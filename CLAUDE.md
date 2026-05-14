# CLAUDE.md — One-Shot Schema Quality Dataset

## Project Overview

This package collects a labeled dataset of `(question, schema,
execution_trace, answer_score, trace_score)` records for training a
reward model that predicts schema quality. There is **no evolutionary
search** here — the agent proposes schemas, executes them, and both
judges score the result. The reward model is trained downstream from
this dataset.

## Pipeline (per task)

1. The **agent** LLM is asked for **4 structurally distinct schemas** in
   a single call. The prompt actively pushes for diversity across round
   count, persona mix, tool selection, instructions, and synthesis
   method, so the four candidates spread along multiple axes.
2. All 4 schemas are executed in parallel against the task.
3. A **single** judge call grades each execution on BOTH dimensions:
   - `answer_score` — binary correctness (SimpleQA-style 3-category
     grading collapsed to 0/1; the full CORRECT/INCORRECT/NOT_ATTEMPTED
     label is preserved in the judge_call record).
   - `trace_score` — 0–5 Likert on multi-round debate process quality.
4. Each `(task, schema)` is persisted as two files under
   `results/{dataset}_{split}/{question_id}/{run_id}/`:
   - `schemaN.json` — overview: question, ground truth, predicted
     answer, both scores, resource counts, and the schema definition.
   - `schemaN_trace.json` — full `execution_trace`, per-round
     `all_responses`, and the single dual-judge call record.
   The frozen run config is written to `results_configs/{run_id}.json`.

The **orchestrator** client is intentionally dormant at data-collection
time — it will be trained later, on the dataset this package produces.
`[llm.orchestrator]` in `config.toml` is commented out; `LLMClients`
accepts a missing orchestrator section without error.

## Layout

```
One_shot_package/
├── README.md, CLAUDE.md
├── config.toml          # [run], [corpus_server], [llm.*], [tools], [oneshot]
├── personas.toml        # persona system prompts
├── pyproject.toml, .env.example
├── src/                 # flat — no sub-packages
│   ├── types.py         # Pydantic models: Schema, Round, DebateMode, Task, ...
│   ├── validator.py     # Mode-aware grammar checks
│   ├── generator.py     # propose_schemas_batch — single entry point
│   ├── engine.py        # Schema executor
│   ├── debate.py        # Per-round persona / sample dispatch
│   ├── supervisor.py    # supervised_execute (retries up to 3)
│   ├── judge.py         # judge_answer + judge_trace_quality
│   ├── fitness.py       # evaluate_schema_dual_judge + DualJudgeResult
│   ├── llm_client.py    # vllm | openai | openrouter — no Anthropic here
│   ├── tools.py         # Tool / ToolRegistry + search_info, fetch_url, code_compute
│   ├── config.py        # TOML loader + .env reader (merged from old utils/env.py)
│   ├── datasets.py      # Dataset registry — nq + hotpotqa, each train/test
│   ├── logging.py       # structlog setup
│   ├── serialization.py # schema_to_dict + save_json_durable
│   └── __init__.py
├── scripts/
│   ├── run_oneshot.py   # main entry point
│   ├── build_train.py   # NQ + HotPotQA → per-dataset train JSONL files
│   ├── download_nq.py, download_hotpotqa.py
│   ├── deploy_model.sh  # vLLM launcher with tool/reasoning parser presets
│   └── ping_model.py, reason_model.py  # vLLM sanity helpers
├── tests/               # test_schema.py, test_engine.py, test_fitness.py, test_llm_client.py
└── data/, results/, results_configs/   # gitignored
```

All imports are one level deep: `from src.X import Y`. There are no
sub-packages under `src/`.

## Output Dataset Format

Per-question, per-run on-disk layout:

```
results/
└── <dataset>_<split>/                # nq_train, nq_test, hotpotqa_train, hotpotqa_test
    └── <question_id>/                # e.g. nq_train_42
        └── <run_id>/                 # oneshot_<ts>_<dataset>_<split>_tasks<N>
            ├── schema1.json          # overview
            ├── schema1_trace.json    # trace + judge calls
            ├── schema2.json
            └── ...

results_configs/
└── <run_id>.json                     # frozen run config
```

`schemaN.json` (overview):

```json
{
  "task_id": "nq_train_42",
  "level": 1,
  "question": "...",
  "ground_truth": "...",
  "predicted_answer": "...",
  "answer_score": 1.0,
  "trace_score": 4.0,
  "num_rounds_executed": 3,
  "num_llm_calls": 9,
  "num_tool_calls": 4,
  "schema_index": 3,
  "schema_idx_in_batch": 2,
  "run_id": "oneshot_20260513_103000_nq_train_tasks50",
  "schema": { "mode": "persona", "max_rounds": 3, "rounds": [...], ... }
}
```

`schemaN_trace.json` (heavy):

```json
{
  "task_id": "nq_train_42",
  "schema_index": 3,
  "run_id": "oneshot_20260513_103000_nq_train_tasks50",
  "execution_trace": [ ... ],
  "all_responses": [ [ {"persona": "analyst", "response": "..."}, ... ], ... ],
  "judge_calls": [
    {
      "method": "dual_judge",
      "answer_score": 1.0,
      "answer_label": "CORRECT",
      "trace_score": 4.0,
      "judgment_summary": "..."
    }
  ]
}
```

`answer_score` is binary (0.0 or 1.0); `trace_score` is a 0–5 Likert.
RM training can use either, both, or a learned combination — out of
scope for this package.

`schema_index` is a flat 1..N index over the single proposal batch;
`schema_idx_in_batch` is preserved for analysis.

## Schema Grammar

A schema is a JSON object with:

- `mode`: `"persona"` or `"self_consistency"`.
- `max_rounds`: hard cap on round count (safety valve).
- `final_synthesis`: `"majority_vote"` | `"last_persona"` | `"synthesizer_persona"`.
- `rounds`: a list of round objects.

Each round has:

- `personas`: 1–5 personas in persona mode (from
  `analyst` / `critic` / `synthesizer`). `analyst` and `critic` may each
  appear up to 2 times in a single round to amplify that voice;
  `synthesizer` appears at most once. In self-consistency mode the
  round has exactly 1 persona — `generic_assistant` for sampling rounds
  or `synthesizer` for aggregation.
- `n_samples`: 1 in persona mode; 2–8 for self-consistency sampling
  rounds, 1 for self-consistency aggregation rounds.
- `tools`: 0+ tools from `search_info` / `fetch_url` / `code_compute`.
  Each round's persona system prompt is built dynamically by
  `debate.py` to ONLY mention this round's enabled tools — calls to
  other tool names are rejected as `[Unknown tool: ...]`.
- `instruction`: one of the `InstructionType` enum values (e.g.
  `think_and_plan`, `independently_research`, `debate_and_refine`,
  `produce_final_answer`).

There is **no early stopping**. Every round defined in the schema runs
in order; `max_rounds` is the only ceiling.

`validator.py` enforces these constraints — every generated schema
passes through `is_valid_schema` before execution. When a persona is
repeated within a round, `debate.py` distinguishes the trace labels
with a `#1`/`#2` suffix (e.g. `analyst#1`, `analyst#2`).

## LLM Backend

Two active roles + one dormant slot (`LLMClients` container):

- **agent** (`~14–30B`): proposes the 8 schemas AND runs the personas
  during schema execution. The only role with tool access.
- **judge** (OpenRouter/OpenAI): scores answers via `judge_answer` and
  traces via `judge_trace_quality`. Same client for both.
- **orchestrator** (`Optional[LLMClient]`): unused at data-collection
  time; reserved for future RM training. If `[llm.orchestrator]` is
  absent or has an empty `model`, the slot stays `None`.

Each role has its own `[llm.<role>]` section. A `[llm.shared]` section,
if present, supplies defaults each role can override.

### Providers

- `vllm` — OpenAI SDK pointed at `http://{ip}:{port}/v1`.
- `openai` — standard OpenAI API; key from `api_key_env`.
- `openrouter` — OpenAI-compatible API at
  `https://openrouter.ai/api/v1`; key from `api_key_env`.

Anthropic is **not** supported in this package; the receiving team can
re-add the branch if they want to swap the judge to a Claude model.

### Concurrency

- `concurrent.futures.ThreadPoolExecutor` (not asyncio).
- Per-round persona / self-consistency-sample calls dispatch concurrently
  via `complete_batch` and `complete_with_tools_batch`.
- The 8 (execute + dual-judge) pipelines per task run concurrently up to
  `[oneshot].eval_workers`.

### Extended thinking

`enable_thinking = true` passes `{"chat_template_kwargs": {"enable_thinking": True}}`
as `extra_body` **only** for `provider = "vllm"` (Qwen3 chat-template
flag). `strip_thinking()` removes `<think>...</think>` blocks from all
responses regardless of provider.

## Tools

`tools.py` exposes three tools, all with the unified
`tool(query: str) -> str` interface:

- `search_info` — corpus / web search; returns numbered title/url/snippet.
- `fetch_url` — full document text by URL.
- `code_compute` — Python expression or full script in a subprocess
  (bare expressions auto-`print(...)`).

Corpus modes: **`static`** hits the local Flask corpus server
(`[corpus_server].base_url`, default `http://localhost:7470`); **`live`**
hits Desearch + direct HTTP. One-shot runs force `static` — the corpus
server is launched **outside this package** and must be running before
`run_oneshot.py`.

`active_tool_names(cfg=None)` reads the `[tools].enabled` list. There is
no mutable `ACTIVE_TOOL_NAMES` global — tool selection is determined
once when `create_tool_registry` is called.

## Coding Conventions

- Python 3.11+ (uses `tomllib`, PEP 604 unions).
- **Pydantic v2** for all data models.
- `concurrent.futures.ThreadPoolExecutor` — NOT asyncio.
- `structlog` JSON logging via `src/logging.py`.
- Type hints throughout; `mypy --strict` on `src/` is the target.
- Tests with `pytest`. The 4 test files cover schema validation,
  execution engine, fitness helpers, and LLM-client argument parsing.
  None require a running vLLM server.
- Config via TOML (`tomllib`, stdlib). Personas live in `personas.toml`
  at the package root.
- Randomness goes through a seeded `random.Random` instance
  (`[oneshot].random_seed`, default 42).

## Key Implementation Notes

1. **Agent diversity is prompt-driven.** `propose_schemas_batch` issues
   one LLM call asking for `n` structurally distinct schemas in a JSON
   array. The system prompt enumerates the diversity axes explicitly
   (round count, persona mix, tool selection, instructions, synthesis
   method, stop conditions) so the four candidates from a single call
   spread across structures rather than emitting near-copies. The call
   uses temperature 0.9 by default (separate from execution
   temperature). If the model returns fewer than `n` valid schemas
   after retries, the run proceeds with whatever was produced (logged
   as `schema_batch_returning_partial`).

2. **Schema validation is critical.** Every batch-proposed schema must
   pass `is_valid_schema` before execution. The validator is mode-aware
   — self-consistency rounds require exactly one persona, only allow
   `generic_assistant` or `synthesizer`, and force `n_samples=1` for
   the synthesizer aggregation case.

3. **Dual-judge scoring (single call).** `evaluate_schema_dual_judge`
   runs the agent once, then issues exactly one `judge_dual` call that
   returns both scores from the same response. The grader prompt is
   modeled on OpenAI's SimpleQA grader — three-category correctness
   (CORRECT / INCORRECT / NOT_ATTEMPTED) for robustness, collapsed to
   binary `answer_score` (1 if CORRECT, else 0). `trace_score` is a
   0-5 Likert on process quality, judged independently. Both scores
   are extracted from two required trailing lines
   (`ANSWER GRADE: <A|B|C>` and `TRACE SCORE: <0-5>`); a follow-up
   call asks specifically for them if either is missing. The original
   three-category label is preserved on the `judge_calls` record as
   `answer_label`. There is no longer an exact-match fast path — the
   trace score requires the LLM call regardless.

4. **Retries.** `supervised_execute` retries failed executions up to 3
   times before returning an empty `ExecutionResult`. Tool calls retry
   up to 2 times via `Tool.safe_call` before returning a sentinel error
   string.

5. **Reproducibility.** The full TOML config plus task IDs and tool
   names are written to `results_configs/{run_id}.json` at the start of
   every run. LLM calls are stochastic (`temperature > 0` for the
   agent), so re-runs with the same seed are not bit-identical.

6. **sys.path bootstrap.** `scripts/run_oneshot.py`,
   `scripts/ping_model.py`, and `scripts/reason_model.py` each prepend
   the package root to `sys.path` so `from src...` resolves against this
   package even when the parent project is installed editable in the
   same venv.

## CLI Quick Reference

```bash
# Build the training data (NQ Open + HotPotQA pools + mixed datasets)
python scripts/download_nq.py
python scripts/download_hotpotqa.py
python scripts/build_train.py
# writes:
#   data/nq/train.jsonl              (per-source NQ Open pool, uniform random)
#   data/hotpotqa/train.jsonl        (per-source HotPotQA pool, stratified by difficulty)
#   data/train_300/train.jsonl       (150 NQ + 150 HP, independently sampled)
#   data/train_4/train.jsonl         (2 NQ + 2 HP, smoke-test sized)
#   data/train_hp_300/train.jsonl    (300 HP: exactly 100 easy + 100 medium + 100 hard)

# Launch a vLLM server for the agent (judge is OpenRouter, no local server)
./scripts/deploy_model.sh -m qwen3-14b -d 0,1

# Verify the vLLM server (tool-call + reasoning-parser smoke tests)
python scripts/ping_model.py
python scripts/reason_model.py

# Run one-shot data collection (any of: nq | hotpotqa | train_300 | train_4)
python scripts/run_oneshot.py --dataset train_4   --split train --max-tasks 4
python scripts/run_oneshot.py --dataset train_300 --split train --max-tasks 300
python scripts/run_oneshot.py --dataset nq        --split train --max-tasks 50

# Tests (no servers required)
pytest -q
```
