# Schema Induction — One-Shot Reward-Model Dataset

This package is a self-contained slice of the larger Schema Induction
project. It produces a dataset of `(question, schema, execution_trace,
answer_score, trace_score)` records that will later train a reward
model for debate schemas. The
agent proposes schemas in two batches, executes them, and both scoring
judges record their verdicts.

## Quickstart

End-to-end: build the train data, bring up the servers, run one-shot
collection on `train_300` (150 NQ + 150 HotPotQA).

### 1. Build the train dataset

```bash
python scripts/build_train.py
# writes data/nq/train.jsonl, data/hotpotqa/train.jsonl,
# data/train_300/train.jsonl, data/train_4/train.jsonl, data/train_hp_300/train.jsonl
```

### 2. Start the corpus server

The static corpus server is launched **outside this package** and must
be reachable at `[corpus_server].base_url` (default
`http://localhost:7470`) before step 5.

### 3. Start the LLM servers

`scripts/deploy_model.sh` launches a vLLM server for one model. Supported
keys (default ports match `config.toml`):

| Role | Default port | Model keys |
| --- | --- | --- |
| **orchestrator** | 7471 | `qwen3-4b`, `gemma-4-e2b` |
| **agent** | 7472 | `qwen3-14b`, `nemotron-30b`, `gemma-4-31b`, `gpt-oss-20b` |
| **judge** | 7473 | `nemotron-super-120b`, `gpt-oss-120b` |

The orchestrator slot is dormant for one-shot collection, so only the
**agent** and **judge** need to be up. Each role can alternatively be
served via `openai` or `openrouter` by editing the matching
`[llm.<role>]` section in `config.toml` (providers: `vllm` | `openai` |
`openrouter` — Anthropic is not supported here).

```bash
# Agent (matches [llm.agent] in config.toml)
./scripts/deploy_model.sh -m qwen3-14b -d 0,1

# Judge (matches [llm.judge] in config.toml)
./scripts/deploy_model.sh -m gpt-oss-120b -d 2,3,4,5
```

Sanity-check each vLLM server with `python scripts/ping_model.py`
(tool-calling) and `python scripts/reason_model.py` (reasoning parser).

### 4. Run one-shot collection on `train_300`

```bash
python scripts/run_oneshot.py --dataset train_300 --split train --max-tasks 300
```

Outputs land under
`results/train_300_train/<question_id>/<run_id>/schemaN{,_trace}.json`,
with the frozen run config at `results_configs/<run_id>.json`.

## Layout

```
One_shot_package/
├── README.md
├── config.toml             # run/agent/judge/oneshot settings
├── personas.toml           # persona system prompts
├── pyproject.toml
├── .env.example
├── src/                    # flat — no sub-packages
│   ├── types.py validator.py generator.py
│   ├── engine.py debate.py supervisor.py
│   ├── judge.py fitness.py
│   ├── llm_client.py tools.py
│   ├── config.py datasets.py logging.py serialization.py
│   └── __init__.py
├── scripts/
│   ├── run_oneshot.py      # main entry
│   ├── build_train.py download_nq.py download_hotpotqa.py
│   ├── deploy_model.sh ping_model.py reason_model.py
└── tests/
```

## Pipeline (per task)

1. The **agent** LLM is asked to produce **4 structurally distinct
   debate schemas** in a single response (batch 1).
2. A second call asks for **4 more schemas that differ structurally** from
   batch 1 (batch 2). Total: 8 schemas per task.
3. All 8 schemas are executed against the task in parallel.
4. Each execution is scored by two judges (same judge client):
   - `answer_score` — correctness of the final answer vs. ground truth.
   - `trace_score` — process quality of the multi-round debate trace.
5. Records are appended to `results/{run_id}/dataset.jsonl`, with a
   per-schema JSON file under `results/{run_id}/traces/`.

The orchestrator client is intentionally **not** used at data-collection
time. It will be trained later on the dataset this package produces, so
the `[llm.orchestrator]` section in `config.toml` is left commented out.

## Setup

### 1. Python environment

```bash
cd One_shot_package
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. API keys

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (judge).
```

### 3. Build the train dataset

NQ + HotPotQA train splits, sampled to 2000 questions (seed 42):

```bash
python scripts/download_nq.py
python scripts/download_hotpotqa.py
python scripts/build_train.py
```

### 4. Deploy the agent LLM (vLLM)

The judge is served via OpenRouter, so only ONE local vLLM server is required for one-shot data collection — the agent. See
`deploy_model.sh` for the supported model keys and reasoning parsers:

```bash
# e.g. Qwen3-14B on GPUs 0,1, port 7472 (matches config.toml's [llm.agent])
./scripts/deploy_model.sh -m qwen3-14b -d 0,1
```

You can use `scripts/ping_model.py` to verify the tool-calling endpoint
and `scripts/reason_model.py` to verify the reasoning parser.

### 5. Static corpus server

By default the package expects it at
`http://localhost:7470` (configurable in `config.toml`'s
`[corpus_server]`).

## Running

```bash
python scripts/run_oneshot.py --dataset train --max-tasks 50
```

Outputs land in `results/{run_id}/`:

- `dataset.jsonl` — one line per evaluated schema (the RM training file).
- `traces/{task_id}_b{batch}_s{slot}.json` — pretty-printed per-schema record.
- `config.json` — the resolved config used for the run.
- `judge_log.json` — every judge call (answer + trace) with its prompt and verdict.
- `run.log` — the structlog stream.

## Dataset row format

```json
{
  "task_id": "nq_train_42",
  "level": 1,
  "question": "...",
  "ground_truth": "...",
  "batch_idx": 0,
  "schema_idx_in_batch": 2,
  "schema": { "mode": "persona", "max_rounds": 3, "rounds": [...], ... },
  "predicted_answer": "...",
  "answer_score": 5.0,
  "trace_score": 4.0,
  "num_rounds_executed": 3,
  "num_llm_calls": 9,
  "num_tool_calls": 4,
  "execution_trace": [ ... ],
  "all_responses": [ [ {"persona": "analyst", "response": "..."}, ... ], ... ]
}
```

`answer_score` and `trace_score` are independent 0–5 Likert signals. The
reward model can be trained on either, both, or a learned combination —
that is downstream work, intentionally out of scope for this package.

## Configuration knobs

`config.toml`:

- `[oneshot].schemas_per_batch` (default 4) — schemas per agent call.
- `[oneshot].num_batches` (default 2) — total schemas per task =
  `schemas_per_batch * num_batches`.
- `[oneshot].debate_mode` — `"persona"`, `"self_consistency"`, or
  `"mixed"` (the agent picks per-schema in mixed mode).
- `[oneshot].eval_workers` — parallelism for (execute + dual-judge) per task.
- `[llm.agent]` — agent model + endpoint.
- `[llm.judge]` — judge model
