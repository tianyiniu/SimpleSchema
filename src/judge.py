"""LLM-as-judge for schema evaluation.

A single judge call grades BOTH dimensions for an executed schema in one
response:

  - **Correctness** — graded as ``CORRECT`` / ``INCORRECT`` / ``NOT_ATTEMPTED``
    using a prompt modeled on OpenAI's SimpleQA grader
    (https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py).
    The three-category internal label keeps grading robust; the public
    ``answer_score`` collapses it to a binary 0/1 (``CORRECT`` → 1, else
    0). The full label is preserved on the per-call record so downstream
    analysis can distinguish wrong-answer vs. no-answer.

  - **Trace quality** — a 0-5 Likert on the multi-round debate trace as
    a *process*, independent of whether the final answer is right.

Both scores are extracted from the same response via two distinct
trailing lines (``ANSWER GRADE: <A|B|C>`` and ``TRACE SCORE: <0-5>``).
A follow-up call is issued if either line is missing.
"""

from __future__ import annotations

import json
import re

from src.llm_client import LLMClient
from src.logging import get_logger

logger = get_logger(__name__)


MIN_TRACE_SCORE = 0.0
MAX_TRACE_SCORE = 5.0

# Tolerate optional angle brackets / quotes / asterisks around the value —
# some judges echo the prompt's placeholder syntax verbatim (e.g.
# "ANSWER GRADE: <C>" or "TRACE SCORE: **3**").
_GRADE_RE = re.compile(
    r"ANSWER\s*GRADE\s*:\s*[<\[\"'*]*\s*([ABC])\s*[>\]\"'*]*",
    re.IGNORECASE,
)
_TRACE_SCORE_RE = re.compile(
    r"TRACE\s*SCORE\s*:\s*[<\[\"'*]*\s*([0-5])\s*[>\]\"'*]*"
)

# Map the three-category grader label to (binary answer_score, label).
_GRADE_TO_ANSWER_SCORE: dict[str, float] = {"A": 1.0, "B": 0.0, "C": 0.0}
_GRADE_TO_LABEL: dict[str, str] = {
    "A": "CORRECT",
    "B": "INCORRECT",
    "C": "NOT_ATTEMPTED",
}


def _record_judge_entry(judge_log: list[dict] | None, entry: dict) -> None:
    """Append an entry to a caller-supplied judge log, if any."""
    if judge_log is not None:
        judge_log.append(entry)


DUAL_JUDGE_SYSTEM_PROMPT = """You are a strict, impartial grader evaluating an AI debate system on TWO independent dimensions.

You will be given:
- The original question
- The gold target answer (ground truth)
- The AI system's full multi-round debate trace (rounds, personas, tool calls, intermediate responses)
- The AI system's final extracted answer

You will grade two things in a single response: ANSWER CORRECTNESS and TRACE QUALITY. The two grades are independent — correctness must not be influenced by trace quality, and vice versa.

=== PART 1: ANSWER CORRECTNESS ===

Look at the question, the gold target, and the AI's final extracted answer, and assign a grade of either CORRECT, INCORRECT, or NOT_ATTEMPTED.

CORRECT: The predicted answer fully contains the important information in the gold target, and contains no information that contradicts the gold target. Only semantic meaning matters — capitalization, punctuation, grammar, and word order do not. Hedging language ("I believe", "it appears that…") is acceptable as long as the gold target is fully covered and no incorrect claims are made. Numeric answers must be accurate to the last significant figure given in the gold target. Minor typos in proper names are tolerated.

INCORRECT: A factual statement in the predicted answer contradicts the gold target. Hedged incorrect claims ("it is possible that …", "I think it might be …") are also INCORRECT — hedging does not rescue a wrong fact.

NOT_ATTEMPTED: The important information in the gold target is not contained in the predicted answer, AND no statements in the predicted answer contradict the gold target. Refusals ("I don't know", "insufficient data", "cannot determine"), empty answers, and off-topic answers all fall here.

=== PART 2: TRACE QUALITY ===

Rate the *process quality* of the debate trace on a 0-5 Likert scale. Score the design and execution of the multi-round debate, not the outcome — a well-run trace can still land on a wrong final answer, and a poorly-run trace can stumble into the right one.

- 5: Exemplary. The schema is well-matched to the question, evidence gathering is targeted and non-redundant, personas/samples contribute distinct value, tools are used purposefully, and the synthesis is grounded in what was actually retrieved or computed.
- 4: Strong trace with minor inefficiencies — e.g. one redundant tool call, one persona that mostly echoes another, or a synthesis step that under-uses gathered evidence.
- 3: Mixed. Meaningful progress, but notable structural problems: schema/question mismatch, wasted rounds, evidence ignored at synthesis, or persona collapse.
- 1-2: Mostly poor. The schema is ill-suited (e.g. heavy debate for a single-fact lookup, or a single round for a multi-hop question), tool use is sparse or unfocused, personas don't differentiate, and synthesis is shallow.
- 0: Degenerate. The trace is empty, the personas all refuse, no tools are used when obviously required, or the synthesis ignores the rounds entirely.

=== OUTPUT FORMAT ===

After thinking carefully, output:

1. A single paragraph summarizing both judgments, referencing both the final answer (vs. gold target) and the process quality of the trace.
2. Then on two separate lines at the END of your response, output EXACTLY:

ANSWER GRADE: A
TRACE SCORE: 3

(substitute your actual letter A/B/C and integer 0-5; do NOT include angle brackets, quotes, or any other punctuation around the value)

where A=CORRECT, B=INCORRECT, C=NOT_ATTEMPTED. Use ONLY these letters for the answer grade; use ONLY a whole-number 0-5 for the trace score. Both lines are required."""

DUAL_JUDGE_USER_TEMPLATE = """Question: {question}

Gold target answer: {ground_truth}

--- AI Debate Trace (round-by-round) ---
{reasoning_context}
--- End of Trace ---

--- Tool calls and results (per round, if any) ---
{tool_calls_context}
--- End of Tool Calls ---

Final extracted answer: {predicted_answer}

Grade BOTH the answer correctness (A=CORRECT / B=INCORRECT / C=NOT_ATTEMPTED) and the trace quality (0-5). End your response with the required two lines:

ANSWER GRADE: A
TRACE SCORE: 3

(substitute your actual letter A/B/C and integer 0-5; do NOT include angle brackets, quotes, or any other punctuation around the value)"""

DUAL_JUDGE_FOLLOWUP_PROMPT = """Your previous response did not include both required final lines in the exact format. Based on your evaluation above, now output ONLY the two final lines, in this exact format:

ANSWER GRADE: A
TRACE SCORE: 3

(substitute your actual letter A/B/C and integer 0-5; do NOT include angle brackets, quotes, or any other punctuation around the value)

where A=CORRECT, B=INCORRECT, C=NOT_ATTEMPTED."""


def _format_reasoning_context(all_responses: list[list[dict[str, str]]]) -> str:
    """Format all persona responses across rounds into a readable context string."""
    if not all_responses:
        return "(no intermediate responses)"
    lines: list[str] = []
    for round_idx, round_responses in enumerate(all_responses):
        lines.append(f"[Round {round_idx + 1}]")
        for entry in round_responses:
            persona = entry.get("persona", "unknown")
            response = entry.get("response", "")
            lines.append(f"  {persona}: {response}")
    return "\n".join(lines)


def _format_tool_calls_context(execution_trace: list[dict] | None) -> str:
    """Render any tool-call traces stored on the ExecutionResult."""
    if not execution_trace:
        return "(no tool calls recorded)"
    lines: list[str] = []
    for entry in execution_trace:
        if entry.get("type") != "tool_call":
            continue
        round_num = entry.get("round", "?")
        persona = entry.get("persona", "?")
        tool = entry.get("tool", "?")
        args = entry.get("arguments", {})
        result = (entry.get("result") or "")[:400]
        lines.append(
            f"[Round {round_num}] {persona} -> {tool}"
            f"({json.dumps(args, ensure_ascii=False)}) -> {result}"
        )
    if not lines:
        return "(no tool calls recorded)"
    return "\n".join(lines)


def _extract_grade(text: str) -> str | None:
    """Extract A/B/C from 'ANSWER GRADE:' line. Returns None if missing."""
    match = _GRADE_RE.search(text)
    return match.group(1).upper() if match else None


def _extract_trace_score(text: str) -> float | None:
    """Extract integer 0-5 from 'TRACE SCORE:' line. Returns None if missing."""
    match = _TRACE_SCORE_RE.search(text)
    if not match:
        return None
    score = float(match.group(1))
    return max(MIN_TRACE_SCORE, min(MAX_TRACE_SCORE, score))


def judge_dual(
    question: str,
    ground_truth: str,
    predicted_answer: str,
    judge_client: LLMClient,
    all_responses: list[list[dict[str, str]]] | None = None,
    execution_trace: list[dict] | None = None,
    judge_log: list[dict] | None = None,
) -> tuple[float, float]:
    """Single-call dual judge: returns ``(answer_score, trace_score)``.

    ``answer_score`` is binary (0.0 or 1.0). Internally the grader uses
    the three-category SimpleQA labels (CORRECT/INCORRECT/NOT_ATTEMPTED)
    for robustness; the public score collapses ``CORRECT`` → 1.0 and the
    other two → 0.0. The original three-way label is preserved on the
    ``judge_log`` record.

    ``trace_score`` is a 0-5 Likert on debate-trace process quality.

    Both scores come from a single LLM response. If the response is
    missing either trailing line, one follow-up call is made to extract
    the missing values; if that also fails the missing dim defaults to 0.
    """
    reasoning_context = _format_reasoning_context(all_responses or [])
    tool_calls_context = _format_tool_calls_context(execution_trace)
    user_prompt = DUAL_JUDGE_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        reasoning_context=reasoning_context,
        tool_calls_context=tool_calls_context,
        predicted_answer=predicted_answer,
    )

    try:
        response = judge_client.complete(
            system_prompt=DUAL_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=16384,
            enable_thinking=True,
        )
    except Exception:
        logger.exception(
            "dual_judge_call_failed",
            predicted=predicted_answer[:200],
            ground_truth=ground_truth[:200],
        )
        _record_judge_entry(judge_log, {
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "answer_score": 0.0,
            "answer_label": "NOT_ATTEMPTED",
            "trace_score": MIN_TRACE_SCORE,
            "method": "dual_judge_call_failed",
            "judgment_summary": None,
        })
        return 0.0, MIN_TRACE_SCORE

    grade = _extract_grade(response)
    trace_score = _extract_trace_score(response)

    logger.info(
        "dual_judge_response",
        predicted=predicted_answer[:200],
        ground_truth=ground_truth[:200],
        grade=grade,
        trace_score=trace_score,
        judgment_summary=response,
    )

    method = "dual_judge"
    followup_response: str | None = None
    if grade is None or trace_score is None:
        # One follow-up retry asking specifically for the trailing lines.
        try:
            followup_response = judge_client.complete(
                system_prompt=DUAL_JUDGE_SYSTEM_PROMPT,
                user_prompt=(
                    user_prompt
                    + "\n\nYour previous response:\n"
                    + response
                    + "\n\n"
                    + DUAL_JUDGE_FOLLOWUP_PROMPT
                ),
                temperature=0.0,
                max_tokens=128,
                enable_thinking=False,
            )
        except Exception:
            logger.exception("dual_judge_followup_call_failed")
        else:
            logger.info("dual_judge_followup_response", followup=followup_response)
            if grade is None:
                grade = _extract_grade(followup_response)
            if trace_score is None:
                trace_score = _extract_trace_score(followup_response)
            method = "dual_judge_followup"

    if grade is None and trace_score is None:
        method = "dual_judge_parse_failed"
        logger.warning(
            "dual_judge_parse_failed",
            judgment_summary=response,
            followup=followup_response,
        )

    answer_label = _GRADE_TO_LABEL.get(grade or "", "NOT_ATTEMPTED")
    answer_score = _GRADE_TO_ANSWER_SCORE.get(grade or "", 0.0)
    final_trace_score = MIN_TRACE_SCORE if trace_score is None else trace_score

    entry = {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "answer_score": answer_score,
        "answer_label": answer_label,
        "trace_score": final_trace_score,
        "method": method,
        "judgment_summary": response,
    }
    if followup_response is not None:
        entry["followup_response"] = followup_response
    _record_judge_entry(judge_log, entry)

    return answer_score, final_trace_score
