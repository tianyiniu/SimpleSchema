"""Debate loop logic — manages persona turns within a round.

Each persona independently calls tools during their turn via the
OpenAI function calling API (per-persona tool calling).
"""

from __future__ import annotations

from src.types import DebateMode, PersonaType, Round
from src.tools import ToolCallCache, ToolRegistry
from src.config import load_personas
from src.llm_client import LLMClient, ToolCallingResult
from src.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Persona prompt loading
# ---------------------------------------------------------------------------

def _load_persona_prompts() -> dict[str, str]:
    """Load persona system prompts from config/personas.toml."""
    return load_personas()


INSTRUCTION_PROMPTS: dict[str, str] = {
    "independently_research": (
        "Research the question independently. Gather relevant facts, data, and evidence "
        "from available sources. Do not yet try to synthesize or finalize an answer — "
        "focus on collecting information."
    ),
    "debate_and_refine": (
        "Review the previous responses and engage in a structured debate. Challenge weak "
        "arguments, reinforce strong ones, and refine the collective understanding. "
        "Aim to converge toward a more accurate answer through critical discussion."
    ),
    "critique_previous": (
        "Critically evaluate the previous responses. Identify logical errors, unsupported "
        "claims, missing evidence, and inconsistencies. Suggest specific corrections or "
        "areas that need further investigation."
    ),
    "gather_facts": (
        "Focus on gathering concrete, verifiable facts relevant to the question. Use "
        "available tools to look up specific data points, statistics, or reference "
        "material. Present findings clearly with sources."
    ),
    "produce_final_answer": (
        "Based on all previous discussion and evidence, produce a clear and definitive "
        "final answer to the question. Synthesize the key findings and state your answer "
        "concisely. Prefix your final answer with 'ANSWER: '."
    ),
    "verify_and_check": (
        "Verify the claims and answers produced so far. Cross-check facts, re-run "
        "calculations, and look for errors. Confirm or correct the current best answer "
        "with evidence."
    ),
    "brainstorm": (
        "Brainstorm multiple possible approaches, interpretations, or answers to the "
        "question. Think creatively and consider diverse angles. Do not commit to a "
        "single answer yet — explore the solution space."
    ),
    "think_and_plan": (
        "Before attempting to answer, create a detailed step-by-step plan for solving "
        "this question. Identify what information is needed, which tools might help, "
        "and what sequence of reasoning steps would lead to the correct answer. "
        "Do NOT produce a final answer yet — only output the plan."
    ),
}

PERSONA_PROMPTS: dict[str, str] | None = None


def get_persona_prompt(persona: PersonaType) -> str:
    """Get the (tool-agnostic) system prompt for a persona."""
    global PERSONA_PROMPTS
    if PERSONA_PROMPTS is None:
        PERSONA_PROMPTS = _load_persona_prompts()
    return PERSONA_PROMPTS[persona.value]


# Per-tool description appended to the persona system prompt at runtime.
# personas.toml intentionally no longer mentions any tools — the persona
# file describes the *role*; the *tools* available are decided per-round
# by the schema and pasted in here. This prevents the model from calling
# tools the schema's round didn't enable (which surfaces as
# ``[Unknown tool: ...]`` errors in the trace).
_TOOL_USAGE_LINES: dict[str, str] = {
    "search_info": (
        "- search_info: focused query (key entities, dates, constraints) -> "
        "ranked title/url/snippet hits. It does NOT summarize."
    ),
    "fetch_url": (
        "- fetch_url: a full URL -> the document's text. Use a URL "
        "already cited in the discussion or one you know directly."
    ),
    "code_compute": (
        "- code_compute: a Python expression or full script -> stdout. "
        "Bare expressions are auto-printed; use print() in scripts."
    ),
}


def _build_system_prompt(
    persona: PersonaType, available_tools: list[str],
) -> str:
    """Combine the persona's role prompt with a per-round tool-availability
    block. Empty tool list yields an explicit no-tools notice."""
    base = get_persona_prompt(persona)
    if not available_tools:
        return (
            base
            + "\nThis round you have NO tools. Reason from the question and "
            "any prior round context. Do not emit tool calls."
        )
    lines = [
        "",
        "Tool usage — for THIS round you have ONLY these tools (do not "
        "call any other tool name; calls to unlisted tools are rejected):",
    ]
    for name in available_tools:
        line = _TOOL_USAGE_LINES.get(name)
        if line is not None:
            lines.append(line)
    lines.append("You may call them multiple times to refine your work.")
    return base + "\n".join(lines)


# Per-instruction output budget (tokens). Caps the agent's max_tokens for
# the round so context-bound models (Qwen3-14B has only 32k) don't burn
# their full budget on every round when the instruction's natural output
# is short. Falls back to the agent's configured max_tokens otherwise.
INSTRUCTION_MAX_TOKENS: dict[str, int] = {
    "think_and_plan":        2048,
    "produce_final_answer":  2048,
    "verify_and_check":      4096,
    "critique_previous":     4096,
    "brainstorm":            4096,
    "gather_facts":          8192,
    "independently_research": 8192,
    "debate_and_refine":     8192,
}

# Same cap table for thinking-enabled models. Reasoning channels routinely
# spend 4-8k tokens BEFORE any visible content (Qwen3 emits <think>...
# </think>; gpt-oss-20b emits a separate reasoning trace that vLLM's
# openai_gptoss parser strips). If the budget exhausts mid-reasoning the
# response leaves visible content empty. These caps reserve generous
# headroom for the reasoning trace on top of the no-thinking visible-
# output budget.
INSTRUCTION_MAX_TOKENS_THINKING: dict[str, int] = {
    "think_and_plan":        12288,
    "produce_final_answer":  12288,
    "verify_and_check":      16384,
    "critique_previous":     16384,
    "brainstorm":            16384,
    "gather_facts":          16384,
    "independently_research": 16384,
    "debate_and_refine":     16384,
}


# Per-instruction depth bound on the inner tool-calling loop. One "round" is
# one assistant turn that may emit multiple parallel tool calls — so this
# limits sequential chaining (search -> fetch -> search -> ...), not total
# tool calls. Research-heavy instructions get more rounds so multi-hop
# browse-then-read patterns can complete; planning/answer instructions need
# fewer because tools shouldn't be required there.
INSTRUCTION_MAX_TOOL_ROUNDS: dict[str, int] = {
    "think_and_plan":        3,
    "produce_final_answer":  3,
    "verify_and_check":      4,
    "critique_previous":     4,
    "brainstorm":            4,
    "gather_facts":          6,
    "independently_research": 6,
    "debate_and_refine":     6,
}
DEFAULT_MAX_TOOL_ROUNDS = 6


# Each prior-round response is truncated to this many characters before
# being inlined into a downstream persona's user prompt. Keeps total
# prompt size bounded as rounds accumulate (the unbounded version would
# pile every previous full response into every subsequent round).
PRIOR_RESPONSE_MAX_CHARS = 2500
_TRUNC_MARKER = "\n... [truncated]\n"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _truncate_prior_response(text: str, max_chars: int = PRIOR_RESPONSE_MAX_CHARS) -> str:
    """Shrink a prior-round response while preserving the ANSWER: line.

    Strategy: if the response fits, return as-is. Otherwise keep an
    ``ANSWER:`` line (always more important than the surrounding rationale)
    and as much leading rationale as fits within ``max_chars``.
    """
    if len(text) <= max_chars:
        return text

    upper = text.upper()
    if "ANSWER:" in upper:
        idx = upper.rfind("ANSWER:")
        answer_line = text[idx:].split("\n", 1)[0]
        prefix_budget = max_chars - len(answer_line) - len(_TRUNC_MARKER)
        if prefix_budget > 100:
            return text[:prefix_budget] + _TRUNC_MARKER + answer_line
    return text[: max_chars - len(_TRUNC_MARKER)] + _TRUNC_MARKER


def _build_user_prompt(
    question: str,
    instruction: str,
    previous_responses: list[dict[str, str]],
    available_tools: list[str],
) -> str:
    """Build the user prompt for a persona turn."""
    parts = [f"Question: {question}", f"\nInstruction: {instruction}"]

    if previous_responses:
        parts.append("\n--- Previous responses from this debate ---")
        for resp in previous_responses:
            truncated = _truncate_prior_response(resp["response"])
            parts.append(f"\n[{resp['persona']}]: {truncated}")

    parts.append(
        "\nProvide your response. Use any available tools to search for information, "
        "perform calculations, or run code as needed. "
        "If you have a specific answer, state it clearly at the end prefixed with 'ANSWER: '."
    )

    if available_tools:
        tool_hints = {
            "search_info": "- search_info args: {'query': '<focused search query with key entities or constraints>'}  # returns ranked title/url/snippet hits",
            "fetch_url": "- fetch_url args: {'query': '<full URL>'}  # returns the full document text",
            "code_compute": "- code_compute args: {'query': '<python expression or full script>'}  # bare expressions are auto-printed",
        }
        parts.append("\nTool call format tips:")
        for tool_name in available_tools:
            hint = tool_hints.get(tool_name)
            if hint is not None:
                parts.append(hint)

    return "\n".join(parts)


def _max_tokens_for_instruction(
    instruction_value: str, fallback: int, enable_thinking: bool = False,
) -> int:
    """Lookup the per-instruction output cap, defaulting to ``fallback``.

    When ``enable_thinking`` is True, the larger thinking-aware cap table
    is consulted so the model has headroom to finish its <think> block
    before emitting visible output. Always returns at most ``fallback``
    so a generous cap can't accidentally exceed the client's max_tokens.
    """
    table = (
        INSTRUCTION_MAX_TOKENS_THINKING
        if enable_thinking else INSTRUCTION_MAX_TOKENS
    )
    cap = table.get(instruction_value, fallback)
    return min(cap, fallback)


def _max_tool_rounds_for_instruction(instruction_value: str) -> int:
    """Lookup the per-instruction tool-loop depth cap."""
    return INSTRUCTION_MAX_TOOL_ROUNDS.get(
        instruction_value, DEFAULT_MAX_TOOL_ROUNDS
    )


# ---------------------------------------------------------------------------
# Debate round execution
# ---------------------------------------------------------------------------

def run_debate_round(
    round_spec: Round,
    question: str,
    previous_round_responses: list[dict[str, str]],
    agent_client: LLMClient,
    tool_registry: ToolRegistry,
    mode: DebateMode = DebateMode.PERSONA,
    tool_cache: ToolCallCache | None = None,
) -> tuple[list[dict[str, str]], int, int, list[dict]]:
    """Execute a single debate round with per-persona tool calling.

    In ``persona`` mode each persona independently calls tools and
    contributes one response. In ``self_consistency`` mode the round's
    single persona is sampled ``round_spec.n_samples`` times in parallel.

    Returns:
        (responses, num_llm_calls, num_tool_calls, trace_entries)
    """
    instruction = INSTRUCTION_PROMPTS.get(
        round_spec.instruction.value,
        round_spec.instruction.value.replace("_", " "),
    )

    # Get OpenAI tool schemas and executors for this round's tools
    tool_names = [t.value for t in round_spec.tools]
    tool_schemas, tool_executors = tool_registry.get_openai_tools(tool_names)

    # Persona-mode: one request per persona. Self-consistency: n_samples
    # parallel requests of the round's single persona.
    if mode == DebateMode.SELF_CONSISTENCY:
        persona = round_spec.personas[0]
        system_prompt = _build_system_prompt(persona, tool_names)
        user_prompt = _build_user_prompt(
            question, instruction, previous_round_responses, tool_names
        )
        # Aggregation rounds (n_samples=1) use the bare persona name so
        # downstream synthesis lookups (e.g. SYNTHESIZER_PERSONA) match.
        # Sampling rounds (n_samples>1) suffix #i to distinguish samples.
        if round_spec.n_samples == 1:
            labels = [persona.value]
        else:
            labels = [
                f"{persona.value}#{i + 1}" for i in range(round_spec.n_samples)
            ]
        requests: list[tuple[str, str, list[dict], dict]] = [
            (system_prompt, user_prompt, tool_schemas, tool_executors)
            for _ in range(round_spec.n_samples)
        ]
    else:
        # If a persona is repeated within the round (analyst/critic may
        # appear up to 2x), suffix the label with #1/#2 so trace entries
        # and the synthesis lookups remain unambiguous.
        persona_totals: dict[str, int] = {}
        for p in round_spec.personas:
            persona_totals[p.value] = persona_totals.get(p.value, 0) + 1
        seen: dict[str, int] = {}
        labels = []
        for p in round_spec.personas:
            name = p.value
            seen[name] = seen.get(name, 0) + 1
            labels.append(f"{name}#{seen[name]}" if persona_totals[name] > 1 else name)
        requests = []
        for persona in round_spec.personas:
            system_prompt = _build_system_prompt(persona, tool_names)
            user_prompt = _build_user_prompt(
                question, instruction, previous_round_responses, tool_names
            )
            requests.append((system_prompt, user_prompt, tool_schemas, tool_executors))

    round_max_tokens = _max_tokens_for_instruction(
        round_spec.instruction.value,
        agent_client.max_tokens,
        enable_thinking=agent_client.enable_thinking,
    )
    round_max_tool_rounds = _max_tool_rounds_for_instruction(
        round_spec.instruction.value
    )

    logger.info(
        "debate_round_start",
        round=round_spec.round,
        mode=mode.value,
        personas=[p.value for p in round_spec.personas],
        n_samples=round_spec.n_samples,
        tools=tool_names,
        instruction=instruction,
        has_tools=bool(tool_schemas),
        max_tokens=round_max_tokens,
        max_tool_rounds=round_max_tool_rounds,
    )

    # Execute all persona calls concurrently (each with its own tool-calling loop)
    if tool_schemas:
        tool_results: list[ToolCallingResult] = agent_client.complete_with_tools_batch(
            requests,
            max_tokens=round_max_tokens,
            max_tool_rounds=round_max_tool_rounds,
            tool_cache=tool_cache,
        )
    else:
        # No tools — use plain batch completion for efficiency
        plain_requests = [(sys, usr) for sys, usr, _, _ in requests]
        raw_responses = agent_client.complete_batch(
            plain_requests, max_tokens=round_max_tokens,
        )
        tool_results = [
            ToolCallingResult(text=resp, num_llm_calls=1, num_tool_calls=0)
            for resp in raw_responses
        ]

    # Collect responses and traces
    responses: list[dict[str, str]] = []
    all_trace: list[dict] = []
    total_llm_calls = 0
    total_tool_calls = 0

    for label, (sys_prompt, usr_prompt, _, _), result in zip(
        labels, requests, tool_results
    ):
        responses.append({"persona": label, "response": result.text})
        total_llm_calls += result.num_llm_calls
        total_tool_calls += result.num_tool_calls

        # Build trace entries for tool calls
        for tc in result.tool_traces:
            all_trace.append({
                "type": "tool_call",
                "round": round_spec.round,
                "persona": label,
                "tool": tc.tool_name,
                "arguments": tc.arguments,
                "result": tc.result,
            })

        # Persona response trace
        all_trace.append({
            "type": "persona_call",
            "round": round_spec.round,
            "persona": label,
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,
            "response": result.text,
            "num_tool_calls": result.num_tool_calls,
            "num_llm_calls": result.num_llm_calls,
        })

    logger.info(
        "debate_round_complete",
        round=round_spec.round,
        num_responses=len(responses),
        total_llm_calls=total_llm_calls,
        total_tool_calls=total_tool_calls,
        response_previews={
            r["persona"]: r["response"][:150] for r in responses
        },
    )

    return responses, total_llm_calls, total_tool_calls, all_trace


