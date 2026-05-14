"""LLM client supporting vLLM (self-hosted), OpenAI, and OpenRouter.

Two roles for one-shot data collection: ``agent`` (handles tool-calling
against vLLM, also proposes schemas) and ``judge`` (OpenRouter or OpenAI).
A third, optional ``orchestrator`` role is kept for forward compatibility
with downstream reward-model training but is unused here.
"""

from __future__ import annotations

import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from openai import OpenAI

from src.config import get_env_value
from src.logging import get_logger
from src.tools import ToolCallCache

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

TOOL_ARGUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "search_info": ("query", "search_query", "q", "question", "topic"),
    "fetch_url": ("query", "url", "link", "href"),
    "code_compute": ("query", "code", "python", "script", "expression"),
}

logger = get_logger(__name__)


def strip_thinking(text: str) -> str:
    """Strip ``<think>...</think>`` blocks (Qwen3-style) from output."""
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def _strip_thinking_with_warning(raw: str) -> str:
    """Strip thinking blocks and warn if it consumed the entire response.

    The classic failure mode is Qwen3 + ``enable_thinking=true`` running
    out of ``max_tokens`` mid-think — the response ends with an unclosed
    ``<think>`` block which strip_thinking() then deletes wholesale,
    leaving the persona response empty. This emits a structured warning
    so that exhaustion is visible in run logs instead of silent.
    """
    if not raw:
        return ""
    cleaned = strip_thinking(raw)
    if not cleaned:
        had_unclosed_think = "<think>" in raw and "</think>" not in raw
        logger.warning(
            "thinking_consumed_entire_response",
            raw_length=len(raw),
            had_unclosed_think=had_unclosed_think,
            raw_head=raw[:300],
            raw_tail=raw[-300:],
        )
    return cleaned


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    """Best-effort parse for tool-call arguments from the model."""
    if isinstance(raw_args, dict):
        return raw_args

    if not isinstance(raw_args, str):
        return {"query": str(raw_args)}

    text = raw_args.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"query": str(parsed)}
    except json.JSONDecodeError:
        pass

    try:
        literal = ast.literal_eval(text)
        if isinstance(literal, dict):
            return {str(k): v for k, v in literal.items()}
        return {"query": str(literal)}
    except (ValueError, SyntaxError):
        pass

    match = _JSON_BLOCK_RE.search(text)
    if match:
        block = match.group(0)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"query": text}


def _resolve_tool_query(tool_name: str, args: dict[str, Any]) -> str:
    """Extract a query string from parsed tool-call args with tool-specific aliases."""
    aliases = TOOL_ARGUMENT_ALIASES.get(tool_name, ("query",))
    for key in aliases:
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, str):
            try:
                return json.dumps(value)
            except TypeError:
                return str(value)

    for value in args.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None:
            try:
                return json.dumps(value)
            except TypeError:
                return str(value)

    return ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Per-role LLM configuration.

    For ``provider == "vllm"``: the OpenAI SDK is pointed at the local
    server via ``ip``/``port`` and ``api_key`` is the literal ``"EMPTY"``
    string vLLM accepts.

    For ``provider == "openai"``: standard OpenAI API. The API key is
    read from the env variable named by ``api_key_env``.

    For ``provider == "openrouter"``: OpenRouter's OpenAI-compatible API
    at ``https://openrouter.ai/api/v1``. Same env-var rule for the key.

    """

    provider: str = "vllm"
    model: str = ""

    # vllm: target the local server
    ip: str = "localhost"
    port: int = 6666
    api_key: str = "EMPTY"

    # openai / openrouter: env var holding the real API key
    api_key_env: str = ""

    # Inference parameters
    temperature: float = 0.7
    tool_calling_temperature: float = 0.2
    max_tokens: int = 16384
    enable_thinking: bool = False  # Qwen3-specific; ignored elsewhere

    # Concurrency
    max_workers: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Build from a parsed TOML section."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class ToolCallTrace:
    tool_name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class ToolCallingResult:
    text: str
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    num_llm_calls: int = 0
    num_tool_calls: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Synchronous LLM client with thread-pool concurrency.

    A single instance is bound to one ``LLMConfig`` (one provider, one
    model). The three roles in ``LLMClients`` each own their own.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model_name = config.model
        self.temperature = config.temperature
        self.tool_calling_temperature = config.tool_calling_temperature
        self.max_tokens = config.max_tokens
        self.enable_thinking = config.enable_thinking
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._call_count = 0
        self._call_time_total = 0.0  # cumulative seconds spent in OpenAI/vLLM calls
        self._call_count_lock = Lock()

        if config.provider == "vllm":
            self._oa_client: OpenAI = OpenAI(
                api_key=config.api_key,
                base_url=f"http://{config.ip}:{config.port}/v1",
            )
        elif config.provider == "openai":
            self._oa_client = OpenAI(api_key=self._require_api_key(config))
        elif config.provider == "openrouter":
            self._oa_client = OpenAI(
                api_key=self._require_api_key(config),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(
                f"unknown LLM provider {config.provider!r}; "
                "expected one of 'vllm', 'openai', 'openrouter'"
            )

    @staticmethod
    def _require_api_key(config: LLMConfig) -> str:
        if not config.api_key_env:
            raise RuntimeError(
                f"provider={config.provider!r} requires api_key_env to be set"
            )
        key = get_env_value(config.api_key_env)
        if not key:
            raise RuntimeError(
                f"environment variable {config.api_key_env!r} is not set"
            )
        return key

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_time_total(self) -> float:
        """Cumulative seconds spent in the underlying chat-completions API call."""
        return self._call_time_total

    @property
    def call_time_avg(self) -> float:
        """Mean per-call latency in seconds (0.0 if no calls have been made)."""
        with self._call_count_lock:
            if not self._call_count:
                return 0.0
            return self._call_time_total / self._call_count

    def reset_call_count(self) -> None:
        with self._call_count_lock:
            self._call_count = 0
            self._call_time_total = 0.0

    def _record_call(self, duration: float) -> None:
        """Bump the per-client call counter and accumulate elapsed time."""
        with self._call_count_lock:
            self._call_count += 1
            self._call_time_total += duration

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Public API: complete / complete_batch
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        return self._openai_complete(
            system_prompt, user_prompt, temp, tokens, enable_thinking
        )

    def complete_batch(
        self,
        requests: list[tuple[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        futures = {
            self.executor.submit(
                self.complete, sys_prompt, usr_prompt, temperature, max_tokens,
                enable_thinking,
            ): i
            for i, (sys_prompt, usr_prompt) in enumerate(requests)
        }
        results: list[str | None] = [None] * len(requests)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        return [r if r is not None else "" for r in results]

    # ------------------------------------------------------------------
    # Public API: tool-calling (OpenAI-compatible providers only)
    # ------------------------------------------------------------------

    def complete_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_executors: dict[str, Any],
        max_tool_rounds: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        tool_cache: ToolCallCache | None = None,
    ) -> ToolCallingResult:
        return self._openai_complete_with_tools(
            system_prompt, user_prompt, tools, tool_executors,
            max_tool_rounds, temperature, max_tokens, enable_thinking,
            tool_cache,
        )

    def complete_with_tools_batch(
        self,
        requests: list[tuple[str, str, list[dict[str, Any]], dict[str, Any]]],
        max_tool_rounds: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        tool_cache: ToolCallCache | None = None,
    ) -> list[ToolCallingResult]:
        # If no cache is supplied, share one across this batch so personas
        # in the same round still benefit from cache hits on each other's
        # tool calls.
        shared_cache = tool_cache if tool_cache is not None else ToolCallCache()
        futures = {
            self.executor.submit(
                self.complete_with_tools,
                sys_prompt, usr_prompt, tools, executors,
                max_tool_rounds, temperature, max_tokens, enable_thinking,
                shared_cache,
            ): i
            for i, (sys_prompt, usr_prompt, tools, executors) in enumerate(requests)
        }
        results: list[ToolCallingResult | None] = [None] * len(requests)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        return [
            r if r is not None else ToolCallingResult(text="", num_llm_calls=0)
            for r in results
        ]

    # ------------------------------------------------------------------
    # OpenAI-compatible implementation (vllm + openai providers)
    # ------------------------------------------------------------------

    def _extra_body(self, enable_thinking: bool | None) -> dict[str, Any]:
        # ``enable_thinking`` is a Qwen3 chat-template flag passed through
        # vLLM's extra_body. OpenAI ignores it; we only emit it when the
        # role is configured for a Qwen-family model.
        if self.config.provider != "vllm":
            return {}
        enabled = enable_thinking if enable_thinking is not None else self.enable_thinking
        return {"chat_template_kwargs": {"enable_thinking": enabled}}

    def _openai_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool | None,
    ) -> str:
        text = self._chat_once(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            tools=None,
        )
        if text:
            return text

        # Empty content recovery: reasoning models (gpt-oss-20b's
        # openai_gptoss parser, Qwen3's <think> blocks) sometimes burn the
        # entire token budget on the reasoning channel and leave visible
        # content empty. Retry once with thinking disabled and an explicit
        # forcing nudge appended to the user prompt.
        logger.warning("complete_returned_empty_retrying")
        forced_user = user_prompt + (
            "\n\n[Note: a previous attempt at this turn returned no visible "
            "content. Respond NOW in plain text and keep it concise. "
            "If the question expects a specific answer, end with a line "
            "in the form 'ANSWER: <your answer>'.]"
        )
        recovered = self._chat_once(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": forced_user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=False,
            tools=None,
        )
        logger.warning("complete_retry_after_empty", recovered=bool(recovered))
        return recovered

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool | None,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """One chat-completions call returning post-stripped text content."""
        assert self._oa_client is not None
        t0 = time.perf_counter()
        try:
            response = self._oa_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body(enable_thinking) or None,
            )
            content = response.choices[0].message.content or ""
            return _strip_thinking_with_warning(content)
        except Exception:
            logger.exception("llm_call_failed")
            raise
        finally:
            self._record_call(time.perf_counter() - t0)

    def _openai_complete_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_executors: dict[str, Any],
        max_tool_rounds: int,
        temperature: float | None,
        max_tokens: int | None,
        enable_thinking: bool | None,
        tool_cache: ToolCallCache | None = None,
    ) -> ToolCallingResult:
        assert self._oa_client is not None
        temp = (
            temperature if temperature is not None
            else self.tool_calling_temperature
        )
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        extra_body = self._extra_body(enable_thinking) or None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        traces: list[ToolCallTrace] = []
        total_llm_calls = 0
        total_tool_calls = 0
        loop_exit_reason = "max_rounds"

        # Shared cache wins; otherwise the call-local fallback isolates this
        # call's results from anything else.
        cache = tool_cache if tool_cache is not None else ToolCallCache()

        available_tool_names = sorted(tool_executors.keys())

        for _ in range(max_tool_rounds):
            total_llm_calls += 1

            t0 = time.perf_counter()
            try:
                response = self._oa_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=temp,
                    max_tokens=tokens,
                    extra_body=extra_body,
                )
            except Exception:
                logger.exception("llm_tool_call_failed")
                raise
            finally:
                self._record_call(time.perf_counter() - t0)

            choice = response.choices[0]

            if choice.finish_reason != "tool_calls" and (
                not choice.message.tool_calls
            ):
                content = choice.message.content or ""
                text = _strip_thinking_with_warning(content)
                return ToolCallingResult(
                    text=text,
                    tool_traces=traces,
                    num_llm_calls=total_llm_calls,
                    num_tool_calls=total_tool_calls,
                )

            assistant_msg = choice.message
            tool_calls_payload: list[dict[str, Any]] = []
            for tc in assistant_msg.tool_calls or []:
                tool_calls_payload.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

            assistant_payload: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content or "",
            }
            if tool_calls_payload:
                assistant_payload["tool_calls"] = tool_calls_payload
            messages.append(assistant_payload)

            if not assistant_msg.tool_calls:
                logger.warning("tool_call_finish_without_calls")
                continue

            new_executions_this_turn = 0
            for tool_call in assistant_msg.tool_calls or []:
                fn_name = tool_call.function.name
                raw_args = tool_call.function.arguments

                args = _parse_tool_arguments(raw_args)
                query = _resolve_tool_query(fn_name, args)
                executor = tool_executors.get(fn_name)

                if executor is not None:
                    if not query:
                        result = (
                            f"[Tool '{fn_name}' received empty arguments: {args}]"
                        )
                        logger.warning(
                            "empty_tool_call_arguments",
                            tool_name=fn_name,
                            arguments=args,
                        )
                    else:
                        cached_result = cache.get(fn_name, query)
                        if cached_result is not None:
                            result = cached_result
                            logger.info("tool_call_cache_hit", tool=fn_name, query=query[:200])
                        else:
                            result = executor.safe_call(query)
                            cache.set(fn_name, query, result)
                            total_tool_calls += 1
                            new_executions_this_turn += 1
                else:
                    # Tool not enabled this round. Tell the model exactly
                    # which tools ARE available so it can adapt next turn
                    # rather than repeating the bogus call.
                    available_str = (
                        ", ".join(available_tool_names)
                        if available_tool_names else "(none)"
                    )
                    result = (
                        f"Error: tool '{fn_name}' is NOT enabled this round. "
                        f"The only tools available are: {available_str}. "
                        f"Do not call '{fn_name}' again."
                    )
                    logger.warning(
                        "unknown_tool_in_call",
                        tool_name=fn_name,
                        available=available_tool_names,
                    )

                traces.append(ToolCallTrace(
                    tool_name=fn_name,
                    arguments=args,
                    result=result,
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                logger.info(
                    "tool_call_executed",
                    tool=fn_name,
                    query=query[:200],
                    result_preview=result[:200],
                )

            # If this turn yielded no new evidence (all cache hits or all
            # rejected unknown tools), the model is looping. Break out and
            # let the fallback push it to a final text response.
            if new_executions_this_turn == 0:
                logger.warning(
                    "tool_loop_no_new_progress",
                    num_tool_calls_in_turn=len(assistant_msg.tool_calls or []),
                )
                loop_exit_reason = "no_new_progress"
                break

        # Tool loop exhausted (or broke early on no-progress). Push the model
        # toward a plain-text final response using only what's already been
        # gathered above — without this nudge gpt-oss-20b in particular tends
        # to return empty text after a long tool-call sequence.
        logger.warning(
            "max_tool_rounds_exhausted",
            max_tool_rounds=max_tool_rounds,
            num_tool_calls=total_tool_calls,
            exit_reason=loop_exit_reason,
        )
        messages.append({
            "role": "user",
            "content": (
                "You have used your tool-call budget for this turn. "
                "Using ONLY the evidence and tool results already gathered "
                "above, write your final response now in plain text. "
                "Do NOT request any more tools. "
                "If the question expects a specific answer, end your response "
                "with a line in the form 'ANSWER: <your answer>'."
            ),
        })
        total_llm_calls += 1
        try:
            text = self._chat_once(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                enable_thinking=enable_thinking,
                tools=None,
            )
        except Exception:
            logger.exception("llm_fallback_call_failed")
            text = ""

        # Same empty-content recovery as the bare complete() path. The
        # fallback nudge usually elicits a response, but reasoning-model
        # token exhaustion can still leave content empty.
        if not text:
            logger.warning("tool_fallback_returned_empty_retrying")
            messages.append({
                "role": "user",
                "content": (
                    "[The previous attempt returned no visible content. "
                    "Respond NOW in plain text — keep it concise — and "
                    "end with 'ANSWER: <answer>' if applicable.]"
                ),
            })
            total_llm_calls += 1
            try:
                text = self._chat_once(
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    enable_thinking=False,
                    tools=None,
                )
            except Exception:
                logger.exception("llm_fallback_retry_failed")
                text = ""
            logger.warning("tool_fallback_retry_after_empty", recovered=bool(text))

        return ToolCallingResult(
            text=text,
            tool_traces=traces,
            num_llm_calls=total_llm_calls,
            num_tool_calls=total_tool_calls,
        )

# ---------------------------------------------------------------------------
# Per-role container
# ---------------------------------------------------------------------------

@dataclass
class LLMClients:
    """Role-specific clients for the agent/judge split.

    The ``orchestrator`` field is retained for forward compatibility with
    reward-model training (the orchestrator will be *trained* on the
    dataset this package produces). For one-shot data collection it is
    unused and may be ``None`` — the ``[llm.orchestrator]`` config section
    is therefore optional in this package.
    """

    agent: LLMClient
    judge: LLMClient
    orchestrator: LLMClient | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LLMClients":
        """Build clients from a parsed ``config.toml`` dict.

        Expects ``[llm.agent]`` and ``[llm.judge]`` sections. The
        ``[llm.orchestrator]`` section is optional — when absent or when
        its ``model`` field is empty, the orchestrator slot stays ``None``.
        A ``[llm.shared]`` section, if present, supplies defaults each
        role may override.
        """
        llm_cfg = cfg.get("llm", {})
        shared = dict(llm_cfg.get("shared", {}))

        def merge(role: str) -> dict[str, Any]:
            section = dict(llm_cfg.get(role, {}))
            return {**shared, **section}

        orchestrator: LLMClient | None = None
        orch_section = llm_cfg.get("orchestrator")
        if orch_section and orch_section.get("model"):
            orchestrator = LLMClient(LLMConfig.from_dict(merge("orchestrator")))

        return cls(
            agent=LLMClient(LLMConfig.from_dict(merge("agent"))),
            judge=LLMClient(LLMConfig.from_dict(merge("judge"))),
            orchestrator=orchestrator,
        )

    def shutdown(self) -> None:
        clients = [self.agent, self.judge]
        if self.orchestrator is not None:
            clients.append(self.orchestrator)
        for client in clients:
            client.shutdown()
