"""Pydantic v2 models for Schema, Round, and related types.

Schemas in this package have no early-stopping mechanism: every defined
round is executed in order. The previous ``StopCondition`` and
``Round.stop_condition`` field have been removed.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PersonaType(str, Enum):
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    # Used only in self-consistency mode: a single generic persona that
    # the orchestrator samples ``n_samples`` times in parallel.
    GENERIC_ASSISTANT = "generic_assistant"


class ToolType(str, Enum):
    SEARCH_INFO = "search_info"
    FETCH_URL = "fetch_url"
    CODE_COMPUTE = "code_compute"


class InstructionType(str, Enum):
    INDEPENDENTLY_RESEARCH = "independently_research"
    DEBATE_AND_REFINE = "debate_and_refine"
    CRITIQUE_PREVIOUS = "critique_previous"
    GATHER_FACTS = "gather_facts"
    PRODUCE_FINAL_ANSWER = "produce_final_answer"
    VERIFY_AND_CHECK = "verify_and_check"
    BRAINSTORM = "brainstorm"
    THINK_AND_PLAN = "think_and_plan"


class SynthesisMethod(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    LAST_PERSONA = "last_persona"
    SYNTHESIZER_PERSONA = "synthesizer_persona"


class DebateMode(str, Enum):
    """Top-level debate strategy.

    - ``persona``: classic 3-persona debate (analyst / critic / synthesizer).
    - ``self_consistency``: generic-assistant rounds sampled ``n_samples``
      times in parallel; no early stop conditions.
    """

    PERSONA = "persona"
    SELF_CONSISTENCY = "self_consistency"


class Round(BaseModel):
    round: int = Field(ge=1)
    # Up to 5 entries to allow persona-mode amplification (analyst x2 +
    # critic x2 + synthesizer x1). Self-consistency mode enforces exactly
    # one persona via validator.py.
    personas: list[PersonaType] = Field(min_length=1, max_length=5)
    tools: list[ToolType] = Field(default_factory=list)
    instruction: InstructionType
    # Self-consistency only: number of parallel samples of the round's
    # single persona. Always 1 in persona-mode rounds.
    n_samples: int = Field(default=1, ge=1, le=8)


class Schema(BaseModel):
    max_rounds: int = Field(ge=1, le=10)
    final_synthesis: SynthesisMethod
    rounds: list[Round] = Field(min_length=1)
    mode: DebateMode = DebateMode.PERSONA


class Task(BaseModel):
    """A single benchmark question.

    ``level`` is a numeric difficulty (defaults to 1 when the source
    dataset doesn't expose one). ``difficulty`` is the string label some
    datasets ship — e.g. HotPotQA's ``easy``/``medium``/``hard`` — and is
    None when the source has no such label.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    level: int = 1
    difficulty: str | None = None
    question: str
    ground_truth: str
    file_name: str = ""
    keep: int = 1
    source: str = ""  # e.g. "natural_questions" / "hotpot_qa"
