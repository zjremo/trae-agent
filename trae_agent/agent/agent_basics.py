# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..tools.base import ToolCall, ToolResult
from ..utils.llm_basics import LLMResponse, LLMUsage

__all__ = [
    "AgentState",
    "AgentStep",
    "AgentExecution",
    "AgentError",
]


class AgentState(Enum):
    """Defines possible states during an agent's execution lifecycle."""

    IDLE = "idle"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentStep:
    """
    Represents a single step in an agent's execution process.

    Tracks the state, thought process, tool interactions, LLM response,
    and any associated metadata or errors.
    """

    step_number: int
    state: AgentState
    thought: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_results: Optional[list[ToolResult]] = None
    llm_response: Optional[LLMResponse] = None
    reflection: Optional[str] = None
    error: Optional[str] = None
    extra: Optional[dict[str, object]] = None
    llm_usage: Optional[LLMUsage] = None

    def __repr__(self) -> str:
        return (
            f"<AgentStep #{self.step_number} "
            f"state={self.state.name} "
            f"thought={repr(self.thought)[:40]}...>"
        )


@dataclass
class AgentExecution:
    """
    Encapsulates the entire execution of an agent task.

    Contains the original task, all intermediate steps,
    final result, execution metadata, and success state.
    """

    task: str
    steps: list[AgentStep]
    final_result: Optional[str] = None
    success: bool = False
    total_tokens: Optional[LLMUsage] = None
    execution_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"<AgentExecution task={self.task!r} "
            f"steps={len(self.steps)} "
            f"success={self.success}>"
        )


class AgentError(Exception):
    """
    Base class for agent-related errors.

    Used to signal execution failures, misconfigurations,
    or unexpected LLM/tool behavior.
    """

    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"<AgentError message={self.message!r}>"
