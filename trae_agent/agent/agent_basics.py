# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum

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

    IDLE = "idle" # 空闲 初始状态
    THINKING = "thinking" # 思考中
    CALLING_TOOL = "calling_tool" # 调用工具中
    REFLECTING = "reflecting" # 反思优化
    COMPLETED = "completed" # 已完成
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
    thought: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    llm_response: LLMResponse | None = None
    reflection: str | None = None
    error: str | None = None
    extra: dict[str, object] | None = None
    llm_usage: LLMUsage | None = None

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
    final_result: str | None = None
    success: bool = False
    total_tokens: LLMUsage | None = None
    execution_time: float = 0.0

    def __repr__(self) -> str:
        return f"<AgentExecution task={self.task!r} steps={len(self.steps)} success={self.success}>"


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
