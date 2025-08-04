# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Base class for OpenAI-compatible clients with shared logic."""

import json
from abc import ABC, abstractmethod
from typing import override

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition

from ...tools.base import Tool, ToolCall
from ..base_client import BaseLLMClient
from ..config import ModelParameters
from ..llm_basics import LLMMessage, LLMResponse, LLMUsage
from ..retry_utils import retry_with
from trae_agent.cli import console


class ProviderConfig(ABC):
    """Abstract base class for provider-specific configurations."""

    @abstractmethod
    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create the OpenAI client instance."""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        pass

    @abstractmethod
    def get_extra_headers(self) -> dict[str, str]:
        """Get any extra headers needed for the API call."""
        pass

    @abstractmethod
    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        pass


class OpenAICompatibleClient(BaseLLMClient):  # OpenAI适配模型商使用
    """Base class for OpenAI-compatible clients with shared logic."""

    def __init__(self, model_parameters: ModelParameters,
                 provider_config: ProviderConfig):
        super().__init__(model_parameters)
        self.provider_config = provider_config
        self.client = provider_config.create_client(
            self.api_key, self.base_url, self.api_version)  # openai.OpenAI
        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_response(
        self,
        model_parameters: ModelParameters,
        tool_schemas: list[ChatCompletionToolParam] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Create a response using the provider's API. This method will be decorated with retry logic."""
        return self.client.chat.completions.create(
            model=model_parameters.model,  # 模型名称
            messages=self.message_history,  # 上下文历史
            tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,  # 可调用工具
            temperature=model_parameters.temperature,
            top_p=model_parameters.top_p, # top_p=0.1，则考虑概率最高的极少数词汇; top_p=0.9则会从概率占90%的词汇中采样
            max_tokens=model_parameters.max_tokens, # 输出最大tokens
            extra_headers=extra_headers if extra_headers else None,
            n=1,
        ) # 没有指定stream字段的布尔值，采用blocking模式进行输出

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_parameters: ModelParameters,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages with optional tool support."""
        # 转换为与大模型交互的消息格式, openai格式兼容
        parsed_messages = self.parse_messages(messages)
        if reuse_history:  # 记录之前的输入历史
            self.message_history = self.message_history + parsed_messages  # list加法
        else:
            self.message_history = parsed_messages
        # console.print(f"\n[green]message_history is {self.message_history}[/green]")

        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.get_name(),
                        description=tool.get_description(),
                        parameters=tool.get_input_schema(),
                    ),
                    type="function",
                ) for tool in tools
            ]

        # Get provider-specific extra headers
        extra_headers = self.provider_config.get_extra_headers()

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_response, # 底层调用openai
            service_name=self.provider_config.get_service_name(),  # Qwen
            max_retries=model_parameters.max_retries,
        )
        response = retry_decorator(model_parameters, tool_schemas,
                                   extra_headers)

        choice = response.choices[0] # 拿到response的相关内容，blocking模式

        tool_calls: list[ToolCall] | None = None
        if choice.message.tool_calls:
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        call_id=tool_call.id,
                        arguments=(json.loads(tool_call.function.arguments)
                                   if tool_call.function.arguments else {}),
                    ))

        llm_response = LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=(LLMUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            ) if response.usage else None), # usage用量
        )

        # print the model response
        console.print(f"\n[yellow]llm_response is {llm_response}[/yellow]")

        # Update message history
        if llm_response.tool_calls:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", # 模型自己的输出以assistant方式加入到上下文
                    content=llm_response.content,
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.call_id,
                            function=Function(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.arguments),
                            ),
                            type="function",
                        ) for tool_call in llm_response.tool_calls
                    ],
                ))
        elif llm_response.content:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    content=llm_response.content, role="assistant")) # 添加上下文 模型输出

        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction( # 记录一次模型交互
                messages=messages,
                response=llm_response,
                provider=self.provider_config.get_provider_name(),
                model=model_parameters.model,
                tools=tools,
            )

        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""
        return self.provider_config.supports_tool_calling(
            model_parameters.model)

    def parse_messages(
            self,
            messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        """Parse LLM messages to OpenAI format."""
        # 转化消息为与大模型进行交互的信息，补全对话类消息格式
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            match msg:
                case msg if msg.tool_call is not None:
                    _msg_tool_call_handler(openai_messages, msg)
                case msg if msg.tool_result is not None:
                    _msg_tool_result_handler(openai_messages, msg)
                case msg if msg.role is not None:
                    _msg_role_handler(openai_messages, msg)
                case _:
                    raise ValueError(f"Invalid message: {msg}")

        return openai_messages

# 格式化tool_call信息函数
def _msg_tool_call_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.tool_call:
        messages.append(
            ChatCompletionFunctionMessageParam(
                content=json.dumps(
                    {
                        "name": msg.tool_call.name,
                        "arguments": msg.tool_call.arguments,
                    }
                ),
                role="function",
                name=msg.tool_call.name,
            )
        )

# 格式化tool_result信息函数
def _msg_tool_result_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.tool_result:
        result: str = ""
        if msg.tool_result.result:
            result = result + msg.tool_result.result + "\n"
        if msg.tool_result.error: # 错误信息打印
            result += "Tool call failed with error:\n"
            result += msg.tool_result.error
        result = result.strip()
        messages.append(
            ChatCompletionToolMessageParam(
                content=result,
                role="tool",
                tool_call_id=msg.tool_result.call_id,
            )
        )

# 格式化role信息格式
def _msg_role_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.role:
        match msg.role:
            case "system":
                if not msg.content:
                    raise ValueError("System message content is required")
                messages.append(
                    ChatCompletionSystemMessageParam(content=msg.content, role="system")
                )
            case "user":
                if not msg.content:
                    raise ValueError("User message content is required")
                messages.append(ChatCompletionUserMessageParam(content=msg.content, role="user"))
            case "assistant":
                if not msg.content:
                    raise ValueError("Assistant message content is required")
                messages.append(
                    ChatCompletionAssistantMessageParam(content=msg.content, role="assistant")
                )
            case _:
                raise ValueError(f"Invalid message role: {msg.role}")
