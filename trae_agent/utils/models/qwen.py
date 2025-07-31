# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Doubao provider configuration."""

import openai

from .openai_compatible_base import ProviderConfig


class QwenProvider(ProviderConfig):
    """Qwen provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create OpenAI client with Qwen base URL."""
        return openai.OpenAI(base_url=base_url, api_key=api_key) # openai库中创建模型客户端

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "Qwen"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "qwen"

    def get_extra_headers(self) -> dict[str, str]:
        """Get Qwen-specific headers (none needed)."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Qwen models generally support tool calling
        return True
