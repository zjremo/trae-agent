# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenRouter provider configuration."""

import os

import openai

from .openai_compatible_base import ProviderConfig


class OpenRouterProvider(ProviderConfig):
    """OpenRouter provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create OpenAI client with OpenRouter base URL."""
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "OpenRouter"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "openrouter"

    def get_extra_headers(self) -> dict[str, str]:
        """Get OpenRouter-specific headers."""
        extra_headers: dict[str, str] = {}

        openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
        if openrouter_site_url:
            extra_headers["HTTP-Referer"] = openrouter_site_url

        openrouter_site_name = os.getenv("OPENROUTER_SITE_NAME")
        if openrouter_site_name:
            extra_headers["X-Title"] = openrouter_site_name

        return extra_headers

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Most modern models on OpenRouter support tool calling
        # We'll be conservative and check for known capable models
        tool_capable_patterns = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3",
            "claude-2",
            "gemini",
            "mistral",
            "llama-3",
            "command-r",
        ]
        return any(pattern in model_name.lower() for pattern in tool_capable_patterns)
