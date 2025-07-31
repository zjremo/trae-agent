# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Qwen client wrapper with tool integrations"""

from .config import ModelParameters
from .models.qwen import QwenProvider
from .models.openai_compatible_base import OpenAICompatibleClient


class QwenClient(OpenAICompatibleClient):
    """Qwen client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters, QwenProvider())
