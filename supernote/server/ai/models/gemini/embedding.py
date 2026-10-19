"""Gemini-backed text embedding model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.genai import types

from supernote.server.ai.base import AIError
from supernote.server.ai.models.embedding import EmbeddingModel, EmbeddingTaskType

from .backend import GeminiBackend

if TYPE_CHECKING:
    from supernote.server.config import ServerConfig

__all__ = [
    "GeminiEmbeddingModel",
    "create_embedding_model",
]


class GeminiEmbeddingModel(EmbeddingModel):
    """Gemini-backed text embedding model."""

    def __init__(self, backend: GeminiBackend, model_name: str) -> None:
        self._backend = backend
        self._model_name = model_name

    @property
    def model_id(self) -> str:
        return self._model_name

    async def _embed(
        self,
        text: str,
        task_type: EmbeddingTaskType,
    ) -> list[float]:
        response = await self._backend.embed_content(
            model=self._model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type.value),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise AIError("No embeddings returned from Gemini API")
        return list(response.embeddings[0].values)


def create_embedding_model(config: ServerConfig) -> EmbeddingModel | None:
    """Create a `GeminiEmbeddingModel` from `ServerConfig`, or `None` if unconfigured."""
    if not config.gemini_is_configured:
        return None
    return GeminiEmbeddingModel(
        GeminiBackend.from_config(config), config.gemini_embedding_model
    )
