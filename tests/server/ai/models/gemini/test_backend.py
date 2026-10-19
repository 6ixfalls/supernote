"""Tests for `supernote.server.ai.models.gemini.backend`."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from supernote.server.ai.base import AIError
from supernote.server.ai.models.embedding import EmbeddingTaskType
from supernote.server.ai.models.gemini.backend import GeminiBackend
from supernote.server.config import ServerConfig


async def test_gemini_backend_translates_client_errors_to_ai_error() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        backend = GeminiBackend(api_key="fake-key")
        mock_client = mock_client_cls.return_value
        backend._client = mock_client

        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("network timeout")
        )
        with pytest.raises(AIError, match="Unexpected error calling Gemini API"):
            await backend.generate_content(model="gemini-2.0-flash", contents="hello")

        mock_client.aio.models.embed_content = AsyncMock(
            side_effect=RuntimeError("network timeout")
        )
        with pytest.raises(AIError, match="Unexpected error calling Gemini API"):
            await backend.embed_content(model="text-embedding-004", contents="hello")


@pytest.mark.parametrize("task", ["ocr", "summary", "embedding"])
async def test_vertex_flex_through_task_factory(task: str) -> None:
    """The AI refactor must retain the fork's Vertex/Flex transport settings."""

    config = ServerConfig(
        gemini_provider="vertex", gemini_project="project", gemini_flex=True
    )
    factory = getattr(
        import_module(f"supernote.server.ai.models.{task}"), f"create_{task}_model"
    )
    with patch("google.genai.Client") as client_cls:
        model = factory(config)
        assert model is not None
        client = client_cls.return_value
        if task == "embedding":
            client.aio.models.embed_content = AsyncMock(
                return_value=types.EmbedContentResponse(
                    embeddings=[types.ContentEmbedding(values=[1.0, 2.0])]
                )
            )
            await model.embed("hello", EmbeddingTaskType.RETRIEVAL_QUERY)
        else:
            client.aio.models.generate_content = AsyncMock(
                return_value=types.GenerateContentResponse()
            )
            if task == "ocr":
                await model.extract_text(b"png", "transcribe")
                assert (
                    client.aio.models.generate_content.call_args.kwargs["contents"][
                        0
                    ].role
                    == "user"
                )
            else:
                await model.summarize("summarize")
            assert (
                client.aio.models.generate_content.call_args.kwargs[
                    "config"
                ].service_tier
                is None
            )
        client_cls.assert_called_once_with(
            vertexai=True,
            project="project",
            location="global",
            http_options={
                "headers": {
                    "X-Vertex-AI-LLM-Request-Type": "shared",
                    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
                }
            },
        )


async def test_backend_google_flex_preserves_request_config() -> None:
    backend = GeminiBackend(api_key="key", flex=True)
    with patch("google.genai.Client") as client_cls:
        client_cls.return_value.aio.models.generate_content = AsyncMock(
            return_value=types.GenerateContentResponse()
        )
        config = types.GenerateContentConfig(response_mime_type="application/json")
        await backend.generate_content(model="model", contents="prompt", config=config)
        sent = client_cls.return_value.aio.models.generate_content.call_args.kwargs[
            "config"
        ]
        assert sent.service_tier == types.ServiceTier.FLEX
        assert sent.response_mime_type == "application/json"
        assert config.service_tier is None
