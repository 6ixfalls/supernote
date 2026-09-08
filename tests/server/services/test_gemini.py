from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from supernote.server.config import ServerConfig
from supernote.server.services.gemini import GeminiService


def test_google_provider_builds_client_with_api_key() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiService(ServerConfig(gemini_api_key="fake-key"))

    mock_client_cls.assert_called_once_with(
        api_key="fake-key", http_options={"api_version": "v1alpha"}
    )
    assert service.is_configured


def test_google_provider_without_api_key_is_unconfigured() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiService(ServerConfig())

    mock_client_cls.assert_not_called()
    assert not service.is_configured


def test_vertex_provider_builds_client_with_project_and_location() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiService(
            ServerConfig(
                gemini_provider="vertex",
                gemini_project="my-project",
                gemini_location="europe-west4",
            )
        )

    mock_client_cls.assert_called_once_with(
        vertexai=True, project="my-project", location="europe-west4"
    )
    assert service.is_configured


def test_vertex_provider_defaults_location() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        GeminiService(ServerConfig(gemini_provider="vertex", gemini_project="p"))

    mock_client_cls.assert_called_once_with(vertexai=True, project="p")


def test_vertex_provider_ignores_api_key() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiService(
            ServerConfig(gemini_provider="vertex", gemini_api_key="fake-key")
        )

    mock_client_cls.assert_not_called()
    assert not service.is_configured


def test_vertex_provider_without_project_is_unconfigured() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiService(ServerConfig(gemini_provider="vertex"))

    mock_client_cls.assert_not_called()
    assert not service.is_configured


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported Gemini provider"):
        GeminiService(ServerConfig(gemini_provider="bedrock"))


async def test_generate_content_applies_flex_service_tier() -> None:
    with patch("google.genai.Client"):
        service = GeminiService(
            ServerConfig(gemini_api_key="fake-key", gemini_flex=True)
        )
    service._client = MagicMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=MagicMock())

    await service.generate_content(
        model="gemini-3.6-flash",
        contents="hello",
        config={"media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH},
    )

    _, kwargs = service._client.aio.models.generate_content.call_args
    assert kwargs["config"] == {
        "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        "service_tier": types.ServiceTier.FLEX,
    }


async def test_generate_content_without_flex_keeps_config() -> None:
    with patch("google.genai.Client"):
        service = GeminiService(ServerConfig(gemini_api_key="fake-key"))
    service._client = MagicMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=MagicMock())

    await service.generate_content(
        model="gemini-3.6-flash",
        contents="hello",
        config={"media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH},
    )

    _, kwargs = service._client.aio.models.generate_content.call_args
    assert kwargs["config"] == {
        "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH
    }


async def test_generate_content_with_flex_and_no_config() -> None:
    with patch("google.genai.Client"):
        service = GeminiService(
            ServerConfig(gemini_api_key="fake-key", gemini_flex=True)
        )
    service._client = MagicMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=MagicMock())

    await service.generate_content(model="gemini-3.6-flash", contents="hello")

    _, kwargs = service._client.aio.models.generate_content.call_args
    assert kwargs["config"] == {"service_tier": types.ServiceTier.FLEX}


def test_vertex_provider_with_flex_sets_flex_headers() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        GeminiService(
            ServerConfig(
                gemini_provider="vertex",
                gemini_project="my-project",
                gemini_flex=True,
            )
        )

    mock_client_cls.assert_called_once_with(
        vertexai=True,
        project="my-project",
        location="global",
        http_options={
            "headers": {
                "X-Vertex-AI-LLM-Request-Type": "shared",
                "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
            }
        },
    )


def test_vertex_provider_with_flex_keeps_explicit_location() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        GeminiService(
            ServerConfig(
                gemini_provider="vertex",
                gemini_project="p",
                gemini_location="europe-west4",
                gemini_flex=True,
            )
        )

    assert mock_client_cls.call_args.kwargs["location"] == "europe-west4"


def test_vertex_provider_without_flex_omits_headers() -> None:
    with patch("google.genai.Client") as mock_client_cls:
        GeminiService(ServerConfig(gemini_provider="vertex", gemini_project="p"))

    assert "http_options" not in mock_client_cls.call_args.kwargs


async def test_vertex_flex_does_not_set_service_tier() -> None:
    with patch("google.genai.Client"):
        service = GeminiService(
            ServerConfig(
                gemini_provider="vertex",
                gemini_project="my-project",
                gemini_flex=True,
            )
        )
    service._client = MagicMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=MagicMock())

    await service.generate_content(
        model="gemini-3.6-flash",
        contents="hello",
        config={"media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH},
    )

    _, kwargs = service._client.aio.models.generate_content.call_args
    assert kwargs["config"] == {
        "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH
    }


def test_require_client_error_messages() -> None:
    google_service = GeminiService(ServerConfig())
    with pytest.raises(ValueError, match="Gemini API key not configured"):
        google_service._require_client()

    vertex_service = GeminiService(ServerConfig(gemini_provider="vertex"))
    with pytest.raises(ValueError, match="gemini_project"):
        vertex_service._require_client()
