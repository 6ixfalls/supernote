from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from ai.providers.ai_gateway import GatewayParams

from supernote.server.config import ServerConfig
from supernote.server.services.gemini import GeminiService


async def test_generate_uses_model_namespace_and_gateway_routing() -> None:
    config = ServerConfig()
    config.ai.api_key = "fake-key"
    config.ai.provider = "vertex"
    config.ai.flex = True
    config.ai.zdr = True
    service = GeminiService(config)

    with patch(
        "ai.experimental_generate", new=AsyncMock(return_value=SimpleNamespace())
    ) as generate:
        await service.generate_content(
            model="gemini-3.6-flash",
            prompt="Transcribe this",
            provider_options={"mediaResolution": "MEDIA_RESOLUTION_HIGH"},
        )

    model, _ = generate.call_args.args
    params = generate.call_args.kwargs["params"]

    assert model.id == "google/gemini-3.6-flash"
    assert params.routing.provider_allowlist == frozenset({"vertex"})
    assert params.extra_body == {
        "providerOptions": {
            "gateway": {"serviceTier": "flex"},
            "google": {"mediaResolution": "MEDIA_RESOLUTION_HIGH"},
        }
    }
    assert params.provider_params[GatewayParams].zero_data_retention is True


async def test_embedding_applies_gateway_routing_and_zdr() -> None:
    config = ServerConfig()
    config.ai.api_key = "fake-key"
    config.ai.provider = "vertex"
    config.ai.zdr = True
    service = GeminiService(config)

    with patch(
        "ai.ops.embed",
        new=AsyncMock(return_value=SimpleNamespace(value=[[0.1, 0.2]])),
    ) as embed:
        result = await service.embed_content("gemini-embedding-001", ["hello"])

    model, values = embed.call_args.args
    params = embed.call_args.kwargs["params"]

    assert result == [[0.1, 0.2]]
    assert model.id == "google/gemini-embedding-001"
    assert values == ["hello"]
    assert params.provider_options == {
        "gateway": {"only": ["vertex"], "zeroDataRetention": True}
    }


def test_rejects_unknown_gateway_provider() -> None:
    config = ServerConfig()
    config.ai.provider = "unknown"

    with pytest.raises(ValueError, match="Unsupported AI provider"):
        GeminiService(config)
