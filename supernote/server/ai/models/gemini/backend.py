"""Google Gemini client wrapper with Prometheus instrumentation."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from supernote.server.config import ServerConfig

from google import genai
from google.genai import errors, types

from supernote.server.ai.base import AIError
from supernote.server.metrics import GEMINI_API_CALLS_TOTAL, GEMINI_API_DURATION_SECONDS

__all__ = [
    "GeminiBackend",
]

logger = logging.getLogger(__name__)


class GeminiBackend:
    """Google Gemini client wrapper with Prometheus instrumentation."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: str = "google",
        project: str | None = None,
        location: str | None = None,
        flex: bool = False,
    ) -> None:
        if provider not in {"google", "vertex"}:
            raise ValueError(f"Unsupported Gemini provider {provider!r}")
        self._api_key = api_key
        self.provider = provider
        self.project = project
        self.location = location
        self.flex = flex
        self._client: genai.Client | None = None

    @classmethod
    def from_config(cls, config: ServerConfig) -> GeminiBackend:
        """Adapt legacy Gemini connection options to task-specific models."""
        return cls(
            config.gemini_api_key,
            provider=config.gemini_provider,
            project=config.gemini_project,
            location=config.gemini_location,
            flex=config.gemini_flex,
        )

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if self.provider == "vertex":
                if not self.project:
                    raise ValueError("Vertex AI provider requires gemini_project")
                kwargs: dict[str, Any] = {"vertexai": True, "project": self.project}
                location = self.location or ("global" if self.flex else None)
                if location:
                    kwargs["location"] = location
                if self.flex:
                    kwargs["http_options"] = {
                        "headers": {
                            "X-Vertex-AI-LLM-Request-Type": "shared",
                            "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
                        }
                    }
                self._client = genai.Client(**kwargs)
            else:
                self._client = genai.Client(
                    api_key=self._api_key, http_options={"api_version": "v1alpha"}
                )
        return self._client

    @contextmanager
    def _track_api_call(self, operation: str) -> Generator[None]:
        start_time = time.monotonic()
        status = "success"
        try:
            yield
        except errors.APIError as err:
            status = "failure"
            logger.error(f"Gemini API error during {operation}: {err}")
            raise AIError(f"Gemini API error during {operation}: {err}") from err
        except Exception as err:
            status = "failure"
            logger.error(f"Unexpected error calling Gemini API ({operation}): {err}")
            raise AIError(
                f"Unexpected error calling Gemini API ({operation}): {err}"
            ) from err
        finally:
            duration = time.monotonic() - start_time
            GEMINI_API_CALLS_TOTAL.labels(operation=operation, status=status).inc()
            GEMINI_API_DURATION_SECONDS.labels(operation=operation).observe(duration)

    def _apply_config_defaults(
        self, config: "types.GenerateContentConfigOrDict | None"
    ) -> "types.GenerateContentConfigOrDict | None":
        """Merge the configured Flex service tier into a request config.

        Only applies to the `google` provider; Vertex AI selects Flex
        PayGo through request headers set on the client instead.
        """
        if not self.flex or self.provider != "google":
            return config
        # google-genai is only needed once a client is configured, which is
        # guaranteed by the time a request is built.
        from google.genai import types as genai_types  # noqa: PLC0415

        service_tier = genai_types.ServiceTier.FLEX
        if config is None:
            return {"service_tier": service_tier}
        if isinstance(config, genai_types.GenerateContentConfig):
            if config.service_tier is None:
                return config.model_copy(update={"service_tier": service_tier})
            return config
        merged: dict[str, Any] = dict(config)
        merged.setdefault("service_tier", service_tier)
        return cast("types.GenerateContentConfigDict", merged)

    async def generate_content(
        self,
        model: str,
        contents: types.ContentListUnion | types.ContentListUnionDict,
        config: types.GenerateContentConfigOrDict | None = None,
    ) -> types.GenerateContentResponse:
        """Call Gemini `generate_content` with Prometheus instrumentation."""
        with self._track_api_call("generate_content"):
            return await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=self._apply_config_defaults(config),
            )

    async def embed_content(
        self,
        model: str,
        contents: types.ContentListUnion | types.ContentListUnionDict,
        config: types.EmbedContentConfigOrDict | None = None,
    ) -> types.EmbedContentResponse:
        """Call Gemini `embed_content` with Prometheus instrumentation."""
        with self._track_api_call("embed_content"):
            return await self.client.aio.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
