import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, cast

from supernote.server.config import ServerConfig
from supernote.server.metrics import GEMINI_API_CALLS_TOTAL, GEMINI_API_DURATION_SECONDS

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = frozenset({"google", "vertex"})

VERTEX_FLEX_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
}


class GeminiService:
    """Shared service for interacting with Google Gemini models.

    Routes requests to either the Gemini Developer API (`google`) using an
    API key, or to Vertex AI (`vertex`) using Application Default
    Credentials with an explicit project and optional location.
    """

    def __init__(self, config: ServerConfig) -> None:
        self.provider = config.gemini_provider
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported Gemini provider {self.provider!r}; expected one of "
                f"{sorted(SUPPORTED_PROVIDERS)}"
            )
        self.flex = config.gemini_flex
        self.max_concurrency = config.gemini_max_concurrency
        self.api_key = config.gemini_api_key
        self.project = config.gemini_project
        self.location = config.gemini_location
        self._client: "genai.Client | None" = None
        self._semaphore: asyncio.Semaphore | None = None
        if self.provider == "vertex":
            if self.project:
                # Deferred: google-genai is a heavy import (pulls in ~300
                # transitive modules), so only pay for it when configured.
                from google import genai  # noqa: PLC0415

                kwargs: dict[str, Any] = {
                    "vertexai": True,
                    "project": self.project,
                }
                location = self.location
                if not location and self.flex:
                    # Flex PayGo is only served on the global endpoint.
                    location = "global"
                if location:
                    kwargs["location"] = location
                if self.flex:
                    # Vertex AI selects Flex PayGo via request headers
                    # instead of the service_tier config field.
                    kwargs["http_options"] = {"headers": dict(VERTEX_FLEX_HEADERS)}
                self._client = genai.Client(**kwargs)
        elif self.api_key:
            # Deferred: google-genai is a heavy import (pulls in ~300
            # transitive modules), so only pay for it when an API key is
            # actually configured.
            from google import genai  # noqa: PLC0415

            self._client = genai.Client(
                api_key=self.api_key, http_options={"api_version": "v1alpha"}
            )

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def _require_client(self) -> "genai.Client":
        """Return the configured client, raising when the service is off."""
        if self._client is None:
            if self.provider == "vertex":
                raise ValueError(
                    "Vertex AI provider requires gemini_project to be configured"
                )
            raise ValueError("Gemini API key not configured")
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of semaphore to ensure it's in the correct event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

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
        contents: Any,
        config: "types.GenerateContentConfigOrDict | None" = None,
    ) -> "types.GenerateContentResponse":
        """Asynchronously generate content using the Gemini API."""
        client = self._require_client()

        start_time = time.perf_counter()
        status = "success"
        try:
            async with self._get_semaphore():
                return await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=self._apply_config_defaults(config),
                )
        except Exception:
            status = "failure"
            raise
        finally:
            duration = time.perf_counter() - start_time
            GEMINI_API_CALLS_TOTAL.labels(
                operation="generate_content", status=status
            ).inc()
            GEMINI_API_DURATION_SECONDS.labels(operation="generate_content").observe(
                duration
            )

    async def embed_content(
        self,
        model: str,
        contents: Any,
        config: "types.EmbedContentConfigOrDict | None" = None,
    ) -> "types.EmbedContentResponse":
        """Asynchronously generate embeddings using the Gemini API."""
        client = self._require_client()

        start_time = time.perf_counter()
        status = "success"
        try:
            async with self._get_semaphore():
                return await client.aio.models.embed_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
        except Exception:
            status = "failure"
            raise
        finally:
            duration = time.perf_counter() - start_time
            GEMINI_API_CALLS_TOTAL.labels(
                operation="embed_content", status=status
            ).inc()
            GEMINI_API_DURATION_SECONDS.labels(operation="embed_content").observe(
                duration
            )
