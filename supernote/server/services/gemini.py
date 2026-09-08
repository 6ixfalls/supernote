import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pydantic

from supernote.server.config import ServerConfig
from supernote.server.metrics import GEMINI_API_CALLS_TOTAL, GEMINI_API_DURATION_SECONDS

if TYPE_CHECKING:
    import ai

logger = logging.getLogger(__name__)

GEMINI_MODEL_NAMESPACE = "google"
FLEX_SERVICE_TIER = "flex"
SUPPORTED_GATEWAY_PROVIDERS = frozenset({"google", "vertex"})


class GeminiService:
    """Shared service for Gemini models via the Vercel AI SDK (AI Gateway)."""

    def __init__(self, config: ServerConfig) -> None:
        self.provider = config.ai.provider
        if self.provider not in SUPPORTED_GATEWAY_PROVIDERS:
            raise ValueError(
                f"Unsupported AI provider {self.provider!r}; expected one of "
                f"{sorted(SUPPORTED_GATEWAY_PROVIDERS)}"
            )
        self.flex = config.ai.flex
        self.zdr = config.ai.zdr
        self.max_concurrency = config.ai.max_concurrency
        self.api_key = config.ai.api_key
        self._semaphore: asyncio.Semaphore | None = None
        self._sdk: ModuleType | None = None
        self._gateway: "ai.Provider | None" = None
        if self.api_key:
            # Deferred: the AI SDK is only needed once an API key is
            # actually configured, keeping key-less startup light.
            import ai  # noqa: PLC0415

            self._sdk = ai
            self._gateway = ai.get_provider("gateway", api_key=self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._gateway is not None

    def _require_sdk(self) -> ModuleType:
        """Return the AI SDK module, raising when the service is not configured."""
        if self._sdk is None or self._gateway is None:
            raise ValueError("AI SDK API key not configured")
        return self._sdk

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of semaphore to ensure it's in the correct event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def _model(self, sdk: ModuleType, model: str) -> "ai.Model":
        return sdk.Model(
            id=f"{GEMINI_MODEL_NAMESPACE}/{model}", provider=self._gateway
        )

    def _build_params(
        self, sdk: ModuleType, provider_options: Mapping[str, Any] | None = None
    ) -> "ai.InferenceRequestParams":
        """Build request params applying the configured flex and zdr settings."""
        kwargs: dict[str, Any] = {
            "routing": sdk.RoutingParams(
                provider_allowlist=frozenset({self.provider})
            )
        }
        request_provider_options: dict[str, Any] = {}
        if self.flex:
            # Use the Gateway-wide option so it is translated correctly for
            # both Google AI Studio and Vertex AI routing.
            request_provider_options["gateway"] = {
                "serviceTier": FLEX_SERVICE_TIER
            }
        if provider_options:
            # Gemini model options stay under the model provider namespace even
            # when the Gateway serves the request through Vertex AI.
            request_provider_options[GEMINI_MODEL_NAMESPACE] = dict(provider_options)
        if request_provider_options:
            kwargs["extra_body"] = {"providerOptions": request_provider_options}
        params = sdk.InferenceRequestParams(**kwargs)
        if self.zdr:
            from ai.providers.ai_gateway import GatewayParams  # noqa: PLC0415

            params = params.with_provider_params(
                GatewayParams(zero_data_retention=True)
            )
        return params

    async def generate_content(
        self,
        model: str,
        prompt: str,
        image: bytes | None = None,
        *,
        output_type: type[pydantic.BaseModel] | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> "ai.types.messages.Message":
        """Asynchronously generate content using the Vercel AI SDK."""
        sdk = self._require_sdk()

        start_time = time.perf_counter()
        status = "success"
        try:
            async with self._get_semaphore():
                if image is not None:
                    messages = [
                        sdk.user_message(
                            prompt,
                            sdk.file_part(image, media_type="image/png"),
                        )
                    ]
                else:
                    messages = [sdk.user_message(prompt)]
                return await sdk.experimental_generate(
                    self._model(sdk, model),
                    messages,
                    output_type=output_type,
                    params=self._build_params(sdk, provider_options),
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
        contents: Sequence[str],
    ) -> list[list[float]]:
        """Asynchronously generate embeddings using the Vercel AI SDK.

        Returns one vector per input string, in the same order.
        """
        sdk = self._require_sdk()

        start_time = time.perf_counter()
        status = "success"
        try:
            async with self._get_semaphore():
                gateway_options: dict[str, Any] = {"only": [self.provider]}
                if self.zdr:
                    gateway_options["zeroDataRetention"] = True
                result = await sdk.ops.embed(
                    self._model(sdk, model),
                    list(contents),
                    params=sdk.ops.EmbedParams(
                        provider_options={"gateway": gateway_options}
                    ),
                )
                return result.value
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
