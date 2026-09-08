import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from mashumaro.config import TO_DICT_ADD_OMIT_NONE_FLAG, BaseConfig
from mashumaro.mixins.yaml import DataClassYAMLMixin

logger = logging.getLogger(__name__)


def _get_bool_env(name: str, default: bool) -> bool:
    """Get a boolean value from an environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


@dataclass
class AuthConfig(DataClassYAMLMixin):
    """Authentication configuration."""

    secret_key: str = ""
    """JWT secret key.

    Env Var: `SUPERNOTE_JWT_SECRET`
    """

    expiration_hours: int = 24
    """JWT expiration time in hours."""

    device_expiration_hours: int = 87600
    """JWT expiration time for devices in hours (default: 10 years)."""

    enable_registration: bool = False
    """When disabled, registration is only allowed if there are no users in the system.

    Env Var: `SUPERNOTE_ENABLE_REGISTRATION`
    """

    enable_remote_password_reset: bool = False
    """When disabled, the public password reset endpoint returns 403.

    Env Var: `SUPERNOTE_ENABLE_REMOTE_PASSWORD_RESET`
    """

    allow_unauthenticated_binds: bool = False
    """Allow legacy clients to bind devices without a valid session.

    When disabled, unauthenticated bind attempts require approval in the web UI.

    Env Var: `SUPERNOTE_ALLOW_UNAUTHENTICATED_BINDS`
    """

    class Config(BaseConfig):
        omit_none = True
        code_generation_options = [TO_DICT_ADD_OMIT_NONE_FLAG]  # type: ignore[list-item]


@dataclass
class AIConfig(DataClassYAMLMixin):
    """Configuration for AI features served through Vercel AI Gateway."""

    api_key: str | None = None
    """Vercel AI Gateway API key.

    Falls back to the `AI_GATEWAY_API_KEY` environment variable when unset.

    Env Var: `SUPERNOTE_AI_API_KEY`
    """

    provider: str = "google"
    """Gateway serving provider for Gemini models (`google` or `vertex`).

    Env Var: `SUPERNOTE_AI_PROVIDER`
    """

    ocr_model: str = "gemini-3.6-flash"
    """Gemini model to use for OCR and summarization.

    Env Var: `SUPERNOTE_AI_OCR_MODEL`
    """

    embedding_model: str = "gemini-embedding-001"
    """Gemini model to use for embeddings.

    Env Var: `SUPERNOTE_AI_EMBEDDING_MODEL`
    """

    max_concurrency: int = 5
    """Maximum number of concurrent AI API calls.

    Env Var: `SUPERNOTE_AI_MAX_CONCURRENCY`
    """

    flex: bool = False
    """Use Gemini Flex inference for lower-cost batch-tolerant processing.

    Env Var: `SUPERNOTE_AI_FLEX`
    """

    zdr: bool = False
    """Restrict Gateway routing to zero-data-retention providers.

    Env Var: `SUPERNOTE_AI_ZDR`
    """

    prompts_dir: str | None = None
    """Directory containing custom AI prompts.

    Env Var: `SUPERNOTE_AI_PROMPTS_DIR`
    """

    class Config(BaseConfig):
        omit_none = True
        code_generation_options = [TO_DICT_ADD_OMIT_NONE_FLAG]  # type: ignore[list-item]


@dataclass
class ServerConfig(DataClassYAMLMixin):
    host: str = "0.0.0.0"
    """Host to bind the server to.

    Env Var: `SUPERNOTE_HOST`
    """

    port: int = 8080
    """Port to bind the server to.

    Env Var: `SUPERNOTE_PORT`
    """

    mcp_port: int = 8081
    """Port to bind the MCP server to.

    Env Var: `SUPERNOTE_MCP_PORT`
    """

    _base_url: str | None = field(default=None, metadata={"name": "base_url"})
    """Base URL for the main server (port 8080).
    Used for generating links and for the MCP Authorization Server issuer.
    """

    _mcp_base_url: str | None = field(default=None, metadata={"name": "mcp_base_url"})
    """Base URL for the MCP server (port 8081).

    Used for RFC 9728 discovery if the server is behind a proxy.
    """

    trace_log_file: str | None = None
    """Path to trace log file.

    Trace logging is disabled when unset.

    Env Var: `SUPERNOTE_TRACE_LOG_FILE`
    """

    storage_dir: str = "storage"
    """Directory for storing files and database.

    Env Var: `SUPERNOTE_STORAGE_DIR`
    """

    proxy_mode: str | None = None
    """Proxy header handling mode: None/'disabled' (ignore proxy headers), 'relaxed' (trust immediate upstream), or 'strict' (require specific trusted IPs). Defaults to None for security.

    Env Var: `SUPERNOTE_PROXY_MODE`
    """

    trusted_proxies: list[str] = field(
        default_factory=lambda: ["127.0.0.1", "::1", "172.17.0.0/16"]
    )
    """List of trusted proxy IPs/networks (used in strict mode). Supports CIDR notation.

    Env Var: `SUPERNOTE_TRUSTED_PROXIES` (comma-separated)
    """

    auth: AuthConfig = field(default_factory=AuthConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    metrics_enabled: bool = True
    """Whether to enable the Prometheus metrics endpoint and logging middleware.

    Env Var: `SUPERNOTE_METRICS_ENABLED`
    """

    metrics_path: str = "/metrics"
    """The path where Prometheus metrics are exposed.

    Env Var: `SUPERNOTE_METRICS_PATH`
    """

    @property
    def configured_base_url(self) -> str | None:
        """Get the explicitly configured base URL, or None if unset.

        Returns `None` when `SUPERNOTE_BASE_URL` (config `base_url`) is not set,
        allowing callers to fall back to the incoming request's base URL.

        Env Var: `SUPERNOTE_BASE_URL`
        """
        if self._base_url:
            return self._base_url.rstrip("/")
        return None

    @property
    def base_url(self) -> str:
        """Get the base URL for the main server.

        Falls back to the configured host/port when `SUPERNOTE_BASE_URL` is unset.

        Env Var: `SUPERNOTE_BASE_URL`
        """
        if self.configured_base_url is not None:
            return self.configured_base_url
        host = "localhost" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}"

    @property
    def mcp_base_url(self) -> str:
        """Get the base URL for the MCP server.

        Env Var: `SUPERNOTE_MCP_BASE_URL`
        """
        if self._mcp_base_url:
            return self._mcp_base_url.rstrip("/")
        host = "localhost" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.mcp_port}"

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.storage_dir}/system/supernote.db"

    @property
    def storage_root(self) -> Path:
        return Path(self.storage_dir)

    @property
    def ephemeral(self) -> bool:
        """Whether the server is running in ephemeral mode."""
        return _get_bool_env("SUPERNOTE_EPHEMERAL", False)

    @classmethod
    def load(
        cls, config_dir: str | Path | None = None, config_file: str | Path | None = None
    ) -> "ServerConfig":
        """Load configuration from directory. READ-ONLY."""
        if config_file is not None:
            config_file = Path(config_file)
        else:
            if config_dir is None:
                config_dir = os.getenv("SUPERNOTE_CONFIG_DIR", "config")
                logger.info(f"Using SUPERNOTE_CONFIG_DIR: {config_dir}")
            config_dir_path = Path(config_dir)
            config_file = config_dir_path / "config.yaml"
            logger.info(f"Using config file: {config_file}")

        config = cls()
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config_data = yaml.safe_load(f) or {}
                config = cls.from_dict(_migrate_legacy_ai_config(config_data))
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")

        # 4. JWT Secret priority: Env > Config > Random(in-memory only)
        env_secret = os.getenv("SUPERNOTE_JWT_SECRET")
        if env_secret:
            logger.info("Using SUPERNOTE_JWT_SECRET")
            config.auth.secret_key = env_secret

        if not config.auth.secret_key:
            logger.warning(
                "No JWT secret key configured. Using a temporary in-memory key."
            )
            config.auth.secret_key = secrets.token_hex(32)

        # Apply other env var overrides
        if os.getenv("SUPERNOTE_HOST"):
            config.host = os.getenv("SUPERNOTE_HOST", config.host)
            logger.info(f"Using SUPERNOTE_HOST: {config.host}")

        if os.getenv("SUPERNOTE_PORT"):
            try:
                config.port = int(os.getenv("SUPERNOTE_PORT", str(config.port)))
                logger.info(f"Using SUPERNOTE_PORT: {config.port}")
            except ValueError:
                pass

        if os.getenv("SUPERNOTE_MCP_PORT"):
            try:
                config.mcp_port = int(
                    os.getenv("SUPERNOTE_MCP_PORT", str(config.mcp_port))
                )
                logger.info(f"Using SUPERNOTE_MCP_PORT: {config.mcp_port}")
            except ValueError:
                pass

        if os.getenv("SUPERNOTE_STORAGE_DIR"):
            config.storage_dir = os.getenv("SUPERNOTE_STORAGE_DIR", config.storage_dir)
            logger.info(f"Using SUPERNOTE_STORAGE_DIR: {config.storage_dir}")

        if "SUPERNOTE_TRACE_LOG_FILE" in os.environ:
            trace_log_file = os.environ["SUPERNOTE_TRACE_LOG_FILE"].strip()
            config.trace_log_file = trace_log_file or None
            logger.info(
                "Trace logging %s via SUPERNOTE_TRACE_LOG_FILE",
                "enabled" if config.trace_log_file else "disabled",
            )

        if os.getenv("SUPERNOTE_BASE_URL"):
            config._base_url = os.getenv("SUPERNOTE_BASE_URL")
            logger.info(f"Using SUPERNOTE_BASE_URL: {config._base_url}")

        if os.getenv("SUPERNOTE_MCP_BASE_URL"):
            config._mcp_base_url = os.getenv("SUPERNOTE_MCP_BASE_URL")
            logger.info(f"Using SUPERNOTE_MCP_BASE_URL: {config._mcp_base_url}")

        # Legacy support/compatibility if USER sets SUPERNOTE_AUTH_URL_BASE
        if os.getenv("SUPERNOTE_AUTH_URL_BASE"):
            if not config._base_url:
                config._base_url = os.getenv("SUPERNOTE_AUTH_URL_BASE")
                logger.info(
                    f"Using legacy SUPERNOTE_AUTH_URL_BASE as base_url: {config._base_url}"
                )

        if os.getenv("SUPERNOTE_ENABLE_REGISTRATION"):
            config.auth.enable_registration = _get_bool_env(
                "SUPERNOTE_ENABLE_REGISTRATION", config.auth.enable_registration
            )
            logger.info(f"Registration Enabled: {config.auth.enable_registration}")

        if os.getenv("SUPERNOTE_ENABLE_REMOTE_PASSWORD_RESET"):
            config.auth.enable_remote_password_reset = _get_bool_env(
                "SUPERNOTE_ENABLE_REMOTE_PASSWORD_RESET",
                config.auth.enable_remote_password_reset,
            )
            logger.info(
                f"Remote Password Reset Enabled: {config.auth.enable_remote_password_reset}"
            )

        if os.getenv("SUPERNOTE_ALLOW_UNAUTHENTICATED_BINDS"):
            config.auth.allow_unauthenticated_binds = _get_bool_env(
                "SUPERNOTE_ALLOW_UNAUTHENTICATED_BINDS",
                config.auth.allow_unauthenticated_binds,
            )
            logger.info(
                "Unauthenticated Device Binds Enabled: %s",
                config.auth.allow_unauthenticated_binds,
            )

        if os.getenv("SUPERNOTE_PROXY_MODE"):
            config.proxy_mode = os.getenv("SUPERNOTE_PROXY_MODE")
            logger.info(f"Using SUPERNOTE_PROXY_MODE: {config.proxy_mode}")

        if os.getenv("SUPERNOTE_TRUSTED_PROXIES"):
            val = os.getenv("SUPERNOTE_TRUSTED_PROXIES", "")
            config.trusted_proxies = [p.strip() for p in val.split(",") if p.strip()]
            logger.info(f"Using SUPERNOTE_TRUSTED_PROXIES: {config.trusted_proxies}")

        ai_api_key = os.getenv("SUPERNOTE_AI_API_KEY")
        if ai_api_key:
            config.ai.api_key = ai_api_key
            logger.info(f"Using AI API key: xxx...{config.ai.api_key[-3:]}")
        elif not config.ai.api_key and (
            gateway_api_key := os.getenv("AI_GATEWAY_API_KEY")
        ):
            config.ai.api_key = gateway_api_key
            logger.info("Using AI_GATEWAY_API_KEY")

        if os.getenv("SUPERNOTE_GEMINI_API_KEY"):
            logger.warning(
                "SUPERNOTE_GEMINI_API_KEY is a Google Gemini credential and cannot "
                "authenticate with Vercel AI Gateway; configure SUPERNOTE_AI_API_KEY "
                "or AI_GATEWAY_API_KEY instead"
            )

        if ai_provider := os.getenv("SUPERNOTE_AI_PROVIDER") or os.getenv(
            "SUPERNOTE_GEMINI_PROVIDER"
        ):
            config.ai.provider = ai_provider
            logger.info(f"Using AI provider: {config.ai.provider}")

        if ai_ocr_model := os.getenv("SUPERNOTE_AI_OCR_MODEL") or os.getenv(
            "SUPERNOTE_GEMINI_OCR_MODEL"
        ):
            config.ai.ocr_model = ai_ocr_model
            logger.info(f"Using AI OCR model: {config.ai.ocr_model}")

        if ai_embedding_model := os.getenv("SUPERNOTE_AI_EMBEDDING_MODEL") or os.getenv(
            "SUPERNOTE_GEMINI_EMBEDDING_MODEL"
        ):
            config.ai.embedding_model = ai_embedding_model
            logger.info(f"Using AI embedding model: {config.ai.embedding_model}")

        ai_max_concurrency = os.getenv("SUPERNOTE_AI_MAX_CONCURRENCY") or os.getenv(
            "SUPERNOTE_GEMINI_MAX_CONCURRENCY"
        )
        if ai_max_concurrency:
            try:
                config.ai.max_concurrency = int(ai_max_concurrency)
                logger.info(f"Using AI max concurrency: {config.ai.max_concurrency}")
            except ValueError:
                pass

        ai_flex_env = (
            "SUPERNOTE_AI_FLEX"
            if "SUPERNOTE_AI_FLEX" in os.environ
            else "SUPERNOTE_GEMINI_FLEX"
        )
        if ai_flex_env in os.environ:
            config.ai.flex = _get_bool_env(ai_flex_env, config.ai.flex)
            logger.info(f"Using AI Flex inference: {config.ai.flex}")

        ai_zdr_env = (
            "SUPERNOTE_AI_ZDR"
            if "SUPERNOTE_AI_ZDR" in os.environ
            else "SUPERNOTE_GEMINI_ZDR"
        )
        if ai_zdr_env in os.environ:
            config.ai.zdr = _get_bool_env(ai_zdr_env, config.ai.zdr)
            logger.info(f"Using AI zero data retention: {config.ai.zdr}")

        if ai_prompts_dir := os.getenv("SUPERNOTE_AI_PROMPTS_DIR") or os.getenv(
            "SUPERNOTE_PROMPTS_DIR"
        ):
            config.ai.prompts_dir = ai_prompts_dir
            logger.info(f"Using AI prompts directory: {config.ai.prompts_dir}")

        if os.getenv("SUPERNOTE_METRICS_ENABLED"):
            config.metrics_enabled = _get_bool_env(
                "SUPERNOTE_METRICS_ENABLED", config.metrics_enabled
            )
            logger.info(f"Metrics Enabled: {config.metrics_enabled}")

        if metrics_path := os.getenv("SUPERNOTE_METRICS_PATH"):
            config.metrics_path = metrics_path
            logger.info(f"Using SUPERNOTE_METRICS_PATH: {config.metrics_path}")

        if not config_file.exists():
            logger.info(f"Saving config to {config_file}")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(cast(str, config.to_yaml()))

        return config

    class Config(BaseConfig):
        omit_none = True
        code_generation_options = [TO_DICT_ADD_OMIT_NONE_FLAG]  # type: ignore[list-item]


_LEGACY_AI_CONFIG_KEYS = {
    "gemini_provider": "provider",
    "gemini_ocr_model": "ocr_model",
    "gemini_embedding_model": "embedding_model",
    "gemini_max_concurrency": "max_concurrency",
    "gemini_flex": "flex",
    "gemini_zdr": "zdr",
    "prompts_dir": "prompts_dir",
}


def _migrate_legacy_ai_config(config_data: Any) -> dict[str, Any]:
    """Move compatible legacy Gemini settings into the unified AI block."""
    if not isinstance(config_data, dict):
        raise ValueError("Server configuration must be a mapping")
    migrated = dict(config_data)
    ai_config = dict(migrated.get("ai") or {})
    if migrated.pop("gemini_api_key", None) is not None:
        logger.warning(
            "Ignoring legacy gemini_api_key because Google Gemini credentials "
            "cannot authenticate with Vercel AI Gateway; configure ai.api_key instead"
        )
    for legacy_key, ai_key in _LEGACY_AI_CONFIG_KEYS.items():
        legacy_value = migrated.pop(legacy_key, None)
        if legacy_value is not None and ai_key not in ai_config:
            ai_config[ai_key] = legacy_value
    if ai_config:
        migrated["ai"] = ai_config
    return migrated
