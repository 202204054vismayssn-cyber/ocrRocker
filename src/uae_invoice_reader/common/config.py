"""Load shared application settings from environment variables or .env file."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the Invoice Reader API powered by PaddleOCR on GPU and Ollama."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama runner settings
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="gemma2:2b")
    ollama_timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)

    # API keys
    savvy_api_key: SecretStr | None = None
    admin_api_key: SecretStr | None = None
    enable_self_restart: bool = False
    restart_delay_seconds: float = Field(default=2.0, ge=1.0, le=10.0)

    # Logging
    log_level: str = "INFO"

    # Concurrency control: allows 4-5 simultaneous extractions on GPU
    max_concurrent_extractions: int = Field(default=4, ge=1, le=20)

    # Document safety limits
    max_upload_mb: int = 15
    max_pdf_pages: int = 20
    max_image_pixels: int = 40_000_000

    # Local PaddleOCR settings (GPU accelerated by default)
    paddle_device: str = "gpu"
    paddle_lang: str = "en"
    paddle_model_cache_dir: Path = Path(".model_cache/paddle")
    paddle_enable_mkldnn: bool = False

    @field_validator("log_level")
    @classmethod
    def require_supported_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return normalized

    @property
    def model_name(self) -> str:
        return self.ollama_model

    # Document token limit
    max_input_tokens: int = Field(default=8192, ge=1000, le=128000)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and reuse across the application."""
    return Settings()
