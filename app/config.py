"""Application configuration using Pydantic settings."""

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Security
    api_key: str = "change-me-to-a-secure-key"

    # Text model backend URLs
    deepseek_base_url: HttpUrl = "http://127.0.0.1:8081"  # type: ignore
    qwen_base_url: HttpUrl = "http://127.0.0.1:8082"  # type: ignore

    # Vision model backend URLs
    minicpm_v_base_url: HttpUrl = "http://127.0.0.1:8083"  # type: ignore
    qwen2_5_vl_base_url: HttpUrl = "http://127.0.0.1:8084"  # type: ignore

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 9000
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
