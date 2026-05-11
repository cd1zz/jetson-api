"""Application configuration using Pydantic settings."""

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Security
    api_key: str = "change-me-to-a-secure-key"

    # Text / code model backend URL
    qwen_coder_base_url: HttpUrl = "http://127.0.0.1:8083"  # type: ignore

    # Vision / multimodal model backend URLs
    qwen3_vl_base_url: HttpUrl = "http://127.0.0.1:8084"  # type: ignore
    # UI-TARS-1.5-7B is a Qwen2.5-VL-architecture GUI agent model served via vLLM
    # in bf16 (in a Docker container) for faithful image preprocessing.
    ui_tars_base_url: HttpUrl = "http://127.0.0.1:8085"  # type: ignore
    # Gemma 4 26B-A4B MoE — multimodal (vision via mmproj-bf16), 4B active params
    gemma_4_base_url: HttpUrl = "http://127.0.0.1:8086"  # type: ignore

    # Embedding model backend URL (moved to 8087 since 8085 is UI-TARS)
    qwen3_embedding_base_url: HttpUrl = "http://127.0.0.1:8087"  # type: ignore

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
