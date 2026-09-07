from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    llm_provider: Literal["groq", "openai", "anthropic", "fake"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    dataforseo_mode: Literal["mock", "live"] = "mock"
    dataforseo_login: str | None = None
    dataforseo_password: str | None = None
    dataforseo_base_url: str = "https://api.dataforseo.com"
    dataforseo_connect_timeout: float = 5.0
    dataforseo_read_timeout: float = 30.0

    mock_failure_rate: float = 0.0
    mock_always_fail_tools: str = ""

    retry_max_attempts: int = 4
    retry_base_delay: float = 0.4
    retry_max_delay: float = 8.0
    circuit_failure_threshold: int = 5
    circuit_reset_timeout: float = 30.0
    retrieval_success_threshold: float = 0.5

    max_planned_calls: int = 8
    max_seed_queries: int = 6

    score_volume_ceiling: int = 50_000
    score_weight_volume: float = 0.35
    score_weight_difficulty: float = 0.25
    score_weight_visibility_gap: float = 0.40
    score_ai_surface_bonus: float = 0.10

    database_url: str = "sqlite:///./search_intel.db"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    @field_validator("mock_always_fail_tools")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def always_fail_tools(self) -> set[str]:
        return {t.strip() for t in self.mock_always_fail_tools.split(",") if t.strip()}

    @property
    def dataforseo_credentials(self) -> tuple[str, str] | None:
        if self.dataforseo_login and self.dataforseo_password:
            return self.dataforseo_login, self.dataforseo_password
        return None

    def api_key_for_provider(self) -> str | None:
        return {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(self.llm_provider)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()

