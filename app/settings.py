from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None

    BASE_URL: AnyUrl | None = None         # ej: https://tucasapy.com
    RESULTS_PATH: str = "/buscar"          # ruta de resultados en tu web

    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    LOG_LEVEL: str = "INFO"                # DEBUG para trazas detalladas

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
