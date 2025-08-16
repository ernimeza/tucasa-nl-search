from pydantic_settings import BaseSettings
from pydantic import AnyUrl

class Settings(BaseSettings):
    # Clave de OpenAI (la usaremos en el main después)
    OPENAI_API_KEY: str | None = None

    # Para construir la URL final (tu dominio)
    BASE_URL: AnyUrl | None = None  # Ej: https://tucasapy.com

    # Ruta base de resultados en tu web (donde tu front lee los query params)
    RESULTS_PATH: str = "/buscar"   # cámbialo si tu ruta es otra

    # CORS
    CORS_ALLOW_ORIGINS: list[str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
