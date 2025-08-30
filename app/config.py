from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Типизированные настройки приложения, читаемые из переменных окружения и .env


class Settings(BaseSettings):
    # Устройство и модель
    WHISPER_DEVICE: str = Field(default="auto", description="auto|cuda|cpu")
    WHISPER_MODEL: str = Field(default="small")
    WHISPER_COMPUTE_TYPE: str | None = Field(default=None, description="int8|float16|int8_float16|...")

    # Параметры распознавания
    WHISPER_LANGUAGE: str | None = None
    WHISPER_TASK: str = "transcribe"  # или "translate"
    WHISPER_BEAM: int = 8
    WHISPER_PATIENCE: float = 1.0
    WHISPER_LENGTH_PENALTY: float = 1.0
    WHISPER_CONDITION_PREV: bool = False
    WHISPER_VAD_MIN_SIL_MS: int = 300
    WHISPER_PROMPT: str | None = None
    WHISPER_VAD_ENABLED: bool = True
    WHISPER_LOG_PROGRESS: bool = False
    WHISPER_WORD_TIMESTAMPS: bool = False
    WHISPER_MULTILINGUAL: bool = False
    WHISPER_HOTWORDS: str | None = None
    WHISPER_MAX_NEW_TOKENS: int | None = None
    WHISPER_BATCHED: bool = False
    WHISPER_BATCH_SIZE: int = 8

    # Конфиг загрузки из .env (если присутствует) и переменных окружения
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    # Возвращает кешированные настройки
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    # Переинициализирует настройки из окружения/.env.
    global _settings
    _settings = Settings()
    return _settings
