from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI API配置
    OPENAI_API_KEY: str = ""  # 需要在.env中设置
    OPENAI_API_BASE: str = "http://103.237.29.236:10069/de_learning/v1"
    OPENAI_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    API_TIMEOUT: int = 30
    API_MAX_RETRIES: int = 3
    API_RETRY_DELAY: int = 1

    # 记忆系统配置
    MEMORY_FILE: str = "memory/chat_history.json"
    MAX_MEMORY_ITEMS: int = 100

    # 意图识别配置
    INTENT_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
