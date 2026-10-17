"""Конфигурация приложения: переменные окружения и настройки RAG-бота."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Ошибка конфигурации: отсутствует обязательная переменная окружения."""


@dataclass(frozen=True)
class Settings:
    """Неизменяемый снимок настроек приложения."""

    # LLM (OpenAI-совместимый провайдер, например ProxyAPI).
    # Внимание: ProxyAPI требует имена моделей с префиксом ("openai/gpt-4o-mini").
    openai_api_key: str
    openai_base_url: str
    openai_model: str

    # Эмбеддинги (имя тоже может быть с префиксом провайдера)
    embedding_model: str
    embedding_dim: int

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_region: str
    pinecone_cloud: str

    # Telegram (валидируется отдельно — нужен только для бота)
    telegram_token: str | None

    # RAG / история
    chunk_size: int
    chunk_overlap: int
    max_search_results: int
    max_history: int
    max_context_documents: int
    max_agent_steps: int

    # Namespaces в Pinecone
    namespace_knowledge: str
    namespace_user_memories: str
    namespace_dialog_context: str

    # Командный помощник (групповые чаты)
    namespace_group_messages: str
    namespace_group_summaries: str
    group_retrieval_top_k: int
    group_always_record: bool

    # Extractive QA (дословные цитаты; опционально)
    use_extractive_reader: bool
    extractive_reader_model: str
    extractive_top_k: int

    def require_telegram_token(self) -> str:
        """Вернуть телеграм-токен или выбросить понятную ошибку."""
        if not self.telegram_token:
            raise ConfigError(
                "Переменная окружения TELEGRAM_TOKEN не задана. "
                "Заполните её в файле .env (см. .env.example)."
            )
        return self.telegram_token


def _require(name: str) -> str:
    """Прочитать обязательную переменную окружения."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Переменная окружения {name} не задана. "
            "Скопируйте .env.example в .env и заполните значения."
        )
    return value


def _optional(name: str, default: str) -> str:
    """Прочитать необязательную переменную окружения."""
    return os.getenv(name, "").strip() or default


def _optional_bool(name: str, default: bool = False) -> bool:
    """Прочитать необязательную булеву переменную окружения."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y", "да"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Загрузить настройки (кэшируется; повторный вызов не читает .env заново)."""
    return Settings(
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_base_url=_optional("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=_optional("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=_optional("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(_optional("EMBEDDING_DIM", "1536")),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index_name=_require("PINECONE_INDEX_NAME"),
        pinecone_region=_optional("PINECONE_REGION", "us-east-1"),
        pinecone_cloud=_optional("PINECONE_CLOUD", "aws"),
        telegram_token=os.getenv("TELEGRAM_TOKEN", "").strip() or None,
        chunk_size=int(_optional("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_optional("CHUNK_OVERLAP", "200")),
        max_search_results=int(_optional("MAX_SEARCH_RESULTS", "4")),
        max_history=int(_optional("MAX_HISTORY", "20")),
        max_context_documents=int(_optional("MAX_CONTEXT_DOCUMENTS", "5")),
        max_agent_steps=int(_optional("MAX_AGENT_STEPS", "8")),
        namespace_knowledge=_optional("NAMESPACE_KNOWLEDGE", "knowledge"),
        namespace_user_memories=_optional("NAMESPACE_USER_MEMORIES", "user_memories"),
        namespace_dialog_context=_optional(
            "NAMESPACE_DIALOG_CONTEXT", "dialog_context"
        ),
        namespace_group_messages=_optional(
            "NAMESPACE_GROUP_MESSAGES", "group_messages"
        ),
        namespace_group_summaries=_optional(
            "NAMESPACE_GROUP_SUMMARIES", "group_summaries"
        ),
        group_retrieval_top_k=int(_optional("GROUP_RETRIEVAL_TOP_K", "8")),
        group_always_record=_optional_bool("GROUP_ALWAYS_RECORD", True),
        use_extractive_reader=_optional_bool("USE_EXTRACTIVE_READER", False),
        extractive_reader_model=_optional(
            "EXTRACTIVE_READER_MODEL", "deepset/roberta-base-squad2"
        ),
        extractive_top_k=int(_optional("EXTRACTIVE_TOP_K", "3")),
    )
