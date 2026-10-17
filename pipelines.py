"""Haystack-пайплайны командного помощника.

Модуль реализует две архитектуры из туториала ``34_Extractive_QA_Pipeline.ipynb``
(индексация и QA), но на актуальных компонентах Haystack 3.x и на
OpenAI-совместимом провайдере (ProxyAPI, ``OPENAI_BASE_URL``):

1. **Индексация** (``get_indexing_pipeline``):
   ``OpenAIDocumentEmbedder → DocumentWriter → PineconeDocumentStore``
   (в туториале — ``SentenceTransformersDocumentEmbedder → DocumentWriter``).

2. **Вопрос-ответ** (``get_qa_pipeline``):
   ``OpenAITextEmbedder → PineconeEmbeddingRetriever → [TransformersExtractiveReader]
   → ChatPromptBuilder → OpenAIChatGenerator``
   (в туториале — ``TextEmbedder → Retriever → TransformersExtractiveReader``;
   мы добавляем генерацию через OpenAI, как требует задача).

3. **Резюме** (``get_summary_pipeline``):
   ``ChatPromptBuilder → OpenAIChatGenerator`` — подведение итогов обсуждения.

ExtractiveReader подключается опционально (``USE_EXTRACTIVE_READER=true`` и
пакет ``transformers-haystack``); если он выключен или не установлен, пайплайн
работает на одних OpenAI-эмбеддингах и остаётся корректным.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)

from config import get_settings
from pinecone_manager import get_pinecone_manager

logger = logging.getLogger(__name__)

# Extractive reader — опциональная зависимость (тянет transformers + torch).
try:  # pragma: no cover - зависит от окружения
    from haystack_integrations.components.readers.transformers import (
        TransformersExtractiveReader,
    )

    _READER_AVAILABLE = True
except Exception:  # noqa: BLE001
    TransformersExtractiveReader = None  # type: ignore[assignment]
    _READER_AVAILABLE = False


# ============================================================================
# Шаблоны промптов (значения system_prompt/user_prompt подставляются рантаймом)
# ============================================================================

_QA_TEMPLATE = [
    ChatMessage.from_system("{{ system_prompt }}"),
    ChatMessage.from_user(
        "Вопрос команды: {{ query }}\n\n"
        "Контекст (сохранённые сообщения команды и, если есть, дословные цитаты):\n"
        "{{ context }}\n\n"
        "Ответь на вопрос по правилам, обязательно указав, КТО и КОГДА это сказал."
    ),
]

_SUMMARY_TEMPLATE = [
    ChatMessage.from_system("{{ system_prompt }}"),
    ChatMessage.from_user("{{ user_prompt }}"),
]


# ============================================================================
# Вспомогательное форматирование документов/цитат
# ============================================================================

def _humanize_ts(value: Any) -> str:
    """Привести ISO-метку времени к виду ``ДД.ММ.ГГГГ ЧЧ:ММ`` (или вернуть как есть)."""
    from datetime import datetime

    if not value:
        return "время неизвестно"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        return text


def format_document_line(doc: Document, index: int) -> str:
    """Одна строка контекста: ``[N] Автор (дата время): текст``."""
    meta = doc.meta or {}
    author = (
        meta.get("author_name")
        or meta.get("author_username")
        or "Участник"
    )
    when = _humanize_ts(meta.get("timestamp") or meta.get("saved_at"))
    content = (doc.content or "").replace("\n", " ").strip()
    return f"[{index}] {author} ({when}): {content}"


def format_extracted_answer(answer: Any) -> str:
    """Одна строка дословной цитаты из ``TransformersExtractiveReader``."""
    data = (getattr(answer, "data", None) or "").strip()
    if not data:
        return ""
    doc = getattr(answer, "document", None)
    meta = (doc.meta if doc is not None else {}) or {}
    author = meta.get("author_name") or meta.get("author_username") or "Участник"
    when = _humanize_ts(meta.get("timestamp") or meta.get("saved_at"))
    score = getattr(answer, "score", None)
    score_text = f", уверенность {score:.2f}" if isinstance(score, (int, float)) else ""
    return f"• «{data}» — {author} ({when}){score_text}"


@component
class MessageContextJoiner:
    """Собирает контекст для промпта: найденные сообщения + дословные цитаты.

    Кастомный компонент Haystack: на вход получает ``documents`` (из ретривера)
    и опционально ``answers`` (из ``TransformersExtractiveReader``), на выходе
    даёт готовую строку ``context`` для ``ChatPromptBuilder``.
    """

    def __init__(self, max_chars: int = 14000) -> None:
        self.max_chars = max_chars

    @component.output_types(context=str)
    def run(
        self,
        documents: list[Document],
        answers: list[Any] | None = None,
    ) -> dict[str, str]:
        parts: list[str] = []

        if answers:
            quotes = [format_extracted_answer(a) for a in answers]
            quotes = [q for q in quotes if q]
            if quotes:
                parts.append("Дословные цитаты (наиболее релевантные):")
                parts.extend(dict.fromkeys(quotes))

        if documents:
            parts.append("Релевантные сообщения команды:")
            parts.extend(
                format_document_line(doc, i)
                for i, doc in enumerate(documents, start=1)
            )

        if not parts:
            return {"context": "— (ничего релевантного в истории чата не найдено)"}

        context = "\n".join(parts)
        if len(context) > self.max_chars:
            context = context[: self.max_chars] + "\n… (контекст обрезан)"
        return {"context": context}


# ============================================================================
# Общие компоненты OpenAI (ProxyAPI base_url)
# ============================================================================

def _openai_secret() -> Secret:
    """Ключ OpenAI-совместимого провайдера как Haystack Secret."""
    return Secret.from_token(get_settings().openai_api_key)


def _text_embedder() -> OpenAITextEmbedder:
    settings = get_settings()
    return OpenAITextEmbedder(
        model=settings.embedding_model,
        api_base_url=settings.openai_base_url,
        api_key=_openai_secret(),
    )


def _document_embedder() -> OpenAIDocumentEmbedder:
    settings = get_settings()
    return OpenAIDocumentEmbedder(
        model=settings.embedding_model,
        api_base_url=settings.openai_base_url,
        api_key=_openai_secret(),
        progress_bar=False,
    )


def _chat_generator() -> OpenAIChatGenerator:
    settings = get_settings()
    return OpenAIChatGenerator(
        model=settings.openai_model,
        api_base_url=settings.openai_base_url,
        api_key=_openai_secret(),
    )


# ============================================================================
# 1. Пайплайн индексации (запись сообщений/резюме в Pinecone)
# ============================================================================

@lru_cache(maxsize=8)
def get_indexing_pipeline(namespace: str) -> Pipeline:
    """Индексация: ``OpenAIDocumentEmbedder → DocumentWriter → PineconeDocumentStore``."""
    document_store = get_pinecone_manager().store(namespace)
    pipeline = Pipeline()
    pipeline.add_component("embedder", _document_embedder())
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
    )
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def run_indexing(documents: list[Document], namespace: str) -> int:
    """Записать документы в Pinecone через пайплайн индексации.

    Возвращает число записанных документов.
    """
    if not documents:
        return 0
    pipeline = get_indexing_pipeline(namespace)
    result = pipeline.run({"embedder": {"documents": documents}})
    written = result.get("writer", {}).get("documents_written")
    return written if isinstance(written, int) else len(documents)


# ============================================================================
# 2. Пайплайн вопрос-ответ (поиск по истории + генерация ответа)
# ============================================================================

def _extractive_reader_enabled() -> bool:
    return get_settings().use_extractive_reader and _READER_AVAILABLE


@lru_cache(maxsize=1)
def get_qa_pipeline() -> Pipeline:
    """QA-пайплайн: эмбеддер → ретривер → [extractive reader] → промпт → генератор."""
    settings = get_settings()
    document_store = get_pinecone_manager().store(settings.namespace_group_messages)

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", _text_embedder())
    pipeline.add_component(
        "retriever",
        PineconeEmbeddingRetriever(
            document_store=document_store,
            top_k=settings.group_retrieval_top_k,
        ),
    )
    pipeline.add_component("joiner", MessageContextJoiner())
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=_QA_TEMPLATE,
            required_variables=["system_prompt", "query", "context"],
        ),
    )
    pipeline.add_component("generator", _chat_generator())

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "joiner.documents")
    pipeline.connect("joiner.context", "prompt_builder.context")
    pipeline.connect("prompt_builder.prompt", "generator.messages")

    if _extractive_reader_enabled():
        reader = TransformersExtractiveReader(
            model=settings.extractive_reader_model,
            no_answer=False,
        )
        pipeline.add_component("reader", reader)
        pipeline.connect("retriever.documents", "reader.documents")
        pipeline.connect("reader.answers", "joiner.answers")
        logger.info(
            "Extractive reader подключён: модель=%r", settings.extractive_reader_model
        )
    else:
        logger.info(
            "Extractive reader выключен — работаю на OpenAI-эмбеддингах "
            "(источник: %s, пакет установлен: %s)",
            "USE_EXTRACTIVE_READER",
            _READER_AVAILABLE,
        )

    return pipeline


def answer_question(
    query: str,
    chat_id: int | str,
    *,
    top_k: int | None = None,
    system_prompt: str | None = None,
) -> tuple[str, list[Document]]:
    """Ответить на вопрос команды по истории чата.

    Аргументы:
        query: вопрос участника (уже без упоминания бота).
        chat_id: id группового чата — фильтр по сообщениям именно этого чата.
        top_k: сколько сообщений достать из Pinecone.
        system_prompt: системный промпт (по умолчанию — ``QA_SYSTEM_PROMPT``).

    Возвращает (текст ответа, найденные документы).
    """
    from prompts import QA_SYSTEM_PROMPT

    settings = get_settings()
    pipeline = get_qa_pipeline()
    filters = {"field": "chat_id", "operator": "==", "value": str(chat_id)}

    run_data: dict[str, Any] = {
        "text_embedder": {"text": query},
        "retriever": {"filters": filters, "top_k": top_k or settings.group_retrieval_top_k},
        "prompt_builder": {
            "system_prompt": system_prompt or QA_SYSTEM_PROMPT,
            "query": query,
        },
    }

    if _extractive_reader_enabled():
        run_data["reader"] = {
            "query": query,
            "top_k": settings.extractive_top_k,
        }
    else:
        run_data["joiner"] = {"answers": []}

    result = pipeline.run(run_data, include_outputs_from={"retriever"})
    replies = result.get("generator", {}).get("replies") or []
    text = (replies[0].text or "").strip() if replies else ""
    if not text:
        text = "Не удалось получить ответ от модели. Попробуйте переформулировать вопрос."

    documents: list[Document] = result.get("retriever", {}).get("documents") or []
    return text, documents


# ============================================================================
# 3. Пайплайн резюме (/stop_listening)
# ============================================================================

@lru_cache(maxsize=1)
def get_summary_pipeline() -> Pipeline:
    """Резюмирующий пайплайн: ``ChatPromptBuilder → OpenAIChatGenerator``."""
    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=_SUMMARY_TEMPLATE,
            required_variables=["system_prompt", "user_prompt"],
        ),
    )
    pipeline.add_component("generator", _chat_generator())
    pipeline.connect("prompt_builder.prompt", "generator.messages")
    return pipeline


def run_summary(system_prompt: str, user_prompt: str) -> str:
    """Сгенерировать текст резюме через OpenAI-совместимый провайдер."""
    pipeline = get_summary_pipeline()
    result = pipeline.run(
        {
            "prompt_builder": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        }
    )
    replies = result.get("generator", {}).get("replies") or []
    text = (replies[0].text or "").strip() if replies else ""
    if not text:
        raise RuntimeError("Модель не вернула текст резюме.")
    return text


def extractive_reader_status() -> dict[str, Any]:
    """Служебная информация о состоянии extractive reader (для логов/README)."""
    return {
        "requested": get_settings().use_extractive_reader,
        "installed": _READER_AVAILABLE,
        "enabled": _extractive_reader_enabled(),
    }
