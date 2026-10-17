"""Единый модуль управления векторной базой Pinecone для RAG-бота (Haystack).

Содержит класс ``PineconeManager`` с чтением и записью в Pinecone через
официальную интеграцию haystack-pinecone (``PineconeDocumentStore``) и
эмбеддингами Haystack (``OpenAITextEmbedder`` / ``OpenAIDocumentEmbedder``
на OpenAI-совместимом провайдере с внешним ``OPENAI_BASE_URL``):

- запись: документы (``add_documents``), личная память с умной фильтрацией
  дубликатов (``save_memory``), тексты сообщений пользователя
  (``save_user_message``);
- чтение: по тексту (``search_by_text``, ``search_by_text_with_score``)
  и по вектору (``search_by_vector``);
- обслуживание: автосоздание индекса, ``delete_namespace``, ``delete_index``.

Перед записью в долговременную память менеджер считает косинусное сходство
между текущим сообщением и уже сохранёнными фрагментами. Если сходство
низкое — это новая информация → запоминаем (``created``). Если высокое —
дубликат или вариация → пропускаем (``skipped``) либо обновляем существующий
слот памяти (``updated``).

Внизу файла — тестовый скрипт проверки подключения и работы фильтрации:

    python pinecone_manager.py                # тест подключения + демо дедупа
    python pinecone_manager.py --delete-index # удалить тестовый индекс
    python pinecone_manager.py --reset-memory # очистить namespace памяти
    python pinecone_manager.py --reset-dialog # очистить контекст диалогов
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable, Sequence

from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone

from config import get_settings

logger = logging.getLogger(__name__)

# ============================================================================
# Глобальные переключатели косинусного сходства (настраиваются здесь)
# ============================================================================

# Граница «низкого» и «высокого» сходства:
#   similarity <  COSINE_SIMILARITY_THRESHOLD → новая информация → записываем
#   similarity >= COSINE_SIMILARITY_THRESHOLD → дубликат/вариация
COSINE_SIMILARITY_THRESHOLD = 0.75

# Что делать с найденным дубликатом: "update" — обновить существующий слот
# памяти, "skip" — пропустить запись.
DUPLICATE_ACTION = "update"

# Общий выключатель умной фильтрации дубликатов (False — всегда записывать).
DEDUPE_ENABLED = True

# Namespace для служебного теста дедупликации (не путать с данными бота).
TEST_NAMESPACE = "pinecone_manager_test"

# Сколько ждать после upsert перед запросом (serverless-индексы Pinecone
# консистентны не мгновенно).
CONSISTENCY_DELAY_SECONDS = 4


@dataclass(frozen=True)
class MemoryWriteResult:
    """Результат записи в долговременную память."""

    action: str          # "created" | "updated" | "skipped"
    similarity: float    # косинусное сходство с лучшим кандидатом
    id: str | None       # id слота памяти (None, если ничего не записано)
    text: str            # исходный текст


@dataclass(frozen=True)
class DuplicateCandidate:
    """Лучший кандидат на дубликат в векторном хранилище."""

    id: str | None
    similarity: float
    text: str | None

    @property
    def found(self) -> bool:
        return self.id is not None


class PineconeManager:
    """Менеджер чтения и записи в векторную базу Pinecone на базе Haystack.

    Ленивая инициализация: raw-клиент Pinecone (для служебных операций),
    haystack-эмбеддеры (OpenAI-совместимый провайдер через ``OPENAI_BASE_URL``)
    и ``PineconeDocumentStore`` для каждого namespace создаются при первом
    обращении; индекс создаётся автоматически при отсутствии.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        *,
        embedding_model: str | None = None,
        base_url: str | None = None,
        auto_create_index: bool = True,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.pinecone_api_key
        self.openai_api_key = settings.openai_api_key
        self.index_name = index_name or settings.pinecone_index_name
        self.embedding_model = embedding_model or settings.embedding_model
        self.base_url = base_url or settings.openai_base_url
        self.embedding_dim = settings.embedding_dim
        self.region = settings.pinecone_region
        self.cloud = settings.pinecone_cloud
        self.auto_create_index = auto_create_index

        self._client: Pinecone | None = None
        self._stores: dict[str, PineconeDocumentStore] = {}
        self._embedder: OpenAITextEmbedder | None = None
        self._document_embedder: OpenAIDocumentEmbedder | None = None

    # ------------------------------------------------------------------
    # Ленивая инициализация
    # ------------------------------------------------------------------

    @property
    def client(self) -> Pinecone:
        """Raw-клиент Pinecone (служебные операции: статистика, удаление)."""
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    @property
    def embedder(self) -> OpenAITextEmbedder:
        """Haystack-эмбеддер запросов (OpenAI-совместимый, с внешним base_url)."""
        if self._embedder is None:
            self._embedder = OpenAITextEmbedder(
                model=self.embedding_model,
                api_base_url=self.base_url,
            )
        return self._embedder

    @property
    def document_embedder(self) -> OpenAIDocumentEmbedder:
        """Haystack-эмбеддер документов (для чанков статей и записей в базу)."""
        if self._document_embedder is None:
            self._document_embedder = OpenAIDocumentEmbedder(
                model=self.embedding_model,
                api_base_url=self.base_url,
                progress_bar=False,
            )
        return self._document_embedder

    def store(self, namespace: str) -> PineconeDocumentStore:
        """PineconeDocumentStore для указанного namespace (создаётся по требованию).

        Индекс создаётся автоматически при первом обращении (serverless,
        метрика cosine, размерность = EMBEDDING_DIM).
        """
        store = self._stores.get(namespace)
        if store is None:
            if not self.auto_create_index:
                raise RuntimeError(
                    f"Автосоздание индекса отключено (namespace={namespace!r})."
                )
            store = PineconeDocumentStore(
                index=self.index_name,
                namespace=namespace,
                dimension=self.embedding_dim,
                metric="cosine",
                spec={"serverless": {"region": self.region, "cloud": self.cloud}},
                show_progress=False,
                api_key=Secret.from_token(self.api_key),
            )
            self._stores[namespace] = store
        return store

    # ------------------------------------------------------------------
    # Эмбеддинги и косинусное сходство
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Преобразовать текст в вектор (эмбеддинг запроса)."""
        return list(self.embedder.run(text)["embedding"])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Преобразовать список текстов в векторы."""
        documents = [Document(content=t) for t in texts]
        embedded = self.document_embedder.run(documents)["documents"]
        return [list(doc.embedding) for doc in embedded]

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Косинусное сходство двух векторов (от 0 до 1 для положительных)."""
        if len(a) != len(b):
            raise ValueError(f"Размерности векторов различаются: {len(a)} != {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: Sequence[Document],
        namespace: str,
        *,
        dedupe: bool = False,
    ) -> int:
        """Сохранить документы (с эмбеддингами) в Pinecone. Число записанных чанков.

        При ``dedupe=True`` каждый документ проверяется на косинусное сходство
        с уже сохранёнными (по правилам ``COSINE_SIMILARITY_THRESHOLD`` и
        ``DUPLICATE_ACTION``).
        """
        if not documents:
            return 0
        if not dedupe:
            return self.store(namespace).write_documents(
                list(documents), policy=DuplicatePolicy.OVERWRITE
            )

        saved = 0
        for document in documents:
            result = self.save_document(document, namespace=namespace)
            if result.action in {"created", "updated"}:
                saved += 1
        return saved

    def save_document(
        self,
        document: Document,
        namespace: str,
        *,
        threshold: float | None = None,
        action: str | None = None,
        dedupe_filter: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Сохранить один документ с проверкой на дубликат."""
        text = (document.content or "").strip()
        if not text:
            return MemoryWriteResult(action="skipped", similarity=0.0, id=None, text="")

        candidate = self.find_best_match(
            text, namespace=namespace, filter=dedupe_filter
        )
        return self._store(
            text=text,
            metadata=dict(document.meta or {}),
            namespace=namespace,
            candidate=candidate,
            threshold=threshold,
            action=action,
        )

    def save_memory(
        self,
        text: str,
        user_id: str,
        namespace: str | None = None,
        *,
        threshold: float | None = None,
        action: str | None = None,
    ) -> MemoryWriteResult:
        """Сохранить личный факт в долговременную память пользователя.

        Перед записью считается косинусное сходство с уже сохранёнными
        фрагментами памяти пользователя:

        - ``similarity <  порога`` → новая информация → ``action: created``;
        - ``similarity >= порога`` → дубликат → ``DUPLICATE_ACTION``
          (``action: updated`` — перезаписываем слот, ``action: skipped`` —
          пропускаем).
        """
        settings = get_settings()
        ns = namespace or settings.namespace_user_memories
        text = (text or "").strip()
        if not text:
            return MemoryWriteResult(action="skipped", similarity=0.0, id=None, text="")
        if not user_id:
            raise ValueError("user_id обязателен для сохранения личной памяти.")

        candidate = self.find_best_match(
            text,
            namespace=ns,
            filter={"field": "user_id", "operator": "==", "value": user_id},
        )
        return self._store(
            text=text,
            metadata={"user_id": user_id, "type": "memory"},
            namespace=ns,
            candidate=candidate,
            threshold=threshold,
            action=action,
        )

    def save_user_message(
        self,
        text: str,
        user_id: str,
        namespace: str | None = None,
        *,
        threshold: float | None = None,
        action: str | None = None,
    ) -> MemoryWriteResult:
        """Сохранить ТЕКСТ СООБЩЕНИЯ ПОЛЬЗОВАТЕЛЯ в контекст диалога.

        Вызывается ботом для каждого неслужебного сообщения пользователя;
        используется агентом для учёта долгосрочного контекста общения
        (поиск по смыслу с косинусным сходством). Ответы бота и служебные
        тексты НЕ сохраняются (фильтрация в bot.py / agent.py).

        Дедупликация работает так же, как для личной памяти: похожие
        сообщения не плодят векторы, а обновляют существующий слот.
        """
        settings = get_settings()
        ns = namespace or settings.namespace_dialog_context
        text = (text or "").strip()
        if not text:
            return MemoryWriteResult(action="skipped", similarity=0.0, id=None, text="")
        if not user_id:
            raise ValueError("user_id обязателен для сохранения контекста диалога.")

        candidate = self.find_best_match(
            text,
            namespace=ns,
            filter={"field": "user_id", "operator": "==", "value": user_id},
        )
        return self._store(
            text=text,
            metadata={"user_id": user_id, "type": "message"},
            namespace=ns,
            candidate=candidate,
            threshold=threshold,
            action=action,
        )

    # ------------------------------------------------------------------
    # Умная фильтрация дубликатов (косинусное сходство)
    # ------------------------------------------------------------------

    def find_best_match(
        self,
        text: str,
        namespace: str,
        *,
        filter: dict[str, Any] | None = None,
        k: int = 1,
    ) -> DuplicateCandidate:
        """Найти самый похожий сохранённый фрагмент для данного текста."""
        query_vector = self.embed_text(text)
        try:
            matches = self.store(namespace)._embedding_retrieval(
                query_embedding=query_vector,
                filters=filter,
                top_k=k,
            )
        except Exception as exc:  # noqa: BLE001 — дедуп не должен ронять запись
            logger.warning("Не удалось выполнить поиск дубликатов: %s", exc)
            return DuplicateCandidate(id=None, similarity=0.0, text=None)

        if not matches:
            return DuplicateCandidate(id=None, similarity=0.0, text=None)

        best = matches[0]
        best_id = str(best.id or "")
        best_score = float(best.score or 0.0)

        # Считаем косинусное сходство явно: вектор запроса против вектора
        # найденного фрагмента (score из Pinecone — величина той же метрики,
        # но переиспользуем её только как запасной вариант).
        if best.embedding:
            try:
                best_score = self.cosine_similarity(query_vector, list(best.embedding))
            except ValueError:
                pass

        return DuplicateCandidate(id=best_id, similarity=best_score, text=best.content)

    def _store(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
        namespace: str,
        candidate: DuplicateCandidate,
        threshold: float | None,
        action: str | None,
    ) -> MemoryWriteResult:
        """Решить судьбу записи (created/updated/skipped) и выполнить её."""
        effective_threshold = (
            COSINE_SIMILARITY_THRESHOLD if threshold is None else threshold
        )
        is_duplicate = (
            DEDUPE_ENABLED
            and candidate.found
            and candidate.similarity >= effective_threshold
        )

        if not is_duplicate:
            document = Document(
                id=uuid.uuid4().hex,
                content=text,
                embedding=self.embed_text(text),
                meta={**metadata, "saved_at": _utc_now_iso()},
            )
            self.store(namespace).write_documents(
                [document], policy=DuplicatePolicy.OVERWRITE
            )
            logger.info(
                "action: created | similarity=%.4f (порог=%.2f) | id=%s | text=%r",
                candidate.similarity,
                effective_threshold,
                document.id,
                text,
            )
            return MemoryWriteResult(
                action="created",
                similarity=candidate.similarity,
                id=document.id,
                text=text,
            )

        effective_action = (action or DUPLICATE_ACTION).lower()
        if effective_action not in {"update", "skip"}:
            logger.warning(
                "Неизвестный DUPLICATE_ACTION=%r — использую 'skip'.",
                effective_action,
            )
            effective_action = "skip"

        if effective_action == "skip":
            logger.info(
                "action: skipped | similarity=%.4f (порог=%.2f) | id=%s | text=%r",
                candidate.similarity,
                effective_threshold,
                candidate.id,
                text,
            )
            return MemoryWriteResult(
                action="skipped",
                similarity=candidate.similarity,
                id=candidate.id,
                text=text,
            )

        document = Document(
            id=candidate.id or uuid.uuid4().hex,
            content=text,
            embedding=self.embed_text(text),
            meta={**metadata, "saved_at": _utc_now_iso()},
        )
        self.store(namespace).write_documents(
            [document], policy=DuplicatePolicy.OVERWRITE
        )
        logger.info(
            "action: updated | similarity=%.4f (порог=%.2f) | id=%s | text=%r",
            candidate.similarity,
            effective_threshold,
            document.id,
            text,
        )
        return MemoryWriteResult(
            action="updated",
            similarity=candidate.similarity,
            id=document.id,
            text=text,
        )

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    def search_by_text(
        self,
        query: str,
        namespace: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Чтение по тексту: семантический поиск по заданному namespace."""
        return self.store(namespace)._embedding_retrieval(
            query_embedding=self.embed_text(query),
            filters=filter,
            top_k=k,
        )

    def search_by_text_with_score(
        self,
        query: str,
        namespace: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Чтение по тексту с косинусным сходством каждого результата."""
        docs = self.search_by_text(query, namespace=namespace, k=k, filter=filter)
        return [(doc, float(doc.score or 0.0)) for doc in docs]

    def search_by_vector(
        self,
        vector: Sequence[float],
        namespace: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Чтение по вектору: поиск от уже готового эмбеддинга."""
        return self.store(namespace)._embedding_retrieval(
            query_embedding=list(vector),
            filters=filter,
            top_k=k,
        )

    def query_vector(
        self,
        vector: Sequence[float],
        namespace: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> Any:
        """Чтение по вектору «сырым» запросом к Pinecone id/score/metadata."""
        return self.client.Index(self.index_name).query(
            vector=list(vector),
            top_k=k,
            namespace=namespace,
            filter=filter,
            include_values=True,
            include_metadata=True,
        )

    def similarity_search(
        self,
        query: str,
        namespace: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Совместимое имя для чтения по тексту."""
        return self.search_by_text(query, namespace=namespace, k=k, filter=filter)

    # ------------------------------------------------------------------
    # Обслуживание
    # ------------------------------------------------------------------

    def stats(self) -> Any:
        """Статистика индекса (подключение + размерность/namespace'ы)."""
        answer = self.client.Index(self.index_name).describe_index_stats()
        # Прогрев haystack-стора (создаст индекс, если его ещё нет).
        _ = self.store(get_settings().namespace_knowledge)
        return answer

    def delete_namespace(self, namespace: str) -> None:
        """Удалить все записи в указанном namespace индекса (без ошибок, если пуст)."""
        self.store(namespace).delete_all_documents()
        logger.info("Namespace %r очищен (или уже был пуст).", namespace)

    def delete_index(self, name: str | None = None) -> bool:
        """Удалить индекс Pinecone целиком (например, тестовый)."""
        target = name or self.index_name
        if not self.client.has_index(target):
            logger.warning("Индекс %r не найден — удалять нечего.", target)
            return False
        logger.info("Удаляю индекс %r...", target)
        self.client.delete_index(target)
        if target == self.index_name:
            self._client = None
            self._stores.clear()
        return True


def _utc_now_iso() -> str:
    """Текущее время в ISO-формате (UTC)."""
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def get_pinecone_manager() -> PineconeManager:
    """Создать (и закэшировать) единый экземпляр менеджера Pinecone."""
    return PineconeManager()


# ============================================================================
# Совместимые функции-обёртки (прежний API модуля)
# ============================================================================

def get_pinecone_index() -> Any:
    """Подключиться к индексу Pinecone (при необходимости — создать)."""
    settings = get_settings()
    return get_pinecone_manager().store(settings.namespace_knowledge)


def upsert_documents(
    documents: Sequence[Document],
    namespace: str,
) -> int:
    """Сохранить документы в Pinecone. Вернуть количество записанных чанков."""
    return get_pinecone_manager().add_documents(documents, namespace)


def similarity_search(
    query: str,
    namespace: str,
    k: int = 4,
    filter: dict[str, Any] | None = None,
) -> list[Document]:
    """Семантический поиск по заданному namespace (опционально — с фильтром)."""
    return get_pinecone_manager().search_by_text(
        query, namespace=namespace, k=k, filter=filter
    )


# ============================================================================
# Тестовый скрипт: подключение, чтение/запись, фильтрация дубликатов
# ============================================================================

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _run_connection_test(manager: PineconeManager) -> None:
    """Проверить подключение: статистика индекса + эмбеддинг слова."""
    stats = manager.stats()
    dimension = stats.get("dimension")
    vectors = stats.get("total_vector_count")
    logger.info(
        "✅ Подключение успешно: индекс=%r, dimension=%s, векторов=%s",
        manager.index_name,
        dimension,
        vectors,
    )
    probe = manager.embed_text("тест подключения")
    logger.info("✅ Эмбеддинг работает: модель=%r, dim=%d", manager.embedding_model, len(probe))


def _run_read_write_test(manager: PineconeManager) -> None:
    """Запись/чтение в тестовом namespace: по тексту и по вектору."""
    namespace = TEST_NAMESPACE
    from dataclasses import replace

    document = Document(
        content="Pinecone — векторная база данных для поиска по смыслу.",
        meta={"source": "self-test", "type": "knowledge"},
    )
    document = replace(document, embedding=manager.embed_text(document.content or ""))
    saved = manager.add_documents([document], namespace=namespace)
    logger.info("Запись документа: %d чанков в namespace=%r", saved, namespace)
    time.sleep(CONSISTENCY_DELAY_SECONDS)

    by_text = manager.search_by_text("векторная база", namespace=namespace, k=2)
    logger.info("Чтение по тексту: найдено %d", len(by_text))
    for doc in by_text:
        logger.info("  - %s", (doc.content or "")[:80])

    vector = manager.embed_text("векторный поиск по смыслу")
    by_vector = manager.search_by_vector(vector, namespace=namespace, k=2)
    logger.info("Чтение по вектору: найдено %d", len(by_vector))

    raw = manager.query_vector(vector, namespace=namespace, k=2)
    matches = getattr(raw, "matches", None) or []
    logger.info("Сырой query_vector: %d совпадений", len(matches))
    for match in matches:
        logger.info("  - id=%s score=%.4f", match.id, match.score)

    manager.delete_namespace(namespace)
    logger.info("Тестовые записи в %r удалены", namespace)


def _run_dedupe_demo(manager: PineconeManager) -> None:
    """Демо умной фильтрации: две схожие по смыслу фразы."""
    namespace = TEST_NAMESPACE
    user_id = "self-test-user"
    phrases = ("Хочу на Марс", "Я полечу на Марс")

    logger.info(
        "Демо дедупа: порог=%.2f, действие=%r",
        COSINE_SIMILARITY_THRESHOLD,
        DUPLICATE_ACTION,
    )
    manager.save_memory(phrases[0], user_id=user_id, namespace=namespace)
    time.sleep(CONSISTENCY_DELAY_SECONDS)
    result = manager.save_memory(phrases[1], user_id=user_id, namespace=namespace)
    logger.info(
        "Итог по второй фразе: action=%s, similarity=%.4f",
        result.action,
        result.similarity,
    )

    manager.delete_namespace(namespace)
    logger.info("Тестовые воспоминания в %r удалены", namespace)


def main(argv: Iterable[str] | None = None) -> int:
    """Тестовый скрипт: подключение, чтение/запись, дедуп, служебные операции."""
    parser = argparse.ArgumentParser(description="Тестовый скрипт PineconeManager")
    parser.add_argument(
        "--delete-index",
        action="store_true",
        help="Удалить индекс Pinecone (например, тестовый) и выйти",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Очистить namespace личной памяти пользователей и выйти",
    )
    parser.add_argument(
        "--reset-dialog",
        action="store_true",
        help="Очистить namespace контекста диалогов и выйти",
    )
    parser.add_argument(
        "--skip-dedupe",
        action="store_true",
        help="Не запускать демо фильтрации дубликатов",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging()

    manager = PineconeManager()
    try:
        if args.delete_index:
            removed = manager.delete_index()
            logger.info("Результат удаления индекса: %s", "удалён" if removed else "не найден")
            return 0
        if args.reset_memory:
            settings = get_settings()
            manager.delete_namespace(settings.namespace_user_memories)
            return 0
        if args.reset_dialog:
            settings = get_settings()
            manager.delete_namespace(settings.namespace_dialog_context)
            return 0

        _run_connection_test(manager)
        _run_read_write_test(manager)
        if not args.skip_dedupe:
            _run_dedupe_demo(manager)
        logger.info("🎉 Все тесты менеджера прошли успешно.")
        return 0
    except Exception as exc:  # noqa: BLE001 — тестовый скрипт сообщает и падает
        logger.exception("❌ Ошибка тестового скрипта PineconeManager: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
