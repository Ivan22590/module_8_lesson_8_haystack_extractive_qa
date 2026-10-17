"""Логика командного помощника: слушание чата, запись в Pinecone, резюме, ответы.

Модуль связывает Telegram-события (см. ``bot.py``) с Haystack-пайплайнами
(``pipelines.py``) и векторной базой Pinecone:

- каждое текстовое сообщение группы превращается в ``Document`` с метаданными
  автора и сохраняется через пайплайн индексации в namespace
  ``group_messages``;
- команды ``/start_listening`` и ``/stop_listening`` открывают и закрывают
  сессию обсуждения; в конце сессии стенограмма уходит в OpenAI и
  резюмируется, резюме сохраняется в namespace ``group_summaries``;
- при упоминании бота вопрос уходит в QA-пайплайн: поиск по истории чата +
  генерация ответа со ссылками «кто и когда сказал».

Состояние сессий хранится в памяти процесса (при рестарте бота активная
сессия теряется, но все записанные сообщения остаются в Pinecone).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from haystack import Document

from config import get_settings
from pipelines import (
    answer_question as _pipeline_answer_question,
    run_indexing,
    run_summary,
)
from prompts import SUMMARY_SYSTEM_PROMPT, build_summary_user_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# Модель данных
# ============================================================================

@dataclass
class ListenedMessage:
    """Одно сообщение участника, попавшее в сессию слушания."""

    message_id: int
    author_id: str
    author_name: str
    author_username: str | None
    timestamp: datetime
    text: str
    reply_to: int | None = None

    def transcript_line(self) -> str:
        when = self.timestamp.strftime("%d.%m.%Y %H:%M")
        text = self.text.replace("\n", " ").strip()
        return f"[{self.author_name} | {when}] {text}"


@dataclass
class ListeningSession:
    """Сессия активного слушания чата (между start и stop)."""

    session_id: str
    started_at: datetime
    started_by: str
    active: bool = True
    messages: list[ListenedMessage] = field(default_factory=list)


# chat_id → текущая сессия слушания.
_sessions: dict[int, ListeningSession] = {}


# ============================================================================
# Служебные утилиты
# ============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def current_session(chat_id: int) -> ListeningSession | None:
    """Вернуть активную сессию слушания для чата (или None)."""
    session = _sessions.get(chat_id)
    return session if session and session.active else None


def is_listening(chat_id: int) -> bool:
    """Идёт ли сейчас активное слушание чата."""
    return current_session(chat_id) is not None


# ============================================================================
# Управление слушанием
# ============================================================================

def start_listening(chat_id: int, started_by: str) -> ListeningSession:
    """Начать (или перезапустить) сессию слушания чата."""
    session = ListeningSession(
        session_id=uuid.uuid4().hex,
        started_at=_utc_now(),
        started_by=started_by,
    )
    _sessions[chat_id] = session
    logger.info(
        "▶️ Слушание начато: chat=%s, session=%s, инициатор=%r",
        chat_id,
        session.session_id,
        started_by,
    )
    return session


def stop_listening(chat_id: int) -> ListeningSession | None:
    """Завершить сессию слушания и вернуть её (или None, если не слушали)."""
    session = _sessions.get(chat_id)
    if session is None:
        return None
    session.active = False
    logger.info(
        "⏹ Слушание остановлено: chat=%s, session=%s, сообщений=%d",
        chat_id,
        session.session_id,
        len(session.messages),
    )
    return session


# ============================================================================
# Запись сообщений
# ============================================================================

def _build_message_document(
    chat_id: int,
    chat_title: str,
    message: ListenedMessage,
    session_id: str | None,
) -> Document:
    """Собрать Haystack Document из сообщения с метаданными автора."""
    meta: dict[str, object] = {
        "kind": "chat_message",
        "chat_id": str(chat_id),
        "chat_title": chat_title or "",
        "author_id": str(message.author_id),
        "author_name": message.author_name,
        "author_username": message.author_username or "",
        "message_id": int(message.message_id),
        "timestamp": message.timestamp.isoformat(),
        "session_id": session_id or "",
        "saved_at": _utc_now_iso(),
    }
    if message.reply_to is not None:
        meta["reply_to_message_id"] = int(message.reply_to)
    return Document(id=uuid.uuid4().hex, content=message.text, meta=meta)


def record_message(
    chat_id: int,
    chat_title: str,
    message: ListenedMessage,
) -> bool:
    """Записать сообщение в Pinecone (и, если идёт сессия, в её буфер).

    По умолчанию сообщения пишутся всегда (``GROUP_ALWAYS_RECORD=true``),
    чтобы у помощника накапливался контекст для ответов на упоминания.
    Если ``GROUP_ALWAYS_RECORD=false`` — запись идёт только во время сессии
    (между ``/start_listening`` и ``/stop_listening``).

    Возвращает True, если сообщение было сохранено.
    """
    settings = get_settings()
    session = current_session(chat_id)
    listening = session is not None
    if not listening and not settings.group_always_record:
        return False

    session_id = session.session_id if listening else None
    document = _build_message_document(chat_id, chat_title, message, session_id)

    try:
        run_indexing([document], settings.namespace_group_messages)
    except Exception:  # noqa: BLE001 — сбой записи не должен ломать бота
        logger.exception("Не удалось записать сообщение в Pinecone (chat=%s)", chat_id)
        saved = False
    else:
        logger.info(
            "💾 Записал сообщение: chat=%s, автор=%r, session=%s, text=%r",
            chat_id,
            message.author_name,
            session_id,
            message.text[:80],
        )
        saved = True

    if listening:
        session.messages.append(message)
    return saved


# ============================================================================
# Резюме сессии
# ============================================================================

def _save_summary(
    chat_id: int,
    chat_title: str,
    session: ListeningSession,
    summary: str,
    period: str,
    participants: list[str],
) -> None:
    """Сохранить резюме обсуждения в Pinecone (namespace summaries)."""
    settings = get_settings()
    document = Document(
        id=uuid.uuid4().hex,
        content=summary,
        meta={
            "kind": "summary",
            "chat_id": str(chat_id),
            "chat_title": chat_title or "",
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "period": period,
            "participants": ", ".join(dict.fromkeys(participants)),
            "messages_count": len(session.messages),
            "timestamp": _utc_now_iso(),
            "saved_at": _utc_now_iso(),
        },
    )
    try:
        run_indexing([document], settings.namespace_group_summaries)
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось сохранить резюме в Pinecone (chat=%s)", chat_id)


@dataclass(frozen=True)
class SummaryResult:
    """Результат резюмирования сессии."""

    text: str
    messages_count: int
    participants: list[str]
    period: str


def summarize_session(chat_id: int, chat_title: str) -> SummaryResult | None:
    """Остановить слушание и подвести итог обсуждения через OpenAI.

    Возвращает ``SummaryResult`` (текст + статистика) или None, если сессия
    не была активна / не было сообщений.
    """
    session = stop_listening(chat_id)
    if session is None:
        return None
    if not session.messages:
        return SummaryResult(
            text="В этой сессии не было сообщений — резюмировать нечего.",
            messages_count=0,
            participants=[],
            period="—",
        )

    messages = sorted(session.messages, key=lambda m: m.timestamp)
    transcript = "\n".join(m.transcript_line() for m in messages)
    participants = [m.author_name for m in messages]
    period = (
        f"{messages[0].timestamp.strftime('%d.%m.%Y %H:%M')} — "
        f"{messages[-1].timestamp.strftime('%d.%m.%Y %H:%M')}"
    )
    user_prompt = build_summary_user_prompt(
        transcript,
        chat_title=chat_title or str(chat_id),
        period=period,
        participants=participants,
    )

    summary_text = run_summary(SUMMARY_SYSTEM_PROMPT, user_prompt)
    _save_summary(
        chat_id, chat_title, session, summary_text, period, participants
    )
    return SummaryResult(
        text=summary_text,
        messages_count=len(messages),
        participants=list(dict.fromkeys(participants)),
        period=period,
    )


# ============================================================================
# Ответы на вопросы команды (упоминание бота)
# ============================================================================

def _format_sources(documents: list[Document], limit: int = 5) -> str:
    """Краткий список источников ответа: автор + время."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        meta = doc.meta or {}
        author = meta.get("author_name") or meta.get("author_username") or "Участник"
        when_raw = meta.get("timestamp") or meta.get("saved_at") or ""
        when = str(when_raw)
        try:
            when = datetime.fromisoformat(when.replace("Z", "+00:00")).strftime(
                "%d.%m.%Y %H:%M"
            )
        except (ValueError, TypeError):
            pass
        key = (str(author), when)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"• {author} — {when}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def answer_question(
    chat_id: int,
    question: str,
    *,
    chat_title: str = "",
) -> str:
    """Ответить на вопрос участника по истории чата (RAG через Haystack)."""
    text, documents = _pipeline_answer_question(question, chat_id)
    sources = _format_sources(documents)
    if sources:
        text = f"{text}\n\n🔎 Опирался на сообщения:\n{sources}"
    return text


def session_stats(chat_id: int) -> str:
    """Короткая сводка по активной сессии (для команды /listening_status)."""
    session = current_session(chat_id)
    if session is None:
        return "Слушание сейчас не активно."
    started = session.started_at.astimezone().strftime("%d.%m.%Y %H:%M")
    return (
        f"🎧 Слушаю с {started} (инициатор: {session.started_by}).\n"
        f"Сообщений в текущей сессии: {len(session.messages)}."
    )
