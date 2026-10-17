"""Логика Telegram-бота (PyTelegramBotAPI).

Бот умеет работать в двух режимах:

- **Личный чат (private)** — персональный ассистент на Haystack Agent
  (поиск по базе знаний, память о пользователе, инструменты).
- **Групповой чат (group/supergroup)** — «Командный помощник»: слушает
  сообщения участников, пишет их в Pinecone через пайплайн индексации,
  по командам подводит итог обсуждения и отвечает на вопросы по истории
  чата со ссылками «кто и когда сказал».

Команды группового режима: ``/start_listening``, ``/stop_listening``,
``/listening_status``, ``/help``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import telebot
from telebot import TeleBot

from agent import AgentResult, reset_chat, run_agent
from config import get_settings
from pinecone_manager import get_pinecone_manager
from service_messages import (
    GROUP_HELP_MESSAGE,
    GROUP_WELCOME_MESSAGE,
    HELP_MESSAGE,
    SERVICE_ECHO_REPLY,
    TOOL_STATUS,
    WELCOME_MESSAGE,
    is_service_text,
)
from team_assistant import (
    ListenedMessage,
    answer_question,
    current_session,
    record_message,
    session_stats,
    start_listening,
    summarize_session,
)

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096

# Текст, когда у пользователя вообще нет ни имени, ни юзернейма.
FALLBACK_USER_NAME = "друг"

# Типы чатов Telegram, относящиеся к группам.
GROUP_CHAT_TYPES = {"group", "supergroup"}

# Типы вложений, которые фиксируем меткой (без содержимого).
_ATTACHMENT_TYPES = (
    "photo",
    "voice",
    "video",
    "document",
    "audio",
    "sticker",
    "video_note",
    "animation",
    "location",
    "contact",
)


# ============================================================================
# Служебные утилиты отправки/правки сообщений
# ============================================================================

def _edit_safely(bot: TeleBot, chat_id: int, message_id: int, text: str) -> bool:
    """Отредактировать статусное сообщение. Вернуть True при успехе."""
    try:
        bot.edit_message_text(text, chat_id, message_id)
        return True
    except Exception:  # noqa: BLE001 — например, «message is not modified»
        logger.debug("Не удалось отредактировать сообщение %d", message_id, exc_info=True)
        return False


def _send_long_message(bot: TeleBot, chat_id: int, text: str) -> None:
    """Отправить ответ, разрезав его по лимиту Telegram (4096 символов)."""
    for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        bot.send_message(chat_id, text[start : start + TELEGRAM_MESSAGE_LIMIT])


def _reply_or_send(bot: TeleBot, message: Any, text: str) -> None:
    """Ответить на сообщение или отправить его в чат целиком (с нарезкой)."""
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        try:
            bot.reply_to(message, text)
            return
        except Exception:  # noqa: BLE001
            logger.debug("reply_to не удался, отправляю в чат", exc_info=True)
    _send_long_message(bot, message.chat.id, text)


# ============================================================================
# Утилиты пользователей и упоминаний
# ============================================================================

def _display_name_from_user(user: Any) -> str:
    """Имя+фамилия → @username → id → fallback (не падаем без данных)."""
    if user is None:
        return FALLBACK_USER_NAME

    first = str(getattr(user, "first_name", "") or "").strip()
    last = str(getattr(user, "last_name", "") or "").strip()
    full_name = " ".join(part for part in (first, last) if part)
    if full_name:
        return full_name

    username = str(getattr(user, "username", "") or "").strip()
    if username:
        return f"@{username}"

    user_id = getattr(user, "id", None)
    if user_id is not None:
        return f"пользователь {user_id}"
    return FALLBACK_USER_NAME


def _user_display_name(message: Any) -> str:
    """Имя собеседника для личного режима."""
    return _display_name_from_user(getattr(message, "from_user", None))


def _is_group(message: Any) -> bool:
    """Сообщение из группового/супергруппового чата?"""
    chat = getattr(message, "chat", None)
    return getattr(chat, "type", None) in GROUP_CHAT_TYPES


def _mentioned_bot(message: Any, bot_username: str | None, bot_id: int | None) -> bool:
    """Упомянут ли бот: через @username, text_mention или reply на его сообщение."""
    text = message.text or ""
    for entity in getattr(message, "entities", None) or []:
        etype = getattr(entity, "type", None)
        if etype == "mention" and bot_username:
            fragment = text[entity.offset : entity.offset + entity.length].lower()
            if fragment == f"@{bot_username.lower()}":
                return True
        elif etype == "text_mention":
            mentioned_user = getattr(entity, "user", None)
            if mentioned_user is not None and getattr(mentioned_user, "id", None) == bot_id:
                return True

    reply = getattr(message, "reply_to_message", None)
    if reply is not None:
        reply_user = getattr(reply, "from_user", None)
        if reply_user is not None and getattr(reply_user, "id", None) == bot_id:
            return True
    return False


def _strip_mention(text: str, bot_username: str | None) -> str:
    """Убрать упоминание бота из текста вопроса."""
    cleaned = text
    if bot_username:
        cleaned = re.sub(
            r"@" + re.escape(bot_username) + r"\b", "", cleaned, flags=re.IGNORECASE
        )
    return cleaned.strip(" \t\n,:;-—!?")


def _attachment_label(message: Any) -> str | None:
    """Человекочитаемая метка вложения (без содержимого)."""
    if getattr(message, "photo", None):
        return "[вложение: фото]"
    if getattr(message, "voice", None):
        return "[вложение: голосовое сообщение]"
    if getattr(message, "audio", None):
        return "[вложение: аудио]"
    if getattr(message, "video", None):
        return "[вложение: видео]"
    if getattr(message, "video_note", None):
        return "[вложение: видеосообщение]"
    if getattr(message, "animation", None):
        return "[вложение: GIF]"
    if getattr(message, "document", None):
        return "[вложение: документ]"
    if getattr(message, "sticker", None):
        return "[вложение: стикер]"
    if getattr(message, "location", None):
        return "[вложение: геолокация]"
    if getattr(message, "contact", None):
        return "[вложение: контакт]"
    return None


def _welcome_message(name: str) -> str:
    """Собрать приветствие с обращением по имени (или без него)."""
    if not name or name == FALLBACK_USER_NAME:
        return WELCOME_MESSAGE
    return WELCOME_MESSAGE.replace("Привет!", f"Привет, {name}!", 1)


# ============================================================================
# Личный режим (private): сохранение контекста и ответ агента
# ============================================================================

def _save_user_message_to_memory(chat_id: int, user_text: str) -> None:
    """Сохранить ТЕКСТ сообщения пользователя в Pinecone (контекст диалога)."""
    try:
        result = get_pinecone_manager().save_user_message(
            user_text, user_id=str(chat_id)
        )
        logger.info(
            "Контекст диалога: action=%s, similarity=%.4f, text=%r",
            result.action,
            result.similarity,
            user_text,
        )
    except Exception:  # noqa: BLE001 — сбой памяти не должен ломать ответ
        logger.exception("Не удалось сохранить сообщение в Pinecone")


def _handle_private_text(bot: TeleBot, message: Any) -> None:
    """Обработать текстовое сообщение в личном чате (личный ассистент)."""
    chat_id = message.chat.id
    user_text = (message.text or "").strip()
    if not user_text:
        return

    if is_service_text(user_text):
        logger.info("Служебное сообщение из чата %s пропущено: %r", chat_id, user_text)
        bot.reply_to(message, SERVICE_ECHO_REPLY)
        return

    _save_user_message_to_memory(chat_id, user_text)

    user_name = _user_display_name(message)
    status_message = bot.send_message(chat_id, "🤔 Думаю…")

    def on_tool_called(tool_name: str) -> None:
        _edit_safely(
            bot,
            chat_id,
            status_message.message_id,
            TOOL_STATUS.get(tool_name, "⚙️ Работаю…"),
        )

    try:
        result: AgentResult = run_agent(
            chat_id,
            user_text,
            user_name=user_name,
            on_tool_called=on_tool_called,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка при обработке сообщения из чата %s", chat_id)
        _edit_safely(
            bot,
            chat_id,
            status_message.message_id,
            "⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.",
        )
        return

    answer = result.text
    if len(answer) > TELEGRAM_MESSAGE_LIMIT:
        _edit_safely(bot, chat_id, status_message.message_id, "✅ Готово, отправляю ответ…")
        _send_long_message(bot, chat_id, answer)
    elif not _edit_safely(bot, chat_id, status_message.message_id, answer):
        _send_long_message(bot, chat_id, answer)

    if result.photo_url:
        try:
            bot.send_photo(chat_id, result.photo_url, caption="🐶 Фото собаки")
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось отправить фото собаки")


# ============================================================================
# Групповой режим: командный помощник
# ============================================================================

def _handle_group_text(bot: TeleBot, message: Any, bot_username: str | None, bot_id: int) -> None:
    """Обработать текстовое сообщение в группе: запись + ответ на упоминание."""
    chat = message.chat
    chat_id = chat.id
    user = getattr(message, "from_user", None)
    text = (message.text or "").strip()

    # Сообщения от ботов (в т.ч. от самого бота) не записываем и не обрабатываем.
    if user is None or getattr(user, "is_bot", False):
        return
    if not text or text.startswith("/"):
        return

    name = _display_name_from_user(user)
    listened = ListenedMessage(
        message_id=message.message_id,
        author_id=str(user.id),
        author_name=name,
        author_username=getattr(user, "username", None),
        timestamp=_to_datetime(message.date),
        text=text,
        reply_to=getattr(message, "reply_to_message", None)
        and message.reply_to_message.message_id,
    )
    record_message(chat_id, getattr(chat, "title", "") or "", listened)

    if not _mentioned_bot(message, bot_username, bot_id):
        return

    question = _strip_mention(text, bot_username)
    if not question:
        _reply_or_send(
            bot,
            message,
            "Сформулируйте, пожалуйста, вопрос — я поищу ответ по истории чата.",
        )
        return

    status_message = bot.send_message(chat_id, "🧠 Ищу по истории чата…")
    try:
        answer = answer_question(chat_id, question, chat_title=getattr(chat, "title", "") or "")
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка ответа на вопрос в чате %s", chat_id)
        _edit_safely(
            bot,
            chat_id,
            status_message.message_id,
            "⚠️ Не удалось обработать вопрос. Попробуйте позже.",
        )
        return

    if len(answer) > TELEGRAM_MESSAGE_LIMIT:
        _edit_safely(bot, chat_id, status_message.message_id, "✅ Готово, отправляю ответ…")
        _send_long_message(bot, chat_id, answer)
    elif not _edit_safely(bot, chat_id, status_message.message_id, answer):
        _send_long_message(bot, chat_id, answer)


def _handle_group_attachment(bot: TeleBot, message: Any) -> None:
    """Зафиксировать вложение меткой (без содержимого), если идёт слушание."""
    chat = message.chat
    user = getattr(message, "from_user", None)
    if user is None or getattr(user, "is_bot", False):
        return
    label = _attachment_label(message)
    if not label:
        return
    listened = ListenedMessage(
        message_id=message.message_id,
        author_id=str(user.id),
        author_name=_display_name_from_user(user),
        author_username=getattr(user, "username", None),
        timestamp=_to_datetime(message.date),
        text=label,
    )
    record_message(chat.id, getattr(chat, "title", "") or "", listened)


def _handle_start_listening(bot: TeleBot, message: Any) -> None:
    """Команда /start_listening: открыть сессию слушания."""
    chat = message.chat
    user = getattr(message, "from_user", None)
    initiator = _display_name_from_user(user)
    session = start_listening(chat.id, initiator)
    _reply_or_send(
        bot,
        message,
        "🎧 Слушаю чат и записываю сообщения…\n"
        f"Сессия: {session.session_id[:8]}\n"
        "Отправьте /stop_listening, чтобы остановить запись и получить резюме.",
    )


def _handle_stop_listening(bot: TeleBot, message: Any) -> None:
    """Команда /stop_listening: закрыть сессию и подвести итог."""
    chat = message.chat
    if current_session(chat.id) is None:
        _reply_or_send(
            bot,
            message,
            "Слушание сейчас не активно. Запустите запись командой /start_listening.",
        )
        return

    status_message = bot.send_message(chat.id, "⏹ Останавливаю запись и подвожу итог…")
    try:
        result = summarize_session(chat.id, getattr(chat, "title", "") or "")
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка резюмирования чата %s", chat.id)
        _edit_safely(
            bot,
            chat.id,
            status_message.message_id,
            "⚠️ Не удалось подвести итог. Попробуйте позже.",
        )
        return

    if result is None:
        _edit_safely(
            bot,
            chat.id,
            status_message.message_id,
            "Слушание сейчас не активно.",
        )
        return

    header = (
        f"🧠 Итог обсуждения ({result.period})\n"
        f"Сообщений: {result.messages_count} • "
        f"Участников: {len(result.participants)}\n\n"
    )
    answer = header + result.text
    if len(answer) > TELEGRAM_MESSAGE_LIMIT:
        _edit_safely(bot, chat.id, status_message.message_id, "✅ Готово, отправляю итог…")
        _send_long_message(bot, chat.id, answer)
    elif not _edit_safely(bot, chat.id, status_message.message_id, answer):
        _send_long_message(bot, chat.id, answer)


def _handle_listening_status(bot: TeleBot, message: Any) -> None:
    """Команда /listening_status: сколько сообщений записано в сессии."""
    _reply_or_send(bot, message, session_stats(message.chat.id))


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _to_datetime(unix_ts: int | None):
    from datetime import datetime, timezone

    if not unix_ts:
        return _now()
    return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)


# ============================================================================
# Сборка бота
# ============================================================================

def create_bot() -> TeleBot:
    """Создать TeleBot с настроенной обработкой сообщений."""
    token = get_settings().require_telegram_token()
    bot = telebot.TeleBot(token)

    bot_id: int | None = None
    bot_username: str | None = None
    try:
        me = bot.get_me()
        bot_id = me.id
        bot_username = me.username
        logger.info("Бот авторизован: @%s (id=%s)", bot_username, bot_id)
    except Exception:  # noqa: BLE001
        logger.warning("Не удалось получить данные бота (get_me) — упоминания по @ снижены.", exc_info=True)

    # --- Приветствие и справка (разные для private и группы) ---
    @bot.message_handler(commands=["start"])
    def handle_start(message: Any) -> None:
        if _is_group(message):
            _reply_or_send(bot, message, GROUP_WELCOME_MESSAGE)
            return
        name = _user_display_name(message)
        bot.reply_to(message, _welcome_message(name))

    @bot.message_handler(commands=["help"])
    def handle_help(message: Any) -> None:
        if _is_group(message):
            _reply_or_send(bot, message, GROUP_HELP_MESSAGE)
            return
        bot.reply_to(message, HELP_MESSAGE)

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: Any) -> None:
        if _is_group(message):
            _reply_or_send(
                bot, message, "В групповом режиме история хранится в Pinecone и не сбрасывается."
            )
            return
        chat_id = message.chat.id
        reset_chat(chat_id)
        bot.reply_to(message, "✅ История диалога очищена.")

    # --- Команды командного помощника ---
    @bot.message_handler(commands=["start_listening"])
    def handle_start_listening(message: Any) -> None:
        if not _is_group(message):
            return
        _handle_start_listening(bot, message)

    @bot.message_handler(commands=["stop_listening"])
    def handle_stop_listening(message: Any) -> None:
        if not _is_group(message):
            return
        _handle_stop_listening(bot, message)

    @bot.message_handler(commands=["listening_status"])
    def handle_listening_status(message: Any) -> None:
        if not _is_group(message):
            return
        _handle_listening_status(bot, message)

    # --- Текстовые сообщения ---
    @bot.message_handler(func=lambda m: True, content_types=["text"])
    def handle_text(message: Any) -> None:
        if _is_group(message):
            _handle_group_text(bot, message, bot_username, bot_id or 0)
            return
        _handle_private_text(bot, message)

    # --- Вложения (фиксируем меткой) ---
    @bot.message_handler(
        func=lambda m: _is_group(m),
        content_types=list(_ATTACHMENT_TYPES),
    )
    def handle_group_attachment(message: Any) -> None:
        _handle_group_attachment(bot, message)

    return bot
