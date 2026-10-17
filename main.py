"""Точка входа: инициализация конфигурации, прогрев Pinecone-индекса и запуск бота."""

from __future__ import annotations

import logging
import sys

from bot import create_bot
from config import ConfigError, get_settings
from pinecone_manager import get_pinecone_manager
from pipelines import extractive_reader_status, get_indexing_pipeline


def main() -> None:
    """Запустить Telegram-бота с RAG-агентом."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("main")

    try:
        settings = get_settings()  # валидация .env до старта
        # Прогрев: подключение (и автосоздание при необходимости) индекса,
        # инициализация haystack-сторов и эмбеддеров.
        get_pinecone_manager().stats()
        # Прогрев пайплайнов индексации сообщений и резюме.
        get_indexing_pipeline(settings.namespace_group_messages)
        get_indexing_pipeline(settings.namespace_group_summaries)
        reader = extractive_reader_status()
        logger.info(
            "Extractive reader: запрошен=%s, установлен=%s, активен=%s",
            reader["requested"],
            reader["installed"],
            reader["enabled"],
        )
    except ConfigError as exc:
        logger.error("❌ Ошибка конфигурации: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ Не удалось выполнить прогрев: %s", exc)
        sys.exit(1)

    bot = create_bot()
    logger.info("🚀 Бот запущен. Ожидаю сообщения…")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    main()
