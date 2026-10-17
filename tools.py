"""Инструменты агента (haystack.tools.Tool): поиск, URL, факты, погода, собаки, память.

Каждый инструмент — обычная функция, обёрнутая в ``Tool`` с ручной JSON-схемой
(так мы контролируем, что пользовательские данные вроде ``user_id`` НЕ попадают
в схему для LLM — они подставляются замыканием при создании инструмента).

Фабрика ``create_agent_tools(user_id, on_tool_called)`` создаёт инструменты
привязанные к конкретному чату: ``user_id`` (chat_id) замыкается в функциях,
``on_tool_called`` вызывается при каждом обращении (бот показывает статус).
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import requests
from bs4 import BeautifulSoup
from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.tools import Tool

from config import get_settings
from pinecone_manager import get_pinecone_manager
from service_messages import is_service_text

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 20
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Маркер в результате get_random_dog_image: bot.py вытаскивает из него URL
# фотографии, чтобы отправить её пользователю вместе с ответом агента.
PHOTO_URL_MARKER_PREFIX = "URL фото:"
_PHOTO_URL_RE = re.compile(r"URL фото:\s*(\S+)")

# Описания погодных кодов Open-Meteo (кратко, на русском).
_WEATHER_CODES_RU: dict[int, str] = {
    0: "ясно ☀️",
    1: "преимущественно ясно 🌤",
    2: "переменная облачность ⛅",
    3: "пасмурно ☁️",
    45: "туман 🌫",
    48: "осаждающийся туман 🌫",
    51: "лёгкая морось 🌦",
    53: "морось 🌦",
    55: "сильная морось 🌦",
    56: "лёгкая гололёдная морось 🌧❄️",
    57: "гололёдная морось 🌧❄️",
    61: "небольшой дождь 🌦",
    63: "дождь 🌧",
    65: "сильный дождь 🌧",
    66: "дождь с гололёдом 🌧❄️",
    67: "сильный дождь с гололёдом 🌧❄️",
    71: "небольшой снег 🌨",
    73: "снег ❄️",
    75: "сильный снег ❄️",
    77: "снежные зёрна 🌨",
    80: "лёгкие ливни 🌦",
    81: "ливни 🌧",
    82: "сильные ливни ⛈",
    85: "снежные ливни 🌨",
    86: "сильные снежные ливни ❄️",
    95: "гроза ⛈",
    96: "гроза с градом ⛈",
    99: "сильная гроза с градом ⛈",
}


# ============================================================================
# Списывающий/вспомогательный код
# ============================================================================

def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    impl,
    on_tool_called,
) -> Tool:
    """Собрать Tool: при вызове сначала сообщает статус, затем выполняет impl."""

    def wrapped_function(**kwargs: Any) -> str:
        if on_tool_called is not None:
            try:
                on_tool_called(name)
            except Exception:  # noqa: BLE001 — статус не должен ронять инструмент
                logger.exception("Ошибка колбэка статуса для инструмента %s", name)
        return impl(**kwargs)

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        function=wrapped_function,
    )


def _embed_query(text: str) -> list[float]:
    """Эмбеддинг запроса через haystack OpenAITextEmbedder (ProxyAPI base_url)."""
    settings = get_settings()
    embedder = OpenAITextEmbedder(
        model=settings.embedding_model,
        api_base_url=settings.openai_base_url,
    )
    return embedder.run(text)["embedding"]


def _fetch_page_text(url: str) -> str:
    """Скачать страницу и извлечь из неё основной текст."""
    response = requests.get(
        url,
        timeout=_HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# ============================================================================
# Реализации инструментов
# ============================================================================

def _search_knowledge_base(query: str, user_id: str | None) -> str:
    """Семантический поиск по базе знаний, памяти и контексту диалога."""
    settings = get_settings()
    manager = get_pinecone_manager()
    parts: list[str] = []

    try:
        docs = manager.search_by_text(
            query,
            namespace=settings.namespace_knowledge,
            k=settings.max_search_results,
        )
    except Exception as exc:  # noqa: BLE001 — инструмент не должен ронять агента
        logger.exception("Ошибка поиска по базе знаний")
        docs = []
        parts.append(f"Ошибка при поиске по базе знаний: {exc}")
    if docs:
        parts.append("Результаты поиска по базе знаний:")
        for doc in docs:
            source = doc.meta.get("source", "источник неизвестен")
            parts.append(f"[Источник: {source}]\n{doc.content}")

    if user_id:
        user_filter = {"field": "user_id", "operator": "==", "value": user_id}

        try:
            memories = manager.search_by_text(
                query,
                namespace=settings.namespace_user_memories,
                k=settings.max_search_results,
                filter=user_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка поиска по памяти пользователя")
            memories = []
            parts.append(f"Ошибка при поиске по памяти пользователя: {exc}")
        if memories:
            parts.append("Личные факты о пользователе:")
            for doc in memories:
                parts.append(f"- {doc.content}")

        try:
            dialog = manager.search_by_text(
                query,
                namespace=settings.namespace_dialog_context,
                k=settings.max_search_results,
                filter=user_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка поиска по контексту диалога")
            dialog = []
        if dialog:
            parts.append("Что вы говорили раньше по этой теме:")
            for doc in dialog:
                parts.append(f"- {doc.content}")

    if not parts:
        return "В базе знаний и памяти пользователя ничего не найдено по этому запросу."
    return "\n\n".join(parts)


def _process_url_and_save(url: str) -> str:
    """Загрузить страницу, разбить на чанки и сохранить в базу знаний."""
    settings = get_settings()
    if not url.lower().startswith(("http://", "https://")):
        return "Ошибка: некорректный URL. Ожидается ссылка, начинающаяся с http:// или https://."

    try:
        text = _fetch_page_text(url)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка загрузки страницы %s", url)
        return f"Не удалось загрузить страницу {url}: {exc}"

    if len(text) < 50:
        return f"Страница {url} загружена, но содержит слишком мало текста для анализа."

    try:
        raw = Document(content=text, meta={"source": url})
        splitter = DocumentSplitter(
            split_by="word",
            split_length=max(settings.chunk_size // 4, 50),
            split_overlap=max(settings.chunk_overlap // 8, 5),
        )
        chunks = splitter.run([raw])["documents"]

        document_embedder = OpenAIDocumentEmbedder(
            model=settings.embedding_model,
            api_base_url=settings.openai_base_url,
            meta_fields_to_embed=[],
        )
        embedded = document_embedder.run(chunks)["documents"]

        saved = get_pinecone_manager().add_documents(
            embedded, namespace=settings.namespace_knowledge
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка сохранения в Pinecone")
        return f"Страница {url} распарсена, но не сохранена в базу: {exc}"

    return f"Страница {url} успешно изучена и сохранена в базу знаний ({saved} чанков)."


def _get_random_cat_fact() -> str:
    """Случайный факт о котиках с catfact.ninja."""
    try:
        response = requests.get("https://catfact.ninja/fact", timeout=10)
        response.raise_for_status()
        fact: str = response.json().get("fact", "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка запроса к catfact.ninja")
        return f"Не удалось получить факт о котиках: {exc}"
    return f"Факт о котиках: {fact}" if fact else "Сервис с фактами о котиках не вернул данных."


def _get_random_dog_fact() -> str:
    """Случайный факт о собаках с dogapi.dog."""
    try:
        response = requests.get(
            "https://dogapi.dog/api/v2/facts?limit=1", timeout=10
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        fact = (data[0].get("attributes") or {}).get("body", "") if data else ""
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка запроса к dogapi.dog")
        return f"Не удалось получить факт о собаках: {exc}"
    return f"Факт о собаках: {fact}" if fact else "Сервис с фактами о собаках не вернул данных."


_VISION_PROMPT = (
    "На картинке — фотография собаки. 1) Определи породу (или ближайшую "
    "визуально похожую). 2) Кратко расскажи предысторию породы: как и для чего "
    "она была выведена. 3) Дай короткую характеристику: характер, "
    "особенности содержания. Отвечай на русском, 5–8 предложений."
)


def _get_random_dog_image() -> str:
    """Случайное фото собаки с dog.ceo + vision-описание породы через LLM."""
    settings = get_settings()
    try:
        response = requests.get(
            "https://dog.ceo/api/breeds/image/random", timeout=10
        )
        response.raise_for_status()
        payload = response.json()
        photo_url: str = payload.get("message", "")
        if payload.get("status") != "success" or not photo_url:
            return "Сервис с фотографиями собак не вернул изображение."
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка запроса к dog.ceo")
        return f"Не удалось получить фото собаки: {exc}"

    try:
        image_content = ImageContent.from_url(photo_url, detail="auto")
        message = ChatMessage.from_user(content_parts=[_VISION_PROMPT, image_content])
        from haystack.components.generators.chat import OpenAIChatGenerator

        generator = OpenAIChatGenerator(
            model=settings.openai_model,
            api_base_url=settings.openai_base_url,
        )
        description = generator.run([message])["replies"][0].text or ""
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка vision-анализа фото %s", photo_url)
        return (
            f"Фото собаки получено ({PHOTO_URL_MARKER_PREFIX} {photo_url}), "
            f"но описать породу не удалось: {exc}"
        )

    return (
        f"Фотография собаки: {photo_url}\n"
        f"{PHOTO_URL_MARKER_PREFIX} {photo_url}\n\n"
        f"Описание:\n{description.strip()}"
    )


def _get_weather(city: str) -> str:
    """Погода по названию города через бесплатное Open-Meteo (без ключа)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "ru", "format": "json"},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            return f"Не нашёл город «{city}». Проверьте название (например: Москва, Казань, London)."
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        place_name = place.get("name", city)
        country = place.get("country", "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка геокодирования %s", city)
        return f"Не удалось найти город «{city}»: {exc}"

    try:
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "wind_speed_10m,weather_code"
                ),
                "wind_speed_unit": "ms",
            },
            timeout=10,
        )
        forecast.raise_for_status()
        current = forecast.json().get("current") or {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка запроса погоды для %s", city)
        return f"Не удалось получить погоду для «{place_name}»: {exc}"

    if not current:
        return f"Сервис погоды не вернул данных для «{place_name}»."

    description = _WEATHER_CODES_RU.get(
        int(current.get("weather_code", -1)), "нет данных"
    )
    return (
        f"Погода в {place_name}"
        + (f", {country}" if country else "")
        + f" сейчас:\n"
        f"• температура: {current.get('temperature_2m')} °C (ощущается как "
        f"{current.get('apparent_temperature')} °C)\n"
        f"• ветер: {current.get('wind_speed_10m')} м/с\n"
        f"• влажность: {current.get('relative_humidity_2m')} %\n"
        f"• описание: {description}"
    )


def _save_user_memory(fact: str, user_id: str | None) -> str:
    """Сохранить важный личный факт о пользователе в долговременную память."""
    settings = get_settings()
    if not user_id:
        return "Ошибка: не удалось определить пользователя для сохранения памяти."

    # Служебные ответы бота не должны попадать в долговременную память.
    if is_service_text(fact):
        logger.info("Служебное сообщение не сохранено в память: %r", fact)
        return "Служебное сообщение не сохраняется в долговременную память."

    try:
        result = get_pinecone_manager().save_memory(
            fact,
            user_id=user_id,
            namespace=settings.namespace_user_memories,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка сохранения памяти пользователя")
        return f"Не удалось сохранить факт в память: {exc}"

    if result.action == "created":
        return f"Запомнил важный факт о пользователе: «{fact}»."
    if result.action == "updated":
        return (
            "Обновил существующий слот памяти — похожий факт уже был "
            f"(косинусное сходство {result.similarity:.2f}): «{fact}»."
        )
    return (
        "Похожий факт уже сохранён в памяти "
        f"(косинусное сходство {result.similarity:.2f}) — пропускаю запись."
    )


# ============================================================================
# Фабрика инструментов для агента
# ============================================================================


def create_agent_tools(user_id: str | None, on_tool_called) -> list[Tool]:
    """Создать список Haystack Tool с привязкой к чату (user_id, статусы)."""
    return [
        _tool(
            name="search_knowledge_base",
            description=(
                "Семантический поиск по базе знаний, личной памяти пользователя "
                "и прошлым диалогам с ним. Используй, когда нужно вспомнить "
                "сохранённые статьи/документы, личные факты о собеседнике (имя, "
                "город, предпочтения) или то, что обсуждали раньше."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос по смыслу (на русском или языке пользователя).",
                    }
                },
                "required": ["query"],
            },
            impl=lambda query: _search_knowledge_base(query, user_id),
            on_tool_called=on_tool_called,
        ),
        _tool(
            name="process_url_and_save",
            description=(
                "Загрузить веб-страницу по URL, разбить на фрагменты и сохранить "
                "в базу знаний. Используй, когда пользователь прислал ссылку и "
                "попросил изучить её содержимое или запомнить/сохранить её."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Полный URL страницы, начинающийся с http:// или https://.",
                    }
                },
                "required": ["url"],
            },
            impl=lambda url: _process_url_and_save(url),
            on_tool_called=on_tool_called,
        ),
        _tool(
            name="get_random_cat_fact",
            description=(
                "Получить случайный факт о котиках из открытого API catfact.ninja. "
                "Используй, когда пользователь просит рассказать что-нибудь "
                "интересное о котиках/кошках."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            impl=lambda: _get_random_cat_fact(),
            on_tool_called=on_tool_called,
        ),
        _tool(
            name="get_random_dog_fact",
            description=(
                "Получить случайный факт о собаках из открытого API dogapi.dog. "
                "Используй, когда пользователь просит рассказать что-нибудь "
                "интересное о собаках."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            impl=lambda: _get_random_dog_fact(),
            on_tool_called=on_tool_called,
        ),
        _tool(
            name="get_random_dog_image",
            description=(
                "Получить случайную фотографию собаки из открытого API dog.ceo, "
                "определить по фото породу и рассказать о ней (предыстория, "
                "характер). Результат содержит URL фото — сообщи пользователю "
                "породу и сведения, фото бот отправит сам."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            impl=lambda: _get_random_dog_image(),
            on_tool_called=on_tool_called,
        ),
        _tool(
            name="get_weather",
            description=(
                "Получить текущую погоду в конкретном городе (температура, ветер, "
                "влажность, описание). Используй, когда пользователь спрашивает о "
                "погоде в городе."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города на русском или английском (например: Москва, Санкт-Петербург, London).",
                    }
                },
                "required": ["city"],
            },
            impl=lambda city: _get_weather(city),
            on_tool_called=on_tool_called,
        ),
        _tool(
            name="save_user_memory",
            description=(
                "Сохранить важный личный факт о пользователе в долговременную "
                "память. Используй, когда в диалоге появляется важная информация "
                "о собеседнике: имя, город, работа, увлечения, предпочтения, "
                "важные даты, цели, планы и мечты. Не сохраняй служебные и "
                "статусные сообщения."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "Краткий факт о пользователе одним предложением, например: «Живёт в Казани».",
                    }
                },
                "required": ["fact"],
            },
            impl=lambda fact: _save_user_memory(fact, user_id),
            on_tool_called=on_tool_called,
        ),
    ]


# Небольшая утилита для тестового запуска инструментов из консоли.
def _selftest() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.stdout.reconfigure(encoding="utf-8")
    print(_get_random_cat_fact())
    print(_get_random_dog_fact())
    print(_get_weather("Москва"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
