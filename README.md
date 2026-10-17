# Командный помощник — Telegram-бот на Haystack + Pinecone

Telegram-бот для **рабочих командных чатов**. Он незаметно фиксирует обсуждения
участников в векторной базе Pinecone, по команде подводит **итог дискуссии**
(позиции сторон, мнение по спорным вопросам, принятые решения) и отвечает на
вопросы по **всей истории чата** со ссылками «кто и когда сказал».

Основа — пайплайны **Haystack 3.x** и эмбеддинги/генерация **OpenAI** через
OpenAI-совместимый провайдер (ProxyAPI, `OPENAI_BASE_URL`).

> Архитектура пайплайнов повторяет туториал `34_Extractive_QA_Pipeline.ipynb`
> (индексация `embedder → writer` и QA `embedder → retriever → reader`), но на
> актуальных компонентах Haystack 3.4, а финальная генерация выполняется через
> `OpenAIChatGenerator` (как требует задача).

## Возможности

**Групповой режим (командный помощник):**
- 🎧 `/start_listening` — начать запись обсуждения;
- ⏹ `/stop_listening` — остановить запись и получить **резюме** через OpenAI:
  суть обсуждения, участники и их позиции, мнение бота при споре, финальное
  решение и следующие шаги;
- 📊 `/listening_status` — сколько сообщений уже записано в текущей сессии;
- ❓ ответ на **упоминание** `@имя_бота …` или reply на сообщение бота: RAG-поиск
  по истории чата + ответ со ссылками «кто и когда сказал»;
- каждое текстовое сообщение пишется в Pinecone с метаданными автора
  (имя, username, id, время, id сообщения, сессия).

**Личный режим (private) — персональный ассистент (как раньше):**
- агент на Haystack Agent с инструментами: поиск по базе знаний, изучение ссылок,
  факты о котиках/собаках, фото собаки с определением породы, погода,
  долговременная память о пользователе; `/reset`, `/help`.

## Архитектура

```
Telegram (PyTelegramBotAPI, bot.py)
        │
        ├── private-чат  → Haystack Agent (agent.py) + инструменты (tools.py)
        │
        └── групповой чат → team_assistant.py
                 │                 │
                 │ запись сообщений │ вопрос / резюме
                 ▼                 ▼
        pipelines.py  (Haystack Pipelines)
         ├── indexing: OpenAIDocumentEmbedder → DocumentWriter ─┐
         ├── QA:       OpenAITextEmbedder → PineconeEmbeddingRetriever
         │             → [TransformersExtractiveReader] → ChatPromptBuilder
         │             → OpenAIChatGenerator
         └── summary:  ChatPromptBuilder → OpenAIChatGenerator
                                                               │
                                                               ▼
                              PineconeDocumentStore (haystack-pinecone)
                                 namespaces:
                                   • group_messages   — сообщения команды
                                   • group_summaries  — резюме сессий
                                   • knowledge / user_memories / dialog_context
                                     (личный режим)
```

- **Все эмбеддинги** — `OpenAITextEmbedder` / `OpenAIDocumentEmbedder` с
  `api_base_url = OPENAI_BASE_URL`.
- **Вся генерация** — `OpenAIChatGenerator` с тем же `api_base_url`.
- **Extractive reader** (`TransformersExtractiveReader`, из туториала)
  подключается опционально и добавляет к ответу дословные цитаты с точным
  местом в тексте; включается `USE_EXTRACTIVE_READER=true`.

## Структура проекта

```
.
├── pipelines.py         # Haystack-пайплайны: индексация, QA, резюме
├── team_assistant.py    # логика командного помощника (слушание, резюме, ответы)
├── prompts.py           # подробные промпты (резюме, ответы на вопросы)
├── pinecone_manager.py  # единый модуль Pinecone (haystack-pinecone) + дедуп
├── config.py            # загрузка .env и настройки (Settings)
├── service_messages.py  # служебные сообщения бота + is_service_text()
├── tools.py             # инструменты личного ассистента
├── agent.py             # Haystack Agent личного ассистента
├── bot.py               # логика Telegram-бота (private + group)
├── main.py              # точка входа (прогрев пайплайнов + запуск бота)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Настройка

### 1. Виртуальное окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` тянет `transformers-haystack` (для extractive reader — это
> крупная установка с `torch`). Он уже установлен и включён
> (`USE_EXTRACTIVE_READER=true`). При первом запуске модель
> `deepset/roberta-base-squad2` скачается в кэш HuggingFace
> (`~/.cache/huggingface`), это разовая загрузка ~500 МБ. Если дословные
> цитаты не нужны — поставьте `USE_EXTRACTIVE_READER=false`.

### 2. Переменные окружения

Скопируйте `.env.example` → `.env` и заполните значения:

| Переменная              | Обязательна | Описание                                                        |
|-------------------------|-------------|-----------------------------------------------------------------|
| `TELEGRAM_TOKEN`        | да          | токен бота от [@BotFather](https://t.me/BotFather)              |
| `PINECONE_API_KEY`      | да          | ключ Pinecone (Console → API Keys)                              |
| `PINECONE_INDEX_NAME`   | да          | имя индекса (создаётся автоматически)                           |
| `OPENAI_API_KEY`        | да          | ключ OpenAI-совместимого провайдера (ProxyAPI)                  |
| `OPENAI_BASE_URL`       | да          | базовый URL API (для ProxyAPI — `https://api.proxyapi.ru/v1`)   |
| `OPENAI_MODEL`          | нет         | модель чата (`openai/gpt-4o-mini`)                              |
| `EMBEDDING_MODEL`       | нет         | модель эмбеддингов (`openai/text-embedding-3-small`)            |
| `EMBEDDING_DIM`         | нет         | размерность эмбеддингов (`1536`)                                |
| `NAMESPACE_GROUP_MESSAGES` | нет      | namespace сообщений команды (`group_messages`)                  |
| `NAMESPACE_GROUP_SUMMARIES`| нет      | namespace резюме (`group_summaries`)                            |
| `GROUP_RETRIEVAL_TOP_K` | нет         | сколько сообщений поднимать на вопрос (`8`)                     |
| `GROUP_ALWAYS_RECORD`   | нет         | писать все сообщения (`true`) или только в сессии (`false`)     |
| `USE_EXTRACTIVE_READER` | нет         | включить дословные цитаты (`true`)                              |
| `EXTRACTIVE_READER_MODEL`| нет        | модель reader (`deepset/roberta-base-squad2`)                   |

> **ProxyAPI:** имена моделей указываются С ПРЕФИКСОМ провайдера
> (`openai/gpt-4o-mini`, `openai/text-embedding-3-small`).

### 3. Настройка бота в Telegram (важно!)

Чтобы бот **видел все сообщения группы**, а не только команды и reply:

1. В [@BotFather](https://t.me/BotFather) → `/mybots` → ваш бот → **Bot Settings**
   → **Group Privacy** → **Turn off** (Privacy Mode отключён).
   *Либо* выдайте боту права **администратора** группы.
2. Добавьте бота в группу.
3. Проверьте: `/start` в группе → должно прийти приветствие командного помощника.

### 4. Запуск

```bash
python main.py
```

В логах: подключение к Pinecone, прогрев пайплайнов, статус extractive reader,
затем `🚀 Бот запущен`.

## Использование (групповой чат)

1. `/start_listening` — бот начинает слушать и записывать сообщения.
2. Участники обсуждают рабочий вопрос.
3. `/stop_listening` — бот подводит итог: суть, позиции сторон, при споре — своё
   аргументированное мнение, при решении — что решили и что делать дальше.
   Резюме сохраняется в Pinecone.
4. В любой момент: `@имя_бота что мы решили по релизу?` — бот ищет по всей
   истории чата и отвечает, **кто и когда** это сказал.

## Примечания

- В Pinecone сохраняются **текстовые** сообщения участников группы; служебные
  реплики и сообщения ботов не пишутся. Вложения фиксируются меткой
  (`[вложение: фото]`) без содержимого.
- По умолчанию (`GROUP_ALWAYS_RECORD=true`) бот пишет сообщения всегда — чтобы
  накапливать контекст для ответов на упоминания. Команды
  `/start_listening`—`/stop_listening` при этом задают **сессию** для резюме.
- История сессий хранится в памяти процесса; все записанные сообщения и резюме
  остаются в Pinecone и переживают рестарт бота.
- Длинные ответы автоматически режутся по лимиту Telegram (4096 символов).
- Пайплайны Haystack 3.x: `Pipeline`, `OpenAIDocumentEmbedder`,
  `PineconeEmbeddingRetriever`, `ChatPromptBuilder`, `OpenAIChatGenerator`,
  `DocumentWriter`.
