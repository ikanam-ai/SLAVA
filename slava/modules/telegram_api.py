import asyncio
import logging
from pathlib import Path

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from telegram import Update
from telegram.ext import Application, CallbackContext, CommandHandler, MessageHandler, filters

from slava.config import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME, TOKEN, model_name
from slava.modules.agent.assistant_graph import AGORAAssistant

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

from slava.modules.agent.handler import AGORAHandler, AGORAOptions


def agora_handler() -> AGORAHandler:
    options = AGORAOptions(
        model_name=model_name,
        psycopg_checkpointer=f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/",
    )
    handler = AGORAHandler(options)
    return handler


handler = agora_handler()


async def start(update: Update, context: CallbackContext):
    """Обработчик команды /start"""

    logging.info(
        f"Команда /start получена от пользователя {update.effective_user.id} в группе {update.effective_chat.id}"
    )
    await update.message.reply_text(
        """
        Добро пожаловать в демо-версию автономного агента **Agora**! 🚀

        ### 🌟 Что умеет Agora:
        1. **ОЦЕНКА С ПОМОЩЬЮ БЕНЧМАРКА** 🧪  
        - Agora может проводить **глубокую проверку возможностей LLM** на основе готовых бенчмарков.  
        - 📊 **Анализирует результаты**, выявляет сильные и слабые стороны модели.  
        - 💡 **Дает советы** по улучшению модели: от точности до обработки специфических запросов.

        2. **СОЗДАНИЕ БЕНЧМАРКОВ С НУЛЯ** ✍️  
        - Автоматически создает **кастомизированные тесты** на основе предоставленных документов. 📂  
        - Учитывает специфику задач, контексты и цели, чтобы оценка была максимально точной.

        3. **ПОИСК ИНФОРМАЦИИ В ИНТЕРНЕТЕ** 🌐  
        - Самостоятельно собирает данные и знания для формирования бенчмарков.  
        - ⚙️ Обрабатывает веб-информацию, чтобы пополнить тестовые кейсы или улучшить результаты анализа.

        ### 🛠️ Для кого подходит Agora?
        - Для разработчиков, которые хотят протестировать свои LLM.  
        - Для исследователей, которым нужен **точный бенчмарк**.  
        - Для бизнеса, который хочет понять, насколько эффективна их LLM в реальных условиях.

        Используйте Agora, чтобы вывести вашу LLM на **новый уровень!** ✨
        """,
        parse_mode="Markdown",
    )


async def handle_message(update: Update, context: CallbackContext):
    """Обработчик текстовых сообщений"""

    user_id = update.message.from_user.id
    user_input = update.message.text
    logging.info(f"Сообщение от пользователя {user_id} в группе {update.effective_chat.id}: {user_input}")

    processing_message = await update.message.reply_text("Ждите, идет обработка запроса...")

    result, response = await handler.ahandle_prompt(user_input, user_id)

    processing_message = await update.message.reply_text("Ждите, идет обработка файла...")

    await processing_message.delete()

    await update.message.reply_text(
        result,
        parse_mode="Markdown",
    )


async def handle_file(update: Update, context: CallbackContext):
    """Обработчик файлов"""
    user_id = update.message.from_user.id
    file = update.message.document

    logging.info(f"Файл от пользователя {user_id}: {file.file_name}")

    if file.mime_type != "application/pdf":
        await update.message.reply_text("Пока я умею работать только с PDF-файлами.")
        return

    file_path = Path(f"downloads/{file.file_name}")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_object = await file.get_file()
    await file_object.download_to_drive(custom_path=str(file_path))
    logging.info(f"Файл сохранен локально: {file_path}")

    processing_message = await update.message.reply_text("Ждите, идет обработка файла...")

    result, response = await handler.ahandle_prompt(str(file_path), user_id)

    logging.info(f"Ответ: {response}, {result}")

    await processing_message.delete()
    await update.message.reply_text(
        result,
        parse_mode="Markdown",
    )


async def run_bot():
    logging.info("Запуск бота...")
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    await application.initialize()
    await application.start()
    logging.info("Бот запущен и готов к получению сообщений.")
    asyncio.create_task(application.updater.start_polling())
    logging.info("Polling запущен.")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logging.info("Цикл событий уже запущен. Создание задачи для бота.")
            loop.create_task(run_bot())
            loop.run_forever()
        else:
            logging.info("Запуск нового цикла событий.")
            loop.run_until_complete(run_bot())
    except RuntimeError as e:
        if str(e) == "This event loop is already running":
            logging.info("Цикл событий уже запущен. Создание задачи для бота.")
            loop = asyncio.get_running_loop()
            loop.create_task(run_bot())
            loop.run_forever()
        else:
            raise
