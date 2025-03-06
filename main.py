import os
import asyncio
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram import BaseMiddleware

from handlers import router
from model import StyleTransferModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


# Define a proper middleware class to inject the style model
class StyleModelMiddleware(BaseMiddleware):
    def __init__(self, style_model):
        self.style_model = style_model
        super().__init__()

    async def __call__(self, handler, event, data):
        # Add style_model to the data
        data["style_model"] = self.style_model
        # Call the handler
        return await handler(event, data)


async def main():
    # Initialize bot and dispatcher
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())

    # Initialize the style transfer model
    style_model = StyleTransferModel()

    # Create and register the middleware properly
    middleware = StyleModelMiddleware(style_model)
    router.message.middleware(middleware)
    router.callback_query.middleware(middleware)  # Also handle callbacks

    # Include the router
    dp.include_router(router)

    # Start the bot
    logger.info("Starting StyleTransfer Bot")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
