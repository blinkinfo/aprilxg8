"""Telegram bot for signal delivery and interactive commands.

Fixes applied:
- Graceful handling of telegram.error.Conflict when another bot instance
  is still polling (common during Railway redeploys). Retries with backoff
  so the new container survives instead of crashing.

UI/UX v2:
- All messages use HTML parse_mode for rich formatting
- Message formatting delegated to formatters module
- Commands return formatted responses via callbacks
"""
import asyncio
import logging
from typing import Optional, Callable, Awaitable

from telegram import Update, Bot
from telegram.error import Conflict, TimedOut, NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from .config import TelegramConfig
from . import formatters

logger = logging.getLogger(__name__)

# How many times to retry start_polling when a Conflict is detected.
_POLLING_MAX_RETRIES = 5
# Initial backoff (seconds); doubles each retry.
_POLLING_INITIAL_BACKOFF = 3


class TelegramBot:
    """Telegram bot for delivering signals and handling commands."""

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.bot: Optional[Bot] = None
        self.application: Optional[Application] = None
        self._stats_callback: Optional[Callable[[], str]] = None
        self._recent_callback: Optional[Callable[[], str]] = None
        self._status_callback: Optional[Callable[[], Awaitable[str]]] = None
        self._retrain_callback: Optional[Callable[[], Awaitable[str]]] = None

    def set_callbacks(
        self,
        stats_cb: Optional[Callable[[], str]] = None,
        recent_cb: Optional[Callable[[], str]] = None,
        status_cb: Optional[Callable[[], Awaitable[str]]] = None,
        retrain_cb: Optional[Callable[[], Awaitable[str]]] = None,
    ):
        """Set callback functions for bot commands."""
        self._stats_callback = stats_cb
        self._recent_callback = recent_cb
        self._status_callback = status_cb
        self._retrain_callback = retrain_cb

    async def initialize(self):
        """Initialize the bot and register handlers."""
        if not self.config.bot_token:
            logger.warning("No Telegram bot token configured")
            return

        self.application = (
            Application.builder()
            .token(self.config.bot_token)
            .build()
        )
        self.bot = self.application.bot

        # Register command handlers
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("stats", self._cmd_stats))
        self.application.add_handler(CommandHandler("recent", self._cmd_recent))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("retrain", self._cmd_retrain))

        await self.application.initialize()
        logger.info("Telegram bot initialized")

    async def start_polling(self):
        """Start polling for commands in the background.

        Retries with exponential backoff when a Conflict error is raised
        (another bot instance is still connected). This is expected during
        Railway redeployments where the old container takes a few seconds
        to fully terminate.
        """
        if self.application is None:
            return

        await self.application.start()

        backoff = _POLLING_INITIAL_BACKOFF
        for attempt in range(1, _POLLING_MAX_RETRIES + 1):
            try:
                await self.application.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES,
                )
                logger.info("Telegram bot polling started")
                return  # success
            except Conflict:
                if attempt < _POLLING_MAX_RETRIES:
                    logger.warning(
                        f"Telegram Conflict (another instance still polling). "
                        f"Retry {attempt}/{_POLLING_MAX_RETRIES} in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(
                        "Telegram Conflict persists after all retries. "
                        "Polling will NOT be active — signals will still be "
                        "sent via send_message(), but commands won't work."
                    )
            except (TimedOut, NetworkError) as e:
                if attempt < _POLLING_MAX_RETRIES:
                    logger.warning(
                        f"Telegram network error on polling start: {e}. "
                        f"Retry {attempt}/{_POLLING_MAX_RETRIES} in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(f"Telegram polling failed after all retries: {e}")

    async def stop(self):
        """Stop the bot."""
        if self.application is not None:
            try:
                if self.application.updater and self.application.updater.running:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("Telegram bot stopped")
            except Exception as e:
                logger.warning(f"Error during Telegram bot shutdown: {e}")

    async def send_message(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Send an HTML-formatted message to the configured chat.

        Args:
            text: Message text (HTML formatted)
            chat_id: Override chat ID (uses config default if None)

        Returns:
            True if sent successfully
        """
        target_chat = chat_id or self.config.chat_id
        if not target_chat or not self.bot:
            logger.warning("Cannot send message: no chat_id or bot not initialized")
            return False

        try:
            # Split long messages
            messages = self._split_message(text)
            for msg in messages:
                await self.bot.send_message(
                    chat_id=target_chat,
                    text=msg,
                    parse_mode="HTML",
                )
                if len(messages) > 1:
                    await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def _split_message(self, text: str) -> list[str]:
        """Split message into chunks that fit Telegram's limit."""
        max_len = self.config.max_message_length
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            # Find last newline before limit
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks

    # --- Command Handlers ---

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        msg = formatters.format_start(chat_id)
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = formatters.format_help()
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._stats_callback:
            text = self._stats_callback()
        else:
            text = "\U0001f4ca Stats not available yet."
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_recent(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._recent_callback:
            text = self._recent_callback()
        else:
            text = "\U0001f4cb No recent signals."
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._status_callback:
            text = await self._status_callback()
        else:
            text = "\u2699\ufe0f Status not available."
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_retrain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            formatters.format_retrain_started(),
            parse_mode="HTML",
        )
        if self._retrain_callback:
            text = await self._retrain_callback()
        else:
            text = "\U0001f504 Retrain not available."
        await update.message.reply_text(text, parse_mode="HTML")
