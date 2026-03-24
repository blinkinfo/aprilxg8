"""Telegram bot for signal delivery and interactive commands.

Fixes applied:
- Graceful handling of telegram.error.Conflict when another bot instance
  is still polling (common during Railway redeploys). Retries with backoff
  so the new container survives instead of crashing.

UI/UX v2:
- All messages use HTML parse_mode for rich formatting
- Message formatting delegated to formatters module
- Commands return formatted responses via callbacks
- Menu button commands registered via set_my_commands for autocomplete

Polymarket commands:
- /autotrade — Toggle auto-trading ON/OFF
- /setamount <value> — Set trade amount in USDC
- /balance — Fetch Polymarket USDC wallet balance
- /positions — Show current open positions
- /pmstatus — Full Polymarket connection status

Interactive retrain:
- /retrain shows comparison stats with inline Keep/Swap buttons
"""
import asyncio
import logging
from typing import Optional, Callable, Awaitable

from telegram import Update, Bot, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import Conflict, TimedOut, NetworkError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
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

# Menu commands shown in the Telegram command picker
_BOT_COMMANDS = [
    BotCommand("start", "Welcome message & chat ID"),
    BotCommand("help", "Show all available commands"),
    BotCommand("stats", "Lifetime signal statistics"),
    BotCommand("recent", "Last 10 signals with outcomes"),
    BotCommand("status", "Bot status & model info"),
    BotCommand("retrain", "Force model retrain now"),
    BotCommand("forcetune", "Force Optuna tuning + retrain"),
    BotCommand("autotrade", "Toggle Polymarket auto-trading"),
    BotCommand("setamount", "Set trade amount (e.g. /setamount 2.50)"),
    BotCommand("balance", "Check Polymarket USDC balance"),
    BotCommand("positions", "View open Polymarket positions"),
    BotCommand("pmstatus", "Full Polymarket connection status"),
    BotCommand("redeem", "Redeem resolved Polymarket positions"),
]


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
        self._retrain_decision_callback: Optional[Callable[[str], Awaitable[str]]] = None
        self._forcetune_callback: Optional[Callable[[], Awaitable[str]]] = None
        # Polymarket callbacks
        self._autotrade_toggle_callback: Optional[Callable[[], Awaitable[str]]] = None
        self._set_amount_callback: Optional[Callable[[float], Awaitable[str]]] = None
        self._balance_callback: Optional[Callable[[], Awaitable[str]]] = None
        self._positions_callback: Optional[Callable[[], Awaitable[str]]] = None
        self._pmstatus_callback: Optional[Callable[[], Awaitable[str]]] = None

    def set_callbacks(
        self,
        stats_cb: Optional[Callable[[], str]] = None,
        recent_cb: Optional[Callable[[], str]] = None,
        status_cb: Optional[Callable[[], Awaitable[str]]] = None,
        retrain_cb: Optional[Callable[[], Awaitable[str]]] = None,
        retrain_decision_cb: Optional[Callable[[str], Awaitable[str]]] = None,
        forcetune_cb: Optional[Callable[[], Awaitable[str]]] = None,
        autotrade_toggle_cb: Optional[Callable[[], Awaitable[str]]] = None,
        set_amount_cb: Optional[Callable[[float], Awaitable[str]]] = None,
        balance_cb: Optional[Callable[[], Awaitable[str]]] = None,
        positions_cb: Optional[Callable[[], Awaitable[str]]] = None,
        pmstatus_cb: Optional[Callable[[], Awaitable[str]]] = None,
        redeem_cb: Optional[Callable[[], Awaitable[str]]] = None,
    ):
        """Set callback functions for bot commands."""
        self._stats_callback = stats_cb
        self._recent_callback = recent_cb
        self._status_callback = status_cb
        self._retrain_callback = retrain_cb
        self._retrain_decision_callback = retrain_decision_cb
        self._forcetune_callback = forcetune_cb
        self._autotrade_toggle_callback = autotrade_toggle_cb
        self._set_amount_callback = set_amount_cb
        self._balance_callback = balance_cb
        self._positions_callback = positions_cb
        self._pmstatus_callback = pmstatus_cb
        self._redeem_cb = redeem_cb

    async def initialize(self):
        """Initialize the bot, register handlers, and set menu commands."""
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
        self.application.add_handler(CommandHandler("forcetune", self._cmd_forcetune))
        # Polymarket commands
        self.application.add_handler(CommandHandler("autotrade", self._cmd_autotrade))
        self.application.add_handler(CommandHandler("setamount", self._cmd_setamount))
        self.application.add_handler(CommandHandler("balance", self._cmd_balance))
        self.application.add_handler(CommandHandler("positions", self._cmd_positions))
        self.application.add_handler(CommandHandler("pmstatus", self._cmd_pmstatus))
        # Redemption command
        self.application.add_handler(CommandHandler("redeem", self._handle_redeem))
        # Inline button callback handler (for retrain swap/keep decisions)
        self.application.add_handler(CallbackQueryHandler(self._handle_callback_query))

        await self.application.initialize()

        # Register menu commands for Telegram's command picker (/ button)
        try:
            await self.bot.set_my_commands(_BOT_COMMANDS)
            logger.info("Telegram menu commands registered successfully")
        except Exception as e:
            logger.warning(f"Failed to set menu commands: {e}")

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
        """Stop the bot and clean up menu commands."""
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
        """Handle /retrain command — interactive retrain with comparison."""
        await update.message.reply_text(
            formatters.format_retrain_started(),
            parse_mode="HTML",
        )
        if self._retrain_callback:
            result = await self._retrain_callback()
            if isinstance(result, dict):
                # Interactive mode: got comparison data, show with inline buttons
                text = result.get("message", "Retrain complete.")
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton(
                            "\U0001f6e1\ufe0f  Keep Old Model",
                            callback_data="retrain_keep",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "\u2705  Swap to New Model",
                            callback_data="retrain_swap",
                        ),
                    ],
                ])
                await update.message.reply_text(
                    text,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            else:
                # Fallback: plain text response (error case)
                await update.message.reply_text(str(result), parse_mode="HTML")
        else:
            await update.message.reply_text(
                "\U0001f504 Retrain not available.", parse_mode="HTML"
            )

    async def _cmd_forcetune(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /forcetune command — force Optuna tuning + interactive retrain."""
        await update.message.reply_text(
            formatters.format_forcetune_started(),
            parse_mode="HTML",
        )
        if self._forcetune_callback:
            result = await self._forcetune_callback()
            if isinstance(result, dict):
                # Interactive mode: got comparison data, show with inline buttons
                text = result.get("message", "Retrain complete.")
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton(
                            "🛡️  Keep Old Model",
                            callback_data="retrain_keep",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "✅  Swap to New Model",
                            callback_data="retrain_swap",
                        ),
                    ],
                ])
                await update.message.reply_text(
                    text,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            else:
                # Fallback: plain text response (error or first model)
                await update.message.reply_text(str(result), parse_mode="HTML")
        else:
            await update.message.reply_text(
                "🔄 Force tune not available.", parse_mode="HTML"
            )

    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()  # Acknowledge the button press

        data = query.data
        if data in ("retrain_keep", "retrain_swap"):
            if self._retrain_decision_callback:
                decision = "swap" if data == "retrain_swap" else "keep"
                text = await self._retrain_decision_callback(decision)
            else:
                text = "\u26a0\ufe0f Decision handler not available."

            # Edit the original comparison message to show the decision
            # and remove the inline buttons
            try:
                await query.edit_message_text(
                    text=query.message.text_html + "\n\n" + text,
                    parse_mode="HTML",
                )
            except Exception:
                # If editing fails (e.g. message too old), send as new message
                await query.message.reply_text(text, parse_mode="HTML")
        else:
            logger.warning(f"Unknown callback query data: {data}")

    # --- Polymarket Command Handlers ---

    async def _cmd_autotrade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._autotrade_toggle_callback:
            text = await self._autotrade_toggle_callback()
        else:
            text = formatters.format_pm_not_configured()
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_setamount(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._set_amount_callback:
            await update.message.reply_text(
                formatters.format_pm_not_configured(), parse_mode="HTML"
            )
            return

        # Parse amount from command arguments
        if not context.args:
            await update.message.reply_text(
                "\u26a0\ufe0f Usage: <code>/setamount 1.50</code>\n\n"
                "Set the USDC amount per trade (0.10 - 100.00).",
                parse_mode="HTML",
            )
            return

        try:
            amount = float(context.args[0])
        except (ValueError, IndexError):
            await update.message.reply_text(
                "\u274c Invalid amount. Use a number like <code>/setamount 2.50</code>",
                parse_mode="HTML",
            )
            return

        text = await self._set_amount_callback(amount)
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._balance_callback:
            text = await self._balance_callback()
        else:
            text = formatters.format_pm_not_configured()
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._positions_callback:
            text = await self._positions_callback()
        else:
            text = formatters.format_pm_not_configured()
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_pmstatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._pmstatus_callback:
            text = await self._pmstatus_callback()
        else:
            text = formatters.format_pm_not_configured()
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_redeem(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /redeem command — trigger manual position redemption."""
        if self._redeem_cb:
            await update.message.reply_text("Scanning for redeemable positions...", parse_mode="HTML")
            text = await self._redeem_cb()
            await update.message.reply_text(text, parse_mode="HTML")
        else:
            await update.message.reply_text("Position redemption is not available.", parse_mode="HTML")