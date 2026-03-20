"""Main bot orchestrator - ties everything together.

Preserves the 15-second pre-signal timing by design.
Integrates: retraining gate messaging, signal strength labels, Optuna status.

Fixes applied:
- Signal predicts the NEXT candle (current_slot + 5min), not the current one
- Resolution uses candle open/close from MEXC (matches Polymarket binary outcome)
- Signals store NEXT candle slot timestamp; resolution only fires once that candle closes
- Resolution waits >= 30 seconds into the candle AFTER the predicted one for data to settle
- Dedup guard prevents double-resolution within the same window
- Startup resolves any stale pending signals from prior sessions
- All timestamps in UTC
- Training now uses full paginated historical data for ALL timeframes (5m, 15m, 1h)

UI/UX v2:
- All message formatting delegated to formatters module
- HTML parse mode via TelegramBot.send_message
"""
import asyncio
import logging
import os
import signal
from datetime import datetime, timezone, timedelta

from .config import BotConfig
from .data_fetcher import MEXCFetcher
from .features import FeatureEngineer
from .model import PredictionModel
from .signal_tracker import SignalTracker
from .telegram_bot import TelegramBot
from . import formatters

logger = logging.getLogger(__name__)

# Seconds into the new candle before we attempt resolution.
# Gives MEXC time to finalize the previous candle's close price.
RESOLUTION_DELAY_SECONDS = 30
# Upper bound of the resolution window (avoid running too late).
RESOLUTION_WINDOW_END = 90

# Candle period in minutes (5-minute candles)
CANDLE_PERIOD_MINUTES = 5


def _candle_slot_open(dt: datetime, period_minutes: int = 5) -> datetime:
    """Return the open timestamp of the candle slot that `dt` falls in.

    E.g. for period_minutes=5:
        09:03:22 -> 09:00:00
        09:07:45 -> 09:05:00
        10:00:01 -> 10:00:00
    """
    total_minutes = dt.hour * 60 + dt.minute
    slot_minute = (total_minutes // period_minutes) * period_minutes
    return dt.replace(
        hour=slot_minute // 60,
        minute=slot_minute % 60,
        second=0,
        microsecond=0,
    )


class SignalBot:
    """Main signal bot orchestrator."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.fetcher = MEXCFetcher(config.mexc)
        self.model = PredictionModel(config.model)
        self.tracker = SignalTracker(config.data_dir)
        self.telegram = TelegramBot(config.telegram)
        self._running = False
        self._last_signal_candle_ts = None  # Prevent duplicate signals per candle
        self._last_resolved_candle_ts = None  # Prevent double-resolution per candle

    async def start(self):
        """Start the bot."""
        logger.info("=" * 50)
        logger.info("BTC 5m Signal Bot starting (aprilxg v2)...")
        logger.info("=" * 50)

        # Create directories
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.model_dir, exist_ok=True)

        # Initialize Telegram bot
        await self.telegram.initialize()
        self.telegram.set_callbacks(
            stats_cb=self._get_stats_text,
            recent_cb=self._get_recent_text,
            status_cb=self._get_status_text,
            retrain_cb=self._retrain_model,
        )
        await self.telegram.start_polling()

        # Try loading existing model
        loaded = self.model.load(self.config.model_dir)
        if not loaded:
            logger.info("No saved model found, training initial model...")
            await self._train_model()
        else:
            logger.info(f"Model loaded (val_acc={self.model.val_accuracy:.4f})")

        # Resolve any stale pending signals from previous session
        await self._resolve_stale_signals()

        # Send startup message
        msg = formatters.format_startup(
            model_accuracy=self.model.val_accuracy,
            confidence_min=self.config.model.confidence_min,
            train_candles=self.config.model.train_candles,
            optuna_enabled=self.config.model.enable_optuna_tuning,
            retrain_gate=self.config.model.retrain_min_improvement,
            tracked_signals=len(self.tracker.signals),
            symbol=self.config.mexc.symbol,
        )
        await self.telegram.send_message(msg)

        # Main loop
        self._running = True
        await self._main_loop()

    async def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self._running = False
        await self.telegram.send_message(formatters.format_shutdown())
        await self.telegram.stop()
        await self.fetcher.close()
        self.model.save(self.config.model_dir)
        logger.info("Bot stopped")

    async def _main_loop(self):
        """Main prediction loop.

        CRITICAL timing design:
        - Signal fires <= 15 seconds before candle close (prediction_lead_seconds).
          The signal is labeled for the NEXT candle (current_slot + 5min) because
          the model predicts the NEXT candle's direction (trained with shift(-1) labels).
        - Resolution fires 30-90 seconds into a new candle. It resolves any pending
          signals whose candle_slot_ts is strictly OLDER than the current candle
          (meaning that predicted candle has fully closed).
        - Dedup guards prevent duplicate signals AND duplicate resolutions.
        """
        logger.info("Entering main prediction loop")

        while self._running:
            try:
                now = datetime.now(timezone.utc)
                current_slot = _candle_slot_open(now)

                # Seconds elapsed within the current 5-min candle
                seconds_in_candle = (now - current_slot).total_seconds()
                seconds_until_close = 300 - seconds_in_candle

                # --- SIGNAL: fire <= prediction_lead_seconds before candle close ---
                if 0 < seconds_until_close <= self.config.prediction_lead_seconds:
                    if self._last_signal_candle_ts != current_slot:
                        await self._run_prediction_cycle(now, current_slot)
                        self._last_signal_candle_ts = current_slot

                # --- RESOLUTION: fire 30-90 seconds into the new candle ---
                if RESOLUTION_DELAY_SECONDS <= seconds_in_candle < RESOLUTION_WINDOW_END:
                    if self._last_resolved_candle_ts != current_slot:
                        await self._resolve_pending_signals(current_slot)
                        self._last_resolved_candle_ts = current_slot

                # --- RETRAIN: check if model needs retraining ---
                if self.model.needs_retrain():
                    logger.info("Model retrain interval reached")
                    await self._train_model()

                await asyncio.sleep(self.config.main_loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _run_prediction_cycle(self, now: datetime, current_slot: datetime):
        """Fetch data and make a prediction.

        The model predicts the direction of the NEXT candle (trained with
        shift(-1) labels). So the signal is labeled for the NEXT slot, not
        the current one that is about to close.

        Args:
            now: Current UTC datetime
            current_slot: The open timestamp of the current 5-min candle
        """
        try:
            # Fetch multi-timeframe data
            data = await self.fetcher.fetch_multi_timeframe(
                intervals=["5m", "15m", "1h"],
                limit=self.config.model.lookback_candles,
            )

            df_5m = data.get("5m")
            if df_5m is None or df_5m.empty:
                logger.warning("No 5m data available")
                return

            higher_tf = {k: v for k, v in data.items() if k != "5m" and not v.empty}

            # Make prediction
            prediction = self.model.predict(df_5m, higher_tf)

            if prediction["signal"] in ("UP", "DOWN"):
                # ----------------------------------------------------------------
                # KEY FIX: The model predicts the NEXT candle, not the current one.
                # current_slot is the candle about to close (e.g. 16:40).
                # The prediction is for the NEXT candle (e.g. 16:45-16:50).
                # ----------------------------------------------------------------
                next_slot = current_slot + timedelta(minutes=CANDLE_PERIOD_MINUTES)
                next_slot_iso = next_slot.isoformat()

                # We don't know the next candle's open price yet (it hasn't started).
                # It will be filled in during resolution when the candle data is available.
                candle_open_price = 0.0

                # Record signal with NEXT candle slot info
                sig = self.tracker.add_signal(
                    direction=prediction["signal"],
                    confidence=prediction["confidence"],
                    entry_price=prediction["current_price"],
                    candle_slot_ts=next_slot_iso,
                    candle_open_price=candle_open_price,
                )

                # Send formatted signal to Telegram
                msg = formatters.format_signal(sig, prediction)
                await self.telegram.send_message(msg)
                logger.info(
                    f"Signal sent: {prediction['signal']} [{prediction.get('strength', 'NORMAL')}] "
                    f"@ ${prediction['current_price']:,.2f} "
                    f"(predicting next slot={next_slot_iso})"
                )
            else:
                logger.info(
                    f"No signal: confidence below {self.config.model.confidence_min} "
                    f"({prediction['confidence']:.4f})"
                )

        except Exception as e:
            logger.error(f"Prediction cycle error: {e}", exc_info=True)

    async def _resolve_pending_signals(self, current_slot: datetime):
        """Resolve pending signals whose predicted candle has fully closed.

        Only resolves signals whose candle_slot_ts is strictly BEFORE
        current_slot, ensuring we never resolve a candle that's still live.

        Example timeline:
        - Signal at 16:44:45 predicts the 16:45-16:50 candle (candle_slot_ts=16:45)
        - At 16:50:30 (current_slot=16:50), 16:45 < 16:50 -> resolvable
        - Fetches the 16:45 candle's open/close from MEXC for WIN/LOSS

        Args:
            current_slot: The open timestamp of the current (live) candle
        """
        current_slot_iso = current_slot.isoformat()
        resolvable = self.tracker.get_resolvable_signals(current_slot_iso)
        if not resolvable:
            return

        try:
            # Fetch enough recent candles to cover any pending signals.
            # Most of the time we only need the last 2-3, but fetch more
            # to handle edge cases (bot was down, multiple pending).
            df = await self.fetcher.fetch_klines(interval="5m", limit=10)
            if df.empty:
                logger.warning("No candle data returned for resolution")
                return

            # Build a lookup: candle open timestamp -> (open_price, close_price)
            # The 'timestamp' column in df is the candle open time (from MEXC).
            candle_lookup = {}
            for _, row in df.iterrows():
                ts = row["timestamp"]
                # Convert pandas Timestamp to an offset-aware datetime for comparison
                if hasattr(ts, "isoformat"):
                    key = ts.isoformat()
                else:
                    key = str(ts)
                candle_lookup[key] = {
                    "open": float(row["open"]),
                    "close": float(row["close"]),
                }

            for sig in resolvable:
                # Find the candle matching this signal's predicted slot
                candle_data = candle_lookup.get(sig.candle_slot_ts)

                if candle_data is None:
                    # Try matching without timezone suffix variations
                    # MEXC returns UTC timestamps; our stored format might differ slightly
                    matched = False
                    try:
                        sig_dt = datetime.fromisoformat(sig.candle_slot_ts)
                        for key, val in candle_lookup.items():
                            key_dt = datetime.fromisoformat(key)
                            if abs((sig_dt - key_dt).total_seconds()) < 5:
                                candle_data = val
                                matched = True
                                break
                    except (ValueError, TypeError):
                        pass

                    if not matched:
                        logger.warning(
                            f"Signal #{sig.signal_id}: candle for slot {sig.candle_slot_ts} "
                            f"not found in fetched data. Will retry next cycle."
                        )
                        continue

                resolved = self.tracker.resolve_signal(
                    sig.signal_id,
                    candle_open=candle_data["open"],
                    candle_close=candle_data["close"],
                )
                if resolved:
                    stats = self.tracker.get_stats()
                    msg = formatters.format_resolution(resolved, stats)
                    await self.telegram.send_message(msg)

        except Exception as e:
            logger.error(f"Signal resolution error: {e}", exc_info=True)

    async def _resolve_stale_signals(self):
        """Resolve any pending signals from previous bot sessions.

        Called once at startup. Fetches historical candle data and resolves
        any signals that should have been resolved while the bot was down.
        """
        pending = self.tracker.get_pending_signals()
        if not pending:
            return

        logger.info(f"Found {len(pending)} stale pending signals at startup, attempting resolution...")

        try:
            # Fetch enough historical candles to cover stale signals
            df = await self.fetcher.fetch_klines(interval="5m", limit=50)
            if df.empty:
                logger.warning("No candle data for stale signal resolution")
                return

            now = datetime.now(timezone.utc)
            current_slot = _candle_slot_open(now)
            current_slot_iso = current_slot.isoformat()

            # Build candle lookup
            candle_lookup = {}
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if hasattr(ts, "isoformat"):
                    key = ts.isoformat()
                else:
                    key = str(ts)
                candle_lookup[key] = {
                    "open": float(row["open"]),
                    "close": float(row["close"]),
                }

            resolvable = self.tracker.get_resolvable_signals(current_slot_iso)
            resolved_count = 0

            for sig in resolvable:
                candle_data = None
                try:
                    sig_dt = datetime.fromisoformat(sig.candle_slot_ts)
                    for key, val in candle_lookup.items():
                        key_dt = datetime.fromisoformat(key)
                        if abs((sig_dt - key_dt).total_seconds()) < 5:
                            candle_data = val
                            break
                except (ValueError, TypeError):
                    pass

                if candle_data is None:
                    logger.warning(
                        f"Stale signal #{sig.signal_id}: candle for slot {sig.candle_slot_ts} "
                        f"not found in historical data."
                    )
                    continue

                resolved = self.tracker.resolve_signal(
                    sig.signal_id,
                    candle_open=candle_data["open"],
                    candle_close=candle_data["close"],
                )
                if resolved:
                    resolved_count += 1
                    stats = self.tracker.get_stats()
                    msg = formatters.format_resolution(resolved, stats)
                    await self.telegram.send_message(msg)

            if resolved_count > 0:
                logger.info(f"Resolved {resolved_count} stale signals at startup")

        except Exception as e:
            logger.error(f"Stale signal resolution error: {e}", exc_info=True)

    async def _train_model(self):
        """Train or retrain the model.

        Uses paginated historical fetch for ALL timeframes to ensure
        higher-TF features (15m, 1h) cover the full training window,
        not just the last 500 candles.
        """
        try:
            train_candles = self.config.model.train_candles
            logger.info(
                f"Fetching training data: {train_candles} 5m candles "
                f"(~{train_candles * 5 // 1440} days)..."
            )

            # Fetch historical 5m data (paginated)
            df_5m = await self.fetcher.fetch_historical_klines(
                interval="5m",
                total_candles=train_candles,
            )

            if df_5m.empty or len(df_5m) < 500:
                logger.error(f"Insufficient 5m training data: {len(df_5m)} candles")
                return

            logger.info(
                f"[DATA DIAGNOSTIC] 5m raw candles fetched: {len(df_5m)}, "
                f"range: {df_5m['timestamp'].iloc[0]} to {df_5m['timestamp'].iloc[-1]}"
            )

            # Fetch higher timeframe data with FULL pagination
            # (covers same calendar window as 5m data)
            higher_tf = await self.fetcher.fetch_historical_multi_timeframe(
                intervals=["15m", "1h"],
                train_candles_5m=train_candles,
            )
            higher_tf = {k: v for k, v in higher_tf.items() if not v.empty}

            # Log diagnostic info for higher TF data
            for tf_name, tf_df in higher_tf.items():
                logger.info(
                    f"[DATA DIAGNOSTIC] {tf_name} candles fetched: {len(tf_df)}, "
                    f"range: {tf_df['timestamp'].iloc[0]} to {tf_df['timestamp'].iloc[-1]}"
                )

            # Capture previous accuracy for delta display
            previous_accuracy = self.model.val_accuracy

            # Train (model internally handles the retraining gate)
            metrics = self.model.train(df_5m, higher_tf)

            # Log post-training diagnostics
            logger.info(
                f"[DATA DIAGNOSTIC] Post-training: "
                f"{metrics['total_samples']} usable samples from {len(df_5m)} raw 5m candles "
                f"({len(df_5m) - metrics['total_samples']} dropped by NaN/alignment)"
            )

            # Save model
            self.model.save(self.config.model_dir)

            # Send formatted training complete message
            msg = formatters.format_training_complete(metrics, previous_accuracy)
            await self.telegram.send_message(msg)
            logger.info("Model training complete")

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            await self.telegram.send_message(
                formatters.format_training_failed(str(e))
            )

    # --- Callback methods for Telegram commands ---

    def _get_stats_text(self) -> str:
        stats = self.tracker.get_stats()
        return formatters.format_stats(stats)

    def _get_recent_text(self) -> str:
        recent = self.tracker.get_recent_signals(10)
        if not recent:
            return "\U0001f4cb No signals recorded yet."
        stats = self.tracker.get_stats()
        return formatters.format_recent(recent, stats)

    async def _get_status_text(self) -> str:
        stats = self.tracker.get_stats()

        retrain_remaining = "N/A"
        if self.model.last_train_time:
            elapsed = (datetime.now(timezone.utc) - self.model.last_train_time).total_seconds()
            remaining = max(0, self.config.model.retrain_interval_hours * 3600 - elapsed)
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            retrain_remaining = f"{hours}h {minutes}m"

        return formatters.format_status(
            running=self._running,
            session_start=self.tracker.session_start,
            symbol=self.config.mexc.symbol,
            model_accuracy=self.model.val_accuracy,
            train_samples=self.model.train_samples,
            last_train_time=self.model.last_train_time,
            retrain_remaining=retrain_remaining,
            confidence_min=self.config.model.confidence_min,
            retrain_gate=self.config.model.retrain_min_improvement,
            optuna_enabled=self.config.model.enable_optuna_tuning,
            optuna_tuned=self.model.best_xgb_params is not None,
            total_signals=stats.total_signals,
            pending=stats.pending,
        )

    async def _retrain_model(self) -> str:
        try:
            await self._train_model()
            return formatters.format_retrain_complete(self.model.val_accuracy)
        except Exception as e:
            return formatters.format_retrain_failed(str(e))


async def run_bot():
    """Entry point to run the bot."""
    config = BotConfig.from_env()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    bot = SignalBot(config)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        await bot.stop()
        raise
