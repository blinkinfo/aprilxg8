"""Walk-forward backtesting engine with no-lookahead bias.

Design principles:
- Uses the CURRENT frozen model (no retraining during backtest)
- Sliding window feature computation: only past data visible at each step
- Higher-timeframe data filtered to timestamps <= current candle
- Completely isolated from live trading, signal tracker, and auto-trader
- Async with progress callbacks for Telegram updates

No-Lookahead Guarantee:
    For each candle i in the backtest window, features are computed using
    only candles [max(0, i - WINDOW_SIZE) : i + 1]. Higher-TF data is
    filtered to timestamps <= candle[i].timestamp. The model predicts
    the direction of candle[i + 1], which is then compared to its actual
    open/close. At no point does any future data leak into the prediction.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable

import pandas as pd

from .config import BotConfig
from .data_fetcher import MEXCFetcher
from .features import FeatureEngineer
from .model import PredictionModel

logger = logging.getLogger(__name__)

# Sliding window size for feature computation.
# Must be large enough for the longest indicator lookback.
# Longest: ATR regime percentile (lookback=100), SMA(50), rolling z-scores(50).
# 200 candles provides ample warm-up headroom.
FEATURE_WINDOW_SIZE = 200

# Minimum candles required for compute_features to produce output
MIN_FEATURE_CANDLES = 50

# Send progress update every N candles processed
PROGRESS_INTERVAL = 250

# Binary market payout constants (must match formatters.py)
WIN_PAYOUT = 0.96
LOSS_PAYOUT = 1.00


@dataclass
class BacktestSignal:
    """A single backtest signal record."""
    index: int  # Index into the backtest candle array
    candle_timestamp: str  # ISO UTC timestamp of the candle being predicted
    direction: str  # UP or DOWN
    confidence: float
    prob_up: float
    prob_down: float
    strength: str  # STRONG or NORMAL
    # Resolution
    actual_direction: str  # UP, DOWN, or NEUTRAL
    result: str  # WIN, LOSS, or NEUTRAL
    pnl: float  # +0.96 or -1.00 or 0.0
    open_price: float
    close_price: float
    is_oos: bool = True  # True if this signal is on out-of-sample data


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    # Request parameters
    n_candles: int
    actual_candles: int  # May be less if MEXC didn't have enough data
    # Time range
    start_time: str
    end_time: str
    # Signal counts
    total_signals: int = 0
    skipped_low_confidence: int = 0
    # Win/Loss
    wins: int = 0
    losses: int = 0
    neutral: int = 0
    win_rate: float = 0.0
    # PnL
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    # Confidence breakdown
    strong_signals: int = 0
    strong_wins: int = 0
    strong_win_rate: float = 0.0
    normal_signals: int = 0
    normal_wins: int = 0
    normal_win_rate: float = 0.0
    # Direction breakdown
    up_signals: int = 0
    up_wins: int = 0
    up_win_rate: float = 0.0
    down_signals: int = 0
    down_wins: int = 0
    down_win_rate: float = 0.0
    # Streaks
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    # Model info
    model_accuracy: Optional[float] = None
    model_train_samples: Optional[int] = None
    # Timing
    elapsed_seconds: float = 0.0
    fetch_seconds: float = 0.0
    # Individual signals for detailed view
    signals: list = field(default_factory=list)
    # Error info
    error: Optional[str] = None
    # Out-of-sample tracking
    train_end_ts: Optional[str] = None  # Model's training data cutoff (ISO string)
    oos_candles: int = 0  # Number of truly out-of-sample candles evaluated
    in_sample_candles: int = 0  # Number of candles that overlapped with training data
    oos_warning: Optional[str] = None  # Warning if backtest includes in-sample data


class Backtester:
    """Walk-forward backtesting engine.

    Uses the current frozen model to simulate predictions on historical data
    with strict no-lookahead bias.
    """

    def __init__(
        self,
        model: PredictionModel,
        fetcher: MEXCFetcher,
        config: BotConfig,
    ):
        self.model = model
        self.fetcher = fetcher
        self.config = config
        self.feature_engineer = FeatureEngineer(config.model)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def run(
        self,
        n_candles: int,
        progress_callback: Optional[Callable[[int, int, int, int], Awaitable[None]]] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest on the last n_candles.

        Args:
            n_candles: Number of 5m candles to backtest.
            progress_callback: async fn(processed, total, signals_so_far, wins_so_far)
                               Called every PROGRESS_INTERVAL candles.

        Returns:
            BacktestResult with full statistics.
        """
        if self.model.model is None:
            return BacktestResult(
                n_candles=n_candles,
                actual_candles=0,
                start_time="",
                end_time="",
                error="No model loaded. Train the model first with /retrain.",
            )

        if self._running:
            return BacktestResult(
                n_candles=n_candles,
                actual_candles=0,
                start_time="",
                end_time="",
                error="A backtest is already running. Please wait for it to finish.",
            )

        self._running = True
        start_time = time.monotonic()

        try:
            return await self._run_internal(n_candles, progress_callback, start_time)
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return BacktestResult(
                n_candles=n_candles,
                actual_candles=0,
                start_time="",
                end_time="",
                error=f"Backtest failed: {e}",
                elapsed_seconds=time.monotonic() - start_time,
            )
        finally:
            self._running = False

    async def _run_internal(
        self,
        n_candles: int,
        progress_callback: Optional[Callable] = None,
        overall_start: float = 0.0,
    ) -> BacktestResult:
        """Internal backtest logic."""
        # ---------------------------------------------------------------
        # 1. Fetch historical data
        # ---------------------------------------------------------------
        # We need extra candles for feature warm-up, plus 1 for the
        # resolution of the last prediction.
        total_needed = n_candles + FEATURE_WINDOW_SIZE + 1

        logger.info(
            f"Backtest: fetching {total_needed} 5m candles "
            f"({n_candles} backtest + {FEATURE_WINDOW_SIZE} warmup + 1 resolution)"
        )

        fetch_start = time.monotonic()

        # Fetch 5m candles
        df_5m = await self.fetcher.fetch_historical_klines(
            interval="5m",
            total_candles=total_needed,
        )

        if df_5m.empty or len(df_5m) < MIN_FEATURE_CANDLES + 2:
            return BacktestResult(
                n_candles=n_candles,
                actual_candles=0,
                start_time="",
                end_time="",
                error="Could not fetch enough historical data from MEXC.",
                elapsed_seconds=time.monotonic() - overall_start,
            )

        # Fetch higher-timeframe data covering the same window
        higher_tf = await self.fetcher.fetch_historical_multi_timeframe(
            intervals=["15m", "1h"],
            train_candles_5m=total_needed,
        )

        fetch_seconds = time.monotonic() - fetch_start
        logger.info(f"Backtest: data fetch complete in {fetch_seconds:.1f}s")

        # ---------------------------------------------------------------
        # 2. Determine actual backtest range
        # ---------------------------------------------------------------
        # The backtest window starts at FEATURE_WINDOW_SIZE and ends at
        # len(df_5m) - 2 (need i+1 for resolution).
        total_available = len(df_5m)
        bt_start_idx = FEATURE_WINDOW_SIZE
        bt_end_idx = total_available - 2  # last index where we can resolve i+1

        if bt_end_idx <= bt_start_idx:
            return BacktestResult(
                n_candles=n_candles,
                actual_candles=0,
                start_time="",
                end_time="",
                error=(
                    f"Not enough data for backtest. Fetched {total_available} candles "
                    f"but need at least {FEATURE_WINDOW_SIZE + 2}."
                ),
                elapsed_seconds=time.monotonic() - overall_start,
            )

        # Clamp to requested n_candles
        actual_bt_candles = min(n_candles, bt_end_idx - bt_start_idx + 1)
        bt_end_idx = bt_start_idx + actual_bt_candles - 1

        start_ts = df_5m["timestamp"].iloc[bt_start_idx].isoformat()
        end_ts = df_5m["timestamp"].iloc[bt_end_idx].isoformat()

        logger.info(
            f"Backtest: simulating {actual_bt_candles} candles "
            f"from {start_ts} to {end_ts}"
        )

        # ---------------------------------------------------------------
        # 2b. Out-of-sample enforcement
        # ---------------------------------------------------------------
        # Check if model has training data cutoff info
        train_end_ts_dt = None
        oos_start_idx = bt_start_idx  # default: all candles are OOS
        in_sample_count = 0

        if hasattr(self.model, 'train_end_ts') and self.model.train_end_ts is not None:
            train_end_ts_dt = self.model.train_end_ts
            # Find the first candle index AFTER the training cutoff
            candle_timestamps = df_5m["timestamp"].values
            for check_idx in range(bt_start_idx, bt_end_idx + 1):
                candle_ts = pd.Timestamp(candle_timestamps[check_idx])
                if candle_ts.tzinfo is None:
                    candle_ts = candle_ts.tz_localize("UTC")
                train_end_aware = train_end_ts_dt
                if train_end_aware.tzinfo is None:
                    from datetime import timezone as tz
                    train_end_aware = train_end_aware.replace(tzinfo=tz.utc)
                if candle_ts > train_end_aware:
                    oos_start_idx = check_idx
                    break
            else:
                # ALL backtest candles are within training data
                oos_start_idx = bt_end_idx + 1  # no OOS candles

            in_sample_count = max(0, oos_start_idx - bt_start_idx)

            if in_sample_count > 0:
                logger.warning(
                    f"Backtest: {in_sample_count} of {actual_bt_candles} candles overlap "
                    f"with training data (train_end={train_end_ts_dt.isoformat()}). "
                    f"Only {actual_bt_candles - in_sample_count} are truly out-of-sample."
                )
        else:
            logger.warning(
                "Backtest: model has no train_end_ts \u2014 cannot verify out-of-sample status. "
                "Results may include in-sample data and appear inflated."
            )

        # ---------------------------------------------------------------
        # 3. Walk-forward simulation
        # ---------------------------------------------------------------
        signals: list[BacktestSignal] = []
        skipped = 0
        confidence_threshold = self.config.model.confidence_min
        confidence_strong = self.config.model.confidence_strong

        # Pre-extract numpy arrays for faster resolution checks
        opens = df_5m["open"].values
        closes = df_5m["close"].values

        for step, i in enumerate(range(bt_start_idx, bt_end_idx + 1)):
            # --- No-lookahead window: only candles up to and including i ---
            window_start = max(0, i - FEATURE_WINDOW_SIZE + 1)
            window_5m = df_5m.iloc[window_start: i + 1].copy()

            # Filter higher-TF data to only candles <= current timestamp
            current_ts = df_5m["timestamp"].iloc[i]
            filtered_htf = {}
            for tf_key, tf_df in higher_tf.items():
                if tf_df is not None and not tf_df.empty:
                    mask = tf_df["timestamp"] <= current_ts
                    filtered_htf[tf_key] = tf_df.loc[mask].copy()
                else:
                    filtered_htf[tf_key] = pd.DataFrame()

            # --- Compute features on the window ---
            try:
                features_df = self.feature_engineer.compute_features(
                    window_5m, higher_tf_data=filtered_htf
                )
            except Exception as e:
                logger.debug(f"Backtest step {step}: feature computation failed: {e}")
                skipped += 1
                continue

            if features_df.empty:
                skipped += 1
                continue

            # --- Predict using frozen model (raw predict_proba) ---
            try:
                expected_cols = self.model.model.get_booster().feature_names
                latest = features_df[expected_cols].iloc[[-1]]
                proba = self.model.model.predict_proba(latest)[0]
                prob_down, prob_up = float(proba[0]), float(proba[1])
                confidence = max(prob_up, prob_down)
                direction = "UP" if prob_up >= prob_down else "DOWN"
            except Exception as e:
                logger.debug(f"Backtest step {step}: prediction failed: {e}")
                skipped += 1
                continue

            # --- Apply confidence filter (same as live) ---
            if confidence < confidence_threshold:
                skipped += 1
                continue

            # --- Resolve against NEXT candle (i+1) ---
            next_idx = i + 1
            next_open = float(opens[next_idx])
            next_close = float(closes[next_idx])

            if next_close > next_open:
                actual_direction = "UP"
            elif next_close < next_open:
                actual_direction = "DOWN"
            else:
                actual_direction = "NEUTRAL"

            # Determine result
            if actual_direction == "NEUTRAL":
                result = "NEUTRAL"
                pnl = 0.0
            elif direction == actual_direction:
                result = "WIN"
                pnl = WIN_PAYOUT
            else:
                result = "LOSS"
                pnl = -LOSS_PAYOUT

            strength = "STRONG" if confidence >= confidence_strong else "NORMAL"

            signal = BacktestSignal(
                index=i,
                candle_timestamp=df_5m["timestamp"].iloc[next_idx].isoformat(),
                direction=direction,
                confidence=confidence,
                prob_up=prob_up,
                prob_down=prob_down,
                strength=strength,
                actual_direction=actual_direction,
                result=result,
                pnl=pnl,
                open_price=next_open,
                close_price=next_close,
            )
            signal.is_oos = (i >= oos_start_idx)
            signals.append(signal)

            # --- Progress callback ---
            if progress_callback and (step + 1) % PROGRESS_INTERVAL == 0:
                wins_so_far = sum(1 for s in signals if s.result == "WIN")
                try:
                    await progress_callback(
                        step + 1, actual_bt_candles, len(signals), wins_so_far
                    )
                except Exception:
                    pass  # Don't let callback errors break the backtest

            # Yield control periodically to keep the event loop responsive
            if (step + 1) % 100 == 0:
                await asyncio.sleep(0)

        # ---------------------------------------------------------------
        # 4. Compute aggregate statistics
        # ---------------------------------------------------------------
        result = self._compute_stats(
            signals=signals,
            n_candles=n_candles,
            actual_candles=actual_bt_candles,
            skipped=skipped,
            start_ts=start_ts,
            end_ts=end_ts,
            elapsed=time.monotonic() - overall_start,
            fetch_seconds=fetch_seconds,
            train_end_ts_str=train_end_ts_dt.isoformat() if train_end_ts_dt else None,
            in_sample_count=in_sample_count,
        )

        logger.info(
            f"Backtest complete: {result.total_signals} signals, "
            f"{result.wins}W/{result.losses}L, "
            f"win rate {result.win_rate:.1%}, "
            f"PnL ${result.total_pnl:+.2f}"
        )

        if result.oos_warning:
            logger.warning(result.oos_warning)

        # Also log OOS-only win rate for comparison
        oos_signals = [s for s in signals if s.is_oos]
        if oos_signals:
            oos_wins = sum(1 for s in oos_signals if s.result == "WIN")
            oos_decided = sum(1 for s in oos_signals if s.result in ("WIN", "LOSS"))
            if oos_decided > 0:
                logger.info(
                    f"Backtest OOS-only: {oos_wins}/{oos_decided} = "
                    f"{oos_wins/oos_decided:.1%} win rate"
                )

        return result

    def _compute_stats(
        self,
        signals: list[BacktestSignal],
        n_candles: int,
        actual_candles: int,
        skipped: int,
        start_ts: str,
        end_ts: str,
        elapsed: float,
        fetch_seconds: float,
        train_end_ts_str: str = None,
        in_sample_count: int = 0,
    ) -> BacktestResult:
        """Compute aggregate statistics from backtest signals."""
        result = BacktestResult(
            n_candles=n_candles,
            actual_candles=actual_candles,
            start_time=start_ts,
            end_time=end_ts,
            total_signals=len(signals),
            skipped_low_confidence=skipped,
            elapsed_seconds=elapsed,
            fetch_seconds=fetch_seconds,
            signals=signals,
        )

        # Model info
        if hasattr(self.model, "val_accuracy") and self.model.val_accuracy is not None:
            result.model_accuracy = self.model.val_accuracy
        if hasattr(self.model, "train_samples") and self.model.train_samples is not None:
            result.model_train_samples = self.model.train_samples

        if not signals:
            return result

        # Win/Loss counts
        result.wins = sum(1 for s in signals if s.result == "WIN")
        result.losses = sum(1 for s in signals if s.result == "LOSS")
        result.neutral = sum(1 for s in signals if s.result == "NEUTRAL")

        decided = result.wins + result.losses
        result.win_rate = result.wins / decided if decided > 0 else 0.0

        # PnL
        result.total_pnl = sum(s.pnl for s in signals)
        result.avg_pnl = result.total_pnl / len(signals) if signals else 0.0

        # --- Confidence breakdown ---
        strong = [s for s in signals if s.strength == "STRONG"]
        normal = [s for s in signals if s.strength == "NORMAL"]

        result.strong_signals = len(strong)
        result.strong_wins = sum(1 for s in strong if s.result == "WIN")
        strong_decided = sum(1 for s in strong if s.result in ("WIN", "LOSS"))
        result.strong_win_rate = (
            result.strong_wins / strong_decided if strong_decided > 0 else 0.0
        )

        result.normal_signals = len(normal)
        result.normal_wins = sum(1 for s in normal if s.result == "WIN")
        normal_decided = sum(1 for s in normal if s.result in ("WIN", "LOSS"))
        result.normal_win_rate = (
            result.normal_wins / normal_decided if normal_decided > 0 else 0.0
        )

        # --- Direction breakdown ---
        up_signals = [s for s in signals if s.direction == "UP"]
        down_signals = [s for s in signals if s.direction == "DOWN"]

        result.up_signals = len(up_signals)
        result.up_wins = sum(1 for s in up_signals if s.result == "WIN")
        up_decided = sum(1 for s in up_signals if s.result in ("WIN", "LOSS"))
        result.up_win_rate = result.up_wins / up_decided if up_decided > 0 else 0.0

        result.down_signals = len(down_signals)
        result.down_wins = sum(1 for s in down_signals if s.result == "WIN")
        down_decided = sum(1 for s in down_signals if s.result in ("WIN", "LOSS"))
        result.down_win_rate = (
            result.down_wins / down_decided if down_decided > 0 else 0.0
        )

        # --- Streaks ---
        current_streak = 0
        current_type = ""
        max_win_streak = 0
        max_loss_streak = 0

        for s in signals:
            if s.result == "NEUTRAL":
                continue
            if s.result == current_type:
                current_streak += 1
            else:
                current_type = s.result
                current_streak = 1

            if current_type == "WIN":
                max_win_streak = max(max_win_streak, current_streak)
            elif current_type == "LOSS":
                max_loss_streak = max(max_loss_streak, current_streak)

        result.longest_win_streak = max_win_streak
        result.longest_loss_streak = max_loss_streak


        # OOS tracking
        result.train_end_ts = train_end_ts_str
        result.in_sample_candles = in_sample_count
        oos_signals = [s for s in signals if s.is_oos]
        result.oos_candles = len(oos_signals)

        if in_sample_count > 0 and signals:
            total = len(signals)
            oos_count = len(oos_signals)
            result.oos_warning = (
                f"WARNING: {in_sample_count} candles in this backtest overlap with model training data. "
                f"Only {oos_count}/{total} signals ({oos_count/total*100:.0f}%) are truly out-of-sample. "
                f"In-sample results appear inflated. Retrain on older data or wait for new candles to get accurate OOS metrics."
            )
        return result
