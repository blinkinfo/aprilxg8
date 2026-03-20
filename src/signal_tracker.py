"""Signal tracking with win/loss, PnL, streaks, and statistics.

Fixed: candle slot timestamps, open/close resolution, UTC time throughout.

UI/UX v2:
- format_signal_message, format_resolution_message, format_stats_message
  moved to formatters.py. This module is now pure data/logic.
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A single signal record."""
    signal_id: int
    direction: str  # UP or DOWN
    confidence: float
    entry_price: float  # Current live price at signal time (kept for display)
    timestamp: str  # When the signal was created (ISO UTC)
    candle_slot_ts: str = ""  # Candle open timestamp in ISO UTC (e.g. 09:00:00)
    candle_open_price: float = 0.0  # The candle's actual open price
    # Filled after candle closes
    exit_price: Optional[float] = None  # Candle close price
    result: Optional[str] = None  # WIN, LOSS, or NEUTRAL
    pnl_pct: Optional[float] = None
    resolved_at: Optional[str] = None


@dataclass
class TrackerStats:
    """Aggregated statistics."""
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    neutral: int = 0
    pending: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    current_streak: int = 0
    current_streak_type: str = ""  # WIN or LOSS
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    avg_confidence: float = 0.0
    # Session info
    session_start: str = ""
    last_signal_time: str = ""


class SignalTracker:
    """Tracks signals and computes performance statistics."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.signals: list[Signal] = []
        self._next_id = 1
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._load()

    @property
    def session_start(self) -> str:
        """Expose session start for status display."""
        return self._session_start

    def add_signal(
        self,
        direction: str,
        confidence: float,
        entry_price: float,
        candle_slot_ts: str = "",
        candle_open_price: float = 0.0,
    ) -> Signal:
        """Add a new signal.

        Args:
            direction: UP or DOWN
            confidence: Model confidence (0-1)
            entry_price: Current live price at signal time (kept for display)
            candle_slot_ts: ISO UTC timestamp of the candle's open (e.g. 09:00:00)
            candle_open_price: The candle's actual open price from MEXC

        Returns:
            The created Signal object
        """
        signal = Signal(
            signal_id=self._next_id,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
            candle_slot_ts=candle_slot_ts,
            candle_open_price=candle_open_price,
        )
        self.signals.append(signal)
        self._next_id += 1
        self._save()
        logger.info(
            f"Signal #{signal.signal_id}: {direction} @ ${entry_price:,.2f} "
            f"(conf={confidence:.4f}, slot={candle_slot_ts})"
        )
        return signal

    def resolve_signal(
        self,
        signal_id: int,
        candle_open: float,
        candle_close: float,
    ) -> Optional[Signal]:
        """Resolve a pending signal using the candle's open and close prices.

        WIN/LOSS is determined by whether the candle closed in the predicted
        direction relative to its OPEN price — matching Polymarket binary outcome.

        Args:
            signal_id: Signal ID to resolve
            candle_open: The candle's open price
            candle_close: The candle's close price

        Returns:
            The resolved Signal or None
        """
        signal = self._find_signal(signal_id)
        if signal is None or signal.result is not None:
            return None

        # Determine actual candle direction
        candle_went_up = candle_close > candle_open
        candle_went_down = candle_close < candle_open

        # Calculate PnL % based on candle open vs close
        if signal.direction == "UP":
            pnl_pct = ((candle_close - candle_open) / candle_open) * 100
        else:  # DOWN
            pnl_pct = ((candle_open - candle_close) / candle_open) * 100

        # WIN/LOSS matches Polymarket: did the candle go in the predicted direction?
        if signal.direction == "UP" and candle_went_up:
            signal.result = "WIN"
        elif signal.direction == "DOWN" and candle_went_down:
            signal.result = "WIN"
        elif candle_open == candle_close:
            signal.result = "NEUTRAL"
        else:
            signal.result = "LOSS"

        # Store candle open as entry, candle close as exit
        signal.candle_open_price = candle_open
        signal.exit_price = candle_close
        signal.pnl_pct = round(pnl_pct, 4)
        signal.resolved_at = datetime.now(timezone.utc).isoformat()

        self._save()
        logger.info(
            f"Signal #{signal_id} resolved: {signal.result} "
            f"(open=${candle_open:,.2f} -> close=${candle_close:,.2f}, pnl={pnl_pct:+.4f}%)"
        )
        return signal

    def get_pending_signals(self) -> list[Signal]:
        """Get all unresolved signals."""
        return [s for s in self.signals if s.result is None]

    def get_resolvable_signals(self, current_candle_open_ts: str) -> list[Signal]:
        """Get pending signals whose candle slot is OLDER than the current candle.

        This prevents resolving a signal for a candle that hasn't closed yet.

        Args:
            current_candle_open_ts: ISO UTC timestamp of the current (live) candle's open

        Returns:
            List of signals safe to resolve
        """
        pending = self.get_pending_signals()
        if not pending:
            return []

        try:
            current_ts = datetime.fromisoformat(current_candle_open_ts)
        except (ValueError, TypeError):
            logger.error(f"Invalid current_candle_open_ts: {current_candle_open_ts}")
            return []

        resolvable = []
        for s in pending:
            if not s.candle_slot_ts:
                # Legacy signal without slot timestamp — skip (can't verify age)
                logger.warning(f"Signal #{s.signal_id} has no candle_slot_ts, skipping resolution")
                continue
            try:
                signal_candle_ts = datetime.fromisoformat(s.candle_slot_ts)
                if signal_candle_ts < current_ts:
                    resolvable.append(s)
                else:
                    logger.debug(
                        f"Signal #{s.signal_id} candle slot {s.candle_slot_ts} "
                        f"not yet closed (current candle: {current_candle_open_ts})"
                    )
            except (ValueError, TypeError):
                logger.warning(f"Signal #{s.signal_id} has invalid candle_slot_ts: {s.candle_slot_ts}")
                continue

        return resolvable

    def get_stats(self) -> TrackerStats:
        """Compute comprehensive statistics."""
        stats = TrackerStats()
        stats.session_start = self._session_start
        stats.total_signals = len(self.signals)

        resolved = [s for s in self.signals if s.result is not None]
        stats.pending = len(self.signals) - len(resolved)

        if not resolved:
            return stats

        stats.wins = sum(1 for s in resolved if s.result == "WIN")
        stats.losses = sum(1 for s in resolved if s.result == "LOSS")
        stats.neutral = sum(1 for s in resolved if s.result == "NEUTRAL")

        decided = stats.wins + stats.losses
        stats.win_rate = (stats.wins / decided * 100) if decided > 0 else 0.0

        pnls = [s.pnl_pct for s in resolved if s.pnl_pct is not None]
        if pnls:
            stats.total_pnl_pct = round(sum(pnls), 4)
            stats.avg_pnl_pct = round(stats.total_pnl_pct / len(pnls), 4)
            stats.best_trade_pct = round(max(pnls), 4)
            stats.worst_trade_pct = round(min(pnls), 4)

            wins_pnl = [p for p in pnls if p > 0]
            losses_pnl = [p for p in pnls if p < 0]
            stats.avg_win_pct = round(sum(wins_pnl) / len(wins_pnl), 4) if wins_pnl else 0.0
            stats.avg_loss_pct = round(sum(losses_pnl) / len(losses_pnl), 4) if losses_pnl else 0.0

        # Streaks
        streak_count = 0
        streak_type = ""
        longest_win = 0
        longest_loss = 0

        for s in resolved:
            if s.result == "NEUTRAL":
                continue
            if s.result == streak_type:
                streak_count += 1
            else:
                streak_type = s.result
                streak_count = 1

            if streak_type == "WIN":
                longest_win = max(longest_win, streak_count)
            elif streak_type == "LOSS":
                longest_loss = max(longest_loss, streak_count)

        stats.current_streak = streak_count
        stats.current_streak_type = streak_type
        stats.longest_win_streak = longest_win
        stats.longest_loss_streak = longest_loss

        # Average confidence
        confidences = [s.confidence for s in self.signals]
        stats.avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # Last signal time
        stats.last_signal_time = self.signals[-1].timestamp if self.signals else ""

        return stats

    def get_recent_signals(self, n: int = 10) -> list[Signal]:
        """Get the N most recent signals."""
        return self.signals[-n:]

    def _find_signal(self, signal_id: int) -> Optional[Signal]:
        for s in self.signals:
            if s.signal_id == signal_id:
                return s
        return None

    def _save(self):
        """Persist signals to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, "signals.json")
        data = {
            "next_id": self._next_id,
            "session_start": self._session_start,
            "signals": [asdict(s) for s in self.signals],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load signals from disk."""
        path = os.path.join(self.data_dir, "signals.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._next_id = data.get("next_id", 1)
            self._session_start = data.get("session_start", self._session_start)
            for s_data in data.get("signals", []):
                # Handle legacy signals that don't have new fields
                s_data.setdefault("candle_slot_ts", "")
                s_data.setdefault("candle_open_price", 0.0)
                self.signals.append(Signal(**s_data))
            logger.info(f"Loaded {len(self.signals)} signals from disk")
        except Exception as e:
            logger.error(f"Failed to load signals: {e}")
