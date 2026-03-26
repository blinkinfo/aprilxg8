"""Regime filter with Telegram-controlled toggles and per-regime performance tracking.

Persists regime enable/disable state and per-regime W/L stats to disk.
Only tracks tradable signals (those that pass the tier gate in TradeManager).

Regimes:
    TRENDING_UP   (0)
    TRENDING_DOWN (1)
    RANGING       (2)
    VOLATILE      (3)
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Regime ID -> name mapping (matches RegimeDetector)
REGIME_NAMES = {
    0: "TRENDING_UP",
    1: "TRENDING_DOWN",
    2: "RANGING",
    3: "VOLATILE",
}

REGIME_NAME_TO_ID = {v: k for k, v in REGIME_NAMES.items()}

# All regimes enabled by default
DEFAULT_ENABLED = {name: True for name in REGIME_NAMES.values()}

# Binary payout constants
WIN_PAYOUT = 0.96
LOSS_PAYOUT = 1.00


class RegimeFilter:
    """Manages regime enable/disable state and per-regime performance stats.

    Persists to data/regime_filter.json.
    Only tracks tradable signals (those that passed the tier gate).
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._config_path = os.path.join(data_dir, "regime_filter.json")

        # Regime enabled state: {"TRENDING_UP": True, ...}
        self.enabled: dict[str, bool] = dict(DEFAULT_ENABLED)

        # Per-regime stats for tradable signals only
        # {"TRENDING_UP": {"wins": 0, "losses": 0}, ...}
        self.regime_stats: dict[str, dict[str, int]] = {
            name: {"wins": 0, "losses": 0}
            for name in REGIME_NAMES.values()
        }

        self._load()

    # ------------------------------------------------------------------
    # Filter Logic
    # ------------------------------------------------------------------

    def is_regime_enabled(self, regime_name: str) -> bool:
        """Check if a regime is enabled for trading.

        Args:
            regime_name: One of TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE

        Returns:
            True if the regime is enabled (trades allowed)
        """
        return self.enabled.get(regime_name, True)

    def is_regime_id_enabled(self, regime_id: int) -> bool:
        """Check if a regime is enabled by its numeric ID."""
        name = REGIME_NAMES.get(regime_id)
        if name is None:
            return True  # Unknown regimes are allowed
        return self.is_regime_enabled(name)

    def toggle_regime(self, regime_name: str) -> bool:
        """Toggle a regime's enabled state.

        Args:
            regime_name: Regime name (e.g. "TRENDING_UP")

        Returns:
            New enabled state (True/False)
        """
        if regime_name not in self.enabled:
            logger.warning(f"Unknown regime: {regime_name}")
            return True

        self.enabled[regime_name] = not self.enabled[regime_name]
        self._save()
        state = "ENABLED" if self.enabled[regime_name] else "DISABLED"
        logger.info(f"Regime {regime_name} toggled to {state}")
        return self.enabled[regime_name]

    def set_regime(self, regime_name: str, enabled: bool) -> None:
        """Set a regime's enabled state explicitly."""
        if regime_name in self.enabled:
            self.enabled[regime_name] = enabled
            self._save()

    # ------------------------------------------------------------------
    # Performance Tracking (tradable signals only)
    # ------------------------------------------------------------------

    def record_result(self, regime_name: str, result: str) -> None:
        """Record a trade result for a specific regime.

        Only call this for signals that passed the tier gate (tradable).

        Args:
            regime_name: Regime name (e.g. "TRENDING_UP")
            result: "WIN", "LOSS", or "NEUTRAL"
        """
        if regime_name not in self.regime_stats:
            self.regime_stats[regime_name] = {"wins": 0, "losses": 0}

        if result == "WIN":
            self.regime_stats[regime_name]["wins"] += 1
        elif result == "LOSS":
            self.regime_stats[regime_name]["losses"] += 1

        self._save()
        logger.debug(f"Regime {regime_name}: recorded {result} -> {self.regime_stats[regime_name]}")

    def get_regime_summary(self, regime_name: str) -> dict:
        """Get W/L summary for a specific regime.

        Returns:
            {"wins": int, "losses": int,
             "total": int, "win_rate": float, "pnl": float}
        """
        stats = self.regime_stats.get(regime_name, {"wins": 0, "losses": 0})
        wins = stats["wins"]
        losses = stats["losses"]
        decided = wins + losses
        win_rate = (wins / decided * 100) if decided > 0 else 0.0
        pnl = (wins * WIN_PAYOUT) - (losses * LOSS_PAYOUT)

        return {
            "wins": wins,
            "losses": losses,
            "total": decided,
            "decided": decided,
            "win_rate": win_rate,
            "pnl": pnl,
        }

    def get_all_regime_summaries(self) -> dict[str, dict]:
        """Get W/L summaries for all regimes.

        Returns:
            {"TRENDING_UP": {...}, "TRENDING_DOWN": {...}, ...}
        """
        return {
            name: self.get_regime_summary(name)
            for name in REGIME_NAMES.values()
        }

    def get_dashboard_data(self) -> list[dict]:
        """Get combined filter state + stats for the Telegram dashboard.

        Returns:
            List of dicts, one per regime, with keys:
            name, enabled, wins, losses, win_rate, pnl
        """
        result = []
        for regime_id in sorted(REGIME_NAMES.keys()):
            name = REGIME_NAMES[regime_id]
            summary = self.get_regime_summary(name)
            result.append({
                "name": name,
                "regime_id": regime_id,
                "enabled": self.enabled.get(name, True),
                "wins": summary["wins"],
                "losses": summary["losses"],
                "decided": summary["decided"],
                "win_rate": summary["win_rate"],
                "pnl": summary["pnl"],
            })
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Persist filter state and stats to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        data = {
            "enabled": self.enabled,
            "regime_stats": self.regime_stats,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._config_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save regime filter config: {e}")

    def _load(self):
        """Load filter state and stats from disk."""
        if not os.path.exists(self._config_path):
            logger.info("No regime filter config found, using defaults (all enabled)")
            return

        try:
            with open(self._config_path, "r") as f:
                data = json.load(f)

            # Load enabled state
            saved_enabled = data.get("enabled", {})
            for name in REGIME_NAMES.values():
                if name in saved_enabled:
                    self.enabled[name] = saved_enabled[name]

            # Load stats
            saved_stats = data.get("regime_stats", {})
            for name in REGIME_NAMES.values():
                if name in saved_stats:
                    self.regime_stats[name] = {
                        "wins": saved_stats[name].get("wins", 0),
                        "losses": saved_stats[name].get("losses", 0),
                    }

            logger.info(
                f"Regime filter loaded: "
                f"enabled={self.enabled}, "
                f"stats={self.regime_stats}"
            )
        except Exception as e:
            logger.error(f"Failed to load regime filter config: {e}")
