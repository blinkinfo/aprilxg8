"""Auto-trade orchestrator bridging XGBoost signals to Polymarket order execution.

Manages:
- Enable/disable toggle with persistent config
- Trade amount configuration
- Signal-to-trade pipeline with safety checks
- Balance verification before each trade
- Duplicate trade prevention per 5-min slot
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from .polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)

# Config persistence path
CONFIG_FILE = "data/autotrade_config.json"

# Trade amount bounds
MIN_TRADE_AMOUNT = 0.10  # Minimum 10 cents USDC
MAX_TRADE_AMOUNT = 100.0  # Safety cap
DEFAULT_TRADE_AMOUNT = 1.0  # Default $1 trade


class AutoTrader:
    """Bridges prediction signals to Polymarket trades."""

    def __init__(self, polymarket_client: PolymarketClient, data_dir: str = "data"):
        """
        Args:
            polymarket_client: Initialized PolymarketClient instance.
            data_dir: Directory for persisting config.
        """
        self._pm = polymarket_client
        self._data_dir = data_dir
        self._config_path = os.path.join(data_dir, "autotrade_config.json")

        # State
        self.enabled: bool = False
        self.trade_amount: float = DEFAULT_TRADE_AMOUNT
        self._last_traded_slot: Optional[int] = None
        self._trade_history: list = []  # Recent trades for this session

        # Load persisted config
        self._load_config()

    # ------------------------------------------------------------------
    # Config Persistence
    # ------------------------------------------------------------------

    def _load_config(self):
        """Load autotrade config from disk."""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r") as f:
                    config = json.load(f)
                self.enabled = config.get("enabled", False)
                self.trade_amount = config.get("trade_amount", DEFAULT_TRADE_AMOUNT)
                logger.info(
                    f"AutoTrader config loaded: enabled={self.enabled}, "
                    f"amount={self.trade_amount} USDC"
                )
            else:
                logger.info("No autotrade config found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load autotrade config: {e}")

    def _save_config(self):
        """Persist autotrade config to disk."""
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            config = {
                "enabled": self.enabled,
                "trade_amount": self.trade_amount,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"AutoTrader config saved: {config}")
        except Exception as e:
            logger.error(f"Failed to save autotrade config: {e}")

    # ------------------------------------------------------------------
    # Toggle & Settings
    # ------------------------------------------------------------------

    def toggle(self, on: Optional[bool] = None) -> dict:
        """Toggle auto-trading on/off.

        Args:
            on: If provided, set to this value. If None, flip current state.

        Returns:
            {enabled: bool, message: str}
        """
        if on is None:
            self.enabled = not self.enabled
        else:
            self.enabled = bool(on)

        self._save_config()
        state = "ENABLED" if self.enabled else "DISABLED"
        msg = f"Auto-trading {state} (amount: {self.trade_amount} USDC)"
        logger.info(msg)
        return {"enabled": self.enabled, "message": msg}

    def set_trade_amount(self, amount: float) -> dict:
        """Set the USDC trade amount per signal.

        Args:
            amount: Trade size in USDC.

        Returns:
            {success: bool, amount: float, message: str}
        """
        if amount < MIN_TRADE_AMOUNT:
            return {
                "success": False,
                "amount": self.trade_amount,
                "message": f"Amount too low. Minimum is {MIN_TRADE_AMOUNT} USDC.",
            }
        if amount > MAX_TRADE_AMOUNT:
            return {
                "success": False,
                "amount": self.trade_amount,
                "message": f"Amount too high. Maximum is {MAX_TRADE_AMOUNT} USDC.",
            }

        self.trade_amount = round(amount, 2)
        self._save_config()
        msg = f"Trade amount set to {self.trade_amount} USDC"
        logger.info(msg)
        return {"success": True, "amount": self.trade_amount, "message": msg}

    def get_config(self) -> dict:
        """Get current autotrade configuration.

        Returns:
            {enabled, trade_amount, last_traded_slot, session_trades}
        """
        return {
            "enabled": self.enabled,
            "trade_amount": self.trade_amount,
            "last_traded_slot": self._last_traded_slot,
            "session_trades": len(self._trade_history),
        }

    # ------------------------------------------------------------------
    # Trade Execution
    # ------------------------------------------------------------------

    async def execute_trade(self, signal: dict) -> dict:
        """Execute a trade based on a prediction signal.

        Called by the bot after a signal is generated. Performs all safety checks
        then delegates to PolymarketClient.place_trade().

        Args:
            signal: Prediction dict with keys:
                - signal: "UP", "DOWN", or "NEUTRAL"
                - confidence: float (0-1)
                - strength: "STRONG", "NORMAL", or "SKIP"
                - current_price: float (BTC price)
                - prob_up: float
                - prob_down: float

        Returns:
            {success: bool, action: str, data: dict|None, error: str|None}
        """
        direction = signal.get("signal", "NEUTRAL")
        confidence = signal.get("confidence", 0)
        strength = signal.get("strength", "SKIP")

        # --- Safety Check 1: Is auto-trading enabled? ---
        if not self.enabled:
            return {
                "success": False,
                "action": "skipped",
                "data": None,
                "error": "Auto-trading is disabled",
            }

        # --- Safety Check 2: Is direction tradeable? ---
        if direction not in ("UP", "DOWN"):
            return {
                "success": False,
                "action": "skipped",
                "data": {"direction": direction, "confidence": confidence},
                "error": f"Signal is {direction} — no trade placed",
            }

        # --- Safety Check 3: Is Polymarket client initialized? ---
        if not self._pm.is_initialized:
            return {
                "success": False,
                "action": "error",
                "data": None,
                "error": "Polymarket client not initialized",
            }

        # --- Safety Check 4: Duplicate slot prevention ---
        current_slot = PolymarketClient.get_current_slot_timestamp()
        if self._last_traded_slot == current_slot:
            return {
                "success": False,
                "action": "skipped",
                "data": {"slot_ts": current_slot},
                "error": f"Already traded this slot ({PolymarketClient.slot_to_datetime(current_slot).strftime('%H:%M')} UTC)",
            }

        # --- Safety Check 5: Sufficient balance ---
        bal_result = await self._pm.get_balance()
        if not bal_result["success"]:
            return {
                "success": False,
                "action": "error",
                "data": None,
                "error": f"Balance check failed: {bal_result['error']}",
            }

        balance = bal_result["data"]["balance"]
        if balance < self.trade_amount:
            return {
                "success": False,
                "action": "error",
                "data": {"balance": balance, "required": self.trade_amount},
                "error": f"Insufficient balance: {balance:.2f} USDC < {self.trade_amount:.2f} USDC",
            }

        # --- Execute Trade ---
        logger.info(
            f"Executing trade: {direction} | conf={confidence:.4f} | "
            f"strength={strength} | amount={self.trade_amount} USDC"
        )

        trade_result = await self._pm.place_trade(
            direction=direction,
            amount=self.trade_amount,
        )

        if trade_result["success"]:
            # Mark slot as traded
            self._last_traded_slot = current_slot

            # Enrich trade data with signal info
            trade_data = trade_result["data"]
            trade_data["confidence"] = confidence
            trade_data["strength"] = strength
            trade_data["balance_before"] = balance
            trade_data["balance_after"] = balance - self.trade_amount

            # Add to session history
            self._trade_history.append(trade_data)

            logger.info(
                f"Trade SUCCESS: {direction} | order={trade_data.get('order_id')} | "
                f"price={trade_data.get('price')} | slot={trade_data.get('slot_dt')}"
            )
            return {
                "success": True,
                "action": "traded",
                "data": trade_data,
                "error": None,
            }
        else:
            logger.error(f"Trade FAILED: {trade_result['error']}")
            return {
                "success": False,
                "action": "error",
                "data": {"direction": direction, "amount": self.trade_amount},
                "error": trade_result["error"],
            }

    # ------------------------------------------------------------------
    # Session Stats
    # ------------------------------------------------------------------

    def get_session_stats(self) -> dict:
        """Get stats for the current session's auto-trades."""
        if not self._trade_history:
            return {
                "total_trades": 0,
                "total_amount": 0.0,
                "directions": {"UP": 0, "DOWN": 0},
                "avg_confidence": 0.0,
            }

        up_count = sum(1 for t in self._trade_history if t.get("direction") == "UP")
        down_count = sum(1 for t in self._trade_history if t.get("direction") == "DOWN")
        total_amount = sum(t.get("amount", 0) for t in self._trade_history)
        avg_conf = sum(t.get("confidence", 0) for t in self._trade_history) / len(self._trade_history)

        return {
            "total_trades": len(self._trade_history),
            "total_amount": round(total_amount, 2),
            "directions": {"UP": up_count, "DOWN": down_count},
            "avg_confidence": round(avg_conf, 4),
        }
