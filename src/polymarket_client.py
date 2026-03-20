"""Polymarket CLOB API client for 5-min BTC Up/Down market trading.

Handles:
- Authentication via private key + derived API credentials
- Market discovery via Gamma API slug-based lookup (deterministic)
- Order placement (FOK market buy) on the correct Up/Down token
- Balance fetching, open positions, connection health checks
- Duplicate trade prevention per slot

Market Discovery Strategy:
    BTC 5-min Up/Down markets follow a deterministic slug pattern:
        btc-updown-5m-{slot_timestamp}
    where slot_timestamp = (unix_time // 300) * 300 (current 5-min boundary).

    We look up the market directly by slug via:
        GET https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-{ts}
    This is 100% reliable — no keyword searching needed.

Slot-Targeted Trading:
    Signals predict the NEXT 5-min candle. The bot fires ~15 seconds before
    the current candle closes, so at 16:44:45 UTC the signal is for the
    16:45-16:50 slot. The target_slot_ts (Unix timestamp of 16:45:00) is
    passed explicitly through the entire trade pipeline to ensure the order
    lands on the correct Polymarket market.

Order Execution (FOK Market Orders):
    Uses MarketOrderArgs with OrderType.FOK (Fill-or-Kill) for immediate
    execution at best available market price.

    For BUY orders: amount = USDC to spend (e.g. 1.0 = $1 USDC).
    The SDK auto-calculates the optimal price from the order book and
    determines the share quantity internally.

    FOK orders either fill entirely and immediately, or are rejected.
    No partial fills, no resting orders on the book.

Requires:
    py-clob-client >= 0.34.6
    Environment variables: POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS,
                           POLYMARKET_SIGNATURE_TYPE (optional, default 2)
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Polymarket API endpoints
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Polygon mainnet chain ID
CHAIN_ID = 137

# 5-minute candle period in seconds
SLOT_PERIOD = 300


class PolymarketClient:
    """Client for Polymarket CLOB API, specialized for 5-min BTC Up/Down markets."""

    def __init__(self, private_key: str, funder_address: str, signature_type: int = 2):
        """
        Args:
            private_key: Wallet private key (hex, with or without 0x prefix)
            funder_address: Funder/proxy wallet address
            signature_type: Signature type (0, 1, or 2). Default: 2
        """
        self._private_key = private_key
        self._funder_address = funder_address
        self._signature_type = signature_type
        self._client = None  # ClobClient instance (lazy init)
        self._api_creds = None
        self._initialized = False
        self._last_traded_slot: Optional[int] = None  # Unix ts of last traded slot
        self._http = httpx.AsyncClient(timeout=15)

    async def initialize(self) -> dict:
        """Initialize the CLOB client and derive API credentials.

        Returns:
            {success: bool, error: str|None}
        """
        try:
            from py_clob_client.client import ClobClient

            self._client = ClobClient(
                host=CLOB_HOST,
                key=self._private_key,
                chain_id=CHAIN_ID,
                signature_type=self._signature_type,
                funder=self._funder_address,
            )

            # Derive or retrieve API credentials
            self._api_creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(self._api_creds)

            self._initialized = True
            logger.info("Polymarket client initialized successfully")
            return {"success": True, "error": None}

        except Exception as e:
            logger.error(f"Polymarket client initialization failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def wallet_address(self) -> str:
        """Return the funder address (display purposes)."""
        return self._funder_address

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict:
        """Fetch USDC balance from Polymarket.

        Uses get_balance_allowance() with AssetType.COLLATERAL which is the
        correct method on ClobClient. Returns balance in wei, converted to USDC.

        Returns:
            {success: bool, data: {balance: float, allowance: float, currency: str}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            result = self._client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )

            # Balance is returned in wei (micro-USDC). Divide by 1e6 for USDC.
            balance_wei = int(result.get("balance", "0"))
            allowance_wei = int(result.get("allowance", "0"))
            bal_usdc = balance_wei / 1e6
            allowance_usdc = allowance_wei / 1e6

            logger.info(f"Polymarket balance: {bal_usdc:.2f} USDC")
            return {
                "success": True,
                "data": {
                    "balance": bal_usdc,
                    "allowance": allowance_usdc,
                    "currency": "USDC",
                },
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Market Discovery (Deterministic Slug-Based)
    # ------------------------------------------------------------------

    @staticmethod
    def get_current_slot_timestamp() -> int:
        """Get the Unix timestamp of the CURRENT 5-min slot (floor to 300s).

        Example: if now is 09:03:22 -> current slot is 09:00:00.
        This is the slot that is currently OPEN and accepting trades.
        """
        now_ts = int(time.time())
        return now_ts - (now_ts % SLOT_PERIOD)

    @staticmethod
    def get_next_slot_timestamp() -> int:
        """Get the Unix timestamp of the NEXT 5-min slot.

        Example: if now is 09:03:22 -> next slot is 09:05:00.
        This market already exists on Polymarket but hasn't started yet.
        """
        now_ts = int(time.time())
        return now_ts - (now_ts % SLOT_PERIOD) + SLOT_PERIOD

    @staticmethod
    def slot_to_datetime(slot_ts: int) -> datetime:
        """Convert a Unix slot timestamp to a UTC datetime."""
        return datetime.fromtimestamp(slot_ts, tz=timezone.utc)

    @staticmethod
    def _build_slug(slot_ts: int) -> str:
        """Build the deterministic Gamma API slug for a BTC 5-min market.

        Format: btc-updown-5m-{slot_timestamp}
        """
        return f"btc-updown-5m-{slot_ts}"

    async def _fetch_market_by_slug(self, slug: str) -> Optional[dict]:
        """Fetch a single market from Gamma API by its slug.

        Args:
            slug: Market slug e.g. 'btc-updown-5m-1774025100'

        Returns:
            Market dict from Gamma API, or None if not found.
        """
        try:
            resp = await self._http.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug},
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict) and data.get("slug"):
                return data
            return None

        except httpx.HTTPStatusError as e:
            logger.warning(f"Gamma API HTTP {e.response.status_code} for slug={slug}")
            return None
        except Exception as e:
            logger.warning(f"Gamma API error for slug={slug}: {e}")
            return None

    def _parse_market(self, market: dict, slot_ts: int) -> dict:
        """Parse a Gamma API market response into our internal format.

        Extracts condition_id, token IDs for Up/Down, outcome prices, etc.
        Maps outcomes to token IDs by matching outcome labels ("Up"/"Down").

        Args:
            market: Raw market dict from Gamma API.
            slot_ts: The slot timestamp this market belongs to.

        Returns:
            Parsed market data dict.
        """
        condition_id = market.get("conditionId", "")

        # clobTokenIds is a JSON string: '["id1", "id2"]'
        clob_raw = market.get("clobTokenIds", "[]")
        if isinstance(clob_raw, str):
            clob_token_ids = json.loads(clob_raw)
        else:
            clob_token_ids = clob_raw

        # outcomes is a JSON string: '["Up", "Down"]'
        outcomes_raw = market.get("outcomes", "[]")
        if isinstance(outcomes_raw, str):
            outcomes = json.loads(outcomes_raw)
        else:
            outcomes = outcomes_raw

        # outcomePrices is a JSON string: '["0.515", "0.485"]'
        prices_raw = market.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        # Map outcomes to token IDs by label
        # Polymarket BTC 5-min markets use ["Up", "Down"] consistently
        up_token_id = None
        down_token_id = None
        up_price = None
        down_price = None

        for i, outcome in enumerate(outcomes):
            label = outcome.strip().lower()
            if i < len(clob_token_ids):
                if label in ("up", "yes"):
                    up_token_id = clob_token_ids[i]
                    if i < len(prices):
                        up_price = float(prices[i])
                elif label in ("down", "no"):
                    down_token_id = clob_token_ids[i]
                    if i < len(prices):
                        down_price = float(prices[i])

        # Safety: if labels didn't match (shouldn't happen), refuse to guess
        if up_token_id is None or down_token_id is None:
            logger.error(
                f"Could not map outcomes to tokens! "
                f"outcomes={outcomes}, tokens={clob_token_ids}"
            )
            return None

        slot_dt = self.slot_to_datetime(slot_ts)
        neg_risk = market.get("negRisk", False)

        return {
            "condition_id": condition_id,
            "up_token_id": up_token_id,
            "down_token_id": down_token_id,
            "up_price": up_price,
            "down_price": down_price,
            "slot_ts": slot_ts,
            "slot_dt": slot_dt.isoformat(),
            "question": market.get("question", "N/A"),
            "outcomes": outcomes,
            "prices": [float(p) for p in prices] if prices else [],
            "market_slug": market.get("slug", ""),
            "enable_order_book": market.get("enableOrderBook", False),
            "neg_risk": neg_risk,
            "active": market.get("active", False),
            "closed": market.get("closed", False),
            "end_date": market.get("end_date_iso", market.get("endDate", market.get("end_date", ""))),
        }

    async def get_market_for_slot(self, target_slot_ts: int) -> dict:
        """Find the Polymarket 5-min BTC Up/Down market for a SPECIFIC slot.

        This is the slot-targeted version of get_current_market(). Instead of
        guessing which market to trade based on what's currently open, it looks
        up the exact market for the given slot timestamp.

        Args:
            target_slot_ts: Unix timestamp of the target slot (must be 300s-aligned).
                            Example: 1774025100 for the 16:45:00 UTC slot.

        Returns:
            {success: bool, data: {condition_id, up_token_id, down_token_id,
             up_price, down_price, slot_ts, slot_dt, question, outcomes,
             prices, neg_risk, ...}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        # Validate slot alignment
        if target_slot_ts % SLOT_PERIOD != 0:
            return {
                "success": False,
                "data": None,
                "error": (
                    f"target_slot_ts {target_slot_ts} is not aligned to "
                    f"{SLOT_PERIOD}s boundary"
                ),
            }

        try:
            slug = self._build_slug(target_slot_ts)
            slot_dt = self.slot_to_datetime(target_slot_ts)
            logger.info(
                f"Looking up target market: {slug} "
                f"(slot {slot_dt.strftime('%H:%M:%S')} UTC)"
            )

            market_raw = await self._fetch_market_by_slug(slug)
            if market_raw is None:
                return {
                    "success": False,
                    "data": None,
                    "error": (
                        f"No market found for target slot "
                        f"{slot_dt.strftime('%H:%M')} UTC (slug={slug}). "
                        f"Market may not exist yet."
                    ),
                }

            # Verify the market is not already closed/resolved
            is_closed = market_raw.get("closed", False)
            if is_closed:
                return {
                    "success": False,
                    "data": None,
                    "error": (
                        f"Target market {slug} is already closed. "
                        f"Slot {slot_dt.strftime('%H:%M')} UTC has ended."
                    ),
                }

            # Parse market data
            parsed = self._parse_market(market_raw, target_slot_ts)
            if parsed is None:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Failed to parse market data for {slug}",
                }

            logger.info(
                f"Target market found: {parsed['question']} | "
                f"slug={slug} | "
                f"Up={parsed['up_token_id'][:16]}... (${parsed['up_price']}) | "
                f"Down={parsed['down_token_id'][:16]}... (${parsed['down_price']})"
            )
            return {"success": True, "data": parsed, "error": None}

        except Exception as e:
            logger.error(f"Market discovery for slot {target_slot_ts} failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    async def get_current_market(self) -> dict:
        """Find the active Polymarket 5-min BTC Up/Down market to trade on.

        NOTE: This is the LEGACY fallback method. For signal-driven trades,
        use get_market_for_slot(target_slot_ts) instead, which guarantees
        the order lands on the correct slot.

        Strategy:
            1. Try the CURRENT slot (floor to 300s) — this is the open, tradeable market.
            2. If current slot market is closed/not found, try the NEXT slot.
            3. Look up by deterministic slug: btc-updown-5m-{slot_ts}

        Returns:
            {success: bool, data: {condition_id, up_token_id, down_token_id,
             up_price, down_price, slot_ts, slot_dt, question, outcomes,
             prices, neg_risk, ...}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        try:
            current_slot = self.get_current_slot_timestamp()
            next_slot = self.get_next_slot_timestamp()

            # Try current slot first (the one currently open for trading)
            for slot_ts in [current_slot, next_slot]:
                slug = self._build_slug(slot_ts)
                logger.info(f"Looking up market: {slug}")

                market_raw = await self._fetch_market_by_slug(slug)
                if market_raw is None:
                    logger.info(f"No market found for slug={slug}, trying next...")
                    continue

                # Check if market is actually tradeable
                is_closed = market_raw.get("closed", False)
                is_active = market_raw.get("active", True)
                enable_ob = market_raw.get("enableOrderBook", True)

                if is_closed:
                    logger.info(f"Market {slug} is closed, trying next...")
                    continue

                if not is_active:
                    logger.info(f"Market {slug} is not active, trying next...")
                    continue

                # Parse market data
                parsed = self._parse_market(market_raw, slot_ts)
                if parsed is None:
                    logger.error(f"Failed to parse market {slug}")
                    continue

                logger.info(
                    f"Market found: {parsed['question']} | "
                    f"slug={slug} | "
                    f"Up={parsed['up_token_id'][:16]}... (${parsed['up_price']}) | "
                    f"Down={parsed['down_token_id'][:16]}... (${parsed['down_price']})"
                )
                return {"success": True, "data": parsed, "error": None}

            # Both slots failed
            return {
                "success": False,
                "data": None,
                "error": (
                    f"No tradeable BTC 5-min market found. "
                    f"Tried slugs: {self._build_slug(current_slot)}, "
                    f"{self._build_slug(next_slot)}"
                ),
            }

        except Exception as e:
            logger.error(f"Market discovery failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Order Book / Pricing (used for logging/display only)
    # ------------------------------------------------------------------

    def get_best_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get the best available price from the CLOB order book.

        Used for logging and display purposes. FOK market orders handle
        pricing automatically via the SDK.

        Args:
            token_id: The CLOB token ID
            side: "BUY" or "SELL"

        Returns:
            Best price as float, or None if no orders available.
        """
        try:
            result = self._client.get_price(token_id, side)
            price = result.get("price")
            if price is not None:
                return float(price)
            return None
        except Exception as e:
            logger.error(f"Price fetch failed for token={token_id[:16]}...: {e}")
            return None

    # ------------------------------------------------------------------
    # Trade Execution (FOK Market Orders)
    # ------------------------------------------------------------------

    async def place_trade(
        self,
        direction: str,
        amount: float,
        target_slot_ts: Optional[int] = None,
    ) -> dict:
        """Place a FOK (Fill-or-Kill) market buy order on the correct Up/Down token.

        Uses MarketOrderArgs with OrderType.FOK for immediate execution at
        the best available market price. The SDK handles:
        - Price calculation from the order book
        - Tick size resolution
        - Neg risk detection
        - Fee rate resolution
        - Order signing

        For BUY orders, amount = USDC to spend (e.g. 1.0 = $1.00 USDC).
        The order either fills entirely and immediately, or is rejected.

        When target_slot_ts is provided (the expected path for signal-driven
        trades), the order is placed on that EXACT slot's market. This prevents
        the bug where a signal for 16:45-16:50 accidentally trades on the
        16:40-16:45 market that's still active.

        When target_slot_ts is None (legacy fallback), falls back to
        get_current_market() which guesses based on what's currently open.

        Args:
            direction: "UP" or "DOWN"
            amount: Trade size in USDC
            target_slot_ts: Unix timestamp of the target slot (300s-aligned).
                            When provided, the order is placed on this exact
                            slot's market. When None, falls back to auto-discovery.

        Returns:
            {success: bool, data: {order_id, direction, amount, price,
             size, token_id, slot_ts, slot_dt, question, filled_at}, error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        if direction not in ("UP", "DOWN"):
            return {"success": False, "data": None, "error": f"Invalid direction: {direction}"}

        try:
            from py_clob_client.clob_types import (
                MarketOrderArgs,
                OrderType,
                PartialCreateOrderOptions,
            )
            from py_clob_client.order_builder.constants import BUY

            # Step 1: Discover the market
            # Use slot-targeted lookup when target_slot_ts is provided (normal path).
            # Fall back to auto-discovery only when no target slot is specified.
            if target_slot_ts is not None:
                slot_dt_str = self.slot_to_datetime(target_slot_ts).strftime('%H:%M:%S UTC')
                logger.info(f"Using slot-targeted market discovery for slot {slot_dt_str}")
                market_result = await self.get_market_for_slot(target_slot_ts)
            else:
                logger.warning(
                    "No target_slot_ts provided — using legacy auto-discovery. "
                    "This may trade the wrong slot!"
                )
                market_result = await self.get_current_market()

            if not market_result["success"]:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Market discovery failed: {market_result['error']}",
                }

            market = market_result["data"]
            slot_ts = market["slot_ts"]

            # Step 2: Verify we got the right slot (safety check)
            if target_slot_ts is not None and slot_ts != target_slot_ts:
                return {
                    "success": False,
                    "data": None,
                    "error": (
                        f"Slot mismatch! Requested slot {target_slot_ts} "
                        f"but got market for slot {slot_ts}. Trade aborted."
                    ),
                }

            # Step 3: Duplicate trade prevention (keyed to the TARGET slot)
            if self._last_traded_slot == slot_ts:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Duplicate trade prevented: already traded slot {market['slot_dt']}",
                }

            # Step 4: Pick the correct token based on signal direction
            if direction == "UP":
                token_id = market["up_token_id"]
                gamma_price = market["up_price"]
            else:
                token_id = market["down_token_id"]
                gamma_price = market["down_price"]

            if not token_id:
                return {
                    "success": False,
                    "data": None,
                    "error": f"No token ID found for direction={direction}",
                }

            # Step 5: Get display price for logging (not used for order)
            display_price = self.get_best_price(token_id, side="BUY")
            if display_price is None:
                display_price = gamma_price if gamma_price and gamma_price > 0 else 0.50

            neg_risk = market.get("neg_risk", False)

            logger.info(
                f"Placing FOK market {direction} order: "
                f"token={token_id[:16]}... | "
                f"amount={amount} USDC | "
                f"display_price={display_price} | "
                f"slot={market['market_slug']} | "
                f"neg_risk={neg_risk}"
            )

            # Step 6: Build FOK market order
            # MarketOrderArgs.amount = USDC to spend for BUY orders.
            # price=0 tells the SDK to auto-calculate the optimal price
            # from the order book via calculate_market_price().
            market_order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=BUY,
                price=0,  # SDK auto-calculates from order book
                order_type=OrderType.FOK,
            )

            options = PartialCreateOrderOptions(
                neg_risk=neg_risk,
            )

            # Step 7: Sign the order locally
            signed_order = self._client.create_market_order(
                market_order_args, options
            )

            # Step 8: Post the signed order with FOK type
            order_resp = self._client.post_order(
                signed_order, orderType=OrderType.FOK
            )

            # Mark slot as traded
            self._last_traded_slot = slot_ts

            # Parse order response
            order_id = "unknown"
            status = "MATCHED"
            if isinstance(order_resp, dict):
                order_id = order_resp.get("orderID", order_resp.get("id", "unknown"))
                status = order_resp.get("status", "MATCHED")
            elif hasattr(order_resp, "orderID"):
                order_id = order_resp.orderID
            elif hasattr(order_resp, "id"):
                order_id = order_resp.id

            # Calculate effective size for display
            # For FOK market buys, size ≈ amount / price
            effective_price = display_price if display_price > 0 else 0.50
            effective_size = round(amount / effective_price, 2)

            trade_data = {
                "order_id": str(order_id),
                "direction": direction,
                "amount": round(amount, 2),  # USDC spent
                "price": effective_price,
                "size": effective_size,
                "token_id": token_id,
                "slot_ts": slot_ts,
                "slot_dt": market["slot_dt"],
                "question": market["question"],
                "market_slug": market["market_slug"],
                "status": status,
                "order_type": "FOK",
                "filled_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"FOK trade placed: {direction} | "
                f"order={order_id} | "
                f"amount={amount} USDC | "
                f"~price={effective_price} | "
                f"~size={effective_size} shares | "
                f"slot={market['market_slug']}"
            )
            return {"success": True, "data": trade_data, "error": None}

        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Open Positions
    # ------------------------------------------------------------------

    async def get_open_positions(self) -> dict:
        """Fetch open positions from the Polymarket Data API.

        Returns:
            {success: bool, data: list[dict], error: str|None}
        """
        if not self._initialized:
            return {"success": False, "data": None, "error": "Client not initialized"}

        try:
            resp = await self._http.get(
                f"{DATA_API}/positions",
                params={"user": self._funder_address},
            )
            resp.raise_for_status()
            positions = resp.json()

            # Normalize positions data
            formatted = []
            if isinstance(positions, list):
                for pos in positions:
                    formatted.append({
                        "market": pos.get("title", pos.get("question", "Unknown")),
                        "outcome": pos.get("outcome", "N/A"),
                        "size": float(pos.get("size", 0)),
                        "avg_price": float(pos.get("avgPrice", pos.get("price", 0))),
                        "current_value": float(pos.get("currentValue", pos.get("value", 0))),
                        "pnl": float(pos.get("cashPnl", pos.get("pnl", pos.get("realizedPnl", 0)))),
                        "token_id": pos.get("asset", pos.get("tokenId", "")),
                    })

            logger.info(f"Fetched {len(formatted)} open positions")
            return {"success": True, "data": formatted, "error": None}

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    async def is_connected(self) -> dict:
        """Check Polymarket connection health.

        Uses ClobClient.get_ok() for API health + get_balance_allowance() for auth check.

        Returns:
            {connected: bool, balance: float|None, error: str|None}
        """
        if not self._initialized:
            return {"connected": False, "balance": None, "error": "Client not initialized"}

        try:
            # Check CLOB API health using the SDK method
            api_ok = False
            try:
                ok_resp = self._client.get_ok()
                api_ok = ok_resp == "OK" or ok_resp is not None
            except Exception:
                # Fallback to HTTP check
                resp = await self._http.get(f"{CLOB_HOST}/")
                api_ok = resp.status_code == 200

            # Check balance (also validates L2 auth works)
            bal_result = await self.get_balance()
            balance = bal_result["data"]["balance"] if bal_result["success"] else None

            connected = api_ok and bal_result["success"]
            return {
                "connected": connected,
                "balance": balance,
                "error": None if connected else "API or balance check failed",
            }

        except Exception as e:
            return {"connected": False, "balance": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Close HTTP client."""
        await self._http.aclose()
