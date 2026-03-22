"""Automated position redemption for resolved Polymarket markets.

Handles:
- Scanning for redeemable positions via Data API
- On-chain redemption via CTF contract (non-negRisk) or NegRisk Adapter (negRisk)
- Safe/proxy wallet transaction execution (signature_type=2)
- Tracks redeemed condition IDs to avoid duplicate attempts

Redemption Flow:
    1. Query Data API for positions with redeemable=true
    2. For each redeemable position:
       a. Determine if negRisk or standard market
       b. Build redeemPositions calldata
       c. If Safe wallet (sig_type=2): wrap in execTransaction
       d. Send transaction on Polygon
    3. Report results via Telegram

Contract Addresses (Polygon Mainnet):
    CTF (Conditional Tokens): 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
    USDC.e:                   0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
    Neg Risk Adapter:         0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296
    Neg Risk Wrapped Col:     0x3A3BD7bb9528E159577F7C2e685CC81A765002E2

Requires:
    web3>=6.14.0 (already in requirements.txt)
    Environment: POLYGON_RPC_URL (defaults to public Polygon RPC)
"""
import asyncio
import logging
import time
from typing import Optional

import httpx
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract addresses (Polygon Mainnet)
# ---------------------------------------------------------------------------
CTF_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDC_E_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
NEG_RISK_ADAPTER_ADDRESS = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")
NEG_RISK_WRAPPED_COL_ADDRESS = Web3.to_checksum_address("0x3A3BD7bb9528E159577F7C2e685CC81A765002E2")

# Default public Polygon RPC (rate-limited but functional)
DEFAULT_POLYGON_RPC = "https://polygon-rpc.com"

# Data API
DATA_API = "https://data-api.polymarket.com"

# ---------------------------------------------------------------------------
# Minimal ABIs (only what we need)
# ---------------------------------------------------------------------------
CTF_REDEEM_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    }
]

NEG_RISK_REDEEM_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    }
]

SAFE_EXEC_ABI = [
    {
        "name": "execTransaction",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures", "type": "bytes"},
        ],
        "outputs": [{"name": "success", "type": "bool"}],
    }
]


class PositionRedeemer:
    """Scans for and redeems resolved Polymarket positions on-chain."""

    def __init__(
        self,
        private_key: str,
        funder_address: str,
        signature_type: int = 2,
        polygon_rpc_url: str = "",
    ):
        self._private_key = private_key
        self._funder_address = Web3.to_checksum_address(funder_address)
        self._signature_type = signature_type
        self._polygon_rpc_url = polygon_rpc_url or DEFAULT_POLYGON_RPC

        # Derive EOA address from private key
        self._account = Account.from_key(self._private_key)
        self._eoa_address = self._account.address

        # Web3 setup
        self._w3 = Web3(Web3.HTTPProvider(self._polygon_rpc_url))
        self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Contract instances
        self._ctf = self._w3.eth.contract(address=CTF_ADDRESS, abi=CTF_REDEEM_ABI)
        self._neg_risk_adapter = self._w3.eth.contract(
            address=NEG_RISK_ADAPTER_ADDRESS, abi=NEG_RISK_REDEEM_ABI
        )

        # Safe contract (only if signature_type == 2)
        self._safe_contract = None
        if self._signature_type == 2:
            self._safe_contract = self._w3.eth.contract(
                address=self._funder_address, abi=SAFE_EXEC_ABI
            )

        # HTTP client for Data API
        self._http = httpx.AsyncClient(timeout=15)

        # Track redeemed condition IDs to skip duplicates within session
        self._redeemed_conditions: set = set()

        # Stats
        self._total_redeemed: int = 0
        self._total_usdc_redeemed: float = 0.0
        self._last_scan_time: Optional[float] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> dict:
        """Verify RPC connection and wallet access.

        Returns:
            {success: bool, error: str|None, data: dict|None}
        """
        try:
            # Test RPC connection
            connected = self._w3.is_connected()
            if not connected:
                return {
                    "success": False,
                    "error": f"Cannot connect to Polygon RPC: {self._polygon_rpc_url}",
                    "data": None,
                }

            chain_id = self._w3.eth.chain_id
            if chain_id != 137:
                return {
                    "success": False,
                    "error": f"Wrong chain ID: {chain_id} (expected 137 for Polygon)",
                    "data": None,
                }

            # Check EOA balance for gas
            eoa_balance = self._w3.eth.get_balance(self._eoa_address)
            pol_balance = self._w3.from_wei(eoa_balance, "ether")

            self._initialized = True
            logger.info(
                f"PositionRedeemer initialized: EOA={self._eoa_address}, "
                f"Safe={self._funder_address}, POL={pol_balance:.4f}, "
                f"RPC={self._polygon_rpc_url}"
            )

            return {
                "success": True,
                "error": None,
                "data": {
                    "eoa_address": self._eoa_address,
                    "safe_address": self._funder_address,
                    "pol_balance": float(pol_balance),
                    "chain_id": chain_id,
                },
            }

        except Exception as e:
            logger.error(f"PositionRedeemer init failed: {e}")
            return {"success": False, "error": str(e), "data": None}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Scan for Redeemable Positions
    # ------------------------------------------------------------------

    async def get_redeemable_positions(self) -> dict:
        """Query Data API for positions that can be redeemed.

        Returns:
            {success: bool, data: list[dict], error: str|None}
        """
        try:
            url = f"{DATA_API}/positions"
            params = {
                "user": self._funder_address.lower(),
                "redeemable": "true",
                "sizeThreshold": "0.01",  # Skip dust positions
            }

            resp = await self._http.get(url, params=params)
            resp.raise_for_status()
            positions = resp.json()

            if not isinstance(positions, list):
                positions = []

            # Filter out already-redeemed conditions (session dedup)
            new_positions = [
                p for p in positions
                if p.get("conditionId") not in self._redeemed_conditions
            ]

            self._last_scan_time = time.time()
            logger.info(
                f"Redeemable scan: {len(positions)} total, "
                f"{len(new_positions)} new (after session dedup)"
            )

            return {"success": True, "data": new_positions, "error": None}

        except httpx.HTTPStatusError as e:
            err = f"Data API HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(err)
            return {"success": False, "data": [], "error": err}
        except Exception as e:
            logger.error(f"Failed to fetch redeemable positions: {e}")
            return {"success": False, "data": [], "error": str(e)}

    # ------------------------------------------------------------------
    # Build Redemption Transaction
    # ------------------------------------------------------------------

    def _build_redeem_calldata(
        self, condition_id: str, neg_risk: bool
    ) -> tuple:
        """Build the redeemPositions calldata.

        Args:
            condition_id: Hex string condition ID (0x-prefixed)
            neg_risk: Whether this is a neg-risk market

        Returns:
            (target_address, calldata_hex_string)
        """
        condition_bytes = bytes.fromhex(
            condition_id[2:] if condition_id.startswith("0x") else condition_id
        )

        # Index sets [1, 2] = redeem both Yes and No outcomes
        index_sets = [1, 2]

        if neg_risk:
            # NegRisk markets: call NegRiskAdapter.redeemPositions(conditionId, indexSets)
            calldata = self._neg_risk_adapter.encodeABI(
                fn_name="redeemPositions",
                args=[condition_bytes, index_sets],
            )
            target = NEG_RISK_ADAPTER_ADDRESS
        else:
            # Standard markets: call CTF.redeemPositions(collateral, parent, conditionId, indexSets)
            parent_collection = b"\x00" * 32
            calldata = self._ctf.encodeABI(
                fn_name="redeemPositions",
                args=[USDC_E_ADDRESS, parent_collection, condition_bytes, index_sets],
            )
            target = CTF_ADDRESS

        return target, calldata

    def _build_safe_signatures(self) -> bytes:
        """Build pre-approved hash signatures for single-owner Safe.

        For a Safe where the EOA is the sole owner, we use the
        pre-validated signature format:
            r = address of owner (padded to 32 bytes)
            s = 0 (32 bytes)
            v = 1

        Returns:
            65-byte signature
        """
        # r: owner address padded to 32 bytes
        r = self._eoa_address.lower().replace("0x", "")
        r_padded = r.zfill(64)

        # s: zero (32 bytes)
        s_padded = "00" * 32

        # v: 1 (pre-validated)
        v = "01"

        sig_hex = r_padded + s_padded + v
        return bytes.fromhex(sig_hex)

    def _get_eip1559_fees(self) -> dict:
        """Get EIP-1559 gas fee parameters for Polygon.

        Returns:
            dict with maxFeePerGas and maxPriorityFeePerGas
        """
        # Get the latest base fee from the pending block
        latest_block = self._w3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas", self._w3.eth.gas_price)

        # Priority fee (tip) — use eth_maxPriorityFeePerGas if available,
        # otherwise default to 30 gwei which is standard for Polygon
        try:
            max_priority_fee = self._w3.eth.max_priority_fee
        except Exception:
            max_priority_fee = self._w3.to_wei(30, "gwei")

        # Max fee = 2x base fee + priority fee (gives headroom for base fee spikes)
        max_fee_per_gas = (2 * base_fee) + max_priority_fee

        return {
            "maxFeePerGas": max_fee_per_gas,
            "maxPriorityFeePerGas": max_priority_fee,
        }

    def _build_safe_tx(
        self, target: str, calldata: str
    ) -> dict:
        """Wrap a call inside Safe's execTransaction.

        Args:
            target: Contract to call (CTF or NegRiskAdapter)
            calldata: Hex-encoded function call (0x-prefixed)

        Returns:
            Transaction dict ready for signing and sending
        """
        signatures = self._build_safe_signatures()
        zero_addr = Web3.to_checksum_address("0x" + "00" * 20)

        # Convert calldata hex string to bytes for the Safe contract call
        calldata_bytes = bytes.fromhex(
            calldata[2:] if calldata.startswith("0x") else calldata
        )

        # Encode the execTransaction call
        exec_calldata = self._safe_contract.encodeABI(
            fn_name="execTransaction",
            args=[
                Web3.to_checksum_address(target),  # to
                0,                                  # value
                calldata_bytes,                     # data
                0,                                  # operation (Call)
                0,                                  # safeTxGas
                0,                                  # baseGas
                0,                                  # gasPrice (Safe internal, not tx-level)
                zero_addr,                          # gasToken
                zero_addr,                          # refundReceiver
                signatures,                         # signatures
            ],
        )

        # Get EIP-1559 gas fees
        fees = self._get_eip1559_fees()

        tx_data = {
            "from": self._eoa_address,
            "to": self._funder_address,
            "data": exec_calldata,
            "chainId": 137,
            "gas": 500_000,  # Will be estimated in redeem_position()
            "nonce": self._w3.eth.get_transaction_count(self._eoa_address),
            "type": 2,  # EIP-1559 transaction
            **fees,
        }

        return tx_data

    def _build_direct_tx(
        self, target: str, calldata: str
    ) -> dict:
        """Build a direct transaction (non-Safe, signature_type != 2).

        Args:
            target: Contract to call
            calldata: Hex-encoded function call (0x-prefixed)

        Returns:
            Transaction dict
        """
        # Get EIP-1559 gas fees
        fees = self._get_eip1559_fees()

        tx_data = {
            "from": self._eoa_address,
            "to": Web3.to_checksum_address(target),
            "data": calldata,
            "chainId": 137,
            "gas": 500_000,  # Will be estimated in redeem_position()
            "nonce": self._w3.eth.get_transaction_count(self._eoa_address),
            "type": 2,  # EIP-1559 transaction
            **fees,
        }
        return tx_data

    # ------------------------------------------------------------------
    # Execute Redemption
    # ------------------------------------------------------------------

    async def redeem_position(self, position: dict) -> dict:
        """Redeem a single resolved position on-chain.

        Args:
            position: Position dict from Data API containing:
                - conditionId, size, outcome, negRisk, title, asset

        Returns:
            {success: bool, tx_hash: str|None, condition_id: str,
             title: str, size: float, error: str|None}
        """
        condition_id = position.get("conditionId", "")
        title = position.get("title", "Unknown Market")
        size = float(position.get("size", 0))
        neg_risk = position.get("curatedByPolymarket", False)  # Approximation

        # The Data API may return negRisk or proxyWallet hints
        # Check multiple fields for neg_risk determination
        if "negRisk" in position:
            neg_risk = bool(position["negRisk"])
        elif position.get("asset", "").lower() == NEG_RISK_WRAPPED_COL_ADDRESS.lower():
            neg_risk = True

        result_base = {
            "condition_id": condition_id,
            "title": title,
            "size": size,
            "neg_risk": neg_risk,
        }

        try:
            # Build calldata (returns hex string now, not bytes)
            target, calldata = self._build_redeem_calldata(condition_id, neg_risk)

            # Build transaction (Safe-wrapped or direct)
            if self._signature_type == 2 and self._safe_contract:
                tx = self._build_safe_tx(target, calldata)
            else:
                tx = self._build_direct_tx(target, calldata)

            # Estimate gas (replace placeholder)
            try:
                estimated_gas = self._w3.eth.estimate_gas(tx)
                tx["gas"] = int(estimated_gas * 1.3)  # 30% buffer
            except Exception as gas_err:
                logger.warning(
                    f"Gas estimation failed for {condition_id[:10]}..., "
                    f"using default 500k: {gas_err}"
                )
                tx["gas"] = 500_000

            # Sign and send
            signed = self._account.sign_transaction(tx)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(
                f"Redemption tx sent: {tx_hash_hex} | "
                f"condition={condition_id[:16]}... | "
                f"negRisk={neg_risk} | size={size}"
            )

            # Wait for receipt in a thread to avoid blocking
            receipt = await asyncio.to_thread(
                self._w3.eth.wait_for_transaction_receipt, tx_hash, timeout=120
            )

            if receipt["status"] == 1:
                # Mark as redeemed
                self._redeemed_conditions.add(condition_id)
                self._total_redeemed += 1
                self._total_usdc_redeemed += size

                logger.info(
                    f"Redemption SUCCESS: {tx_hash_hex} | "
                    f"gas_used={receipt['gasUsed']} | size={size}"
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash_hex,
                    "error": None,
                    **result_base,
                }
            else:
                logger.error(f"Redemption REVERTED: {tx_hash_hex}")
                return {
                    "success": False,
                    "tx_hash": tx_hash_hex,
                    "error": "Transaction reverted on-chain",
                    **result_base,
                }

        except Exception as e:
            logger.error(
                f"Redemption failed for {condition_id[:16]}...: {e}"
            )
            return {
                "success": False,
                "tx_hash": None,
                "error": str(e),
                **result_base,
            }

    # ------------------------------------------------------------------
    # Batch Redemption (main entry point)
    # ------------------------------------------------------------------

    async def redeem_all(self) -> dict:
        """Scan for and redeem all redeemable positions.

        This is the main entry point called by the bot's main loop
        and the /redeem Telegram command.

        Returns:
            {success: bool, redeemed: list[dict], skipped: int,
             errors: list[dict], total_usdc: float}
        """
        if not self._initialized:
            return {
                "success": False,
                "redeemed": [],
                "skipped": 0,
                "errors": [{"error": "Redeemer not initialized"}],
                "total_usdc": 0.0,
            }

        # Step 1: Scan for redeemable positions
        scan_result = await self.get_redeemable_positions()
        if not scan_result["success"]:
            return {
                "success": False,
                "redeemed": [],
                "skipped": 0,
                "errors": [{"error": scan_result["error"]}],
                "total_usdc": 0.0,
            }

        positions = scan_result["data"]
        if not positions:
            logger.debug("No redeemable positions found")
            return {
                "success": True,
                "redeemed": [],
                "skipped": 0,
                "errors": [],
                "total_usdc": 0.0,
            }

        logger.info(f"Found {len(positions)} redeemable position(s), processing...")

        # Step 2: Check MATIC balance for gas
        eoa_balance = self._w3.eth.get_balance(self._eoa_address)
        pol_balance = self._w3.from_wei(eoa_balance, "ether")
        if pol_balance < 0.005:  # Need at least ~0.005 POL for gas
            return {
                "success": False,
                "redeemed": [],
                "skipped": len(positions),
                "errors": [{
                    "error": f"Insufficient POL for gas: {pol_balance:.6f} POL"
                }],
                "total_usdc": 0.0,
            }

        # Step 3: Redeem each position sequentially
        redeemed = []
        errors = []
        total_usdc = 0.0

        for pos in positions:
            result = await self.redeem_position(pos)
            if result["success"]:
                redeemed.append(result)
                total_usdc += result.get("size", 0)
            else:
                errors.append(result)

            # Small delay between transactions to avoid nonce issues
            if len(positions) > 1:
                await asyncio.sleep(2)

        return {
            "success": len(redeemed) > 0 or (len(errors) == 0),
            "redeemed": redeemed,
            "skipped": 0,
            "errors": errors,
            "total_usdc": round(total_usdc, 6),
        }

    # ------------------------------------------------------------------
    # Status / Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get redemption statistics for the current session."""
        return {
            "total_redeemed": self._total_redeemed,
            "total_usdc": round(self._total_usdc_redeemed, 6),
            "redeemed_conditions": len(self._redeemed_conditions),
            "last_scan": self._last_scan_time,
            "initialized": self._initialized,
        }

    async def close(self):
        """Clean up HTTP client."""
        await self._http.aclose()
