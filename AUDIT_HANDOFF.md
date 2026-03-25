# AprilXG7 — Codebase Audit & Bug Fix Plan

> **Purpose:** This document is a complete handoff for an AI agent (or developer) to implement all identified bug fixes without needing to re-audit the codebase. Every bug includes the exact file, line-level context, root cause, and step-by-step fix instructions.
>
> **Generated:** 2025-03-25 | **Audited by:** AI deep audit (all 17 source files, 4 test files, config, and plan files read in full)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Map & Responsibilities](#file-map--responsibilities)
3. [Bug Registry (6 Bugs)](#bug-registry)
   - [BUG-1: TradeManager state not persisted (P1)](#bug-1-trademanager-state-not-persisted-p1)
   - [BUG-2: Hardcoded payout constants in formatters (P2)](#bug-2-hardcoded-payout-constants-in-formatters-p2)
   - [BUG-3: V5_RECENT_WEIGHT is a silent no-op (P2)](#bug-3-v5_recent_weight-is-a-silent-no-op-p2)
   - [BUG-4: V5 ensemble lacks interactive retrain (P3)](#bug-4-v5-ensemble-lacks-interactive-retrain-p3)
   - [BUG-5: calibration_v2.py is dead code (P3)](#bug-5-calibration_v2py-is-dead-code-p3)
   - [BUG-6: get_prediction_model() bridge is dead code (P3)](#bug-6-get_prediction_model-bridge-is-dead-code-p3)
4. [.env Configuration Cleanup](#env-configuration-cleanup)
5. [Implementation Order](#implementation-order)
6. [Testing Checklist](#testing-checklist)
7. [Detailed Architecture Notes](#detailed-architecture-notes)

---

## Architecture Overview

AprilXG7 is a **BTC 5-minute binary options signal bot** for Polymarket. It predicts whether BTC will go Up or Down in the next 5-minute candle, optionally auto-trades on Polymarket, and reports signals/results via Telegram.

The codebase has **two parallel prediction systems**, controlled by `USE_V5_ENSEMBLE=true` in `.env`:

| System | Files | Model | Status |
|--------|-------|-------|--------|
| **V4** (legacy) | `model.py`, `features.py` | Single XGBoost + isotonic calibration | Fallback only |
| **V5** (primary) | `ensemble.py`, `features_v2.py`, `regime.py`, `trade_manager.py` | 3-model ensemble (XGB/LGBM/CatBoost) with regime-aware calibration | Active |

**Signal flow:**
```
main.py → bot.py (orchestrator)
  → data_fetcher.py        Fetches MEXC API klines (5m candles)
  → features_v2.py (V5)    Computes 76 features in 8 groups
  → ensemble.py (V5)       3 sub-models predict, weighted average
  → trade_manager.py       3-tier confidence check + session risk
  → auto_trader.py         Places Polymarket FOK orders
  → telegram_bot.py        Reports signal + resolution via Telegram
  → signal_tracker.py      Persists win/loss history to JSON
```

---

## File Map & Responsibilities

```
aprilxg7/
├── main.py                    Entry point: asyncio.run(bot.run())
├── src/
│   ├── __init__.py
│   ├── config.py              Centralized config from env vars
│   │                          Classes: BotConfig, MEXCConfig, ModelConfig,
│   │                          TelegramConfig, PolymarketConfig, EnsembleConfig
│   │                          Key: EnsembleConfig.sample_weight_recent_multiplier
│   │                          (reads V5_RECENT_WEIGHT but value is UNUSED)
│   │
│   ├── bot.py                 Main orchestrator (48K chars)
│   │                          Imports BOTH V4 and V5 systems
│   │                          use_v5 flag from config.ensemble.enabled
│   │                          Handles: training, prediction, resolution,
│   │                          auto-trading, Telegram commands
│   │
│   ├── data_fetcher.py        MEXC REST API for kline data
│   │                          Returns list of OHLCV candles
│   │
│   ├── features.py            V4 feature engineering (~30 indicators)
│   ├── features_v2.py         V5 feature engineering (76 features, 8 groups)
│   │                          Groups: price, volume, momentum, volatility,
│   │                          microstructure, regime, interaction, meta
│   │
│   ├── model.py               V4 single XGBoost model
│   │                          Has train_for_comparison()/apply()/reject()
│   │                          for interactive retrain
│   │                          DEAD CODE at bottom: get_prediction_model()
│   │
│   ├── ensemble.py            V5 ensemble model (XGB + LGBM + CatBoost)
│   │                          Inline calibration (isotonic/platt per regime)
│   │                          Feature pruning (top N by importance)
│   │                          train() directly replaces model (NO interactive)
│   │                          MISSING: sample_weight_recent_multiplier usage
│   │
│   ├── regime.py              Market regime detection
│   │                          4 regimes: TRENDING_UP, TRENDING_DOWN,
│   │                          RANGING, VOLATILE
│   │
│   ├── trade_manager.py       Session risk management
│   │                          3-tier confidence: TIER1(high), TIER2(med), TIER3(low)
│   │                          3 risk modes: NORMAL, CAUTIOUS, DEFENSIVE
│   │                          Rolling results tracking (wins/losses)
│   │                          BUG: ALL STATE IS IN-MEMORY ONLY
│   │
│   ├── calibration_v2.py      DEAD CODE — never imported anywhere
│   │                          Was planned per implementation docs but
│   │                          ensemble.py has its own inline calibration
│   │
│   ├── signal_tracker.py      Win/loss JSON persistence
│   │                          Saves to data/signal_history.json
│   │
│   ├── auto_trader.py         Polymarket order placement
│   │                          Uses PolymarketClient for FOK orders
│   │
│   ├── polymarket_client.py   CLOB API client (31K chars)
│   │                          Auth (HMAC-SHA256), market discovery,
│   │                          FOK orders, duplicate prevention,
│   │                          slot-targeted trading
│   │
│   ├── position_redeemer.py   On-chain CTF position redemption (Polygon)
│   │
│   ├── telegram_bot.py        Telegram command handlers
│   │                          /stats, /status, /autotrade, /retrain, etc.
│   │
│   └── formatters.py          Telegram HTML message formatting
│                               HARDCODED: WIN_PAYOUT=0.96, LOSS_PAYOUT=1.00,
│                               TRADE_COST=1.00 at module level
│
├── tests/
│   ├── test_ensemble.py       V5 ensemble model tests
│   ├── test_features_v2.py    V5 feature engineering tests
│   ├── test_integration.py    End-to-end V5 pipeline test
│   └── conftest.py            Shared test fixtures
│
├── docs/
│   ├── V5_IMPLEMENTATION_PLAN.md
│   └── V5_PROGRESS.md
│
├── .env.example               Environment variable template
├── requirements.txt
├── Dockerfile
├── railway.toml               Railway deployment config
└── Procfile
```

---

## Bug Registry

### BUG-1: TradeManager state not persisted (P1)

**Severity:** P1 — CRITICAL (affects live trading safety)  
**File:** `src/trade_manager.py`  
**Impact:** A Railway redeploy or crash resets all session state to defaults. If the bot was in DEFENSIVE mode (reduced trading after losses), it immediately returns to NORMAL mode and resumes aggressive trading.

**Root Cause:**  
`TradeManager` stores all state in instance variables:
- `self.session_results` — list of recent trade results (win/loss)
- `self.current_risk_mode` — NORMAL / CAUTIOUS / DEFENSIVE
- `self.consecutive_losses` — loss streak counter
- `self.session_start_time` — when current session began

None of this is written to disk. Compare with `signal_tracker.py` which correctly persists to `data/signal_history.json`.

**Fix Plan:**

1. Add a `STATE_FILE` path constant (e.g., `data/trade_manager_state.json`)
2. Create `_save_state()` method that writes current state to JSON:
   ```python
   def _save_state(self):
       state = {
           "session_results": [(r.value if hasattr(r, 'value') else r) for r in self.session_results],
           "current_risk_mode": self.current_risk_mode.value,  # or .name
           "consecutive_losses": self.consecutive_losses,
           "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
           "saved_at": datetime.utcnow().isoformat()
       }
       Path(self.STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
       with open(self.STATE_FILE, 'w') as f:
           json.dump(state, f, indent=2)
   ```
3. Create `_load_state()` method called in `__init__`:
   ```python
   def _load_state(self):
       if not Path(self.STATE_FILE).exists():
           return
       try:
           with open(self.STATE_FILE) as f:
               state = json.load(f)
           self.session_results = state.get("session_results", [])
           self.current_risk_mode = RiskMode(state["current_risk_mode"])
           self.consecutive_losses = state.get("consecutive_losses", 0)
           # ... restore other fields
       except Exception as e:
           logger.warning(f"Could not restore trade manager state: {e}")
   ```
4. Call `_save_state()` at the end of every method that modifies state:
   - `record_result()` — after appending to session_results
   - `_update_risk_mode()` — after changing risk mode
   - Any method that resets state
5. Add a staleness check: if `saved_at` is >24h old, reset to fresh state (old session data is meaningless)

**Testing:**
- Create state file, instantiate TradeManager, verify state loads
- Record results, kill process, re-instantiate, verify state persists
- Verify stale state (>24h) is discarded

**Estimated effort:** 1-2 hours

---

### BUG-2: Hardcoded payout constants in formatters (P2)

**Severity:** P2 — causes incorrect PnL display in Telegram  
**File:** `src/formatters.py`  
**Impact:** If trade size or payout rate changes in config, Telegram messages will show wrong PnL numbers.

**Root Cause:**  
At the top of `formatters.py`, constants are hardcoded:
```python
WIN_PAYOUT = 0.96
LOSS_PAYOUT = 1.00
TRADE_COST = 1.00
```
Meanwhile, `config.py` has configurable fields for these values. The formatter never reads from config.

**Fix Plan:**

1. Remove the hardcoded constants from the top of `formatters.py`
2. Option A (simple): Import config values
   ```python
   # At top of formatters.py
   from .config import Config
   # Then in functions that use these values:
   config = Config.from_env()
   ```
3. Option B (better, no global state): Pass values as parameters to formatting functions that calculate PnL. Find every function that references `WIN_PAYOUT`, `LOSS_PAYOUT`, or `TRADE_COST` and add them as parameters with defaults matching current values:
   ```python
   def format_resolution_message(
       signal, result, 
       win_payout=0.96, loss_payout=1.00, trade_cost=1.00
   ):
   ```
4. Update callers in `bot.py` and `telegram_bot.py` to pass config values when calling formatters

**Testing:**
- Change payout values in config, verify Telegram messages reflect new values
- Verify default values still work if no config is passed

**Estimated effort:** 30 minutes

---

### BUG-3: V5_RECENT_WEIGHT is a silent no-op (P2)

**Severity:** P2 — user thinks they're tuning a parameter that does nothing  
**File:** `src/config.py` (defined) + `src/ensemble.py` (should use, doesn't)  
**Impact:** Setting `V5_RECENT_WEIGHT=3.0` in `.env` has zero effect. The config field `EnsembleConfig.sample_weight_recent_multiplier` is read from env but never referenced in `ensemble.py`'s training logic.

**Root Cause:**  
In `config.py`, `EnsembleConfig` has:
```python
self.sample_weight_recent_multiplier = float(os.getenv('V5_RECENT_WEIGHT', '1.5'))
```
But in `ensemble.py`, the `train()` method never accesses `self.config.sample_weight_recent_multiplier` to create sample weights.

**Fix Plan — Choose one:**

**Option A: Remove it** (if recency weighting isn't wanted)
1. Remove `sample_weight_recent_multiplier` from `EnsembleConfig` in `config.py`
2. Remove `V5_RECENT_WEIGHT` from `.env.example`
3. Done

**Option B: Implement it** (if recency weighting IS wanted)
1. In `ensemble.py`'s `train()` method, after preparing X_train/y_train:
   ```python
   # Create sample weights that increase for more recent samples
   n = len(X_train)
   multiplier = self.config.sample_weight_recent_multiplier
   if multiplier != 1.0:
       # Linear ramp from 1.0 (oldest) to multiplier (newest)
       sample_weights = np.linspace(1.0, multiplier, n)
   else:
       sample_weights = None
   ```
2. Pass `sample_weight=sample_weights` to each sub-model's `.fit()` call
3. XGBoost, LightGBM, and CatBoost all support `sample_weight` parameter
4. Add a test verifying weights are applied when multiplier > 1.0

**Estimated effort:** 10 min (remove) or 30 min (implement)

---

### BUG-4: V5 ensemble lacks interactive retrain (P3)

**Severity:** P3 — feature gap, not a crash bug  
**File:** `src/ensemble.py` vs `src/model.py`  
**Impact:** V4's `PredictionModel` has `train_for_comparison()` / `apply()` / `reject()` for safe A/B model swaps via Telegram. V5's `EnsembleModel` only has `train()` which directly replaces the current model. The `/retrain`, `/apply`, `/reject` Telegram commands likely error or silently fail in V5 mode.

**Root Cause:**  
`ensemble.py`'s `EnsembleModel` class has:
- `train(candles)` — trains and directly replaces `self.models`, `self.calibrators`, etc.

It does NOT have:
- `train_for_comparison()` — train a candidate without replacing current
- `apply()` — swap candidate into production
- `reject()` — discard candidate

Meanwhile `bot.py` has retrain command handlers that likely call these methods.

**Fix Plan:**

1. Add candidate storage to `EnsembleModel.__init__`:
   ```python
   self._candidate_models = None
   self._candidate_calibrators = None
   self._candidate_metrics = None
   ```
2. Add `train_for_comparison(candles)` method:
   - Same logic as `train()` but stores results in `self._candidate_*` instead of `self.models`
   - Returns comparison metrics (candidate vs current OOS accuracy)
3. Add `apply()` method:
   ```python
   def apply(self):
       if self._candidate_models is None:
           raise ValueError("No candidate model to apply")
       self.models = self._candidate_models
       self.calibrators = self._candidate_calibrators
       self._candidate_models = None
       # ... etc
   ```
4. Add `reject()` method:
   ```python
   def reject(self):
       self._candidate_models = None
       self._candidate_calibrators = None
       self._candidate_metrics = None
   ```
5. Verify `bot.py`'s retrain command handlers work with these new methods
6. Add tests for the candidate lifecycle

**Estimated effort:** 1-2 hours

---

### BUG-5: calibration_v2.py is dead code (P3)

**Severity:** P3 — no runtime impact, but causes confusion  
**File:** `src/calibration_v2.py`  
**Impact:** File exists per the V5 implementation plan but is never imported by any other file. `ensemble.py` has its own inline calibration logic instead.

**Root Cause:**  
The V5 implementation plan called for a standalone calibrator module. During implementation, the calibration was built directly into `ensemble.py` instead. The standalone file was never wired up.

**Fix Plan — Choose one:**

**Option A: Delete it**
1. `git rm src/calibration_v2.py`
2. Update `docs/V5_PROGRESS.md` to note calibration is inline in `ensemble.py`

**Option B: Refactor ensemble.py to use it**
1. Extract calibration logic from `ensemble.py` into `calibration_v2.py`
2. Import and use it from `ensemble.py`
3. This is cleaner but more work

**Recommendation:** Option A (delete). The inline calibration in `ensemble.py` works fine and is already tested.

**Estimated effort:** 5 minutes (delete) or 1 hour (refactor)

---

### BUG-6: get_prediction_model() bridge is dead code (P3)

**Severity:** P3 — no runtime impact  
**File:** `src/model.py` (at the bottom of the file)  
**Impact:** A factory function `get_prediction_model()` exists at the bottom of `model.py` that was intended to bridge V4/V5 model selection. It is never called by `bot.py` or any other file — `bot.py` handles the V4/V5 switch with its own `use_v5` flag.

**Fix Plan:**
1. Delete the `get_prediction_model()` function from the bottom of `model.py`
2. Remove any related imports if present

**Estimated effort:** 5 minutes

---

## .env Configuration Cleanup

The current `.env.example` mixes V4 and V5 settings without clear boundaries. When `USE_V5_ENSEMBLE=true`, many V4 settings are silently ignored.

**Settings IGNORED when V5 is active:**
- `PREDICTION_THRESHOLD` (V4 only)
- `CONFIDENCE_MIN` (V4 only)
- `EV_THRESHOLD` (V4 only)
- `ENABLE_CALIBRATION` (V4 only — V5 always calibrates)
- `ENABLE_FEATURE_PRUNING` (V4 only — V5 has own pruning via `V5_PRUNE_TOP_N`)

**Confusing overlapping names:**
- `TRAIN_CANDLES` vs `V5_TRAIN_CANDLES`
- `RETRAIN_INTERVAL_HOURS` vs `V5_RETRAIN_HOURS`

**Fix Plan:**
Reorganize `.env.example` with clear sections:
```env
# ============================================================
# SYSTEM SELECTION
# ============================================================
USE_V5_ENSEMBLE=true       # true = V5 ensemble, false = V4 single model

# ============================================================
# SHARED SETTINGS (used by both V4 and V5)
# ============================================================
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
MEXC_API_KEY=...
# ... etc

# ============================================================
# V5 ENSEMBLE SETTINGS (only used when USE_V5_ENSEMBLE=true)
# ============================================================
V5_TRAIN_CANDLES=2000
V5_RETRAIN_HOURS=6
V5_TIER1_THRESHOLD=0.62
# ... etc

# ============================================================
# V4 LEGACY SETTINGS (only used when USE_V5_ENSEMBLE=false)
# ============================================================
PREDICTION_THRESHOLD=0.55
CONFIDENCE_MIN=0.58
# ... etc
```

---

## Implementation Order

Recommended order — fix the most dangerous bugs first, keep each commit atomic:

| Step | Bug | Commit Message | Effort |
|------|-----|---------------|--------|
| 1 | BUG-1 | `fix: persist TradeManager state to survive restarts` | 1-2h |
| 2 | BUG-2 | `fix: pass payout config to formatters instead of hardcoding` | 30m |
| 3 | BUG-3 | `fix: remove unused V5_RECENT_WEIGHT config (or implement it)` | 10-30m |
| 4 | BUG-5 | `chore: remove dead calibration_v2.py module` | 5m |
| 5 | BUG-6 | `chore: remove dead get_prediction_model() bridge function` | 5m |
| 6 | BUG-4 | `feat: add interactive retrain support to V5 ensemble` | 1-2h |
| 7 | .env | `docs: reorganize .env.example with V4/V5 sections` | 15m |

**Total estimated effort: 3-5 hours**

---

## Testing Checklist

After implementing fixes, verify:

- [ ] **BUG-1:** Start bot → make trades → kill process → restart → verify risk mode persisted
- [ ] **BUG-1:** Verify stale state (>24h old) is discarded on restart
- [ ] **BUG-2:** Change payout values in .env → verify Telegram shows correct PnL
- [ ] **BUG-3:** If removed: verify no references to V5_RECENT_WEIGHT remain. If implemented: verify sample weights change training behavior
- [ ] **BUG-4:** Run `/retrain` in Telegram → see candidate metrics → `/apply` or `/reject` works
- [ ] **BUG-5:** Verify no import errors after deleting calibration_v2.py
- [ ] **BUG-6:** Verify no import errors after deleting get_prediction_model()
- [ ] **All existing tests pass:** `pytest tests/`
- [ ] **.env.example:** Verify all env vars are documented in correct sections

---

## Detailed Architecture Notes

### V5 Ensemble Training Flow (ensemble.py)
1. Receive candles from data_fetcher
2. FeatureEngineV2 computes 76 features
3. 3-way data split: TRAIN | CALIBRATION | OUT-OF-SAMPLE (with purge gaps)
4. Train 3 sub-models (XGBoost, LightGBM, CatBoost) on TRAIN split
5. Calibrate each model per-regime on CALIBRATION split (isotonic or platt)
6. Evaluate on OOS split
7. Feature pruning: keep top N features by importance, retrain
8. Store models, calibrators, feature list, and regime detector

### V5 Prediction Flow
1. FeatureEngineV2 computes features for latest candles
2. Regime detector identifies current market regime
3. Each sub-model predicts, calibrated by regime-specific calibrator
4. Weighted average of calibrated probabilities
5. TradeManager applies 3-tier confidence check:
   - TIER1 (≥0.62): Full-size trade
   - TIER2 (0.55-0.62): Reduced size
   - TIER3 (<0.55): Skip or signal-only
6. Risk mode (NORMAL/CAUTIOUS/DEFENSIVE) further gates trades

### Polymarket Integration (polymarket_client.py)
- Discovers BTC 5-min binary markets by slug pattern
- Places Fill-or-Kill (FOK) orders to avoid partial fills
- Handles: auth (HMAC-SHA256 with API key/secret/passphrase), market discovery, order placement, position tracking, duplicate order prevention
- Position redeemer (`position_redeemer.py`) handles on-chain CTF token redemption on Polygon

### Key Config Classes (config.py)
- `BotConfig`: candle interval, timezone, data dir
- `MEXCConfig`: API credentials for price data
- `ModelConfig`: V4 model params (threshold, calibration, pruning)
- `EnsembleConfig`: V5 params (train_candles, retrain_hours, tier thresholds, sub-model weights)
- `TelegramConfig`: bot token, chat ID
- `PolymarketConfig`: private key, funder address, signature type, API credentials
- All loaded via `Config.from_env()` class method

### Test Coverage
- `tests/test_ensemble.py` — V5 model training, prediction, calibration
- `tests/test_features_v2.py` — Feature computation, group validation
- `tests/test_integration.py` — End-to-end V5 pipeline (fetch → features → train → predict)
- `tests/conftest.py` — Shared fixtures (sample candle data)
- **Not tested:** TradeManager, formatters, Polymarket client, Telegram bot, auto_trader

---

## What's Working Well (Don't Touch)

- 76-feature V5 engine with proper normalization, no data leakage
- Regime-aware calibration (isotonic/platt per market regime)
- 3-tier confidence + session risk management design
- Slot-targeted Polymarket trading with deterministic slug discovery
- Solid test suite for V5 components
- polymarket_client.py (31K chars, handles auth, FOK orders, duplicate prevention)
- Position redeemer for on-chain settlement
- signal_tracker.py correctly persists to JSON (use as reference pattern for BUG-1)

---

*End of handoff document. All information needed to implement fixes is contained above.*
