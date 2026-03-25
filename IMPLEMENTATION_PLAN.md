# AprilXG V5 — Multi-Model Ensemble Implementation Plan

## MASTER STATUS: PHASE 2 OF 5 — COMPLETE

> **This file is the SINGLE SOURCE OF TRUTH for implementation.**
> Every AI agent session MUST read this file FIRST, check the status boxes,
> and work on the NEXT unchecked phase. Do NOT skip phases or reinterpret specs.
>
> ## CRITICAL: ONE PHASE PER SESSION — HARD STOP RULE
>
> **You MUST work on EXACTLY ONE phase per session. No exceptions.**
>
> - Find the first phase with unchecked `[ ]` items — that is YOUR phase.
> - Complete ALL items in that ONE phase.
> - After completing that phase: update checkboxes, write session log, commit, and STOP.
> - Do NOT start the next phase. Do NOT "get ahead". Do NOT even read the next phase's files.
> - If you finish early, use remaining time to VERIFY your work, not start new work.
> - The user has strict usage limits. Starting a second phase risks hitting the limit
>   mid-implementation, leaving broken incomplete code. This is WORSE than doing nothing.
>
> **What "STOP" means:** After your phase commit, your final message should be:
> *"Phase N complete. All validation criteria passed. Committed and pushed.
> Next session should work on Phase N+1: [brief description]."*
> Then STOP. Do not continue.
>
> After completing work, the agent MUST:
> 1. Check off completed items with `[x]`
> 2. Add a `### Session Log` entry at the bottom with what was done
> 3. Commit with message: `Phase N: <description>`
> 4. **STOP. Do not proceed to the next phase.**

---

## OBJECTIVE

**55%+ prediction accuracy on 5-minute BTC candle direction with 70+ trades per day.**

Current system: 51.7% accuracy, 3-10 trades/day, negative EV.
Target system: 55-58% accuracy, 70-100+ trades/day, positive EV.

---

## ARCHITECTURE OVERVIEW

```
Regime Detector (volatility + trend classification)
        │
        ├──▶ Momentum Model (XGBoost — trending regimes)
        ├──▶ Mean Reversion Model (LightGBM — ranging regimes)
        └──▶ Microstructure Model (CatBoost — all regimes)
                │
        Ensemble Aggregator (regime-weighted soft vote)
                │
        Regime-Conditional Calibration (per-regime Platt + Isotonic)
                │
        Tiered Trade Filter (3 confidence tiers → 70+ trades/day)
                │
        Session Risk Manager (rolling accuracy monitor)
```

### Files Changed/Created

| File | Action | Phase |
|------|--------|-------|
| `src/features_v2.py` | **NEW** — 70+ microstructure features | Phase 1 |
| `src/regime.py` | **NEW** — Regime detector | Phase 2 |
| `src/ensemble.py` | **NEW** — Multi-model ensemble + training | Phase 2 |
| `src/calibration_v2.py` | **NEW** — Regime-conditional calibration | Phase 3 |
| `src/trade_manager.py` | **NEW** — Tiered confidence + session risk | Phase 3 |
| `src/config.py` | **MODIFY** — Add V5 config dataclasses | Phase 4 |
| `src/bot.py` | **MODIFY** — Wire in ensemble pipeline | Phase 4 |
| `src/model.py` | **MODIFY** — Keep as fallback, add V5 bridge | Phase 4 |
| `src/formatters.py` | **MODIFY** — Ensemble signal formatting | Phase 4 |
| `src/data_fetcher.py` | **MODIFY** — Add trades endpoint for volume delta | Phase 1 |
| `requirements.txt` | **MODIFY** — Add lightgbm, catboost | Phase 1 |
| `.env.example` | **MODIFY** — Add V5 config vars | Phase 4 |
| `tests/test_features_v2.py` | **NEW** — Feature validation tests | Phase 5 |
| `tests/test_ensemble.py` | **NEW** — Ensemble + regime tests | Phase 5 |
| `tests/test_integration.py` | **NEW** — End-to-end pipeline test | Phase 5 |

### Key Constraint: Backward Compatibility
- All new code goes in NEW files until Phase 4 integration
- `features.py` and `model.py` remain untouched until Phase 4
- Phase 4 adds a `USE_V5_ENSEMBLE=true` env flag — when false, old pipeline runs
- Existing Telegram commands, Polymarket integration, signal tracker are NOT modified

---

## PHASE 1: Feature Engine V2 (`features_v2.py` + `data_fetcher.py` changes)

**Goal:** Build the new feature engine with 70+ features optimized for 5-minute prediction.
**Files:** `src/features_v2.py` (new), `src/data_fetcher.py` (modify), `requirements.txt` (modify)
**Estimated size:** ~500 lines

### Status
- [x] 1.1 Update `requirements.txt`
- [x] 1.2 Add trade data fetching to `data_fetcher.py`
- [x] 1.3 Create `src/features_v2.py` with all feature groups
- [x] 1.4 Verify features compile and produce no NaN on sample data
- [x] 1.5 Commit: `Phase 1: Feature Engine V2`

### 1.1 — Update `requirements.txt`

Add these lines (keep all existing):
```
# V5 Ensemble
lightgbm==4.6.0
catboost==1.2.7
```

### 1.2 — Add Trade Data Fetching to `data_fetcher.py`

Add this method to the `MEXCFetcher` class (append after `fetch_historical_multi_timeframe`):

```python
async def fetch_recent_trades(
    self,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch recent trades for volume delta / buy-sell imbalance.

    MEXC aggTrades endpoint returns recent aggregated trades with
    buy/sell side indicator.

    Returns:
        DataFrame with columns: id, price, qty, quoteQty, time, isBuyerMaker
    """
    await self._rate_limit()
    client = await self._get_client()

    params = {
        "symbol": self.config.symbol,
        "limit": min(limit, 1000),
    }

    try:
        response = await client.get("/api/v3/aggTrades", params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # MEXC aggTrades fields: a (id), p (price), q (qty), f (first trade id),
        # l (last trade id), T (timestamp), m (isBuyerMaker)
        df = df.rename(columns={
            "a": "id", "p": "price", "q": "qty",
            "T": "time", "m": "isBuyerMaker",
        })
        df["price"] = df["price"].astype(np.float64)
        df["qty"] = df["qty"].astype(np.float64)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch recent trades (non-fatal): {e}")
        return pd.DataFrame()
```

### 1.3 — Create `src/features_v2.py`

Complete specification below. The class must match this EXACT interface:

```python
class FeatureEngineV2:
    """V5 feature engine — 70+ features optimized for 5-minute BTC prediction."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def compute_features(
        self,
        df: pd.DataFrame,
        higher_tf_data: dict[str, pd.DataFrame] | None = None,
        trade_data: pd.DataFrame | None = None,
        ffill: bool = False,
    ) -> pd.DataFrame:
        """Compute all V5 features.

        Args:
            df: 5m OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)
            higher_tf_data: Optional dict of higher TF DataFrames {"15m": df, "1h": df}
            trade_data: Optional recent trades DataFrame for volume delta features
            ffill: If True, forward-fill NaN instead of dropping

        Returns:
            DataFrame with feature columns only (no OHLCV, no timestamp)
        """

    def get_feature_names(self) -> list[str]:
        """Return ordered list of all feature names."""
```

#### Feature Groups (ALL must be implemented)

**Group 1: Price Action Microstructure (12 features)**
```
body_ratio          = abs(close - open) / (high - low + 1e-10)
upper_wick_ratio    = (high - max(open, close)) / (high - low + 1e-10)
lower_wick_ratio    = (min(open, close) - low) / (high - low + 1e-10)
candle_direction    = sign(close - open)  # -1, 0, 1
consec_up           = count of consecutive up closes (reset on down)
consec_down          = count of consecutive down closes (reset on up)
hl_range_pct        = (high - low) / close * 100
gap_pct             = (open - prev_close) / prev_close * 100
close_position       = (close - low) / (high - low + 1e-10)  # where close sits in candle
real_body_pct       = abs(close - open) / close * 100
candle_pattern_3    = encode last 3 candle directions as single int (-1/0/1 → base3)
candle_pattern_5    = encode last 5 candle directions as single int
```

**Group 2: Multi-Scale Momentum (16 features)**
```
return_1            = pct_change(close, 1)
return_2            = pct_change(close, 2)
return_3            = pct_change(close, 3)
return_5            = pct_change(close, 5)
return_8            = pct_change(close, 8)
return_13           = pct_change(close, 13)
return_21           = pct_change(close, 21)
return_34           = pct_change(close, 34)
rsi_3               = RSI(close, 3)
rsi_5               = RSI(close, 5)
rsi_8               = RSI(close, 8)
rsi_14              = RSI(close, 14)
roc_3               = rate_of_change(close, 3)
roc_5               = rate_of_change(close, 5)
momentum_accel      = return_1 - return_1.shift(1)  # acceleration
momentum_jerk       = momentum_accel - momentum_accel.shift(1)  # jerk (3rd derivative)
```

**Group 3: Volatility & Regime (10 features)**
```
atr_5               = ATR(5) / close * 100  # normalized
atr_14              = ATR(14) / close * 100
atr_ratio           = atr_5 / (atr_14 + 1e-10)  # volatility expansion/contraction
bb_width_10         = (upper_bb_10 - lower_bb_10) / mid_bb_10 * 100
bb_position_10      = (close - lower_bb_10) / (upper_bb_10 - lower_bb_10 + 1e-10)
bb_squeeze          = 1 if bb_width_10 < bb_width_10.rolling(50).quantile(0.2) else 0
std_5               = close.pct_change().rolling(5).std() * 100
std_20              = close.pct_change().rolling(20).std() * 100
volatility_ratio    = std_5 / (std_20 + 1e-10)
range_expansion     = hl_range_pct / hl_range_pct.rolling(20).mean()
```

**Group 4: Volume Profile (10 features)**
```
volume_sma_5        = volume / volume.rolling(5).mean()  # relative volume
volume_sma_20       = volume / volume.rolling(20).mean()
volume_change       = safe_pct_change(volume, 1)  # guard against zero
vwap_deviation      = (close - vwap) / close * 100  # where: vwap = cumsum(close*volume)/cumsum(volume) over session
obv_roc_5           = OBV.pct_change(5)  # on-balance volume rate of change
obv_roc_14          = OBV.pct_change(14)
volume_price_corr_10 = rolling_corr(close.pct_change(), volume.pct_change(), 10)
mfi_5               = MFI(5)
mfi_14              = MFI(14)
volume_direction    = volume * candle_direction  # signed volume
```

Note for VWAP: Use a rolling 60-candle (5 hour) window as session proxy:
```python
typical_price = (high + low + close) / 3
vwap = (typical_price * volume).rolling(60).sum() / volume.rolling(60).sum()
```

**Group 5: Order Flow Proxy (6 features — uses trade_data if available, else estimates)**
```
buy_volume_ratio    = buy_volume / (buy_volume + sell_volume + 1e-10)  # from aggTrades if available
volume_delta        = buy_volume - sell_volume  # net buying pressure
delta_pct           = volume_delta / (total_volume + 1e-10) * 100
cvd_5               = cumulative volume delta over last 5 candles
cvd_divergence      = sign(cvd_5) != sign(return_5)  # 1 if diverging
trade_intensity     = number of trades / candle duration (trades per minute)
```

If `trade_data` is None, estimate buy/sell split:
```python
# Estimate: if close > open, assume (close-open)/(high-low) fraction was buying
buy_frac = np.where(high != low, (close - low) / (high - low), 0.5)
buy_volume = volume * buy_frac
sell_volume = volume * (1 - buy_frac)
```

**Group 6: Trend Indicators (8 features)**
```
ema_9               = EMA(close, 9)
ema_21              = EMA(close, 21)
ema_cross           = (ema_9 - ema_21) / close * 100  # normalized distance
ema_cross_signal    = sign(ema_cross) != sign(ema_cross.shift(1))  # crossover flag
macd_5_13           = MACD(5, 13, 4) — optimized for 5m
macd_signal_5_13    = MACD signal line
macd_hist_5_13      = MACD histogram
adx_10              = ADX(10) / 100  # normalized 0-1
```

**Group 7: Time/Session Features (6 features)**
```
hour_sin            = sin(2 * pi * hour / 24)
hour_cos            = cos(2 * pi * hour / 24)
day_of_week_sin     = sin(2 * pi * day_of_week / 7)
day_of_week_cos     = cos(2 * pi * day_of_week / 7)
is_asian_session    = 1 if 0 <= hour < 8 else 0  # UTC
is_us_session       = 1 if 13 <= hour < 21 else 0  # UTC
```

**Group 8: Higher Timeframe Context (8 features — from 15m and 1h data)**
```
htf_15m_rsi_5       = RSI(close_15m, 5)
htf_15m_return_3    = pct_change(close_15m, 3)
htf_15m_atr_ratio   = ATR_15m(5) / ATR_15m(14)
htf_15m_adx         = ADX(close_15m, 10) / 100
htf_1h_rsi_5        = RSI(close_1h, 5)
htf_1h_return_3     = pct_change(close_1h, 3)
htf_1h_atr_ratio    = ATR_1h(5) / ATR_1h(14)
htf_1h_trend        = sign(EMA(close_1h, 9) - EMA(close_1h, 21))  # 1h trend direction
```

Reindex higher TF features to 5m index using ffill (same as current `features.py`).

**Total: 76 features**

#### Implementation Notes for `features_v2.py`

1. **All price-scale features MUST be normalized** (percentage, z-score, or ratio). No raw prices.
2. **Guard all divisions** with `+ 1e-10` in denominator.
3. **NaN handling**: Compute all features, then at the end:
   - If `ffill=True`: `feat.ffill().bfill()` (for inference)
   - If `ffill=False`: `feat.dropna()` (for training)
4. **Return DataFrame with ONLY feature columns** — drop timestamp, open, high, low, close, volume.
5. **Feature names must be deterministic** — `get_feature_names()` returns the exact same list every time.
6. **Use numpy vectorized operations** — no Python loops over rows.
7. **RSI, ATR, MFI, ADX, EMA, MACD**: Implement as static methods or inline (do NOT import ta-lib).
   Copy the implementations from existing `features.py` where available.
8. **The existing `features.py` RSI/ATR/MFI/ADX implementations** should be extracted into shared
   helper functions or reimplemented in `features_v2.py` (prefer self-contained).

#### Validation Criteria (Phase 1 is DONE when):
- [x] `features_v2.py` imports without error
- [x] `FeatureEngineV2(config).compute_features(df_5m, htf_data)` returns DataFrame with 76 columns
- [x] No NaN in output when `ffill=True`
- [x] All feature values are finite (no inf)
- [x] Feature names list has 76 entries matching column names
- [x] `data_fetcher.py` has `fetch_recent_trades` method
- [x] `requirements.txt` includes lightgbm and catboost

---

## PHASE 2: Multi-Model Ensemble + Regime Detection

**Goal:** Build regime detector and 3-model ensemble with proper training pipeline.
**Files:** `src/regime.py` (new), `src/ensemble.py` (new)
**Estimated size:** ~700 lines total
**Depends on:** Phase 1 complete

### Status
- [ ] 2.1 Create `src/regime.py`
- [ ] 2.2 Create `src/ensemble.py`
- [ ] 2.3 Verify ensemble trains on sample data without errors
- [ ] 2.4 Commit: `Phase 2: Multi-Model Ensemble + Regime Detection`

### 2.1 — Create `src/regime.py`

```python
class RegimeDetector:
    """Classifies market into regimes for model routing.

    Regimes:
        TRENDING_UP   = 0 — Strong uptrend (ADX > 25, EMA9 > EMA21)
        TRENDING_DOWN = 1 — Strong downtrend (ADX > 25, EMA9 < EMA21)
        RANGING       = 2 — Low volatility, sideways (ADX < 20)
        VOLATILE      = 3 — High volatility, no clear trend (ADX 20-25 or ATR spike)
    """

    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """Classify each row into a regime.

        Uses these features (must exist in features df):
        - adx_10: ADX normalized 0-1 (multiply by 100 for thresholds)
        - ema_cross: EMA9-EMA21 distance (positive = uptrend)
        - atr_ratio: ATR5/ATR14 (>1.2 = volatility expansion)
        - bb_squeeze: Bollinger squeeze flag

        Returns:
            Series of regime labels (int: 0-3)
        """

    def get_regime_weights(self, regime: int) -> dict[str, float]:
        """Return model weights for a given regime.

        Returns:
            {"momentum": w1, "mean_reversion": w2, "microstructure": w3}
            Weights sum to 1.0.
        """
```

**Regime weight matrix:**

| Regime | Momentum | Mean Reversion | Microstructure |
|--------|----------|----------------|----------------|
| TRENDING_UP | 0.50 | 0.10 | 0.40 |
| TRENDING_DOWN | 0.50 | 0.10 | 0.40 |
| RANGING | 0.10 | 0.50 | 0.40 |
| VOLATILE | 0.25 | 0.25 | 0.50 |

### 2.2 — Create `src/ensemble.py`

```python
class EnsembleModel:
    """Multi-model ensemble with regime-aware weighting.

    Contains three sub-models:
    1. Momentum Model (XGBoost) — trained on TRENDING regime data only
    2. Mean Reversion Model (LightGBM) — trained on RANGING regime data only
    3. Microstructure Model (CatBoost) — trained on ALL data

    Training pipeline:
    1. Compute features using FeatureEngineV2
    2. Create labels: 1 if next candle close > open, else 0
    3. Detect regimes on training data
    4. Split: INNER(65%) | PURGE(20 candles) | CAL(10%) | PURGE(20) | OOS(10%)
    5. Train momentum model on INNER rows where regime in [TRENDING_UP, TRENDING_DOWN]
    6. Train mean_reversion model on INNER rows where regime in [RANGING]
    7. Train microstructure model on ALL INNER rows
    8. Optuna tune each model independently (15 trials each, 300s timeout)
    9. Feature prune each model: keep top 25 features per model
    10. Calibrate each model on CAL split (per-regime calibrators)
    11. Evaluate ensemble on OOS split
    12. Quality gate: ensemble OOS accuracy >= 53% AND >= old_accuracy - 0.5%
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.momentum_model = None       # XGBoost
        self.mean_reversion_model = None  # LightGBM
        self.microstructure_model = None  # CatBoost
        self.calibrators = {}             # {regime_id: CalibratorV2}
        self.feature_engine = FeatureEngineV2(config)
        self.feature_names = {"momentum": [], "mean_reversion": [], "microstructure": []}
        self.training_stats = {}

    async def train(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        """Full training pipeline.

        Returns:
            Dict with training stats:
            {
                "oos_accuracy": float,
                "oos_accuracy_per_regime": {regime: accuracy},
                "model_accuracies": {"momentum": acc, "mean_reversion": acc, "micro": acc},
                "ensemble_accuracy": float,
                "feature_counts": {"momentum": N, "mean_reversion": N, "micro": N},
                "regime_distribution": {regime: count},
                "cal_spread": {"min": float, "max": float, "mean": float},
            }
        """

    def predict(self, features: pd.DataFrame) -> dict:
        """Generate prediction from ensemble.

        Args:
            features: Single row or few rows of features from FeatureEngineV2

        Returns:
            {
                "signal": "UP" or "DOWN",
                "raw_prob_up": float,   # ensemble raw probability
                "cal_prob_up": float,   # calibrated probability
                "confidence": float,    # max(cal_prob_up, 1 - cal_prob_up)
                "regime": int,          # detected regime
                "regime_name": str,     # human-readable
                "model_agreement": int, # how many models agree (1-3)
                "model_probs": {"momentum": p, "mean_reversion": p, "micro": p},
                "ev": float,            # expected value
            }
        """

    def save(self, path: str):
        """Save all models, calibrators, feature names, and metadata to path/"""

    @classmethod
    def load(cls, path: str, config: ModelConfig) -> "EnsembleModel":
        """Load ensemble from path/"""
```

#### Optuna Search Spaces

**Momentum Model (XGBoost):**
```python
{
    "max_depth": trial.suggest_int("max_depth", 3, 8),
    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
    "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    "gamma": trial.suggest_float("gamma", 0.0, 1.0),
}
```

**Mean Reversion Model (LightGBM):**
```python
{
    "max_depth": trial.suggest_int("max_depth", 3, 8),
    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
}
```

**Microstructure Model (CatBoost):**
```python
{
    "depth": trial.suggest_int("depth", 4, 8),
    "iterations": trial.suggest_int("iterations", 200, 600, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
    "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
}
```

#### Training Data Split (CRITICAL — same approach as V4, extended to ensemble)
```
|<------- INNER 65% ------->|<-PURGE->|<-CAL 10%->|<-PURGE->|<-OOS 10%->|
|  20 candle gap between each section to prevent feature leakage         |
```

- PURGE_GAP = 20 candles (100 minutes) — same as V4
- Inner: used for Optuna CV + final model training
- CAL: used for calibrator fitting
- OOS: used ONLY for evaluation — never seen during training or calibration

#### Model Save Format
```
data/ensemble_model/
    momentum_model.json          # XGBoost native save
    mean_reversion_model.txt     # LightGBM native save
    microstructure_model.cbm     # CatBoost native save
    calibrators.pkl              # Dict of calibrator objects
    feature_names.json           # {"momentum": [...], "mean_reversion": [...], "micro": [...]}
    metadata.json                # Training stats, timestamp, version
```

#### Validation Criteria (Phase 2 is DONE when):
- [ ] `regime.py` classifies regimes correctly (test on synthetic trending/ranging data)
- [ ] `ensemble.py` trains all 3 models without error on real MEXC data
- [ ] Each model uses feature pruning (top 25 per model)
- [ ] Optuna runs 15 trials per model within 300s each
- [ ] Calibrators fit on CAL split
- [ ] OOS evaluation produces accuracy metrics per regime
- [ ] Save/load round-trips successfully
- [ ] Training completes in < 20 minutes total on Railway (2 vCPU)

---

## PHASE 3: Adaptive Calibration + Trade Frequency System

**Goal:** Regime-conditional calibration that preserves probability spread + tiered trading system guaranteeing 70+ trades/day.
**Files:** `src/calibration_v2.py` (new), `src/trade_manager.py` (new)
**Estimated size:** ~350 lines total
**Depends on:** Phase 2 complete

### Status
- [ ] 3.1 Create `src/calibration_v2.py`
- [ ] 3.2 Create `src/trade_manager.py`
- [ ] 3.3 Verify calibrators produce spread probabilities
- [ ] 3.4 Verify trade manager produces 70+ signals on 24h of historical data
- [ ] 3.5 Commit: `Phase 3: Adaptive Calibration + Trade Manager`

### 3.1 — Create `src/calibration_v2.py`

```python
class CalibratorV2:
    """Regime-conditional probability calibration.

    Instead of one global isotonic calibrator (which collapsed all probs to ~50.4%),
    fits separate calibrators per regime. Also uses Platt scaling as a fallback
    when a regime has too few calibration samples.

    Calibrator selection:
    - If regime has >= 100 samples in CAL split: Isotonic regression
    - If regime has 30-99 samples: Platt scaling (logistic regression)
    - If regime has < 30 samples: No calibration (pass-through raw probability)
    """

    def __init__(self):
        self.calibrators = {}  # {regime_id: fitted calibrator}
        self.calibrator_types = {}  # {regime_id: "isotonic" | "platt" | "passthrough"}
        self.is_fitted = False

    def fit(
        self,
        raw_probs: np.ndarray,
        true_labels: np.ndarray,
        regimes: np.ndarray,
    ):
        """Fit per-regime calibrators.

        Args:
            raw_probs: Raw model P(UP) probabilities
            true_labels: True binary labels (1=UP, 0=DOWN)
            regimes: Regime labels (0-3) for each sample
        """

    def calibrate(self, raw_prob: float, regime: int) -> float:
        """Calibrate a single probability.

        Returns:
            Calibrated probability, clipped to [0.01, 0.99]
        """

    def get_stats(self) -> dict:
        """Return calibration statistics per regime."""

    def save(self, path: str): ...

    @classmethod
    def load(cls, path: str) -> "CalibratorV2": ...
```

### 3.2 — Create `src/trade_manager.py`

```python
class TradeManager:
    """Manages trade frequency and session risk.

    Tiered confidence system ensures 70+ trades/day while maintaining quality.
    Session risk monitor temporarily tightens filters during losing streaks.

    Tiers (evaluated in order — first match wins):
        Tier 1 (High):   cal_prob >= 0.57  → Trade (highest conviction)
        Tier 2 (Medium): cal_prob >= 0.54  → Trade (good conviction)
        Tier 3 (Base):   cal_prob >= 0.52 AND model_agreement >= 2  → Trade

    Session Risk:
        Tracks rolling accuracy over last 20 decided trades.
        If rolling accuracy drops below 48%, enter CAUTIOUS mode:
        - Tier 3 disabled (only Tier 1 + 2 trade)
        - Stays cautious for 30 minutes, then reverts
        If rolling accuracy drops below 42%, enter DEFENSIVE mode:
        - Only Tier 1 trades (cal_prob >= 0.57)
        - Stays defensive for 60 minutes, then reverts
    """

    # Risk modes
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"

    def __init__(self, config: ModelConfig):
        self.config = config
        self.risk_mode = self.NORMAL
        self._mode_until = None  # datetime when mode reverts to NORMAL
        self._recent_results = []  # list of bools, True=WIN
        self._rolling_window = 20

    def should_trade(self, prediction: dict) -> dict:
        """Decide whether to trade based on prediction and current risk mode.

        Args:
            prediction: Dict from EnsembleModel.predict() containing:
                - confidence: float
                - cal_prob_up: float
                - model_agreement: int
                - ev: float
                - regime: int

        Returns:
            {
                "trade": bool,
                "tier": int or None,     # 1, 2, 3, or None if no trade
                "reason": str,           # why trade/skip
                "risk_mode": str,        # NORMAL, CAUTIOUS, DEFENSIVE
                "rolling_accuracy": float or None,
            }
        """

    def record_result(self, won: bool):
        """Record a trade result for rolling accuracy tracking."""

    def get_stats(self) -> dict:
        """Return trade manager statistics."""

    def _check_risk_mode(self):
        """Update risk mode based on rolling accuracy and time."""
```

#### Trade Frequency Math

With 288 five-minute slots per day:
- Tier 1 (prob >= 0.57): ~15-25% of slots → 43-72 trades
- Tier 2 (prob >= 0.54): ~20-30% more → 58-86 additional
- Tier 3 (agreement >= 2): ~10-20% more → 29-58 additional
- **Conservative estimate with overlap: 70-120 trades/day**

If actual count is lower after calibration, Phase 5 adjusts thresholds.

#### Validation Criteria (Phase 3 is DONE when):
- [ ] CalibratorV2 produces different calibration per regime
- [ ] Calibrated probabilities have spread > 0.10 (not all collapsed to 0.504)
- [ ] TradeManager correctly gates tiers based on confidence
- [ ] Risk mode transitions work (NORMAL → CAUTIOUS → DEFENSIVE → NORMAL)
- [ ] TradeManager produces 70+ trade signals on a simulated 24h prediction set

---

## PHASE 4: Integration — Wire Everything Into Bot

**Goal:** Replace the prediction pipeline in `bot.py` with the V5 ensemble, preserving all existing functionality.
**Files:** Modify `config.py`, `bot.py`, `model.py`, `formatters.py`, `.env.example`
**Estimated size:** ~300 lines of modifications
**Depends on:** Phases 1-3 complete

### Status
- [ ] 4.1 Add V5 config to `config.py`
- [ ] 4.2 Modify `bot.py` — add V5 ensemble pipeline
- [ ] 4.3 Add bridge method to `model.py` (fallback support)
- [ ] 4.4 Update `formatters.py` — ensemble signal format
- [ ] 4.5 Update `.env.example` with V5 variables
- [ ] 4.6 Verify full bot startup with `USE_V5_ENSEMBLE=true`
- [ ] 4.7 Verify fallback to V4 with `USE_V5_ENSEMBLE=false`
- [ ] 4.8 Commit: `Phase 4: V5 Ensemble Integration`

### 4.1 — Add V5 Config to `config.py`

Add new dataclass (do NOT modify existing ModelConfig — extend it):

```python
@dataclass
class EnsembleConfig:
    """V5 Ensemble configuration."""
    use_v5_ensemble: bool = field(default_factory=lambda: os.getenv("USE_V5_ENSEMBLE", "true").lower() == "true")

    # Training
    train_candles: int = field(default_factory=lambda: int(os.getenv("V5_TRAIN_CANDLES", "20000")))  # ~70 days, fresher data
    retrain_interval_hours: int = field(default_factory=lambda: int(os.getenv("V5_RETRAIN_HOURS", "2")))
    sample_weight_recent_multiplier: float = field(default_factory=lambda: float(os.getenv("V5_RECENT_WEIGHT", "3.0")))
    recent_window_frac: float = 0.33  # Last 33% of training data gets weight multiplier

    # Optuna
    optuna_trials_per_model: int = field(default_factory=lambda: int(os.getenv("V5_OPTUNA_TRIALS", "15")))
    optuna_timeout_per_model: int = field(default_factory=lambda: int(os.getenv("V5_OPTUNA_TIMEOUT", "300")))

    # Feature pruning
    feature_prune_top_n: int = field(default_factory=lambda: int(os.getenv("V5_PRUNE_TOP_N", "25")))

    # Trade tiers
    tier1_threshold: float = field(default_factory=lambda: float(os.getenv("V5_TIER1_THRESHOLD", "0.57")))
    tier2_threshold: float = field(default_factory=lambda: float(os.getenv("V5_TIER2_THRESHOLD", "0.54")))
    tier3_threshold: float = field(default_factory=lambda: float(os.getenv("V5_TIER3_THRESHOLD", "0.52")))
    tier3_min_agreement: int = field(default_factory=lambda: int(os.getenv("V5_TIER3_MIN_AGREEMENT", "2")))

    # Session risk
    cautious_accuracy_threshold: float = 0.48
    defensive_accuracy_threshold: float = 0.42
    cautious_duration_minutes: int = 30
    defensive_duration_minutes: int = 60
    rolling_window: int = 20

    # Quality gate
    min_oos_accuracy: float = field(default_factory=lambda: float(os.getenv("V5_MIN_OOS_ACC", "0.53")))

    # Model save path
    model_dir: str = field(default_factory=lambda: os.getenv("V5_MODEL_DIR", "data/ensemble_model"))
```

### 4.2 — Modify `bot.py`

Key changes (SURGICAL — only modify what's needed):

1. **Import V5 modules** at top:
```python
from .features_v2 import FeatureEngineV2
from .ensemble import EnsembleModel
from .trade_manager import TradeManager
from .config import EnsembleConfig
```

2. **In `__init__`**, add V5 initialization:
```python
self.ensemble_config = EnsembleConfig()
if self.ensemble_config.use_v5_ensemble:
    self.ensemble = EnsembleModel(self.config.model)
    self.trade_manager = TradeManager(self.config.model)
    self.feature_engine_v2 = FeatureEngineV2(self.config.model)
else:
    self.ensemble = None
    self.trade_manager = None
```

3. **In `_generate_signal()`**, add V5 branch:
```python
if self.ensemble_config.use_v5_ensemble and self.ensemble is not None:
    return await self._generate_signal_v5(current_data, htf_data)
else:
    return await self._generate_signal_v4(current_data, htf_data)  # existing code
```

4. **New method `_generate_signal_v5()`**:
   - Fetches trade data via `self.fetcher.fetch_recent_trades()`
   - Computes features via `self.feature_engine_v2.compute_features()`
   - Gets prediction via `self.ensemble.predict()`
   - Gets trade decision via `self.trade_manager.should_trade()`
   - Returns same signal dict format as V4 (plus extra fields: regime, tier, model_agreement)

5. **In `_retrain()`**, add V5 branch:
   - If V5: call `self.ensemble.train()` instead of `self.model.retrain()`
   - Use `ensemble_config.retrain_interval_hours` (2h instead of 6h)
   - Same quality gate logic, adapted for ensemble stats

6. **In resolution**, call `self.trade_manager.record_result(won)` to update rolling accuracy.

### 4.3 — Bridge in `model.py`

Add at the very end of model.py (do NOT modify existing code):
```python
# V5 Bridge — allows bot.py to use either V4 or V5 pipeline
def get_prediction_model(config, use_v5=False):
    if use_v5:
        from .ensemble import EnsembleModel
        return EnsembleModel(config)
    return PredictionModel(config)
```

### 4.4 — Update `formatters.py`

Add new formatter function (append, do not modify existing):
```python
def format_ensemble_signal_message(
    signal: dict,
    tracker_stats,
    trade_decision: dict,
) -> str:
    """Format V5 ensemble signal for Telegram.

    Shows: direction, calibrated confidence, EV, regime, tier, model agreement,
    risk mode, rolling accuracy.
    """
```

The message should include:
- Signal direction + confidence (same visual as V4)
- `Regime: TRENDING_UP` (or whichever)
- `Models: 3/3 agree` (or `2/3 agree`)
- `Tier: 1 (High Conviction)` / `2 (Medium)` / `3 (Base)`
- `Risk Mode: NORMAL` (or CAUTIOUS/DEFENSIVE)
- `Rolling Accuracy: 57.5% (20 trades)`
- EV calculation

### 4.5 — Update `.env.example`

Add V5 section (append after existing content):
```ini
# =============================================================================
# V5: MULTI-MODEL ENSEMBLE (Optional — defaults shown)
# =============================================================================

# Enable V5 ensemble (false = use V4 single XGBoost)
USE_V5_ENSEMBLE=true

# Training data window (~70 days of 5m data, fresher = better)
V5_TRAIN_CANDLES=20000

# Retrain interval (faster adaptation)
V5_RETRAIN_HOURS=2

# Optuna trials per model (3 models × 15 = 45 total)
V5_OPTUNA_TRIALS=15
V5_OPTUNA_TIMEOUT=300

# Feature pruning (per model)
V5_PRUNE_TOP_N=25

# Trade tiers
V5_TIER1_THRESHOLD=0.57
V5_TIER2_THRESHOLD=0.54
V5_TIER3_THRESHOLD=0.52
V5_TIER3_MIN_AGREEMENT=2

# Minimum OOS accuracy for model swap
V5_MIN_OOS_ACC=0.53

# Weight multiplier for recent training data
V5_RECENT_WEIGHT=3.0
```

### Validation Criteria (Phase 4 is DONE when):
- [ ] Bot starts with `USE_V5_ENSEMBLE=true` and runs signal loop
- [ ] Bot starts with `USE_V5_ENSEMBLE=false` and runs V4 pipeline (unchanged)
- [ ] V5 signals appear in Telegram with ensemble metadata
- [ ] Retraining works with V5 (trains 3 models, calibrates, evaluates OOS)
- [ ] Auto-trading works with V5 signals (same signal dict format)
- [ ] Resolution updates trade_manager rolling accuracy
- [ ] All existing Telegram commands work (/stats, /recent, /status, etc.)
- [ ] No import errors, no runtime crashes

---

## PHASE 5: Testing, Validation & Deployment Hardening

**Goal:** Comprehensive testing, backtest validation, threshold tuning, deployment verification.
**Files:** `tests/` (new), tuning adjustments to config
**Depends on:** Phase 4 complete

### Status
- [ ] 5.1 Create `tests/test_features_v2.py`
- [ ] 5.2 Create `tests/test_ensemble.py`
- [ ] 5.3 Create `tests/test_integration.py`
- [ ] 5.4 Run full backtest: 7 days of historical data, measure accuracy + trade count
- [ ] 5.5 Tune thresholds if needed to hit 70+ trades/day
- [ ] 5.6 Verify Railway deployment (Docker build, memory, runtime)
- [ ] 5.7 Final commit: `Phase 5: Testing & Validation — V5 Production Ready`

### 5.1 — `tests/test_features_v2.py`
```python
# Test cases:
# 1. Feature count is exactly 76
# 2. No NaN when ffill=True
# 3. No inf values
# 4. Feature names match column names
# 5. Output is pure features (no OHLCV columns)
# 6. Works without higher_tf_data (those features should be NaN/filled)
# 7. Works without trade_data (uses estimation)
# 8. Deterministic: same input → same output
```

### 5.2 — `tests/test_ensemble.py`
```python
# Test cases:
# 1. Regime detector assigns all 4 regime types
# 2. Regime weights sum to 1.0 for each regime
# 3. Ensemble trains without error on 2000 candles of sample data
# 4. Each sub-model produces probabilities in [0, 1]
# 5. Ensemble predict returns correct dict structure
# 6. Save/load roundtrip preserves predictions
# 7. Quality gate correctly rejects bad models
# 8. Calibrator produces spread > 0.10
```

### 5.3 — `tests/test_integration.py`
```python
# Test cases:
# 1. Full pipeline: raw OHLCV → features → ensemble → trade decision
# 2. TradeManager produces 70+ trades on 288 simulated slots
# 3. Risk mode transitions (simulate losing streak)
# 4. V4 fallback works when USE_V5_ENSEMBLE=false
# 5. Signal dict has all required fields for auto_trader.py
```

### 5.4 — Historical Backtest

Procedure:
1. Fetch 7 days of 5m data (2016 candles)
2. Train ensemble on preceding 70 days
3. Walk-forward: predict each of the 2016 candles
4. Measure: accuracy, trade count (with trade manager), EV per trade, regime distribution
5. Log results to `data/backtest_results.json`

Target:
- Accuracy >= 55%
- Trade count >= 70/day (490+ over 7 days)
- Average EV > $0.00

### 5.5 — Threshold Tuning

If backtest shows < 70 trades/day:
- Lower `V5_TIER3_THRESHOLD` from 0.52 to 0.51
- Lower `V5_TIER2_THRESHOLD` from 0.54 to 0.53
- Re-run backtest

If backtest shows < 55% accuracy:
- Increase `V5_PRUNE_TOP_N` from 25 to 30
- Increase `V5_OPTUNA_TRIALS` from 15 to 25
- Increase `V5_TRAIN_CANDLES` from 20000 to 30000
- Re-run backtest

### 5.6 — Railway Deployment Verification

- [ ] `docker build` succeeds with lightgbm + catboost
- [ ] Memory usage < 512MB during training
- [ ] Training completes in < 20 minutes
- [ ] Bot connects to Telegram and runs signal loop
- [ ] First 10 signals have realistic accuracy (not 100% or 0%)

### Validation Criteria (Phase 5 is DONE when):
- [ ] All tests pass
- [ ] Backtest shows >= 55% accuracy on 7-day window
- [ ] Backtest shows >= 70 trades per day
- [ ] Railway deployment runs stable for 1 hour
- [ ] V4 fallback still works

---

## AGENT INSTRUCTIONS

### Starting a New Session

1. **Read this file first.** Find the first unchecked `[ ]` item — that's your work.
2. **Read the files** listed for that phase to understand current state.
3. **Implement exactly as specified.** Do not deviate from function signatures, feature names, or architecture.
4. **Test your work** before marking items complete.
5. **Update this file** — check off completed items, add session log entry.
6. **Commit** with the phase message format.

> **🛑 HARD STOP: ONE PHASE PER SESSION.** After completing one phase, STOP. Do not proceed to the next phase.

### Rules

> **🛑 HARD STOP RULE: ONE PHASE PER SESSION.**
> An agent MUST complete only ONE phase per session, then STOP and commit.
> Do NOT continue to the next phase. Do NOT "get a head start." STOP.
> This prevents exceeding usage limits and ensures review between phases.

- **ONE PHASE PER SESSION. This is non-negotiable.** Complete your phase, commit, and STOP.
- **Do NOT skip validation criteria.** Each phase must pass ALL checks before moving on.
- **Do NOT modify files outside the current phase** unless fixing a blocking bug from a prior phase.
- **Do NOT rename features or change function signatures** from what's specified above.
- **Do NOT add extra features or models** beyond what's specified. Scope creep kills projects.
- **Do NOT start the next phase** even if you have "extra time" or think you can squeeze it in.
- **If a prior phase has a bug**, fix it, add a note in the session log, and re-check its validation criteria.
- **If you finish early**, verify your work by running tests or re-reading validation criteria — do NOT move ahead.

### File Reading Order for Each Phase

| Phase | Read First |
|-------|------------|
| 1 | `features.py` (reference), `data_fetcher.py`, `config.py` |
| 2 | `features_v2.py` (from Phase 1), `model.py` (reference for split/optuna), `config.py` |
| 3 | `ensemble.py` (from Phase 2), `config.py` |
| 4 | `bot.py`, `config.py`, `formatters.py`, `ensemble.py`, `trade_manager.py` |
| 5 | All `src/` files, `tests/` directory |

---

## SESSION LOG

### Session 0 — Planning (2026-03-25)
- Created this implementation plan
- Analyzed deployment logs: 51.7% OOS accuracy, ~3 trades/day, negative EV
- Root cause: standard TA features have near-zero signal on 5m BTC
- Solution: Multi-model ensemble with microstructure features, regime switching, tiered trading
- Phases 1-5 fully specified with exact function signatures, feature lists, and validation criteria

### Session 1 — Phase 1: Feature Engine V2 (2025-03-25)
- Added `lightgbm==4.6.0` and `catboost==1.2.7` to `requirements.txt`
- Added `fetch_recent_trades()` method to `MEXCFetcher` in `data_fetcher.py` (MEXC `/api/v3/aggTrades` endpoint)
- Created `src/features_v2.py` with `FeatureEngineV2` class implementing all 8 feature groups (76 features total):
  - Group 1: Price Action Microstructure (12 features)
  - Group 2: Multi-Scale Momentum (16 features)
  - Group 3: Volatility & Regime (10 features)
  - Group 4: Volume Profile (10 features)
  - Group 5: Order Flow Proxy (6 features — with OHLCV estimation fallback)
  - Group 6: Trend Indicators (8 features)
  - Group 7: Time/Session Features (6 features)
  - Group 8: Higher Timeframe Context (8 features — with neutral defaults fallback)
- All TA indicators (RSI, ATR, MFI, ADX, MACD, EMA) implemented as self-contained static methods (no ta-lib dependency)
- Validation passed: 76 columns, 0 NaN (ffill=True), 0 inf, feature names deterministic
- Fixed `_fill_htf_nan()` to use neutral defaults (RSI=50, returns=0, ATR ratio=1) instead of NaN for HTF-missing case
- Fixed `fetch_recent_trades()`: corrected endpoint from `/api/v3/trades` to `/api/v3/aggTrades`, removed extra `symbol` parameter, added aggTrades column renames (`a→id, p→price, q→qty, T→time, m→isBuyerMaker`), changed error handling to non-fatal (returns empty DataFrame)
