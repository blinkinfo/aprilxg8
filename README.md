# AprilXG V5 — BTC 5-Min Binary Signal Bot

> **Multi-model ensemble** ML bot that predicts the direction of 5-minute BTC candles and optionally auto-trades on [Polymarket](https://polymarket.com) BTC Up/Down binary markets.

---

## 🚀 Overview

AprilXG V5 is a major upgrade from V4, introducing a **3-model ensemble** with regime-aware weighting, 76 microstructure features, per-regime probability calibration, and a tiered trade frequency system with session risk management.

| Feature | V4 | V5 |
|---|---|---|
| Model | Single XGBoost | 3-Model Ensemble (XGBoost + LightGBM + CatBoost) |
| Features | 50+ | 76 microstructure features across 8 groups |
| Calibration | Global isotonic | Per-regime (isotonic / Platt / passthrough) |
| Trade Management | Flat confidence threshold | 3-tier system + session risk modes |
| Regime Detection | Basic ATR flag | 4-regime classifier (ADX + EMA + ATR) |
| Retrain Interval | Every 6 hours | Every 2 hours (faster adaptation) |
| Training Window | ~150 days (~43,200 candles) | ~70 days (~20,000 candles, fresher data) |

---

## 📡 How It Works

1. **~90 seconds before each 5-min candle closes**, the bot fetches multi-timeframe data (5m, 15m, 1h) from MEXC
2. **76 features are engineered** via `FeatureEngineV2` across 8 feature groups
3. **Market regime is detected** (Trending Up, Trending Down, Ranging, Volatile)
4. **3-model ensemble predicts** the next candle direction with regime-weighted soft voting
5. **Per-regime calibration** adjusts raw probabilities for reliability
6. **TradeManager evaluates** the signal against a 3-tier confidence system and session risk mode
7. **Qualifying signals** are posted to Telegram; optionally auto-traded on Polymarket
8. **30-90 seconds later**, the signal is resolved as WIN or LOSS
9. **Every 2 hours**, the ensemble retrains with a quality gate (OOS accuracy >= 53%)
10. **Resolved Polymarket positions** are automatically redeemed on-chain for USDC

---

## 🧠 V5 Ensemble Architecture

### 🎯 Three Specialized Models

| Model | Algorithm | Training Data | Role |
|---|---|---|---|
| **Momentum** | XGBoost | TRENDING regime data only | Captures trend continuation/reversal |
| **Mean Reversion** | LightGBM | RANGING regime data only | Captures mean-reversion signals |
| **Microstructure** | CatBoost | ALL data | General-purpose fallback + microstructure patterns |

Each model is independently tuned via **Optuna** (15 trials per model) and **feature-pruned** to the top 25 features.

### 🌐 Regime Detection

The `RegimeDetector` classifies each 5-minute bar into one of four market states:

| Regime | Condition | Description |
|---|---|---|
| `TRENDING_UP` | ADX > 25, EMA9 > EMA21 | Strong uptrend |
| `TRENDING_DOWN` | ADX > 25, EMA9 < EMA21 | Strong downtrend |
| `RANGING` | ADX < 20 | Low volatility, sideways |
| `VOLATILE` | ADX 20-25 or ATR spike | High volatility, no clear trend |

### ⚖️ Regime-Weighted Soft Voting

Each regime assigns different weights to the three models:

| Regime | Momentum | Mean Reversion | Microstructure |
|---|---|---|---|
| TRENDING_UP | 0.50 | 0.10 | 0.40 |
| TRENDING_DOWN | 0.50 | 0.10 | 0.40 |
| RANGING | 0.10 | 0.50 | 0.40 |
| VOLATILE | 0.25 | 0.25 | 0.50 |

### 📊 Per-Regime Probability Calibration

Instead of a single global calibrator (which collapsed spreads to ~50.4% in V4), V5 fits separate calibrators per regime:

| Samples in CAL Split | Calibrator Type |
|---|---|
| >= 100 | Isotonic Regression |
| 30 - 99 | Platt Scaling (Logistic Regression) |
| < 30 | Passthrough (no calibration) |

---

## 💹 Trade Management

### 🏆 3-Tier Confidence System

Designed to maintain **70+ trades/day** while preserving signal quality:

| Tier | Threshold | Condition | Description |
|---|---|---|---|
| **Tier 1** (High) | `cal_prob >= 0.57` | Always active | Highest conviction |
| **Tier 2** (Medium) | `cal_prob >= 0.54` | Active in NORMAL + CAUTIOUS | Good conviction |
| **Tier 3** (Base) | `cal_prob >= 0.52` | NORMAL only, `agreement >= 2` | Model consensus required |

### 🛡️ Session Risk Management

Tracks rolling accuracy over the last 20 decided trades:

| Risk Mode | Trigger | Effect | Duration |
|---|---|---|---|
| **NORMAL** | Default | All tiers active | -- |
| **CAUTIOUS** | Rolling accuracy < 48% | Tier 3 disabled | 30 minutes |
| **DEFENSIVE** | Rolling accuracy < 42% | Only Tier 1 active | 60 minutes |

---

## 📁 Feature Engine V2

76 normalized features across 8 groups (no raw prices -- all percentage, z-score, or ratio based):

| # | Group | Features | Examples |
|---|---|---|---|
| 1 | Price Action Microstructure | 12 | Body ratio, wick ratios, consecutive up/down, candle patterns |
| 2 | Multi-Scale Momentum | 16 | Returns (1-34 bars), RSI (3/5/8/14), ROC, momentum accel/jerk |
| 3 | Volatility & Regime | 10 | ATR ratio, Bollinger width/position/squeeze, vol ratio |
| 4 | Volume Profile | 10 | Volume SMA ratios, VWAP deviation, OBV ROC, MFI |
| 5 | Order Flow Proxy | 6 | Buy volume ratio, volume delta, CVD, trade intensity |
| 6 | Trend Indicators | 8 | EMA cross/signal, MACD (5/13), ADX |
| 7 | Time/Session | 6 | Hour/day cyclical encoding, Asian/US session flags |
| 8 | Higher Timeframe Context | 8 | 15m + 1h RSI, returns, ATR ratio, trend |

---

## 💻 Training Pipeline

```
1. Fetch multi-TF OHLCV data from MEXC (5m, 15m, 1h)
2. Compute 76 features via FeatureEngineV2
3. Create binary labels: next candle close > open = 1, else 0
4. Detect regimes on training data
5. Split: INNER (65%) | PURGE (20 bars) | CAL (10%) | PURGE (20) | OOS (10%)
6. Train Momentum model (XGBoost) on INNER TRENDING rows
7. Train Mean Reversion model (LightGBM) on INNER RANGING rows
8. Train Microstructure model (CatBoost) on ALL INNER rows
9. Optuna-tune each model independently (15 trials, 300s timeout each)
10. Feature-prune each model to top 25 features
11. Calibrate per-regime on CAL split
12. Evaluate ensemble on OOS split
13. Quality gate: OOS accuracy >= 53% AND >= old_accuracy - 0.5%
```

---

## 🎰 Polymarket Integration

### 🎯 Market Discovery

BTC 5-min Up/Down markets follow a deterministic slug pattern:
```
btc-updown-5m-{slot_timestamp}
```
where `slot_timestamp = (unix_time // 300) * 300`. Markets are looked up by slug via the Gamma API.

### 💰 Order Execution -- FOK Market Orders

- **Order type:** `MarketOrderArgs` with `OrderType.FOK` (Fill-or-Kill)
- **Amount:** USDC to spend (e.g., `1.0` = $1.00)
- **Pricing:** SDK auto-calculates optimal price from the order book
- **Execution:** Fills entirely and immediately, or is rejected

### Binary Market Payout

| Outcome | Amount |
|---|---|
| Win | +$0.96 |
| Loss | -$1.00 |
| Breakeven Win Rate | ~51.04% |

### 🛡️ Safety Checks (6 Layers)

1. Auto-trading enabled?
2. Direction is UP or DOWN (not NEUTRAL)?
3. Polymarket client initialized?
4. Valid target slot timestamp?
5. Duplicate slot prevention (one trade per slot)
6. Sufficient USDC balance?

### 🔄 On-Chain Position Redemption

- Scans for redeemable (resolved) positions via the Polymarket Data API
- Supports both **CTF** (standard) and **NegRisk Adapter** markets
- **Safe/proxy wallet support** (signature_type=2): wraps calls in `execTransaction`
- EIP-1559 gas pricing with 30% gas buffer
- Session-level dedup to avoid re-redeeming
- Requires a small amount of POL in the EOA wallet for gas (~0.005 POL)

---

## 📂 Project Structure

```
main.py                     Entry point
Dockerfile                  Docker build (Python 3.11-slim)
railway.toml                Railway deployment config
requirements.txt            Python dependencies
.env.example                Environment variable reference

src/
  bot.py                    Main orchestrator (5s loop, timing, signal flow)
  config.py                 5 config dataclasses loaded from env vars
                            (MEXC, Model, Telegram, Polymarket, Ensemble)
  data_fetcher.py           MEXC API client (OHLCV, multi-TF, paginated history)
  ensemble.py               V5 3-model ensemble (XGBoost + LightGBM + CatBoost)
  features_v2.py            V5 feature engine (76 microstructure features)
  features.py               V4 feature engine (50+ features, legacy)
  regime.py                 4-regime market state classifier
  trade_manager.py          3-tier confidence + session risk management
  calibration_v2.py         Per-regime probability calibration (isotonic/Platt)
  model.py                  V4 XGBoost classifier (legacy, used when V5 disabled)
  signal_tracker.py         Signal lifecycle (add, resolve, stats, persistence)
  polymarket_client.py      Polymarket CLOB API (FOK orders, balance, positions)
  auto_trader.py            Trade orchestrator (safety checks, slot-targeted exec)
  position_redeemer.py      On-chain position redemption (CTF + NegRisk, Polygon)
  telegram_bot.py           Telegram interface (13 commands, HTML formatting)
  formatters.py             Centralized HTML message formatting

tests/
  test_features_v2.py       Feature engine V2 unit tests
  test_ensemble.py          Ensemble model unit tests
  test_integration.py       Integration tests
  helpers.py                Test utilities
```

---

## 🤖 Telegram Commands

| Command | Description |
|---|---|
| `/start` | Welcome message and chat ID |
| `/help` | Full command reference |
| `/stats` | Performance dashboard (W/L, PnL, streaks) |
| `/recent` | Last 10 signals with outcomes |
| `/status` | Bot status, model info, uptime |
| `/retrain` | Force ensemble retrain with inline Keep/Swap buttons |
| `/forcetune` | Force Optuna hyperparameter tuning |
| `/autotrade` | Toggle Polymarket auto-trading ON/OFF |
| `/setamount` | Set trade amount in USDC (e.g., `/setamount 2.50`) |
| `/balance` | Check Polymarket USDC wallet balance |
| `/positions` | View open Polymarket positions |
| `/pmstatus` | Full Polymarket connection and config status |
| `/redeem` | Manually trigger position redemption |

---

## ⚙️ Setup

### Prerequisites

- Python 3.11+
- Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Polymarket wallet with USDC (optional, for auto-trading)
- Small amount of POL for gas (optional, for on-chain redemption)

### 🔑 Environment Variables

See [`.env.example`](.env.example) for the full reference. Key variables:

```env
# -- Required --
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# -- Trading (Optional) --
TRADING_SYMBOL=BTCUSDT
LOG_LEVEL=INFO

# -- V5 Ensemble (Optional, defaults shown) --
USE_V5_ENSEMBLE=true              # false = fallback to V4 single XGBoost
V5_TRAIN_CANDLES=20000            # ~70 days of 5m data
V5_RETRAIN_HOURS=2                # Retrain every 2 hours
V5_OPTUNA_TRIALS=15               # Trials per model (3 models x 15 = 45)
V5_OPTUNA_TIMEOUT=300             # Timeout per model in seconds
V5_PRUNE_TOP_N=25                 # Top features to keep per model
V5_TIER1_THRESHOLD=0.57           # High conviction tier
V5_TIER2_THRESHOLD=0.54           # Medium conviction tier
V5_TIER3_THRESHOLD=0.52           # Base tier
V5_TIER3_MIN_AGREEMENT=2          # Minimum model agreement for Tier 3
V5_MIN_OOS_ACC=0.53               # Quality gate for model swap
V5_RECENT_WEIGHT=3.0              # Weight multiplier for recent training data

# -- Session Risk (Optional) --
V5_CAUTIOUS_ACCURACY=0.48         # Enter CAUTIOUS below this
V5_DEFENSIVE_ACCURACY=0.42        # Enter DEFENSIVE below this
V5_CAUTIOUS_DURATION=30           # Minutes in CAUTIOUS mode
V5_DEFENSIVE_DURATION=60          # Minutes in DEFENSIVE mode
V5_ROLLING_WINDOW=20              # Rolling accuracy window size

# -- V4 Model Settings (Optional, used when V5 disabled) --
RETRAIN_INTERVAL_HOURS=6
TRAIN_CANDLES=43200
ENABLE_OPTUNA=true
OPTUNA_TRIALS=40
OPTUNA_TIMEOUT=750

# -- Polymarket (Optional -- enables auto-trading) --
POLYMARKET_PRIVATE_KEY=your_wallet_private_key
POLYMARKET_FUNDER_ADDRESS=your_funder_address
POLYMARKET_SIGNATURE_TYPE=2       # 0 = EOA, 1 = EIP-1271, 2 = Gnosis Safe

# -- Position Redemption (Optional) --
POLYGON_RPC_URL=https://polygon-rpc.com
POLYMARKET_AUTO_REDEEM=true
POLYMARKET_REDEEM_INTERVAL=120
```

---

## 🚀 Deployment

### Local Development

```bash
git clone https://github.com/blinkinfo/aprilxg8.git
cd aprilxg8
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your values

python main.py
```

### 🐳 Docker

```bash
docker build -t aprilxg8 .
docker run -d --env-file .env aprilxg8
```

### 🚂 Railway

The bot is designed for Railway deployment with:
- `Dockerfile` for builds (Python 3.11-slim with gcc/g++ for ML libs)
- Telegram polling conflict retry (handles redeploys gracefully)
- Persistent data in `data/` directory (signal history, autotrade config)
- Model persistence in `models/` directory
- Auto-restart on crash

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `xgboost` | 2.1.4 | Momentum model (V5) + V4 classifier |
| `lightgbm` | 4.6.0 | Mean Reversion model (V5) |
| `catboost` | 1.2.7 | Microstructure model (V5) |
| `scikit-learn` | 1.6.1 | Calibration, CV, metrics |
| `numpy` | 1.26.4 | Numerical computation |
| `pandas` | 2.2.3 | Data handling |
| `optuna` | 4.2.1 | Bayesian hyperparameter optimization |
| `httpx` | 0.28.1 | Async HTTP client (MEXC API) |
| `python-telegram-bot` | 21.10 | Telegram interface |
| `python-dotenv` | 1.0.1 | Environment variable loading |
| `py-clob-client` | >= 0.34.6 | Polymarket CLOB API |
| `web3` | >= 6.14.0 | On-chain redemption (Polygon) |

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

Test suite includes:
- **Feature Engine V2** -- validates all 76 features compute correctly
- **Ensemble Model** -- unit tests for training, prediction, save/load
- **Integration Tests** -- end-to-end pipeline verification

---

## 📜 License

Private repository.
