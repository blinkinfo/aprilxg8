# AprilXG V4 — BTC 5-Min Binary Signal Bot

ML-powered signal bot that predicts the direction of 5-minute BTC candles and optionally auto-trades on [Polymarket](https://polymarket.com) BTC Up/Down binary markets.

## How It Works

1. **~15 seconds before each 5-min candle closes**, the bot fetches multi-timeframe data (5m, 15m, 1h) from MEXC
2. **XGBoost predicts** whether the **next** candle will close Up or Down
3. **Signals above 55% confidence** are posted to Telegram; signals >= 60% are marked **STRONG**
4. **Auto-trade (optional)** places a FOK market order on Polymarket for the predicted direction
5. **30-90 seconds into the next candle**, the bot resolves the signal as WIN or LOSS
6. **Every 6 hours**, the model retrains on ~150 days of historical data with a quality gate
7. **Resolved Polymarket positions** are automatically redeemed on-chain for USDC

### Binary Market Payout

| Outcome | Amount |
|---------|--------|
| Win     | +$0.96 |
| Loss    | -$1.00 |
| Breakeven win rate | ~51.04% |

## Architecture

```
main.py                  Entry point
src/
  bot.py                 Main orchestrator (5s loop, timing, signal flow)
  config.py              4 config dataclasses (MEXC, Model, Telegram,
                         Polymarket) loaded from env vars
  data_fetcher.py        MEXC API client (OHLCV, multi-TF, paginated history)
  features.py            50+ features (RSI, MACD, Bollinger, ATR, ADX, MFI,
                         volume, multi-TF alignment, regime detection, z-scores)
  model.py               XGBoost classifier (Optuna tuning, TimeSeriesSplit CV,
                         retrain gate)
  signal_tracker.py      Signal lifecycle (add, resolve, stats, persistence)
  polymarket_client.py   Polymarket CLOB API (FOK market orders, balance,
                         positions, slug-based market discovery)
  auto_trader.py         Trade orchestrator (safety checks, balance verify,
                         duplicate prevention, slot-targeted execution)
  position_redeemer.py   On-chain position redemption (CTF + NegRisk,
                         Safe wallet, Polygon)
  telegram_bot.py        Telegram interface (12 commands, HTML formatting,
                         conflict retry on redeploy)
  formatters.py          Centralized HTML message formatting
```

## Signal Pipeline

```
[MEXC 5m/15m/1h Data] -> [Feature Engineering] -> [XGBoost Predict]
       |                                                |
       |                    UP/DOWN (>= 55% conf)       |
       |                         |                      |
       v                         v                      v
  [Resolution]           [Telegram Signal]      [Polymarket FOK Trade]
  (30-90s later)         (formatted HTML)       (slot-targeted market order)
                                                        |
                                                        v
                                                [PositionRedeemer]
                                                (on-chain USDC claim)
```

## Polymarket Integration

### Market Discovery

BTC 5-min Up/Down markets follow a deterministic slug pattern:
```
btc-updown-5m-{slot_timestamp}
```
where `slot_timestamp = (unix_time // 300) * 300`. The bot looks up the exact market by slug via the Gamma API — no keyword searching needed.

### Order Execution — FOK Market Orders

Orders use **Fill-or-Kill (FOK)** market orders via the `py-clob-client` SDK:

- **Order type:** `MarketOrderArgs` with `OrderType.FOK`
- **Amount:** USDC to spend (e.g. `1.0` = $1.00)
- **Pricing:** SDK auto-calculates optimal price from the order book
- **Execution:** Fills entirely and immediately, or is rejected — no partial fills, no resting orders
- **Minimum:** ~$1 USDC (vs 5 shares minimum for limit orders)

### Slot-Targeted Trading

Signals include a `target_slot_ts` (Unix timestamp) that flows through the entire pipeline:

1. Bot predicts the **next** candle direction at 16:44:45 UTC
2. Signal targets slot `16:45:00` (ts=1774025100)
3. Polymarket order placed on market `btc-updown-5m-1774025100`
4. Safety check verifies the discovered market matches the target slot

This prevents the bug where a signal for 16:45-16:50 accidentally trades on the 16:40-16:45 market.

### Safety Checks (6 layers)

1. Auto-trading enabled?
2. Direction is UP or DOWN (not NEUTRAL)?
3. Polymarket client initialized?
4. Valid target slot timestamp?
5. Duplicate slot prevention (one trade per slot)
6. Sufficient USDC balance?

### On-Chain Position Redemption

The bot automatically redeems resolved Polymarket positions on Polygon (chain ID 137):

- Scans the Polymarket Data API for redeemable (resolved) positions
- Supports both **CTF** (standard) and **NegRisk Adapter** markets
- **Safe/proxy wallet support** (signature_type=2): wraps calls in `execTransaction`
- EIP-1559 gas pricing with 30% gas buffer
- Session-level dedup to avoid re-redeeming
- Requires a small amount of POL in the EOA wallet for gas (~0.005 POL)

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Labels:** `next_candle_close > next_candle_open` (predicting the NEXT candle)
- **Features:** 50+ engineered features across 3 timeframes
  - Price action: returns (1/3/5/10), log returns, candle body/wick ratios, high/low ratios
  - Moving averages: EMA(9/21) crossover + slopes, SMA(50) ratio
  - Momentum: RSI(14), Stochastic RSI, MACD (normalized to % of price), ADX(14), MFI(14)
  - Volatility: Bollinger Bands (width + %B), ATR(14) normalized, vol ratio (5/20)
  - Volume: OBV ratio, volume ratio, VWAP deviation
  - Multi-timeframe: 15m and 1h trend, RSI, momentum aligned to 5m bars via timestamp mapping
  - Regime detection: ATR percentile, binary high-vol flag, vol expansion/contraction
  - Statistical: Z-scores of momentum, RSI, volume ratio (rolling window)
  - Patterns: Higher highs, lower lows, consecutive green/red candles
  - Lag features: RSI, MACD histogram, and volume ratio at lags 1, 2, 3, 5
- **Training data:** ~43,200 candles (~150 days), paginated fetch for all timeframes
- **Validation:** TimeSeriesSplit cross-validation (5 splits for training, 2 splits for Optuna tuning)
- **Retrain:** Every 6 hours with quality gate (new model must beat old by >= 0.002 accuracy)
- **Optuna:** 40 Bayesian hyperparameter trials every 24 hours with 750s timeout, MedianPruner after 10 startup trials
- **Persistence:** Model saved as pickle (includes params, feature names, accuracy, tune time)

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and chat ID |
| `/help` | Full command reference |
| `/stats` | Performance dashboard (W/L, PnL, streaks) |
| `/recent` | Last 10 signals with outcomes |
| `/status` | Bot status, model info, uptime |
| `/retrain` | Force model retrain now |
| `/autotrade` | Toggle Polymarket auto-trading ON/OFF |
| `/setamount` | Set trade amount in USDC (e.g. `/setamount 2.50`) |
| `/balance` | Check Polymarket USDC wallet balance |
| `/positions` | View open Polymarket positions |
| `/pmstatus` | Full Polymarket connection and config status |
| `/redeem` | Manually trigger position redemption |

## Setup

### Prerequisites

- Python 3.11+
- Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Polymarket wallet with USDC (optional, for auto-trading)
- Small amount of POL for gas (optional, for on-chain redemption)

### Environment Variables

```env
# Required
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading (optional)
TRADING_SYMBOL=BTCUSDT                # Default: BTCUSDT
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR

# Model settings (optional)
PREDICTION_THRESHOLD=0.52              # Base prediction threshold
CONFIDENCE_MIN=0.55                    # Minimum confidence to emit signal
RETRAIN_INTERVAL_HOURS=6               # Hours between retrains
LOOKBACK_CANDLES=100                   # Recent candles for live feature engineering
TRAIN_CANDLES=43200                    # ~150 days of 5m candles for training
RETRAIN_MIN_IMPROVEMENT=0.002          # Quality gate for model swap
ENABLE_OPTUNA=true                     # Enable hyperparameter tuning
OPTUNA_TRIALS=40                       # Bayesian optimization trials
OPTUNA_TIMEOUT=750                     # Optuna timeout in seconds

# Polymarket (optional — enables auto-trading)
POLYMARKET_PRIVATE_KEY=your_wallet_private_key
POLYMARKET_FUNDER_ADDRESS=your_funder_address
POLYMARKET_SIGNATURE_TYPE=2            # 0 = EOA, 1 = EIP-1271, 2 = Gnosis Safe

# Position Redemption (optional — requires POLYMARKET_PRIVATE_KEY)
POLYGON_RPC_URL=https://polygon-rpc.com   # Use Alchemy/Infura for reliability
POLYMARKET_AUTO_REDEEM=true                # Enable automatic redemption
POLYMARKET_REDEEM_INTERVAL=120             # Scan interval in seconds
```

### Local Development

```bash
git clone https://github.com/blinkinfo/aprilxg4.git
cd aprilxg4
pip install -r requirements.txt

# Set environment variables (or use .env file)
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...

python main.py
```

### Docker

```bash
docker build -t aprilxg4 .
docker run -d --env-file .env aprilxg4
```

### Railway Deployment

The bot is designed for Railway with:
- `Dockerfile` for builds (Python 3.11-slim)
- Auto-restart on crash (max 5 retries)
- Telegram polling conflict retry (handles redeploys gracefully)
- Persistent data in `data/` directory (signal history, autotrade config)
- Model persistence in `models/` directory

## Dependencies

```
xgboost==2.1.4          # ML model
scikit-learn==1.6.1     # Preprocessing, CV
numpy==1.26.4           # Numerical
pandas==2.2.3           # Data handling
optuna==4.2.1           # Hyperparameter optimization
httpx==0.28.1           # Async HTTP client
python-telegram-bot==21.10  # Telegram interface
python-dotenv==1.0.1    # Env var loading
py-clob-client>=0.34.6  # Polymarket CLOB API
web3>=6.14.0            # On-chain redemption (Polygon)
```

## License

Private repository.
