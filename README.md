# AprilXG v2 — BTC 5-Min Binary Signal Bot

ML-powered signal bot that predicts the direction of 5-minute BTC candles and optionally auto-trades on [Polymarket](https://polymarket.com) BTC Up/Down binary markets.

## How It Works

1. **~15 seconds before each 5-min candle closes**, the bot fetches multi-timeframe data (5m, 15m, 1h) from MEXC
2. **XGBoost predicts** whether the **next** candle will close Up or Down
3. **Signals above 55% confidence** are posted to Telegram; signals >= 60% are marked **STRONG**
4. **Auto-trade (optional)** places a FOK market order on Polymarket for the predicted direction
5. **30-90 seconds into the next candle**, the bot resolves the signal as WIN or LOSS
6. **Every 6 hours**, the model retrains on ~150 days of historical data with a quality gate

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
  data_fetcher.py        MEXC API client (OHLCV, multi-TF, paginated history)
  features.py            50+ features (RSI, MACD, Bollinger, ATR, volume,
                         multi-TF alignment, regime detection, z-scores)
  model.py               XGBoost classifier (Optuna tuning, TimeSeriesSplit CV,
                         retrain gate)
  signal_tracker.py      Signal lifecycle (add, resolve, stats, persistence)
  polymarket_client.py   Polymarket CLOB API (FOK market orders, balance,
                         positions, slug-based market discovery)
  auto_trader.py         Trade orchestrator (safety checks, balance verify,
                         duplicate prevention, slot-targeted execution)
  telegram_bot.py        Telegram interface (11 commands, HTML formatting,
                         conflict retry on redeploy)
  formatters.py          Centralized HTML message formatting
  config.py              4 config dataclasses (Bot, MEXC, Model, Telegram,
                         Polymarket) loaded from env vars
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

## Model Details

- **Algorithm:** XGBoost binary classifier
- **Labels:** `next_candle_close > next_candle_open` (predicting the NEXT candle)
- **Features:** 50+ engineered features across 3 timeframes
  - Price action: returns, log returns, candle body/wick ratios
  - Momentum: RSI (14), MACD, Stochastic %K/%D
  - Volatility: Bollinger Bands width/position, ATR (14)
  - Volume: OBV, volume ratio, VWAP deviation
  - Multi-timeframe: 15m and 1h trend, RSI, momentum aligned to 5m bars
  - Regime detection: ATR percentile, volatility scoring, high-vol flag
  - Statistical: Z-scores of price, volume, and returns
- **Training data:** ~43,200 candles (~150 days), paginated fetch for all timeframes
- **Validation:** TimeSeriesSplit cross-validation (5 splits)
- **Retrain:** Every 6 hours with quality gate (new model must beat old by >= 0.002 accuracy)
- **Optuna:** 30 hyperparameter trials every 24 hours (depth, learning rate, subsample, etc.)

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

## Setup

### Prerequisites

- Python 3.11+
- Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Polymarket wallet with USDC (optional, for auto-trading)

### Environment Variables

```env
# Required
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Polymarket (optional — enables auto-trading)
POLYMARKET_PRIVATE_KEY=your_wallet_private_key
POLYMARKET_FUNDER_ADDRESS=your_funder_address
POLYMARKET_SIGNATURE_TYPE=2          # Optional, default: 2

# Model tuning (optional)
TRAIN_CANDLES=43200                   # ~150 days of 5m candles
CONFIDENCE_MIN=0.55                   # Minimum confidence to emit signal
RETRAIN_INTERVAL_HOURS=6
RETRAIN_MIN_IMPROVEMENT=0.002
ENABLE_OPTUNA=true
OPTUNA_TRIALS=30

# General
LOG_LEVEL=INFO
```

### Local Development

```bash
git clone https://github.com/blinkinfo/aprilxg2.git
cd aprilxg2
pip install -r requirements.txt

# Set environment variables (or use .env file)
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...

python main.py
```

### Docker

```bash
docker build -t aprilxg2 .
docker run -d --env-file .env aprilxg2
```

### Railway Deployment

The bot is designed for Railway with:
- `Dockerfile` for builds
- Auto-restart on crash
- Telegram polling conflict retry (handles redeploys gracefully)
- Persistent data in `data/` directory (signal history, model, autotrade config)

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
web3>=6.14.0            # Ethereum signing
```

## License

Private repository.
