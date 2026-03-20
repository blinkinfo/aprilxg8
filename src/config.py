"""Configuration module for BTC Signal Bot."""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum training candles to ensure meaningful model training.
# 10,000 5m candles ~ 34.7 days -- absolute floor for feature engineering
# to produce enough valid samples after NaN drop.
MIN_TRAIN_CANDLES = 10000


@dataclass
class MEXCConfig:
    """MEXC API configuration."""
    base_url: str = "https://api.mexc.com"
    klines_endpoint: str = "/api/v3/klines"
    symbol: str = "BTCUSDT"
    intervals: dict = field(default_factory=lambda: {
        "5m": "5m",
        "15m": "15m",
        "1h": "60m",
        "4h": "4h",
        "1d": "1d",
    })
    max_klines: int = 500  # MEXC max per request
    request_timeout: int = 15
    rate_limit_delay: float = 0.12  # ~8 req/s to stay under 10/s limit


@dataclass
class ModelConfig:
    """ML model configuration."""
    lookback_candles: int = 100  # Number of candles to use as features
    train_candles: int = 43200  # ~150 days of 5m candles (was 5000)
    retrain_interval_hours: int = 6  # Retrain every N hours
    prediction_threshold: float = 0.55  # Minimum confidence to emit signal (raised from 0.52)
    # Feature engineering
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stoch_period: int = 14
    mfi_period: int = 14
    adx_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    # Confidence filtering (Improvement 2)
    confidence_min: float = 0.55  # Skip trades below this confidence
    confidence_strong: float = 0.60  # Label as "strong" signal above this
    # Regime detection (Improvement 6)
    atr_regime_lookback: int = 100  # Lookback period for ATR percentile
    # Optuna tuning (Improvement 5)
    enable_optuna_tuning: bool = True
    optuna_n_trials: int = 30  # Number of Bayesian optimization trials
    optuna_tune_interval_hours: int = 24  # Re-tune hyperparams every N hours
    # Walk-forward retraining gate (Improvement 4)
    retrain_min_improvement: float = 0.002  # New model must beat old by this margin
    # XGBoost default params (may be overridden by Optuna)
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    })


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = ""
    chat_id: str = ""
    signal_cooldown_seconds: int = 30  # Min time between signals
    max_message_length: int = 4096


@dataclass
class BotConfig:
    """Main bot configuration."""
    mexc: MEXCConfig = field(default_factory=MEXCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    # Timing
    prediction_lead_seconds: int = 15  # Predict N seconds before candle close
    main_loop_interval: int = 5  # Check every N seconds
    # Data storage
    data_dir: str = "data"
    model_dir: str = "models"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.telegram.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        config.telegram.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        config.log_level = os.environ.get("LOG_LEVEL", "INFO")
        config.mexc.symbol = os.environ.get("TRADING_SYMBOL", "BTCUSDT")

        # Optional model overrides
        if os.environ.get("PREDICTION_THRESHOLD"):
            config.model.prediction_threshold = float(os.environ["PREDICTION_THRESHOLD"])
        if os.environ.get("RETRAIN_INTERVAL_HOURS"):
            config.model.retrain_interval_hours = int(os.environ["RETRAIN_INTERVAL_HOURS"])
        if os.environ.get("LOOKBACK_CANDLES"):
            config.model.lookback_candles = int(os.environ["LOOKBACK_CANDLES"])
        if os.environ.get("CONFIDENCE_MIN"):
            config.model.confidence_min = float(os.environ["CONFIDENCE_MIN"])
        if os.environ.get("ENABLE_OPTUNA"):
            config.model.enable_optuna_tuning = os.environ["ENABLE_OPTUNA"].lower() in ("1", "true", "yes")
        if os.environ.get("OPTUNA_TRIALS"):
            config.model.optuna_n_trials = int(os.environ["OPTUNA_TRIALS"])
        if os.environ.get("TRAIN_CANDLES"):
            config.model.train_candles = int(os.environ["TRAIN_CANDLES"])

        # Enforce minimum training candles to prevent under-training
        if config.model.train_candles < MIN_TRAIN_CANDLES:
            logger.warning(
                f"train_candles={config.model.train_candles} is below minimum "
                f"{MIN_TRAIN_CANDLES}. Overriding to {MIN_TRAIN_CANDLES}."
            )
            config.model.train_candles = MIN_TRAIN_CANDLES

        # Log final resolved config for diagnostics
        logger.info(
            f"Config loaded: train_candles={config.model.train_candles} "
            f"(~{config.model.train_candles * 5 // 1440} days of 5m data), "
            f"retrain_interval={config.model.retrain_interval_hours}h, "
            f"confidence_min={config.model.confidence_min}, "
            f"optuna={'ON' if config.model.enable_optuna_tuning else 'OFF'}"
        )

        return config
