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
    # ---------------------------------------------------------------
    # Confidence & EV filtering (Improvement: Calibration + EV-based)
    # ---------------------------------------------------------------
    # Raw confidence floor — predictions below this are always skipped
    # even before calibration (saves compute on clearly garbage predictions).
    confidence_min: float = 0.52
    # EV-based filtering (replaces old confidence_strong threshold).
    # Expected Value = (calibrated_prob * win_payout) - ((1 - calibrated_prob) * loss_amount)
    # Default 0.0 means: only trade when EV >= 0 (i.e. positive expected value).
    ev_threshold: float = 0.0
    # EV above this marks the signal as "STRONG" (higher edge).
    ev_strong_threshold: float = 0.05
    # Binary market payout structure
    win_payout: float = 0.96   # Profit on a win ($)
    loss_amount: float = 1.00  # Loss on a loss ($)
    # ---------------------------------------------------------------
    # Probability calibration (Improvement 1)
    # ---------------------------------------------------------------
    enable_calibration: bool = True  # Fit isotonic regression on OOS split
    # ---------------------------------------------------------------
    # Feature pruning (Improvement 3)
    # ---------------------------------------------------------------
    enable_feature_pruning: bool = True  # Prune low-importance features
    feature_prune_top_n: int = 20        # Keep top N features by importance
    # Regime detection (Improvement 6)
    atr_regime_lookback: int = 100  # Lookback period for ATR percentile
    # Optuna tuning (Improvement 5)
    enable_optuna_tuning: bool = True
    optuna_n_trials: int = 40  # Bayesian optimization trials
    optuna_tune_interval_hours: int = 24  # Re-tune hyperparams every N hours
    optuna_timeout_seconds: int = 750  # Hard cap per tuning session (~12.5 min)
    # Walk-forward retraining gate (Improvement 4)
    retrain_min_improvement: float = 0.002  # New model must beat old by this margin
    # XGBoost default params (may be overridden by Optuna)
    # NOTE: use_label_encoder was removed — it is deprecated in xgboost >= 1.7
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
class EnsembleConfig:
    """V5 Ensemble configuration."""
    use_v5_ensemble: bool = field(default_factory=lambda: os.getenv("USE_V5_ENSEMBLE", "true").lower() == "true")

    # Training
    train_candles: int = field(default_factory=lambda: int(os.getenv("V5_TRAIN_CANDLES", "20000")))  # ~70 days, fresher data
    retrain_interval_hours: int = field(default_factory=lambda: int(os.getenv("V5_RETRAIN_HOURS", "2")))
    sample_weight_recent_multiplier: float = field(default_factory=lambda: float(os.getenv("V5_RECENT_WEIGHT", "3.0")))
    recent_window_frac: float = field(default_factory=lambda: float(os.getenv("V5_RECENT_WINDOW_FRAC", "0.33")))  # Last 33% of training data gets weight multiplier

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
    cautious_accuracy_threshold: float = field(default_factory=lambda: float(os.getenv("V5_CAUTIOUS_ACCURACY", "0.48")))
    defensive_accuracy_threshold: float = field(default_factory=lambda: float(os.getenv("V5_DEFENSIVE_ACCURACY", "0.42")))
    cautious_duration_minutes: int = field(default_factory=lambda: int(os.getenv("V5_CAUTIOUS_DURATION", "30")))
    defensive_duration_minutes: int = field(default_factory=lambda: int(os.getenv("V5_DEFENSIVE_DURATION", "60")))
    rolling_window: int = field(default_factory=lambda: int(os.getenv("V5_ROLLING_WINDOW", "20")))

    # Quality gate
    min_oos_accuracy: float = field(default_factory=lambda: float(os.getenv("V5_MIN_OOS_ACC", "0.53")))

    # Model save path
    model_dir: str = field(default_factory=lambda: os.getenv("V5_MODEL_DIR", "data/ensemble_model"))


@dataclass
class PolymarketConfig:
    """Polymarket auto-trading configuration."""
    private_key: str = ""          # Wallet private key (hex)
    funder_address: str = ""       # Funder/proxy wallet address
    signature_type: int = 2        # Signature type (0, 1, or 2)
    enabled: bool = False           # Derived: True if private_key is set
    # Position redemption settings
    polygon_rpc_url: str = ""      # Polygon RPC URL (defaults to public RPC if empty)
    auto_redeem: bool = True       # Automatically redeem resolved positions
    redeem_check_interval: int = 120  # Seconds between redemption scans


@dataclass
class BotConfig:
    """Main bot configuration."""
    mexc: MEXCConfig = field(default_factory=MEXCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    # Timing
    prediction_lead_seconds: int = 90  # Predict N seconds before candle close
    main_loop_interval: int = 5  # Check every N seconds
    # Data storage
    data_dir: str = "data"
    model_dir: str = "models"
    log_level: str = "INFO"
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

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
        if os.environ.get("OPTUNA_TIMEOUT"):
            config.model.optuna_timeout_seconds = int(os.environ["OPTUNA_TIMEOUT"])
        if os.environ.get("TRAIN_CANDLES"):
            config.model.train_candles = int(os.environ["TRAIN_CANDLES"])
        # EV-based filtering
        if os.environ.get("EV_THRESHOLD"):
            config.model.ev_threshold = float(os.environ["EV_THRESHOLD"])
        if os.environ.get("EV_STRONG_THRESHOLD"):
            config.model.ev_strong_threshold = float(os.environ["EV_STRONG_THRESHOLD"])
        # Calibration
        if os.environ.get("ENABLE_CALIBRATION"):
            config.model.enable_calibration = os.environ["ENABLE_CALIBRATION"].lower() in ("1", "true", "yes")
        # Feature pruning
        if os.environ.get("ENABLE_FEATURE_PRUNING"):
            config.model.enable_feature_pruning = os.environ["ENABLE_FEATURE_PRUNING"].lower() in ("1", "true", "yes")
        if os.environ.get("FEATURE_PRUNE_TOP_N"):
            config.model.feature_prune_top_n = int(os.environ["FEATURE_PRUNE_TOP_N"])
        # Payout structure
        if os.environ.get("WIN_PAYOUT"):
            config.model.win_payout = float(os.environ["WIN_PAYOUT"])
        if os.environ.get("LOSS_AMOUNT"):
            config.model.loss_amount = float(os.environ["LOSS_AMOUNT"])

        # Enforce minimum training candles to prevent under-training
        if config.model.train_candles < MIN_TRAIN_CANDLES:
            logger.warning(
                f"train_candles={config.model.train_candles} is below minimum "
                f"{MIN_TRAIN_CANDLES}. Overriding to {MIN_TRAIN_CANDLES}."
            )
            config.model.train_candles = MIN_TRAIN_CANDLES

        # Polymarket configuration (all optional — bot works without them)
        config.polymarket.private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        config.polymarket.funder_address = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
        if os.environ.get("POLYMARKET_SIGNATURE_TYPE"):
            config.polymarket.signature_type = int(os.environ["POLYMARKET_SIGNATURE_TYPE"])
        # Position redemption settings
        config.polymarket.polygon_rpc_url = os.environ.get("POLYGON_RPC_URL", "")
        if os.environ.get("POLYMARKET_AUTO_REDEEM"):
            config.polymarket.auto_redeem = os.environ["POLYMARKET_AUTO_REDEEM"].lower() in ("1", "true", "yes")
        if os.environ.get("POLYMARKET_REDEEM_INTERVAL"):
            config.polymarket.redeem_check_interval = int(os.environ["POLYMARKET_REDEEM_INTERVAL"])
        # Derived flag: Polymarket is enabled if private key is provided
        config.polymarket.enabled = bool(config.polymarket.private_key)

        # Log final resolved config for diagnostics
        pm_status = "ON" if config.polymarket.enabled else "OFF"
        logger.info(
            f"Config loaded: train_candles={config.model.train_candles} "
            f"(~{config.model.train_candles * 5 // 1440} days of 5m data), "
            f"retrain_interval={config.model.retrain_interval_hours}h, "
            f"confidence_min={config.model.confidence_min}, "
            f"ev_threshold={config.model.ev_threshold}, "
            f"calibration={'ON' if config.model.enable_calibration else 'OFF'}, "
            f"feature_pruning={'ON' if config.model.enable_feature_pruning else 'OFF'} "
            f"(top {config.model.feature_prune_top_n}), "
            f"optuna={'ON' if config.model.enable_optuna_tuning else 'OFF'} "
            f"({config.model.optuna_n_trials} trials, {config.model.optuna_timeout_seconds}s timeout), "
            f"polymarket={pm_status}, "
            f"auto_redeem={'ON' if config.polymarket.auto_redeem else 'OFF'}"
        )

        return config
