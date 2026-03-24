"""Feature engineering for BTC price prediction.

Improvements over original:
- Improvement 3: All raw price-scale features normalized to percentages/z-scores
- Improvement 6: Volatility regime detection (ATR percentile)

Fixes applied (v5 — accuracy truthfulness overhaul):
- Fix C: Added ffill parameter to compute_features() for NaN handling
  consistency between training and inference. When ffill=True, NaN values
  from indicator warm-up are forward-filled instead of dropping rows.
- Fix D: Labels and features use robust index alignment via intersection.
"""
import logging

import numpy as np
import pandas as pd

from .config import ModelConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes technical indicators and features from OHLCV data."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def compute_features(
        self,
        df: pd.DataFrame,
        higher_tf_data: dict[str, pd.DataFrame] | None = None,
        ffill: bool = False,
    ) -> pd.DataFrame:
        """Compute all features from OHLCV data.

        Args:
            df: DataFrame with open, high, low, close, volume columns
            higher_tf_data: Optional dict of higher timeframe DataFrames for multi-TF features
            ffill: If True, forward-fill NaN values instead of dropping rows.
                   Use ffill=True for both training and inference to ensure
                   consistent NaN handling (Fix C).

        Returns:
            DataFrame with all computed features (NaN handled per ffill param).
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for feature computation")
            return pd.DataFrame()

        feat = df.copy()

        # --- Price Action Features (all normalized as pct of price) ---
        feat["returns_1"] = feat["close"].pct_change(1)
        feat["returns_3"] = feat["close"].pct_change(3)
        feat["returns_5"] = feat["close"].pct_change(5)
        feat["returns_10"] = feat["close"].pct_change(10)

        feat["candle_body"] = (feat["close"] - feat["open"]) / feat["open"]
        feat["upper_wick"] = (feat["high"] - feat[["open", "close"]].max(axis=1)) / feat["open"]
        feat["lower_wick"] = (feat[["open", "close"]].min(axis=1) - feat["low"]) / feat["open"]
        feat["candle_range"] = (feat["high"] - feat["low"]) / feat["open"]

        # --- Trend Indicators ---
        # RSI
        delta = feat["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.config.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi"] = 100 - (100 / (1 + rs))
        feat["rsi_norm"] = feat["rsi"] / 100.0  # 0-1 normalized

        # MACD (normalized as pct of price)
        ema_fast = feat["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = feat["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config.macd_signal, adjust=False).mean()
        feat["macd_pct"] = macd_line / feat["close"]
        feat["macd_signal_pct"] = signal_line / feat["close"]
        feat["macd_hist_pct"] = (macd_line - signal_line) / feat["close"]

        # Bollinger Bands (normalized)
        bb_mid = feat["close"].rolling(self.config.bb_period).mean()
        bb_std = feat["close"].rolling(self.config.bb_period).std()
        feat["bb_pctb"] = (feat["close"] - (bb_mid - self.config.bb_std * bb_std)) / (
            (2 * self.config.bb_std * bb_std).replace(0, np.nan)
        )
        feat["bb_width"] = (2 * self.config.bb_std * bb_std) / bb_mid.replace(0, np.nan)

        # ATR (normalized as pct of price)
        high_low = feat["high"] - feat["low"]
        high_close = (feat["high"] - feat["close"].shift(1)).abs()
        low_close = (feat["low"] - feat["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(self.config.atr_period).mean()
        feat["atr_pct"] = atr / feat["close"]

        # Stochastic Oscillator
        low_n = feat["low"].rolling(self.config.stoch_period).min()
        high_n = feat["high"].rolling(self.config.stoch_period).max()
        feat["stoch_k"] = ((feat["close"] - low_n) / (high_n - low_n).replace(0, np.nan)) * 100
        feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()
        feat["stoch_k_norm"] = feat["stoch_k"] / 100.0
        feat["stoch_d_norm"] = feat["stoch_d"] / 100.0

        # MFI (Money Flow Index)
        typical_price = (feat["high"] + feat["low"] + feat["close"]) / 3
        money_flow = typical_price * feat["volume"]
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0.0)
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0.0)
        pos_mf = pos_flow.rolling(self.config.mfi_period).sum()
        neg_mf = neg_flow.rolling(self.config.mfi_period).sum()
        mfi_ratio = pos_mf / neg_mf.replace(0, np.nan)
        feat["mfi"] = 100 - (100 / (1 + mfi_ratio))
        feat["mfi_norm"] = feat["mfi"] / 100.0

        # ADX (Average Directional Index)
        plus_dm = feat["high"].diff()
        minus_dm = -feat["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr_adx = true_range.rolling(self.config.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.config.adx_period).mean() / atr_adx.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(self.config.adx_period).mean() / atr_adx.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        feat["adx"] = dx.rolling(self.config.adx_period).mean()
        feat["adx_norm"] = feat["adx"] / 100.0

        # EMA crossover
        ema_f = feat["close"].ewm(span=self.config.ema_fast, adjust=False).mean()
        ema_s = feat["close"].ewm(span=self.config.ema_slow, adjust=False).mean()
        feat["ema_crossover"] = (ema_f - ema_s) / feat["close"]

        # --- Volume Features ---
        feat["volume_sma20"] = feat["volume"].rolling(20).mean()
        feat["volume_ratio"] = feat["volume"] / feat["volume_sma20"].replace(0, np.nan)
        feat["volume_change"] = feat["volume"].pct_change(1)

        # --- Momentum / Z-score Features ---
        for lookback in [5, 10, 20]:
            roll_mean = feat["close"].pct_change(1).rolling(lookback).mean()
            roll_std = feat["close"].pct_change(1).rolling(lookback).std()
            feat[f"momentum_{lookback}_zscore"] = (
                (feat["close"].pct_change(1) - roll_mean) / roll_std.replace(0, np.nan)
            )

        # --- Lag Features ---
        feat["rsi_lag1"] = feat["rsi_norm"].shift(1)
        feat["rsi_lag3"] = feat["rsi_norm"].shift(3)
        feat["rsi_lag5"] = feat["rsi_norm"].shift(5)
        feat["macd_hist_lag1"] = feat["macd_hist_pct"].shift(1)
        feat["volume_ratio_lag1"] = feat["volume_ratio"].shift(1)

        # --- Volatility Regime Detection (Improvement 6) ---
        lookback = self.config.atr_regime_lookback
        feat["atr_regime"] = feat["atr_pct"].rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # --- SMA features ---
        feat["sma_10_dist"] = (feat["close"] - feat["close"].rolling(10).mean()) / feat["close"]
        feat["sma_20_dist"] = (feat["close"] - feat["close"].rolling(20).mean()) / feat["close"]
        feat["sma_50_dist"] = (feat["close"] - feat["close"].rolling(50).mean()) / feat["close"]

        # --- Higher Timeframe Features (multi-TF alignment) ---
        if higher_tf_data:
            for tf_label, tf_df in higher_tf_data.items():
                if tf_df.empty or len(tf_df) < 20:
                    continue

                prefix = tf_label.replace(" ", "_").lower()

                # RSI on higher TF
                delta_htf = tf_df["close"].diff()
                gain_htf = delta_htf.where(delta_htf > 0, 0.0).rolling(14).mean()
                loss_htf = (-delta_htf.where(delta_htf < 0, 0.0)).rolling(14).mean()
                rs_htf = gain_htf / loss_htf.replace(0, np.nan)
                htf_rsi = (100 - (100 / (1 + rs_htf))) / 100.0

                # MACD on higher TF
                ema_f_htf = tf_df["close"].ewm(span=12, adjust=False).mean()
                ema_s_htf = tf_df["close"].ewm(span=26, adjust=False).mean()
                htf_macd = (ema_f_htf - ema_s_htf) / tf_df["close"]

                # EMA crossover on higher TF
                ema9_htf = tf_df["close"].ewm(span=9, adjust=False).mean()
                ema21_htf = tf_df["close"].ewm(span=21, adjust=False).mean()
                htf_ema_cross = (ema9_htf - ema21_htf) / tf_df["close"]

                # Map to 5m index via forward-fill (last known HTF value)
                htf_features = pd.DataFrame({
                    f"{prefix}_rsi": htf_rsi,
                    f"{prefix}_macd": htf_macd,
                    f"{prefix}_ema_cross": htf_ema_cross,
                }, index=tf_df.index)

                # Reindex to 5m timestamps using forward-fill
                if hasattr(feat.index, 'freq') or feat.index.dtype == 'datetime64[ns]':
                    htf_reindexed = htf_features.reindex(feat.index, method="ffill")
                else:
                    # Integer index: merge on nearest timestamp if available
                    htf_reindexed = htf_features.reindex(
                        range(len(feat)), method="ffill"
                    )

                for col in htf_reindexed.columns:
                    feat[col] = htf_reindexed[col].values[:len(feat)]

        # --- Select feature columns only (drop OHLCV and intermediate columns) ---
        feature_cols = [
            col for col in feat.columns
            if col not in ["open", "high", "low", "close", "volume",
                           "close_time", "quote_volume", "timestamp",
                           "volume_sma20"]
        ]
        result = feat[feature_cols].copy()

        # --- NaN handling (Fix C) ---
        if ffill:
            # Forward-fill: use last known value for each feature.
            # This is semantically correct for time series ("use the most
            # recent observation") and ensures training/inference consistency.
            result = result.ffill()
            # Drop only the leading rows where ffill can't help
            # (the very first rows before any indicator has a value).
            result = result.dropna()
        else:
            # Legacy behavior: drop all rows with any NaN.
            # Used only by backtester which manages its own windowing.
            result = result.dropna()

        if result.empty:
            logger.warning("All rows dropped after NaN handling")

        return result

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create binary labels: 1 if next candle closes above its open (green), 0 otherwise.

        The label for row i indicates the direction of candle i+1.
        This means the LAST row will have NaN label (no next candle to observe).

        Returns:
            pd.Series of 0/1 labels with the same index as df.
        """
        labels = (df["close"].shift(-1) > df["open"].shift(-1)).astype(float)
        # Last row has no next candle — mark as NaN so it gets excluded
        # during index alignment in _prepare_data()
        labels.iloc[-1] = np.nan
        return labels
