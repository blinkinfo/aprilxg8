"""Regime detection for market state classification.

Phase 2 of the AprilXG V5 multi-model ensemble upgrade.
Classifies market into 4 regimes for model routing and weight assignment.

Regimes:
    TRENDING_UP   (0) — Strong uptrend (ADX > 25, EMA9 > EMA21)
    TRENDING_DOWN (1) — Strong downtrend (ADX > 25, EMA9 < EMA21)
    RANGING       (2) — Low volatility, sideways (ADX < 20)
    VOLATILE      (3) — High volatility, no clear trend (ADX 20-25 or ATR spike)
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    REGIME_NAMES = {
        0: "TRENDING_UP",
        1: "TRENDING_DOWN",
        2: "RANGING",
        3: "VOLATILE",
    }

    # Regime weight matrix: {regime: {model_name: weight}}
    # Weights sum to 1.0 for each regime.
    REGIME_WEIGHTS = {
        0: {"momentum": 0.50, "mean_reversion": 0.10, "microstructure": 0.40},  # TRENDING_UP
        1: {"momentum": 0.50, "mean_reversion": 0.10, "microstructure": 0.40},  # TRENDING_DOWN
        2: {"momentum": 0.10, "mean_reversion": 0.50, "microstructure": 0.40},  # RANGING
        3: {"momentum": 0.25, "mean_reversion": 0.25, "microstructure": 0.50},  # VOLATILE
    }

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
        n = len(features)
        regimes = np.full(n, self.VOLATILE, dtype=np.int32)  # default: VOLATILE

        # Extract required features
        adx_raw = features["adx_10"].values * 100  # denormalize to 0-100 scale
        ema_cross = features["ema_cross"].values
        atr_ratio = features["atr_ratio"].values
        bb_squeeze = features["bb_squeeze"].values

        # Classification logic (evaluated in priority order)
        # 1. Strong trend: ADX > 25
        strong_trend = adx_raw > 25
        up_trend = ema_cross > 0
        down_trend = ema_cross < 0

        trending_up_mask = strong_trend & up_trend
        trending_down_mask = strong_trend & down_trend

        # 2. Ranging: ADX < 20 (low directional movement)
        ranging_mask = adx_raw < 20

        # 3. Volatile: ADX 20-25, or ATR spike (atr_ratio > 1.2 without clear trend)
        #    This is the default — anything not trending or ranging is volatile.
        #    Also classify as volatile if ATR is spiking even with low ADX.
        volatile_spike = (atr_ratio > 1.2) & ~strong_trend

        # Apply in priority order: trending > ranging > volatile
        # Volatile is default, so we override with more specific classifications
        regimes[ranging_mask] = self.RANGING
        # Volatile spike overrides ranging (high vol takes priority over low ADX)
        regimes[volatile_spike] = self.VOLATILE
        # Trending overrides everything
        regimes[trending_up_mask] = self.TRENDING_UP
        regimes[trending_down_mask] = self.TRENDING_DOWN

        result = pd.Series(regimes, index=features.index, name="regime")

        # Log regime distribution
        unique, counts = np.unique(regimes, return_counts=True)
        dist = {self.REGIME_NAMES.get(r, str(r)): int(c) for r, c in zip(unique, counts)}
        logger.info(f"Regime distribution: {dist} (n={n})")

        return result

    def get_regime_weights(self, regime: int) -> dict[str, float]:
        """Return model weights for a given regime.

        Returns:
            {"momentum": w1, "mean_reversion": w2, "microstructure": w3}
            Weights sum to 1.0.
        """
        if regime not in self.REGIME_WEIGHTS:
            logger.warning(f"Unknown regime {regime}, defaulting to VOLATILE weights")
            return self.REGIME_WEIGHTS[self.VOLATILE]
        return self.REGIME_WEIGHTS[regime]

    def get_regime_name(self, regime: int) -> str:
        """Return human-readable name for a regime."""
        return self.REGIME_NAMES.get(regime, f"UNKNOWN({regime})")
