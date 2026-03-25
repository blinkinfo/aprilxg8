"""Multi-model ensemble with regime-aware weighting.

Phase 2 of the AprilXG V5 multi-model ensemble upgrade.

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
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .config import ModelConfig
from .features_v2 import FeatureEngineV2
from .regime import RegimeDetector

logger = logging.getLogger(__name__)

# Purge gap between splits (same as V4)
PURGE_GAP = 20

# Minimum samples required to train a regime-specific model
MIN_REGIME_SAMPLES = 200

# Fallback: if a regime-specific model can't be trained, use microstructure
# model which trains on all data.


def _safe_import_lightgbm():
    """Import LightGBM with graceful error handling."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        logger.error("lightgbm not installed. Run: pip install lightgbm==4.6.0")
        raise


def _safe_import_catboost():
    """Import CatBoost with graceful error handling."""
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        logger.error("catboost not installed. Run: pip install catboost==1.2.7")
        raise


class EnsembleModel:
    """Multi-model ensemble with regime-aware weighting."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.momentum_model = None       # XGBoost
        self.mean_reversion_model = None  # LightGBM
        self.microstructure_model = None  # CatBoost
        self.calibrators = {}             # {regime_id: calibrator}
        self.calibrator_types = {}        # {regime_id: "isotonic" | "platt" | "passthrough"}
        self.feature_engine = FeatureEngineV2(config)
        self.feature_names = {
            "momentum": [],
            "mean_reversion": [],
            "microstructure": [],
        }
        self.training_stats = {}
        self.is_trained = False
        self.last_train_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    @staticmethod
    def _create_labels(df: pd.DataFrame) -> pd.Series:
        """Create binary labels: 1 if next candle close > open, else 0."""
        next_close = df["close"].shift(-1)
        next_open = df["open"].shift(-1)
        labels = (next_close > next_open).astype(int)
        # Last row has no next candle — will be NaN after shift
        labels.iloc[-1] = np.nan
        return labels

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    def _tune_momentum(self, X: pd.DataFrame, y: pd.Series,
                       n_trials: int = 15, timeout: int = 300) -> dict:
        """Optuna tuning for Momentum Model (XGBoost)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, using default params")
            return self._default_momentum_params()

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }
            return self._optuna_cv_objective(X, y, XGBClassifier, params, trial)

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best.update({"objective": "binary:logistic", "eval_metric": "logloss",
                     "random_state": 42, "n_jobs": -1})
        logger.info(f"Momentum tuning: {len(study.trials)} trials, best CV={study.best_value:.4f}")
        return best

    def _tune_mean_reversion(self, X: pd.DataFrame, y: pd.Series,
                             n_trials: int = 15, timeout: int = 300) -> dict:
        """Optuna tuning for Mean Reversion Model (LightGBM)."""
        LGBMClassifier = _safe_import_lightgbm()
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, using default params")
            return self._default_mean_reversion_params()

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "verbose": -1,
                "random_state": 42,
                "n_jobs": -1,
            }
            return self._optuna_cv_objective(X, y, LGBMClassifier, params, trial)

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best.update({"verbose": -1, "random_state": 42, "n_jobs": -1})
        logger.info(f"Mean reversion tuning: {len(study.trials)} trials, best CV={study.best_value:.4f}")
        return best

    def _tune_microstructure(self, X: pd.DataFrame, y: pd.Series,
                             n_trials: int = 15, timeout: int = 300) -> dict:
        """Optuna tuning for Microstructure Model (CatBoost)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, using default params")
            return self._default_microstructure_params()

        CatBoostClassifier = _safe_import_catboost()

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 4, 8),
                "iterations": trial.suggest_int("iterations", 200, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "verbose": 0,
                "random_seed": 42,
                "thread_count": -1,
                "allow_writing_files": False,
            }
            return self._optuna_cv_objective_catboost(X, y, params, trial)

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best.update({"verbose": 0, "random_seed": 42, "thread_count": -1,
                     "allow_writing_files": False})
        logger.info(f"Microstructure tuning: {len(study.trials)} trials, best CV={study.best_value:.4f}")
        return best

    def _optuna_cv_objective(self, X: pd.DataFrame, y: pd.Series,
                             model_cls, params: dict, trial) -> float:
        """Shared CV objective for XGBoost and LightGBM."""
        import optuna

        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(train_idx) > PURGE_GAP:
                train_idx = train_idx[:-PURGE_GAP]
            else:
                continue

            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            if len(X_tr) < 100 or len(X_va) < 50:
                continue

            model = model_cls(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = model.predict(X_va)
            acc = accuracy_score(y_va, preds)
            fold_scores.append(acc)

            if fold_idx >= 1:
                trial.report(np.mean(fold_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return np.mean(fold_scores) if fold_scores else 0.0

    def _optuna_cv_objective_catboost(self, X: pd.DataFrame, y: pd.Series,
                                     params: dict, trial) -> float:
        """CV objective for CatBoost (different fit API)."""
        import optuna

        CatBoostClassifier = _safe_import_catboost()
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(train_idx) > PURGE_GAP:
                train_idx = train_idx[:-PURGE_GAP]
            else:
                continue

            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            if len(X_tr) < 100 or len(X_va) < 50:
                continue

            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
            preds = model.predict(X_va)
            # CatBoost predict returns strings or arrays — ensure int
            preds = np.array(preds).astype(int)
            acc = accuracy_score(y_va, preds)
            fold_scores.append(acc)

            if fold_idx >= 1:
                trial.report(np.mean(fold_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return np.mean(fold_scores) if fold_scores else 0.0

    # ------------------------------------------------------------------
    # Default params (fallback if Optuna unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_momentum_params() -> dict:
        return {
            "max_depth": 6, "n_estimators": 300, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.1,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "random_state": 42, "n_jobs": -1,
        }

    @staticmethod
    def _default_mean_reversion_params() -> dict:
        return {
            "max_depth": 6, "n_estimators": 300, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "num_leaves": 31,
            "verbose": -1, "random_state": 42, "n_jobs": -1,
        }

    @staticmethod
    def _default_microstructure_params() -> dict:
        return {
            "depth": 6, "iterations": 400, "learning_rate": 0.05,
            "l2_leaf_reg": 3.0, "subsample": 0.8,
            "random_strength": 1.0, "bagging_temperature": 0.5,
            "verbose": 0, "random_seed": 42, "thread_count": -1,
            "allow_writing_files": False,
        }

    # ------------------------------------------------------------------
    # Feature pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _prune_features(model, feature_names: list[str], top_n: int = 25) -> list[str]:
        """Keep top N features by importance."""
        importances = model.feature_importances_
        if len(importances) <= top_n:
            return list(feature_names)
        top_indices = np.argsort(importances)[-top_n:]
        pruned = [feature_names[i] for i in sorted(top_indices)]
        logger.info(f"Feature pruning: {len(feature_names)} -> {len(pruned)} features")
        return pruned

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------

    async def train(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        """Full training pipeline.

        Returns:
            Dict with training stats.
        """
        logger.info("=" * 60)
        logger.info("ENSEMBLE TRAINING PIPELINE START")
        logger.info("=" * 60)

        LGBMClassifier = _safe_import_lightgbm()
        CatBoostClassifier = _safe_import_catboost()

        # ----------------------------------------------------------
        # Step 1: Compute features
        # ----------------------------------------------------------
        logger.info("Step 1: Computing V2 features...")
        features = self.feature_engine.compute_features(
            df_5m, higher_tf_data, ffill=True,
        )
        if features.empty:
            logger.error("Feature computation returned empty DataFrame")
            return {"success": False, "error": "Empty features"}

        # ----------------------------------------------------------
        # Step 2: Create labels
        # ----------------------------------------------------------
        logger.info("Step 2: Creating labels...")
        labels = self._create_labels(df_5m)

        # Align features and labels
        valid_idx = features.index.intersection(labels.dropna().index)
        if len(valid_idx) == 0:
            logger.error("No valid samples after alignment")
            return {"success": False, "error": "No valid samples"}

        X = features.loc[valid_idx]
        y = labels.loc[valid_idx].astype(int)
        all_feature_names = list(X.columns)

        logger.info(f"Aligned data: {len(X)} samples, {len(all_feature_names)} features, "
                    f"class balance: UP={int(y.sum())}/{len(y)} ({y.mean():.1%})")

        # ----------------------------------------------------------
        # Step 3: Detect regimes
        # ----------------------------------------------------------
        logger.info("Step 3: Detecting regimes...")
        regimes = self.regime_detector.detect(X)

        # ----------------------------------------------------------
        # Step 4: Split data
        # INNER(65%) | PURGE(20) | CAL(10%) | PURGE(20) | OOS(10%)
        # ----------------------------------------------------------
        n_total = len(X)
        inner_end = int(n_total * 0.65)
        cal_start = inner_end + PURGE_GAP
        cal_end = min(cal_start + int(n_total * 0.10), n_total)
        oos_start = cal_end + PURGE_GAP
        oos_end = min(oos_start + int(n_total * 0.10), n_total)

        if oos_start >= n_total or (oos_end - oos_start) < 50:
            logger.warning(
                f"Dataset too small for 3-way split (n={n_total}). "
                f"Falling back to simplified split."
            )
            inner_end = int(n_total * 0.70)
            cal_start = inner_end + PURGE_GAP
            remaining = n_total - cal_start
            cal_end = cal_start + remaining // 2
            oos_start = cal_end + PURGE_GAP
            oos_end = n_total

        X_inner, y_inner = X.iloc[:inner_end], y.iloc[:inner_end]
        X_cal, y_cal = X.iloc[cal_start:cal_end], y.iloc[cal_start:cal_end]
        X_oos, y_oos = X.iloc[oos_start:oos_end], y.iloc[oos_start:oos_end]
        regimes_inner = regimes.iloc[:inner_end]
        regimes_cal = regimes.iloc[cal_start:cal_end]
        regimes_oos = regimes.iloc[oos_start:oos_end]

        logger.info(
            f"Data split: INNER={len(X_inner)} | PURGE={PURGE_GAP} | "
            f"CAL={len(X_cal)} | PURGE={PURGE_GAP} | "
            f"OOS={len(X_oos)} | Total={n_total}"
        )

        # Regime distribution in inner split
        regime_dist_inner = {}
        for r in [0, 1, 2, 3]:
            count = int((regimes_inner == r).sum())
            regime_dist_inner[self.regime_detector.get_regime_name(r)] = count
        logger.info(f"Inner regime distribution: {regime_dist_inner}")

        # ----------------------------------------------------------
        # Step 5-7: Optuna tune + train each model
        # ----------------------------------------------------------
        n_trials = 15  # per model
        optuna_timeout = 300  # seconds per model
        top_n_features = 25

        # --- 5. Momentum Model (XGBoost) — TRENDING regimes ---
        trending_mask = regimes_inner.isin(
            [RegimeDetector.TRENDING_UP, RegimeDetector.TRENDING_DOWN]
        )
        X_trending = X_inner[trending_mask]
        y_trending = y_inner[trending_mask]

        if len(X_trending) >= MIN_REGIME_SAMPLES:
            logger.info(f"Training momentum model on {len(X_trending)} trending samples...")
            momentum_params = self._tune_momentum(
                X_trending, y_trending, n_trials=n_trials, timeout=optuna_timeout,
            )
            # Initial train for feature importance
            momentum_initial = XGBClassifier(**momentum_params)
            momentum_initial.fit(X_trending, y_trending, verbose=False)

            # Feature pruning
            momentum_features = self._prune_features(
                momentum_initial, all_feature_names, top_n=top_n_features,
            )
            self.feature_names["momentum"] = momentum_features

            # Retrain on pruned features
            self.momentum_model = XGBClassifier(**momentum_params)
            self.momentum_model.fit(
                X_trending[momentum_features], y_trending, verbose=False,
            )
            mom_acc = accuracy_score(
                y_trending, self.momentum_model.predict(X_trending[momentum_features]),
            )
            logger.info(f"Momentum model trained: train_acc={mom_acc:.4f}, "
                        f"features={len(momentum_features)}")
        else:
            logger.warning(
                f"Not enough trending data ({len(X_trending)} < {MIN_REGIME_SAMPLES}). "
                f"Momentum model will use microstructure as fallback."
            )
            self.momentum_model = None
            self.feature_names["momentum"] = []

        # --- 6. Mean Reversion Model (LightGBM) — RANGING regime ---
        ranging_mask = regimes_inner == RegimeDetector.RANGING
        X_ranging = X_inner[ranging_mask]
        y_ranging = y_inner[ranging_mask]

        if len(X_ranging) >= MIN_REGIME_SAMPLES:
            logger.info(f"Training mean reversion model on {len(X_ranging)} ranging samples...")
            mr_params = self._tune_mean_reversion(
                X_ranging, y_ranging, n_trials=n_trials, timeout=optuna_timeout,
            )
            mr_initial = LGBMClassifier(**mr_params)
            mr_initial.fit(X_ranging, y_ranging, verbose=False)

            mr_features = self._prune_features(
                mr_initial, all_feature_names, top_n=top_n_features,
            )
            self.feature_names["mean_reversion"] = mr_features

            self.mean_reversion_model = LGBMClassifier(**mr_params)
            self.mean_reversion_model.fit(
                X_ranging[mr_features], y_ranging, verbose=False,
            )
            mr_acc = accuracy_score(
                y_ranging, self.mean_reversion_model.predict(X_ranging[mr_features]),
            )
            logger.info(f"Mean reversion model trained: train_acc={mr_acc:.4f}, "
                        f"features={len(mr_features)}")
        else:
            logger.warning(
                f"Not enough ranging data ({len(X_ranging)} < {MIN_REGIME_SAMPLES}). "
                f"Mean reversion model will use microstructure as fallback."
            )
            self.mean_reversion_model = None
            self.feature_names["mean_reversion"] = []

        # --- 7. Microstructure Model (CatBoost) — ALL data ---
        logger.info(f"Training microstructure model on ALL {len(X_inner)} inner samples...")
        micro_params = self._tune_microstructure(
            X_inner, y_inner, n_trials=n_trials, timeout=optuna_timeout,
        )
        micro_initial = CatBoostClassifier(**micro_params)
        micro_initial.fit(X_inner, y_inner, verbose=0)

        micro_features = self._prune_features(
            micro_initial, all_feature_names, top_n=top_n_features,
        )
        self.feature_names["microstructure"] = micro_features

        self.microstructure_model = CatBoostClassifier(**micro_params)
        self.microstructure_model.fit(
            X_inner[micro_features], y_inner, verbose=0,
        )
        micro_acc = accuracy_score(
            y_inner,
            np.array(self.microstructure_model.predict(X_inner[micro_features])).astype(int),
        )
        logger.info(f"Microstructure model trained: train_acc={micro_acc:.4f}, "
                    f"features={len(micro_features)}")

        # ----------------------------------------------------------
        # Step 8: Calibrate on CAL split (per-regime)
        # ----------------------------------------------------------
        logger.info("Step 8: Calibrating per-regime...")
        self._fit_calibrators(X_cal, y_cal, regimes_cal)

        # ----------------------------------------------------------
        # Step 9: Evaluate ensemble on OOS
        # ----------------------------------------------------------
        logger.info(f"Step 9: Evaluating on OOS ({len(X_oos)} samples)...")
        oos_results = self._evaluate_oos(X_oos, y_oos, regimes_oos)

        # Model accuracies on their respective subsets
        model_accuracies = {
            "momentum": mom_acc if self.momentum_model else None,
            "mean_reversion": mr_acc if self.mean_reversion_model else None,
            "microstructure": micro_acc,
        }

        # Regime distribution in full dataset
        regime_distribution = {}
        for r in [0, 1, 2, 3]:
            regime_distribution[self.regime_detector.get_regime_name(r)] = int((regimes == r).sum())

        # Cal spread stats
        cal_probs = []
        for i in range(len(X_cal)):
            row_features = X_cal.iloc[[i]]
            regime = int(regimes_cal.iloc[i])
            raw_prob = self._raw_ensemble_prob(row_features, regime)
            cal_prob = self._calibrate_prob(raw_prob, regime)
            cal_probs.append(cal_prob)
        cal_probs = np.array(cal_probs)

        self.training_stats = {
            "oos_accuracy": oos_results["accuracy"],
            "oos_accuracy_per_regime": oos_results["accuracy_per_regime"],
            "model_accuracies": model_accuracies,
            "ensemble_accuracy": oos_results["accuracy"],
            "feature_counts": {
                "momentum": len(self.feature_names["momentum"]),
                "mean_reversion": len(self.feature_names["mean_reversion"]),
                "microstructure": len(self.feature_names["microstructure"]),
            },
            "regime_distribution": regime_distribution,
            "cal_spread": {
                "min": float(np.min(cal_probs)) if len(cal_probs) > 0 else 0.0,
                "max": float(np.max(cal_probs)) if len(cal_probs) > 0 else 0.0,
                "mean": float(np.mean(cal_probs)) if len(cal_probs) > 0 else 0.0,
            },
            "n_total": n_total,
            "n_inner": len(X_inner),
            "n_cal": len(X_cal),
            "n_oos": len(X_oos),
            "calibrator_types": dict(self.calibrator_types),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.is_trained = True
        self.last_train_time = datetime.now(timezone.utc)

        logger.info("=" * 60)
        logger.info(
            f"ENSEMBLE TRAINING COMPLETE | "
            f"OOS accuracy={oos_results['accuracy']:.4f} | "
            f"Features: mom={len(self.feature_names['momentum'])}, "
            f"mr={len(self.feature_names['mean_reversion'])}, "
            f"micro={len(self.feature_names['microstructure'])} | "
            f"Cal spread: [{cal_probs.min():.3f}, {cal_probs.max():.3f}]"
        )
        logger.info("=" * 60)

        return self.training_stats

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _fit_calibrators(self, X_cal: pd.DataFrame, y_cal: pd.Series,
                         regimes_cal: pd.Series):
        """Fit per-regime calibrators on the CAL split.

        Selection logic:
        - >= 100 samples: Isotonic regression
        - 30-99 samples: Platt scaling (logistic regression)
        - < 30 samples: Passthrough (no calibration)
        """
        self.calibrators = {}
        self.calibrator_types = {}

        for regime in [0, 1, 2, 3]:
            mask = regimes_cal == regime
            n_regime = int(mask.sum())
            regime_name = self.regime_detector.get_regime_name(regime)

            if n_regime < 2:
                logger.info(f"Calibration [{regime_name}]: {n_regime} samples -> passthrough")
                self.calibrators[regime] = None
                self.calibrator_types[regime] = "passthrough"
                continue

            # Get raw ensemble probabilities for this regime's CAL samples
            X_regime = X_cal[mask]
            y_regime = y_cal[mask]
            raw_probs = []
            for i in range(len(X_regime)):
                row = X_regime.iloc[[i]]
                prob = self._raw_ensemble_prob(row, regime)
                raw_probs.append(prob)
            raw_probs = np.array(raw_probs)

            if n_regime >= 100:
                # Isotonic regression
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(raw_probs, y_regime.values)
                self.calibrators[regime] = calibrator
                self.calibrator_types[regime] = "isotonic"
                logger.info(f"Calibration [{regime_name}]: {n_regime} samples -> isotonic")

            elif n_regime >= 30:
                # Platt scaling (logistic regression on raw probs)
                calibrator = LogisticRegression(random_state=42)
                calibrator.fit(raw_probs.reshape(-1, 1), y_regime.values)
                self.calibrators[regime] = calibrator
                self.calibrator_types[regime] = "platt"
                logger.info(f"Calibration [{regime_name}]: {n_regime} samples -> platt")

            else:
                self.calibrators[regime] = None
                self.calibrator_types[regime] = "passthrough"
                logger.info(f"Calibration [{regime_name}]: {n_regime} samples -> passthrough")

    def _calibrate_prob(self, raw_prob: float, regime: int) -> float:
        """Calibrate a single probability."""
        cal_type = self.calibrator_types.get(regime, "passthrough")
        calibrator = self.calibrators.get(regime)

        if cal_type == "passthrough" or calibrator is None:
            return float(np.clip(raw_prob, 0.01, 0.99))
        elif cal_type == "isotonic":
            cal = float(calibrator.predict(np.array([raw_prob]))[0])
            return float(np.clip(cal, 0.01, 0.99))
        elif cal_type == "platt":
            cal = float(calibrator.predict_proba(np.array([[raw_prob]]))[0, 1])
            return float(np.clip(cal, 0.01, 0.99))
        else:
            return float(np.clip(raw_prob, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Raw ensemble probability
    # ------------------------------------------------------------------

    def _raw_ensemble_prob(self, features: pd.DataFrame, regime: int) -> float:
        """Compute raw ensemble P(UP) using regime-weighted soft vote."""
        weights = self.regime_detector.get_regime_weights(regime)
        probs = {}
        total_weight = 0.0

        # Momentum
        if self.momentum_model is not None and self.feature_names["momentum"]:
            mom_feats = self.feature_names["momentum"]
            # Handle missing features
            available = [f for f in mom_feats if f in features.columns]
            if len(available) == len(mom_feats):
                X_mom = features[mom_feats]
                prob = float(self.momentum_model.predict_proba(X_mom)[0, 1])
                probs["momentum"] = prob
                total_weight += weights["momentum"]
        
        # Mean reversion
        if self.mean_reversion_model is not None and self.feature_names["mean_reversion"]:
            mr_feats = self.feature_names["mean_reversion"]
            available = [f for f in mr_feats if f in features.columns]
            if len(available) == len(mr_feats):
                X_mr = features[mr_feats]
                prob = float(self.mean_reversion_model.predict_proba(X_mr)[0, 1])
                probs["mean_reversion"] = prob
                total_weight += weights["mean_reversion"]

        # Microstructure (always available — trained on all data)
        if self.microstructure_model is not None and self.feature_names["microstructure"]:
            micro_feats = self.feature_names["microstructure"]
            available = [f for f in micro_feats if f in features.columns]
            if len(available) == len(micro_feats):
                X_micro = features[micro_feats]
                prob_raw = self.microstructure_model.predict_proba(X_micro)
                # CatBoost may return shape (1, 2) or similar
                prob = float(np.array(prob_raw)[0, 1])
                probs["microstructure"] = prob
                total_weight += weights["microstructure"]

        if total_weight == 0:
            logger.warning("No models contributed to ensemble — returning 0.5")
            return 0.5

        # Weighted average
        ensemble_prob = 0.0
        for model_name, prob in probs.items():
            ensemble_prob += (weights[model_name] / total_weight) * prob

        return ensemble_prob

    # ------------------------------------------------------------------
    # OOS evaluation
    # ------------------------------------------------------------------

    def _evaluate_oos(self, X_oos: pd.DataFrame, y_oos: pd.Series,
                      regimes_oos: pd.Series) -> dict:
        """Evaluate ensemble on OOS split."""
        predictions = []
        probabilities = []

        for i in range(len(X_oos)):
            row = X_oos.iloc[[i]]
            regime = int(regimes_oos.iloc[i])
            raw_prob = self._raw_ensemble_prob(row, regime)
            cal_prob = self._calibrate_prob(raw_prob, regime)
            pred = 1 if cal_prob > 0.5 else 0
            predictions.append(pred)
            probabilities.append(cal_prob)

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        y_true = y_oos.values

        overall_acc = accuracy_score(y_true, predictions)

        # Per-regime accuracy
        acc_per_regime = {}
        for r in [0, 1, 2, 3]:
            mask = regimes_oos.values == r
            if mask.sum() > 0:
                regime_acc = accuracy_score(y_true[mask], predictions[mask])
                acc_per_regime[self.regime_detector.get_regime_name(r)] = {
                    "accuracy": float(regime_acc),
                    "count": int(mask.sum()),
                }
                logger.info(
                    f"OOS [{self.regime_detector.get_regime_name(r)}]: "
                    f"accuracy={regime_acc:.4f}, n={int(mask.sum())}"
                )

        logger.info(f"OOS overall accuracy: {overall_acc:.4f} ({len(y_true)} samples)")

        return {
            "accuracy": float(overall_acc),
            "accuracy_per_regime": acc_per_regime,
            "n_samples": len(y_true),
            "predictions": predictions,
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> dict:
        """Generate prediction from ensemble.

        Args:
            features: Single row or few rows of features from FeatureEngineV2

        Returns:
            Dict with signal, probabilities, regime, confidence, etc.
        """
        if not self.is_trained:
            logger.warning("Ensemble not trained — cannot predict")
            return {
                "signal": "SKIP",
                "raw_prob_up": 0.5,
                "cal_prob_up": 0.5,
                "confidence": 0.5,
                "regime": 3,
                "regime_name": "VOLATILE",
                "model_agreement": 0,
                "model_probs": {},
                "ev": 0.0,
            }

        # Use last row for prediction
        row = features.iloc[[-1]]

        # Detect regime
        regime = int(self.regime_detector.detect(row).iloc[0])
        regime_name = self.regime_detector.get_regime_name(regime)

        # Get raw ensemble probability
        raw_prob_up = self._raw_ensemble_prob(row, regime)

        # Calibrate
        cal_prob_up = self._calibrate_prob(raw_prob_up, regime)

        # Direction
        signal = "UP" if cal_prob_up > 0.5 else "DOWN"
        confidence = max(cal_prob_up, 1.0 - cal_prob_up)

        # Model agreement
        model_probs = {}
        model_directions = []

        if self.momentum_model is not None and self.feature_names["momentum"]:
            mom_feats = self.feature_names["momentum"]
            if all(f in row.columns for f in mom_feats):
                prob = float(self.momentum_model.predict_proba(row[mom_feats])[0, 1])
                model_probs["momentum"] = prob
                model_directions.append("UP" if prob > 0.5 else "DOWN")

        if self.mean_reversion_model is not None and self.feature_names["mean_reversion"]:
            mr_feats = self.feature_names["mean_reversion"]
            if all(f in row.columns for f in mr_feats):
                prob = float(self.mean_reversion_model.predict_proba(row[mr_feats])[0, 1])
                model_probs["mean_reversion"] = prob
                model_directions.append("UP" if prob > 0.5 else "DOWN")

        if self.microstructure_model is not None and self.feature_names["microstructure"]:
            micro_feats = self.feature_names["microstructure"]
            if all(f in row.columns for f in micro_feats):
                prob_raw = self.microstructure_model.predict_proba(row[micro_feats])
                prob = float(np.array(prob_raw)[0, 1])
                model_probs["microstructure"] = prob
                model_directions.append("UP" if prob > 0.5 else "DOWN")

        model_agreement = sum(1 for d in model_directions if d == signal)

        # EV calculation
        ev = (confidence * self.config.win_payout) - ((1.0 - confidence) * self.config.loss_amount)

        return {
            "signal": signal,
            "raw_prob_up": float(raw_prob_up),
            "cal_prob_up": float(cal_prob_up),
            "confidence": float(confidence),
            "regime": regime,
            "regime_name": regime_name,
            "model_agreement": model_agreement,
            "model_probs": model_probs,
            "ev": float(ev),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all models, calibrators, feature names, and metadata to path/"""
        os.makedirs(path, exist_ok=True)

        # 1. Momentum model (XGBoost native save)
        if self.momentum_model is not None:
            self.momentum_model.save_model(os.path.join(path, "momentum_model.json"))
            logger.info("Saved momentum model")

        # 2. Mean reversion model (LightGBM native save)
        if self.mean_reversion_model is not None:
            self.mean_reversion_model.booster_.save_model(
                os.path.join(path, "mean_reversion_model.txt")
            )
            # Also save params for reconstruction
            with open(os.path.join(path, "mean_reversion_params.pkl"), "wb") as f:
                pickle.dump(self.mean_reversion_model.get_params(), f)
            logger.info("Saved mean reversion model")

        # 3. Microstructure model (CatBoost native save)
        if self.microstructure_model is not None:
            self.microstructure_model.save_model(os.path.join(path, "microstructure_model.cbm"))
            logger.info("Saved microstructure model")

        # 4. Calibrators
        with open(os.path.join(path, "calibrators.pkl"), "wb") as f:
            pickle.dump({
                "calibrators": self.calibrators,
                "calibrator_types": self.calibrator_types,
            }, f)
        logger.info("Saved calibrators")

        # 5. Feature names
        with open(os.path.join(path, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info("Saved feature names")

        # 6. Metadata
        metadata = {
            "training_stats": self.training_stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v5",
            "has_momentum": self.momentum_model is not None,
            "has_mean_reversion": self.mean_reversion_model is not None,
            "has_microstructure": self.microstructure_model is not None,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved metadata")

        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str, config: ModelConfig) -> "EnsembleModel":
        """Load ensemble from path/"""
        ensemble = cls(config)

        # 1. Feature names (load first — needed for model reconstruction)
        fn_path = os.path.join(path, "feature_names.json")
        if os.path.exists(fn_path):
            with open(fn_path, "r") as f:
                ensemble.feature_names = json.load(f)
            logger.info(f"Loaded feature names: {[f'{k}={len(v)}' for k, v in ensemble.feature_names.items()]}")

        # 2. Momentum model
        mom_path = os.path.join(path, "momentum_model.json")
        if os.path.exists(mom_path):
            ensemble.momentum_model = XGBClassifier()
            ensemble.momentum_model.load_model(mom_path)
            logger.info("Loaded momentum model")

        # 3. Mean reversion model
        mr_model_path = os.path.join(path, "mean_reversion_model.txt")
        mr_params_path = os.path.join(path, "mean_reversion_params.pkl")
        if os.path.exists(mr_model_path):
            LGBMClassifier = _safe_import_lightgbm()
            import lightgbm as lgb

            # Load params if available
            mr_params = {}
            if os.path.exists(mr_params_path):
                with open(mr_params_path, "rb") as f:
                    mr_params = pickle.load(f)

            # Reconstruct via booster
            booster = lgb.Booster(model_file=mr_model_path)
            ensemble.mean_reversion_model = LGBMClassifier(**mr_params)
            ensemble.mean_reversion_model._Booster = booster
            ensemble.mean_reversion_model._fitted = True
            ensemble.mean_reversion_model._n_features = booster.num_feature()
            ensemble.mean_reversion_model._n_classes = 2
            ensemble.mean_reversion_model.fitted_ = True
            ensemble.mean_reversion_model._le = None
            # Set classes_ attribute for predict_proba
            ensemble.mean_reversion_model.classes_ = np.array([0, 1])
            ensemble.mean_reversion_model._n_features_in = booster.num_feature()
            logger.info("Loaded mean reversion model")

        # 4. Microstructure model
        micro_path = os.path.join(path, "microstructure_model.cbm")
        if os.path.exists(micro_path):
            CatBoostClassifier = _safe_import_catboost()
            ensemble.microstructure_model = CatBoostClassifier()
            ensemble.microstructure_model.load_model(micro_path)
            logger.info("Loaded microstructure model")

        # 5. Calibrators
        cal_path = os.path.join(path, "calibrators.pkl")
        if os.path.exists(cal_path):
            with open(cal_path, "rb") as f:
                cal_data = pickle.load(f)
            ensemble.calibrators = cal_data["calibrators"]
            ensemble.calibrator_types = cal_data["calibrator_types"]
            logger.info(f"Loaded calibrators: {ensemble.calibrator_types}")

        # 6. Metadata
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ensemble.training_stats = meta.get("training_stats", {})
            logger.info(f"Loaded metadata (version={meta.get('version', 'unknown')})")

        ensemble.is_trained = True
        logger.info(f"Ensemble loaded from {path}")
        return ensemble
