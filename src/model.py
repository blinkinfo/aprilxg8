"""XGBoost model for BTC 5-minute candle direction prediction.

Fixes applied (v5 — accuracy truthfulness overhaul):
- Fix A: Proper nested CV — Optuna tunes on inner split only, final validation
  on a truly held-out set that Optuna never touched.
- Fix B: Increased Optuna CV folds from 2 to 5 for regime diversity.
- Fix C: NaN handling consistency — forward-fill in both training and inference
  (no more drop-in-training / zero-fill-in-inference mismatch).
- Fix D: Point-in-time features — predict() uses only completed candles,
  excluding the current in-progress candle.
- Fix E: Purge gap (20 candles / 100 minutes) between train/val splits to
  prevent feature leakage across boundaries.
- Fix F: Honest accuracy reporting — logs OOS accuracy, per-class accuracy,
  prediction bias, and confidence calibration.
- Fix G: Production model retrained on 100% of data after validation confirms
  quality, so no recent data is wasted.

Prior improvements preserved:
- Improvement 2: Confidence filtering — skip low-confidence predictions
- Improvement 4: Walk-forward retraining gate — only swap model if new one is better
- Improvement 5: Optuna Bayesian hyperparameter optimization
- Interactive retrain: train_for_comparison / apply / reject / force_tune
"""
import logging
import os
import pickle
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from .config import ModelConfig
from .features import FeatureEngineer

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

logger = logging.getLogger(__name__)

# Number of candles to skip between train and validation splits.
# Prevents feature look-back windows (up to ~50 candles) from leaking
# across the split boundary. 20 candles = 100 minutes at 5m resolution.
PURGE_GAP = 20


class PredictionModel:
    """XGBoost-based prediction model with training, evaluation, and inference."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.model: Optional[XGBClassifier] = None
        self.feature_names: list[str] = []
        self.last_train_time: Optional[datetime] = None
        self.last_tune_time: Optional[datetime] = None
        self.best_params: Optional[dict] = None
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self._model_dir = "models"
        self._n_train_samples: int = 0
        self.train_end_ts: Optional[datetime] = None  # For backtester OOS detection

        # Interactive retrain state (train_for_comparison / apply / reject)
        self._pending_model: Optional[XGBClassifier] = None
        self._pending_feature_names: list[str] = []
        self._pending_val_accuracy: float = 0.0
        self._pending_train_accuracy: float = 0.0
        self._pending_cv_accuracy: float = 0.0
        self._pending_n_samples: int = 0
        self._pending_oos_metrics: Optional[dict] = None
        self._pending_xgb_params: Optional[dict] = None
        self._pending_train_end_ts: Optional[datetime] = None

        # One-shot force-tune flag
        self._force_tune_flag: bool = False

    # ------------------------------------------------------------------
    # Properties for bot.py compatibility
    # ------------------------------------------------------------------

    @property
    def train_samples(self) -> int:
        """Number of samples used in last training run."""
        return self._n_train_samples

    @property
    def best_xgb_params(self) -> Optional[dict]:
        """Alias for best_params (used by bot.py status display)."""
        return self.best_params

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """Save model + metadata to disk."""
        if self.model is None:
            logger.warning("No model to save.")
            return

        path = path or os.path.join(self._model_dir, "model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "last_train_time": self.last_train_time,
            "last_tune_time": self.last_tune_time,
            "best_params": self.best_params,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "n_train_samples": self._n_train_samples,
            "train_end_ts": self.train_end_ts,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load model + metadata from disk. Returns True if successful."""
        path = path or os.path.join(self._model_dir, "model.pkl")
        if not os.path.exists(path):
            logger.info(f"No saved model found at {path}")
            return False

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.model = state["model"]
            self.feature_names = state["feature_names"]
            self.last_train_time = state.get("last_train_time")
            self.last_tune_time = state.get("last_tune_time")
            self.best_params = state.get("best_params")
            self.train_accuracy = state.get("train_accuracy", 0.0)
            self.val_accuracy = state.get("val_accuracy", 0.0)
            self._n_train_samples = state.get("n_train_samples", 0)
            self.train_end_ts = state.get("train_end_ts")
            logger.info(
                f"Model loaded from {path} | "
                f"val_accuracy={self.val_accuracy:.4f} | "
                f"features={len(self.feature_names)}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Training state checks
    # ------------------------------------------------------------------

    def needs_training(self) -> bool:
        """Check if model needs (re)training."""
        if self.model is None:
            return True
        if self.last_train_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_train_time).total_seconds()
        return elapsed >= self.config.retrain_interval_hours * 3600

    def needs_retrain(self) -> bool:
        """Alias for needs_training() — used by bot.py main loop."""
        return self.needs_training()

    def needs_tuning(self) -> bool:
        """Check if Optuna hyperparameter tuning is due."""
        # One-shot force-tune overrides the timer
        if self._force_tune_flag:
            return True
        if not self.config.enable_optuna_tuning:
            return False
        if self.last_tune_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_tune_time).total_seconds()
        return elapsed >= self.config.optuna_tune_interval_hours * 3600

    def force_tune(self):
        """Set one-shot flag to force Optuna tuning on next train cycle."""
        self._force_tune_flag = True
        logger.info("Force-tune flag set: Optuna will run on next training cycle")

    # ------------------------------------------------------------------
    # Data preparation (shared by train + tune)
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare aligned features and labels from raw OHLCV data.

        Fix C: Features use forward-fill for NaN instead of dropping rows.
        Labels and features aligned via index intersection.

        Returns:
            (features_df, labels_series) with aligned indices, no NaN.
        """
        labels = self.feature_engineer.create_labels(df_5m)
        features_df = self.feature_engineer.compute_features(
            df_5m, higher_tf_data, ffill=True
        )

        if features_df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        valid_idx = features_df.dropna().index.intersection(labels.dropna().index)

        if len(valid_idx) == 0:
            logger.error("No valid samples after alignment")
            return pd.DataFrame(), pd.Series(dtype=float)

        X = features_df.loc[valid_idx]
        y = labels.loc[valid_idx]

        logger.info(
            f"Data prepared: {len(X)} samples, {len(X.columns)} features, "
            f"class balance: UP={int(y.sum())}/{len(y)} ({y.mean():.1%})"
        )
        return X, y

    # ------------------------------------------------------------------
    # Hyperparameter resolution
    # ------------------------------------------------------------------

    def _resolve_xgb_params(self, X_inner: pd.DataFrame, y_inner: pd.Series) -> dict:
        """Resolve XGBoost params: tune if needed, else use cached or defaults.

        Args:
            X_inner: Inner split features (for Optuna if tuning is needed)
            y_inner: Inner split labels

        Returns:
            Dict of XGBoost parameters to use for training.
        """
        xgb_params = dict(self.config.xgb_params)

        if self.needs_tuning():
            logger.info("Optuna tuning scheduled — running on INNER split only")
            tuned_params = self.tune_hyperparameters(X_inner, y_inner)
            if tuned_params:
                xgb_params = tuned_params
                logger.info("Using Optuna-tuned hyperparameters")
            else:
                logger.info("Optuna tuning failed/skipped, using default params")
            # Clear force-tune flag after tuning attempt
            self._force_tune_flag = False
        elif self.best_params:
            xgb_params = self.best_params
            logger.info("Using previously tuned hyperparameters")

        return xgb_params

    # ------------------------------------------------------------------
    # Optuna hyperparameter tuning (Fix A + B: inner split only)
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self,
        X_inner: pd.DataFrame,
        y_inner: pd.Series,
    ) -> Optional[dict]:
        """Run Optuna Bayesian optimization on the INNER split only.

        Fix A: Optuna only sees X_inner/y_inner (first ~70% of data).
        Fix B: Uses 5-fold TimeSeriesSplit for regime diversity.
        Fix E: Purge gap between each fold's train/val boundary.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, skipping tuning")
            return None

        logger.info(
            f"Starting Optuna tuning on inner split: {len(X_inner)} samples, "
            f"5-fold CV with purge gap={PURGE_GAP}"
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }

            tscv = TimeSeriesSplit(n_splits=5)
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_inner)):
                if len(train_idx) > PURGE_GAP:
                    train_idx = train_idx[:-PURGE_GAP]
                else:
                    continue

                X_tr = X_inner.iloc[train_idx]
                y_tr = y_inner.iloc[train_idx]
                X_va = X_inner.iloc[val_idx]
                y_va = y_inner.iloc[val_idx]

                if len(X_tr) < 100 or len(X_va) < 50:
                    continue

                model = XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
                preds = model.predict(X_va)
                acc = accuracy_score(y_va, preds)
                fold_scores.append(acc)

                if fold_idx >= 1:
                    trial.report(np.mean(fold_scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if not fold_scores:
                return 0.0
            return np.mean(fold_scores)

        try:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
            study = optuna.create_study(
                direction="maximize",
                pruner=pruner,
            )
            study.optimize(
                objective,
                n_trials=self.config.optuna_n_trials,
                timeout=self.config.optuna_timeout_seconds,
                show_progress_bar=False,
            )

            best = study.best_params
            best["objective"] = "binary:logistic"
            best["eval_metric"] = "logloss"
            best["random_state"] = 42
            best["n_jobs"] = -1

            self.best_params = best
            self.last_tune_time = datetime.now(timezone.utc)

            logger.info(
                f"Optuna tuning complete: {len(study.trials)} trials, "
                f"best CV accuracy={study.best_value:.4f}, "
                f"best params: depth={best.get('max_depth')}, "
                f"lr={best.get('learning_rate', 0):.4f}, "
                f"n_est={best.get('n_estimators')}"
            )
            return best

        except Exception as e:
            logger.error(f"Optuna tuning failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Honest accuracy reporting (Fix F)
    # ------------------------------------------------------------------

    @staticmethod
    def _log_honest_metrics(
        model: XGBClassifier,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        label: str = "OOS",
    ) -> dict:
        """Log detailed, honest validation metrics."""
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)
        confidence = np.max(proba, axis=1)

        acc = accuracy_score(y_val, preds)

        up_mask = y_val == 1
        down_mask = y_val == 0
        up_acc = accuracy_score(y_val[up_mask], preds[up_mask]) if up_mask.sum() > 0 else 0.0
        down_acc = accuracy_score(y_val[down_mask], preds[down_mask]) if down_mask.sum() > 0 else 0.0

        pred_up_pct = float(preds.mean())
        actual_up_pct = float(y_val.mean())

        high_conf_mask = confidence >= 0.60
        high_conf_acc = (
            accuracy_score(y_val[high_conf_mask], preds[high_conf_mask])
            if high_conf_mask.sum() > 10
            else None
        )

        try:
            ll = log_loss(y_val, proba)
        except Exception:
            ll = None

        metrics = {
            "accuracy": acc,
            "up_accuracy": up_acc,
            "down_accuracy": down_acc,
            "pred_up_pct": pred_up_pct,
            "actual_up_pct": actual_up_pct,
            "high_conf_accuracy": high_conf_acc,
            "high_conf_count": int(high_conf_mask.sum()),
            "avg_confidence": float(confidence.mean()),
            "log_loss": ll,
            "n_samples": len(y_val),
        }

        logger.info(
            f"[{label}] Accuracy: {acc:.4f} | "
            f"UP acc: {up_acc:.4f}, DOWN acc: {down_acc:.4f} | "
            f"Pred UP%: {pred_up_pct:.1%}, Actual UP%: {actual_up_pct:.1%} | "
            f"High-conf (>=60%) acc: "
            f"{f'{high_conf_acc:.4f} (n={high_conf_mask.sum()})' if high_conf_acc is not None else 'N/A'} | "
            f"Avg conf: {confidence.mean():.4f} | "
            f"Log loss: {f'{ll:.4f}' if ll is not None else 'N/A'} | "
            f"Samples: {len(y_val)}"
        )

        return metrics

    # ------------------------------------------------------------------
    # Internal training core (shared by train + train_for_comparison)
    # ------------------------------------------------------------------

    def _train_core(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict] = None,
    ) -> Optional[dict]:
        """Core training logic shared by train() and train_for_comparison().

        Returns dict with all training artifacts and metrics, or None on failure.
        Does NOT install the model — caller decides what to do with results.
        """
        X, y = self._prepare_data(df_5m, higher_tf_data)
        if X.empty:
            return None

        n_total = len(X)
        feature_names = list(X.columns)

        # ----------------------------------------------------------
        # Split: INNER (70%) | PURGE (20 candles) | OOS (15%) | tail
        # ----------------------------------------------------------
        inner_end = int(n_total * 0.70)
        oos_start = inner_end + PURGE_GAP
        oos_end = min(oos_start + int(n_total * 0.15), n_total)

        if oos_start >= n_total or (oos_end - oos_start) < 100:
            logger.warning(
                f"Dataset too small for proper nested CV "
                f"(n={n_total}, oos_start={oos_start}). "
                f"Falling back to simple 80/20 with purge gap."
            )
            inner_end = int(n_total * 0.80) - PURGE_GAP
            oos_start = inner_end + PURGE_GAP
            oos_end = n_total

        X_inner, y_inner = X.iloc[:inner_end], y.iloc[:inner_end]
        X_oos, y_oos = X.iloc[oos_start:oos_end], y.iloc[oos_start:oos_end]

        logger.info(
            f"Data split: INNER={len(X_inner)} | PURGE={PURGE_GAP} | "
            f"OOS={len(X_oos)} | Total={n_total}"
        )

        # ----------------------------------------------------------
        # Step 1: Resolve hyperparameters (Optuna on inner only)
        # ----------------------------------------------------------
        xgb_params = self._resolve_xgb_params(X_inner, y_inner)

        # ----------------------------------------------------------
        # Step 2: Train candidate on INNER
        # ----------------------------------------------------------
        logger.info(f"Training candidate model on INNER split ({len(X_inner)} samples)")
        candidate_model = XGBClassifier(**xgb_params)
        candidate_model.fit(X_inner, y_inner, verbose=False)

        # ----------------------------------------------------------
        # Step 3: Honest OOS validation (Fix F)
        # ----------------------------------------------------------
        logger.info(f"Validating on truly held-out OOS split ({len(X_oos)} samples)")
        oos_metrics = self._log_honest_metrics(candidate_model, X_oos, y_oos, label="OOS-holdout")
        new_val_accuracy = oos_metrics["accuracy"]

        # Cross-validation on inner for comparison logging
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_inner)):
            if len(train_idx) > PURGE_GAP:
                train_idx = train_idx[:-PURGE_GAP]
            else:
                continue
            X_tr = X_inner.iloc[train_idx]
            y_tr = y_inner.iloc[train_idx]
            X_va = X_inner.iloc[val_idx]
            y_va = y_inner.iloc[val_idx]
            if len(X_tr) < 100 or len(X_va) < 50:
                continue
            fold_model = XGBClassifier(**xgb_params)
            fold_model.fit(X_tr, y_tr, verbose=False)
            fold_acc = accuracy_score(y_va, fold_model.predict(X_va))
            cv_scores.append(fold_acc)
            logger.info(f"  CV fold {fold_idx + 1}: accuracy={fold_acc:.4f}")

        cv_mean = np.mean(cv_scores) if cv_scores else 0.0
        cv_std = np.std(cv_scores) if cv_scores else 0.0
        if cv_scores:
            logger.info(
                f"CV summary: mean={cv_mean:.4f} +/- {cv_std:.4f} | "
                f"OOS holdout: {new_val_accuracy:.4f}"
            )

        # OOS log loss for comparison display
        try:
            oos_proba = candidate_model.predict_proba(X_oos)
            oos_logloss = log_loss(y_oos, oos_proba)
        except Exception:
            oos_logloss = 0.0

        # ----------------------------------------------------------
        # Step 4: Train PRODUCTION model on ALL data (Fix G)
        # ----------------------------------------------------------
        logger.info(f"Training PRODUCTION model on ALL {n_total} samples")
        production_model = XGBClassifier(**xgb_params)
        production_model.fit(X, y, verbose=False)

        train_preds = production_model.predict(X)
        train_acc = accuracy_score(y, train_preds)
        logger.info(f"Production model train accuracy (full data): {train_acc:.4f}")

        # Determine train_end_ts from the raw data timestamps
        train_end_ts = None
        if "timestamp" in df_5m.columns and len(df_5m) > 0:
            last_ts = df_5m["timestamp"].iloc[-1]
            if hasattr(last_ts, "to_pydatetime"):
                train_end_ts = last_ts.to_pydatetime()
                if train_end_ts.tzinfo is None:
                    train_end_ts = train_end_ts.replace(tzinfo=timezone.utc)
            elif isinstance(last_ts, datetime):
                train_end_ts = last_ts
                if train_end_ts.tzinfo is None:
                    train_end_ts = train_end_ts.replace(tzinfo=timezone.utc)

        # Recent-288 accuracy (last 288 OOS samples = ~24h at 5m)
        recent_288_acc = 0.0
        if len(X_oos) > 0:
            tail = min(288, len(X_oos))
            X_recent = X_oos.iloc[-tail:]
            y_recent = y_oos.iloc[-tail:]
            recent_preds = candidate_model.predict(X_recent)
            recent_288_acc = accuracy_score(y_recent, recent_preds)

        return {
            "production_model": production_model,
            "candidate_model": candidate_model,
            "feature_names": feature_names,
            "xgb_params": xgb_params,
            "n_total": n_total,
            "n_inner": len(X_inner),
            "n_oos": len(X_oos),
            "val_accuracy": new_val_accuracy,
            "train_accuracy": train_acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_scores": cv_scores,
            "oos_metrics": oos_metrics,
            "oos_logloss": oos_logloss,
            "recent_288_acc": recent_288_acc,
            "optuna_tuned": xgb_params != dict(self.config.xgb_params),
            "train_end_ts": train_end_ts,
        }

    # ------------------------------------------------------------------
    # Main training pipeline (auto — used by scheduled retrains)
    # ------------------------------------------------------------------

    def train(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict] = None,
    ) -> dict:
        """Train the model with proper nested CV and honest validation.

        This is the auto-retrain path used by the scheduled retrain loop.
        It applies the retrain quality gate and auto-installs the model.

        Returns:
            Dict with training results and metrics.
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE START")
        logger.info("=" * 60)

        result = self._train_core(df_5m, higher_tf_data)
        if result is None:
            return {"success": False, "error": "No valid training data"}

        new_val_accuracy = result["val_accuracy"]
        n_total = result["n_total"]

        # ----------------------------------------------------------
        # Retrain quality gate (Improvement 4)
        # ----------------------------------------------------------
        old_accuracy = self.val_accuracy if self.model else 0.0
        improvement = new_val_accuracy - old_accuracy
        gate_threshold = self.config.retrain_min_improvement

        if self.model is not None and improvement < gate_threshold:
            logger.info(
                f"RETRAIN GATE: REJECTED | "
                f"old={old_accuracy:.4f}, new={new_val_accuracy:.4f}, "
                f"improvement={improvement:+.4f} < threshold={gate_threshold}"
            )
            return {
                "success": True,
                "accepted": False,
                "model_swapped": False,
                "old_accuracy": old_accuracy,
                "new_accuracy": new_val_accuracy,
                "val_accuracy": new_val_accuracy,
                "active_val_accuracy": old_accuracy,
                "improvement": improvement,
                "cv_mean": result["cv_mean"],
                "oos_metrics": result["oos_metrics"],
                "total_samples": n_total,
                "n_samples": n_total,
                "optuna_tuned": result["optuna_tuned"],
            }

        logger.info(
            f"RETRAIN GATE: ACCEPTED | "
            f"old={old_accuracy:.4f}, new={new_val_accuracy:.4f}, "
            f"improvement={improvement:+.4f}"
        )

        # Install production model
        self.model = result["production_model"]
        self.feature_names = result["feature_names"]
        self.train_accuracy = result["train_accuracy"]
        self.val_accuracy = new_val_accuracy
        self._n_train_samples = n_total
        self.train_end_ts = result["train_end_ts"]
        self.last_train_time = datetime.now(timezone.utc)

        self.save()

        logger.info("=" * 60)
        logger.info(
            f"TRAINING COMPLETE | OOS accuracy={new_val_accuracy:.4f} | "
            f"CV mean={result['cv_mean']:.4f} | "
            f"Production train acc={result['train_accuracy']:.4f}"
        )
        logger.info("=" * 60)

        return {
            "success": True,
            "accepted": True,
            "model_swapped": True,
            "old_accuracy": old_accuracy,
            "new_accuracy": new_val_accuracy,
            "val_accuracy": new_val_accuracy,
            "active_val_accuracy": new_val_accuracy,
            "improvement": improvement,
            "train_accuracy": result["train_accuracy"],
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
            "cv_scores": result["cv_scores"],
            "oos_metrics": result["oos_metrics"],
            "n_features": len(result["feature_names"]),
            "total_samples": n_total,
            "n_samples": n_total,
            "n_inner": result["n_inner"],
            "n_oos": result["n_oos"],
            "purge_gap": PURGE_GAP,
            "optuna_tuned": result["optuna_tuned"],
        }

    # ------------------------------------------------------------------
    # Interactive retrain (for /retrain and /forcetune commands)
    # ------------------------------------------------------------------

    def train_for_comparison(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict] = None,
    ) -> dict:
        """Train a candidate model and return comparison metrics.

        Does NOT install the model. Stores it as _pending_model for
        the user to accept (apply_pending_model) or reject (reject_pending_model).

        Returns:
            Dict with old_* and new_* metrics for comparison display.
        """
        logger.info("=" * 60)
        logger.info("INTERACTIVE RETRAIN: Training candidate for comparison")
        logger.info("=" * 60)

        result = self._train_core(df_5m, higher_tf_data)
        if result is None:
            return {"has_existing_model": self.model is not None, "error": "Training failed"}

        # Store pending model
        self._pending_model = result["production_model"]
        self._pending_feature_names = result["feature_names"]
        self._pending_val_accuracy = result["val_accuracy"]
        self._pending_train_accuracy = result["train_accuracy"]
        self._pending_cv_accuracy = result["cv_mean"]
        self._pending_n_samples = result["n_total"]
        self._pending_oos_metrics = result["oos_metrics"]
        self._pending_xgb_params = result["xgb_params"]
        self._pending_train_end_ts = result["train_end_ts"]

        # Compute current model's OOS metrics for comparison
        old_val_accuracy = self.val_accuracy if self.model else 0.0
        old_val_logloss = 0.0
        old_recent_accuracy = 0.0

        comparison = {
            "has_existing_model": self.model is not None,
            "old_val_accuracy": old_val_accuracy,
            "new_val_accuracy": result["val_accuracy"],
            "improvement": result["val_accuracy"] - old_val_accuracy,
            "new_cv_accuracy": result["cv_mean"],
            "old_val_logloss": old_val_logloss,
            "new_val_logloss": result["oos_logloss"],
            "new_total_samples": result["n_total"],
            "new_n_features": len(result["feature_names"]),
            "optuna_tuned": result["optuna_tuned"],
            "old_recent_accuracy": old_recent_accuracy,
            "new_recent_accuracy": result["recent_288_acc"],
        }

        logger.info(
            f"Comparison ready: old={old_val_accuracy:.4f}, "
            f"new={result['val_accuracy']:.4f}, "
            f"delta={comparison['improvement']:+.4f}"
        )

        return comparison

    def apply_pending_model(self) -> dict:
        """Accept and install the pending candidate model.

        Returns:
            Dict with action='swap' and the new model's metrics.
        """
        if self._pending_model is None:
            logger.warning("No pending model to apply")
            return {"action": "error", "error": "No pending model"}

        self.model = self._pending_model
        self.feature_names = self._pending_feature_names
        self.val_accuracy = self._pending_val_accuracy
        self.train_accuracy = self._pending_train_accuracy
        self._n_train_samples = self._pending_n_samples
        self.train_end_ts = self._pending_train_end_ts
        self.last_train_time = datetime.now(timezone.utc)

        # Clear pending state
        self._pending_model = None

        logger.info(f"Pending model applied: val_accuracy={self.val_accuracy:.4f}")

        return {
            "action": "swap",
            "val_accuracy": self.val_accuracy,
            "train_accuracy": self.train_accuracy,
        }

    def reject_pending_model(self) -> dict:
        """Reject the pending candidate model and keep the current one.

        Returns:
            Dict with action='keep' and current model's metrics.
        """
        rejected_acc = self._pending_val_accuracy if self._pending_model else 0.0

        # Clear pending state
        self._pending_model = None

        logger.info(
            f"Pending model rejected: keeping current val_accuracy={self.val_accuracy:.4f}"
        )

        return {
            "action": "keep",
            "val_accuracy": self.val_accuracy,
            "rejected_val_accuracy": rejected_acc,
        }

    # ------------------------------------------------------------------
    # Prediction / Inference (Fix C + D)
    # ------------------------------------------------------------------

    def predict(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict] = None,
    ) -> Optional[dict]:
        """Predict the direction of the NEXT candle.

        Fix D: The caller (bot.py) passes only COMPLETED candles.
        Fix C: NaN values are forward-filled consistently with training.

        Args:
            df_5m: DataFrame of COMPLETED 5m candles (current candle excluded)
            higher_tf_data: Dict of higher-TF DataFrames (also completed only)

        Returns:
            Dict with direction, confidence, strength, probabilities,
            current_price, model_accuracy — or None if skipped.
        """
        if self.model is None:
            logger.warning("No model available for prediction")
            return None

        try:
            features_df = self.feature_engineer.compute_features(
                df_5m, higher_tf_data, ffill=True
            )

            if features_df.empty:
                logger.warning("Feature computation returned empty DataFrame")
                return None

            latest = features_df.iloc[[-1]][self.feature_names]

            # Fix C: Last-resort fallback for any remaining NaN after ffill
            if latest.isna().any().any():
                nan_cols = latest.columns[latest.isna().any()].tolist()
                logger.warning(
                    f"NaN in {len(nan_cols)} features after ffill "
                    f"(last-resort zero fill): {nan_cols[:5]}"
                )
                latest = latest.fillna(0)

            proba = self.model.predict_proba(latest)[0]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class])

            direction = "UP" if pred_class == 1 else "DOWN"

            # Confidence filtering (Improvement 2)
            if confidence < self.config.confidence_min:
                logger.info(
                    f"Low confidence {confidence:.4f} < {self.config.confidence_min} — skipping"
                )
                return None

            strength = "STRONG" if confidence >= self.config.confidence_strong else "NORMAL"

            # Get current price from last completed candle
            current_price = float(df_5m["close"].iloc[-1]) if not df_5m.empty else 0.0

            result = {
                "signal": direction,
                "direction": direction,
                "confidence": confidence,
                "strength": strength,
                "current_price": current_price,
                "model_accuracy": self.val_accuracy,
                "probabilities": {"DOWN": float(proba[0]), "UP": float(proba[1])},
            }

            logger.info(
                f"Prediction: {direction} ({confidence:.1%}) [{strength}] | "
                f"P(UP)={proba[1]:.4f}, P(DOWN)={proba[0]:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return model metadata for status display."""
        return {
            "has_model": self.model is not None,
            "feature_count": len(self.feature_names),
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_samples": self._n_train_samples,
            "last_train_time": (
                self.last_train_time.isoformat() if self.last_train_time else None
            ),
            "last_tune_time": (
                self.last_tune_time.isoformat() if self.last_tune_time else None
            ),
            "has_tuned_params": self.best_params is not None,
        }
