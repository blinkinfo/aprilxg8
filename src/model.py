"""XGBoost model for BTC 5-minute candle direction prediction.

Improvements:
- Improvement 2: Confidence filtering — skip low-confidence predictions
- Improvement 4: Walk-forward retraining gate — only swap model if new one is better
- Improvement 5: Optuna Bayesian hyperparameter optimization

Fixes applied:
- Removed deprecated use_label_encoder param (xgboost >= 1.7)
- Added timeout guard to Optuna tuning to prevent container kills
- Reduced Optuna CV folds from 3 to 2 for faster tuning on large datasets
- Suppressed noisy xgboost deprecation warnings
- Added MedianPruner to early-stop underperforming Optuna trials after fold 1
- Increased default trials to 40 and timeout to 750s for better TPE convergence
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

# Suppress the noisy xgboost "use_label_encoder" deprecation warning
# and any other non-critical xgboost warnings that spam logs.
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

logger = logging.getLogger(__name__)


class PredictionModel:
    """XGBoost-based prediction model with training, evaluation, and inference."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.model: Optional[XGBClassifier] = None
        self.feature_names: list[str] = []
        self.last_train_time: Optional[datetime] = None
        self.last_tune_time: Optional[datetime] = None
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self.val_logloss: float = 1.0
        self.train_samples: int = 0
        self.best_xgb_params: Optional[dict] = None  # Optuna-tuned params
        # Pending model state for interactive retrain comparison
        self._pending_model: Optional[XGBClassifier] = None
        self._pending_metrics: Optional[dict] = None

    def _get_xgb_params(self) -> dict:
        """Return the best known XGBoost params (Optuna-tuned or defaults)."""
        if self.best_xgb_params is not None:
            return self.best_xgb_params
        return self.config.xgb_params.copy()

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """Improvement 5: Bayesian hyperparameter optimization with Optuna.

        Uses TimeSeriesSplit CV to find optimal XGBoost parameters.
        Enforces both a trial count limit AND a wall-clock timeout so
        tuning cannot exceed the container's health-check window.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Dict of best parameters
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, using default params")
            return self.config.xgb_params.copy()

        n_trials = self.config.optuna_n_trials
        timeout_sec = self.config.optuna_timeout_seconds

        logger.info(
            f"Starting Optuna hyperparameter tuning "
            f"({n_trials} trials, {timeout_sec}s timeout)..."
        )

        # Use 2-fold CV for speed on large datasets.
        # With 43k samples each fold still has ~21k train / ~21k val.
        tscv = TimeSeriesSplit(n_splits=2)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }

            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

                model = XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
                val_proba = model.predict_proba(X_va)[:, 1]
                fold_loss = log_loss(y_va, val_proba)
                cv_scores.append(fold_loss)

                # Report intermediate CV fold result for pruning.
                # After fold 0 completes, the pruner can compare this
                # trial's loss against the median of completed trials
                # and kill it early if it's clearly underperforming.
                trial.report(np.mean(cv_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,  # Let first 10 trials complete fully (TPE warm-up)
                n_warmup_steps=0,     # Can prune after first CV fold
            ),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_sec,  # Hard wall-clock cap
            show_progress_bar=False,
        )

        best = study.best_params
        best.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        })

        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        logger.info(
            f"Optuna finished: {completed} completed, {pruned} pruned "
            f"(out of {len(study.trials)} total trials), "
            f"best logloss: {study.best_value:.6f}"
        )
        logger.info(
            f"Best params: max_depth={best['max_depth']}, lr={best['learning_rate']:.4f}, "
            f"n_est={best['n_estimators']}, subsample={best['subsample']:.2f}"
        )

        self.best_xgb_params = best
        self.last_tune_time = datetime.now(timezone.utc)
        return best

    def needs_tuning(self) -> bool:
        """Check if hyperparameters need re-tuning."""
        if not self.config.enable_optuna_tuning:
            return False
        if self.last_tune_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_tune_time).total_seconds()
        return elapsed > self.config.optuna_tune_interval_hours * 3600

    def train(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """Train the model on historical data.

        Args:
            df_5m: 5-minute OHLCV DataFrame
            higher_tf_data: Optional higher timeframe data

        Returns:
            Dict with training metrics
        """
        logger.info(f"Training model on {len(df_5m)} candles...")

        # Create labels before computing features (need close column)
        labels = self.feature_engineer.create_labels(df_5m)

        # Compute features
        features_df = self.feature_engineer.compute_features(df_5m, higher_tf_data)

        if features_df.empty:
            raise ValueError("Feature computation returned empty DataFrame")

        # Align labels with features (drop last row since label is NaN)
        min_len = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_len]
        labels = labels.iloc[:min_len]

        # Align indices
        valid_idx = features_df.dropna().index.intersection(labels.dropna().index)
        X = features_df.loc[valid_idx]
        y = labels.loc[valid_idx]

        if len(X) < 200:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need >= 200)")

        self.feature_names = list(X.columns)
        logger.info(f"Training with {len(X)} samples, {len(self.feature_names)} features")

        # --- Improvement 5: Optuna tuning if needed ---
        if self.needs_tuning():
            self.tune_hyperparameters(X, y)

        xgb_params = self._get_xgb_params()

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = XGBClassifier(**xgb_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = fold_model.predict(X_val)
            fold_acc = accuracy_score(y_val, val_pred)
            cv_scores.append(fold_acc)
            logger.info(f"  Fold {fold + 1}/5: val_accuracy={fold_acc:.4f}")

        avg_cv = np.mean(cv_scores)
        logger.info(f"Average CV accuracy: {avg_cv:.4f}")

        # Train final model on all data
        # Use 80/20 chronological split for final metrics
        split_idx = int(len(X) * 0.8)
        X_train_final = X.iloc[:split_idx]
        y_train_final = y.iloc[:split_idx]
        X_val_final = X.iloc[split_idx:]
        y_val_final = y.iloc[split_idx:]

        new_model = XGBClassifier(**xgb_params)
        new_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False,
        )

        # Evaluate
        train_pred = new_model.predict(X_train_final)
        val_pred = new_model.predict(X_val_final)
        val_proba = new_model.predict_proba(X_val_final)[:, 1]

        new_train_accuracy = accuracy_score(y_train_final, train_pred)
        new_val_accuracy = accuracy_score(y_val_final, val_pred)
        new_val_logloss = log_loss(y_val_final, val_proba)

        # --- Improvement 4: Walk-forward retraining gate ---
        # Only swap to new model if it improves on the current one
        swapped = True
        if self.model is not None:
            improvement = new_val_accuracy - self.val_accuracy
            if improvement < self.config.retrain_min_improvement:
                logger.info(
                    f"Retraining gate: new val_acc={new_val_accuracy:.4f} vs "
                    f"current={self.val_accuracy:.4f} (improvement={improvement:+.4f} "
                    f"< threshold={self.config.retrain_min_improvement}). Keeping current model."
                )
                swapped = False
            else:
                logger.info(
                    f"Retraining gate: PASSED. new val_acc={new_val_accuracy:.4f} vs "
                    f"current={self.val_accuracy:.4f} (improvement={improvement:+.4f})"
                )

        if swapped:
            self.model = new_model
            self.train_accuracy = new_train_accuracy
            self.val_accuracy = new_val_accuracy
            self.val_logloss = new_val_logloss
            self.train_samples = len(X)
            self.last_train_time = datetime.now(timezone.utc)
        else:
            # Still update the train time so we don't re-trigger immediately
            self.last_train_time = datetime.now(timezone.utc)

        # Class distribution in validation
        val_class_dist = y_val_final.value_counts().to_dict()

        metrics = {
            "train_accuracy": new_train_accuracy,
            "val_accuracy": new_val_accuracy,
            "cv_accuracy": avg_cv,
            "cv_scores": cv_scores,
            "val_logloss": new_val_logloss,
            "train_samples": len(X_train_final),
            "val_samples": len(X_val_final),
            "total_samples": len(X),
            "n_features": len(self.feature_names),
            "val_class_distribution": val_class_dist,
            "train_time": datetime.now(timezone.utc).isoformat(),
            "model_swapped": swapped,
            "active_val_accuracy": self.val_accuracy,  # The model actually in use
            "optuna_tuned": self.best_xgb_params is not None,
        }

        logger.info(f"New model - Train acc: {new_train_accuracy:.4f}, Val acc: {new_val_accuracy:.4f}")
        logger.info(f"Active model val acc: {self.val_accuracy:.4f} (swapped={swapped})")
        logger.info(f"Val log loss: {new_val_logloss:.4f}")

        # Feature importance
        importances = new_model.feature_importances_
        top_features = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
        logger.info("Top 15 features:")
        for fname, imp in top_features:
            logger.info(f"  {fname}: {imp:.4f}")
        metrics["top_features"] = {f: float(i) for f, i in top_features}

        return metrics

    def train_for_comparison(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """Train a candidate model and store it as pending WITHOUT swapping.

        Returns comparison metrics so the user can decide whether to
        keep the current model or swap to the new one.

        Args:
            df_5m: 5-minute OHLCV DataFrame
            higher_tf_data: Optional higher timeframe data

        Returns:
            Dict with old vs new metrics for comparison display
        """
        logger.info(f"Training candidate model on {len(df_5m)} candles (comparison mode)...")

        # Create labels before computing features
        labels = self.feature_engineer.create_labels(df_5m)
        features_df = self.feature_engineer.compute_features(df_5m, higher_tf_data)

        if features_df.empty:
            raise ValueError("Feature computation returned empty DataFrame")

        min_len = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_len]
        labels = labels.iloc[:min_len]

        valid_idx = features_df.dropna().index.intersection(labels.dropna().index)
        X = features_df.loc[valid_idx]
        y = labels.loc[valid_idx]

        if len(X) < 200:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need >= 200)")

        self.feature_names = list(X.columns)
        logger.info(f"Comparison training with {len(X)} samples, {len(self.feature_names)} features")

        if self.needs_tuning():
            self.tune_hyperparameters(X, y)

        xgb_params = self._get_xgb_params()

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_model = XGBClassifier(**xgb_params)
            fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = fold_model.predict(X_val)
            cv_scores.append(accuracy_score(y_val, val_pred))

        avg_cv = np.mean(cv_scores)

        # Train final candidate on 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train_final = X.iloc[:split_idx]
        y_train_final = y.iloc[:split_idx]
        X_val_final = X.iloc[split_idx:]
        y_val_final = y.iloc[split_idx:]

        candidate = XGBClassifier(**xgb_params)
        candidate.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False,
        )

        train_pred = candidate.predict(X_train_final)
        val_pred = candidate.predict(X_val_final)
        val_proba = candidate.predict_proba(X_val_final)[:, 1]

        new_train_acc = accuracy_score(y_train_final, train_pred)
        new_val_acc = accuracy_score(y_val_final, val_pred)
        new_val_logloss = log_loss(y_val_final, val_proba)

        # Store as pending (do NOT swap yet)
        self._pending_model = candidate
        self._pending_metrics = {
            "train_accuracy": new_train_acc,
            "val_accuracy": new_val_acc,
            "cv_accuracy": avg_cv,
            "cv_scores": cv_scores,
            "val_logloss": new_val_logloss,
            "train_samples": len(X_train_final),
            "val_samples": len(X_val_final),
            "total_samples": len(X),
            "n_features": len(self.feature_names),
            "optuna_tuned": self.best_xgb_params is not None,
        }

        # Build comparison result
        comparison = {
            "old_val_accuracy": self.val_accuracy,
            "old_train_accuracy": self.train_accuracy,
            "old_val_logloss": self.val_logloss,
            "old_train_samples": self.train_samples,
            "new_val_accuracy": new_val_acc,
            "new_train_accuracy": new_train_acc,
            "new_val_logloss": new_val_logloss,
            "new_cv_accuracy": avg_cv,
            "new_total_samples": len(X),
            "new_n_features": len(self.feature_names),
            "improvement": new_val_acc - self.val_accuracy,
            "optuna_tuned": self.best_xgb_params is not None,
            "has_existing_model": self.model is not None,
        }

        logger.info(
            f"Candidate ready: new_val_acc={new_val_acc:.4f} vs "
            f"current={self.val_accuracy:.4f} (delta={comparison['improvement']:+.4f})"
        )
        return comparison

    def apply_pending_model(self) -> dict:
        """Swap the pending candidate model into production.

        Returns:
            Dict with the applied model's metrics
        """
        if self._pending_model is None or self._pending_metrics is None:
            raise RuntimeError("No pending model to apply")

        metrics = self._pending_metrics
        self.model = self._pending_model
        self.train_accuracy = metrics["train_accuracy"]
        self.val_accuracy = metrics["val_accuracy"]
        self.val_logloss = metrics["val_logloss"]
        self.train_samples = metrics["total_samples"]
        self.last_train_time = datetime.now(timezone.utc)

        # Clear pending state
        self._pending_model = None
        self._pending_metrics = None

        logger.info(f"Pending model applied: val_acc={self.val_accuracy:.4f}")
        return {
            "action": "swap",
            "val_accuracy": self.val_accuracy,
            "train_accuracy": self.train_accuracy,
            "val_logloss": self.val_logloss,
        }

    def reject_pending_model(self) -> dict:
        """Discard the pending candidate model and keep the current one.

        Returns:
            Dict confirming the kept model's metrics
        """
        old_pending = self._pending_metrics
        self._pending_model = None
        self._pending_metrics = None
        # Still update train time to avoid immediate re-trigger
        self.last_train_time = datetime.now(timezone.utc)

        logger.info(f"Pending model rejected, keeping val_acc={self.val_accuracy:.4f}")
        return {
            "action": "keep",
            "val_accuracy": self.val_accuracy,
            "rejected_val_accuracy": old_pending["val_accuracy"] if old_pending else 0.0,
        }

    def predict(self, df_5m: pd.DataFrame, higher_tf_data: Optional[dict[str, pd.DataFrame]] = None) -> dict:
        """Make a prediction for the next candle.

        Improvement 2: Uses confidence_min filtering — predictions below
        the minimum confidence threshold are returned as NEUTRAL/SKIP.

        Args:
            df_5m: Recent 5-minute OHLCV data
            higher_tf_data: Optional higher timeframe data

        Returns:
            Dict with prediction, confidence, and signal info
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        features_df = self.feature_engineer.compute_features(df_5m, higher_tf_data)

        if features_df.empty:
            return {"signal": "SKIP", "confidence": 0.0, "reason": "Insufficient data"}

        # Use the last row (most recent candle)
        latest = features_df.iloc[[-1]]

        # Ensure feature alignment
        missing_feats = set(self.feature_names) - set(latest.columns)
        for feat in missing_feats:
            latest[feat] = 0.0
        latest = latest[self.feature_names]

        # Handle any NaN in the latest row
        if latest.isna().any().any():
            latest = latest.fillna(0)

        # Predict
        proba = self.model.predict_proba(latest)[0]
        prob_up = float(proba[1])
        prob_down = float(proba[0])

        # --- Improvement 2: Confidence filtering ---
        confidence = max(prob_up, prob_down)
        confidence_min = self.config.confidence_min
        confidence_strong = self.config.confidence_strong

        if prob_up >= confidence_min and prob_up > prob_down:
            signal = "UP"
            direction_confidence = prob_up
            strength = "STRONG" if prob_up >= confidence_strong else "NORMAL"
        elif prob_down >= confidence_min and prob_down > prob_up:
            signal = "DOWN"
            direction_confidence = prob_down
            strength = "STRONG" if prob_down >= confidence_strong else "NORMAL"
        else:
            signal = "NEUTRAL"
            direction_confidence = confidence
            strength = "SKIP"

        current_price = float(df_5m["close"].iloc[-1])

        result = {
            "signal": signal,
            "confidence": round(direction_confidence, 4),
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "current_price": current_price,
            "model_accuracy": self.val_accuracy,
            "strength": strength,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Prediction: {signal} [{strength}] "
            f"(confidence={direction_confidence:.4f}, min={confidence_min}, price={current_price})"
        )
        return result

    def needs_retrain(self) -> bool:
        """Check if model needs retraining based on time interval."""
        if self.model is None or self.last_train_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_train_time).total_seconds()
        return elapsed > self.config.retrain_interval_hours * 3600

    def save(self, model_dir: str) -> str:
        """Save model to disk."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "xgb_model.pkl")
        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "last_train_time": self.last_train_time,
            "last_tune_time": self.last_tune_time,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "val_logloss": self.val_logloss,
            "train_samples": self.train_samples,
            "best_xgb_params": self.best_xgb_params,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, model_dir: str) -> bool:
        """Load model from disk."""
        path = os.path.join(model_dir, "xgb_model.pkl")
        if not os.path.exists(path):
            logger.info("No saved model found")
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.model = state["model"]
            self.feature_names = state["feature_names"]
            self.last_train_time = state["last_train_time"]
            self.last_tune_time = state.get("last_tune_time")
            self.train_accuracy = state["train_accuracy"]
            self.val_accuracy = state["val_accuracy"]
            self.val_logloss = state.get("val_logloss", 1.0)
            self.train_samples = state["train_samples"]
            self.best_xgb_params = state.get("best_xgb_params")
            logger.info(f"Model loaded from {path} (val_acc={self.val_accuracy:.4f})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
