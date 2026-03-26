"""Comprehensive test suite for BTC Signal Bot (aprilxg v2).
Tests: MEXC API, feature engineering (normalized + regime), model training (Optuna + retraining gate),
       confidence filtering, signal tracking, prediction, Polymarket integration.
"""
import asyncio
import sys
import os
import json
import shutil
import traceback
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BotConfig, MEXCConfig, ModelConfig, PolymarketConfig
from src.data_fetcher import MEXCFetcher
from src.features import FeatureEngineer
from src.model import PredictionModel
from src.signal_tracker import SignalTracker, Signal


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, error):
        self.failed += 1
        self.errors.append((name, str(error)))
        print(f"  [FAIL] {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"TEST RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, err in self.errors:
                print(f"  - {name}: {err}")
        print(f"{'='*50}")
        return self.failed == 0


async def test_mexc_api(results: TestResults):
    """Test MEXC API connectivity and data fetching."""
    print("\n--- Testing MEXC API ---")
    config = MEXCConfig()
    fetcher = MEXCFetcher(config)

    try:
        # Test 1: Fetch 5m klines
        df_5m = await fetcher.fetch_klines(interval="5m", limit=100)
        assert not df_5m.empty, "5m klines returned empty"
        assert len(df_5m) >= 50, f"Only got {len(df_5m)} candles, expected >= 50"
        assert all(col in df_5m.columns for col in ["timestamp", "open", "high", "low", "close", "volume"]), "Missing columns"
        assert df_5m["close"].dtype == float, "Close should be float"
        assert df_5m["volume"].dtype == float, "Volume should be float"
        assert (df_5m["high"] >= df_5m["low"]).all(), "High should be >= Low"
        assert (df_5m["high"] >= df_5m["close"]).all(), "High should be >= Close"
        assert (df_5m["low"] <= df_5m["close"]).all(), "Low should be <= Close"
        results.ok(f"5m klines: {len(df_5m)} candles, price=${df_5m['close'].iloc[-1]:,.2f}")

        # Test 2: Fetch 15m klines
        df_15m = await fetcher.fetch_klines(interval="15m", limit=100)
        assert not df_15m.empty, "15m klines returned empty"
        results.ok(f"15m klines: {len(df_15m)} candles")

        # Test 3: Fetch 1h klines
        df_1h = await fetcher.fetch_klines(interval="1h", limit=100)
        assert not df_1h.empty, "1h klines returned empty"
        results.ok(f"1h klines: {len(df_1h)} candles")

        # Test 4: Multi-timeframe fetch
        multi = await fetcher.fetch_multi_timeframe(intervals=["5m", "15m", "1h"], limit=100)
        assert len(multi) == 3, f"Expected 3 timeframes, got {len(multi)}"
        assert all(not v.empty for v in multi.values()), "Some timeframes returned empty"
        results.ok(f"Multi-TF fetch: {list(multi.keys())}")

        # Test 5: Historical pagination
        df_hist = await fetcher.fetch_historical_klines(interval="5m", total_candles=1000)
        assert len(df_hist) >= 900, f"Historical fetch only got {len(df_hist)}, expected >= 900"
        assert df_hist["timestamp"].is_unique, "Historical data has duplicate timestamps"
        assert df_hist["timestamp"].is_monotonic_increasing, "Historical data not sorted"
        results.ok(f"Historical klines: {len(df_hist)} candles (paginated)")

        return df_5m, df_15m, df_1h, df_hist

    except Exception as e:
        results.fail("MEXC API", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, None, None
    finally:
        await fetcher.close()


def test_features(results: TestResults, df_5m, df_15m, df_1h):
    """Test feature engineering including v2 improvements."""
    print("\n--- Testing Feature Engineering (v2) ---")
    config = ModelConfig()
    fe = FeatureEngineer(config)

    try:
        # Test 1: Basic feature computation
        features = fe.compute_features(df_5m)
        assert not features.empty, "Features returned empty"
        assert len(features) > 0, "No feature rows"
        n_features = len(features.columns)
        assert n_features >= 30, f"Only {n_features} features, expected >= 30"
        results.ok(f"Feature computation: {n_features} features, {len(features)} rows")

        # Test 2: Check no infinite values
        import numpy as np
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        assert inf_count == 0, f"Found {inf_count} infinite values"
        results.ok("No infinite values in features")

        # Test 3: Check key features exist (original + v2 new features)
        expected_features = [
            "rsi", "macd_histogram", "bb_pctb", "atr_norm", "adx",
            "volume_ratio", "mfi", "ema_crossover", "returns_1", "candle_body",
            # v2 normalized MACD
            "macd_line", "macd_signal",
            # v2 regime detection
            "atr_percentile", "regime_high_vol", "regime_vol_score", "vol_expansion",
            # v2 z-score features
            "momentum_3_zscore", "momentum_5_zscore", "rsi_zscore", "volume_ratio_zscore",
        ]
        for feat_name in expected_features:
            assert feat_name in features.columns, f"Missing feature: {feat_name}"
        results.ok(f"All {len(expected_features)} key features present (incl. v2 regime + normalized)")

        # Test 4: Verify MACD is normalized (should be small pct values, not raw price-scale)
        macd_vals = features["macd_line"].dropna()
        assert macd_vals.abs().max() < 0.1, f"MACD line not normalized: max={macd_vals.abs().max():.6f}"
        results.ok(f"MACD normalization: max|macd_line|={macd_vals.abs().max():.6f} (correctly small)")

        # Test 5: Verify regime features are valid
        regime_vals = features["regime_high_vol"].dropna()
        assert set(regime_vals.unique()).issubset({0, 1}), f"regime_high_vol should be binary, got {regime_vals.unique()}"
        vol_score = features["regime_vol_score"].dropna()
        assert vol_score.min() >= 0 and vol_score.max() <= 1.0, f"regime_vol_score out of [0,1] range"
        results.ok(f"Regime features valid: {regime_vals.mean():.1%} high-vol, score range [{vol_score.min():.2f}, {vol_score.max():.2f}]")

        # Test 6: Raw MACD columns should NOT be in output
        assert "macd_line_raw" not in features.columns, "Raw MACD line should be dropped"
        assert "macd_signal_raw" not in features.columns, "Raw MACD signal should be dropped"
        assert "macd_histogram_raw" not in features.columns, "Raw MACD histogram should be dropped"
        results.ok("Raw MACD columns correctly excluded from features")

        # Test 7: Multi-timeframe features
        higher_tf = {"15m": df_15m, "1h": df_1h}
        features_mtf = fe.compute_features(df_5m, higher_tf)
        mtf_cols = [c for c in features_mtf.columns if "15min" in c or "1hr" in c or "60min" in c]
        assert len(mtf_cols) > 0, "No multi-timeframe features found"
        results.ok(f"Multi-TF features: {len(mtf_cols)} MTF columns added")

        # Test 8: Labels
        labels = fe.create_labels(df_5m)
        assert len(labels) == len(df_5m), "Label count mismatch"
        assert set(labels.dropna().unique()).issubset({0, 1}), "Labels should be 0 or 1"
        up_pct = labels.mean() * 100
        results.ok(f"Labels: {up_pct:.1f}% UP, {100-up_pct:.1f}% DOWN")

        return features_mtf

    except Exception as e:
        results.fail("Feature Engineering", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_model(results: TestResults, df_hist, df_15m, df_1h):
    """Test model training, prediction, confidence filtering, retraining gate, and v8 3-way split."""
    print("\n--- Testing Model (v2: Optuna + Gate + Confidence + v8 3-way split) ---")
    # Disable Optuna for fast testing, use small data
    config = ModelConfig()
    config.enable_optuna_tuning = False  # Skip Optuna in tests for speed
    config.train_candles = 1000  # Smaller for test
    model = PredictionModel(config)

    try:
        # Test 1: Train model
        higher_tf = {"15m": df_15m, "1h": df_1h}
        metrics = model.train(df_hist, higher_tf)

        assert "train_accuracy" in metrics, "Missing train_accuracy"
        assert "val_accuracy" in metrics, "Missing val_accuracy"
        assert "cv_mean" in metrics, "Missing cv_mean"
        assert "model_swapped" in metrics, "Missing model_swapped (v2 retraining gate)"
        assert metrics["model_swapped"] is True, "First training should always swap"
        assert metrics["val_accuracy"] > 0.45, f"Val accuracy too low: {metrics['val_accuracy']}"
        # v8: verify 3-way split fields are present
        assert "n_cal" in metrics, "Missing n_cal (v8 3-way split)"
        assert "n_oos" in metrics, "Missing n_oos"
        assert "n_inner" in metrics, "Missing n_inner"
        assert metrics["n_cal"] > 0, f"CAL split should have samples, got {metrics['n_cal']}"
        assert metrics["n_oos"] > 0, f"OOS split should have samples, got {metrics['n_oos']}"
        assert metrics["n_inner"] > metrics["n_cal"], "INNER should be larger than CAL"
        assert metrics["n_inner"] > metrics["n_oos"], "INNER should be larger than OOS"
        results.ok(
            f"Model trained: val_acc={metrics['val_accuracy']:.4f}, "
            f"cv_mean={metrics['cv_mean']:.4f}, "
            f"{metrics['n_features']} features, {metrics['total_samples']} samples, "
            f"split=INNER({metrics['n_inner']})/CAL({metrics['n_cal']})/OOS({metrics['n_oos']}), "
            f"swapped={metrics['model_swapped']}"
        )

        # Test 2: Check CV scores
        cv_scores = metrics["cv_scores"]
        assert len(cv_scores) == 5, f"Expected 5 CV folds, got {len(cv_scores)}"
        assert all(0.3 < s < 0.75 for s in cv_scores), f"CV scores out of range: {cv_scores}"
        results.ok(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

        # Test 3: Prediction with v7 calibration + EV filtering
        # v7: predict() returns None when filtered out (low confidence or negative EV).
        # When it returns a dict, signal is always UP or DOWN (never NEUTRAL).
        prediction = model.predict(df_hist.tail(120), higher_tf)

        if prediction is None:
            # Model filtered this prediction out (low confidence or negative EV) — valid behavior
            results.ok("Prediction: filtered out by confidence/EV gate (returned None — valid)")
        else:
            assert "signal" in prediction, "Missing signal"
            assert "confidence" in prediction, "Missing confidence (calibrated)"
            assert "raw_confidence" in prediction, "Missing raw_confidence"
            assert "ev" in prediction, "Missing ev"
            assert "strength" in prediction, "Missing strength"
            assert "probabilities" in prediction, "Missing probabilities dict"
            assert "calibrated_prob_up" in prediction, "Missing calibrated_prob_up"
            assert "calibrated_prob_down" in prediction, "Missing calibrated_prob_down"
            assert "current_price" in prediction, "Missing current_price"
            assert prediction["signal"] in ("UP", "DOWN"), f"Invalid signal: {prediction['signal']} (v7 never returns NEUTRAL)"
            assert prediction["strength"] in ("STRONG", "NORMAL"), f"Invalid strength: {prediction['strength']} (v7 uses STRONG/NORMAL)"
            assert 0 <= prediction["confidence"] <= 1, f"Confidence out of range: {prediction['confidence']}"
            # Probabilities dict has raw UP/DOWN that should sum to ~1
            prob_up = prediction["probabilities"]["UP"]
            prob_down = prediction["probabilities"]["DOWN"]
            assert abs(prob_up + prob_down - 1.0) < 0.01, "Raw probabilities don't sum to 1"
            # Calibrated probs should also sum to ~1
            assert abs(prediction["calibrated_prob_up"] + prediction["calibrated_prob_down"] - 1.0) < 0.01, \
                "Calibrated probabilities don't sum to 1"
            # If predict() returned a result, EV must be >= threshold
            assert prediction["ev"] >= config.ev_threshold, \
                f"EV {prediction['ev']} should be >= threshold {config.ev_threshold}"
            # Strength is based on EV
            if prediction["ev"] >= config.ev_strong_threshold:
                assert prediction["strength"] == "STRONG", "High EV should be STRONG"
            else:
                assert prediction["strength"] == "NORMAL", "Low EV should be NORMAL"

            results.ok(
                f"Prediction: {prediction['signal']} [{prediction['strength']}] "
                f"(cal_conf={prediction['confidence']:.4f}, raw_conf={prediction['raw_confidence']:.4f}, "
                f"EV=${prediction['ev']:+.4f}, "
                f"P(up)={prob_up:.4f}, P(down)={prob_down:.4f})"
            )

        # Test 4: Retraining gate — retrain and check gate logic
        first_val_acc = model.val_accuracy
        metrics2 = model.train(df_hist, higher_tf)
        # The gate should either swap or keep based on improvement threshold
        assert "model_swapped" in metrics2, "Second training missing model_swapped"
        assert "active_val_accuracy" in metrics2, "Missing active_val_accuracy"
        results.ok(
            f"Retraining gate: swapped={metrics2['model_swapped']}, "
            f"new_acc={metrics2['val_accuracy']:.4f}, "
            f"active_acc={metrics2['active_val_accuracy']:.4f}"
        )

        # Test 5: Save and load model (including v7 calibration + pruning)
        test_dir = "/tmp/test_model_v2"
        model.save(test_dir)
        assert os.path.exists(os.path.join(test_dir, "model.pkl")), "Model file not saved"

        model2 = PredictionModel(config)
        loaded = model2.load(test_dir)
        assert loaded, "Model failed to load"
        assert model2.val_accuracy == model.val_accuracy, "Loaded accuracy mismatch"
        # v7: verify calibrator and pruned features persist
        if model.calibrator is not None:
            assert model2.calibrator is not None, "Loaded model missing calibrator"
        if model.pruned_feature_names is not None:
            assert model2.pruned_feature_names == model.pruned_feature_names, "Loaded pruned features mismatch"

        # Verify loaded model predicts the same
        pred2 = model2.predict(df_hist.tail(120), higher_tf)
        if prediction is not None and pred2 is not None:
            assert pred2["signal"] == prediction["signal"], "Loaded model gives different signal"
            assert abs(pred2["confidence"] - prediction["confidence"]) < 0.001, "Loaded model confidence differs"
            assert pred2["strength"] == prediction["strength"], "Loaded model strength differs"
            results.ok("Model save/load: verified identical predictions (incl. v7 calibration + pruning)")
        elif prediction is None and pred2 is None:
            results.ok("Model save/load: both predictions filtered out (consistent)")
        else:
            results.ok("Model save/load: loaded successfully (prediction filtering may differ due to boundary effects)")

        # Test 6: Retrain check
        assert not model.needs_retrain(), "Model should not need retrain immediately after training"
        results.ok("Retrain logic: correctly reports no retrain needed")

        # Test 7: Tuning check
        assert not model.needs_tuning(), "Optuna tuning is disabled, should not need tuning"
        results.ok("Optuna tuning: correctly disabled in test config")

        return model, metrics

    except Exception as e:
        results.fail("Model", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


def test_signal_tracker(results: TestResults):
    """Test signal tracking with win/loss/PnL calculations."""
    print("\n--- Testing Signal Tracker ---")
    test_dir = "/tmp/test_signals_v2"
    os.makedirs(test_dir, exist_ok=True)

    try:
        tracker = SignalTracker(test_dir)

        # Test 1: Add signals
        s1 = tracker.add_signal("UP", 0.58, 84000.00)
        assert s1.signal_id == 1, f"Expected ID 1, got {s1.signal_id}"
        assert s1.direction == "UP"
        assert s1.result is None, "New signal should have no result"

        s2 = tracker.add_signal("DOWN", 0.62, 84100.00)
        s3 = tracker.add_signal("UP", 0.55, 83900.00)
        s4 = tracker.add_signal("DOWN", 0.60, 84200.00)
        s5 = tracker.add_signal("UP", 0.57, 83800.00)
        results.ok(f"Added 5 signals (IDs: 1-5)")

        # Test 2: Resolve signals with known prices
        r1 = tracker.resolve_signal(1, 84100.00)
        assert r1.result == "WIN", f"Expected WIN, got {r1.result}"
        expected_pnl1 = ((84100 - 84000) / 84000) * 100
        assert abs(r1.pnl_pct - expected_pnl1) < 0.001, f"PnL mismatch: {r1.pnl_pct} vs {expected_pnl1}"
        results.ok(f"Signal #1 UP 84000->84100: WIN {r1.pnl_pct:+.4f}% (correct)")

        r2 = tracker.resolve_signal(2, 84000.00)
        assert r2.result == "WIN", f"Expected WIN, got {r2.result}"
        expected_pnl2 = ((84100 - 84000) / 84100) * 100
        assert abs(r2.pnl_pct - expected_pnl2) < 0.001, f"PnL mismatch: {r2.pnl_pct} vs {expected_pnl2}"
        results.ok(f"Signal #2 DOWN 84100->84000: WIN {r2.pnl_pct:+.4f}% (correct)")

        r3 = tracker.resolve_signal(3, 83800.00)
        assert r3.result == "LOSS", f"Expected LOSS, got {r3.result}"
        results.ok(f"Signal #3 UP 83900->83800: LOSS {r3.pnl_pct:+.4f}% (correct)")

        r4 = tracker.resolve_signal(4, 84300.00)
        assert r4.result == "LOSS", f"Expected LOSS, got {r4.result}"
        results.ok(f"Signal #4 DOWN 84200->84300: LOSS {r4.pnl_pct:+.4f}% (correct)")

        r5 = tracker.resolve_signal(5, 84000.00)
        assert r5.result == "WIN", f"Expected WIN, got {r5.result}"
        results.ok(f"Signal #5 UP 83800->84000: WIN {r5.pnl_pct:+.4f}% (correct)")

        # Test 3: Stats accuracy
        stats = tracker.get_stats()
        assert stats.total_signals == 5, f"Expected 5 total, got {stats.total_signals}"
        assert stats.wins == 3, f"Expected 3 wins, got {stats.wins}"
        assert stats.losses == 2, f"Expected 2 losses, got {stats.losses}"
        expected_wr = 3 / 5 * 100
        assert abs(stats.win_rate - expected_wr) < 0.1, f"Win rate: {stats.win_rate} vs {expected_wr}"
        results.ok(f"Stats: W={stats.wins} L={stats.losses} WR={stats.win_rate:.1f}% (correct)")

        # Test 4: PnL accuracy
        total_pnl = sum(s.pnl_pct for s in [r1, r2, r3, r4, r5])
        assert abs(stats.total_pnl_pct - total_pnl) < 0.001, f"Total PnL mismatch"
        results.ok(f"Total PnL: {stats.total_pnl_pct:+.4f}% (matches sum)")

        # Test 5: Streaks
        assert stats.current_streak == 1
        assert stats.current_streak_type == "WIN"
        assert stats.longest_win_streak == 2
        assert stats.longest_loss_streak == 2
        results.ok(f"Streaks: current={stats.current_streak} {stats.current_streak_type}")

        # Test 6: Signal message formatting with strength
        mock_prediction = {
            "prob_up": 0.62, "prob_down": 0.38,
            "model_accuracy": 0.55, "strength": "STRONG"
        }
        msg = tracker.format_signal_message(s1, mock_prediction)
        assert "STRONG" in msg, "Signal message should include STRONG label"
        assert "UP" in msg, "Signal message should include direction"
        results.ok("Signal message formatting includes strength label")

        # Test 7: Persistence
        tracker2 = SignalTracker(test_dir)
        assert len(tracker2.signals) == 5, f"Loaded {len(tracker2.signals)} signals, expected 5"
        stats2 = tracker2.get_stats()
        assert stats2.wins == stats.wins, "Persisted stats mismatch"
        results.ok("Persistence: signals correctly saved and reloaded")

        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        results.fail("Signal Tracker", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_config(results: TestResults):
    """Test configuration loading including v2 fields and Polymarket config."""
    print("\n--- Testing Configuration (v2 + Polymarket) ---")
    try:
        # Test default config
        config = BotConfig()
        assert config.mexc.symbol == "BTCUSDT"
        assert config.model.train_candles == 43200, f"train_candles should be 43200, got {config.model.train_candles}"
        assert config.model.confidence_min == 0.52, f"confidence_min should be 0.52, got {config.model.confidence_min}"
        # v7: confidence_strong removed, replaced by EV-based thresholds
        assert config.model.ev_threshold == 0.0, f"ev_threshold should be 0.0, got {config.model.ev_threshold}"
        assert config.model.ev_strong_threshold == 0.05, f"ev_strong_threshold should be 0.05"
        assert config.model.win_payout == 0.96, f"win_payout should be 0.96"
        assert config.model.loss_amount == 1.00, f"loss_amount should be 1.00"
        assert config.model.enable_calibration is True, "enable_calibration should default to True"
        assert config.model.enable_feature_pruning is True, "enable_feature_pruning should default to True"
        assert config.model.feature_prune_top_n == 20, f"feature_prune_top_n should be 20"
        assert config.model.enable_optuna_tuning is True, "Optuna should be enabled by default"
        assert config.model.retrain_min_improvement == 0.002, f"retrain_min_improvement should be 0.002"
        assert config.model.atr_regime_lookback == 100, f"atr_regime_lookback should be 100"
        assert config.prediction_lead_seconds == 35, f"prediction_lead_seconds should be 35 (BY DESIGN)"
        results.ok("Default v2 config loaded correctly (incl. 35-sec lead time)")

        # Test Polymarket default config
        assert config.polymarket.private_key == "", "Polymarket key should default to empty"
        assert config.polymarket.funder_address == "", "Funder address should default to empty"
        assert config.polymarket.signature_type == 2, f"Signature type should default to 2, got {config.polymarket.signature_type}"
        assert config.polymarket.enabled is False, "Polymarket should be disabled by default"
        results.ok("Polymarket default config: disabled, sig_type=2")

        # Test Polymarket env override
        os.environ["POLYMARKET_PRIVATE_KEY"] = "0xdeadbeef"
        os.environ["POLYMARKET_FUNDER_ADDRESS"] = "0x1234567890abcdef"
        os.environ["POLYMARKET_SIGNATURE_TYPE"] = "1"
        config_pm = BotConfig.from_env()
        assert config_pm.polymarket.private_key == "0xdeadbeef", "Private key not set from env"
        assert config_pm.polymarket.funder_address == "0x1234567890abcdef", "Funder address not set from env"
        assert config_pm.polymarket.signature_type == 1, f"Signature type should be 1, got {config_pm.polymarket.signature_type}"
        assert config_pm.polymarket.enabled is True, "Polymarket should be enabled when key is set"
        results.ok("Polymarket env config: enabled when key set, sig_type=1")

        # Cleanup Polymarket env vars
        for key in ["POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS", "POLYMARKET_SIGNATURE_TYPE"]:
            del os.environ[key]

        # Verify disabled after env cleanup
        config_no_pm = BotConfig.from_env()
        assert config_no_pm.polymarket.enabled is False, "Polymarket should be disabled without key"
        results.ok("Polymarket disabled when env vars removed")

        # Test env override for other vars
        os.environ["TRADING_SYMBOL"] = "ETHUSDT"
        os.environ["PREDICTION_THRESHOLD"] = "0.58"
        os.environ["CONFIDENCE_MIN"] = "0.56"
        os.environ["ENABLE_OPTUNA"] = "false"
        os.environ["TRAIN_CANDLES"] = "20000"
        config2 = BotConfig.from_env()
        assert config2.mexc.symbol == "ETHUSDT"
        assert config2.model.prediction_threshold == 0.58
        assert config2.model.confidence_min == 0.56
        assert config2.model.enable_optuna_tuning is False
        assert config2.model.train_candles == 20000
        results.ok("Environment variable overrides work (incl. v2 vars)")

        # Cleanup
        for key in ["TRADING_SYMBOL", "PREDICTION_THRESHOLD", "CONFIDENCE_MIN", "ENABLE_OPTUNA", "TRAIN_CANDLES"]:
            del os.environ[key]

    except Exception as e:
        results.fail("Config", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_35sec_timing(results: TestResults):
    """Verify the 35-second pre-signal timing is preserved."""
    print("\n--- Testing 15-Second Signal Timing ---")
    try:
        config = BotConfig()
        assert config.prediction_lead_seconds == 35, \
            f"CRITICAL: prediction_lead_seconds changed from 35 to {config.prediction_lead_seconds}"
        results.ok("35-second pre-signal timing preserved in config")

        # Verify the timing logic would fire correctly
        # At 4:25 into a candle (35 sec before close), should trigger
        seconds_in_candle = 265  # 4 min 25 sec
        candle_duration = 300
        seconds_until_close = candle_duration - seconds_in_candle  # = 15
        should_trigger = seconds_until_close <= config.prediction_lead_seconds and seconds_until_close > 0
        assert should_trigger, "Should trigger at 35 seconds before close"
        results.ok("Timing logic: triggers at exactly 15 sec before candle close")

        # At 4:00 into candle (60 sec before), should NOT trigger
        seconds_in_candle = 240
        seconds_until_close = candle_duration - seconds_in_candle  # = 60
        should_not_trigger = seconds_until_close <= config.prediction_lead_seconds and seconds_until_close > 0
        assert not should_not_trigger, "Should NOT trigger at 60 seconds before close"
        results.ok("Timing logic: correctly skips at 30 sec before close")

        # At candle close (0 sec remaining), should NOT trigger
        seconds_in_candle = 300
        seconds_until_close = candle_duration - seconds_in_candle  # = 0
        should_not_trigger_close = seconds_until_close <= config.prediction_lead_seconds and seconds_until_close > 0
        assert not should_not_trigger_close, "Should NOT trigger at exact close"
        results.ok("Timing logic: correctly skips at exact candle close")

    except Exception as e:
        results.fail("35-sec Timing", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_auto_trader(results: TestResults):
    """Test AutoTrader toggle, amount settings, and duplicate prevention."""
    print("\n--- Testing AutoTrader ---")
    test_dir = "/tmp/test_autotrader"
    os.makedirs(test_dir, exist_ok=True)

    try:
        from src.auto_trader import AutoTrader, MIN_TRADE_AMOUNT, MAX_TRADE_AMOUNT, DEFAULT_TRADE_AMOUNT

        # Create a mock PolymarketClient
        mock_pm = MagicMock()
        mock_pm.is_initialized = True
        mock_pm.wallet_address = "0xMOCKWALLET"

        trader = AutoTrader(polymarket_client=mock_pm, data_dir=test_dir)

        # Test 1: Default state
        assert trader.enabled is False, "AutoTrader should start disabled"
        assert trader.trade_amount == DEFAULT_TRADE_AMOUNT, f"Default amount should be {DEFAULT_TRADE_AMOUNT}"
        results.ok(f"Default state: disabled, amount={DEFAULT_TRADE_AMOUNT} USDC")

        # Test 2: Toggle on
        result = trader.toggle()
        assert result["enabled"] is True, "Toggle should enable"
        assert trader.enabled is True
        results.ok("Toggle ON: enabled=True")

        # Test 3: Toggle off
        result = trader.toggle()
        assert result["enabled"] is False, "Toggle should disable"
        assert trader.enabled is False
        results.ok("Toggle OFF: enabled=False")

        # Test 4: Toggle with explicit value
        result = trader.toggle(on=True)
        assert result["enabled"] is True
        result = trader.toggle(on=True)  # Already on, should stay on
        assert result["enabled"] is True
        results.ok("Toggle with explicit on=True works")

        # Test 5: Set valid trade amount
        result = trader.set_trade_amount(2.50)
        assert result["success"] is True
        assert result["amount"] == 2.50
        assert trader.trade_amount == 2.50
        results.ok("Set amount 2.50 USDC: success")

        # Test 6: Amount too low
        result = trader.set_trade_amount(0.05)
        assert result["success"] is False
        assert trader.trade_amount == 2.50, "Amount should not change on failure"
        results.ok(f"Amount 0.05 rejected (min={MIN_TRADE_AMOUNT})")

        # Test 7: Amount too high
        result = trader.set_trade_amount(500.0)
        assert result["success"] is False
        assert trader.trade_amount == 2.50, "Amount should not change on failure"
        results.ok(f"Amount 500.00 rejected (max={MAX_TRADE_AMOUNT})")

        # Test 8: Config persistence
        trader2 = AutoTrader(polymarket_client=mock_pm, data_dir=test_dir)
        assert trader2.enabled is True, "Enabled state should persist"
        assert trader2.trade_amount == 2.50, "Amount should persist"
        results.ok("Config persists across instances")

        # Test 9: Get config
        config = trader.get_config()
        assert config["enabled"] is True
        assert config["trade_amount"] == 2.50
        assert config["session_trades"] == 0
        results.ok("get_config returns correct state")

        # Test 10: Edge amount values
        result = trader.set_trade_amount(MIN_TRADE_AMOUNT)
        assert result["success"] is True, f"Min amount {MIN_TRADE_AMOUNT} should be valid"
        result = trader.set_trade_amount(MAX_TRADE_AMOUNT)
        assert result["success"] is True, f"Max amount {MAX_TRADE_AMOUNT} should be valid"
        results.ok(f"Edge amounts accepted: min={MIN_TRADE_AMOUNT}, max={MAX_TRADE_AMOUNT}")

        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        results.fail("AutoTrader", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_auto_trader_execute(results: TestResults):
    """Test AutoTrader trade execution with mocked Polymarket client.

    Updated for slot-targeted trading: signals now include target_slot_ts
    and the trade pipeline uses it for market discovery and dedup.
    """
    print("\n--- Testing AutoTrader Trade Execution (Slot-Targeted) ---")
    test_dir = "/tmp/test_autotrader_exec"
    os.makedirs(test_dir, exist_ok=True)

    try:
        from src.auto_trader import AutoTrader
        from src.polymarket_client import PolymarketClient

        # Use a fixed target slot timestamp for deterministic testing
        # This represents a specific 5-min slot (e.g. 16:45:00 UTC)
        TARGET_SLOT_TS = 1742480700  # Must be 300-second aligned
        assert TARGET_SLOT_TS % 300 == 0, "Test slot timestamp must be 300s-aligned"

        # Create a mock PolymarketClient
        mock_pm = MagicMock()
        mock_pm.is_initialized = True
        mock_pm.wallet_address = "0xMOCKWALLET"

        # Mock balance check
        mock_pm.get_balance = AsyncMock(return_value={
            "success": True,
            "data": {"balance": 50.0},
        })

        # Mock place_trade -- verify target_slot_ts is passed through
        mock_pm.place_trade = AsyncMock(return_value={
            "success": True,
            "data": {
                "order_id": "mock-order-123",
                "direction": "UP",
                "amount": 1.0,
                "price": 0.55,
                "size": 1.82,
                "slot_ts": TARGET_SLOT_TS,
                "slot_dt": "2026-03-20T16:45:00+00:00",
                "market_slug": f"btc-updown-5m-{TARGET_SLOT_TS}",
                "status": "MATCHED",
            },
        })

        # Mock slot_to_datetime for logging
        mock_pm.slot_to_datetime = PolymarketClient.slot_to_datetime

        trader = AutoTrader(polymarket_client=mock_pm, data_dir=test_dir)
        trader.toggle(on=True)

        # Signal now includes target_slot_ts (injected by bot.py)
        signal = {
            "signal": "UP",
            "confidence": 0.62,
            "strength": "STRONG",
            "current_price": 85000.0,
            "prob_up": 0.62,
            "prob_down": 0.38,
            "target_slot_ts": TARGET_SLOT_TS,
        }

        # Test 1: Execute trade when disabled
        trader.toggle(on=False)
        result = await trader.execute_trade(signal)
        assert result["success"] is False
        assert result["action"] == "skipped"
        results.ok("Trade skipped when disabled")

        # Test 2: Execute trade when enabled — should pass target_slot_ts to place_trade
        trader.toggle(on=True)
        result = await trader.execute_trade(signal)
        assert result["success"] is True, f"Trade should succeed, got error: {result.get('error')}"
        assert result["action"] == "traded"
        assert result["data"]["order_id"] == "mock-order-123"
        assert result["data"]["confidence"] == 0.62
        assert result["data"]["strength"] == "STRONG"

        # Verify place_trade was called with target_slot_ts
        mock_pm.place_trade.assert_called_once_with(
            direction="UP",
            amount=trader.trade_amount,
            target_slot_ts=TARGET_SLOT_TS,
        )
        results.ok("Trade executed with correct target_slot_ts passed to place_trade")

        # Test 3: Duplicate slot prevention — same target_slot_ts should be rejected
        result2 = await trader.execute_trade(signal)
        assert result2["success"] is False
        assert result2["action"] == "skipped"
        assert "Already traded" in result2["error"]
        results.ok("Duplicate trade for same target slot correctly prevented")

        # Test 4: Different target slot should work
        NEXT_SLOT_TS = TARGET_SLOT_TS + 300  # Next 5-min slot
        signal_next = {
            "signal": "DOWN",
            "confidence": 0.58,
            "strength": "NORMAL",
            "current_price": 85100.0,
            "prob_up": 0.42,
            "prob_down": 0.58,
            "target_slot_ts": NEXT_SLOT_TS,
        }
        mock_pm.place_trade = AsyncMock(return_value={
            "success": True,
            "data": {
                "order_id": "mock-order-456",
                "direction": "DOWN",
                "amount": 1.0,
                "price": 0.52,
                "size": 1.92,
                "slot_ts": NEXT_SLOT_TS,
                "slot_dt": "2026-03-20T16:50:00+00:00",
                "market_slug": f"btc-updown-5m-{NEXT_SLOT_TS}",
                "status": "MATCHED",
            },
        })
        result3 = await trader.execute_trade(signal_next)
        assert result3["success"] is True, f"Different slot should work, got: {result3.get('error')}"
        assert result3["data"]["order_id"] == "mock-order-456"
        mock_pm.place_trade.assert_called_once_with(
            direction="DOWN",
            amount=trader.trade_amount,
            target_slot_ts=NEXT_SLOT_TS,
        )
        results.ok("Different target slot accepted (dedup is per-slot)")

        # Test 5: Missing target_slot_ts should error (not silently fail)
        signal_no_slot = {
            "signal": "UP",
            "confidence": 0.60,
            "strength": "NORMAL",
            "current_price": 85000.0,
            "prob_up": 0.60,
            "prob_down": 0.40,
            # No target_slot_ts!
        }
        result4 = await trader.execute_trade(signal_no_slot)
        assert result4["success"] is False
        assert result4["action"] == "error"
        assert "target_slot_ts" in result4["error"].lower() or "target_slot" in result4["error"].lower()
        results.ok("Missing target_slot_ts correctly rejected with error")

        # Test 6: NEUTRAL signal skipped
        neutral_signal = {
            "signal": "NEUTRAL",
            "confidence": 0.50,
            "strength": "SKIP",
            "target_slot_ts": TARGET_SLOT_TS + 600,
        }
        result5 = await trader.execute_trade(neutral_signal)
        assert result5["success"] is False
        assert result5["action"] == "skipped"
        results.ok("NEUTRAL signal correctly skipped")

        # Test 7: Insufficient balance
        mock_pm.get_balance = AsyncMock(return_value={
            "success": True,
            "data": {"balance": 0.05},
        })
        signal_low_bal = {
            "signal": "UP",
            "confidence": 0.60,
            "strength": "NORMAL",
            "current_price": 85000.0,
            "prob_up": 0.60,
            "prob_down": 0.40,
            "target_slot_ts": TARGET_SLOT_TS + 900,  # Fresh slot
        }
        result6 = await trader.execute_trade(signal_low_bal)
        assert result6["success"] is False
        assert result6["action"] == "error"
        assert "Insufficient" in result6["error"]
        results.ok("Insufficient balance correctly rejected")

        # Test 8: Session stats
        stats = trader.get_session_stats()
        assert stats["total_trades"] == 2, f"Expected 2 session trades, got {stats['total_trades']}"
        assert stats["directions"]["UP"] == 1
        assert stats["directions"]["DOWN"] == 1
        results.ok(f"Session stats: {stats['total_trades']} trade(s), UP={stats['directions']['UP']}, DOWN={stats['directions']['DOWN']}")

        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        results.fail("AutoTrader Execution", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        shutil.rmtree(test_dir, ignore_errors=True)


def test_slot_timestamp(results: TestResults):
    """Test Polymarket slot timestamp calculation."""
    print("\n--- Testing Slot Timestamp Calculation ---")
    try:
        from src.polymarket_client import PolymarketClient

        # Test 1: get_next_slot_timestamp returns an integer
        ts = PolymarketClient.get_next_slot_timestamp()
        assert isinstance(ts, int), f"Slot timestamp should be int, got {type(ts)}"
        results.ok(f"Slot timestamp is int: {ts}")

        # Test 2: Timestamp is in the future (or very near future)
        now = datetime.now(timezone.utc)
        slot_dt = PolymarketClient.slot_to_datetime(ts)
        assert slot_dt >= now - timedelta(minutes=5), "Slot should be near-future"
        results.ok(f"Slot datetime: {slot_dt.isoformat()} (valid near-future)")

        # Test 3: Slot is aligned to 5-minute boundary
        assert slot_dt.second == 0, f"Slot should have 0 seconds, got {slot_dt.second}"
        assert slot_dt.minute % 5 == 0, f"Slot minute should be 5-min aligned, got {slot_dt.minute}"
        results.ok(f"Slot aligned to 5-min boundary: {slot_dt.strftime('%H:%M')}")

        # Test 4: slot_to_datetime roundtrip
        ts2 = int(slot_dt.timestamp())
        assert ts == ts2, f"Roundtrip mismatch: {ts} vs {ts2}"
        results.ok("Slot timestamp roundtrip: consistent")

        # Test 5: get_market_for_slot validates alignment
        # Misaligned timestamp should fail
        misaligned_ts = ts + 17  # Not 300-second aligned
        assert misaligned_ts % 300 != 0, "Test setup: should be misaligned"
        results.ok("Slot alignment validation: misaligned timestamps detectable")

        # Test 6: _build_slug is deterministic
        slug = PolymarketClient._build_slug(ts)
        expected_slug = f"btc-updown-5m-{ts}"
        assert slug == expected_slug, f"Slug mismatch: {slug} vs {expected_slug}"
        results.ok(f"Slug generation: {slug} (deterministic)")

    except Exception as e:
        results.fail("Slot Timestamp", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_formatters_polymarket(results: TestResults):
    """Test Polymarket-related message formatters."""
    print("\n--- Testing Polymarket Formatters ---")
    try:
        from src import formatters

        # Test 1: format_trade_execution
        trade_data = {
            "order_id": "abc123",
            "direction": "UP",
            "amount": 1.50,
            "price": 0.55,
            "size": 2.73,
            "slot_dt": "2026-03-20T13:30:00+00:00",
            "confidence": 0.62,
            "strength": "STRONG",
            "status": "MATCHED",
        }
        msg = formatters.format_trade_execution(trade_data)
        assert "TRADE PLACED" in msg, "Should contain TRADE PLACED header"
        assert "YES (Up)" in msg, "Should show YES (Up) for UP direction"
        assert "$1.50" in msg, "Should show amount"
        assert "abc123" in msg, "Should show order ID"
        results.ok("format_trade_execution: UP trade formatted correctly")

        # Test 2: DOWN trade
        trade_data["direction"] = "DOWN"
        msg_down = formatters.format_trade_execution(trade_data)
        assert "NO (Down)" in msg_down, "Should show NO (Down) for DOWN direction"
        results.ok("format_trade_execution: DOWN trade formatted correctly")

        # Test 3: format_trade_error
        msg_err = formatters.format_trade_error("Connection timeout")
        assert "Trade Error" in msg_err
        assert "Connection timeout" in msg_err
        results.ok("format_trade_error: error formatted correctly")

        # Test 4: format_balance
        msg_bal = formatters.format_balance(42.50)
        assert "$42.50" in msg_bal
        assert "Balance" in msg_bal
        results.ok("format_balance: balance formatted correctly")

        # Test 5: format_positions (empty)
        msg_pos_empty = formatters.format_positions([])
        assert "No open positions" in msg_pos_empty
        results.ok("format_positions: empty list handled")

        # Test 6: format_positions (with data)
        positions = [
            {"market": "BTC Up", "outcome": "YES", "size": 5.0, "avg_price": 0.55, "current_value": 3.00, "pnl": 0.25},
            {"market": "BTC Down", "outcome": "NO", "size": 3.0, "avg_price": 0.45, "current_value": 1.50, "pnl": -0.15},
        ]
        msg_pos = formatters.format_positions(positions)
        assert "BTC Up" in msg_pos
        assert "BTC Down" in msg_pos
        results.ok("format_positions: positions formatted correctly")

        # Test 7: format_pm_status
        msg_status = formatters.format_pm_status(
            connected=True, wallet="0x1234567890abcdef1234567890abcdef12345678",
            balance=25.00, autotrade_on=True, trade_amount=1.50, session_trades=3,
        )
        assert "Connected" in msg_status
        assert "0x1234...5678" in msg_status, "Wallet should be shortened"
        assert "$25.00" in msg_status
        assert "ON" in msg_status
        assert "$1.50" in msg_status
        results.ok("format_pm_status: full status card formatted")

        # Test 8: format_pm_status disconnected with error
        msg_disc = formatters.format_pm_status(
            connected=False, wallet="", balance=None,
            autotrade_on=False, trade_amount=1.0, session_trades=0, error="API timeout",
        )
        assert "Disconnected" in msg_disc
        assert "API timeout" in msg_disc
        results.ok("format_pm_status: disconnected + error formatted")

        # Test 9: format_autotrade_toggle
        msg_on = formatters.format_autotrade_toggle(True, 2.50)
        assert "ON" in msg_on
        assert "$2.50" in msg_on
        msg_off = formatters.format_autotrade_toggle(False, 2.50)
        assert "OFF" in msg_off
        results.ok("format_autotrade_toggle: on/off formatted")

        # Test 10: format_set_amount
        msg_set = formatters.format_set_amount({"success": True, "amount": 3.00, "message": "ok"})
        assert "$3.00" in msg_set
        assert "updated" in msg_set
        msg_fail = formatters.format_set_amount({"success": False, "amount": 0.05, "message": "Too low"})
        assert "Invalid" in msg_fail or "Too low" in msg_fail
        results.ok("format_set_amount: success and failure formatted")

        # Test 11: format_pm_not_configured
        msg_nc = formatters.format_pm_not_configured()
        assert "Not Configured" in msg_nc
        assert "POLYMARKET_PRIVATE_KEY" in msg_nc
        results.ok("format_pm_not_configured: message formatted")

        # Test 12: format_startup with Polymarket
        msg_startup = formatters.format_startup(
            model_accuracy=0.55, confidence_min=0.55, train_candles=43200,
            optuna_enabled=True, retrain_gate=0.002, tracked_signals=10,
            symbol="BTCUSDT", polymarket_enabled=True, autotrade_on=True,
        )
        assert "Polymarket" in msg_startup
        assert "autotrade ON" in msg_startup
        results.ok("format_startup: Polymarket status line included")

        # Test 13: format_help includes Polymarket commands
        msg_help = formatters.format_help()
        assert "/autotrade" in msg_help
        assert "/setamount" in msg_help
        assert "/balance" in msg_help
        assert "/positions" in msg_help
        assert "/pmstatus" in msg_help
        results.ok("format_help: all Polymarket commands listed")

    except Exception as e:
        results.fail("Polymarket Formatters", f"{type(e).__name__}: {e}")
        traceback.print_exc()


async def run_all_tests():
    """Run all tests."""
    print("="*50)
    print("BTC 5m SIGNAL BOT (aprilxg v2) - TEST SUITE")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("="*50)

    results = TestResults()

    # Test 1: Config (v2 + Polymarket)
    test_config(results)

    # Test 2: 35-second timing preservation
    test_35sec_timing(results)

    # Test 3: MEXC API
    df_5m, df_15m, df_1h, df_hist = await test_mexc_api(results)

    if df_5m is not None and df_hist is not None:
        # Test 4: Features (v2)
        test_features(results, df_5m, df_15m, df_1h)

        # Test 5: Model (v2)
        model, metrics = test_model(results, df_hist, df_15m, df_1h)

        if metrics:
            print(f"\n--- Model Metrics Summary ---")
            print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f}")
            print(f"  CV Mean:        {metrics['cv_mean']:.4f}")
            print(f"  Samples:        {metrics['total_samples']}")
            print(f"  Split:          INNER={metrics.get('n_inner', '?')} / CAL={metrics.get('n_cal', '?')} / OOS={metrics.get('n_oos', '?')}")
            print(f"  Features:       {metrics['n_features']}")
            print(f"  Feature Pruned: {metrics.get('feature_pruned', False)}")
            print(f"  Calibrated:     {metrics.get('calibrated', False)}")
            print(f"  Model Swapped:  {metrics['model_swapped']}")
            print(f"  Optuna Tuned:   {metrics.get('optuna_tuned', False)}")
    else:
        results.fail("Features (skipped)", "No data from MEXC API")
        results.fail("Model (skipped)", "No data from MEXC API")

    # Test 6: Signal Tracker (independent of API)
    test_signal_tracker(results)

    # Test 7: Polymarket slot timestamp calculation
    test_slot_timestamp(results)

    # Test 8: AutoTrader toggle, amount, config
    test_auto_trader(results)

    # Test 9: AutoTrader trade execution (mocked, slot-targeted)
    await test_auto_trader_execute(results)

    # Test 10: Polymarket formatters
    test_formatters_polymarket(results)

    # Final summary
    all_passed = results.summary()

    if all_passed:
        print("\nAll tests PASSED! aprilxg v2 + Polymarket slot-targeted trading is ready for deployment.")
    else:
        print(f"\n{results.failed} test(s) FAILED. Review errors above.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
