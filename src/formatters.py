"""Telegram message formatters for AprilXG v2.

All message formatting is centralized here, separated from business logic.
Uses Telegram HTML parse mode for rich formatting.

Conventions:
- All messages use HTML parse_mode
- Dollar PnL is shown for binary trading ($0.96 win / $1.00 loss)
- Emoji used purposefully for visual hierarchy, not decoration
- Modular design for future Polymarket integration
"""
from datetime import datetime, timezone, timedelta
from typing import Optional

# Binary market payout constants
WIN_PAYOUT = 0.96   # Profit on a win
LOSS_PAYOUT = 1.00  # Loss on a loss
TRADE_COST = 1.00   # Cost per trade


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_slot(iso_ts: str) -> str:
    """Format an ISO timestamp into a readable UTC slot string.

    E.g. '2026-03-19T09:00:00+00:00' -> '09:00 - 09:05 UTC'
    """
    try:
        dt = datetime.fromisoformat(iso_ts)
        end = dt + timedelta(minutes=5)
        return f"{dt.strftime('%H:%M')} - {end.strftime('%H:%M')} UTC"
    except Exception:
        return iso_ts[:19] + "Z"


def _format_utc(iso_ts: str) -> str:
    """Format an ISO timestamp to a short UTC string like '09:04:45 UTC'."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%H:%M:%S UTC")
    except Exception:
        return iso_ts[:19] + "Z"


def _format_utc_short(iso_ts: str) -> str:
    """Format an ISO timestamp to HH:MM UTC."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%H:%M UTC")
    except Exception:
        return iso_ts[:19]


def _dollar_pnl(result: Optional[str]) -> str:
    """Return dollar PnL string based on WIN/LOSS result."""
    if result == "WIN":
        return f"+${WIN_PAYOUT:.2f}"
    elif result == "LOSS":
        return f"-${LOSS_PAYOUT:.2f}"
    elif result == "NEUTRAL":
        return "$0.00"
    return "---"


def _total_dollar_pnl(wins: int, losses: int) -> float:
    """Calculate total dollar PnL from win/loss counts."""
    return (wins * WIN_PAYOUT) - (losses * LOSS_PAYOUT)


def _win_rate_bar(win_rate: float, width: int = 20) -> str:
    """Create a visual win rate bar.

    E.g. 60% -> '||||||||||||--------'
    """
    filled = round(win_rate / 100 * width)
    empty = width - filled
    return "|" * filled + "-" * empty


def _prob_bar(prob_up: float, total_width: int = 20) -> str:
    """Create a visual probability split bar.

    E.g. 57% up -> '|||||||||||---------'
    """
    up_blocks = round(prob_up * total_width)
    down_blocks = total_width - up_blocks
    return "|" * up_blocks + "-" * down_blocks


def _streak_display(streak_count: int, streak_type: str) -> str:
    """Format streak display with emoji."""
    if streak_count == 0 or not streak_type:
        return "--"
    emoji = "\U0001f525" if streak_type == "WIN" else "\u2744\ufe0f"  # fire / snowflake
    return f"{streak_count}{streak_type[0]} {emoji}" if streak_count >= 3 else f"{streak_count}{streak_type[0]}"


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def format_signal(signal, prediction: dict) -> str:
    """Format a new signal as an HTML Telegram message.

    Args:
        signal: Signal dataclass instance
        prediction: Dict with prob_up, prob_down, model_accuracy, strength, etc.

    Returns:
        HTML-formatted message string
    """
    direction = signal.direction
    confidence = signal.confidence
    slot_str = _format_slot(signal.candle_slot_ts) if signal.candle_slot_ts else "N/A"
    sent_at = _format_utc(signal.timestamp)
    price = f"${signal.entry_price:,.2f}"

    prob_up = prediction.get("prob_up", 0)
    prob_down = prediction.get("prob_down", 0)
    model_acc = prediction.get("model_accuracy", 0)
    strength = prediction.get("strength", "NORMAL")

    # Direction styling
    if direction == "UP":
        dir_emoji = "\U0001f7e2"  # green circle
        dir_arrow = "\u2b06\ufe0f"   # up arrow
    else:
        dir_emoji = "\U0001f534"  # red circle
        dir_arrow = "\u2b07\ufe0f"   # down arrow

    # Strength badge
    strength_badge = "  \u26a1 <b>STRONG</b>" if strength == "STRONG" else ""

    # Probability bar
    bar = _prob_bar(prob_up)

    lines = [
        f"{dir_emoji} <b>SIGNAL #{signal.signal_id}</b>{strength_badge}",
        "",
        f"{dir_arrow} <b>{direction}</b>  |  <code>{slot_str}</code>",
        "",
        f"\U0001f4ca <b>Confidence</b>   <code>{confidence:.1%}</code>",
        f"\U0001f4b0 <b>Price</b>            <code>{price}</code>",
        "",
        f"  P(Up) <code>{prob_up:.1%}</code>  <code>{bar}</code>  P(Down) <code>{prob_down:.1%}</code>",
        "",
        f"\U0001f916 Model Accuracy   <code>{model_acc:.1%}</code>",
        f"\U0001f552 Sent   <code>{sent_at}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# RESOLUTION MESSAGE
# ============================================================

def format_resolution(signal, stats) -> str:
    """Format a signal resolution as an HTML Telegram message.

    Args:
        signal: Resolved Signal dataclass instance
        stats: TrackerStats dataclass instance

    Returns:
        HTML-formatted message string
    """
    result = signal.result
    slot_str = _format_slot(signal.candle_slot_ts) if signal.candle_slot_ts else "N/A"
    resolved_at = _format_utc(signal.resolved_at) if signal.resolved_at else "N/A"

    # Result styling
    if result == "WIN":
        result_emoji = "\u2705"  # green check
        result_label = "WIN"
        dollar = f"+${WIN_PAYOUT:.2f}"
    elif result == "LOSS":
        result_emoji = "\u274c"  # red X
        result_label = "LOSS"
        dollar = f"-${LOSS_PAYOUT:.2f}"
    else:
        result_emoji = "\u2796"  # neutral
        result_label = "NEUTRAL"
        dollar = "$0.00"

    # Direction emoji for the prediction
    pred_emoji = "\u2b06\ufe0f" if signal.direction == "UP" else "\u2b07\ufe0f"

    # Calculate move percentage
    move_pct = signal.pnl_pct if signal.pnl_pct is not None else 0.0

    # Running totals
    total_dollar = _total_dollar_pnl(stats.wins, stats.losses)
    total_sign = "+" if total_dollar >= 0 else ""
    streak_str = _streak_display(stats.current_streak, stats.current_streak_type)

    lines = [
        f"{result_emoji} <b>RESULT</b>  |  Signal #{signal.signal_id}",
        "",
        f"    <b>{result_label}</b>  <code>{dollar}</code>",
        "",
        f"\U0001f552 <code>{slot_str}</code>",
        f"{pred_emoji} Predicted   <b>{signal.direction}</b>",
        f"\U0001f4c2 Open            <code>${signal.candle_open_price:,.2f}</code>",
        f"\U0001f4c3 Close           <code>${signal.exit_price:,.2f}</code>",
        f"\U0001f4c8 Move            <code>{move_pct:+.4f}%</code>",
        "",
        f"\U0001f3af Record     <code>{stats.wins}W - {stats.losses}L  ({stats.win_rate:.1f}%)</code>",
        f"\U0001f4b5 Total PnL  <code>{total_sign}${abs(total_dollar):.2f}</code>",
        f"\U0001f525 Streak     <code>{streak_str}</code>",
        "",
        f"\u23f1 Resolved   <code>{resolved_at}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# STATS / PERFORMANCE DASHBOARD
# ============================================================

def format_stats(stats) -> str:
    """Format full performance stats as an HTML Telegram message.

    Args:
        stats: TrackerStats dataclass instance

    Returns:
        HTML-formatted message string
    """
    if stats.total_signals == 0:
        return "\U0001f4ca No signals recorded yet."

    resolved = stats.wins + stats.losses + stats.neutral
    total_dollar = _total_dollar_pnl(stats.wins, stats.losses)
    total_sign = "+" if total_dollar >= 0 else ""

    # Average dollar PnL per trade
    if resolved > 0:
        avg_dollar = total_dollar / resolved
        avg_sign = "+" if avg_dollar >= 0 else ""
    else:
        avg_dollar = 0.0
        avg_sign = ""

    # Win rate bar
    bar = _win_rate_bar(stats.win_rate)

    # Streak display
    streak_str = _streak_display(stats.current_streak, stats.current_streak_type)

    # Session / timing
    session_str = _format_utc_short(stats.session_start) if stats.session_start else "--"
    last_sig_str = _format_utc_short(stats.last_signal_time) if stats.last_signal_time else "--"

    lines = [
        "\U0001f4ca <b>PERFORMANCE DASHBOARD</b>",
        "",
        "\U0001f4cb <b>Overview</b>",
        f"  Total  <code>{stats.total_signals}</code>  |  Resolved  <code>{resolved}</code>  |  Pending  <code>{stats.pending}</code>",
        "",
        "\U0001f3af <b>Win Rate</b>",
        f"  <code>{stats.wins}W - {stats.losses}L</code>  <b>{stats.win_rate:.1f}%</b>",
        f"  <code>{bar}</code>",
        "",
        "\U0001f4b0 <b>Profit &amp; Loss</b>",
        f"  Total         <code>{total_sign}${abs(total_dollar):.2f}</code>  <code>({stats.total_pnl_pct:+.4f}%)</code>",
        f"  Avg/Trade     <code>{avg_sign}${abs(avg_dollar):.2f}</code>",
        f"  Avg Win       <code>+${WIN_PAYOUT:.2f}</code>  |  Avg Loss  <code>-${LOSS_PAYOUT:.2f}</code>",
        f"  Best          <code>{stats.best_trade_pct:+.4f}%</code>  |  Worst  <code>{stats.worst_trade_pct:+.4f}%</code>",
        "",
        "\U0001f525 <b>Streaks</b>",
        f"  Current      <code>{streak_str}</code>",
        f"  Best Win     <code>{stats.longest_win_streak}</code>  |  Worst Loss  <code>{stats.longest_loss_streak}</code>",
        "",
        "\U0001f916 <b>Model</b>",
        f"  Confidence Avg  <code>{stats.avg_confidence:.1%}</code>",
        f"  Session Start   <code>{session_str}</code>",
        f"  Last Signal     <code>{last_sig_str}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# RECENT SIGNALS
# ============================================================

def format_recent(signals: list, stats=None) -> str:
    """Format recent signals list as an HTML Telegram message.

    Args:
        signals: List of Signal dataclass instances (most recent last)
        stats: Optional TrackerStats for summary line

    Returns:
        HTML-formatted message string
    """
    if not signals:
        return "\U0001f4cb No signals recorded yet."

    lines = ["\U0001f4cb <b>RECENT SIGNALS</b>", ""]

    # Count wins/losses in this batch for summary
    batch_wins = 0
    batch_losses = 0

    for s in reversed(signals):  # Show newest first
        # Result styling
        if s.result == "WIN":
            r_emoji = "\u2705"
            r_label = "WIN"
            dollar = f"+${WIN_PAYOUT:.2f}"
            batch_wins += 1
        elif s.result == "LOSS":
            r_emoji = "\u274c"
            r_label = "LOSS"
            dollar = f"-${LOSS_PAYOUT:.2f}"
            batch_losses += 1
        else:
            r_emoji = "\u23f3"  # hourglass
            r_label = "PENDING"
            dollar = "   ---"

        # Direction emoji
        d_emoji = "\u2b06\ufe0f" if s.direction == "UP" else "\u2b07\ufe0f"

        # Slot time
        if s.candle_slot_ts:
            try:
                dt = datetime.fromisoformat(s.candle_slot_ts)
                end_dt = dt + timedelta(minutes=5)
                slot = f"{dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} UTC"
            except (ValueError, TypeError):
                slot = "--"
        else:
            slot = "--"

        lines.append(
            f"  {r_emoji} <b>#{s.signal_id}</b>  {d_emoji} {s.direction}  "
            f"<b>{r_label}</b>  <code>{dollar}</code>"
        )
        lines.append(
            f"       <code>{slot}</code>  |  <code>{s.confidence:.1%}</code>"
        )
        lines.append("")

    # Summary line
    batch_total = _total_dollar_pnl(batch_wins, batch_losses)
    batch_sign = "+" if batch_total >= 0 else ""
    batch_resolved = batch_wins + batch_losses
    if batch_resolved > 0:
        lines.append(
            f"\U0001f4ca Summary: <code>{batch_wins}W - {batch_losses}L</code> last {len(signals)}"
            f"  |  <code>{batch_sign}${abs(batch_total):.2f}</code>"
        )

    return "\n".join(lines)


# ============================================================
# STATUS
# ============================================================

def format_status(
    running: bool,
    session_start: str,
    symbol: str,
    model_accuracy: float,
    train_samples: int,
    last_train_time: Optional[datetime],
    retrain_remaining: str,
    confidence_min: float,
    retrain_gate: float,
    optuna_enabled: bool,
    optuna_tuned: bool,
    total_signals: int,
    pending: int,
) -> str:
    """Format bot status as an HTML Telegram message."""
    status_emoji = "\U0001f7e2" if running else "\U0001f534"  # green/red circle
    status_label = "Online" if running else "Offline"

    # Uptime calculation
    uptime_str = "--"
    if session_start:
        try:
            start_dt = datetime.fromisoformat(session_start)
            elapsed = datetime.now(timezone.utc) - start_dt
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            uptime_str = f"{hours}h {minutes}m"
        except Exception:
            uptime_str = "--"

    trained_str = last_train_time.strftime("%H:%M UTC") if last_train_time else "Never"
    optuna_str = "ON" if optuna_enabled else "OFF"
    tuned_str = "(tuned)" if optuna_tuned else "(defaults)"

    lines = [
        "\u2699\ufe0f <b>SYSTEM STATUS</b>",
        "",
        f"  {status_emoji} Bot          <code>{status_label}</code>",
        f"  \u23f1 Uptime       <code>{uptime_str}</code>",
        f"  \U0001f4b9 Symbol       <code>{symbol}</code>",
        "",
        "\U0001f916 <b>Model</b>",
        f"  Accuracy       <code>{model_accuracy:.1%}</code>",
        f"  Samples        <code>{train_samples:,}</code>",
        f"  Last Trained   <code>{trained_str}</code>",
        f"  Next Retrain   <code>{retrain_remaining}</code>",
        f"  Optuna         <code>{optuna_str} {tuned_str}</code>",
        "",
        "\u2699\ufe0f <b>Config</b>",
        f"  Confidence     <code>&gt;= {confidence_min:.0%}</code>",
        f"  Retrain Gate   <code>{retrain_gate:.3f}</code>",
        "",
        "\U0001f4e1 <b>Signals</b>",
        f"  Total  <code>{total_signals}</code>  |  Pending  <code>{pending}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# START / WELCOME
# ============================================================

def format_start(chat_id: int) -> str:
    """Format the /start welcome message."""
    lines = [
        "\U0001f680 <b>Welcome to AprilXG v2</b>",
        "",
        "\U0001f4b9 <b>BTC 5-Min Binary Signal Bot</b>",
        "Powered by XGBoost ML",
        "",
        "Signals are posted automatically",
        "before each 5-min candle opens.",
        "",
        f"\u2705 Win: <code>+$0.96</code>  |  \u274c Loss: <code>-$1.00</code>",
        f"\u2696\ufe0f Breakeven: <code>51.04%</code> win rate",
        "",
        "\U0001f4dd <b>Commands</b>",
        "  /stats     \U0001f4ca  Performance dashboard",
        "  /recent    \U0001f4cb  Last 10 signals",
        "  /status    \u2699\ufe0f   Bot &amp; model info",
        "  /retrain   \U0001f504  Force model retrain",
        "  /help      \u2753  Command reference",
        "",
        f"\U0001f194 Chat ID: <code>{chat_id}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# HELP
# ============================================================

def format_help() -> str:
    """Format the /help message."""
    lines = [
        "\u2753 <b>AprilXG v2 — Help</b>",
        "",
        "This bot predicts the direction of the next",
        "BTC 5-minute candle using an XGBoost ML model",
        "with multi-timeframe feature engineering.",
        "",
        "\U0001f4dd <b>Commands</b>",
        "  /stats     \U0001f4ca  Full performance stats (W/L, PnL, streaks)",
        "  /recent    \U0001f4cb  Last 10 signals with results",
        "  /status    \u2699\ufe0f   Model &amp; bot status",
        "  /retrain   \U0001f504  Force model retraining",
        "  /start     \U0001f680  Show welcome &amp; chat ID",
        "  /help      \u2753  This help message",
        "",
        "\U0001f4a1 <b>Signal Strength</b>",
        f"  \u26a1 <b>STRONG</b> — Confidence \u2265 60%",
        f"  \U0001f7e2 <b>NORMAL</b> — Confidence 55-60%",
        "  Signals below 55% are skipped.",
        "",
        "\U0001f4b0 <b>Payouts</b>",
        f"  Win: <code>+$0.96</code>  |  Loss: <code>-$1.00</code>",
        f"  Breakeven win rate: <code>51.04%</code>",
    ]

    return "\n".join(lines)


# ============================================================
# TRAINING COMPLETE
# ============================================================

def format_training_complete(metrics: dict, previous_accuracy: float) -> str:
    """Format model training completion message.

    Args:
        metrics: Training metrics dict from model.train()
        previous_accuracy: The accuracy before this training run

    Returns:
        HTML-formatted message string
    """
    swapped = metrics.get("model_swapped", False)
    new_acc = metrics.get("val_accuracy", 0)
    active_acc = metrics.get("active_val_accuracy", 0)
    cv_acc = metrics.get("cv_accuracy", 0)
    logloss = metrics.get("val_logloss", 0)
    samples = metrics.get("total_samples", 0)
    features = metrics.get("n_features", 0)
    optuna_tuned = metrics.get("optuna_tuned", False)

    # Delta from previous
    delta = new_acc - previous_accuracy
    delta_str = f"({delta:+.1%})" if previous_accuracy > 0 else ""

    if swapped:
        status_emoji = "\u2705"
        status_label = "New model active"
    else:
        status_emoji = "\U0001f6e1\ufe0f"
        status_label = "Kept previous model"

    params_str = "Optuna-tuned" if optuna_tuned else "Default params"

    lines = [
        "\U0001f504 <b>MODEL RETRAINED</b>",
        "",
        f"  {status_emoji} Status   <b>{status_label}</b>",
        "",
        "\U0001f4ca <b>Metrics</b>",
        f"  Val Accuracy    <code>{new_acc:.1%}</code>  <code>{delta_str}</code>",
        f"  Active Accuracy <code>{active_acc:.1%}</code>",
        f"  CV Accuracy     <code>{cv_acc:.1%}</code>",
        f"  Log Loss        <code>{logloss:.4f}</code>",
        f"  Samples         <code>{samples:,}</code>",
        f"  Features        <code>{features}</code>",
        f"  Params          <code>{params_str}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# STARTUP MESSAGE
# ============================================================

def format_startup(
    model_accuracy: float,
    confidence_min: float,
    train_candles: int,
    optuna_enabled: bool,
    retrain_gate: float,
    tracked_signals: int,
    symbol: str,
) -> str:
    """Format bot startup/online message."""
    days = train_candles * 5 // 1440
    optuna_str = "ON" if optuna_enabled else "OFF"

    lines = [
        "\U0001f680 <b>AprilXG v2 Online</b>",
        "",
        f"  \U0001f916 Model        <code>{model_accuracy:.1%} accuracy</code>",
        f"  \U0001f3af Threshold    <code>&gt;= {confidence_min:.0%} confidence</code>",
        f"  \U0001f4ca Data         <code>{train_candles:,} candles (~{days}d)</code>",
        f"  \u2699\ufe0f  Optuna       <code>{optuna_str}</code>",
        f"  \U0001f6e1\ufe0f Gate         <code>{retrain_gate:.3f} min improvement</code>",
        f"  \U0001f4e1 Signals      <code>{tracked_signals} tracked</code>",
        f"  \U0001f4b9 Symbol       <code>{symbol}</code>",
        "",
        "Signals posted automatically.",
        "Type /help for commands.",
    ]

    return "\n".join(lines)


# ============================================================
# SHUTDOWN MESSAGE
# ============================================================

def format_shutdown() -> str:
    """Format bot shutdown message."""
    return "\U0001f534 <b>AprilXG v2 Offline</b>\n\nBot is shutting down..."


# ============================================================
# RETRAIN STARTED / COMPLETE
# ============================================================

def format_retrain_started() -> str:
    """Format retrain-in-progress message."""
    return "\U0001f504 <b>Retraining model...</b>\n\nThis may take a few minutes."


def format_retrain_complete(accuracy: float) -> str:
    """Format retrain success message for /retrain command."""
    return (
        f"\u2705 <b>Retrain complete!</b>\n\n"
        f"Active model accuracy: <code>{accuracy:.1%}</code>"
    )


def format_retrain_failed(error: str) -> str:
    """Format retrain failure message."""
    safe_error = _escape_html(error[:200])
    return f"\u274c <b>Retrain failed</b>\n\n<code>{safe_error}</code>"


# ============================================================
# TRAINING FAILED
# ============================================================

def format_training_failed(error: str) -> str:
    """Format training failure notification."""
    safe_error = _escape_html(error[:200])
    return f"\u274c <b>Model training failed</b>\n\n<code>{safe_error}</code>"
