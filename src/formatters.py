"""Telegram message formatters for AprilXG v2.

All message formatting is centralized here, separated from business logic.
Uses Telegram HTML parse mode for rich formatting.

Design System:
- One emoji per section header, not per line
- No pipe separators, no ASCII bars
- <code> blocks for monospace data alignment
- Hero info first (direction, result, P&L)
- Consistent emoji vocabulary across all messages
- Streak emojis only at 3+
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
    return ""


def _total_dollar_pnl(wins: int, losses: int) -> float:
    """Calculate total dollar PnL from win/loss counts."""
    return (wins * WIN_PAYOUT) - (losses * LOSS_PAYOUT)


def _streak_display(streak_count: int, streak_type: str) -> str:
    """Format streak display with emoji only at 3+."""
    if streak_count == 0 or not streak_type:
        return "--"
    label = f"{streak_count}{streak_type[0]}"
    if streak_count >= 3:
        emoji = "\U0001f525" if streak_type == "WIN" else "\u2744\ufe0f"
        return f"{label} {emoji}"
    return label


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
    confidence = signal.confidence  # Now calibrated probability
    raw_confidence = prediction.get("raw_confidence", confidence)
    ev = prediction.get("ev", 0.0)
    slot_str = _format_slot(signal.candle_slot_ts) if signal.candle_slot_ts else "N/A"
    price = f"${signal.entry_price:,.2f}"

    model_acc = prediction.get("model_accuracy", 0)
    strength = prediction.get("strength", "NORMAL")

    # Direction styling
    if direction == "UP":
        dir_emoji = "\U0001f7e2"  # green circle
    else:
        dir_emoji = "\U0001f534"  # red circle

    # Strength badge
    strength_badge = "            \u26a1 <b>STRONG</b>" if strength == "STRONG" else ""

    lines = [
        f"\U0001f4e1  <b>SIGNAL #{signal.signal_id}</b>{strength_badge}",
        "",
        f"{dir_emoji} <b>{direction}</b>     {slot_str}",
        "",
        "<code>"
        f"Confidence (cal)  {confidence:.1%}\n"
        f"Confidence (raw)  {raw_confidence:.1%}\n"
        f"Expected Value    {'+' if ev >= 0 else ''}{ev:.4f}\n"
        f"Price            {price}\n"
        f"Model            {model_acc:.1%}"
        "</code>",
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

    # Direction emoji
    if signal.direction == "UP":
        dir_emoji = "\U0001f7e2"
    else:
        dir_emoji = "\U0001f534"

    # Running totals
    total_dollar = _total_dollar_pnl(stats.wins, stats.losses)
    total_sign = "+" if total_dollar >= 0 else ""
    streak_str = _streak_display(stats.current_streak, stats.current_streak_type)

    lines = [
        f"{result_emoji}  <b>{result_label}</b>  <code>{dollar}</code>            Signal #{signal.signal_id}",
        "",
        f"{dir_emoji} <b>{signal.direction}</b>     {slot_str}",
        "",
        "<code>"
        f"Open         ${signal.candle_open_price:,.2f}\n"
        f"Close        ${signal.exit_price:,.2f}"
        "</code>",
        "",
        "<code>"
        f"Record       {stats.wins}W - {stats.losses}L  ({stats.win_rate:.1f}%)\n"
        f"P&amp;L          {total_sign}${abs(total_dollar):.2f}\n"
        f"Streak       {streak_str}"
        "</code>",
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

    # Streak display
    streak_str = _streak_display(stats.current_streak, stats.current_streak_type)

    # Session / timing
    session_str = _format_utc_short(stats.session_start) if stats.session_start else "--"
    last_sig_str = _format_utc_short(stats.last_signal_time) if stats.last_signal_time else "--"

    lines = [
        "\U0001f4ca  <b>PERFORMANCE</b>",
        "",
        "\U0001f3c6 <b>Win Rate</b>",
        f"<code>    {stats.wins}W  -  {stats.losses}L         {stats.win_rate:.1f}%</code>",
        "",
        "\U0001f4b0 <b>Profit &amp; Loss</b>",
        f"<code>    Total               {total_sign}${abs(total_dollar):.2f}\n"
        f"    Per Trade            {avg_sign}${abs(avg_dollar):.2f}</code>",
        "",
        "\U0001f525 <b>Streaks</b>",
        f"<code>    Current              {streak_str}\n"
        f"    Best Win             {stats.longest_win_streak}\n"
        f"    Worst Loss           {stats.longest_loss_streak}</code>",
        "",
        "\U0001f916 <b>Model</b>",
        f"<code>    Avg Confidence       {stats.avg_confidence:.1%}\n"
        f"    Session Start        {session_str}\n"
        f"    Last Signal          {last_sig_str}</code>",
        "",
        "\U0001f4cb <b>Signals</b>",
        f"<code>    Total  {stats.total_signals}     Resolved  {resolved}     Pending  {stats.pending}</code>",
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

    lines = ["\U0001f4cb  <b>RECENT SIGNALS</b>", ""]

    # Count wins/losses in this batch for summary
    batch_wins = 0
    batch_losses = 0

    for s in reversed(signals):  # Show newest first
        # Result styling
        if s.result == "WIN":
            r_emoji = "\u2705"
            dollar = f"+${WIN_PAYOUT:.2f}"
            batch_wins += 1
        elif s.result == "LOSS":
            r_emoji = "\u274c"
            dollar = f"-${LOSS_PAYOUT:.2f}"
            batch_losses += 1
        else:
            r_emoji = "\u23f3"  # hourglass
            dollar = ""

        # Direction emoji
        if s.direction == "UP":
            d_emoji = "\U0001f7e2"
        else:
            d_emoji = "\U0001f534"

        # Slot time
        if s.candle_slot_ts:
            try:
                dt = datetime.fromisoformat(s.candle_slot_ts)
                end_dt = dt + timedelta(minutes=5)
                slot = f"{dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')} UTC"
            except (ValueError, TypeError):
                slot = "--"
        else:
            slot = "--"

        # Build signal line - dollar only shown if resolved
        dollar_part = f"   {dollar}" if dollar else ""
        lines.append(
            f" {r_emoji}  <b>#{s.signal_id}</b>  {d_emoji} <b>{s.direction}</b>"
            f"     <code>{s.confidence:.1%}</code>{dollar_part}"
        )
        lines.append(
            f"          <code>{slot}</code>"
        )
        lines.append("")

    # Summary line
    batch_total = _total_dollar_pnl(batch_wins, batch_losses)
    batch_sign = "+" if batch_total >= 0 else ""
    batch_resolved = batch_wins + batch_losses
    if batch_resolved > 0:
        lines.append(
            f"<code>Summary   {batch_wins}W - {batch_losses}L"
            f"     {batch_sign}${abs(batch_total):.2f}</code>"
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
    calibration_on: bool = False,
    pruning_on: bool = False,
    n_features: int = 0,
    n_total_features: int = 0,
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
        "\u2699\ufe0f  <b>SYSTEM STATUS</b>",
        "",
        "<code>"
        f"Status           {status_emoji} {status_label}\n"
        f"Uptime           {uptime_str}"
        "</code>",
        "",
        "\U0001f916 <b>Model</b>",
        "<code>"
        f"Accuracy         {model_accuracy:.1%}\n"
        f"Samples          {train_samples:,}\n"
        f"Last Trained     {trained_str}\n"
        f"Next Retrain     {retrain_remaining}\n"
        f"Optuna           {optuna_str} {tuned_str}\n"
        f"Calibration      {'ON' if calibration_on else 'OFF'}\n"
        f"Features         {n_features}/{n_total_features} {'(pruned)' if pruning_on else ''}"
        "</code>",
        "",
        "\U0001f4d0 <b>Config</b>",
        "<code>"
        f"Confidence       &gt;= {confidence_min:.0%}\n"
        f"Retrain Gate     {retrain_gate:.3f}"
        "</code>",
        "",
        "\U0001f4cb <b>Signals</b>",
        f"<code>Total  {total_signals}     Pending  {pending}</code>",
    ]

    return "\n".join(lines)


# ============================================================
# START / WELCOME
# ============================================================

def format_start(chat_id: int) -> str:
    """Format the /start welcome message."""
    lines = [
        "\U0001f680  <b>AprilXG v2</b>",
        "",
        "BTC 5-min binary signals",
        "Win <code>+$0.96</code>  \u00b7  Loss <code>-$1.00</code>",
        "",
        "Signals fire automatically before",
        "each candle opens.",
        "",
        "\U0001f4cb <b>Commands</b>",
        "<code>/stats          Performance\n"
        "/recent         Last 10 signals\n"
        "/status         Bot &amp; model info</code>",
        "",
        "\U0001f4b0 <b>Trading</b>",
        "<code>/autotrade      Toggle trading\n"
        "/balance        Wallet balance\n"
        "/positions      Open positions</code>",
        "",
        "Type /help for all commands.",
    ]

    return "\n".join(lines)


# ============================================================
# HELP
# ============================================================

def format_help() -> str:
    """Format the /help message."""
    lines = [
        "\u2753  <b>HELP</b>",
        "",
        "\U0001f4cb <b>Signals</b>",
        "<code>/stats           Full performance dashboard\n"
        "/recent          Last 10 signals with results\n"
        "/status          Bot &amp; model info\n"
        "/retrain         Force model retraining\n"
        "/forcetune       Force Optuna tuning + retrain</code>",
        "",
        "\U0001f4b0 <b>Trading</b>",
        "<code>/autotrade       Toggle auto-trading ON/OFF\n"
        "/setamount       Set trade size (e.g. /setamount 1.50)\n"
        "/balance         Wallet USDC balance\n"
        "/positions       Open positions\n"
        "/pmstatus        Connection status\n"
        "/redeem          Redeem resolved positions</code>",
        "",
        "\u26a1 <b>Signal Strength</b>",
        "<code>  STRONG         EV &gt;= $0.05 per trade\n"
        "  NORMAL         EV &gt;= $0.00 (positive EV)\n"
        "  Negative EV signals are skipped.</code>",
        "",
        "\U0001f4b5 <b>Payouts</b>",
        "<code>  Win  +$0.96    Loss  -$1.00\n"
        "  Breakeven      51.04% win rate</code>",
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
    samples = metrics.get("total_samples", 0)
    optuna_tuned = metrics.get("optuna_tuned", False)

    # Delta from previous
    delta = new_acc - previous_accuracy
    delta_str = f"  ({delta:+.1%})" if previous_accuracy > 0 else ""

    params_str = "Optuna-tuned" if optuna_tuned else "Default params"

    if swapped:
        status_emoji = "\u2705"
        status_label = "Active"

        lines = [
            f"\U0001f504  <b>MODEL RETRAINED</b>        {status_emoji} {status_label}",
            "",
            "<code>"
            f"Accuracy         {new_acc:.1%}{delta_str}\n"
            f"Samples          {samples:,}\n"
            f"Params           {params_str}"
            "</code>",
        ]
    else:
        status_emoji = "\U0001f6e1\ufe0f"
        status_label = "Kept Previous"

        lines = [
            f"\U0001f504  <b>MODEL RETRAINED</b>        {status_emoji} {status_label}",
            "",
            "<code>"
            f"New Accuracy     {new_acc:.1%}\n"
            f"Active           {active_acc:.1%}  (better)\n"
            f"Samples          {samples:,}"
            "</code>",
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
    polymarket_enabled: bool = False,
    autotrade_on: bool = False,
    calibration_on: bool = False,
    pruning_on: bool = False,
    n_features: int = 0,
) -> str:
    """Format bot startup/online message."""
    days = train_candles * 5 // 1440

    pm_str = ""
    if polymarket_enabled:
        at_str = "ON" if autotrade_on else "OFF"
        pm_str = f"\nPolymarket       Connected (autotrade {at_str})"
    else:
        pm_str = "\nPolymarket       Disabled"

    lines = [
        "\U0001f7e2  <b>AprilXG v2 Online</b>",
        "",
        "<code>"
        f"Model            {model_accuracy:.1%} accuracy\n"
        f"Threshold        &gt;= {confidence_min:.0%} raw confidence\n"
        f"Calibration      {'ON' if calibration_on else 'OFF'}\n"
        f"Feature Pruning  {'ON (' + str(n_features) + ' features)' if pruning_on else 'OFF'}\n"
        f"Data             {train_candles:,} candles (~{days}d)\n"
        f"Signals          {tracked_signals} tracked"
        f"{pm_str}"
        "</code>",
        "",
        "Type /help for commands.",
    ]

    return "\n".join(lines)


# ============================================================
# SHUTDOWN MESSAGE
# ============================================================

def format_shutdown() -> str:
    """Format bot shutdown message."""
    return "\U0001f534  <b>AprilXG v2 Offline</b>"


# ============================================================
# RETRAIN STARTED / COMPLETE
# ============================================================

def format_retrain_started() -> str:
    """Format retrain-in-progress message."""
    return "\U0001f504  <b>Retraining model...</b>\nThis may take a few minutes."


def format_forcetune_started() -> str:
    """Format force-tune in-progress message."""
    return (
        "\U0001f9ec  <b>Force Optuna tuning + retrain...</b>\n"
        "Running 40 Optuna trials then retraining.\n"
        "This may take several minutes."
    )


def format_retrain_complete(accuracy: float) -> str:
    """Format retrain success message for /retrain command."""
    return (
        f"\u2705  <b>Retrain complete</b>\n"
        f"Accuracy  <code>{accuracy:.1%}</code>"
    )


def format_retrain_failed(error: str) -> str:
    """Format retrain failure message."""
    safe_error = _escape_html(error[:200])
    return f"\u274c  <b>Retrain failed</b>\n\n<code>{safe_error}</code>"


# ============================================================
# INTERACTIVE RETRAIN COMPARISON
# ============================================================

def format_retrain_comparison(comparison: dict) -> str:
    """Format old-vs-new model comparison for interactive retrain.

    Args:
        comparison: Dict from model.train_for_comparison() with
                    old_* and new_* metrics plus improvement.

    Returns:
        HTML-formatted comparison message shown with Keep/Swap buttons.
    """
    old_acc = comparison.get("old_val_accuracy", 0)
    new_acc = comparison.get("new_val_accuracy", 0)
    improvement = comparison.get("improvement", 0)
    new_cv = comparison.get("new_cv_accuracy", 0)
    old_logloss = comparison.get("old_val_logloss", 0)
    new_logloss = comparison.get("new_val_logloss", 0)
    samples = comparison.get("new_total_samples", 0)
    n_features = comparison.get("new_n_features", 0)
    optuna = comparison.get("optuna_tuned", False)
    has_existing = comparison.get("has_existing_model", False)
    old_recent = comparison.get("old_recent_accuracy", 0)
    new_recent = comparison.get("new_recent_accuracy", 0)

    if improvement > 0:
        delta_icon = "\U0001f7e2"  # green circle
    elif improvement < 0:
        delta_icon = "\U0001f534"  # red circle
    else:
        delta_icon = "\u26aa"     # white circle

    params_str = "Optuna" if optuna else "Default"

    lines = [
        "\U0001f504  <b>RETRAIN COMPARISON</b>",
        "",
        "<code>",
    ]

    if has_existing:
        lines.append(f"           Old       New")
        lines.append(f"Val Acc    {old_acc:.1%}     {new_acc:.1%}  {delta_icon} {improvement:+.1%}")
        lines.append(f"Recent288  {old_recent:.1%}     {new_recent:.1%}")
        lines.append(f"Log Loss   {old_logloss:.4f}    {new_logloss:.4f}")
    else:
        lines.append(f"Val Acc    {new_acc:.1%}  (first model)")
        lines.append(f"Recent288  {new_recent:.1%}")
        lines.append(f"Log Loss   {new_logloss:.4f}")

    lines.append(f"CV Acc     {new_cv:.1%}")
    lines.append(f"Samples    {samples:,}")
    lines.append(f"Features   {n_features}")
    lines.append(f"Params     {params_str}")
    lines.append("</code>")
    lines.append("")
    lines.append("Choose an action below:")

    return "\n".join(lines)


def format_retrain_decision(result: dict) -> str:
    """Format the outcome after user clicks Keep or Swap.

    Args:
        result: Dict from model.apply_pending_model() or
                model.reject_pending_model().

    Returns:
        HTML string appended below the comparison message.
    """
    action = result.get("action", "unknown")

    if action == "swap":
        acc = result.get("val_accuracy", 0)
        return (
            f"\u2705  <b>Model swapped</b>\n"
            f"Active accuracy  <code>{acc:.1%}</code>"
        )
    elif action == "keep":
        acc = result.get("val_accuracy", 0)
        rejected = result.get("rejected_val_accuracy", 0)
        return (
            f"\U0001f6e1\ufe0f  <b>Old model kept</b>\n"
            f"Active accuracy  <code>{acc:.1%}</code>\n"
            f"Rejected candidate  <code>{rejected:.1%}</code>"
        )
    else:
        return "\u26a0\ufe0f  Unknown retrain decision."


def format_retrain_result(result: dict) -> str:
    """Format retrain result when there is no existing model (auto-apply).

    Args:
        result: Dict from model.apply_pending_model().

    Returns:
        HTML-formatted message confirming the new model is active.
    """
    acc = result.get("val_accuracy", 0)
    return (
        f"\u2705  <b>Model trained and activated</b>\n"
        f"Accuracy  <code>{acc:.1%}</code>"
    )


# ============================================================
# TRAINING FAILED
# ============================================================

def format_training_failed(error: str) -> str:
    """Format training failure notification."""
    safe_error = _escape_html(error[:200])
    return f"\u274c  <b>Model training failed</b>\n\n<code>{safe_error}</code>"


# ============================================================
# POLYMARKET TRADE FORMATTERS
# ============================================================

def format_trade_execution(trade_data: dict) -> str:
    """Format a Polymarket trade execution confirmation.

    Args:
        trade_data: Dict with order_id, direction, amount, price, size,
                    slot_dt, question, confidence, strength, etc.

    Returns:
        HTML-formatted message string
    """
    direction = trade_data.get("direction", "?")
    amount = trade_data.get("amount", 0)
    price = trade_data.get("price", 0)
    slot_dt = trade_data.get("slot_dt", "")
    confidence = trade_data.get("confidence", 0)
    strength = trade_data.get("strength", "NORMAL")

    # Direction styling
    if direction == "UP":
        dir_emoji = "\U0001f7e2"  # green circle
    else:
        dir_emoji = "\U0001f534"  # red circle

    strength_badge = "     \u26a1" if strength == "STRONG" else ""

    # Slot time formatting
    slot_str = ""
    if slot_dt:
        try:
            dt = datetime.fromisoformat(slot_dt)
            end = dt + timedelta(minutes=5)
            slot_str = f"{dt.strftime('%H:%M')} - {end.strftime('%H:%M')} UTC"
        except Exception:
            slot_str = str(slot_dt)[:19]

    lines = [
        "\U0001f4b0  <b>TRADE PLACED</b>",
        "",
        f"{dir_emoji} <b>{direction}</b>     {slot_str}{strength_badge}",
        "",
        "<code>"
        f"Amount           ${amount:.2f}\n"
        f"Confidence       {confidence:.1%}\n"
        f"Fill Price       {price:.4f}"
        "</code>",
    ]

    return "\n".join(lines)


def format_trade_error(error: str) -> str:
    """Format a trade execution error."""
    safe_error = _escape_html(str(error)[:300])
    return (
        f"\u274c  <b>TRADE ERROR</b>\n\n"
        f"<code>{safe_error}</code>\n\n"
        f"Check /pmstatus for details."
    )


def format_balance(balance: float) -> str:
    """Format Polymarket wallet balance display."""
    return (
        f"\U0001f4b0  <b>BALANCE</b>\n\n"
        f"<code>${balance:.2f} USDC</code>"
    )


def format_positions(positions: list) -> str:
    """Format open Polymarket positions.

    Args:
        positions: List of position dicts with market, outcome, size,
                   avg_price, current_value, pnl fields.
    """
    if not positions:
        return "\U0001f4cb  <b>OPEN POSITIONS</b>\n\nNo open positions."

    lines = ["\U0001f4cb  <b>OPEN POSITIONS</b>", ""]

    for i, pos in enumerate(positions, 1):
        outcome = pos.get("outcome", "?")
        size = pos.get("size", 0)
        avg_price = pos.get("avg_price", 0)
        current_value = pos.get("current_value", 0)
        pnl = pos.get("pnl", 0)

        pnl_sign = "+" if pnl >= 0 else ""
        pnl_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"

        # Map outcome to trader-friendly language
        display_outcome = outcome
        outcome_lower = outcome.strip().lower() if outcome else ""
        if outcome_lower in ("yes", "up"):
            display_outcome = "UP"
            out_emoji = "\U0001f7e2"
        elif outcome_lower in ("no", "down"):
            display_outcome = "DOWN"
            out_emoji = "\U0001f534"
        else:
            out_emoji = pnl_emoji

        # Extract slot from market title if possible
        market = _escape_html(str(pos.get("market", "Unknown"))[:50])

        lines.append(f"<b>{i}.</b>  {market}")
        lines.append(
            f"<code>    {out_emoji} {display_outcome}   {size:.2f} shares @ {avg_price:.4f}\n"
            f"    Value ${current_value:.2f}     PnL {pnl_sign}${abs(pnl):.2f}</code>"
        )
        lines.append("")

    return "\n".join(lines)


def format_pm_status(
    connected: bool,
    wallet: str,
    balance: Optional[float],
    autotrade_on: bool,
    trade_amount: float,
    session_trades: int,
    error: Optional[str] = None,
) -> str:
    """Format full Polymarket connection status card."""
    conn_emoji = "\U0001f7e2" if connected else "\U0001f534"
    conn_label = "Connected" if connected else "Disconnected"
    at_emoji = "\U0001f7e2" if autotrade_on else "\u26ab"
    at_label = "ON" if autotrade_on else "OFF"

    balance_str = f"${balance:.2f}" if balance is not None else "N/A"
    wallet_short = f"{wallet[:6]}...{wallet[-4:]}" if wallet and len(wallet) > 10 else (wallet or "N/A")

    lines = [
        "\U0001f4b0  <b>POLYMARKET STATUS</b>",
        "",
        "<code>"
        f"Connection       {conn_emoji} {conn_label}\n"
        f"Wallet           {_escape_html(wallet_short)}\n"
        f"Balance          {balance_str}\n"
        f"\n"
        f"Auto-Trade       {at_emoji} {at_label}\n"
        f"Trade Amount     ${trade_amount:.2f} USDC\n"
        f"Session Trades   {session_trades}"
        "</code>",
    ]

    if error:
        safe_error = _escape_html(str(error)[:150])
        lines.extend([
            "",
            f"\u26a0\ufe0f Error: <code>{safe_error}</code>",
        ])

    return "\n".join(lines)


def format_autotrade_toggle(enabled: bool, amount: float) -> str:
    """Format autotrade toggle confirmation."""
    if enabled:
        return (
            f"\U0001f7e2  <b>AUTO-TRADE ON</b>\n\n"
            f"<code>${amount:.2f} USDC</code> per signal\n"
            f"Trades execute on every signal."
        )
    else:
        return (
            f"\u26ab  <b>AUTO-TRADE OFF</b>\n\n"
            f"Signals only, no trades."
        )


def format_set_amount(result: dict) -> str:
    """Format set-amount confirmation.

    Args:
        result: Dict with success (bool), amount (float), message (str)
    """
    if result.get("success"):
        amount = result.get("amount", 0)
        return (
            f"\u2705  <b>Trade amount updated</b>\n\n"
            f"<code>${amount:.2f} USDC</code> per trade"
        )
    else:
        msg = _escape_html(result.get("message", "Unknown error"))
        return f"\u274c  <b>Invalid amount</b>\n\n{msg}"


def format_pm_not_configured() -> str:
    """Format message when Polymarket is not configured."""
    return (
        "\u26a0\ufe0f  <b>Polymarket not configured</b>\n\n"
        "Set <code>POLYMARKET_PRIVATE_KEY</code> to enable trading."
    )


# ---------------------------------------------------------------------------
# Position Redemption Formatters
# ---------------------------------------------------------------------------

def format_redemption_result(result: dict) -> str:
    """Format the result of a batch redemption scan."""
    redeemed = result.get("redeemed", [])
    errors = result.get("errors", [])
    total_usdc = result.get("total_usdc", 0)

    if not redeemed and not errors:
        return (
            "\U0001f50d  <b>Redemption Scan</b>\n\n"
            "No redeemable positions found.\n"
            "Resolved positions are auto-scanned every 2 minutes."
        )

    lines = []

    if redeemed:
        lines.append(
            f"\U0001f4b0  <b>REDEMPTION COMPLETE</b>\n\n"
            f"Redeemed  <b>{len(redeemed)} positions</b>     <code>${total_usdc:.2f}</code>"
        )
        lines.append("")
        for r in redeemed:
            title = _escape_html(r.get("title", "Unknown")[:50])
            size = r.get("size", 0)
            # Try to extract just the slot time from the title for a cleaner display
            lines.append(f"  \u2705 {title}   <code>${size:.2f}</code>")
        lines.append("")

    if errors:
        lines.append(f"Errors  <b>{len(errors)}</b>")
        for e in errors:
            title = _escape_html(e.get("title", "Unknown")[:50])
            err_msg = _escape_html(str(e.get("error", "Unknown error"))[:100])
            lines.append(f"  \u274c {title}   {err_msg}")

    return "\n".join(lines)


def format_redeem_status(stats: dict, redeemer_initialized: bool) -> str:
    """Format redemption system status."""
    if not redeemer_initialized:
        return (
            "\U0001f4b0  <b>Redemption Status</b>\n\n"
            "Redeemer not initialized.\n"
            "Set <code>POLYGON_RPC_URL</code> to enable on-chain redemption."
        )

    total = stats.get("total_redeemed", 0)
    total_usdc = stats.get("total_usdc", 0)
    last_scan = stats.get("last_scan")

    import time
    if last_scan:
        ago = int(time.time() - last_scan)
        scan_str = f"{ago}s ago"
    else:
        scan_str = "Never"

    return (
        "\U0001f4b0  <b>Redemption Status</b>\n\n"
        "<code>"
        f"Status           \u2705 Active\n"
        f"Redeemed         {total} positions\n"
        f"Session USDC     ${total_usdc:.4f}\n"
        f"Last Scan        {scan_str}"
        "</code>"
    )


def format_redeem_error(error: str) -> str:
    """Format a redemption error message."""
    return (
        f"\u274c  <b>Redemption Error</b>\n\n"
        f"<code>{_escape_html(error)}</code>"
    )


def format_ensemble_signal_message(
    signal,
    tracker_stats,
    trade_decision: dict,
) -> str:
    """Format V5 ensemble signal for Telegram.

    Shows: direction, calibrated confidence, EV, regime, tier, model agreement,
    risk mode, rolling accuracy.

    Args:
        signal: Signal dataclass instance from tracker
        tracker_stats: TrackerStats dataclass instance
        trade_decision: Dict from TradeManager.should_trade()
    """
    direction = signal.direction
    confidence = signal.confidence

    # Direction emoji
    if direction == "UP":
        dir_emoji = "\U0001f7e2"  # Green circle
        dir_arrow = "\u2b06\ufe0f"   # Up arrow
    else:
        dir_emoji = "\U0001f534"  # Red circle
        dir_arrow = "\u2b07\ufe0f"   # Down arrow

    # Tier display
    tier = trade_decision.get("tier")
    if tier == 1:
        tier_label = "Tier 1 (High Conviction)"
        tier_emoji = "\U0001f525"  # Fire
    elif tier == 2:
        tier_label = "Tier 2 (Medium)"
        tier_emoji = "\u26a1"     # Lightning
    elif tier == 3:
        tier_label = "Tier 3 (Base)"
        tier_emoji = "\u2705"     # Check
    else:
        tier_label = "No Trade"
        tier_emoji = "\u26d4"     # No entry

    # Risk mode display
    risk_mode = trade_decision.get("risk_mode", "NORMAL")
    if risk_mode == "DEFENSIVE":
        risk_emoji = "\U0001f6e1\ufe0f"  # Shield
    elif risk_mode == "CAUTIOUS":
        risk_emoji = "\u26a0\ufe0f"       # Warning
    else:
        risk_emoji = "\u2705"              # Check

    # Rolling accuracy
    rolling_count = trade_decision.get("rolling_count", 0)
    rolling_acc = trade_decision.get("rolling_accuracy")
    if rolling_acc is not None:
        rolling_str = f"{rolling_acc:.1%}"
    else:
        rolling_str = "N/A (warming up)"

    # Model agreement
    model_agreement = getattr(signal, "model_agreement", None)
    if model_agreement is not None:
        agreement_str = f"{model_agreement}/3 agree"
    else:
        agreement_str = "N/A"

    # Regime
    regime_name = getattr(signal, "regime_name", None)
    if regime_name is None:
        regime_name = "UNKNOWN"

    # EV calculation
    ev = getattr(signal, "ev", None)
    if ev is not None:
        ev_str = f"{ev:+.4f}"
        ev_label = "\u2705 Positive" if ev > 0 else "\u274c Negative"
    else:
        ev_str = "N/A"
        ev_label = ""

    # Stats
    total = tracker_stats.total_signals if tracker_stats else 0
    wins = tracker_stats.wins if tracker_stats else 0
    losses = tracker_stats.losses if tracker_stats else 0
    accuracy = tracker_stats.accuracy if tracker_stats else 0.0

    lines = [
        f"{dir_emoji} <b>V5 ENSEMBLE SIGNAL</b> {dir_arrow}",
        "",
        f"<b>Direction:</b> {direction}",
        f"<b>Confidence:</b> {confidence:.1%}",
        f"<b>EV:</b> {ev_str} {ev_label}",
        "",
        f"\U0001f30d <b>Regime:</b> {regime_name}",
        f"\U0001f916 <b>Models:</b> {agreement_str}",
        f"{tier_emoji} <b>Tier:</b> {tier_label}",
        f"{risk_emoji} <b>Risk Mode:</b> {risk_mode}",
        f"\U0001f4ca <b>Rolling Accuracy:</b> {rolling_str} ({rolling_count} trades)",
        "",
        f"<b>Session:</b> {wins}W / {losses}L ({accuracy:.1%})",
    ]

    if not trade_decision.get("trade", True):
        lines.append("")
        reason = trade_decision.get("reason", "")
        lines.append(f"\u26d4 <b>Trade Skipped:</b> {_escape_html(reason)}")

    return "\n".join(lines)

