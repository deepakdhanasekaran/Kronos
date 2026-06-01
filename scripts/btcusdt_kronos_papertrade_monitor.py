#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crypto_predictor import (  # noqa: E402
    evaluate_trade_signal,
    fetch_binance_klines,
    load_pretrained_predictor,
    predict_binance_direction_with_predictor,
)


def log_line(message: str) -> None:
    print(message, flush=True)


def fetch_latest_closed_candle(symbol: str, interval: str) -> tuple[datetime, float]:
    frame = fetch_binance_klines(symbol, interval, 3)
    if len(frame) < 2:
        raise ValueError(f"Not enough Binance candles returned for {symbol} @ {interval}")
    closed_row = frame.iloc[-2]
    timestamp = closed_row["timestamps"].to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp, float(closed_row["close"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 10-second Kronos paper-trade monitor for BTCUSDT."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol to monitor.")
    parser.add_argument("--interval", default="5m", help="Binance candle interval.")
    parser.add_argument("--lookback", type=int, default=256, help="Number of candles to feed Kronos.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon in candles.")
    parser.add_argument("--sample-count", type=int, default=5, help="Stochastic sampling count.")
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.2, help="Neutral threshold percent.")
    parser.add_argument("--confidence-samples", type=int, default=5, help="Confidence sample count.")
    parser.add_argument("--poll-seconds", type=int, default=10, help="How often to poll for a new opportunity.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N loops. 0 means run until Ctrl-C.")
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=50.0,
        help="Starting paper balance in USD.",
    )
    parser.add_argument(
        "--stake-fraction",
        type=float,
        default=1.0,
        help="Fraction of current balance to use for each paper trade.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/btcusdt_kronos_papertrade_monitor.jsonl",
        help="Path to write JSONL audit records.",
    )
    parser.add_argument(
        "--no-beep",
        action="store_true",
        help="Disable the terminal bell when a trade closes profitably.",
    )
    return parser


def summarize_run(stats: dict[str, Any], balance: float) -> str:
    evaluated = stats["closed_trades"]
    profitable = stats["profitable_trades"]
    correct = stats["correct_predictions"]
    correct_rate = (correct / evaluated) if evaluated else 0.0
    win_rate = (profitable / evaluated) if evaluated else 0.0
    return "\n".join(
        [
            "",
            "Kronos paper-trade summary:",
            f"Cycles: {stats['cycles']}",
            f"Open trades: {stats['opened_trades']}",
            f"Closed trades: {stats['closed_trades']}",
            f"Correct predictions: {correct}",
            f"Correct rate: {correct_rate:.2%}",
            f"Profitable trades: {profitable}",
            f"Win rate: {win_rate:.2%}",
            f"Realized PnL: ${stats['realized_pnl']:.2f}",
            f"Starting balance: ${stats['starting_balance']:.2f}",
            f"Ending balance: ${balance:.2f}",
        ]
    )


def open_trade_from_summary(summary: dict[str, Any], current_closed_ts: datetime, balance: float, stake_fraction: float) -> dict[str, Any]:
    stake_usd = max(0.0, balance * max(0.0, stake_fraction))
    return {
        "entry_closed_timestamp": current_closed_ts.isoformat(),
        "entry_price": float(summary["live_price"]),
        "stake_usd": float(stake_usd),
        "summary": summary,
        "opened_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    args = build_parser().parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "cycles": 0,
        "opened_trades": 0,
        "closed_trades": 0,
        "correct_predictions": 0,
        "profitable_trades": 0,
        "realized_pnl": 0.0,
        "starting_balance": float(args.starting_balance),
    }
    balance = float(args.starting_balance)
    pending_trade: dict[str, Any] | None = None
    predictor = load_pretrained_predictor()

    log_line(f"Starting Kronos paper-trade monitor for {args.symbol} @ {args.interval}")
    log_line(f"Polling every {args.poll_seconds}s")
    log_line(f"Starting balance: ${balance:.2f}")
    log_line(f"Logging to: {log_path}")

    try:
        while True:
            if args.max_cycles and stats["cycles"] >= args.max_cycles:
                break

            current_closed_ts, current_closed_close = fetch_latest_closed_candle(args.symbol, args.interval)

            if pending_trade is not None:
                entry_closed_ts = datetime.fromisoformat(pending_trade["entry_closed_timestamp"])
                if current_closed_ts > entry_closed_ts:
                    summary = pending_trade["summary"]
                    stake_usd = float(pending_trade["stake_usd"])
                    evaluation = evaluate_trade_signal(
                        reference_close=float(pending_trade["entry_price"]),
                        predicted_signal=str(summary.get("signal", "NEUTRAL")),
                        actual_close=current_closed_close,
                        neutral_threshold_pct=args.neutral_threshold_pct,
                        stake_usd=stake_usd,
                    )
                    pnl = float(evaluation["trade_pnl_usd"] or 0.0)
                    balance += pnl
                    stats["closed_trades"] += 1
                    stats["realized_pnl"] += pnl
                    if evaluation["match_signal"]:
                        stats["correct_predictions"] += 1
                    if pnl > 0:
                        stats["profitable_trades"] += 1
                        if not args.no_beep:
                            print("\a", end="", flush=True)
                        log_line("ALARM: paper trade was profitable")

                    record = {
                        "event": "trade_close",
                        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                        "closed_candle_timestamp": current_closed_ts.isoformat(),
                        "entry_price": pending_trade["entry_price"],
                        "exit_price": current_closed_close,
                        "stake_usd": stake_usd,
                        "pnl_usd": pnl,
                        "balance_after": balance,
                        "summary": summary,
                        "evaluation": evaluation,
                        "stats": stats.copy(),
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

                    log_line(
                        f"Closed trade: {summary['action']} -> pnl=${pnl:.2f} "
                        f"(balance=${balance:.2f}, match={evaluation['match_signal']})"
                    )
                    pending_trade = None

            if pending_trade is None:
                result = predict_binance_direction_with_predictor(
                    predictor=predictor,
                    symbol=args.symbol,
                    interval=args.interval,
                    lookback=args.lookback,
                    pred_len=args.pred_len,
                    sample_count=args.sample_count,
                    neutral_threshold_pct=args.neutral_threshold_pct,
                    confidence_samples=args.confidence_samples,
                )
                summary = result["summary"]
                stats["cycles"] += 1

                log_line(
                    f"Snapshot {stats['cycles']}: {summary['verdict']} "
                    f"(signal={summary['signal']}, action={summary['action']}, "
                    f"trade_conf={summary['trade_confidence']:.2f}) "
                    f"pred_close={summary['predicted_close']:.2f} live={summary['live_price']:.2f}"
                )

                record = {
                    "event": "trade_open_candidate",
                    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "closed_candle_timestamp": current_closed_ts.isoformat(),
                    "symbol": args.symbol,
                    "interval": args.interval,
                    "summary": summary,
                    "stats": stats.copy(),
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

                if summary["action"] != "NO_TRADE":
                    pending_trade = open_trade_from_summary(summary, current_closed_ts, balance, args.stake_fraction)
                    stats["opened_trades"] += 1
                    log_line(
                        f"Opened paper trade: {summary['action']} at ${pending_trade['entry_price']:.2f} "
                        f"(stake=${pending_trade['stake_usd']:.2f})"
                    )
                else:
                    log_line("No trade opened.")

            time.sleep(max(1, int(args.poll_seconds)))

    except KeyboardInterrupt:
        log_line("\nStopped by user.")

    log_line(summarize_run(stats, balance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
