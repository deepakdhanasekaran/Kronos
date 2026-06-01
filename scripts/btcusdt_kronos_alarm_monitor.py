#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crypto_predictor import (
    evaluate_binance_prediction,
    fetch_binance_klines,
    interval_to_timedelta,
    predict_binance_direction,
)


def log_line(message: str) -> None:
    print(message, flush=True)


def seconds_until_next_close(interval: str, buffer_seconds: int) -> int:
    interval_seconds = int(interval_to_timedelta(interval).total_seconds())
    now = time.time()
    next_boundary = math.floor(now / interval_seconds + 1.0) * interval_seconds
    return max(1, int(math.ceil(next_boundary + buffer_seconds - now)))


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
        description="Poll BTCUSDT every few seconds, ring an alarm on correct Kronos predictions."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol to monitor.")
    parser.add_argument("--interval", default="5m", help="Binance candle interval.")
    parser.add_argument("--lookback", type=int, default=256, help="Number of candles to feed Kronos.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon in candles.")
    parser.add_argument("--sample-count", type=int, default=5, help="Stochastic sampling count.")
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.2, help="Neutral threshold percent.")
    parser.add_argument("--confidence-samples", type=int, default=5, help="Confidence sample count.")
    parser.add_argument("--poll-seconds", type=int, default=5, help="How often to poll and print a snapshot.")
    parser.add_argument("--buffer-seconds", type=int, default=8, help="Extra delay after candle close.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N evaluations. 0 means run until Ctrl-C.")
    parser.add_argument(
        "--log-file",
        default="logs/btcusdt_kronos_alarm_monitor.jsonl",
        help="Path to write JSONL audit records.",
    )
    parser.add_argument(
        "--no-beep",
        action="store_true",
        help="Disable the terminal bell when a prediction is correct.",
    )
    return parser


def summarize_run(stats: dict[str, Any]) -> str:
    cycles = stats["cycles"]
    correct = stats["correct"]
    evaluated = stats["evaluated"]
    tradeable = stats["tradeable"]
    tradeable_correct = stats["tradeable_correct"]
    win_rate = (correct / evaluated) if evaluated else 0.0
    tradeable_rate = (tradeable_correct / tradeable) if tradeable else 0.0
    return "\n".join(
        [
            "",
            "Kronos alarm summary:",
            f"Cycles: {cycles}",
            f"Evaluated: {evaluated}",
            f"Correct: {correct}",
            f"Correct rate: {win_rate:.2%}",
            f"Tradeable predictions: {tradeable}",
            f"Tradeable correct: {tradeable_correct}",
            f"Tradeable correct rate: {tradeable_rate:.2%}",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "cycles": 0,
        "evaluated": 0,
        "correct": 0,
        "tradeable": 0,
        "tradeable_correct": 0,
    }

    log_line(f"Starting Kronos alarm monitor for {args.symbol} @ {args.interval}")
    log_line(f"Polling every {args.poll_seconds}s")
    log_line(f"Logging to: {log_path}")

    last_closed_timestamp: datetime | None = None
    pending_summary: dict[str, Any] | None = None
    pending_reference_close: float | None = None
    pending_observed_at: datetime | None = None

    try:
        while True:
            if args.max_cycles and stats["cycles"] >= args.max_cycles:
                break

            current_closed_ts, current_closed_close = fetch_latest_closed_candle(args.symbol, args.interval)

            if pending_summary is not None and last_closed_timestamp is not None and current_closed_ts > last_closed_timestamp:
                evaluation = evaluate_binance_prediction(
                    pending_summary,
                    actual_close=current_closed_close,
                    neutral_threshold_pct=args.neutral_threshold_pct,
                )
                stats["evaluated"] += 1
                if evaluation["match_signal"]:
                    stats["correct"] += 1
                    if not args.no_beep:
                        print("\a", end="", flush=True)
                    log_line("ALARM: prediction was correct")

                if evaluation["predicted_action"] != "NO_TRADE":
                    stats["tradeable"] += 1
                    if evaluation["match_action"]:
                        stats["tradeable_correct"] += 1

                record = {
                    "event": "evaluation",
                    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "closed_candle_timestamp": current_closed_ts.isoformat(),
                    "reference_close": pending_reference_close,
                    "actual_close": current_closed_close,
                    "prediction": pending_summary,
                    "evaluation": evaluation,
                    "stats": stats.copy(),
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

                log_line(
                    f"Evaluated previous prediction: actual={evaluation['actual_signal']} "
                    f"(change={evaluation['actual_change_pct']:.2f}%, match={evaluation['match_signal']})"
                )

            result = predict_binance_direction(
                symbol=args.symbol,
                interval=args.interval,
                lookback=args.lookback,
                pred_len=args.pred_len,
                sample_count=args.sample_count,
                neutral_threshold_pct=args.neutral_threshold_pct,
                confidence_samples=args.confidence_samples,
            )
            summary = result["summary"]
            pending_summary = summary
            pending_reference_close = float(summary["last_close"])
            pending_observed_at = datetime.now(timezone.utc)
            last_closed_timestamp = current_closed_ts

            stats["cycles"] += 1
            log_line(
                f"Snapshot {stats['cycles']}: {summary['verdict']} "
                f"(signal={summary['signal']}, action={summary['action']}, trade_conf={summary['trade_confidence']:.2f}) "
                f"last_close={summary['last_close']:.2f} closed={current_closed_ts.isoformat()}"
            )

            record = {
                "event": "snapshot",
                "timestamp_utc": pending_observed_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "closed_candle_timestamp": current_closed_ts.isoformat(),
                "symbol": args.symbol,
                "interval": args.interval,
                "prediction": summary,
                "stats": stats.copy(),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            sleep_seconds = max(1, int(args.poll_seconds))
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        log_line("\nStopped by user.")

    log_line(summarize_run(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
