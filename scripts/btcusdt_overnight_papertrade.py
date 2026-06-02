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
    DEFAULT_MODEL_NAME,
    DEFAULT_TOKENIZER_NAME,
    evaluate_binance_prediction,
    fetch_binance_klines,
    interval_to_timedelta,
    predict_binance_direction,
    MODEL_NAME_BY_SIZE,
)


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
        description="Run an overnight BTCUSDT paper-trade loop using Kronos forecasts."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol to monitor.")
    parser.add_argument("--interval", default="5m", help="Binance candle interval.")
    parser.add_argument("--lookback", type=int, default=256, help="Number of candles to feed Kronos.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon in candles.")
    parser.add_argument("--sample-count", type=int, default=5, help="Stochastic sampling count.")
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.2, help="Neutral threshold percent.")
    parser.add_argument("--confidence-samples", type=int, default=5, help="Confidence sample count.")
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_TOKENIZER_NAME,
        help="Hugging Face tokenizer id or local path.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(MODEL_NAME_BY_SIZE.keys()),
        default="small",
        help="Shortcut for Kronos model size; overrides --model-name when set.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=512,
        help="Maximum context length for the model.",
    )
    parser.add_argument("--buffer-seconds", type=int, default=8, help="Extra delay after candle close.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles. 0 means run until Ctrl-C.")
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=50.0,
        help="Starting paper balance in USD used as trade stake.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/btcusdt_overnight_papertrade.jsonl",
        help="Path to write JSONL audit records.",
    )
    return parser


def summarize_run(stats: dict[str, Any]) -> str:
    cycles = stats["cycles"]
    trade_cycles = stats["trade_cycles"]
    signal_hits = stats["signal_hits"]
    action_hits = stats["action_hits"]
    trade_return_sum = stats["trade_return_sum_pct"]
    trade_wins = stats["trade_wins"]
    starting_balance = stats["starting_balance"]
    ending_balance = stats["balance_usd"]
    realized_pnl_usd = stats["realized_pnl_usd"]

    signal_hit_rate = (signal_hits / cycles) if cycles else 0.0
    action_hit_rate = (action_hits / trade_cycles) if trade_cycles else 0.0
    avg_trade_return = (trade_return_sum / trade_cycles) if trade_cycles else 0.0
    trade_win_rate = (trade_wins / trade_cycles) if trade_cycles else 0.0

    return "\n".join(
        [
            "",
            "Overnight summary:",
            f"Cycles: {cycles}",
            f"Trade cycles: {trade_cycles}",
            f"Signal hit rate: {signal_hit_rate:.2%}",
            f"Action hit rate: {action_hit_rate:.2%}",
            f"Trade win rate: {trade_win_rate:.2%}",
            f"Average trade return: {avg_trade_return:.2f}%",
            f"Total trade return: {trade_return_sum:.2f}%",
            f"Starting balance: ${starting_balance:.2f}",
            f"Ending balance: ${ending_balance:.2f}",
            f"Realized PnL: ${realized_pnl_usd:.2f}",
        ]
    )


def log_line(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = build_parser().parse_args()
    model_name = MODEL_NAME_BY_SIZE.get(args.model_size, args.model_name) if args.model_size else args.model_name
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "cycles": 0,
        "trade_cycles": 0,
        "signal_hits": 0,
        "action_hits": 0,
        "trade_wins": 0,
        "trade_return_sum_pct": 0.0,
        "starting_balance": float(args.starting_balance),
        "balance_usd": float(args.starting_balance),
        "realized_pnl_usd": 0.0,
    }

    log_line(f"Starting overnight paper-trade loop for {args.symbol} @ {args.interval}")
    log_line(f"Logging to: {log_path}")

    try:
        while True:
            if args.max_cycles and stats["cycles"] >= args.max_cycles:
                break

            cycle_num = stats["cycles"] + 1
            started_at = datetime.now(timezone.utc)
            result = predict_binance_direction(
                symbol=args.symbol,
                interval=args.interval,
                lookback=args.lookback,
                pred_len=args.pred_len,
                sample_count=args.sample_count,
                neutral_threshold_pct=args.neutral_threshold_pct,
                confidence_samples=args.confidence_samples,
                tokenizer_name=args.tokenizer_name,
                model_name=model_name,
                max_context=args.max_context,
            )
            summary = result["summary"]

            wait_seconds = seconds_until_next_close(args.interval, args.buffer_seconds)
            log_line(
                f"Cycle {cycle_num}: {summary['verdict']} "
                f"(signal={summary['signal']}, action={summary['action']}, trade_conf={summary['trade_confidence']:.2f})"
            )
            log_line(f"Waiting {wait_seconds}s for the next candle close...")
            time.sleep(wait_seconds)

            actual_ts, actual_close = fetch_latest_closed_candle(args.symbol, args.interval)
            evaluation = evaluate_binance_prediction(
                summary,
                actual_close=actual_close,
                neutral_threshold_pct=args.neutral_threshold_pct,
                stake_usd=stats["balance_usd"],
            )

            stats["cycles"] += 1
            if evaluation["match_signal"]:
                stats["signal_hits"] += 1
            if evaluation["match_action"]:
                stats["action_hits"] += 1
            if evaluation["predicted_action"] != "NO_TRADE":
                stats["trade_cycles"] += 1
                stats["trade_return_sum_pct"] += evaluation["strategy_return_pct"]
                pnl_usd = float(evaluation["trade_pnl_usd"] or 0.0)
                stats["realized_pnl_usd"] += pnl_usd
                stats["balance_usd"] += pnl_usd
                if evaluation["strategy_return_pct"] > 0:
                    stats["trade_wins"] += 1
            else:
                pnl_usd = 0.0

            record = {
                "cycle": cycle_num,
                "timestamp_utc": started_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "evaluation_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "symbol": args.symbol,
                "interval": args.interval,
                "prediction": {
                    "signal": summary["signal"],
                    "action": summary["action"],
                    "verdict": summary["verdict"],
                    "predicted_close": summary["predicted_close"],
                    "last_close": summary["last_close"],
                    "close_change_pct": summary["close_change_pct"],
                    "trade_confidence": summary["trade_confidence"],
                    "confidence": summary["confidence"],
                    "signal_counts": summary.get("signal_counts", {}),
                },
                "actual": {
                    "timestamp": actual_ts.isoformat().replace("+00:00", "Z"),
                    "close": actual_close,
                    "signal": evaluation["actual_signal"],
                    "action": evaluation["actual_action"],
                    "change_pct": evaluation["actual_change_pct"],
                },
                "evaluation": {
                    "match_signal": evaluation["match_signal"],
                    "match_action": evaluation["match_action"],
                    "strategy_return_pct": evaluation["strategy_return_pct"],
                    "trade_pnl_usd": pnl_usd,
                },
                "stats": stats.copy(),
            }

            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            log_line(
                f"Actual: {evaluation['actual_signal']} "
                f"(change={evaluation['actual_change_pct']:.2f}%, strategy={evaluation['strategy_return_pct']:.2f}%, pnl=${pnl_usd:.2f}, balance=${stats['balance_usd']:.2f}) "
                f"match={evaluation['match_signal']}"
            )

    except KeyboardInterrupt:
        log_line("\nStopped by user.")

    log_line(summarize_run(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
