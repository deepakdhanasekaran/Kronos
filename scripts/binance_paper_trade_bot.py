#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crypto_predictor import (  # noqa: E402
    predict_binance_direction,
    predict_binance_direction_from_live_server,
)
from paper_trading import BinanceSpotDemoExecutionGateway, PaperExecutionGateway, PaperTradingBot  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Binance-style paper trading bot.")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="Comma-separated USDT symbols to monitor.")
    parser.add_argument("--interval", default="15m", help="Binance candle interval.")
    parser.add_argument("--lookback", type=int, default=256, help="Number of candles to feed Kronos.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon in candles.")
    parser.add_argument("--sample-count", type=int, default=5, help="Stochastic sampling count.")
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.05, help="Neutral threshold percent.")
    parser.add_argument("--confidence-samples", type=int, default=5, help="Confidence sample count.")
    parser.add_argument("--poll-seconds", type=int, default=10, help="How often to refresh each symbol.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N loops. 0 means run until Ctrl-C.")
    parser.add_argument(
        "--execution-mode",
        choices=["sim", "demo"],
        default=os.environ.get("BINANCE_EXECUTION_MODE", "sim"),
        help="Execution backend: local simulator or Binance spot demo account.",
    )
    parser.add_argument(
        "--binance-base-url",
        default=os.environ.get("BINANCE_SPOT_BASE_URL", "https://demo-api.binance.com"),
        help="Binance spot base URL for demo execution.",
    )
    parser.add_argument("--starting-balance", type=float, default=1000.0, help="Starting paper balance per symbol.")
    parser.add_argument("--stake-fraction", type=float, default=1.0, help="Fraction of balance to deploy on entry.")
    parser.add_argument(
        "--max-trade-usdt",
        type=float,
        default=50.0,
        help="Maximum quote notional per entry order in USDT.",
    )
    parser.add_argument("--min-trade-confidence", type=float, default=0.6, help="Minimum trade confidence to enter.")
    parser.add_argument("--take-profit-pct", type=float, default=0.5, help="Auto-close long positions after this gain percent.")
    parser.add_argument("--stop-loss-pct", type=float, default=0.25, help="Auto-close long positions after this loss percent.")
    parser.add_argument(
        "--exit-on-opposite-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit an open long when the model flips to SELL.",
    )
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Paper trading fee rate.")
    parser.add_argument("--slippage-bps", type=float, default=10.0, help="Paper trading slippage in basis points.")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel refresh workers.")
    parser.add_argument("--live-url", default="http://127.0.0.1:8765", help="Warm prediction server URL.")
    parser.add_argument("--local-model", action="store_true", help="Load the model locally instead of using --live-url.")
    parser.add_argument("--model-size", default="base", choices=["small", "base"], help="Local model size.")
    parser.add_argument("--log-file", default="logs/binance_paper_trade_bot.jsonl", help="JSONL audit log path.")
    return parser


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _fmt_money(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "--"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "--"


def _render_blotter(snapshot: dict[str, Any]) -> str:
    sessions = snapshot.get("sessions") if isinstance(snapshot, dict) else {}
    if not isinstance(sessions, dict) or not sessions:
        return "No sessions."

    header = (
        f"{'SYMBOL':<10} {'STATE':<8} {'ENTRY':>12} {'LIVE':>12} "
        f"{'TP':>12} {'SL':>12} {'UPNL':>12} {'RPNL':>12}"
    )
    lines = [header, "-" * len(header)]
    for symbol, session in sessions.items():
        last_snapshot = session.get("last_snapshot") if isinstance(session, dict) else {}
        event = str(last_snapshot.get("event", "idle")).upper() if isinstance(last_snapshot, dict) else "IDLE"
        position = session.get("position") if isinstance(session, dict) else None
        if isinstance(position, dict):
            state = "OPEN"
            entry = _fmt_money(position.get("entry_price"))
            live = _fmt_money(position.get("live_price"))
            tp = _fmt_money(position.get("take_profit_price"))
            sl = _fmt_money(position.get("stop_loss_price"))
            upnl = _fmt_money(position.get("unrealized_pnl"))
            rpnl = "--"
        else:
            state = "CLOSED" if event == "EXIT" else "FLAT"
            closed_position = last_snapshot.get("closed_position") if isinstance(last_snapshot, dict) else None
            entry = _fmt_money(closed_position.get("entry_price")) if isinstance(closed_position, dict) else "--"
            live = _fmt_money(closed_position.get("live_price")) if isinstance(closed_position, dict) else "--"
            tp = _fmt_money(closed_position.get("take_profit_price")) if isinstance(closed_position, dict) else "--"
            sl = _fmt_money(closed_position.get("stop_loss_price")) if isinstance(closed_position, dict) else "--"
            upnl = "--"
            rpnl = _fmt_money(last_snapshot.get("profit_loss")) if isinstance(last_snapshot, dict) else "--"

        lines.append(
            f"{symbol:<10} {state:<8} {entry:>12} {live:>12} {tp:>12} {sl:>12} {upnl:>12} {rpnl:>12}"
        )
    return "\n".join(lines)


def _build_summary_provider(args: argparse.Namespace):
    if args.local_model:
        def provider(symbol: str) -> dict[str, Any]:
            result = predict_binance_direction(
                symbol=symbol,
                interval=args.interval,
                lookback=args.lookback,
                pred_len=args.pred_len,
                sample_count=args.sample_count,
                neutral_threshold_pct=args.neutral_threshold_pct,
                confidence_samples=args.confidence_samples,
                model_name=f"NeoQuasar/Kronos-{args.model_size}",
            )
            summary = dict(result["summary"])
            summary["symbol"] = symbol
            summary["timestamp_utc"] = _utc_now_iso()
            return summary

        return provider

    def provider(symbol: str) -> dict[str, Any]:
        result = predict_binance_direction_from_live_server(
            args.live_url,
            symbol=symbol,
            interval=args.interval,
            lookback=args.lookback,
            pred_len=args.pred_len,
            sample_count=args.sample_count,
            neutral_threshold_pct=args.neutral_threshold_pct,
            confidence_samples=args.confidence_samples,
        )
        summary = dict(result["summary"])
        summary["symbol"] = symbol
        summary["timestamp_utc"] = _utc_now_iso()
        return summary

    return provider


def main() -> int:
    args = build_parser().parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.execution_mode == "demo":
        gateway = BinanceSpotDemoExecutionGateway(
            api_key=os.environ.get("BINANCE_DEMO_API_KEY", ""),
            secret_key=os.environ.get("BINANCE_DEMO_SECRET_KEY", ""),
            base_url=args.binance_base_url,
        )
    else:
        gateway = PaperExecutionGateway(fee_rate=args.fee_rate, slippage_bps=args.slippage_bps)
    bot = PaperTradingBot(
        symbols,
        summary_provider=_build_summary_provider(args),
        gateway=gateway,
        starting_balance=args.starting_balance,
        stake_fraction=args.stake_fraction,
        max_trade_quote_usdt=args.max_trade_usdt,
        min_trade_confidence=args.min_trade_confidence,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        exit_on_opposite_signal=args.exit_on_opposite_signal,
        max_workers=args.max_workers,
    )

    print(f"Starting paper trade bot for {', '.join(symbols)}", flush=True)
    print(f"Refresh cadence: {args.poll_seconds}s", flush=True)
    print(f"Logging to: {log_path}", flush=True)

    cycles = 0
    try:
        while True:
            if args.max_cycles and cycles >= args.max_cycles:
                break
            cycles += 1
            events = bot.refresh_all()
            snapshot = bot.snapshot()
            record = {
                "event": "cycle",
                "timestamp_utc": _utc_now_iso(),
                "cycle": cycles,
                "events": events,
                "snapshot": snapshot,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default) + "\n")
            for event in events:
                suffix_parts = [f"balance={event.get('balance_after', 0.0):.2f}"]
                if event.get("event") == "exit":
                    suffix_parts.append(f"pnl={event.get('profit_loss', 0.0):.2f}")
                    if event.get("exit_reason"):
                        suffix_parts.append(f"reason={event['exit_reason']}")
                print(f"{event['symbol']} {event['event']} " + " ".join(suffix_parts), flush=True)
            print(_render_blotter(snapshot), flush=True)
            time.sleep(max(1, int(args.poll_seconds)))
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)

    final_snapshot = bot.snapshot()
    print(_render_blotter(final_snapshot), flush=True)
    print(json.dumps(final_snapshot, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
