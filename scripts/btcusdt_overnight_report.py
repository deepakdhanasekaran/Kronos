#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crypto_predictor import load_jsonl_records, render_overnight_audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a readable backtest report from an overnight Kronos JSONL log."
    )
    parser.add_argument(
        "--log-file",
        default="logs/btcusdt_overnight_papertrade.jsonl",
        help="Path to the overnight JSONL log file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the last N rows in the table. 0 means show all rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_jsonl_records(args.log_file)
    report = render_overnight_audit_report(records, limit=args.limit or None)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
