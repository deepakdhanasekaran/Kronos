#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/polymarket_btc_actions.jsonl"

mkdir -p "$LOG_DIR"

slug="$("$PYTHON_BIN" - <<'PY'
from crypto_predictor import resolve_current_polymarket_slug

print(resolve_current_polymarket_slug())
PY
)"

action="$("$PYTHON_BIN" "$ROOT_DIR/crypto_predictor.py" --polymarket-slug "$slug" --polymarket-only | tail -n 1 | tr -d '\r')"

record="$("$PYTHON_BIN" - <<PY
from datetime import datetime, timezone
import json

from crypto_predictor import build_polymarket_summary, fetch_polymarket_market_by_slug

slug = "${slug}"
summary = build_polymarket_summary(slug)
market = fetch_polymarket_market_by_slug(slug)
record = {
    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "slug": slug,
    "title": summary["title"],
    "action": summary["action"],
    "market_signal": summary["market_signal"],
    "market_probability": summary["market_probability"],
    "closed": bool(market.get("closed", False)),
    "outcome": summary["market_signal"] if market.get("closed", False) else "pending",
}
print(json.dumps(record))
PY
)"

printf '%s\n' "$record" >> "$LOG_FILE"
printf '%s\n' "$action"
