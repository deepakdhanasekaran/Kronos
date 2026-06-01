#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/polymarket_btc_audit.jsonl"
SLUG="${1:-}"
EDGE_PCT="${EDGE_PCT:-0.05}"
POLL_SECONDS="${POLL_SECONDS:-15}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
WAIT_FOR_CLOSE="${WAIT_FOR_CLOSE:-1}"

mkdir -p "$LOG_DIR"

if [[ -z "$SLUG" ]]; then
  SLUG="$("$PYTHON_BIN" - <<'PY'
from crypto_predictor import resolve_current_polymarket_slug

print(resolve_current_polymarket_slug())
PY
)"
fi

initial_record="$("$PYTHON_BIN" - <<PY
import json

from crypto_predictor import build_polymarket_summary, fetch_polymarket_market_by_slug, extract_polymarket_final_outcome

slug = "${SLUG}"
edge_pct = float("${EDGE_PCT}")
summary = build_polymarket_summary(slug, edge_threshold_pct=edge_pct)
market = fetch_polymarket_market_by_slug(slug)
closed = bool(market.get("closed", False) or market.get("resolved", False) or market.get("resolvedAt"))
record = {
    "slug": summary["slug"],
    "title": summary["title"],
    "prediction": summary["action"],
    "market_signal": summary["market_signal"],
    "market_probability": summary["market_probability"],
    "closed": closed,
    "final_outcome": extract_polymarket_final_outcome(market) or "pending",
}
print(json.dumps(record))
PY
)"

printf 'Slug: %s\n' "$SLUG"
printf '%s\n' "$initial_record"
printf '%s\n' "$initial_record" >> "$LOG_FILE"

if [[ "$WAIT_FOR_CLOSE" != "1" ]]; then
  exit 0
fi

printf 'Waiting for close... poll=%ss max_wait=%ss\n' "$POLL_SECONDS" "$MAX_WAIT_SECONDS"
start_epoch="$(date +%s)"
while true; do
  closed="$("$PYTHON_BIN" - <<PY
from crypto_predictor import resolve_polymarket_market, market_is_closed
slug = "${SLUG}"
event, market = resolve_polymarket_market(slug)
print("1" if (market_is_closed(market) or market_is_closed(event)) else "0")
PY
)"

  if [[ "$closed" == "1" ]]; then
    record="$("$PYTHON_BIN" - <<PY
import json

from crypto_predictor import build_polymarket_summary, extract_polymarket_final_outcome, resolve_polymarket_market, market_is_closed

slug = "${SLUG}"
edge_pct = float("${EDGE_PCT}")
summary = build_polymarket_summary(slug, edge_threshold_pct=edge_pct)
event, market = resolve_polymarket_market(slug)
payload = {
    "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "slug": summary["slug"],
    "title": summary["title"],
    "prediction": summary["action"],
    "market_signal": summary["market_signal"],
    "market_probability": summary["market_probability"],
    "closed": market_is_closed(market) or market_is_closed(event),
    "final_outcome": extract_polymarket_final_outcome(market) or extract_polymarket_final_outcome(event) or "pending",
}
print(json.dumps(payload))
PY
)"
    printf '%s\n' "$record" >> "$LOG_FILE"
    printf '%s\n' "$record"
    exit 0
  fi

  now_epoch="$(date +%s)"
  elapsed="$((now_epoch - start_epoch))"
  if (( elapsed >= MAX_WAIT_SECONDS )); then
    timeout_record="$("$PYTHON_BIN" - <<PY
import json

from crypto_predictor import build_polymarket_summary, extract_polymarket_final_outcome, resolve_polymarket_market, market_is_closed

slug = "${SLUG}"
edge_pct = float("${EDGE_PCT}")
summary = build_polymarket_summary(slug, edge_threshold_pct=edge_pct)
event, market = resolve_polymarket_market(slug)
payload = {
    "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "slug": summary["slug"],
    "title": summary["title"],
    "prediction": summary["action"],
    "market_signal": summary["market_signal"],
    "market_probability": summary["market_probability"],
    "closed": market_is_closed(market) or market_is_closed(event),
    "final_outcome": extract_polymarket_final_outcome(market) or extract_polymarket_final_outcome(event) or "pending",
    "timed_out": True,
}
print(json.dumps(payload))
PY
)"
    printf '%s\n' "$timeout_record" >> "$LOG_FILE"
    printf 'Still open after %ss, stopping.\n' "$elapsed"
    printf '%s\n' "$timeout_record"
    exit 0
  fi

  printf 'Still open after %ss; polling again in %ss...\n' "$elapsed" "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done
