from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DASHBOARD_INTERVAL = "15m"
DEFAULT_DASHBOARD_LOOKBACK = 256
DEFAULT_DASHBOARD_PRED_LEN = 1
DEFAULT_REFRESH_SECONDS = 10
DEFAULT_TOP_SYMBOL_LIMIT = 30
WATCHLIST_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}USDT$")


def normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def validate_usdt_symbol(symbol: Any) -> str:
    normalized = normalize_symbol(symbol)
    if not WATCHLIST_SYMBOL_RE.fullmatch(normalized):
        raise ValueError("Symbol must be a USDT pair such as BTCUSDT.")
    return normalized


def dedupe_symbols(symbols: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def merge_symbol_lists(*symbol_groups: Iterable[Any]) -> list[str]:
    merged: list[str] = []
    for group in symbol_groups:
        merged.extend(group)
    return dedupe_symbols(merged)


def select_top_usdt_symbols(
    tickers: Iterable[dict[str, Any]],
    limit: int = DEFAULT_TOP_SYMBOL_LIMIT,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for entry in tickers:
        symbol = normalize_symbol(entry.get("symbol"))
        if not symbol.endswith("USDT"):
            continue
        if not WATCHLIST_SYMBOL_RE.fullmatch(symbol):
            continue
        try:
            quote_volume = float(entry.get("quoteVolume", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        ranked.append(
            {
                "symbol": symbol,
                "quote_volume": quote_volume,
                "last_price": float(entry.get("lastPrice", 0.0) or 0.0),
                "price_change_percent": float(entry.get("priceChangePercent", 0.0) or 0.0),
            }
        )

    ranked.sort(key=lambda item: (-item["quote_volume"], item["symbol"]))
    for index, item in enumerate(ranked[:limit], start=1):
        item["rank"] = index
    return ranked[:limit]


def dashboard_row_from_summary(
    symbol: str,
    summary: dict[str, Any],
    *,
    source: str = "Top 30",
    rank: int | None = None,
) -> dict[str, Any]:
    signal_counts = summary.get("signal_counts") or {}
    row = {
        "symbol": normalize_symbol(symbol),
        "source": source,
        "rank": rank,
        "last_close": float(summary.get("last_close", 0.0) or 0.0),
        "predicted_close": float(summary.get("predicted_close", 0.0) or 0.0),
        "verdict": str(summary.get("verdict", "NO_TRADE")),
        "action": str(summary.get("action", "NO_TRADE")),
        "signal": str(summary.get("signal", "NEUTRAL")),
        "agreement": float(summary.get("confidence", summary.get("agreement", 0.0)) or 0.0),
        "trade_confidence": float(summary.get("trade_confidence", summary.get("confidence", 0.0)) or 0.0),
        "signal_counts": {
            "UP": int(signal_counts.get("UP", 0) or 0),
            "DOWN": int(signal_counts.get("DOWN", 0) or 0),
            "NEUTRAL": int(signal_counts.get("NEUTRAL", 0) or 0),
        },
    }
    if "live_price" in summary:
        row["live_price"] = float(summary.get("live_price", 0.0) or 0.0)
    if "confidence_samples" in summary:
        row["confidence_samples"] = int(summary.get("confidence_samples", 0) or 0)
    row["emphasis"] = row_emphasis_from_verdict(row["verdict"], row["action"])
    row["needs_refresh"] = row_needs_refresh(row)
    return row


def dashboard_placeholder_row(
    symbol: str,
    *,
    source: str = "Top 30",
    rank: int | None = None,
) -> dict[str, Any]:
    return {
        "symbol": normalize_symbol(symbol),
        "source": source,
        "rank": rank,
        "ready": False,
        "last_close": None,
        "predicted_close": None,
        "verdict": "WARMING",
        "action": "WARMING",
        "signal": "WARMING",
        "agreement": None,
        "trade_confidence": None,
        "signal_counts": {"UP": 0, "DOWN": 0, "NEUTRAL": 0},
        "emphasis": "neutral",
        "needs_refresh": True,
    }


def row_emphasis_from_verdict(verdict: Any, action: Any = None) -> str:
    normalized_verdict = str(verdict or "").upper()
    normalized_action = str(action or "").upper()
    if "STRONG BUY" in normalized_verdict or normalized_action == "BUY":
        return "strong-buy"
    if "STRONG SELL" in normalized_verdict or normalized_action == "SELL":
        return "strong-sell"
    return "neutral"


def row_needs_refresh(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("refreshing"):
        return True
    if row.get("last_error"):
        return True
    if row.get("ready") is False:
        return True
    return False


def parse_watchlist_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return dedupe_symbols(payload)
    if isinstance(payload, dict):
        raw_symbols = payload.get("symbols", [])
        if isinstance(raw_symbols, list):
            return dedupe_symbols(raw_symbols)
    return []


class WatchlistStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid watchlist registry: {self.path}") from exc
        return parse_watchlist_payload(data)

    def save(self, symbols: Iterable[Any]) -> list[str]:
        ordered = dedupe_symbols(symbols)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbols": ordered,
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)
        return ordered

    def add(self, symbol: Any) -> list[str]:
        normalized = validate_usdt_symbol(symbol)
        return self.save(merge_symbol_lists(self.load(), [normalized]))

    def remove(self, symbol: Any) -> list[str]:
        normalized = validate_usdt_symbol(symbol)
        return self.save([item for item in self.load() if item != normalized])
