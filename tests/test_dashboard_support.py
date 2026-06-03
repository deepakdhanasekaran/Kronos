from __future__ import annotations

import json

import pytest

from dashboard_support import (
    WatchlistStore,
    dashboard_row_from_summary,
    dashboard_placeholder_row,
    dedupe_symbols,
    merge_symbol_lists,
    select_top_usdt_symbols,
    row_emphasis_from_verdict,
    row_needs_refresh,
    validate_usdt_symbol,
)


def test_select_top_usdt_symbols_ranks_by_quote_volume():
    tickers = [
        {"symbol": "XRPUSDT", "quoteVolume": "100"},
        {"symbol": "BTCUSDT", "quoteVolume": "900"},
        {"symbol": "ETHUSDT", "quoteVolume": "400"},
        {"symbol": "ETHBTC", "quoteVolume": "999"},
    ]

    ranked = select_top_usdt_symbols(tickers, limit=2)

    assert [item["symbol"] for item in ranked] == ["BTCUSDT", "ETHUSDT"]
    assert [item["rank"] for item in ranked] == [1, 2]


def test_validate_usdt_symbol_rejects_invalid_pairs():
    with pytest.raises(ValueError, match="USDT pair"):
        validate_usdt_symbol("BTC")


def test_merge_symbol_lists_preserves_order_and_removes_duplicates():
    merged = merge_symbol_lists(["BTCUSDT", "ETHUSDT"], ["ETHUSDT", "SOLUSDT"])

    assert merged == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_dashboard_row_from_summary_maps_requested_fields():
    row = dashboard_row_from_summary(
        "BTCUSDT",
        {
            "last_close": 100.0,
            "predicted_close": 110.0,
            "verdict": "Strong BUY",
            "action": "BUY",
            "signal": "UP",
            "confidence": 0.6,
            "trade_confidence": 0.4,
            "signal_counts": {"UP": 2, "DOWN": 3, "NEUTRAL": 0},
        },
    )

    assert row["symbol"] == "BTCUSDT"
    assert row["last_close"] == 100.0
    assert row["predicted_close"] == 110.0
    assert row["verdict"] == "Strong BUY"
    assert row["action"] == "BUY"
    assert row["signal"] == "UP"
    assert row["agreement"] == 0.6
    assert row["trade_confidence"] == 0.4
    assert row["signal_counts"] == {"UP": 2, "DOWN": 3, "NEUTRAL": 0}
    assert row["emphasis"] == "strong-buy"
    assert row["needs_refresh"] is False


def test_dashboard_placeholder_row_marks_symbol_as_loading():
    row = dashboard_placeholder_row("BTCUSDT", source="Custom", rank=7)

    assert row["symbol"] == "BTCUSDT"
    assert row["source"] == "Custom"
    assert row["rank"] == 7
    assert row["ready"] is False
    assert row["signal"] == "WARMING"
    assert row["signal_counts"] == {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    assert row["emphasis"] == "neutral"
    assert row["needs_refresh"] is True


def test_row_emphasis_from_verdict_highlights_strong_directions():
    assert row_emphasis_from_verdict("Strong BUY", "BUY") == "strong-buy"
    assert row_emphasis_from_verdict("Strong SELL", "SELL") == "strong-sell"
    assert row_emphasis_from_verdict("NO_TRADE", "NO_TRADE") == "neutral"


def test_row_needs_refresh_detects_pending_and_error_rows():
    assert row_needs_refresh({"ready": False}) is True
    assert row_needs_refresh({"refreshing": True}) is True
    assert row_needs_refresh({"last_error": "boom"}) is True
    assert row_needs_refresh({"ready": True, "refreshing": False, "last_error": None}) is False


def test_watchlist_store_persists_and_dedupes(tmp_path):
    path = tmp_path / "watchlist.json"
    store = WatchlistStore(path)

    assert store.load() == []
    assert store.add("ethusdt") == ["ETHUSDT"]
    assert store.add("btcusdt") == ["ETHUSDT", "BTCUSDT"]
    assert store.add("btcusdt") == ["ETHUSDT", "BTCUSDT"]

    payload = json.loads(path.read_text())
    assert payload["symbols"] == ["ETHUSDT", "BTCUSDT"]

    assert store.remove("ETHUSDT") == ["BTCUSDT"]
    assert store.load() == ["BTCUSDT"]


def test_dedupe_symbols_preserves_first_seen_order():
    assert dedupe_symbols(["BTCUSDT", "ethusdt", "BTCUSDT"]) == ["BTCUSDT", "ETHUSDT"]
