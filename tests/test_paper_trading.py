from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

import pytest

from paper_trading import PaperExecutionGateway, PaperTradeSession, PaperTradingBot


def test_paper_execution_gateway_applies_slippage_and_fees():
    gateway = PaperExecutionGateway(fee_rate=0.001, slippage_bps=10)

    buy_order = gateway.submit_market_order(
        symbol="BTCUSDT",
        side="BUY",
        reference_price=100.0,
        quote_amount=1000.0,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert buy_order.status == "FILLED"
    assert buy_order.executed_price == pytest.approx(100.1)
    assert buy_order.fee_paid == pytest.approx(1.0)
    assert buy_order.cash_delta == pytest.approx(-1000.0)
    assert buy_order.executed_quantity == pytest.approx((1000.0 - 1.0) / 100.1)

    sell_order = gateway.submit_market_order(
        symbol="BTCUSDT",
        side="SELL",
        reference_price=100.0,
        quantity=2.0,
    )

    assert sell_order.status == "FILLED"
    assert sell_order.executed_price == pytest.approx(99.9)
    assert sell_order.fee_paid == pytest.approx(0.1998)
    assert sell_order.cash_delta == pytest.approx((2.0 * 99.9) - 0.1998)


def test_paper_trade_session_opens_and_closes_long_positions():
    gateway = PaperExecutionGateway(fee_rate=0.0, slippage_bps=0.0)
    session = PaperTradeSession(
        "BTCUSDT",
        gateway=gateway,
        starting_balance=1000.0,
        stake_fraction=0.5,
        min_trade_confidence=0.6,
    )

    entry = session.step(
        {
            "action": "BUY",
            "trade_confidence": 0.8,
            "signal": "UP",
            "live_price": 100.0,
        },
        live_price=100.0,
    )

    assert entry["event"] == "entry"
    assert session.position is not None
    assert session.balance == pytest.approx(500.0)

    exit_event = session.step(
        {
            "action": "SELL",
            "trade_confidence": 0.8,
            "signal": "DOWN",
            "live_price": 110.0,
        },
        live_price=110.0,
    )

    assert exit_event["event"] == "exit"
    assert session.position is None
    assert session.balance == pytest.approx(1050.0)
    assert exit_event["profit_loss"] == pytest.approx(50.0)


def test_paper_trading_bot_refreshes_symbols_in_parallel():
    release_btc = threading.Event()
    eth_called = threading.Event()
    btc_called = threading.Event()

    def summary_provider(symbol: str):
        if symbol == "BTCUSDT":
            btc_called.set()
            release_btc.wait(timeout=5)
        else:
            eth_called.set()
        return {
            "symbol": symbol,
            "action": "NO_TRADE",
            "trade_confidence": 0.0,
            "signal": "NEUTRAL",
            "live_price": 100.0,
            "last_close": 100.0,
            "predicted_close": 100.0,
            "verdict": "NO_TRADE",
            "timestamp_utc": "2026-06-01T12:00:00Z",
        }

    bot = PaperTradingBot(
        ["BTCUSDT", "ETHUSDT"],
        summary_provider=summary_provider,
        starting_balance=1000.0,
        stake_fraction=0.5,
        max_workers=2,
    )

    result_holder: dict[str, object] = {}

    def run_bot() -> None:
        result_holder["events"] = bot.refresh_all()

    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()

    assert btc_called.wait(timeout=1.0) is True
    assert eth_called.wait(timeout=1.0) is True
    release_btc.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    events = result_holder["events"]
    assert isinstance(events, list)
    assert {event["symbol"] for event in events} == {"BTCUSDT", "ETHUSDT"}
    assert all(event["event"] == "idle" for event in events)
