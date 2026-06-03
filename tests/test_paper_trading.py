from __future__ import annotations

import asyncio
import importlib.util
import json
import hmac
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from paper_trading import PaperExecutionGateway, PaperTradeSession, PaperTradingBot
from paper_trading import BinanceSpotDemoExecutionGateway


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
        max_trade_quote_usdt=1000.0,
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


def test_paper_trade_session_caps_entry_notional_at_fifty_usdt():
    gateway = PaperExecutionGateway(fee_rate=0.0, slippage_bps=0.0)
    session = PaperTradeSession(
        "BTCUSDT",
        gateway=gateway,
        starting_balance=1000.0,
        stake_fraction=1.0,
        max_trade_quote_usdt=50.0,
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
    assert entry["order"].gross_notional == pytest.approx(50.0)
    assert entry["order"].executed_quantity == pytest.approx(0.5)
    assert session.balance == pytest.approx(950.0)


def test_paper_trade_session_auto_closes_on_take_profit():
    gateway = PaperExecutionGateway(fee_rate=0.0, slippage_bps=0.0)
    session = PaperTradeSession(
        "BTCUSDT",
        gateway=gateway,
        starting_balance=1000.0,
        stake_fraction=1.0,
        max_trade_quote_usdt=1000.0,
        min_trade_confidence=0.6,
        take_profit_pct=1.0,
        stop_loss_pct=1.0,
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

    exit_event = session.step(
        {
            "action": "BUY",
            "trade_confidence": 0.8,
            "signal": "UP",
            "live_price": 101.0,
        },
        live_price=101.0,
    )

    assert exit_event["event"] == "exit"
    assert exit_event["exit_reason"] == "take_profit"
    assert session.position is None
    assert session.balance == pytest.approx(1010.0)
    assert exit_event["profit_loss"] == pytest.approx(10.0)


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


def test_binance_paper_trade_bot_serializes_dataclass_records():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "binance_paper_trade_bot.py"
    spec = importlib.util.spec_from_file_location("binance_paper_trade_bot", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gateway = PaperExecutionGateway(fee_rate=0.0, slippage_bps=0.0)
    order = gateway.submit_market_order(
        symbol="BTCUSDT",
        side="BUY",
        reference_price=100.0,
        quote_amount=100.0,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    record = {
        "event": "cycle",
        "events": [{"order": order}],
        "snapshot": {"last_snapshot": {"order": order}},
    }

    encoded = json.dumps(record, default=module._json_default)

    assert "paper-" in encoded


def test_paper_trading_bot_snapshot_includes_open_position_mark_to_market():
    def summary_provider(symbol: str):
        return {
            "symbol": symbol,
            "action": "BUY",
            "trade_confidence": 0.9,
            "signal": "UP",
            "live_price": 100.0,
            "last_close": 100.0,
            "predicted_close": 101.0,
            "verdict": "Strong BUY",
            "timestamp_utc": "2026-06-01T12:00:00Z",
        }

    bot = PaperTradingBot(
        ["BTCUSDT"],
        summary_provider=summary_provider,
        starting_balance=1000.0,
        stake_fraction=1.0,
        gateway=PaperExecutionGateway(fee_rate=0.0, slippage_bps=0.0),
        take_profit_pct=1.0,
        stop_loss_pct=1.0,
        max_workers=1,
    )

    bot.refresh_all()
    snapshot = bot.snapshot()
    session = snapshot["sessions"]["BTCUSDT"]

    assert session["position"] is not None
    assert session["position"]["live_price"] == pytest.approx(100.0)
    assert session["position"]["take_profit_price"] == pytest.approx(101.0)
    assert session["position"]["stop_loss_price"] == pytest.approx(99.0)


def test_binance_demo_gateway_signs_and_parses_market_orders(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "symbol": "BTCUSDT",
                    "orderId": 12345,
                    "clientOrderId": "agent-BTCUSDT-entry",
                    "status": "FILLED",
                    "executedQty": "0.10000000",
                    "cummulativeQuoteQty": "10.00000000",
                    "fills": [
                        {
                            "price": "100.00000000",
                            "qty": "0.10000000",
                            "commission": "0.01000000",
                            "commissionAsset": "USDT",
                        }
                    ],
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        captured["headers"] = dict(request.headers)
        captured["method"] = request.get_method()
        return FakeResponse()

    monkeypatch.setattr("paper_trading.urlopen", fake_urlopen)

    gateway = BinanceSpotDemoExecutionGateway(
        api_key="demo-key",
        secret_key="demo-secret",
        base_url="https://demo-api.binance.com",
    )

    order = gateway.submit_market_order(
        symbol="BTCUSDT",
        side="BUY",
        reference_price=100.0,
        quote_amount=10.0,
        client_order_id="BTCUSDT-entry",
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert captured["url"].endswith("/api/v3/order")
    assert captured["method"] == "POST"
    assert any(key.lower() == "x-mbx-apikey" for key in captured["headers"])
    assert "newClientOrderId=agent-BTCUSDT-entry" in captured["body"]
    assert "symbol=BTCUSDT" in captured["body"]
    assert "side=BUY" in captured["body"]
    assert "type=MARKET" in captured["body"]
    assert "signature=" in captured["body"]

    expected_query = (
        "symbol=BTCUSDT&side=BUY&type=MARKET&newOrderRespType=FULL&recvWindow=5000"
        f"&timestamp={int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)}"
        "&newClientOrderId=agent-BTCUSDT-entry&quoteOrderQty=10.00000000"
    )
    expected_signature = hmac.new(
        b"demo-secret",
        expected_query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert captured["body"] == f"{expected_query}&signature={expected_signature}"
    assert order.order_id == "12345"
    assert order.status == "FILLED"
    assert order.executed_quantity == pytest.approx(0.1)
    assert order.gross_notional == pytest.approx(10.0)
    assert order.fee_paid == pytest.approx(0.01)
