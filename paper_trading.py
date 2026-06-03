from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional
from uuid import uuid4

from dashboard_support import dedupe_symbols, normalize_symbol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_z(value: Optional[datetime] = None) -> str:
    timestamp = value or _utc_now()
    return timestamp.isoformat().replace("+00:00", "Z")


def _positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    symbol: str
    side: str
    status: str
    reference_price: float
    executed_price: float
    executed_quantity: float
    gross_notional: float
    fee_paid: float
    cash_delta: float
    slippage_pct: float
    requested_quote_amount: float | None = None
    requested_quantity: float | None = None
    client_order_id: str | None = None
    created_at: str = field(default_factory=_iso_z)
    filled_at: str = field(default_factory=_iso_z)


@dataclass
class PaperPosition:
    symbol: str
    quantity: float
    entry_price: float
    entry_order_id: str
    opened_at: str
    invested_quote: float
    entry_fee: float
    entry_slippage_pct: float
    take_profit_price: float
    stop_loss_price: float

    def exit_reason(self, live_price: float) -> str | None:
        if live_price >= self.take_profit_price:
            return "take_profit"
        if live_price <= self.stop_loss_price:
            return "stop_loss"
        return None

    def mark_to_market(self, live_price: float) -> dict[str, Any]:
        gross_value = self.quantity * live_price
        cost_basis = self.invested_quote + self.entry_fee
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "live_price": float(live_price),
            "gross_value": gross_value,
            "cost_basis": cost_basis,
            "unrealized_pnl": gross_value - cost_basis,
            "unrealized_return_pct": ((gross_value - cost_basis) / cost_basis * 100.0) if cost_basis else 0.0,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
        }


class PaperExecutionGateway:
    def __init__(self, *, fee_rate: float = 0.001, slippage_bps: float = 10.0) -> None:
        self.fee_rate = max(0.0, float(fee_rate))
        self.slippage_bps = max(0.0, float(slippage_bps))

    def _slippage_multiplier(self, side: str) -> float:
        slip = self.slippage_bps / 10000.0
        if side == "BUY":
            return 1.0 + slip
        return 1.0 - slip

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        reference_price: float,
        quote_amount: float | None = None,
        quantity: float | None = None,
        client_order_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> PaperOrder:
        normalized_symbol = normalize_symbol(symbol)
        normalized_side = str(side or "").strip().upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL.")
        if quote_amount is None and quantity is None:
            raise ValueError("Either quote_amount or quantity is required.")
        if quote_amount is not None and quantity is not None:
            raise ValueError("Provide either quote_amount or quantity, not both.")

        reference_price = _positive_float(reference_price, "reference_price")
        executed_price = reference_price * self._slippage_multiplier(normalized_side)

        requested_quote_amount = float(quote_amount) if quote_amount is not None else None
        requested_quantity = float(quantity) if quantity is not None else None

        if normalized_side == "BUY":
            gross_notional = float(quote_amount)
            fee_paid = gross_notional * self.fee_rate
            effective_quote = gross_notional - fee_paid
            executed_quantity = effective_quote / executed_price
            cash_delta = -gross_notional
        else:
            executed_quantity = float(quantity)
            gross_notional = executed_quantity * executed_price
            fee_paid = gross_notional * self.fee_rate
            cash_delta = gross_notional - fee_paid

        created_at = _iso_z(timestamp)
        return PaperOrder(
            order_id=f"paper-{uuid4().hex}",
            symbol=normalized_symbol,
            side=normalized_side,
            status="FILLED",
            reference_price=reference_price,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            gross_notional=gross_notional,
            fee_paid=fee_paid,
            cash_delta=cash_delta,
            slippage_pct=self.slippage_bps / 100.0,
            requested_quote_amount=requested_quote_amount,
            requested_quantity=requested_quantity,
            client_order_id=client_order_id,
            created_at=created_at,
            filled_at=created_at,
        )


class BinanceSpotDemoExecutionGateway:
    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        base_url: str = "https://demo-api.binance.com",
        user_agent: str = "KronosPaperTrader/1.0",
        recv_window: int = 5000,
        client_order_prefix: str = "agent-",
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.secret_key = str(secret_key or "").strip()
        self.base_url = str(base_url or "").rstrip("/")
        self.user_agent = str(user_agent or "").strip()
        self.recv_window = max(1, int(recv_window))
        self.client_order_prefix = client_order_prefix if client_order_prefix.endswith("-") else f"{client_order_prefix}-"

        if not self.api_key:
            raise ValueError("api_key is required for Binance demo trading.")
        if not self.secret_key:
            raise ValueError("secret_key is required for Binance demo trading.")

    def _sign_query(self, params: list[tuple[str, Any]]) -> str:
        query = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={signature}"

    def _client_order_id(self, client_order_id: str | None, side: str, symbol: str) -> str:
        raw = str(client_order_id or f"{symbol.lower()}-{side.lower()}-{uuid4().hex[:12]}").strip()
        if not raw.startswith(self.client_order_prefix):
            raw = f"{self.client_order_prefix}{raw}"
        return raw[:36]

    def _request_json(self, path: str, params: list[tuple[str, Any]]) -> dict[str, Any]:
        signed_query = self._sign_query(params)
        request = Request(
            f"{self.base_url}{path}",
            data=signed_query.encode("utf-8"),
            headers={
                "X-MBX-APIKEY": self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else exc.reason
            raise RuntimeError(f"Binance demo order failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Binance demo unavailable: {exc.reason}") from exc

        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Binance demo response: {data}")
        return data

    def _commission_paid(self, fills: list[dict[str, Any]], gross_notional: float) -> float:
        quote_commission = 0.0
        for fill in fills:
            commission_asset = str(fill.get("commissionAsset") or "").upper()
            if commission_asset == "USDT":
                try:
                    quote_commission += float(fill.get("commission", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
        if quote_commission > 0.0:
            return quote_commission
        return gross_notional * 0.001

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        reference_price: float,
        quote_amount: float | None = None,
        quantity: float | None = None,
        client_order_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> PaperOrder:
        normalized_symbol = normalize_symbol(symbol)
        normalized_side = str(side or "").strip().upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL.")
        if quote_amount is None and quantity is None:
            raise ValueError("Either quote_amount or quantity is required.")
        if quote_amount is not None and quantity is not None:
            raise ValueError("Provide either quote_amount or quantity, not both.")

        reference_price = _positive_float(reference_price, "reference_price")
        created_at = _iso_z(timestamp)
        timestamp_ms = int((timestamp or _utc_now()).timestamp() * 1000)
        params: list[tuple[str, Any]] = [
            ("symbol", normalized_symbol),
            ("side", normalized_side),
            ("type", "MARKET"),
            ("newOrderRespType", "FULL"),
            ("recvWindow", self.recv_window),
            ("timestamp", timestamp_ms),
        ]
        if client_order_id is not None:
            params.append(("newClientOrderId", self._client_order_id(client_order_id, normalized_side, normalized_symbol)))
        if normalized_side == "BUY":
            params.append(("quoteOrderQty", f"{float(quote_amount):.8f}"))
        else:
            params.append(("quantity", f"{float(quantity):.8f}"))

        response = self._request_json("/api/v3/order", params)
        status = str(response.get("status", "FILLED"))
        order_id = str(response.get("orderId") or response.get("clientOrderId") or f"binance-demo-{uuid4().hex}")
        fills = response.get("fills") if isinstance(response.get("fills"), list) else []
        fills = [fill for fill in fills if isinstance(fill, dict)]

        if normalized_side == "BUY":
            executed_quantity = float(response.get("executedQty") or 0.0)
            if executed_quantity <= 0.0:
                executed_quantity = sum(float(fill.get("qty", 0.0) or 0.0) for fill in fills)
            gross_notional = float(response.get("cummulativeQuoteQty") or 0.0)
            if gross_notional <= 0.0:
                gross_notional = sum(float(fill.get("qty", 0.0) or 0.0) * float(fill.get("price", 0.0) or 0.0) for fill in fills)
            fee_paid = self._commission_paid(fills, gross_notional)
            cash_delta = -gross_notional
            executed_price = gross_notional / executed_quantity if executed_quantity > 0 else reference_price
            slippage_pct = max(0.0, ((executed_price - reference_price) / reference_price) * 100.0)
        else:
            executed_quantity = float(response.get("executedQty") or 0.0)
            if executed_quantity <= 0.0:
                executed_quantity = sum(float(fill.get("qty", 0.0) or 0.0) for fill in fills)
            gross_notional = float(response.get("cummulativeQuoteQty") or 0.0)
            if gross_notional <= 0.0:
                gross_notional = sum(float(fill.get("qty", 0.0) or 0.0) * float(fill.get("price", 0.0) or 0.0) for fill in fills)
            fee_paid = self._commission_paid(fills, gross_notional)
            cash_delta = gross_notional - fee_paid
            executed_price = gross_notional / executed_quantity if executed_quantity > 0 else reference_price
            slippage_pct = max(0.0, ((reference_price - executed_price) / reference_price) * 100.0)

        return PaperOrder(
            order_id=order_id,
            symbol=normalized_symbol,
            side=normalized_side,
            status=status,
            reference_price=reference_price,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            gross_notional=gross_notional,
            fee_paid=fee_paid,
            cash_delta=cash_delta,
            slippage_pct=slippage_pct,
            requested_quote_amount=float(quote_amount) if quote_amount is not None else None,
            requested_quantity=float(quantity) if quantity is not None else None,
            client_order_id=response.get("clientOrderId") or client_order_id,
            created_at=created_at,
            filled_at=created_at,
        )


class PaperTradeSession:
    def __init__(
        self,
        symbol: str,
        *,
        gateway: PaperExecutionGateway | None = None,
        starting_balance: float = 1000.0,
        stake_fraction: float = 1.0,
        max_trade_quote_usdt: float = 50.0,
        min_trade_confidence: float = 0.6,
        take_profit_pct: float = 0.5,
        stop_loss_pct: float = 0.25,
        exit_on_opposite_signal: bool = True,
    ) -> None:
        self.symbol = normalize_symbol(symbol)
        self.gateway = gateway or PaperExecutionGateway()
        self.starting_balance = float(starting_balance)
        self.balance = float(starting_balance)
        self.stake_fraction = max(0.0, float(stake_fraction))
        self.max_trade_quote_usdt = _positive_float(max_trade_quote_usdt, "max_trade_quote_usdt")
        self.min_trade_confidence = max(0.0, float(min_trade_confidence))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.exit_on_opposite_signal = bool(exit_on_opposite_signal)
        self.position: PaperPosition | None = None
        self.orders: list[PaperOrder] = []
        self.events: list[dict[str, Any]] = []
        self.last_snapshot: dict[str, Any] | None = None

    def _stake_quote(self) -> float:
        stake_quote = max(0.0, self.balance * self.stake_fraction)
        return min(stake_quote, self.max_trade_quote_usdt)

    def _position_risk_levels(self, entry_price: float) -> tuple[float, float]:
        take_profit_multiplier = 1.0 + (self.take_profit_pct / 100.0)
        stop_loss_multiplier = max(0.0, 1.0 - (self.stop_loss_pct / 100.0))
        return entry_price * take_profit_multiplier, entry_price * stop_loss_multiplier

    def _entry_from_summary(self, summary: dict[str, Any], live_price: float, timestamp: datetime | None) -> dict[str, Any]:
        stake_quote = self._stake_quote()
        if stake_quote <= 0:
            return {"event": "skipped", "reason": "empty_balance", "symbol": self.symbol}

        order = self.gateway.submit_market_order(
            symbol=self.symbol,
            side="BUY",
            reference_price=live_price,
            quote_amount=stake_quote,
            client_order_id=f"{self.symbol}-entry",
            timestamp=timestamp,
        )
        self.balance += order.cash_delta
        take_profit_price, stop_loss_price = self._position_risk_levels(order.executed_price)
        self.position = PaperPosition(
            symbol=self.symbol,
            quantity=order.executed_quantity,
            entry_price=order.executed_price,
            entry_order_id=order.order_id,
            opened_at=order.filled_at,
            invested_quote=stake_quote,
            entry_fee=order.fee_paid,
            entry_slippage_pct=order.slippage_pct,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
        )
        self.orders.append(order)
        event = {
            "event": "entry",
            "symbol": self.symbol,
            "order": order,
            "position": self.position.mark_to_market(live_price),
            "balance_after": self.balance,
            "timestamp_utc": _iso_z(timestamp),
            "summary": dict(summary),
        }
        self.events.append(event)
        self.last_snapshot = event
        return event

    def _exit_from_summary(
        self,
        summary: dict[str, Any],
        live_price: float,
        timestamp: datetime | None,
        *,
        reason: str,
    ) -> dict[str, Any]:
        assert self.position is not None
        order = self.gateway.submit_market_order(
            symbol=self.symbol,
            side="SELL",
            reference_price=live_price,
            quantity=self.position.quantity,
            client_order_id=f"{self.symbol}-exit",
            timestamp=timestamp,
        )
        self.balance += order.cash_delta
        pnl = order.cash_delta - (self.position.invested_quote + self.position.entry_fee)
        closed_position = self.position.mark_to_market(live_price)
        self.orders.append(order)
        self.position = None
        event = {
            "event": "exit",
            "exit_reason": reason,
            "symbol": self.symbol,
            "order": order,
            "closed_position": closed_position,
            "balance_after": self.balance,
            "profit_loss": pnl,
            "timestamp_utc": _iso_z(timestamp),
            "summary": dict(summary),
        }
        self.events.append(event)
        self.last_snapshot = event
        return event

    def step(
        self,
        summary: dict[str, Any],
        *,
        live_price: float,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        action = str(summary.get("action", "NO_TRADE")).upper()
        confidence = float(summary.get("trade_confidence", summary.get("confidence", 0.0)) or 0.0)
        timestamp = timestamp or _utc_now()

        if self.position is None:
            if action == "BUY" and confidence >= self.min_trade_confidence:
                return self._entry_from_summary(summary, live_price, timestamp)
            event = {
                "event": "idle",
                "symbol": self.symbol,
                "timestamp_utc": _iso_z(timestamp),
                "summary": dict(summary),
                "balance_after": self.balance,
                "position": None,
            }
            self.events.append(event)
            self.last_snapshot = event
            return event

        exit_reason = self.position.exit_reason(live_price)
        if exit_reason is not None:
            exit_summary = dict(summary)
            exit_summary["exit_reason"] = exit_reason
            return self._exit_from_summary(exit_summary, live_price, timestamp, reason=exit_reason)

        if action == "SELL" and self.exit_on_opposite_signal:
            return self._exit_from_summary(summary, live_price, timestamp, reason="signal_reversal")

        event = {
            "event": "hold",
            "symbol": self.symbol,
            "timestamp_utc": _iso_z(timestamp),
            "summary": dict(summary),
            "balance_after": self.balance,
            "position": self.position.mark_to_market(live_price),
        }
        self.events.append(event)
        self.last_snapshot = event
        return event

    def snapshot(self, live_price: float | None = None) -> dict[str, Any]:
        position = self.position.mark_to_market(live_price) if self.position and live_price is not None else None
        return {
            "symbol": self.symbol,
            "balance": self.balance,
            "starting_balance": self.starting_balance,
            "position": position,
            "open_orders": [order.order_id for order in self.orders if order.status != "FILLED"],
            "orders": [order.order_id for order in self.orders],
            "last_snapshot": self.last_snapshot,
        }


class PaperTradingBot:
    def __init__(
        self,
        symbols: Iterable[Any],
        *,
        summary_provider: Callable[[str], dict[str, Any]],
        gateway: PaperExecutionGateway | None = None,
        starting_balance: float = 1000.0,
        stake_fraction: float = 1.0,
        max_trade_quote_usdt: float = 50.0,
        min_trade_confidence: float = 0.6,
        take_profit_pct: float = 0.5,
        stop_loss_pct: float = 0.25,
        exit_on_opposite_signal: bool = True,
        max_workers: int | None = None,
    ) -> None:
        self.symbols = dedupe_symbols(symbols)
        self.summary_provider = summary_provider
        self.gateway = gateway or PaperExecutionGateway()
        self.starting_balance = float(starting_balance)
        self.stake_fraction = max(0.0, float(stake_fraction))
        self.max_trade_quote_usdt = _positive_float(max_trade_quote_usdt, "max_trade_quote_usdt")
        self.min_trade_confidence = max(0.0, float(min_trade_confidence))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.exit_on_opposite_signal = bool(exit_on_opposite_signal)
        self.max_workers = max(1, int(max_workers or max(1, len(self.symbols) or 1)))
        self.sessions = {
            symbol: PaperTradeSession(
                symbol,
                gateway=self.gateway,
                starting_balance=starting_balance,
                stake_fraction=stake_fraction,
                max_trade_quote_usdt=max_trade_quote_usdt,
                min_trade_confidence=min_trade_confidence,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                exit_on_opposite_signal=exit_on_opposite_signal,
            )
            for symbol in self.symbols
        }

    def refresh_symbol(self, symbol: str) -> dict[str, Any]:
        normalized = normalize_symbol(symbol)
        if normalized not in self.sessions:
            raise ValueError(f"Unknown symbol: {normalized}")
        summary = self.summary_provider(normalized)
        live_price = float(summary.get("live_price") or summary.get("last_close") or 0.0)
        if live_price <= 0:
            raise ValueError(f"Invalid live price for {normalized}.")
        timestamp_value = summary.get("timestamp_utc")
        timestamp = None
        if isinstance(timestamp_value, str) and timestamp_value:
            try:
                timestamp = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
            except ValueError:
                timestamp = None
        return self.sessions[normalized].step(summary, live_price=live_price, timestamp=timestamp)

    def refresh_all(self) -> list[dict[str, Any]]:
        if not self.symbols:
            return []
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.refresh_symbol, symbol): symbol for symbol in self.symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                results[symbol] = future.result()
        return [results[symbol] for symbol in self.symbols if symbol in results]

    def snapshot(self) -> dict[str, Any]:
        sessions: dict[str, dict[str, Any]] = {}
        for symbol, session in self.sessions.items():
            live_price = None
            last_snapshot = session.last_snapshot or {}
            summary = last_snapshot.get("summary") if isinstance(last_snapshot, dict) else None
            if isinstance(summary, dict):
                live_price = summary.get("live_price") or summary.get("last_close")
            sessions[symbol] = session.snapshot(live_price=float(live_price) if live_price is not None else None)

        return {
            "symbols": list(self.symbols),
            "sessions": sessions,
        }
