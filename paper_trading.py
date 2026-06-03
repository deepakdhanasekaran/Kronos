from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional
from uuid import uuid4

from dashboard_support import dedupe_symbols, normalize_symbol


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


class PaperTradeSession:
    def __init__(
        self,
        symbol: str,
        *,
        gateway: PaperExecutionGateway | None = None,
        starting_balance: float = 1000.0,
        stake_fraction: float = 1.0,
        min_trade_confidence: float = 0.6,
    ) -> None:
        self.symbol = normalize_symbol(symbol)
        self.gateway = gateway or PaperExecutionGateway()
        self.starting_balance = float(starting_balance)
        self.balance = float(starting_balance)
        self.stake_fraction = max(0.0, float(stake_fraction))
        self.min_trade_confidence = max(0.0, float(min_trade_confidence))
        self.position: PaperPosition | None = None
        self.orders: list[PaperOrder] = []
        self.events: list[dict[str, Any]] = []
        self.last_snapshot: dict[str, Any] | None = None

    def _stake_quote(self) -> float:
        return max(0.0, self.balance * self.stake_fraction)

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
        self.position = PaperPosition(
            symbol=self.symbol,
            quantity=order.executed_quantity,
            entry_price=order.executed_price,
            entry_order_id=order.order_id,
            opened_at=order.filled_at,
            invested_quote=stake_quote,
            entry_fee=order.fee_paid,
            entry_slippage_pct=order.slippage_pct,
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

    def _exit_from_summary(self, summary: dict[str, Any], live_price: float, timestamp: datetime | None) -> dict[str, Any]:
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

        if action == "SELL":
            return self._exit_from_summary(summary, live_price, timestamp)

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
        min_trade_confidence: float = 0.6,
        max_workers: int | None = None,
    ) -> None:
        self.symbols = dedupe_symbols(symbols)
        self.summary_provider = summary_provider
        self.gateway = gateway or PaperExecutionGateway()
        self.starting_balance = float(starting_balance)
        self.stake_fraction = max(0.0, float(stake_fraction))
        self.min_trade_confidence = max(0.0, float(min_trade_confidence))
        self.max_workers = max(1, int(max_workers or max(1, len(self.symbols) or 1)))
        self.sessions = {
            symbol: PaperTradeSession(
                symbol,
                gateway=self.gateway,
                starting_balance=starting_balance,
                stake_fraction=stake_fraction,
                min_trade_confidence=min_trade_confidence,
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
        return {
            "symbols": list(self.symbols),
            "sessions": {symbol: session.snapshot() for symbol, session in self.sessions.items()},
        }
