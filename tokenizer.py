from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from model import Kronos, KronosPredictor, KronosTokenizer


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
DEFAULT_CONTEXT_LEN = 256
DEFAULT_PRED_LEN = 1
DEFAULT_HISTORY_LIMIT = 1000
DEFAULT_POLYMARKET_SLUG = "btc-updown-5m-1779918300"
DEFAULT_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MODEL_NAME = "NeoQuasar/Kronos-small"

DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

BINANCE_KLINES_BASE = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_PRICE_BASE = "https://api.binance.com/api/v3/ticker/price"
POLYMARKET_EVENT_BASE = "https://gamma-api.polymarket.com/events/slug"
POLYMARKET_MARKET_BASE = "https://gamma-api.polymarket.com/markets/slug"


def fetch_json(url: str, headers: Optional[dict[str, str]] = None) -> Any:
    request = Request(url, headers={**DEFAULT_HTTP_HEADERS, **(headers or {})})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def current_polymarket_slug() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    bucket_minute = (now_et.minute // 5) * 5
    start = now_et.replace(minute=bucket_minute, second=0, microsecond=0)
    return f"btc-updown-5m-{int(start.timestamp())}"


def interval_to_timedelta(interval: str) -> timedelta:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"Unsupported Binance interval: {interval}")


def fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    payload = fetch_json(
        f"{BINANCE_KLINES_BASE}?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    )
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Binance klines response: {payload}")
    if not payload:
        raise ValueError(f"No Binance klines returned for {symbol} @ {interval}")

    rows = []
    for candle in payload:
        rows.append(
            {
                "timestamps": pd.to_datetime(candle[0], unit="ms", utc=True),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "amount": float(candle[7]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def fetch_binance_spot_price(symbol: str) -> float:
    payload = fetch_json(f"{BINANCE_TICKER_PRICE_BASE}?symbol={symbol.upper()}")
    price = payload.get("price") if isinstance(payload, dict) else None
    if price is None:
        raise ValueError(f"Unexpected Binance ticker response: {payload}")
    return float(price)


def fetch_polymarket_event(slug: str) -> dict[str, Any]:
    payload = fetch_json(
        f"{POLYMARKET_EVENT_BASE}/{slug}",
        headers={
            "Referer": "https://polymarket.com/",
            "Origin": "https://polymarket.com",
        },
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Polymarket event response: {payload}")
    return payload


def fetch_polymarket_market(slug: str) -> dict[str, Any]:
    payload = fetch_json(
        f"{POLYMARKET_MARKET_BASE}/{slug}",
        headers={
            "Referer": "https://polymarket.com/",
            "Origin": "https://polymarket.com",
        },
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Polymarket market response: {payload}")
    return payload


def parse_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        if not cleaned:
            return []
        return [part.strip().strip('"').strip("'") for part in cleaned.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def parse_string_floats(value: Any) -> list[float]:
    return [float(item) for item in parse_string_list(value)]


def resolve_polymarket_market(slug: str) -> tuple[dict[str, Any], dict[str, Any]]:
    event = fetch_polymarket_event(slug)
    markets = event.get("markets") or []
    market: dict[str, Any] | None = markets[0] if markets else None
    if not market:
        market = fetch_polymarket_market(slug)
    return event, market


def load_predictor() -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
    model = Kronos.from_pretrained(DEFAULT_MODEL_NAME)
    return KronosPredictor(model, tokenizer, max_context=512)


def build_future_timestamps(last_timestamp: pd.Timestamp, interval: str, pred_len: int) -> pd.Series:
    step = interval_to_timedelta(interval)
    return pd.Series([last_timestamp + step * (i + 1) for i in range(pred_len)])


def run_kronos_prediction(
    symbol: str,
    interval: str,
    context_len: int,
    pred_len: int,
    sample_count: int,
    top_k: int,
    top_p: float,
    temperature: float,
    history_limit: int,
    polymarket_slug: Optional[str],
) -> None:
    history = fetch_binance_klines(symbol, interval, history_limit)
    if len(history) < context_len + pred_len:
        raise ValueError(
            f"Not enough history for {symbol} @ {interval}: "
            f"need at least {context_len + pred_len} rows, got {len(history)}"
        )

    context = history.tail(context_len).reset_index(drop=True)
    x_timestamp = context["timestamps"]
    y_timestamp = build_future_timestamps(context["timestamps"].iloc[-1], interval, pred_len)

    last_close = float(context["close"].iloc[-1])
    live_price = fetch_binance_spot_price(symbol)
    live_label = "Binance BTC spot"

    predictor = load_predictor()
    pred_df = predictor.predict(
        df=context[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )

    predicted_close = float(pred_df["close"].iloc[-1])
    direction_vs_live = "UP" if predicted_close >= live_price else "DOWN"
    direction_vs_last_close = "UP" if predicted_close >= last_close else "DOWN"
    live_change = ((predicted_close - live_price) / live_price) * 100.0
    close_change = ((predicted_close - last_close) / last_close) * 100.0

    print(f"Symbol: {symbol.upper()} @ {interval}")
    print(f"{live_label}: {live_price:,.2f}")
    print(f"Last candle close: {last_close:,.2f}")
    print(f"Predicted next close: {predicted_close:,.2f}")
    print(f"Direction vs live: {direction_vs_live}")
    print(f"Direction vs last close: {direction_vs_last_close}")
    print(f"Move vs live: {live_change:+.2f}%")
    print(f"Move vs last close: {close_change:+.2f}%")

    if polymarket_slug:
        event, market = resolve_polymarket_market(polymarket_slug)
        title = (
            market.get("title")
            or market.get("question")
            or event.get("title")
            or event.get("question")
            or polymarket_slug
        )
        resolved_slug = market.get("slug") or event.get("slug") or polymarket_slug
        outcomes = parse_string_list(market.get("outcomes"))
        outcome_prices = parse_string_floats(market.get("outcomePrices"))

        print()
        print(f"Polymarket event: {title}")
        print(f"Slug: {resolved_slug}")

        if len(outcomes) >= 2 and len(outcome_prices) >= 2:
            up_label = outcomes[0]
            down_label = outcomes[1]
            up_price = float(outcome_prices[0])
            down_price = float(outcome_prices[1])
            market_bias = up_label if up_price >= down_price else down_label
            kronos_bias = up_label if direction_vs_last_close == "UP" else down_label
            matches = "YES" if kronos_bias == market_bias else "NO"
            current_price = float(market.get("lastTradePrice") or up_price)

            print(f"Polymarket bias: {market_bias}")
            print(f"Polymarket current price: {current_price:.4f}")
            print(f"{up_label} / {down_label}: {up_price:.2%} / {down_price:.2%}")
            print(f"Kronos bias: {kronos_bias}")
            print(f"Kronos matches market bias: {matches}")

    print()
    print(pred_df.head())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use the Kronos tokenizer to predict the next BTC candle direction."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help="Binance symbol to fetch, e.g. BTCUSDT",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        help="Binance kline interval, e.g. 5m or 1h",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=DEFAULT_CONTEXT_LEN,
        help="Number of most recent candles to feed into Kronos",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=DEFAULT_PRED_LEN,
        help="How many future candles to predict",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="How many stochastic samples Kronos should average",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling for Kronos decoding",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling for Kronos decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for Kronos decoding",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help="How many recent Binance candles to fetch before slicing context",
    )
    parser.add_argument(
        "--polymarket-slug",
        type=str,
        default=None,
        help="Optional Polymarket BTC up/down market slug to compare against",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Compare against the currently active 5-minute Polymarket market",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    polymarket_slug = current_polymarket_slug() if args.current else args.polymarket_slug
    run_kronos_prediction(
        symbol=args.symbol,
        interval=args.interval,
        context_len=args.context_len,
        pred_len=args.pred_len,
        sample_count=args.sample_count,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        history_limit=args.history_limit,
        polymarket_slug=polymarket_slug,
    )


if __name__ == "__main__":
    main()
