from __future__ import annotations

import asyncio
import argparse
import json
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

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
POLYMARKET_EVENTS_LIST_BASE = "https://gamma-api.polymarket.com/events"
LIVE_SERVER_DEFAULT_HOST = "127.0.0.1"
LIVE_SERVER_DEFAULT_PORT = 8765

DEFAULT_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MODEL_NAME = "NeoQuasar/Kronos-small"
MODEL_NAME_BY_SIZE = {
    "small": "NeoQuasar/Kronos-small",
    "base": "NeoQuasar/Kronos-base",
}


def fetch_json(url: str, headers: Optional[dict[str, str]] = None) -> Any:
    request = Request(url, headers={**DEFAULT_HTTP_HEADERS, **(headers or {})})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_json_post(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None) -> Any:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**DEFAULT_HTTP_HEADERS, "Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def interval_to_timedelta(interval: str) -> timedelta:
    if not interval or len(interval) < 2:
        raise ValueError(f"Unsupported interval: {interval}")

    unit = interval[-1]
    try:
        value = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported interval: {interval}") from exc

    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"Unsupported interval: {interval}")


def build_future_timestamps(last_timestamp: pd.Timestamp, interval: str, pred_len: int) -> pd.Series:
    step = interval_to_timedelta(interval)
    return pd.Series([last_timestamp + step * (i + 1) for i in range(pred_len)])


def current_polymarket_slug() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    bucket_minute = (now_et.minute // 5) * 5
    start = now_et.replace(minute=bucket_minute, second=0, microsecond=0)
    return f"btc-updown-5m-{int(start.timestamp())}"


def _parse_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _market_end_timestamp(payload: dict[str, Any]) -> Optional[pd.Timestamp]:
    if not isinstance(payload, dict):
        return None

    for key in ("endDate", "endDateIso", "closeTime", "close_time", "closeAt", "close_at", "expiresAt", "expires_at"):
        end_dt = _parse_datetime(payload.get(key))
        if end_dt is not None:
            return end_dt

    return None


def market_is_closed(payload: dict[str, Any], now: Optional[pd.Timestamp] = None) -> bool:
    if not isinstance(payload, dict):
        return False

    if payload.get("closed", False) or payload.get("resolved", False) or payload.get("resolvedAt"):
        return True

    end_dt = _market_end_timestamp(payload)
    if end_dt is None:
        return False

    now_ts = now or pd.Timestamp.now(tz="UTC")
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    return now_ts >= end_dt


def resolve_current_polymarket_slug() -> str:
    payload = fetch_json(
        f"{POLYMARKET_EVENTS_LIST_BASE}?active=true&closed=false&limit=100&order=end_date&ascending=true"
    )
    data = json.loads(payload)
    if isinstance(data, list) and data:
        now_utc = pd.Timestamp.now(tz="UTC")
        candidates: list[tuple[pd.Timestamp, dict[str, Any]]] = []
        for event in data:
            if not isinstance(event, dict):
                continue
            title = str(event.get("title") or "")
            slug = str(event.get("slug") or "")
            end_dt = _parse_datetime(event.get("endDate") or event.get("endDateIso"))
            if "btc up or down 5m" not in title.lower() and not slug.startswith("btc-updown-5m-"):
                continue
            if end_dt is None:
                continue
            candidates.append((abs(end_dt - now_utc), event))

        if candidates:
            candidates.sort(key=lambda item: (item[0], str(item[1].get("slug") or "")))
            return str(candidates[0][1].get("slug"))

    return current_polymarket_slug()


def prepare_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    required_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required OHLC columns: {required_cols}")

    prepared = df.copy()
    if "volume" not in prepared.columns:
        prepared["volume"] = 0.0
        prepared["amount"] = 0.0
    elif "amount" not in prepared.columns:
        prepared["amount"] = prepared["volume"] * prepared[required_cols].mean(axis=1)

    if "amount" not in prepared.columns:
        prepared["amount"] = 0.0

    ohlcv_cols = required_cols + ["volume", "amount"]
    if prepared[ohlcv_cols].isnull().values.any():
        raise ValueError("Input DataFrame contains NaN values in OHLCV columns.")

    return prepared[ohlcv_cols].copy()


def compute_direction_summary(predicted_close: float, last_close: float, live_price: float) -> dict[str, Any]:
    if last_close == 0 or live_price == 0:
        raise ValueError("Price values must be non-zero.")

    direction_vs_last_close = "UP" if predicted_close >= last_close else "DOWN"
    direction_vs_live = "UP" if predicted_close >= live_price else "DOWN"

    return {
        "predicted_close": float(predicted_close),
        "last_close": float(last_close),
        "live_price": float(live_price),
        "direction_vs_last_close": direction_vs_last_close,
        "direction_vs_live": direction_vs_live,
        "close_change_pct": ((predicted_close - last_close) / last_close) * 100.0,
        "live_change_pct": ((predicted_close - live_price) / live_price) * 100.0,
    }


def compute_technical_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_period: int = 14,
    momentum_period: int = 3,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must include a 'close' column.")
    if ema_fast <= 0 or ema_slow <= 0 or rsi_period <= 0 or momentum_period <= 0:
        raise ValueError("Indicator periods must be positive.")

    frame = df.copy()
    close = frame["close"].astype(float)

    frame["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    frame["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()
    frame["ema_diff_pct"] = ((frame["ema_fast"] - frame["ema_slow"]) / close) * 100.0

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    frame["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    frame["rsi"] = frame["rsi"].fillna(100.0)

    frame["momentum_pct"] = (close / close.shift(momentum_period) - 1.0) * 100.0
    frame["momentum_pct"] = frame["momentum_pct"].fillna(0.0)

    return frame


def indicator_trade_signal(
    row: pd.Series,
    overbought: float = 70.0,
    oversold: float = 30.0,
    min_score: int = 2,
) -> dict[str, Any]:
    if min_score <= 0:
        raise ValueError("min_score must be positive.")

    score = 0
    reasons: list[str] = []

    ema_fast = float(row.get("ema_fast", np.nan))
    ema_slow = float(row.get("ema_slow", np.nan))
    rsi = float(row.get("rsi", np.nan))
    momentum_pct = float(row.get("momentum_pct", np.nan))

    if np.isnan(ema_fast) or np.isnan(ema_slow):
        ema_signal = 0
    elif ema_fast > ema_slow:
        ema_signal = 1
        reasons.append("EMA fast above slow")
    elif ema_fast < ema_slow:
        ema_signal = -1
        reasons.append("EMA fast below slow")
    else:
        ema_signal = 0

    if np.isnan(rsi):
        rsi_signal = 0
    elif rsi <= oversold:
        rsi_signal = 1
        reasons.append("RSI oversold")
    elif rsi >= overbought:
        rsi_signal = -1
        reasons.append("RSI overbought")
    else:
        rsi_signal = 0

    if np.isnan(momentum_pct):
        momentum_signal = 0
    elif momentum_pct > 0:
        momentum_signal = 1
        reasons.append("Positive momentum")
    elif momentum_pct < 0:
        momentum_signal = -1
        reasons.append("Negative momentum")
    else:
        momentum_signal = 0

    score = ema_signal + rsi_signal + momentum_signal
    if score >= min_score:
        signal = "BUY"
    elif score <= -min_score:
        signal = "SELL"
    else:
        signal = "NO_TRADE"

    return {
        "signal": signal,
        "action": signal,
        "score": score,
        "reasons": reasons,
        "ema_signal": ema_signal,
        "rsi_signal": rsi_signal,
        "momentum_signal": momentum_signal,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi": rsi,
        "momentum_pct": momentum_pct,
    }


def evaluate_trade_signal(
    reference_close: float,
    predicted_signal: str,
    actual_close: float,
    neutral_threshold_pct: float = 0.2,
    stake_usd: Optional[float] = None,
) -> dict[str, Any]:
    if reference_close <= 0:
        raise ValueError("reference_close must be positive.")
    if actual_close <= 0:
        raise ValueError("actual_close must be positive.")

    actual_change_pct = ((float(actual_close) - float(reference_close)) / float(reference_close)) * 100.0
    actual_signal = signal_from_change_pct(actual_change_pct, neutral_threshold_pct=neutral_threshold_pct)
    predicted_signal = str(predicted_signal)
    if predicted_signal not in {"UP", "DOWN", "NEUTRAL", "BUY", "SELL", "NO_TRADE"}:
        raise ValueError(f"Unsupported predicted_signal: {predicted_signal}")

    normalized_signal = predicted_signal
    if normalized_signal == "BUY":
        normalized_signal = "UP"
    elif normalized_signal == "SELL":
        normalized_signal = "DOWN"
    elif normalized_signal == "NO_TRADE":
        normalized_signal = "NEUTRAL"

    predicted_action = action_from_signal(normalized_signal)
    actual_action = action_from_signal(actual_signal)

    if predicted_action == "BUY":
        strategy_return_pct = actual_change_pct
    elif predicted_action == "SELL":
        strategy_return_pct = -actual_change_pct
    else:
        strategy_return_pct = 0.0

    trade_pnl_usd = None
    if stake_usd is not None:
        if stake_usd < 0:
            raise ValueError("stake_usd must be non-negative.")
        trade_pnl_usd = float(stake_usd) * (strategy_return_pct / 100.0)

    return {
        "reference_close": float(reference_close),
        "actual_close": float(actual_close),
        "actual_change_pct": actual_change_pct,
        "actual_signal": actual_signal,
        "actual_action": actual_action,
        "predicted_signal": normalized_signal,
        "predicted_action": predicted_action,
        "match_signal": normalized_signal == actual_signal,
        "match_action": predicted_action == actual_action,
        "strategy_return_pct": strategy_return_pct,
        "trade_pnl_usd": trade_pnl_usd,
        "neutral_threshold_pct": neutral_threshold_pct,
    }


def combine_kronos_and_indicator_signals(
    kronos_summary: dict[str, Any],
    indicator_signal: dict[str, Any],
    require_agreement: bool = True,
) -> dict[str, Any]:
    kronos_action = str(kronos_summary.get("action", "NO_TRADE"))
    if kronos_action not in {"BUY", "SELL", "NO_TRADE"}:
        verdict = str(kronos_summary.get("verdict", "NO_TRADE"))
        if verdict in {"Strong BUY", "BUY"}:
            kronos_action = "BUY"
        elif verdict in {"Strong SELL", "SELL"}:
            kronos_action = "SELL"
        else:
            kronos_action = "NO_TRADE"

    indicator_action = str(indicator_signal.get("action", "NO_TRADE"))
    if indicator_action not in {"BUY", "SELL", "NO_TRADE"}:
        indicator_action = "NO_TRADE"

    agreement = kronos_action == indicator_action and kronos_action != "NO_TRADE"
    if require_agreement and agreement:
        combined_action = kronos_action
    else:
        combined_action = "NO_TRADE"

    reasons = []
    if kronos_action != "NO_TRADE":
        reasons.append(f"Kronos={kronos_action}")
    if indicator_action != "NO_TRADE":
        reasons.append(f"Indicator={indicator_action}")
    if agreement:
        reasons.append("Agreement")

    return {
        "signal": "UP" if combined_action == "BUY" else "DOWN" if combined_action == "SELL" else "NEUTRAL",
        "action": combined_action,
        "verdict": verdict_from_summary({"action": combined_action, "trade_confidence": kronos_summary.get("trade_confidence", 0.0)}),
        "kronos_action": kronos_action,
        "indicator_action": indicator_action,
        "agreement": agreement,
        "reasons": reasons,
    }


def signal_from_change_pct(change_pct: float, neutral_threshold_pct: float = 0.2) -> str:
    if neutral_threshold_pct < 0:
        raise ValueError("neutral_threshold_pct must be non-negative.")
    if abs(change_pct) < neutral_threshold_pct:
        return "NEUTRAL"
    return "UP" if change_pct >= 0 else "DOWN"


def action_from_signal(signal: str) -> str:
    mapping = {
        "UP": "BUY",
        "DOWN": "SELL",
        "NEUTRAL": "NO_TRADE",
    }
    if signal not in mapping:
        raise ValueError(f"Unsupported signal: {signal}")
    return mapping[signal]


def verdict_from_summary(summary: dict[str, Any], strong_threshold: float = 0.6) -> str:
    action = str(summary.get("action", "NO_TRADE"))
    trade_confidence = float(summary.get("trade_confidence", summary.get("confidence", 0.0)))

    if action == "NO_TRADE":
        return "NO_TRADE"
    if action == "BUY":
        return "Strong BUY" if trade_confidence >= strong_threshold else "BUY"
    if action == "SELL":
        return "Strong SELL" if trade_confidence >= strong_threshold else "SELL"
    return "NO_TRADE"


def evaluate_binance_prediction(
    summary: dict[str, Any],
    actual_close: float,
    neutral_threshold_pct: Optional[float] = None,
    stake_usd: Optional[float] = None,
) -> dict[str, Any]:
    if "last_close" not in summary:
        raise ValueError("summary must include last_close.")
    threshold = (
        float(neutral_threshold_pct)
        if neutral_threshold_pct is not None
        else float(summary.get("neutral_threshold_pct", 0.2))
    )
    evaluation = evaluate_trade_signal(
        reference_close=float(summary["last_close"]),
        predicted_signal=str(summary.get("signal", "NEUTRAL")),
        actual_close=actual_close,
        neutral_threshold_pct=threshold,
        stake_usd=stake_usd,
    )
    evaluation["predicted_action"] = str(summary.get("action", evaluation["predicted_action"]))
    evaluation["match_action"] = evaluation["predicted_action"] == evaluation["actual_action"]
    return evaluation


def polymarket_action_from_probability(
    market_signal: str,
    market_probability: float,
    edge_threshold_pct: float = 0.05,
) -> str:
    if not 0.0 <= market_probability <= 1.0:
        raise ValueError("market_probability must be between 0 and 1.")
    if edge_threshold_pct < 0:
        raise ValueError("edge_threshold_pct must be non-negative.")
    if market_signal not in {"UP", "DOWN"}:
        raise ValueError("market_signal must be either 'UP' or 'DOWN'.")

    if market_probability < 0.5 - edge_threshold_pct or market_probability > 0.5 + edge_threshold_pct:
        return "BUY" if market_signal == "UP" else "SELL"
    return "NO_TRADE"


def score_sampled_directions(directions: list[str]) -> dict[str, Any]:
    if not directions:
        raise ValueError("directions must not be empty.")

    counts: dict[str, int] = {}
    for direction in directions:
        counts[direction] = counts.get(direction, 0) + 1

    dominant_signal = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    confidence = counts[dominant_signal] / len(directions)
    trade_confidence = confidence if dominant_signal != "NEUTRAL" else 0.0
    return {
        "sample_count": len(directions),
        "dominant_signal": dominant_signal,
        "confidence": confidence,
        "trade_confidence": trade_confidence,
        "counts": counts,
    }


def parse_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = json.loads(cleaned.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [part.strip().strip('"').strip("'") for part in cleaned.strip("[]").split(",") if part.strip()]
    return []


def parse_string_floats(value: Any) -> list[float]:
    return [float(item) for item in parse_string_list(value)]


def fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    from urllib.parse import urlencode

    query = urlencode({"symbol": symbol.upper(), "interval": interval, "limit": limit})
    payload = fetch_json(f"{BINANCE_KLINES_BASE}?{query}")
    data = json.loads(payload)
    if not isinstance(data, list) or not data:
        raise ValueError(f"No Binance klines returned for {symbol} @ {interval}")

    rows = []
    for candle in data:
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

    return pd.DataFrame(rows).sort_values("timestamps").reset_index(drop=True)


def fetch_binance_spot_price(symbol: str) -> float:
    from urllib.parse import urlencode

    query = urlencode({"symbol": symbol.upper()})
    payload = fetch_json(f"{BINANCE_TICKER_PRICE_BASE}?{query}")
    data = json.loads(payload)
    price = data.get("price") if isinstance(data, dict) else None
    if price is None:
        raise ValueError(f"Unexpected Binance ticker response: {data}")
    return float(price)


def load_pretrained_predictor(
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    max_context: int = 512,
):
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


async def load_pretrained_predictor_async(
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    max_context: int = 512,
):
    return await asyncio.to_thread(
        load_pretrained_predictor,
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        device=device,
        max_context=max_context,
    )


async def fetch_binance_klines_async(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_binance_klines, symbol, interval, limit)


async def fetch_binance_spot_price_async(symbol: str) -> float:
    return await asyncio.to_thread(fetch_binance_spot_price, symbol)


class LivePredictionService:
    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_context: int = 512,
    ) -> None:
        self.predictor = load_pretrained_predictor(
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            device=device,
            max_context=max_context,
        )
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.max_context = max_context
        self._lock = threading.Lock()

    def predict(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            result = asyncio.run(
                predict_binance_direction_with_predictor_async(
                    predictor=self.predictor,
                    **kwargs,
                )
            )

        prediction = result["prediction"].copy()
        summary = dict(result["summary"])
        samples = [dict(sample) for sample in result["samples"]]
        return {
            "prediction": prediction.to_dict(orient="records"),
            "summary": summary,
            "samples": samples,
            "model_info": {
                "model_name": self.model_name,
                "tokenizer_name": self.tokenizer_name,
                "device": self.device,
                "max_context": self.max_context,
            },
        }


class LivePredictionRequestHandler(BaseHTTPRequestHandler):
    server_version = "KronosLive/1.0"

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        if not raw_body:
            return {}
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            service: LivePredictionService = self.server.live_service  # type: ignore[attr-defined]
            self._write_json(
                200,
                {
                    "ok": True,
                    "model_loaded": True,
                    "model_info": {
                        "model_name": service.model_name,
                        "tokenizer_name": service.tokenizer_name,
                        "device": service.device,
                        "max_context": service.max_context,
                    },
                },
            )
            return

        self._write_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/predict":
            self._write_json(404, {"error": "Not found"})
            return

        try:
            payload = self._read_json_body()
            service: LivePredictionService = self.server.live_service  # type: ignore[attr-defined]
            result = service.predict(
                symbol=str(payload.get("symbol", "BTCUSDT")),
                interval=str(payload.get("interval", "5m")),
                lookback=int(payload.get("lookback", 256)),
                pred_len=int(payload.get("pred_len", 1)),
                sample_count=int(payload.get("sample_count", 5)),
                top_k=int(payload.get("top_k", 0)),
                top_p=float(payload.get("top_p", 0.9)),
                temperature=float(payload.get("temperature", 1.0)),
                neutral_threshold_pct=float(payload.get("neutral_threshold_pct", 0.2)),
                confidence_samples=int(payload.get("confidence_samples", 5)),
            )
            self._write_json(200, {"success": True, **result})
        except Exception as exc:
            self._write_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def run_live_prediction_server(
    host: str = LIVE_SERVER_DEFAULT_HOST,
    port: int = LIVE_SERVER_DEFAULT_PORT,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    max_context: int = 512,
) -> None:
    service = LivePredictionService(
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        device=device,
        max_context=max_context,
    )
    httpd = ThreadingHTTPServer((host, port), LivePredictionRequestHandler)
    httpd.live_service = service  # type: ignore[attr-defined]
    print(
        f"Kronos live server running on http://{host}:{port} "
        f"({model_name}, tokenizer={tokenizer_name})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def request_live_prediction(live_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = live_url.rstrip("/") + "/predict"
    response = fetch_json_post(url, payload)
    data = json.loads(response)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected live prediction response: {data}")
    if data.get("success") is not True:
        raise ValueError(data.get("error") or "Live prediction failed.")
    return data


def predict_binance_direction_from_live_server(
    live_url: str,
    symbol: str,
    interval: str,
    lookback: int,
    pred_len: int = 1,
    sample_count: int = 5,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 1.0,
    neutral_threshold_pct: float = 0.2,
    confidence_samples: int = 5,
) -> dict[str, Any]:
    payload = {
        "symbol": symbol,
        "interval": interval,
        "lookback": lookback,
        "pred_len": pred_len,
        "sample_count": sample_count,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "neutral_threshold_pct": neutral_threshold_pct,
        "confidence_samples": confidence_samples,
    }
    data = request_live_prediction(live_url, payload)
    prediction = pd.DataFrame(data.get("prediction", []))
    summary = dict(data.get("summary", {}))
    samples = list(data.get("samples", []))
    return {"prediction": prediction, "summary": summary, "samples": samples, "model_info": data.get("model_info", {})}


def fetch_polymarket_event_by_slug(slug: str) -> dict[str, Any]:
    payload = fetch_json(f"{POLYMARKET_EVENT_BASE}/{slug}")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected Polymarket event response: {data}")
    return data


def fetch_polymarket_market_by_slug(slug: str) -> dict[str, Any]:
    payload = fetch_json(f"{POLYMARKET_MARKET_BASE}/{slug}")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected Polymarket market response: {data}")
    return data


def resolve_polymarket_market(slug: str) -> tuple[dict[str, Any], dict[str, Any]]:
    market = fetch_polymarket_market_by_slug(slug)
    event: dict[str, Any] = {}
    try:
        event = fetch_polymarket_event_by_slug(slug)
    except Exception:
        event = {}
    if not market.get("outcomes") or not market.get("outcomePrices"):
        markets = event.get("markets") or []
        if markets and isinstance(markets[0], dict):
            fallback_market = markets[0]
            if fallback_market.get("outcomes") and fallback_market.get("outcomePrices"):
                market = fallback_market
    return event, market


def predict_direction_from_prediction_frame(pred_df: pd.DataFrame, context_df: pd.DataFrame, live_price: float) -> dict[str, Any]:
    if "close" not in pred_df.columns:
        raise ValueError("Prediction frame must include a 'close' column.")
    if "close" not in context_df.columns:
        raise ValueError("Context frame must include a 'close' column.")

    predicted_close = float(pred_df["close"].iloc[-1])
    last_close = float(context_df["close"].iloc[-1])
    return compute_direction_summary(predicted_close, last_close, live_price)


def build_sample_summary(
    predictor: Any,
    context: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    top_k: int,
    top_p: float,
    temperature: float,
    live_price: float,
    neutral_threshold_pct: float,
) -> dict[str, Any]:
    sampled_pred_df = predictor.predict(
        df=context[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_count=1,
        verbose=False,
    )
    sample_summary = predict_direction_from_prediction_frame(sampled_pred_df, context, live_price)
    sample_summary["signal"] = signal_from_change_pct(sample_summary["close_change_pct"], neutral_threshold_pct)
    sample_summary["action"] = action_from_signal(sample_summary["signal"])
    return sample_summary


def polymarket_direction_summary(outcomes: list[str], outcome_prices: list[float]) -> dict[str, Any]:
    if len(outcomes) < 2 or len(outcome_prices) < 2:
        raise ValueError("Polymarket markets need at least two outcomes and prices.")

    up_label, down_label = outcomes[:2]
    up_price, down_price = float(outcome_prices[0]), float(outcome_prices[1])
    market_bias = up_label if up_price >= down_price else down_label
    market_probability = max(up_price, down_price)

    return {
        "up_label": up_label,
        "down_label": down_label,
        "up_price": up_price,
        "down_price": down_price,
        "market_bias": market_bias,
        "market_probability": market_probability,
        "market_signal": "UP" if market_bias == up_label else "DOWN",
    }


def build_polymarket_summary(
    slug: str,
    edge_threshold_pct: float = 0.05,
) -> dict[str, Any]:
    event, market = resolve_polymarket_market(slug)
    title = (
        market.get("title")
        or market.get("question")
        or event.get("title")
        or event.get("question")
        or slug
    )
    outcomes = parse_string_list(market.get("outcomes"))
    outcome_prices = parse_string_floats(market.get("outcomePrices"))
    market_summary = polymarket_direction_summary(outcomes, outcome_prices)
    return {
        "slug": market.get("slug") or event.get("slug") or slug,
        "title": title,
        **market_summary,
        "action": polymarket_action_from_probability(
            market_summary["market_signal"],
            market_summary["market_probability"],
            edge_threshold_pct=edge_threshold_pct,
        ),
    }


def extract_polymarket_final_outcome(market: dict[str, Any]) -> Optional[str]:
    if not isinstance(market, dict):
        return None

    direct_keys = ("resolvedOutcome", "winningOutcome", "outcome", "result")
    for key in direct_keys:
        value = market.get(key)
        if isinstance(value, str) and value.strip():
            normalized = value.strip().upper()
            if normalized in {"UP", "DOWN", "YES", "NO"}:
                return "UP" if normalized in {"UP", "YES"} else "DOWN"
            return value.strip()

    tokens = market.get("tokens")
    if isinstance(tokens, list):
        for token in tokens:
            if not isinstance(token, dict):
                continue
            if token.get("winner") is True or str(token.get("status", "")).lower() == "winner":
                token_outcome = token.get("outcome") or token.get("name") or token.get("title")
                if isinstance(token_outcome, str) and token_outcome.strip():
                    normalized = token_outcome.strip().upper()
                    if normalized in {"UP", "YES"}:
                        return "UP"
                    if normalized in {"DOWN", "NO"}:
                        return "DOWN"
                    return token_outcome.strip()

    return None


def compare_with_polymarket(
    summary: dict[str, Any],
    slug: str,
    edge_threshold_pct: float = 0.05,
) -> dict[str, Any]:
    market_summary = build_polymarket_summary(slug, edge_threshold_pct=edge_threshold_pct)
    kronos_signal = summary.get("signal", "NEUTRAL")
    market_signal = market_summary["market_signal"]
    return {
        **market_summary,
        "kronos_signal": kronos_signal,
        "kronos_action": summary.get("action", "NO_TRADE"),
        "agreement": kronos_signal in {"UP", "DOWN"} and kronos_signal == market_signal,
    }


def build_polymarket_audit_record(
    slug: str,
    edge_threshold_pct: float = 0.05,
    poll_seconds: int = 15,
    max_wait_seconds: int = 300,
) -> dict[str, Any]:
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative.")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be non-negative.")

    summary = build_polymarket_summary(slug, edge_threshold_pct=edge_threshold_pct)
    event, market = resolve_polymarket_market(slug)

    def closed_and_outcome(payload: dict[str, Any], fallback: Optional[dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        closed = market_is_closed(payload) or (market_is_closed(fallback) if fallback else False)
        outcome = extract_polymarket_final_outcome(payload) or (extract_polymarket_final_outcome(fallback) if fallback else None)
        return closed, outcome

    initial_closed, initial_outcome = closed_and_outcome(market, event)
    closed = initial_closed
    final_outcome = initial_outcome
    waited_seconds = 0

    while not closed and waited_seconds < max_wait_seconds:
        sleep_for = min(poll_seconds, max_wait_seconds - waited_seconds)
        if sleep_for <= 0:
            break
        time.sleep(sleep_for)
        waited_seconds += sleep_for
        event, market = resolve_polymarket_market(slug)
        closed, final_outcome = closed_and_outcome(market, event)

    return {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "slug": summary["slug"],
        "title": summary["title"],
        "prediction": summary["action"],
        "market_signal": summary["market_signal"],
        "market_probability": summary["market_probability"],
        "initial_closed": initial_closed,
        "initial_outcome": initial_outcome or "pending",
        "closed": closed,
        "final_outcome": final_outcome or "pending",
        "matched": bool(final_outcome and summary["market_signal"] == final_outcome),
        "waited_seconds": waited_seconds,
    }


def predict_binance_direction_with_predictor(
    predictor: Any,
    symbol: str,
    interval: str,
    lookback: int,
    pred_len: int = 1,
    sample_count: int = 5,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 1.0,
    neutral_threshold_pct: float = 0.2,
    confidence_samples: int = 5,
):
    history = fetch_binance_klines(symbol, interval, max(lookback + pred_len, lookback))
    if len(history) < lookback + pred_len:
        raise ValueError(
            f"Not enough Binance history for {symbol} @ {interval}: "
            f"need at least {lookback + pred_len} rows, got {len(history)}"
        )

    context = history.tail(lookback).reset_index(drop=True)
    x_timestamp = context["timestamps"]
    y_timestamp = build_future_timestamps(context["timestamps"].iloc[-1], interval, pred_len)

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

    live_price = fetch_binance_spot_price(symbol)
    summary = predict_direction_from_prediction_frame(pred_df, context, live_price)
    summary["signal"] = signal_from_change_pct(summary["close_change_pct"], neutral_threshold_pct)
    summary["action"] = action_from_signal(summary["signal"])

    sample_summaries = [summary]
    if confidence_samples > 1:
        sample_summaries = [
            build_sample_summary(
                predictor,
                context,
                x_timestamp,
                y_timestamp,
                pred_len,
                top_k,
                top_p,
                temperature,
                live_price,
                neutral_threshold_pct,
            )
            for _ in range(confidence_samples)
        ]

    direction_score = score_sampled_directions([item["signal"] for item in sample_summaries])
    summary["confidence"] = direction_score["confidence"]
    summary["trade_confidence"] = direction_score["trade_confidence"]
    summary["confidence_samples"] = direction_score["sample_count"]
    summary["dominant_signal"] = direction_score["dominant_signal"]
    summary["signal_counts"] = direction_score["counts"]
    summary["neutral_threshold_pct"] = neutral_threshold_pct
    summary["verdict"] = verdict_from_summary(summary)

    return {"prediction": pred_df, "summary": summary, "samples": sample_summaries}


async def predict_binance_direction_with_predictor_async(
    predictor: Any,
    symbol: str,
    interval: str,
    lookback: int,
    pred_len: int = 1,
    sample_count: int = 5,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 1.0,
    neutral_threshold_pct: float = 0.2,
    confidence_samples: int = 5,
    history: Optional[pd.DataFrame] = None,
    live_price: Optional[float] = None,
):
    if history is None and live_price is None:
        history, live_price = await asyncio.gather(
            fetch_binance_klines_async(symbol, interval, max(lookback + pred_len, lookback)),
            fetch_binance_spot_price_async(symbol),
        )
    else:
        if history is None:
            history = await fetch_binance_klines_async(symbol, interval, max(lookback + pred_len, lookback))
        if live_price is None:
            live_price = await fetch_binance_spot_price_async(symbol)

    if len(history) < lookback + pred_len:
        raise ValueError(
            f"Not enough Binance history for {symbol} @ {interval}: "
            f"need at least {lookback + pred_len} rows, got {len(history)}"
        )

    context = history.tail(lookback).reset_index(drop=True)
    x_timestamp = context["timestamps"]
    y_timestamp = build_future_timestamps(context["timestamps"].iloc[-1], interval, pred_len)

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

    summary = predict_direction_from_prediction_frame(pred_df, context, live_price)
    summary["signal"] = signal_from_change_pct(summary["close_change_pct"], neutral_threshold_pct)
    summary["action"] = action_from_signal(summary["signal"])

    sample_summaries = [summary]
    if confidence_samples > 1:
        sample_summaries = await asyncio.gather(
            *[
                asyncio.to_thread(
                    build_sample_summary,
                    predictor,
                    context,
                    x_timestamp,
                    y_timestamp,
                    pred_len,
                    top_k,
                    top_p,
                    temperature,
                    live_price,
                    neutral_threshold_pct,
                )
                for _ in range(confidence_samples)
            ]
        )

    direction_score = score_sampled_directions([item["signal"] for item in sample_summaries])
    summary["confidence"] = direction_score["confidence"]
    summary["trade_confidence"] = direction_score["trade_confidence"]
    summary["confidence_samples"] = direction_score["sample_count"]
    summary["dominant_signal"] = direction_score["dominant_signal"]
    summary["signal_counts"] = direction_score["counts"]
    summary["neutral_threshold_pct"] = neutral_threshold_pct
    summary["verdict"] = verdict_from_summary(summary)

    return {"prediction": pred_df, "summary": summary, "samples": sample_summaries}


def predict_binance_direction(
    symbol: str,
    interval: str,
    lookback: int,
    pred_len: int = 1,
    sample_count: int = 5,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 1.0,
    neutral_threshold_pct: float = 0.2,
    confidence_samples: int = 5,
    device: Optional[str] = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    max_context: int = 512,
):
    return asyncio.run(
        predict_binance_direction_async(
            symbol=symbol,
            interval=interval,
            lookback=lookback,
            pred_len=pred_len,
            sample_count=sample_count,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            neutral_threshold_pct=neutral_threshold_pct,
            confidence_samples=confidence_samples,
            device=device,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_context=max_context,
        )
    )


async def predict_binance_direction_async(
    symbol: str,
    interval: str,
    lookback: int,
    pred_len: int = 1,
    sample_count: int = 5,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 1.0,
    neutral_threshold_pct: float = 0.2,
    confidence_samples: int = 5,
    device: Optional[str] = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    max_context: int = 512,
):
    predictor_task = asyncio.create_task(
        load_pretrained_predictor_async(
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            device=device,
            max_context=max_context,
        )
    )
    history_task = asyncio.create_task(
        fetch_binance_klines_async(symbol, interval, max(lookback + pred_len, lookback))
    )
    live_price_task = asyncio.create_task(fetch_binance_spot_price_async(symbol))

    predictor, history, live_price = await asyncio.gather(predictor_task, history_task, live_price_task)
    return await predict_binance_direction_with_predictor_async(
        predictor=predictor,
        symbol=symbol,
        interval=interval,
        lookback=lookback,
        pred_len=pred_len,
        sample_count=sample_count,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        neutral_threshold_pct=neutral_threshold_pct,
        confidence_samples=confidence_samples,
        history=history,
        live_price=live_price,
    )


def render_cli_output(
    symbol: str,
    interval: str,
    prediction: pd.DataFrame,
    summary: dict[str, Any],
    polymarket: Optional[dict[str, Any]] = None,
    action_only: Optional[str] = None,
) -> str:
    if action_only is not None:
        return action_only

    lines = [
        f"Symbol: {symbol}",
        f"Interval: {interval}",
        f"Predicted close: {summary['predicted_close']:.4f}",
        f"Last close: {summary['last_close']:.4f}",
        f"Live price: {summary['live_price']:.4f}",
        f"Signal: {summary.get('signal', summary['direction_vs_last_close'])}",
        f"Action: {summary.get('action', action_from_signal(summary.get('signal', summary['direction_vs_last_close'])))}",
        f"Verdict: {summary.get('verdict', 'NO_TRADE')}",
        f"Direction vs last close: {summary['direction_vs_last_close']}",
        f"Direction vs live: {summary['direction_vs_live']}",
        f"Close change pct: {summary['close_change_pct']:.2f}%",
        f"Live change pct: {summary['live_change_pct']:.2f}%",
        f"Agreement: {summary.get('confidence', 1.0):.2f}",
        f"Trade confidence: {summary.get('trade_confidence', summary.get('confidence', 0.0)):.2f}",
        f"Confidence samples: {summary.get('confidence_samples', 1)}",
    ]
    if "signal_counts" in summary:
        counts = summary["signal_counts"]
        lines.append(f"Signal counts: UP={counts.get('UP', 0)} DOWN={counts.get('DOWN', 0)} NEUTRAL={counts.get('NEUTRAL', 0)}")
    if polymarket is not None:
        lines.extend(
            [
                "",
                "Polymarket comparison:",
                f"Title: {polymarket['title']}",
                f"Slug: {polymarket['slug']}",
                f"Market probability: {polymarket['market_probability']:.2%}",
                f"Market signal: {polymarket['market_signal']}",
                f"Action: {polymarket['action']}",
                f"Kronos signal: {polymarket['kronos_signal']}",
                f"Agreement: {'YES' if polymarket['agreement'] else 'NO'}",
            ]
        )
    lines.extend(
        [
            "",
            "Prediction head:",
            prediction.head().to_string(),
        ]
    )
    return "\n".join(lines)


def load_jsonl_records(log_file: str | Path) -> list[dict[str, Any]]:
    path = Path(log_file)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc.msg}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _format_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not headers:
        return ""

    rendered_rows = [[_format_table_value(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def extract_overnight_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prediction = record.get("prediction")
        actual = record.get("actual")
        evaluation = record.get("evaluation")
        stats = record.get("stats") or {}

        if not isinstance(prediction, dict) or not isinstance(actual, dict) or not isinstance(evaluation, dict):
            continue

        rows.append(
            {
                "cycle": record.get("cycle", index),
                "timestamp": record.get("timestamp_utc", ""),
                "verdict": prediction.get("verdict", ""),
                "signal": prediction.get("signal", ""),
                "action": prediction.get("action", ""),
                "predicted_close": prediction.get("predicted_close"),
                "actual_close": actual.get("close"),
                "actual_signal": actual.get("signal", ""),
                "actual_action": actual.get("action", ""),
                "match_signal": evaluation.get("match_signal", False),
                "match_action": evaluation.get("match_action", False),
                "strategy_return_pct": evaluation.get("strategy_return_pct", 0.0),
                "trade_pnl_usd": evaluation.get("trade_pnl_usd", evaluation.get("pnl_usd", 0.0)),
                "balance_usd": stats.get("balance_usd"),
                "starting_balance": stats.get("starting_balance"),
            }
        )
    return rows


def summarize_overnight_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cycles = len(rows)
    trade_cycles = sum(1 for row in rows if row.get("action") != "NO_TRADE")
    signal_hits = sum(1 for row in rows if row.get("match_signal"))
    action_hits = sum(1 for row in rows if row.get("match_action"))
    trade_wins = sum(1 for row in rows if float(row.get("trade_pnl_usd") or 0.0) > 0)
    trade_return_sum_pct = sum(float(row.get("strategy_return_pct") or 0.0) for row in rows)
    starting_balance = 0.0
    if rows:
        starting_balance = float(
            rows[0].get("starting_balance")
            if rows[0].get("starting_balance") is not None
            else float(rows[0].get("balance_usd") or 0.0) - float(rows[0].get("trade_pnl_usd") or 0.0)
        )
    ending_balance = float(rows[-1].get("balance_usd") or starting_balance) if rows else starting_balance

    return {
        "cycles": cycles,
        "trade_cycles": trade_cycles,
        "signal_hits": signal_hits,
        "action_hits": action_hits,
        "trade_wins": trade_wins,
        "trade_return_sum_pct": trade_return_sum_pct,
        "starting_balance": starting_balance,
        "ending_balance": ending_balance,
        "realized_pnl_usd": ending_balance - starting_balance,
    }


def render_overnight_audit_report(records: list[dict[str, Any]], limit: Optional[int] = None) -> str:
    rows = extract_overnight_audit_rows(records)
    if not rows:
        return "No completed overnight audit rows were found in the log."

    display_rows = rows[-limit:] if limit and limit > 0 else rows
    summary = summarize_overnight_audit(rows)

    table_rows = [
        [
            row["cycle"],
            row["timestamp"],
            row["verdict"],
            row["signal"],
            row["action"],
            row["predicted_close"],
            row["actual_close"],
            row["actual_signal"],
            row["actual_action"],
            row["match_signal"],
            row["trade_pnl_usd"],
            row["balance_usd"],
        ]
        for row in display_rows
    ]

    lines = [
        "Overnight backtest report:",
        f"Cycles: {summary['cycles']}",
        f"Trade cycles: {summary['trade_cycles']}",
        f"Signal hit rate: {(summary['signal_hits'] / summary['cycles']) if summary['cycles'] else 0.0:.2%}",
        f"Action hit rate: {(summary['action_hits'] / summary['trade_cycles']) if summary['trade_cycles'] else 0.0:.2%}",
        f"Trade win rate: {(summary['trade_wins'] / summary['trade_cycles']) if summary['trade_cycles'] else 0.0:.2%}",
        f"Average trade return: {(summary['trade_return_sum_pct'] / summary['trade_cycles']) if summary['trade_cycles'] else 0.0:.2f}%",
        f"Total trade return: {summary['trade_return_sum_pct']:.2f}%",
        f"Starting balance: ${summary['starting_balance']:.2f}",
        f"Ending balance: ${summary['ending_balance']:.2f}",
        f"Realized PnL: ${summary['realized_pnl_usd']:.2f}",
        "",
        render_markdown_table(
            [
                "Cycle",
                "Timestamp",
                "Verdict",
                "Signal",
                "Action",
                "Pred Close",
                "Actual Close",
                "Actual Signal",
                "Actual Action",
                "Match",
                "PnL USD",
                "Balance USD",
            ],
            table_rows,
        ),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict crypto candle direction with pretrained Kronos."
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default="5m", help="Binance interval, e.g. 5m, 15m, 1h")
    parser.add_argument("--lookback", type=int, default=256, help="Number of candles to use as context")
    parser.add_argument("--pred-len", type=int, default=1, help="Number of future candles to predict")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of stochastic samples to average")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling cutoff")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--neutral-threshold-pct",
        type=float,
        default=0.2,
        help="Treat tiny moves within this percent as NEUTRAL",
    )
    parser.add_argument(
        "--confidence-samples",
        type=int,
        default=5,
        help="How many single-sample forecasts to use for confidence scoring",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_TOKENIZER_NAME,
        help="Hugging Face tokenizer id or local path",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(MODEL_NAME_BY_SIZE.keys()),
        default="small",
        help="Shortcut for Kronos model size; overrides --model-name when set.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=512,
        help="Maximum context length for the model",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run a warm local prediction server instead of a one-shot prediction.",
    )
    parser.add_argument(
        "--live-url",
        type=str,
        default=None,
        help="Use a running live prediction server at this base URL instead of loading the model locally.",
    )
    parser.add_argument("--host", type=str, default=LIVE_SERVER_DEFAULT_HOST, help="Host for --serve mode.")
    parser.add_argument("--port", type=int, default=LIVE_SERVER_DEFAULT_PORT, help="Port for --serve mode.")
    parser.add_argument(
        "--polymarket-slug",
        type=str,
        default=None,
        help="Optional Polymarket slug to compare against",
    )
    parser.add_argument(
        "--polymarket-edge-pct",
        type=float,
        default=0.05,
        help="Probability edge around 50%% required to emit a Polymarket action",
    )
    parser.add_argument(
        "--polymarket-only",
        action="store_true",
        help="Print only the Polymarket BUY/SELL/NO_TRADE action",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Use the current BTC up/down Polymarket market slug",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_name = MODEL_NAME_BY_SIZE.get(args.model_size, args.model_name) if args.model_size else args.model_name

    if args.serve:
        run_live_prediction_server(
            host=args.host,
            port=args.port,
            tokenizer_name=args.tokenizer_name,
            model_name=model_name,
            device=args.device,
            max_context=args.max_context,
        )
        return 0

    polymarket_slug = resolve_current_polymarket_slug() if args.current else args.polymarket_slug

    if args.polymarket_only:
        polymarket_summary = build_polymarket_summary(
            polymarket_slug or resolve_current_polymarket_slug(),
            edge_threshold_pct=args.polymarket_edge_pct,
        )
        print(
            render_cli_output(
                args.symbol,
                args.interval,
                prediction=pd.DataFrame(),
                summary={},
                polymarket=None,
                action_only=polymarket_summary["action"],
            )
        )
        return 0

    if args.live_url:
        result = predict_binance_direction_from_live_server(
            live_url=args.live_url,
            symbol=args.symbol,
            interval=args.interval,
            lookback=args.lookback,
            pred_len=args.pred_len,
            sample_count=args.sample_count,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            neutral_threshold_pct=args.neutral_threshold_pct,
            confidence_samples=args.confidence_samples,
        )
    else:
        result = predict_binance_direction(
            symbol=args.symbol,
            interval=args.interval,
            lookback=args.lookback,
            pred_len=args.pred_len,
            sample_count=args.sample_count,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            neutral_threshold_pct=args.neutral_threshold_pct,
            confidence_samples=args.confidence_samples,
            device=args.device,
            tokenizer_name=args.tokenizer_name,
            model_name=model_name,
            max_context=args.max_context,
        )

    polymarket = (
        compare_with_polymarket(
            result["summary"],
            polymarket_slug,
            edge_threshold_pct=args.polymarket_edge_pct,
        )
        if polymarket_slug
        else None
    )

    print(
        render_cli_output(
            args.symbol,
            args.interval,
            result["prediction"],
            result["summary"],
            polymarket,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
