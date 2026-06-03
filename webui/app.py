from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dashboard_support import (
    DEFAULT_DASHBOARD_INTERVAL,
    DEFAULT_DASHBOARD_LOOKBACK,
    DEFAULT_DASHBOARD_PRED_LEN,
    DEFAULT_REFRESH_SECONDS,
    DEFAULT_TOP_SYMBOL_LIMIT,
    WatchlistStore,
    dashboard_placeholder_row,
    dedupe_symbols,
    merge_symbol_lists,
    normalize_symbol,
    validate_usdt_symbol,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _backend_request_json(
    backend_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    url = backend_url.rstrip("/") + path
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request_obj = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request_obj, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else exc.reason
        raise RuntimeError(f"Backend request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Backend unavailable: {exc.reason}") from exc


def _load_custom_symbols(store: WatchlistStore) -> list[str]:
    try:
        return store.load()
    except ValueError:
        return []


@dataclass(slots=True)
class DashboardCache:
    payload: dict[str, Any] | None = None
    rows_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    refreshing: bool = False
    last_error: str | None = None
    last_refreshed_at: str | None = None
    selected_top_symbols: set[str] | None = None


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)

    defaults = {
        "BACKEND_URL": os.environ.get("KRONOS_BACKEND_URL", "http://127.0.0.1:8765"),
        "WATCHLIST_PATH": os.environ.get("DASHBOARD_WATCHLIST_PATH", "/data/custom_coins.json"),
        "DASHBOARD_INTERVAL": os.environ.get("DASHBOARD_INTERVAL", DEFAULT_DASHBOARD_INTERVAL),
        "DASHBOARD_LOOKBACK": _env_int("DASHBOARD_LOOKBACK", DEFAULT_DASHBOARD_LOOKBACK),
        "DASHBOARD_PRED_LEN": _env_int("DASHBOARD_PRED_LEN", DEFAULT_DASHBOARD_PRED_LEN),
        "DASHBOARD_REFRESH_SECONDS": _env_int("DASHBOARD_REFRESH_SECONDS", DEFAULT_REFRESH_SECONDS),
        "DASHBOARD_SAMPLE_COUNT": _env_int("DASHBOARD_SAMPLE_COUNT", 5),
        "DASHBOARD_CONFIDENCE_SAMPLES": _env_int("DASHBOARD_CONFIDENCE_SAMPLES", 1),
        "DASHBOARD_TOP_K": _env_int("DASHBOARD_TOP_K", 0),
        "DASHBOARD_TOP_P": float(os.environ.get("DASHBOARD_TOP_P", "0.9")),
        "DASHBOARD_TEMPERATURE": float(os.environ.get("DASHBOARD_TEMPERATURE", "1.0")),
        "DASHBOARD_NEUTRAL_THRESHOLD_PCT": float(os.environ.get("DASHBOARD_NEUTRAL_THRESHOLD_PCT", "0.05")),
        "DASHBOARD_TOP_SYMBOL_LIMIT": _env_int("DASHBOARD_TOP_SYMBOL_LIMIT", DEFAULT_TOP_SYMBOL_LIMIT),
        "DASHBOARD_BACKGROUND_REFRESH": os.environ.get("DASHBOARD_BACKGROUND_REFRESH", "1") not in {"0", "false", "False"},
    }
    app.config.from_mapping(defaults)
    if config:
        app.config.update(config)

    watchlist_store = WatchlistStore(app.config["WATCHLIST_PATH"])
    app.extensions["watchlist_store"] = watchlist_store

    snapshot_lock = threading.Lock()
    snapshot_state: dict[str, Any] = {
        "data": None,
        "refreshing": False,
        "last_error": None,
        "last_refreshed_at": None,
    }
    refresh_event = threading.Event()
    stop_event = threading.Event()

    def get_backend_health() -> dict[str, Any]:
        return _backend_request_json(app.config["BACKEND_URL"], "/health")

    def get_top_symbols() -> list[dict[str, Any]]:
        response = _backend_request_json(
            app.config["BACKEND_URL"],
            f"/top-symbols?{urlencode({'limit': app.config['DASHBOARD_TOP_SYMBOL_LIMIT']})}",
        )
        items = response.get("items", [])
        return [item for item in items if isinstance(item, dict)]

    def build_dashboard_payload(selected_top_symbols: list[str] | None = None) -> dict[str, Any]:
        top_items = get_top_symbols()
        top_symbols = [normalize_symbol(item.get("symbol")) for item in top_items]
        custom_symbols = _load_custom_symbols(watchlist_store)
        if selected_top_symbols is None:
            active_top_symbols = top_symbols
            annotated_top_items = [dict(item, selected=True) for item in top_items]
        else:
            selected_set = set(dedupe_symbols(selected_top_symbols))
            active_top_symbols = [symbol for symbol in top_symbols if symbol in selected_set]
            annotated_top_items = []
            for item in top_items:
                symbol = normalize_symbol(item.get("symbol"))
                annotated = dict(item)
                annotated["selected"] = symbol in selected_set
                annotated_top_items.append(annotated)
        merged_symbols = merge_symbol_lists(active_top_symbols, custom_symbols)

        if merged_symbols:
            prediction_response = _backend_request_json(
                app.config["BACKEND_URL"],
                "/predict/batch",
                method="POST",
                payload={
                    "symbols": merged_symbols,
                    "interval": app.config["DASHBOARD_INTERVAL"],
                    "lookback": app.config["DASHBOARD_LOOKBACK"],
                    "pred_len": app.config["DASHBOARD_PRED_LEN"],
                    "sample_count": app.config["DASHBOARD_SAMPLE_COUNT"],
                    "top_k": app.config["DASHBOARD_TOP_K"],
                    "top_p": app.config["DASHBOARD_TOP_P"],
                    "temperature": app.config["DASHBOARD_TEMPERATURE"],
                    "neutral_threshold_pct": app.config["DASHBOARD_NEUTRAL_THRESHOLD_PCT"],
                    "confidence_samples": app.config["DASHBOARD_CONFIDENCE_SAMPLES"],
                },
            )
        else:
            prediction_response = {"items": [], "model_info": {}, "cached": False}

        return {
            "generated_at": _utc_timestamp(),
            "backend": get_backend_health(),
            "top_symbols": annotated_top_items,
            "custom_symbols": custom_symbols,
            "symbols": merged_symbols,
            "items": prediction_response.get("items", []),
            "selected_top_symbols": active_top_symbols,
            "model_info": prediction_response.get("model_info", {}),
            "cached": bool(prediction_response.get("cached", False)),
            "refresh_seconds": app.config["DASHBOARD_REFRESH_SECONDS"],
            "interval": app.config["DASHBOARD_INTERVAL"],
            "lookback": app.config["DASHBOARD_LOOKBACK"],
            "pred_len": app.config["DASHBOARD_PRED_LEN"],
        }

    def build_placeholder_payload() -> dict[str, Any]:
        return {
            "generated_at": _utc_timestamp(),
            "backend": {"ok": False, "status": "warming"},
            "top_symbols": [],
            "custom_symbols": [],
            "symbols": [],
            "items": [],
            "selected_top_symbols": [],
            "model_info": {},
            "cached": False,
            "refresh_seconds": app.config["DASHBOARD_REFRESH_SECONDS"],
            "interval": app.config["DASHBOARD_INTERVAL"],
            "lookback": app.config["DASHBOARD_LOOKBACK"],
            "pred_len": app.config["DASHBOARD_PRED_LEN"],
        }

    def top_source_label() -> str:
        return f"Top {app.config['DASHBOARD_TOP_SYMBOL_LIMIT']}"

    class CoinSessionManager:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._cache = DashboardCache(payload=build_placeholder_payload())
            self._prime_roster()
            threading.Thread(target=self._refresh_loop, daemon=True).start()

        def set_selected_top_symbols(self, symbols: list[str] | None) -> None:
            with self._lock:
                self._cache.selected_top_symbols = None if symbols is None else set(dedupe_symbols(symbols))
            self.request_refresh()

        def _refresh_once(self) -> None:
            with self._lock:
                if self._cache.refreshing:
                    return
                self._cache.refreshing = True
                selected_top_symbols = (
                    None if self._cache.selected_top_symbols is None else list(self._cache.selected_top_symbols)
                )

            try:
                payload = build_dashboard_payload(selected_top_symbols)
                rows_by_symbol = {
                    normalize_symbol(item.get("symbol")): dict(item)
                    for item in payload.get("items", [])
                    if isinstance(item, dict) and normalize_symbol(item.get("symbol"))
                }
                with self._lock:
                    self._cache.payload = payload
                    self._cache.rows_by_symbol.update(rows_by_symbol)
                    self._cache.last_error = None
                    self._cache.last_refreshed_at = payload.get("generated_at", _utc_timestamp())
            except Exception as exc:
                with self._lock:
                    self._cache.last_error = str(exc)
                    if self._cache.payload is None:
                        self._cache.payload = build_placeholder_payload()
            finally:
                with self._lock:
                    self._cache.refreshing = False

        def _prime_roster(self) -> None:
            self._refresh_once()

        def request_refresh(self) -> None:
            threading.Thread(target=self._refresh_once, daemon=True).start()

        def _refresh_loop(self) -> None:
            interval_seconds = max(1, int(app.config["DASHBOARD_REFRESH_SECONDS"]))
            while not self._stop_event.wait(interval_seconds):
                self._refresh_once()

        def snapshot(self) -> dict[str, Any]:
            with self._lock:
                cache = self._cache
                cached_payload = dict(cache.payload or build_placeholder_payload())
                rows_by_symbol = {symbol: dict(row) for symbol, row in cache.rows_by_symbol.items()}
                selected_top_symbols = None if cache.selected_top_symbols is None else set(cache.selected_top_symbols)
                refreshing = cache.refreshing
                last_error = cache.last_error
                last_refreshed_at = cache.last_refreshed_at or cached_payload.get("last_refreshed_at")

            top_symbols = [dict(item) for item in cached_payload.get("top_symbols", []) if isinstance(item, dict)]
            custom_symbols = [normalize_symbol(symbol) for symbol in cached_payload.get("custom_symbols", []) if normalize_symbol(symbol)]

            if selected_top_symbols is None:
                active_top_items = [dict(item, selected=bool(item.get("selected", True))) for item in top_symbols]
                active_top_symbols = [normalize_symbol(item.get("symbol")) for item in active_top_items if item.get("selected", True)]
            else:
                active_top_items = []
                active_top_symbols = []
                for item in top_symbols:
                    symbol = normalize_symbol(item.get("symbol"))
                    if not symbol:
                        continue
                    annotated = dict(item)
                    annotated["selected"] = symbol in selected_top_symbols
                    active_top_items.append(annotated)
                    if annotated["selected"]:
                        active_top_symbols.append(symbol)

            symbols = merge_symbol_lists(active_top_symbols, custom_symbols)
            meta_by_symbol: dict[str, dict[str, Any]] = {}
            for item in active_top_items:
                symbol = normalize_symbol(item.get("symbol"))
                if not symbol:
                    continue
                meta_by_symbol[symbol] = {
                    "source": item.get("source", top_source_label()),
                    "rank": int(item.get("rank", 0) or 0) or None,
                }

            items: list[dict[str, Any]] = []
            for symbol in symbols:
                row = rows_by_symbol.get(symbol)
                if row is not None:
                    item = dict(row)
                    item["refreshing"] = refreshing
                    item["last_error"] = last_error
                else:
                    meta = meta_by_symbol.get(symbol, {"source": "Custom", "rank": None})
                    item = dashboard_placeholder_row(symbol, source=str(meta.get("source", "Custom")), rank=meta.get("rank"))
                    item["refreshing"] = refreshing
                    item["last_error"] = last_error
                items.append(item)

            cached = any(bool(item.get("cached")) for item in items if item.get("ready"))
            model_info = cached_payload.get("model_info", {})
            if not isinstance(model_info, dict):
                model_info = {}
            payload = {
                "generated_at": _utc_timestamp(),
                "backend": dict(cached_payload.get("backend", {"ok": False, "status": "warming"})),
                "top_symbols": active_top_items,
                "custom_symbols": custom_symbols,
                "symbols": symbols,
                "items": items,
                "selected_top_symbols": active_top_symbols,
                "model_info": dict(model_info),
                "cached": cached,
                "refresh_seconds": app.config["DASHBOARD_REFRESH_SECONDS"],
                "interval": app.config["DASHBOARD_INTERVAL"],
                "lookback": app.config["DASHBOARD_LOOKBACK"],
                "pred_len": app.config["DASHBOARD_PRED_LEN"],
                "refreshing": refreshing,
                "last_error": last_error,
                "last_refreshed_at": last_refreshed_at,
            }
            return payload

    dashboard_manager: CoinSessionManager | None = None
    if app.config["DASHBOARD_BACKGROUND_REFRESH"]:
        dashboard_manager = CoinSessionManager()
        app.extensions["dashboard_manager"] = dashboard_manager

    def refresh_snapshot() -> None:
        try:
            payload = build_dashboard_payload()
            with snapshot_lock:
                snapshot_state["data"] = payload
                snapshot_state["last_error"] = None
                snapshot_state["last_refreshed_at"] = payload["generated_at"]
        except Exception as exc:
            with snapshot_lock:
                snapshot_state["last_error"] = str(exc)
        finally:
            with snapshot_lock:
                snapshot_state["refreshing"] = False

    def trigger_refresh() -> bool:
        with snapshot_lock:
            if snapshot_state["refreshing"]:
                return False
            snapshot_state["refreshing"] = True

        threading.Thread(target=refresh_snapshot, daemon=True).start()
        return True

    def refresh_loop() -> None:
        trigger_refresh()
        while not stop_event.is_set():
            refresh_event.wait(app.config["DASHBOARD_REFRESH_SECONDS"])
            refresh_event.clear()
            if stop_event.is_set():
                break
            trigger_refresh()

    if app.config["DASHBOARD_BACKGROUND_REFRESH"] and dashboard_manager is None:
        threading.Thread(target=refresh_loop, daemon=True).start()

    @app.route("/")
    def index() -> str:
        if dashboard_manager is not None:
            initial_dashboard = dashboard_manager.snapshot()
        else:
            with snapshot_lock:
                initial_dashboard = dict(snapshot_state["data"] or build_placeholder_payload())
                initial_dashboard["refreshing"] = bool(snapshot_state["refreshing"])
                initial_dashboard["last_error"] = snapshot_state["last_error"]
                initial_dashboard["last_refreshed_at"] = snapshot_state["last_refreshed_at"]
        initial_dashboard.setdefault("selected_top_symbols", initial_dashboard.get("selected_top_symbols", []))
        return render_template(
            "index.html",
            backend_url=app.config["BACKEND_URL"],
            refresh_seconds=app.config["DASHBOARD_REFRESH_SECONDS"],
            interval=app.config["DASHBOARD_INTERVAL"],
            watchlist_limit=app.config["DASHBOARD_TOP_SYMBOL_LIMIT"],
            top_source_label=top_source_label(),
            initial_dashboard=initial_dashboard,
        )

    @app.route("/api/health")
    def health() -> Any:
        try:
            if dashboard_manager is not None:
                backend = dashboard_manager.snapshot()["backend"]
            else:
                backend = get_backend_health()
            return jsonify(
                {
                    "ok": True,
                    "backend": backend,
                    "refresh_seconds": app.config["DASHBOARD_REFRESH_SECONDS"],
                }
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503

    @app.route("/api/watchlist", methods=["GET", "POST"])
    def watchlist() -> Any:
        if request.method == "GET":
            return jsonify({"symbols": _load_custom_symbols(watchlist_store)})

        payload = request.get_json(silent=True) or {}
        raw_symbols = payload.get("symbols")
        if isinstance(raw_symbols, list) and raw_symbols:
            updated = watchlist_store.load()
            for raw_symbol in raw_symbols:
                updated = watchlist_store.add(raw_symbol)
            return jsonify({"symbols": updated})

        symbol = payload.get("symbol")
        if not symbol:
            return jsonify({"error": "symbol is required"}), 400

        try:
            validated = validate_usdt_symbol(symbol)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        updated = watchlist_store.add(validated)
        if dashboard_manager is not None:
            dashboard_manager.request_refresh()
        else:
            refresh_event.set()
            trigger_refresh()
        return jsonify({"symbols": updated, "symbol": validated})

    @app.route("/api/watchlist/<symbol>", methods=["DELETE"])
    def delete_watchlist_symbol(symbol: str) -> Any:
        try:
            validated = validate_usdt_symbol(symbol)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        updated = watchlist_store.remove(validated)
        if dashboard_manager is not None:
            dashboard_manager.request_refresh()
        else:
            refresh_event.set()
            trigger_refresh()
        return jsonify({"symbols": updated, "removed": validated})

    @app.route("/api/top-symbols")
    def top_symbols() -> Any:
        try:
            if dashboard_manager is not None:
                return jsonify({"items": dashboard_manager.snapshot()["top_symbols"]})
            return jsonify({"items": get_top_symbols()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 503

    @app.route("/api/dashboard")
    def dashboard() -> Any:
        try:
            selected_top_symbols_raw = request.args.get("selected_top_symbols")
            selected_top_symbols = None
            if selected_top_symbols_raw is not None:
                selected_top_symbols = dedupe_symbols(
                    [symbol for symbol in selected_top_symbols_raw.split(",") if symbol.strip()]
                )
                if dashboard_manager is not None:
                    dashboard_manager.set_selected_top_symbols(selected_top_symbols)
            elif dashboard_manager is not None:
                selected_top_symbols = None
            if dashboard_manager is not None:
                payload = dashboard_manager.snapshot()
                if selected_top_symbols is not None:
                    payload["selected_top_symbols"] = selected_top_symbols
                return jsonify(payload)
            with snapshot_lock:
                data_payload = dict(snapshot_state["data"] or {})
                refreshing = bool(snapshot_state["refreshing"])
                last_error = snapshot_state["last_error"]
                last_refreshed_at = snapshot_state["last_refreshed_at"]
            if not data_payload and not app.config["DASHBOARD_BACKGROUND_REFRESH"]:
                data_payload = build_dashboard_payload(selected_top_symbols)
                with snapshot_lock:
                    snapshot_state["data"] = data_payload
                    snapshot_state["last_error"] = None
                    snapshot_state["last_refreshed_at"] = data_payload["generated_at"]
                refreshing = False
                last_error = None
                last_refreshed_at = data_payload["generated_at"]
            if not data_payload:
                trigger_refresh()
                placeholder = build_placeholder_payload()
                placeholder["refreshing"] = True
                placeholder["last_error"] = last_error
                placeholder["last_refreshed_at"] = last_refreshed_at
                return jsonify(placeholder)
            payload = dict(data_payload)
            payload["refreshing"] = refreshing
            payload["last_error"] = last_error
            payload["last_refreshed_at"] = last_refreshed_at
            return jsonify(payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 503

    return app


app = create_app()


if __name__ == "__main__":
    port = _env_int("PORT", 7070)
    app.run(host="0.0.0.0", port=port, debug=False)
