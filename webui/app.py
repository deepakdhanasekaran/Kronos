from __future__ import annotations

import json
import os
import sys
import threading
import time
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
    dashboard_row_from_summary,
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
            self._refresh_event = threading.Event()
            self._backend_health: dict[str, Any] = {"ok": False, "status": "warming"}
            self._top_symbols: list[dict[str, Any]] = []
            self._custom_symbols: list[str] = []
            self._symbol_meta: dict[str, dict[str, Any]] = {}
            self._sessions: dict[str, dict[str, Any]] = {}
            self._selected_top_symbols: set[str] | None = None
            self._last_error: str | None = None
            self._last_refreshed_at: str | None = None
            self._roster_refreshing = False
            self._prime_roster()
            threading.Thread(target=self._refresh_loop, daemon=True).start()

        def set_selected_top_symbols(self, symbols: list[str] | None) -> None:
            if symbols is None:
                selected: set[str] | None = None
            else:
                selected = set(dedupe_symbols(symbols))
            with self._lock:
                self._selected_top_symbols = selected
            self.request_refresh()

        def _sync_session_meta(self, symbol: str, *, source: str, rank: int | None) -> dict[str, Any]:
            session = self._sessions.setdefault(
                symbol,
                {
                    "data": None,
                    "refreshing": False,
                    "last_error": None,
                    "last_refreshed_at": None,
                    "cached": False,
                    "worker_running": False,
                    "worker_stop": threading.Event(),
                    "wake_event": threading.Event(),
                },
            )
            session["source"] = source
            session["rank"] = rank
            self._symbol_meta[symbol] = {"source": source, "rank": rank}
            return session

        def _stop_symbol_worker(self, symbol: str) -> None:
            with self._lock:
                session = self._sessions.get(symbol)
                if not session:
                    return
                stop_event = session.get("worker_stop")
                wake_event = session.get("wake_event")
                session["worker_running"] = False
            if isinstance(stop_event, threading.Event):
                stop_event.set()
            if isinstance(wake_event, threading.Event):
                wake_event.set()

        def _wake_symbol_worker(self, symbol: str) -> None:
            with self._lock:
                session = self._sessions.get(symbol)
                wake_event = session.get("wake_event") if session else None
            if isinstance(wake_event, threading.Event):
                wake_event.set()

        def _fetch_prediction(self, symbol: str) -> dict[str, Any]:
            response = _backend_request_json(
                app.config["BACKEND_URL"],
                "/predict",
                method="POST",
                payload={
                    "symbol": symbol,
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
            summary = response.get("summary")
            if not isinstance(summary, dict):
                raise ValueError(f"Unexpected prediction response for {symbol}")
            return {
                "summary": summary,
                "model_info": response.get("model_info", {}),
                "cached": bool(response.get("cached", False)),
            }

        def _refresh_symbol(self, symbol: str) -> None:
            with self._lock:
                session = self._sessions.get(symbol)
                if session is None:
                    return
                meta = {"source": session.get("source", top_source_label()), "rank": session.get("rank")}

            try:
                prediction = self._fetch_prediction(symbol)
                row = dashboard_row_from_summary(symbol, prediction["summary"], source=meta["source"], rank=meta["rank"])
                row["ready"] = True
                row["cached"] = prediction["cached"]
                row["refreshing"] = False
                row["status"] = "ready"
                with self._lock:
                    session = self._sessions.setdefault(symbol, {})
                    session["data"] = row
                    session["last_error"] = None
                    session["last_refreshed_at"] = _utc_timestamp()
                    session["cached"] = prediction["cached"]
                    session["refreshing"] = False
                    session["model_info"] = prediction["model_info"]
                    self._last_refreshed_at = session["last_refreshed_at"]
            except Exception as exc:
                with self._lock:
                    session = self._sessions.setdefault(symbol, {})
                    session["last_error"] = str(exc)
                    session["refreshing"] = False
                    self._last_error = str(exc)

        def _symbol_worker(self, symbol: str) -> None:
            while not self._stop_event.is_set():
                with self._lock:
                    session = self._sessions.get(symbol)
                    if session is None:
                        return
                    stop_event = session.get("worker_stop")
                    wake_event = session.get("wake_event")
                    session["worker_running"] = True
                    session["refreshing"] = True
                if isinstance(stop_event, threading.Event) and stop_event.is_set():
                    break

                self._refresh_symbol(symbol)

                with self._lock:
                    session = self._sessions.get(symbol)
                    if session is None:
                        return
                    session["refreshing"] = False

                interval_seconds = max(1, int(app.config["DASHBOARD_REFRESH_SECONDS"]))
                if isinstance(wake_event, threading.Event) and wake_event.wait(interval_seconds):
                    wake_event.clear()
                if isinstance(stop_event, threading.Event) and stop_event.is_set():
                    break

            with self._lock:
                session = self._sessions.get(symbol)
                if session is not None:
                    session["worker_running"] = False

        def _ensure_symbol_worker(self, symbol: str) -> None:
            with self._lock:
                session = self._sessions.get(symbol)
                if session is None:
                    return
                if session.get("worker_running"):
                    return
                session["worker_running"] = True
                session["worker_stop"] = threading.Event()
                session["wake_event"] = threading.Event()
            threading.Thread(target=self._symbol_worker, args=(symbol,), daemon=True).start()

        def _refresh_roster(self) -> None:
            with self._lock:
                if self._roster_refreshing:
                    return
                self._roster_refreshing = True

            try:
                backend_health = get_backend_health()
                top_items = get_top_symbols()
                custom_symbols = _load_custom_symbols(watchlist_store)
                top_symbols = [normalize_symbol(item.get("symbol")) for item in top_items]
                selected_top_symbols = self._selected_top_symbols
                if selected_top_symbols is None:
                    active_top_items = [dict(item, selected=True) for item in top_items]
                    active_top_symbols = top_symbols
                else:
                    active_top_items = []
                    active_top_symbols = []
                    for item in top_items:
                        symbol = normalize_symbol(item.get("symbol"))
                        selected = symbol in selected_top_symbols
                        annotated = dict(item)
                        annotated["selected"] = selected
                        active_top_items.append(annotated)
                        if selected:
                            active_top_symbols.append(symbol)

                merged_symbols = merge_symbol_lists(active_top_symbols, custom_symbols)

                with self._lock:
                    self._backend_health = backend_health
                    self._top_symbols = active_top_items
                    self._custom_symbols = custom_symbols
                    self._symbol_meta = {}
                    for item in active_top_items:
                        symbol = normalize_symbol(item.get("symbol"))
                        if not symbol:
                            continue
                        if item.get("selected"):
                            self._sync_session_meta(symbol, source=top_source_label(), rank=int(item.get("rank", 0) or 0) or None)
                        else:
                            self._symbol_meta[symbol] = {"source": top_source_label(), "rank": int(item.get("rank", 0) or 0) or None}
                    for symbol in custom_symbols:
                        if symbol in self._symbol_meta:
                            continue
                        self._sync_session_meta(symbol, source="Custom", rank=None)
                    for symbol in merged_symbols:
                        self._sessions.setdefault(
                            symbol,
                            {
                                "data": None,
                                "refreshing": False,
                                "last_error": None,
                                "last_refreshed_at": None,
                                "cached": False,
                                "worker_running": False,
                                "worker_stop": threading.Event(),
                                "wake_event": threading.Event(),
                                "source": self._symbol_meta.get(symbol, {}).get("source", top_source_label()),
                                "rank": self._symbol_meta.get(symbol, {}).get("rank"),
                            },
                        )
                    stale_symbols = [symbol for symbol in list(self._sessions.keys()) if symbol not in merged_symbols]
                    self._last_error = None
                    self._last_refreshed_at = _utc_timestamp()
                    schedule_symbols = list(merged_symbols)
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._backend_health = {"ok": False, "status": "warning", "error": str(exc)}
                    schedule_symbols = []
                    stale_symbols = []
            finally:
                with self._lock:
                    self._roster_refreshing = False

            for symbol in stale_symbols:
                self._stop_symbol_worker(symbol)
                with self._lock:
                    self._sessions.pop(symbol, None)
                    self._symbol_meta.pop(symbol, None)

            for symbol in schedule_symbols:
                self._ensure_symbol_worker(symbol)

        def _prime_roster(self) -> None:
            self._refresh_roster()

        def request_refresh(self) -> None:
            self._refresh_event.set()
            threading.Thread(target=self._refresh_roster, daemon=True).start()

        def _refresh_loop(self) -> None:
            while not self._stop_event.is_set():
                self._refresh_roster()
                if self._refresh_event.wait(app.config["DASHBOARD_REFRESH_SECONDS"]):
                    self._refresh_event.clear()
                if self._stop_event.is_set():
                    break

        def snapshot(self) -> dict[str, Any]:
            with self._lock:
                top_symbols = [dict(item) for item in self._top_symbols]
                custom_symbols = list(self._custom_symbols)
                active_top_symbols = [normalize_symbol(item.get("symbol")) for item in top_symbols if item.get("selected", True)]
                symbols = merge_symbol_lists(active_top_symbols, custom_symbols)
                items: list[dict[str, Any]] = []
                for symbol in symbols:
                    meta = self._symbol_meta.get(symbol, {"source": top_source_label(), "rank": None})
                    session = self._sessions.get(symbol)
                    if session and session.get("data"):
                        row = dict(session["data"])
                        row["refreshing"] = bool(session.get("refreshing"))
                        row["last_error"] = session.get("last_error")
                    else:
                        row = dashboard_placeholder_row(symbol, source=str(meta.get("source", top_source_label())), rank=meta.get("rank"))
                        row["refreshing"] = bool(session.get("refreshing")) if session else False
                        row["last_error"] = session.get("last_error") if session else None
                    items.append(row)

                ready_rows = [item for item in items if item.get("ready")]
                cached = any(bool(item.get("cached")) for item in ready_rows)
                last_refreshed_candidates = [
                    str(session.get("last_refreshed_at"))
                    for session in self._sessions.values()
                    if session.get("last_refreshed_at")
                ]
                last_refreshed_candidates.append(self._last_refreshed_at or "")
                last_refreshed_at = max((value for value in last_refreshed_candidates if value), default=None)
                refreshing = self._roster_refreshing or any(bool(session.get("refreshing")) for session in self._sessions.values())
                payload = {
                    "generated_at": _utc_timestamp(),
                    "backend": dict(self._backend_health),
                    "top_symbols": top_symbols,
                    "custom_symbols": custom_symbols,
                    "symbols": symbols,
                    "items": items,
                    "selected_top_symbols": active_top_symbols,
                    "model_info": next((dict(session.get("model_info", {})) for session in self._sessions.values() if session.get("model_info")), {}),
                    "cached": cached,
                    "refresh_seconds": app.config["DASHBOARD_REFRESH_SECONDS"],
                    "interval": app.config["DASHBOARD_INTERVAL"],
                    "lookback": app.config["DASHBOARD_LOOKBACK"],
                    "pred_len": app.config["DASHBOARD_PRED_LEN"],
                    "refreshing": refreshing,
                    "last_error": self._last_error,
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
