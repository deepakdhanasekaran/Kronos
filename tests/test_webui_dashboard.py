from __future__ import annotations

from webui.app import create_app


def make_backend_stub(calls=None):
    calls = [] if calls is None else calls

    def fake_backend_request_json(backend_url, path, *, method="GET", payload=None, timeout=300):
        calls.append({"backend_url": backend_url, "path": path, "method": method, "payload": payload})
        if path.startswith("/health"):
            return {"ok": True, "model_loaded": True}
        if path.startswith("/top-symbols"):
            return {
                "items": [
                    {"symbol": "BTCUSDT", "quote_volume": 1000.0, "rank": 1},
                    {"symbol": "ETHUSDT", "quote_volume": 800.0, "rank": 2},
                ]
            }
        if path == "/predict/batch":
            symbols = payload["symbols"]
            return {
                "items": [
                    {
                        "symbol": symbol,
                        "rank": index + 1,
                        "source": "Top 30" if index < 2 else "Custom",
                        "last_close": 100.0 + index,
                        "predicted_close": 101.0 + index,
                        "verdict": "Strong BUY",
                        "action": "BUY",
                        "signal": "UP",
                        "agreement": 0.6,
                        "trade_confidence": 0.6,
                        "signal_counts": {"UP": 2, "DOWN": 3, "NEUTRAL": 0},
                    }
                    for index, symbol in enumerate(symbols)
                ],
                "model_info": {"model_name": "Kronos-base"},
            }
        raise AssertionError(f"Unexpected backend path: {path}")

    return fake_backend_request_json


def test_dashboard_merges_top_symbols_with_persisted_custom_symbols(monkeypatch, tmp_path):
    calls = []
    app = create_app(
        {
            "BACKEND_URL": "http://backend",
            "WATCHLIST_PATH": str(tmp_path / "custom_coins.json"),
            "DASHBOARD_TOP_SYMBOL_LIMIT": 30,
            "DASHBOARD_BACKGROUND_REFRESH": False,
        }
    )
    monkeypatch.setattr("webui.app._backend_request_json", make_backend_stub(calls))
    app.extensions["watchlist_store"].save(["XRPUSDT"])

    client = app.test_client()
    response = client.get("/api/dashboard")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["custom_symbols"] == ["XRPUSDT"]
    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    assert len(payload["items"]) == 3
    assert payload["items"][2]["source"] == "Custom"
    assert any(call["path"] == "/predict/batch" for call in calls)
    assert not any(call["path"] == "/predict" for call in calls)


def test_dashboard_placeholder_response_includes_safe_timestamp(monkeypatch, tmp_path):
    app = create_app(
        {
            "BACKEND_URL": "http://backend",
            "WATCHLIST_PATH": str(tmp_path / "custom_coins.json"),
            "DASHBOARD_BACKGROUND_REFRESH": False,
        }
    )
    monkeypatch.setattr("webui.app._backend_request_json", make_backend_stub())
    app.config["DASHBOARD_BACKGROUND_REFRESH"] = True

    client = app.test_client()
    response = client.get("/api/dashboard")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["refreshing"] is True
    assert "generated_at" in payload
    assert payload["generated_at"]


def test_dashboard_manager_refreshes_in_batch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("webui.app._backend_request_json", make_backend_stub(calls))

    app = create_app(
        {
            "BACKEND_URL": "http://backend",
            "WATCHLIST_PATH": str(tmp_path / "custom_coins.json"),
            "DASHBOARD_BACKGROUND_REFRESH": True,
            "DASHBOARD_REFRESH_SECONDS": 3600,
        }
    )

    client = app.test_client()
    response = client.get("/api/dashboard")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert len(payload["items"]) == 2
    assert any(call["path"] == "/predict/batch" for call in calls)
    assert not any(call["path"] == "/predict" for call in calls)


def test_dashboard_respects_selected_top_symbols(monkeypatch, tmp_path):
    app = create_app(
        {
            "BACKEND_URL": "http://backend",
            "WATCHLIST_PATH": str(tmp_path / "custom_coins.json"),
            "DASHBOARD_BACKGROUND_REFRESH": False,
        }
    )
    monkeypatch.setattr("webui.app._backend_request_json", make_backend_stub())

    client = app.test_client()
    response = client.get("/api/dashboard?selected_top_symbols=BTCUSDT")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["selected_top_symbols"] == ["BTCUSDT"]
    assert payload["symbols"] == ["BTCUSDT"]
    assert [item["symbol"] for item in payload["items"]] == ["BTCUSDT"]
    assert payload["top_symbols"][0]["selected"] is True
    assert payload["top_symbols"][1]["selected"] is False


def test_watchlist_add_and_delete_routes_validate_symbols(monkeypatch, tmp_path):
    app = create_app(
        {
            "BACKEND_URL": "http://backend",
            "WATCHLIST_PATH": str(tmp_path / "custom_coins.json"),
            "DASHBOARD_BACKGROUND_REFRESH": False,
        }
    )
    monkeypatch.setattr("webui.app._backend_request_json", make_backend_stub())
    client = app.test_client()

    added = client.post("/api/watchlist", json={"symbol": "ethusdt"})
    assert added.status_code == 200
    assert added.get_json()["symbols"] == ["ETHUSDT"]

    duplicate = client.post("/api/watchlist", json={"symbol": "ETHUSDT"})
    assert duplicate.status_code == 200
    assert duplicate.get_json()["symbols"] == ["ETHUSDT"]

    invalid = client.post("/api/watchlist", json={"symbol": "BTC"})
    assert invalid.status_code == 400

    deleted = client.delete("/api/watchlist/ETHUSDT")
    assert deleted.status_code == 200
    assert deleted.get_json()["symbols"] == []
