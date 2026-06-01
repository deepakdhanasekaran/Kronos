import json
from datetime import timedelta

import pandas as pd
import pytest

from crypto_predictor import (
    action_from_signal,
    build_future_timestamps,
    build_parser,
    compute_direction_summary,
    compare_with_polymarket,
    build_polymarket_summary,
    build_polymarket_audit_record,
    evaluate_binance_prediction,
    extract_polymarket_final_outcome,
    compute_technical_indicators,
    combine_kronos_and_indicator_signals,
    indicator_trade_signal,
    interval_to_timedelta,
    load_jsonl_records,
    market_is_closed,
    polymarket_action_from_probability,
    resolve_current_polymarket_slug,
    score_sampled_directions,
    signal_from_change_pct,
    render_overnight_audit_report,
    render_cli_output,
    prepare_ohlcv_frame,
    verdict_from_summary,
)


def test_interval_to_timedelta_supports_common_crypto_bars():
    assert interval_to_timedelta("5m") == timedelta(minutes=5)
    assert interval_to_timedelta("1h") == timedelta(hours=1)
    assert interval_to_timedelta("1d") == timedelta(days=1)
    assert interval_to_timedelta("1w") == timedelta(weeks=1)


def test_interval_to_timedelta_rejects_unknown_intervals():
    with pytest.raises(ValueError, match="Unsupported interval"):
        interval_to_timedelta("15x")


def test_build_future_timestamps_steps_forward_from_last_timestamp():
    last_timestamp = pd.Timestamp("2026-05-30T12:00:00Z")
    future = build_future_timestamps(last_timestamp, "5m", 3)

    assert list(future) == [
        pd.Timestamp("2026-05-30T12:05:00Z"),
        pd.Timestamp("2026-05-30T12:10:00Z"),
        pd.Timestamp("2026-05-30T12:15:00Z"),
    ]


def test_prepare_ohlcv_frame_fills_missing_volume_and_amount():
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )

    prepared = prepare_ohlcv_frame(frame)

    assert list(prepared.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert prepared["volume"].tolist() == [0.0, 0.0]
    assert prepared["amount"].tolist() == [0.0, 0.0]


def test_prepare_ohlcv_frame_derives_amount_from_volume_when_needed():
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 20.0],
        }
    )

    prepared = prepare_ohlcv_frame(frame)

    expected = frame["volume"] * frame[["open", "high", "low", "close"]].mean(axis=1)
    assert prepared["amount"].tolist() == pytest.approx(expected.tolist())


def test_compute_direction_summary_uses_close_and_live_reference():
    summary = compute_direction_summary(
        predicted_close=112.0,
        last_close=100.0,
        live_price=110.0,
    )

    assert summary["direction_vs_last_close"] == "UP"
    assert summary["direction_vs_live"] == "UP"
    assert summary["close_change_pct"] == pytest.approx(12.0)
    assert summary["live_change_pct"] == pytest.approx(1.8181818)


def test_signal_from_change_pct_supports_neutral_band():
    assert signal_from_change_pct(0.25, neutral_threshold_pct=0.2) == "UP"
    assert signal_from_change_pct(-0.25, neutral_threshold_pct=0.2) == "DOWN"
    assert signal_from_change_pct(0.05, neutral_threshold_pct=0.2) == "NEUTRAL"


def test_action_from_signal_maps_signals_to_actions():
    assert action_from_signal("UP") == "BUY"
    assert action_from_signal("DOWN") == "SELL"
    assert action_from_signal("NEUTRAL") == "NO_TRADE"


def test_verdict_from_summary_promotes_only_high_confidence_trades():
    assert verdict_from_summary({"action": "NO_TRADE", "trade_confidence": 1.0}) == "NO_TRADE"
    assert verdict_from_summary({"action": "BUY", "trade_confidence": 0.8}) == "Strong BUY"
    assert verdict_from_summary({"action": "SELL", "trade_confidence": 0.8}) == "Strong SELL"
    assert verdict_from_summary({"action": "BUY", "trade_confidence": 0.2}) == "BUY"


def test_evaluate_binance_prediction_scores_direction_and_return():
    summary = {
        "last_close": 100.0,
        "signal": "UP",
        "action": "BUY",
        "neutral_threshold_pct": 0.2,
    }

    evaluation = evaluate_binance_prediction(summary, actual_close=103.0)

    assert evaluation["actual_signal"] == "UP"
    assert evaluation["actual_action"] == "BUY"
    assert evaluation["match_signal"] is True
    assert evaluation["match_action"] is True
    assert evaluation["strategy_return_pct"] == pytest.approx(3.0)


def test_evaluate_binance_prediction_calculates_pnl_from_stake():
    summary = {
        "last_close": 100.0,
        "signal": "DOWN",
        "action": "SELL",
        "neutral_threshold_pct": 0.2,
    }

    evaluation = evaluate_binance_prediction(summary, actual_close=97.0, stake_usd=50.0)

    assert evaluation["strategy_return_pct"] == pytest.approx(3.0)
    assert evaluation["trade_pnl_usd"] == pytest.approx(1.5)


def test_compute_technical_indicators_and_trade_signal_buy():
    frame = pd.DataFrame(
        {
            "close": [float(i) for i in range(1, 40)],
        }
    )
    indicators = compute_technical_indicators(frame, ema_fast=5, ema_slow=10, rsi_period=5, momentum_period=3)
    row = indicators.iloc[-1]

    signal = indicator_trade_signal(row, overbought=101.0, oversold=-1.0, min_score=2)

    assert row["ema_fast"] > row["ema_slow"]
    assert row["momentum_pct"] > 0
    assert signal["signal"] == "BUY"
    assert signal["action"] == "BUY"
    assert signal["score"] >= 2


def test_compute_technical_indicators_and_trade_signal_sell():
    frame = pd.DataFrame(
        {
            "close": [float(i) for i in range(40, 1, -1)],
        }
    )
    indicators = compute_technical_indicators(frame, ema_fast=5, ema_slow=10, rsi_period=5, momentum_period=3)
    row = indicators.iloc[-1]

    signal = indicator_trade_signal(row, overbought=101.0, oversold=-1.0, min_score=2)

    assert row["ema_fast"] < row["ema_slow"]
    assert row["momentum_pct"] < 0
    assert signal["signal"] == "SELL"
    assert signal["action"] == "SELL"
    assert signal["score"] <= -2


def test_combine_kronos_and_indicator_signals_requires_agreement():
    kronos_summary = {"action": "BUY", "verdict": "Strong BUY", "trade_confidence": 0.8}
    indicator_signal = {"action": "BUY", "signal": "BUY"}

    combined = combine_kronos_and_indicator_signals(kronos_summary, indicator_signal)

    assert combined["action"] == "BUY"
    assert combined["agreement"] is True
    assert combined["signal"] == "UP"


def test_combine_kronos_and_indicator_signals_blocks_disagreement():
    kronos_summary = {"action": "BUY", "verdict": "Strong BUY", "trade_confidence": 0.8}
    indicator_signal = {"action": "SELL", "signal": "SELL"}

    combined = combine_kronos_and_indicator_signals(kronos_summary, indicator_signal)

    assert combined["action"] == "NO_TRADE"
    assert combined["agreement"] is False
    assert combined["signal"] == "NEUTRAL"


def test_polymarket_action_from_probability_uses_neutral_band():
    assert polymarket_action_from_probability("UP", 0.62, edge_threshold_pct=0.05) == "BUY"
    assert polymarket_action_from_probability("DOWN", 0.62, edge_threshold_pct=0.05) == "SELL"
    assert polymarket_action_from_probability("UP", 0.52, edge_threshold_pct=0.05) == "NO_TRADE"


def test_score_sampled_directions_measures_agreement():
    scored = score_sampled_directions(
        [
            "UP",
            "UP",
            "DOWN",
            "UP",
            "NEUTRAL",
        ]
    )

    assert scored["sample_count"] == 5
    assert scored["dominant_signal"] == "UP"
    assert scored["confidence"] == pytest.approx(0.6)
    assert scored["trade_confidence"] == pytest.approx(0.6)


def test_score_sampled_directions_reports_zero_trade_confidence_for_neutral_majority():
    scored = score_sampled_directions(["NEUTRAL", "NEUTRAL", "NEUTRAL", "UP"])

    assert scored["dominant_signal"] == "NEUTRAL"
    assert scored["confidence"] == pytest.approx(0.75)
    assert scored["trade_confidence"] == pytest.approx(0.0)


def test_build_parser_exposes_useful_cli_flags():
    parser = build_parser()
    args = parser.parse_args(["--symbol", "ETHUSDT", "--interval", "15m", "--lookback", "128"])

    assert args.symbol == "ETHUSDT"
    assert args.interval == "15m"
    assert args.lookback == 128
    assert args.pred_len == 1
    assert args.polymarket_slug is None
    assert args.current is False
    assert args.polymarket_only is False


def test_resolve_current_polymarket_slug_uses_active_events_api(monkeypatch):
    events_payload = [
        {
            "title": "BTC Up or Down 5m",
            "slug": "btc-updown-5m-1780166400",
            "endDate": "2026-05-30T18:45:00Z",
        },
        {
            "title": "ETH Up or Down 5m",
            "slug": "eth-updown-5m-1780166400",
            "endDate": "2026-05-30T18:45:00Z",
        },
    ]

    monkeypatch.setattr("crypto_predictor.fetch_json", lambda url, headers=None: json.dumps(events_payload))
    monkeypatch.setattr("crypto_predictor.pd.Timestamp.now", lambda tz=None: pd.Timestamp("2026-05-30T18:40:00Z"))

    assert resolve_current_polymarket_slug() == "btc-updown-5m-1780166400"


def test_compare_with_polymarket_uses_api_payload(monkeypatch):
    event_payload = {
        "slug": "btc-updown-5m-123",
        "title": "BTC up/down 5m",
        "markets": [
            {
                "slug": "btc-updown-5m-123",
                "question": "Will BTC go up?",
                "outcomes": "['Yes', 'No']",
                "outcomePrices": "['0.70', '0.30']",
            }
        ],
    }

    monkeypatch.setattr("crypto_predictor.fetch_json", lambda url, headers=None: json.dumps(event_payload))

    summary = {
        "signal": "UP",
        "action": "BUY",
    }
    comparison = compare_with_polymarket(summary, "btc-updown-5m-123")

    assert comparison["market_signal"] == "UP"
    assert comparison["kronos_signal"] == "UP"
    assert comparison["agreement"] is True
    assert comparison["slug"] == "btc-updown-5m-123"
    assert comparison["market_probability"] == pytest.approx(0.7)
    assert comparison["action"] == "BUY"


def test_build_polymarket_summary_returns_action(monkeypatch):
    event_payload = {
        "slug": "btc-updown-5m-123",
        "title": "BTC up/down 5m",
        "markets": [
            {
                "slug": "btc-updown-5m-123",
                "question": "Will BTC go up?",
                "outcomes": "['Yes', 'No']",
                "outcomePrices": "['0.38', '0.62']",
            }
        ],
    }

    monkeypatch.setattr("crypto_predictor.fetch_json", lambda url, headers=None: json.dumps(event_payload))

    summary = build_polymarket_summary("btc-updown-5m-123")

    assert summary["market_signal"] == "DOWN"
    assert summary["action"] == "SELL"
    assert summary["market_probability"] == pytest.approx(0.62)


def test_build_polymarket_summary_prefers_market_payload(monkeypatch):
    market_payload = {
        "slug": "btc-updown-5m-123",
        "question": "Will BTC go up?",
        "outcomes": "['Yes', 'No']",
        "outcomePrices": "['0.18', '0.82']",
    }
    event_payload = {
        "slug": "btc-updown-5m-123",
        "title": "BTC up/down 5m",
        "markets": [
            {
                "slug": "btc-updown-5m-123",
                "question": "Will BTC go up?",
                "outcomes": "['Yes', 'No']",
                "outcomePrices": "['0.50', '0.50']",
            }
        ],
    }

    def fake_fetch_json(url, headers=None):
        if "/markets/slug/" in url:
            return json.dumps(market_payload)
        return json.dumps(event_payload)

    monkeypatch.setattr("crypto_predictor.fetch_json", fake_fetch_json)

    summary = build_polymarket_summary("btc-updown-5m-123")

    assert summary["market_probability"] == pytest.approx(0.82)
    assert summary["action"] == "SELL"


def test_build_polymarket_audit_record_tracks_prediction_and_final_outcome(monkeypatch):
    market_payload = {
        "slug": "btc-updown-5m-123",
        "question": "Will BTC go up?",
        "outcomes": "['Yes', 'No']",
        "outcomePrices": "['0.18', '0.82']",
        "closed": True,
        "resolvedOutcome": "Down",
    }
    event_payload = {
        "slug": "btc-updown-5m-123",
        "title": "BTC up/down 5m",
        "markets": [market_payload],
    }

    def fake_fetch_json(url, headers=None):
        if "/markets/slug/" in url:
            return json.dumps(market_payload)
        return json.dumps(event_payload)

    monkeypatch.setattr("crypto_predictor.fetch_json", fake_fetch_json)
    monkeypatch.setattr("crypto_predictor.time.sleep", lambda seconds: None)

    record = build_polymarket_audit_record("btc-updown-5m-123", poll_seconds=1, max_wait_seconds=1)

    assert record["prediction"] == "SELL"
    assert record["market_signal"] == "DOWN"
    assert record["final_outcome"] == "DOWN"
    assert record["matched"] is True
    assert record["closed"] is True
    assert record["waited_seconds"] == 0


def test_extract_polymarket_final_outcome_reads_common_fields():
    assert extract_polymarket_final_outcome({"winningOutcome": "Down"}) == "DOWN"
    assert extract_polymarket_final_outcome({"resolvedOutcome": "Up"}) == "UP"
    assert extract_polymarket_final_outcome(
        {"tokens": [{"outcome": "Down", "winner": True}]}
    ) == "DOWN"


def test_market_is_closed_uses_end_timestamp_when_status_is_missing(monkeypatch):
    monkeypatch.setattr("crypto_predictor.pd.Timestamp.now", lambda tz=None: pd.Timestamp("2026-05-30T18:56:00Z"))

    assert market_is_closed({"endDate": "2026-05-30T18:55:00Z"}) is True
    assert market_is_closed({"endDate": "2026-05-30T19:00:00Z"}) is False


def test_render_cli_output_includes_summary_and_head():
    summary = {
        "predicted_close": 112.0,
        "last_close": 100.0,
        "live_price": 110.0,
        "direction_vs_last_close": "UP",
        "direction_vs_live": "UP",
        "close_change_pct": 12.0,
        "live_change_pct": 1.8181818,
        "signal": "UP",
        "action": "BUY",
        "verdict": "Strong BUY",
        "confidence": 0.8,
        "confidence_samples": 5,
    }
    pred_df = pd.DataFrame(
        {
            "open": [111.0],
            "high": [113.0],
            "low": [109.0],
            "close": [112.0],
            "volume": [42.0],
            "amount": [4704.0],
        }
    )

    rendered = render_cli_output(
        symbol="BTCUSDT",
        interval="5m",
        prediction=pred_df,
        summary=summary,
    )

    assert "Symbol: BTCUSDT" in rendered
    assert "Action: BUY" in rendered
    assert "Verdict: Strong BUY" in rendered
    assert "Agreement: 0.80" in rendered
    assert "Trade confidence: 0.80" in rendered
    assert "112.0000" in rendered


def test_render_cli_output_includes_polymarket_comparison():
    summary = {
        "predicted_close": 112.0,
        "last_close": 100.0,
        "live_price": 110.0,
        "direction_vs_last_close": "UP",
        "direction_vs_live": "UP",
        "close_change_pct": 12.0,
        "live_change_pct": 1.8181818,
        "signal": "UP",
        "action": "BUY",
        "confidence": 0.8,
        "confidence_samples": 5,
    }
    pred_df = pd.DataFrame(
        {
            "open": [111.0],
            "high": [113.0],
            "low": [109.0],
            "close": [112.0],
            "volume": [42.0],
            "amount": [4704.0],
        }
    )
    polymarket = {
        "title": "Bitcoin Up or Down - May 30, 2:15PM-2:20PM ET",
        "slug": "btc-updown-5m-1780164300",
        "market_signal": "UP",
        "market_probability": 0.62,
        "action": "BUY",
        "kronos_signal": "UP",
        "kronos_action": "BUY",
        "agreement": True,
    }

    rendered = render_cli_output(
        symbol="BTCUSDT",
        interval="5m",
        prediction=pred_df,
        summary=summary,
        polymarket=polymarket,
    )

    assert "Polymarket comparison:" in rendered
    assert "Market probability: 62.00%" in rendered
    assert "Agreement: YES" in rendered


def test_render_polymarket_action_only_output():
    rendered = render_cli_output(
        symbol="BTCUSDT",
        interval="5m",
        prediction=pd.DataFrame(),
        summary={},
        polymarket=None,
        action_only="BUY",
    )

    assert rendered == "BUY"


def test_load_jsonl_records_parses_valid_lines(tmp_path):
    log_file = tmp_path / "overnight.jsonl"
    log_file.write_text(
        "\n".join(
            [
                json.dumps({"event": "snapshot", "value": 1}),
                "",
                json.dumps({"event": "trade_close", "value": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_jsonl_records(log_file)

    assert len(records) == 2
    assert records[0]["event"] == "snapshot"
    assert records[1]["value"] == 2


def test_render_overnight_audit_report_formats_summary_and_table():
    records = [
        {
            "cycle": 1,
            "timestamp_utc": "2026-06-01T19:30:00Z",
            "prediction": {
                "verdict": "BUY",
                "signal": "UP",
                "action": "BUY",
                "predicted_close": 71452.5547,
            },
            "actual": {
                "close": 71480.0,
                "signal": "UP",
                "action": "BUY",
            },
            "evaluation": {
                "match_signal": True,
                "match_action": True,
                "strategy_return_pct": 0.05,
                "trade_pnl_usd": 0.025,
            },
            "stats": {
                "balance_usd": 50.025,
            },
        },
        {
            "cycle": 2,
            "timestamp_utc": "2026-06-01T19:45:00Z",
            "prediction": {
                "verdict": "SELL",
                "signal": "DOWN",
                "action": "SELL",
                "predicted_close": 71400.0,
            },
            "actual": {
                "close": 71360.0,
                "signal": "DOWN",
                "action": "SELL",
            },
            "evaluation": {
                "match_signal": True,
                "match_action": True,
                "strategy_return_pct": 0.06,
                "trade_pnl_usd": 0.03,
            },
            "stats": {
                "balance_usd": 50.055,
            },
        },
    ]

    report = render_overnight_audit_report(records)

    assert "Overnight backtest report:" in report
    assert "Cycles: 2" in report
    assert "Trade win rate: 100.00%" in report
    assert "| Cycle | Timestamp" in report
    assert "2026-06-01T19:30:00Z" in report
    assert "71452.5547" in report
