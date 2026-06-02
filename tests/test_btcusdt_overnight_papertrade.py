from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "btcusdt_overnight_papertrade.py"
    spec = importlib.util.spec_from_file_location("btcusdt_overnight_papertrade", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_parser_accepts_model_selection_flags():
    module = _load_script_module()
    args = module.build_parser().parse_args(
        [
            "--model-size",
            "base",
            "--tokenizer-name",
            "custom/tokenizer",
            "--model-name",
            "custom/model",
        ]
    )

    assert args.model_size == "base"
    assert args.tokenizer_name == "custom/tokenizer"
    assert args.model_name == "custom/model"


def test_main_forwards_resolved_model_name(monkeypatch, tmp_path):
    module = _load_script_module()
    captured = {}

    def fake_parser():
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace(
            symbol="BTCUSDT",
            interval="15m",
            lookback=256,
            pred_len=1,
            sample_count=5,
            neutral_threshold_pct=0.05,
            confidence_samples=5,
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            model_name="NeoQuasar/Kronos-small",
            model_size="base",
            max_context=512,
            buffer_seconds=8,
            max_cycles=1,
            starting_balance=50.0,
            log_file=str(tmp_path / "overnight.jsonl"),
        )
        parser.parse_args = lambda argv=None: namespace
        return parser

    monkeypatch.setattr(module, "build_parser", fake_parser)
    monkeypatch.setattr(
        module,
        "fetch_latest_closed_candle",
        lambda symbol, interval: (datetime(2026, 6, 1, tzinfo=timezone.utc), 100.0),
    )
    monkeypatch.setattr(
        module,
        "predict_binance_direction",
        lambda **kwargs: captured.update(kwargs)
        or {
            "summary": {
                "signal": "UP",
                "action": "BUY",
                "verdict": "BUY",
                "predicted_close": 101.0,
                "last_close": 100.0,
                "close_change_pct": 1.0,
                "trade_confidence": 0.8,
                "confidence": 0.8,
                "confidence_samples": 5,
                "signal_counts": {"UP": 1, "DOWN": 0, "NEUTRAL": 0},
            }
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate_binance_prediction",
        lambda summary, actual_close, neutral_threshold_pct, stake_usd=None: {
            "match_signal": True,
            "match_action": True,
            "predicted_action": "BUY",
            "strategy_return_pct": 1.0,
            "trade_pnl_usd": 0.5,
            "actual_signal": "UP",
            "actual_action": "BUY",
            "actual_change_pct": 1.0,
        },
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    exit_code = module.main()

    assert exit_code == 0
    assert captured["model_name"] == "NeoQuasar/Kronos-base"
    assert captured["tokenizer_name"] == "NeoQuasar/Kronos-Tokenizer-base"
    assert captured["max_context"] == 512
