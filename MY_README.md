while true; do python crypto_predictor.py --current --polymarket-only sleep 3 done


(.venv) deepakdhanasekaran ~/repos/private/Kronos [master] $ python crypto_predictor.py \
  --symbol BTCUSDT \
  --interval 5m \
  --lookback 256 \
  --pred-len 1 \
  --neutral-threshold-pct 0.05 \
  --confidence-samples 5 \
  --model-size base


Load the Model faster by keeping it hot
python crypto_predictor.py --serve --model-size base --host 127.0.0.1 --port 8765
python crypto_predictor.py --live-url http://127.0.0.1:8765 --symbol BTCUSDT --interval 15m --lookback 256 --pred-len 1 --neutral-threshold-pct 0.05 --confidence-samples 5


Demo Account  export BINANCE_EXECUTION_MODE=demo
  export BINANCE_DEMO_API_KEY="your_demo_api_key"
  export BINANCE_DEMO_SECRET_KEY="your_demo_secret_key"

  python scripts/binance_paper_trade_bot.py \
    --execution-mode demo \
    --binance-base-url https://demo-api.binance.com \
    --live-url http://127.0.0.1:8765 \
    --symbols BTCUSDT,ETHUSDT \
    --interval 15m \
    --pred-len 1 \
    --neutral-threshold-pct 0.05 \
    --confidence-samples 5 \
    --take-profit-pct 0.5 \
    --stop-loss-pct 0.25 \
    --exit-on-opposite-signal \
    --max-cycles 2 \
    --poll-seconds 2
