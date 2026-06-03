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
