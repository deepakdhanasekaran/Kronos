while true; do python crypto_predictor.py --current --polymarket-only sleep 3 done


(.venv) deepakdhanasekaran ~/repos/private/Kronos [master] $ python crypto_predictor.py \
  --symbol BTCUSDT \
  --interval 5m \
  --lookback 256 \
  --pred-len 1 \
  --neutral-threshold-pct 0.05 \
  --confidence-samples 5 \
  --model-size base
