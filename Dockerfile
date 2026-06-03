FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KRONOS_MODEL_SIZE=base \
    KRONOS_HOST=0.0.0.0 \
    KRONOS_PORT=8765 \
    KRONOS_CACHE_TTL_SECONDS=10 \
    KRONOS_MAX_CONCURRENT_PREDICTIONS=2

WORKDIR /app

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8765

CMD ["sh", "-c", "python crypto_predictor.py --serve --model-size \"$KRONOS_MODEL_SIZE\" --host \"$KRONOS_HOST\" --port \"$KRONOS_PORT\" --cache-ttl-seconds \"$KRONOS_CACHE_TTL_SECONDS\" --max-concurrent-predictions \"$KRONOS_MAX_CONCURRENT_PREDICTIONS\""]
