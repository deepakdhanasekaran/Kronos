FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=7070 \
    HOST=0.0.0.0

WORKDIR /app

COPY webui/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard_support.py ./
COPY webui ./webui

WORKDIR /app/webui

EXPOSE 7070

CMD ["python", "app.py"]
