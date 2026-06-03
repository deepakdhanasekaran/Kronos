#!/usr/bin/env python3
from __future__ import annotations

import os

from app import app


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


if __name__ == "__main__":
    port = _env_int("PORT", 7070)
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(debug=False, host=host, port=port)
