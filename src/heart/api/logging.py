"""Structured JSON logging: one line per event, easy to search and ship to a log stack.

    {"ts": "2026-10-14T06:40:12.345Z", "level": "INFO", "logger": "heart.api",
     "msg": "request", "request_id": "3f9c...", "path": "/predict", "status": 200,
     "duration_ms": 4.1, "prediction": 1, "probability": 0.9787}

Patient measurements are deliberately never logged.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Attributes every LogRecord has; anything else came from `extra=` and is emitted as a field
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}
# uvicorn adds a terminal-coloured duplicate of the message; drop it from JSON
_SKIP = _STANDARD | {"color_message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _SKIP})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    for name in ("heart", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False
    # The request middleware already logs every request with more detail
    logging.getLogger("uvicorn.access").disabled = True
