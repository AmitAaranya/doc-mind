import logging
import logging.config
import sys
from typing import Any

from app.core.config import get_settings

_CONFIGURED = False


def _build_config(level: str, use_json: bool) -> dict[str, Any]:
    fmt_text = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    fmt_json = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","line":%(lineno)d,"msg":"%(message)s"}'
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "text": {"format": fmt_text, "datefmt": "%Y-%m-%dT%H:%M:%S"},
            "json": {"format": fmt_json, "datefmt": "%Y-%m-%dT%H:%M:%S"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json" if use_json else "text",
            }
        },
        "root": {"level": level, "handlers": ["stdout"]},
        # Quiet noisy third-party loggers
        "loggers": {
            "uvicorn.access": {"level": "WARNING", "propagate": True},
            "httpx": {"level": "WARNING", "propagate": True},
        },
    }


def setup_logging() -> None:
    """Configure logging once; subsequent calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logging.config.dictConfig(_build_config(settings.LOG_LEVEL, settings.LOG_JSON))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is set up first."""
    setup_logging()
    return logging.getLogger(name)
