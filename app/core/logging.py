import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

from app.core.config import get_settings

_CONFIGURED = False


def _build_formatter(use_json: bool) -> logging.Formatter:
    if use_json:
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","line":%(lineno)d,"msg":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")


def setup_logging() -> None:
    """Configure logging once; subsequent calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    formatter = _build_formatter(settings.LOG_JSON)

    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        handlers=[console_handler, file_handler],
        force=True,
    )

    # Quiet noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is set up first."""
    setup_logging()
    return logging.getLogger(name)
