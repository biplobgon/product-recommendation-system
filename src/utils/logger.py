"""
utils/logger.py
---------------
Centralised logging setup.
"""
from __future__ import annotations

import logging
import sys


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured: set[str] = set()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger with a consistent format.

    Parameters
    ----------
    name:
        Logger name — use ``__name__`` for module-level loggers.
    level:
        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if name not in _configured:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _configured.add(name)

    return logger
