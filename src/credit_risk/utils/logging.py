"""Logging configuration for command-line pipeline runs."""

from __future__ import annotations

import logging

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", enabled: bool = True) -> None:
    """Configure console logging without replacing existing application handlers."""

    package_logger = logging.getLogger("credit_risk")
    package_logger.disabled = not enabled
    if not enabled:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise TypeError(f"Unsupported logging level: {level}")

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        return

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
