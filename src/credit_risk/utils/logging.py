"""Logging configuration for command-line pipeline runs."""

from __future__ import annotations

import logging
import sys

DEFAULT_FORMAT = "%(asctime)s  %(levelname)s  %(name)-12s  %(message)s"
LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
RESET_COLOR = "\033[0m"


class ConsoleFormatter(logging.Formatter):
    """Render compact, readable pipeline logs for the console."""

    def __init__(self, use_colors: bool) -> None:
        super().__init__(DEFAULT_FORMAT, datefmt="%H:%M:%S")
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        original_name = record.name
        original_levelname = record.levelname

        record.name = record.name.removeprefix("credit_risk.")
        record.name = record.name.removeprefix("pipelines.")
        if record.name == "main":
            record.name = "pipeline"

        record.levelname = f"{record.levelname:<8}"
        if self.use_colors:
            color = LEVEL_COLORS.get(original_levelname)
            if color:
                record.levelname = f"{color}{record.levelname}{RESET_COLOR}"

        try:
            return super().format(record)
        finally:
            record.name = original_name
            record.levelname = original_levelname


def configure_logging(
    level: str = "INFO",
    enabled: bool = True,
    color: bool = True,
) -> None:
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
        datefmt="%H:%M:%S",
    )
    root_logger.handlers[0].setFormatter(ConsoleFormatter(use_colors=color))
