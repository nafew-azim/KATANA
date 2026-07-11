"""Logging utilities shared across the KATANA framework."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(log_file: str = "katana.log", level: int = logging.INFO) -> logging.Logger:
    """Configure the ``katana`` logger to write to both a file and the console.

    Mirrors the logging setup used during the original RL discovery runs so
    that search transcripts (candidates, rewards, dashboards) are preserved.
    """
    logger = logging.getLogger("katana")
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("katana")
