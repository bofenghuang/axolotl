"""
Common logging module for axolotl
"""

import os
import sys
from logging import Formatter
from logging.config import dictConfig
from typing import Any, Dict

from colorama import Fore, Style, init


class ColorfulFormatter(Formatter):
    """
    Formatter to add coloring to log messages by log type
    """

    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        record.rank = int(os.getenv("LOCAL_RANK", "0"))
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET


DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
        },
        "colorful": {
            "()": ColorfulFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] [RANK:%(rank)d] %(message)s",
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "filters": [],
            "stream": sys.stdout,
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "colorful",
            "filters": [],
            "stream": sys.stdout,
        },
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {
        "axolotl": {
            "handlers": ["color_console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}


def configure_logging():
    """Configure with default logging"""
    init()  # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)

    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()
