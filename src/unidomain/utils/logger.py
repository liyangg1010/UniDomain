"""
Logging Utilities
=================

This module configures the global logging state and provides factory functions
for context-aware loggers. It uses a "LoggerAdapter" to inject directory
context into log records, allowing custom file handlers to route logs dynamically
(e.g., creating a separate log file inside each task's folder).

Architecture Position:
    [Shared Infrastructure / Utility] -> Central logging facility.
"""

import logging
import logging.config

from unidomain.configs.constants import LoggerName
from unidomain.configs.logger import LOGGING_CONFIG
from unidomain.schemas.typings import Paths

logging.config.dictConfig(LOGGING_CONFIG)
execution_logger = logging.getLogger(LoggerName.EXECUTION)
llm_usage_logger = logging.getLogger(LoggerName.LLM_USAGE)

__all__ = ["get_task_logger", "llm_usage_logger", "execution_logger"]


def get_task_logger(log_dir: Paths) -> logging.LoggerAdapter:
    """Create a logger adapter injected with the specific directory context.

    This adapter allows log messages to carry the "log_dir" extra field.
    The underlying logging handler (e.g., a custom FileHandler) can use this
    field to determine where to write the log file for the specific task.

    Args:
        log_dir (Paths): The directory path associated with the current task.

    Returns:
        logging.LoggerAdapter: An adapter wrapping the execution logger.
    """
    
    log_context = {"log_dir": str(log_dir)}
    adapter = logging.LoggerAdapter(execution_logger, log_context)

    return adapter
