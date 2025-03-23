# logging_config.py
import logging
import logging.handlers
import os
from pathlib import Path
import json


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from log records."""

    def __init__(self, sensitive_patterns=None):
        super().__init__()
        self.sensitive_patterns = sensitive_patterns or ['password', 'token', 'api_key', 'secret']

    def filter(self, record):
        if isinstance(record.msg, str):
            for pattern in self.sensitive_patterns:
                # Simple pattern replacement, could be more sophisticated
                record.msg = record.msg.replace(pattern, '****REDACTED****')
        return True


def setup_logging(
        log_name="app",
        log_dir="logs",
        log_level=logging.INFO,
        rotation_size=10 * 1024 * 1024,
        backup_count=10,
        console_output=True,
        console_level=logging.DEBUG
):
    """Configure application logging with rotation."""
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []

    # Add sensitive data filter
    logger.addFilter(SensitiveDataFilter())

    # Detailed formatter for debugging
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - [%(threadName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / f"{log_name}.log",
        maxBytes=rotation_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(detailed_formatter)
        logger.addHandler(console_handler)

    return logger


# Utility functions for logging structured data
def log_structured(logger, level, message, data=None, **kwargs):
    """Log a message with structured data."""
    if data is not None:
        if isinstance(data, (dict, list)):
            try:
                message = f"{message} {json.dumps(data, default=str)}"
            except (TypeError, ValueError):
                message = f"{message} {str(data)}"
        else:
            message = f"{message} {data}"

    # Add any additional kwargs to the message
    if kwargs:
        message = f"{message} {json.dumps(kwargs, default=str)}"

    if level == 'debug':
        logger.debug(message)
    elif level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'critical':
        logger.critical(message)


# Create default application logger
app_logger = setup_logging()