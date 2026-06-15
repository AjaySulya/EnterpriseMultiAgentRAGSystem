"""
Structured logging via structlog.
"""

import logging
import structlog
from app.config import settings

def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        )
    
    structlog.configure(
        processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
                ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    
configure_logging()
    


def get_logger(name:str)-> structlog.BoundLogger:
    return structlog.get_logger(name)