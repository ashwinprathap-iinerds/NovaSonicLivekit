import logging
import sys
from typing import Optional

# Create a logger instance
logger = logging.getLogger("livekit-agent")

# Set default logging level
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

def set_log_level(level: str) -> None:
    """
    Set the logging level for the logger.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logger.setLevel(numeric_level)
    console_handler.setLevel(numeric_level)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    If no name is provided, returns the default logger.
    
    Args:
        name: Optional name for the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logging.getLogger(f"livekit-agent.{name}")
    return logger 