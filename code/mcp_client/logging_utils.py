"""
Logging utilities for MCP Client testing.

Provides response trimming and structured logging for debugging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import get_settings


def trim_response(response: Any, max_chars: int = 1000) -> tuple[str, int]:
    """
    Trim response to max_chars and return (trimmed_text, original_length).
    
    Args:
        response: Response data (dict, str, or any JSON-serializable type)
        max_chars: Maximum characters to keep
    
    Returns:
        Tuple of (trimmed_string, original_length)
    """
    if isinstance(response, dict):
        text = json.dumps(response, indent=2, default=str)
    elif isinstance(response, str):
        text = response
    else:
        text = str(response)
    
    original_length = len(text)
    
    if original_length > max_chars:
        trimmed = text[:max_chars] + f"\n... [TRIMMED: {original_length - max_chars} more chars]"
        return trimmed, original_length
    
    return text, original_length


def setup_test_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "DEBUG",
    session_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for test sessions.
    
    Creates a new log file for each test session with timestamp.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        session_name: Optional name for the session
    
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    log_dir = log_dir or settings.log_dir
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = session_name or "mcp_test"
    log_file = log_dir / f"{session_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("mcp_client_test")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)-8s | %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.info(f"Test session started, logging to: {log_file}")
    
    return logger


class ResultLogger:
    """
    Helper class for logging test results with response trimming.
    
    Note: Named 'ResultLogger' instead of 'TestResultLogger' to avoid 
    pytest collection warnings (pytest tries to collect classes starting with 'Test').
    """
    
    def __init__(self, logger: logging.Logger, max_chars: int = 1000):
        """
        Initialize result logger.
        
        Args:
            logger: Logger instance
            max_chars: Maximum characters for response trimming
        """
        self.logger = logger
        self.max_chars = max_chars
    
    def log_request(self, method: str, params: Optional[dict] = None) -> None:
        """Log an outgoing request."""
        params_str = json.dumps(params, default=str) if params else "{}"
        self.logger.info(f"REQUEST: {method}")
        self.logger.debug(f"  Params: {params_str}")
    
    def log_response(
        self,
        method: str,
        response: Any,
        success: bool = True,
        error_msg: Optional[str] = None,
    ) -> None:
        """
        Log a response with trimming.
        
        Args:
            method: Method name for context
            response: Response data
            success: Whether the request succeeded
            error_msg: Optional error message
        """
        trimmed, original_len = trim_response(response, self.max_chars)
        
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"RESPONSE: {method} - {status} (original: {original_len} chars)")
        
        if error_msg:
            self.logger.error(f"  Error: {error_msg}")
        
        self.logger.debug(f"  Content:\n{trimmed}")
    
    def log_test_result(
        self,
        test_name: str,
        passed: bool,
        response: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Log a test result summary.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            response: Response data if available
            error: Exception if test failed
        """
        status = "PASSED" if passed else "FAILED"
        self.logger.info(f"TEST: {test_name} - {status}")
        
        if error:
            self.logger.error(f"  Exception: {type(error).__name__}: {error}")
        
        if response is not None:
            trimmed, original_len = trim_response(response, self.max_chars)
            self.logger.debug(f"  Response ({original_len} chars):\n{trimmed}")
