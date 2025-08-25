"""
Crash-Safe Logging Module

This module provides decorators and utilities for crash-safe logging with separate process logging,
signal handling, heartbeat monitoring, and comprehensive error capture.
"""

from .crash_safe_logging import (
    crash_safe_log,
    setup_logging,
    list_log_files,
    get_latest_log_file,
    print_log_summary,
    CrashSafeLogger
)

from .config import (
    DEFAULT_CONFIG,
    ADVANCED_CONFIG,
    get_config,
    get_advanced_config,
    update_config,
    get_logs_directory
)

__version__ = "1.0.0"
__author__ = "SQW Project"

__all__ = [
    "crash_safe_log",
    "setup_logging", 
    "list_log_files",
    "get_latest_log_file",
    "print_log_summary",
    "CrashSafeLogger",
    "DEFAULT_CONFIG",
    "ADVANCED_CONFIG",
    "get_config",
    "get_advanced_config", 
    "update_config",
    "get_logs_directory"
]
