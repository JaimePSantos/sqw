"""
Configuration settings for the crash-safe logging module.

This file contains default settings and configuration options for the logging system.
Users can modify these settings or override them when using the decorators.
"""

import logging
import os

# Default logging configuration
DEFAULT_CONFIG = {
    "log_file_prefix": "execution",
    "heartbeat_interval": 10.0,
    "log_level": logging.DEBUG,
    "log_system_info": True,
    "logs_base_directory": "logs",
    "date_format": "%Y-%m-%d",
    "time_format": "%H-%M-%S",
    "log_format": "%(asctime)s - %(levelname)s - %(message)s"
}

# Advanced configuration options
ADVANCED_CONFIG = {
    "queue_timeout": 0.5,
    "shutdown_timeout": 3.0,
    "heartbeat_error_retry_delay": 1.0,
    "log_flush_immediate": True,
    "signal_shutdown_delay": 0.2,
    "process_join_timeout": 3.0,
    "force_kill_timeout": 2.0
}

def get_config(key: str, default=None):
    """
    Get a configuration value with fallback to default.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return DEFAULT_CONFIG.get(key, default)

def get_advanced_config(key: str, default=None):
    """
    Get an advanced configuration value with fallback to default.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return ADVANCED_CONFIG.get(key, default)

def update_config(**kwargs):
    """
    Update default configuration values.
    
    Args:
        **kwargs: Configuration key-value pairs to update
    """
    DEFAULT_CONFIG.update(kwargs)

def get_logs_directory():
    """
    Get the base logs directory path.
    
    Returns:
        str: Path to the logs directory
    """
    return get_config("logs_base_directory", "logs")
