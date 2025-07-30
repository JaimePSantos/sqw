# Crash-Safe Logging Decorator Module

This module provides a comprehensive crash-safe logging system with organized directory structure, separate process logging, signal handling, heartbeat monitoring, and comprehensive error capture.

## Features:

### 🗂️ **Organized Directory Structure**
- **Main directory**: `logs/`
- **Date subdirectories**: `logs/YYYY-MM-DD/` (e.g., `logs/2025-07-30/`)
- **Time-stamped files**: `logs/YYYY-MM-DD/prefix_HH-MM-SS.log` (e.g., `logs/2025-07-30/sqw_execution_21-02-34.log`)
- **Readable 24-hour format**: Files use HH-MM-SS format for easy identification

### 🔧 **Separate Logging Process**
- The logging runs in a completely separate process, ensuring logs are written even if your main code crashes
- Uses multiprocessing queues for safe communication between processes

### 🛡️ **Signal Handling** 
- Catches Ctrl+C (SIGINT) and other termination signals
- Logs the exact signal received and the reason for termination
- Uses `atexit` handlers for additional termination detection

### ❤️ **Heartbeat Monitoring**
- Sends periodic "heartbeat" messages (configurable interval, default 10 seconds)
- Shows the process is alive and how long it's been running
- Helps identify if/when the process stops responding

### 📝 **Comprehensive Logging**
- **System info**: Python version, platform, process IDs, working directory
- **Execution flow**: Function start/completion messages
- **Errors**: Full tracebacks for all exceptions
- **Interruptions**: Keyboard interrupts, system exits, signals
- **Timestamps**: All messages include precise timestamps

### 💾 **Crash-Safe File Writing**
- Immediate flush to disk after each log message
- Continues logging even during crashes

## Usage Examples:

### 1. Basic Decorator Usage
```python
from crash_safe_logging import crash_safe_log

@crash_safe_log()
def my_function():
    # Your code here
    pass
```

### 2. Custom Settings
```python
@crash_safe_log(
    log_file_prefix="my_task",
    heartbeat_interval=5.0,
    log_level=logging.INFO
)
def my_custom_function():
    # Your code here
    pass
```

### 3. Manual Setup (Alternative to Decorator)
```python
from crash_safe_logging import setup_logging

logger, crash_logger = setup_logging(log_file_prefix="manual_task")
try:
    logger.info("Starting my process")
    # Your code here
finally:
    crash_logger.cleanup()
```

### 4. Log Management Utilities
```python
from crash_safe_logging import print_log_summary, get_latest_log_file, list_log_files

# Print summary of all recent log files
print_log_summary()

# Get path to most recent log file
latest = get_latest_log_file()

# Get log files from last 7 days
recent_logs = list_log_files(days_back=7)
```

## Directory Structure Example:

```
your_project/
├── logs/
│   ├── 2025-07-30/
│   │   ├── sqw_execution_09-15-23.log
│   │   ├── sqw_execution_14-30-45.log
│   │   └── test_run_21-02-34.log
│   ├── 2025-07-29/
│   │   ├── experiment_08-45-12.log
│   │   └── sqw_execution_16-22-18.log
│   └── 2025-07-28/
│       └── old_run_10-15-30.log
├── your_script.py
└── crash_safe_logging.py
```

## Log Levels:

- **INFO**: Normal execution flow, heartbeats, system information
- **WARNING**: Non-critical issues
- **ERROR**: Recoverable errors
- **CRITICAL**: Severe errors, crashes, interruptions

## Configuration Options:

- **log_file_prefix**: Prefix for log file names (default: "execution")
- **heartbeat_interval**: Seconds between heartbeat messages (default: 10.0)
- **log_level**: Minimum logging level (default: logging.DEBUG)
- **log_system_info**: Whether to log system info at startup (default: True)

## Benefits:

### ✅ **Easy Organization**
- Logs are automatically organized by date
- Easy to find logs from specific days
- Readable time format for quick identification

### ✅ **Crash Protection**
- Separate logging process continues even if main code crashes
- Captures Python-level interruptions and errors
- Records everything up to the point of failure

### ✅ **Zero Configuration**
- Works out of the box with sensible defaults
- Just add the decorator to any function
- Automatically creates directory structure

### ✅ **Comprehensive Monitoring**
- Heartbeat monitoring shows process health
- System information for debugging
- Complete execution timeline

## Notes:

- The system captures Python-level interruptions and errors
- Some low-level interruptions (like Fortran library aborts) may not be catchable at the Python level, but the logging system will still record everything up to that point
- Each run creates a new log file with timestamp, so you have a complete history of all executions
- Logs are organized by date for easy management and cleanup
- The directory structure is created automatically - no manual setup required
