# I/O Error Fix Summary

## Root Cause Identified

The process logs showed an **OSError: [Errno 5] Input/output error** occurring during the mean probability computation. This error was caused by **print statements failing** in a cluster environment.

### Error Details:
```
[2025-08-20 11:25:11,279] ERROR: Error in mean probability process for dev max0.000_min0.000: [Errno 5] Input/output error
Traceback (most recent call last):
  File "static_cluster_logged_mp.py", line 248, in log_system_resources
    print(msg)
OSError: [Errno 5] Input/output error
```

### Why This Happens in Cluster Environments:
1. **Stdout/stderr redirection issues** - Cluster schedulers often redirect or close these streams
2. **Process isolation** - Worker processes may not have access to terminal output
3. **Network filesystem problems** - When stdout is redirected to network-mounted files
4. **Resource limits** - Cluster systems may limit I/O operations
5. **Job termination** - Partial job termination can close output streams

## Problematic Code Patterns

### 1. **Resource Monitoring Function**
```python
# BEFORE (causing I/O errors):
def log_system_resources(logger=None, prefix="[SYSTEM]"):
    msg = f"{prefix} Memory: {memory.percent:.1f}% used..."
    if logger:
        logger.info(msg)
    print(msg)  # ❌ FAILS in cluster environments
```

### 2. **Progress Update Function**
```python
# BEFORE (causing I/O errors):
def log_progress_update(phase, completed, total, start_time, logger=None):
    msg = f"[{phase}] Progress: {completed}/{total}..."
    if logger:
        logger.info(msg)
    print(msg)  # ❌ FAILS in cluster environments
```

### 3. **Log-and-Print Helper Functions**
```python
# BEFORE (causing I/O errors):
def log_and_print(message, level="info"):
    print(message)  # ❌ FAILS in cluster environments
    if logger:
        logger.info(message)
```

## Fixes Applied

### 1. **Removed All Print Statements from Worker Functions**
```python
# AFTER (cluster-safe):
def log_system_resources(logger=None, prefix="[SYSTEM]"):
    msg = f"{prefix} Memory: {memory.percent:.1f}% used..."
    if logger:
        logger.info(msg)  # ✅ Only use logger, no print()
```

### 2. **Made Helper Functions Logger-Only**
```python
# AFTER (cluster-safe):
def log_and_print(message, level="info"):
    """Helper function to log messages (cluster-safe, no print)"""
    if logger:
        if level == "info":
            logger.info(message)  # ✅ Only use logger
```

### 3. **Enhanced Error Handling**
```python
# AFTER (robust error handling):
except Exception as e:
    msg = f"{prefix} Error monitoring resources: {e}"
    if logger:
        logger.error(msg)  # ✅ No print() that could fail
```

## Files Fixed

### Both versions updated:
- `static_cluster_logged_mp.py` - Primary cluster version
- `static_local_logged_mp.py` - Local development version

### Functions Modified:
1. **`log_system_resources()`** - Removed all print statements
2. **`log_progress_update()`** - Removed print statement
3. **`log_and_print()` (2 instances)** - Removed print statements, made logger-only

## Benefits

### 1. **Cluster Compatibility**
- ✅ No more I/O errors from print statements
- ✅ Works with stdout redirection
- ✅ Compatible with batch schedulers (SLURM, PBS, etc.)

### 2. **Robust Logging**
- ✅ All output goes to log files (persistent and reliable)
- ✅ Proper log levels (info, warning, error)
- ✅ Structured, searchable log format

### 3. **Better Error Recovery**
- ✅ Process doesn't crash on stdout/stderr issues
- ✅ Continues execution even if terminal is unavailable
- ✅ Complete execution logs preserved

### 4. **Production Ready**
- ✅ Suitable for long-running cluster jobs
- ✅ No dependency on interactive terminal
- ✅ Professional logging practices

## Key Insight

**The lesson**: In cluster/production environments, **never rely on print() statements** for critical logging. Always use proper logging frameworks that write to files, as stdout/stderr can be unreliable or unavailable.

## Testing

Both files now:
- ✅ Compile without syntax errors
- ✅ Use logger-only approach for all worker processes
- ✅ Maintain the same functionality without I/O vulnerabilities
- ✅ Are ready for cluster deployment

The cluster job should now complete successfully without I/O errors interrupting the mean probability computation phase.
