# Cluster Process Failure Analysis and Fixes

## Problem Analysis

Your cluster job (N=20000, steps=5000, samples=5) stopped during the mean probability computation phase. Based on the logs:

### What Happened:
1. **Sample computation completed successfully** - All 5 processes finished in ~16 seconds
2. **Mean probability computation started** but the main process stopped logging
3. **Individual processes were still running** (as shown in process logs up to step 3501/5000)
4. **Likely cluster timeout or resource limit hit** - Most clusters have job time limits

### Root Cause:
The mean probability computation phase used `as_completed(future_to_dev)` **without a timeout**, while sample computation used `as_completed(future_to_dev, timeout=PROCESS_TIMEOUT)`. This meant:
- No timeout protection for the mean probability phase
- No proper error handling if processes got stuck or killed
- No graceful recovery mechanism
- Limited progress monitoring

## Fixes Implemented

### 1. **Added Proper Timeout Handling**
```python
# Separate timeout for mean probability (typically takes longer)
MEAN_PROB_TIMEOUT_MULTIPLIER = 2.0
MEAN_PROB_TIMEOUT = max(7200, int(PROCESS_TIMEOUT * MEAN_PROB_TIMEOUT_MULTIPLIER))

# Applied timeout to mean probability computation
for future in as_completed(future_to_dev, timeout=MEAN_PROB_TIMEOUT):
```

### 2. **Enhanced Logging and Monitoring**
- **Progress updates every 5 minutes** with ETA calculation
- **System resource monitoring** (memory, CPU usage)
- **More frequent step logging** in worker processes
- **Detailed timeout and error messages**
- **Timestamps on all major operations**

### 3. **Graceful Shutdown Handling**
```python
# Signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Cluster termination
signal.signal(signal.SIGHUP, signal_handler)   # Hangup signal

# Shutdown checks in critical loops
if SHUTDOWN_REQUESTED:
    log_and_print("[SHUTDOWN] Graceful shutdown requested, cancelling remaining processes...")
```

### 4. **Improved Error Recovery**
- **Timeout recovery**: Continue with completed processes if some timeout
- **Partial result handling**: Save progress even if not all processes complete
- **Better error messages** with specific timeout information
- **Resource usage warnings** if memory/CPU is high

### 5. **Better Resource Management**
- **Conservative process allocation** for large problems (N > 10000)
- **Memory monitoring** with warnings at 90%+ usage
- **CPU monitoring** with warnings at 95%+ usage
- **Timeout scaling** based on problem size

## For Cluster Usage

### Timeout Configuration:
- **Sample timeout**: 4.2 hours (for N=20000, steps=5000, samples=5)
- **Mean prob timeout**: 8.4 hours (2x sample timeout)
- **Total estimated time**: ~12-15 hours for full pipeline

### Recommended Cluster Settings:
```bash
# Example SLURM settings
#SBATCH --time=15:00:00          # 15 hours
#SBATCH --mem=32G                # 32GB memory
#SBATCH --cpus-per-task=10       # Adjust based on MAX_PROCESSES
```

### Environment Variables for Control:
```bash
export CALCULATE_SAMPLES_ONLY=true    # Only compute samples
export SKIP_SAMPLE_COMPUTATION=true   # Only do analysis
export ENABLE_PLOTTING=false          # Disable plotting on cluster
export CREATE_TAR_ARCHIVE=true        # Create archive of results
```

## Testing the Fixes

Run a small test first:
```python
# In the script, temporarily change:
N = 1000          # Smaller problem
steps = 100       # Fewer steps
samples = 2       # Fewer samples

# This should complete in ~5-10 minutes and test all the new logging/error handling
```

## Key Improvements for Debugging

1. **Process logs now show**:
   - System resource usage every 5 minutes
   - Step-by-step progress with timestamps
   - Memory and CPU warnings

2. **Main log now shows**:
   - Progress updates with ETA
   - Timeout information and recovery
   - Graceful shutdown messages

3. **Better error handling**:
   - Specific timeout vs error messages
   - Partial result recovery
   - Resource exhaustion detection

The improved version should be much more robust for cluster environments and provide clear information about what went wrong if issues occur.
