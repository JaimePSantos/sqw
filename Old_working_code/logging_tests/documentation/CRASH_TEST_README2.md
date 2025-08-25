# Comprehensive Crash Test Suite for Enhanced Logging

This test suite provides exhaustive testing of all possible scenarios that could cause your quantum walk experiments to suddenly disappear on cluster systems.

## 🎯 Test Coverage

### Comprehensive Crash Tests (`comprehensive_crash_tests.py`)
Tests fundamental termination scenarios:

1. **Normal Termination** - Baseline successful completion
2. **Keyboard Interrupt (Ctrl+C)** - Manual interruption
3. **SIGTERM Signal** - Standard termination request
4. **Memory Exhaustion** - Tests leading to potential OOM kills
5. **CPU Intensive** - High CPU load scenarios
6. **Exception Handling** - Various unhandled exceptions
7. **File System Stress** - Disk space/I/O limit scenarios
8. **Subprocess Termination** - External process kills
9. **Resource Limits** - Testing various system limits
10. **Deadman's Switch** - Sudden termination detection

### Cluster-Specific Tests (`cluster_specific_crash_tests.py`)
Tests cluster environment scenarios:

1. **SLURM Job Timeout** - Job scheduler time limits with SIGUSR1 warning
2. **PBS Job Termination** - PBS/Torque job manager termination
3. **OOM Kill Simulation** - Kernel memory killer scenarios
4. **Network/Storage Failure** - Shared filesystem interruptions
5. **Compute Node Failure** - Hardware failure simulation
6. **CPU Time Limit Exceeded** - SIGXCPU resource limit violations
7. **Checkpointing on Warning** - Graceful shutdown with state saving
8. **Resource Contention** - Multi-user cluster competition

## 🚀 Quick Start

### 1. Setup and Verify Installation
```bash
python quick_setup_test.py
```
This will:
- Check Python version and dependencies
- Install psutil if needed
- Verify enhanced logging module
- Run a quick functionality test

### 2. Run Individual Test Suites

**Comprehensive Tests:**
```bash
python comprehensive_crash_tests.py
```

**Cluster-Specific Tests:**
```bash
python cluster_specific_crash_tests.py
```

### 3. Run All Tests
```bash
# Full test suite
python run_all_crash_tests.py

# Quick essential tests only
python run_all_crash_tests.py --quick

# Skip specific test categories
python run_all_crash_tests.py --skip-cluster
python run_all_crash_tests.py --skip-comprehensive
```

## 📊 Understanding Test Results

### Success Indicators
- **✅ PASS** - Test executed and behaved as expected
- **Normal termination** - Function completed successfully
- **Signal caught** - Termination signal was properly logged
- **Exception handled** - Error was caught and logged appropriately

### Log Analysis
After running tests, check the generated logs:

```bash
# Check for evidence of crashed processes
python -m logging_module.crash_safe_logging --check-crashes

# Generate cluster diagnostic script
python -m logging_module.crash_safe_logging --generate-diagnostics
```

### Log Directories
- `logs_test/` - Comprehensive test logs
- `logs_test_cluster/` - Cluster-specific test logs  
- `logs_test_master/` - Master test runner logs
- `logs_quick_test/` - Quick setup test logs

## 🔍 Interpreting Results

### What Each Test Validates

**Memory Exhaustion Test:**
- ✅ **Expected**: Gradual memory allocation with warnings logged
- ✅ **Good**: "HIGH MEMORY USAGE" warnings in logs
- ❌ **Concerning**: Process disappears without memory warnings

**Signal Tests (SIGTERM, SIGINT):**
- ✅ **Expected**: "SIGNAL RECEIVED" messages in logs
- ✅ **Good**: Proper cleanup and shutdown sequence
- ❌ **Concerning**: No signal logs but process terminates

**OOM Kill Simulation:**
- ✅ **Expected**: Resource monitoring shows memory pressure
- ✅ **Good**: Deadman's switch detects sudden termination
- ❌ **Concerning**: No termination evidence at all

**Cluster Job Tests:**
- ✅ **Expected**: Cluster environment variables logged
- ✅ **Good**: Warning signals (SIGUSR1) caught before termination
- ❌ **Concerning**: Job disappears without cluster context

### Red Flags in Your Actual Experiments

Look for these patterns in your quantum walk experiment logs:

1. **Sudden Silence Pattern:**
   ```
   HEARTBEAT #67 - Process alive
   HIGH MEMORY USAGE: 1847.3 MB
   [No further logs] ← OOM kill likely
   ```

2. **Resource Exhaustion Pattern:**
   ```
   SYSTEM MEMORY CRITICAL: 94.2% used
   Available memory: 45.2 MB
   [Process disappears] ← Kernel intervention
   ```

3. **Cluster Timeout Pattern:**
   ```
   SLURM_JOB_ID: 123456
   SLURM_TIMELIMIT: 120
   [Logs stop at ~119 minutes] ← Job time limit
   ```

4. **Hardware Failure Pattern:**
   ```
   HEARTBEAT #45 - Process alive
   DEADMAN'S SWITCH TRIGGERED ← Sudden hardware failure
   ```

## 🛠️ Diagnostic Tools Generated

### Cluster Diagnostic Script
After running tests, you'll have `cluster_diagnostics.sh`:

```bash
# Run on your cluster to gather system information
bash cluster_diagnostics.sh > my_cluster_info.txt
```

This script gathers:
- Cluster environment variables (SLURM, PBS, etc.)
- Resource limits (`ulimit -a`)
- Memory and CPU information
- System logs for OOM kills
- Disk space and load information

### Crash Analysis
```bash
# Analyze test results for crash patterns
python -m logging_module.crash_safe_logging --check-crashes
```

This will:
- Find orphaned deadman switch files
- Analyze log patterns for termination causes
- Identify potential OOM kills, resource limits, etc.
- Provide specific recommendations

## 🎯 Applying Results to Your Quantum Walk Experiments

### 1. Deploy Enhanced Logging
```python
from logging_module import crash_safe_log

@crash_safe_log(
    log_file_prefix="quantum_walk_experiment",
    heartbeat_interval=30.0,  # Cluster-appropriate interval
    log_system_info=True      # Log cluster environment
)
def run_quantum_walk():
    # Your existing quantum walk code
    pass
```

### 2. Monitor for Patterns
Based on test results, watch for:
- Memory usage trends approaching system limits
- CPU time approaching cluster job limits
- Cluster environment variables indicating job constraints
- Deadman switch triggers indicating sudden termination

### 3. Implement Preventive Measures
```python
@crash_safe_log(heartbeat_interval=10.0)
def quantum_walk_with_checkpointing():
    # Set up SIGUSR1 handler for cluster warnings
    import signal
    
    def checkpoint_handler(signum, frame):
        logger.warning("Cluster warning signal - saving checkpoint")
        save_checkpoint()
        
    signal.signal(signal.SIGUSR1, checkpoint_handler)
    
    # Your computation with periodic checkpointing
    for step in range(total_steps):
        run_quantum_walk_step(step)
        
        if step % checkpoint_interval == 0:
            save_checkpoint()
```

## 📋 Test Checklist

Before deploying to your cluster, ensure:

- [ ] ✅ Quick setup test passes
- [ ] ✅ Comprehensive tests show expected signal handling
- [ ] ✅ Memory exhaustion test shows proper warnings
- [ ] ✅ Cluster tests demonstrate environment detection
- [ ] ✅ Deadman's switch detects sudden termination
- [ ] ✅ Crash analysis tool works on test data
- [ ] ✅ Cluster diagnostic script generated successfully

## 🔧 Troubleshooting

### Tests Fail to Run
```bash
# Check Python and dependencies
python quick_setup_test.py

# Install missing dependencies
pip install psutil
```

### No Crash Evidence Detected
This might indicate:
- Tests are working too well (good!)
- Signal handlers are preventing crashes (expected)
- Need to check specific test log files manually

### Cluster Tests Don't Reflect Your Environment
- Modify cluster environment variables in test scripts
- Add your specific cluster's signal patterns
- Customize resource limits to match your cluster

### Memory Tests Cause System Issues
The tests are designed with safety limits, but if you experience issues:
- Reduce memory allocation in `test_4_memory_exhaustion()`
- Lower the maximum iterations in memory stress tests
- Run tests on a dedicated test system

## 📞 Support

For issues specific to your cluster environment:
1. Run the generated `cluster_diagnostics.sh` script
2. Compare your cluster's signals/limits with test scenarios
3. Customize test parameters to match your environment
4. Check the comprehensive test report for specific recommendations

The test suite is designed to be comprehensive but safe - it should not cause system damage while thoroughly exercising all termination scenarios you might encounter in production.
