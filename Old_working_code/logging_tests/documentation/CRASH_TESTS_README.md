# Crash Tests for Enhanced Logging System

This directory contains comprehensive crash detection tests for quantum walk experiments.

## Files

### Test Scripts
- `realistic_angle_crash_test.py` - Realistic crash scenarios that inject failures during actual angle experiments
- `quick_crash_test.py` - Quick basic tests for logging functionality validation

### Logging Module
- `logging_module/` - Enhanced crash-safe logging system with:
  - `crash_safe_logging.py` - Main logging class with heartbeat monitoring
  - `config.py` - Configuration settings
  - `__init__.py` - Module initialization

## Running Tests

### Quick Tests (Recommended first)
```batch
# From parent directory:
.\run_quick_tests.bat
```
Runs 4 basic tests in ~30 seconds to validate core functionality.

### Realistic Crash Tests  
```batch
# From parent directory:
.\run_crash_tests.bat
```
Runs 5 realistic scenarios that inject failures during actual quantum walk experiments.

## Test Scenarios

### Quick Tests
1. **Normal Execution** - Validates successful execution logging
2. **Exception Handling** - Tests error capture and logging
3. **Resource Monitoring** - Checks memory/CPU monitoring
4. **Logging Output** - Verifies log file generation

### Realistic Tests
1. **Memory OOM During Dev=2** - Simulates memory crash during deviation processing
2. **Import Error Mid-Experiment** - Tests module import failures during execution
3. **Computation Error During Walk** - Matrix dimension errors during quantum walk
4. **Cluster Timeout Mid-Experiment** - Simulates cluster job termination
5. **Simplified Successful Experiment** - Complete successful experiment

## Output

All tests generate detailed log files in the `../logs/YYYY-MM-DD/` directory with:
- Real-time heartbeat monitoring
- Complete error tracebacks  
- Resource usage tracking
- Process state information

## Key Features Tested

✅ **Mid-experiment failure detection** - Catches errors between deviation runs  
✅ **Heartbeat monitoring** - Regular process alive confirmations  
✅ **Resource tracking** - Memory and CPU usage monitoring  
✅ **Cross-platform compatibility** - Windows/conda environment support  
✅ **Process isolation** - Each test runs in separate logging process  
✅ **Comprehensive error capture** - Full tracebacks with context  

## Requirements

- Python with numpy, networkx
- QWAK2 conda environment (recommended)
- psutil package (for enhanced monitoring)

The enhanced logging system ensures your quantum walk experiments will never disappear silently on clusters again!
