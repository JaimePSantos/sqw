# Logging Module Tests

This folder contains tests and examples for the enhanced crash-safe logging module.

## Test Files

### Crash Detection Tests (New!)

#### `realistic_angle_crash_test.py` 🎯
**Purpose**: Realistic crash scenarios that inject failures during actual angle experiments.
- Tests 5 scenarios: Memory OOM, Import errors, Computation errors, Cluster timeouts, Successful completion
- Simulates failures occurring DURING experiment execution (between dev runs)
- Uses enhanced CrashSafeLogger with heartbeat monitoring
- **Status**: ✅ Working with QWAK2 environment
- **Use**: `.\run_crash_tests.bat` - Validates crash detection for quantum walk experiments

#### `quick_crash_test.py` ⚡
**Purpose**: Fast validation of core crash detection functionality.
- Tests 4 basic scenarios: Normal execution, Exception handling, Resource monitoring, Logging output
- Quick smoke test for logging system validation
- **Status**: ✅ Working - completes in ~30 seconds
- **Use**: `.\run_quick_tests.bat` - Quick verification of logging functionality

### Legacy Tests

#### `final_validation.py`
**Purpose**: Final comprehensive validation test that confirms the logging module works correctly.
- Tests basic functionality, loops, string operations, and list operations
- Uses `@crash_safe_log(log_file_prefix="final_validation")` decorator
- **Status**: ✅ Working - completes without hanging
- **Use**: Run this to verify the module is working properly

### `simple_logging_test.py`
**Purpose**: Simple, fast test for basic logging functionality.
- Minimal test that should complete quickly
- Uses `@crash_safe_log(log_file_prefix="simple_test", heartbeat_interval=2.0)` decorator
- **Status**: ✅ Working - good for quick verification
- **Use**: Run this for quick smoke tests

### `logging_examples.py`
**Purpose**: Comprehensive examples showing different ways to use the logging module.
- Examples of basic decorator usage
- Custom settings examples
- Error handling demonstration
- Manual setup examples
- **Status**: ⚠️ May hang - use with caution
- **Use**: Reference for learning how to use the module features

## Running Tests

### Quick Test
```bash
python logging_tests\simple_logging_test.py
```

### Full Validation
```bash
python logging_tests\final_validation.py
```

### View Examples (Reference Only)
```bash
# Be careful - this might hang
python logging_tests\logging_examples.py
```

## Expected Output

When tests run successfully, you should see:
1. Console output showing the test progress
2. Log files created in `logs/YYYY-MM-DD/` directory
3. Final message showing log file location
4. Success message

## Troubleshooting

If a test hangs:
1. Press `Ctrl+C` to interrupt
2. Check if log files were created (they should be, even if interrupted)
3. Use the simpler tests first (`simple_logging_test.py`)

## Log Files

All tests create log files in the organized directory structure:
```
logs/
└── YYYY-MM-DD/
    ├── final_validation_HH-MM-SS.log
    ├── simple_test_HH-MM-SS.log
    └── examples_HH-MM-SS.log
```

Each log file contains detailed execution information including system info, heartbeats, and any errors.
