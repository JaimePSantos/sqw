# Logging Tests - Organized Structure

This directory contains a comprehensive suite of tests for the enhanced crash-safe logging system, organized into logical subdirectories.

## Directory Structure

```
logging_tests/
├── basic_tests/                    # Simple tests and examples
│   ├── simple_logging_test.py         # Quick smoke test
│   ├── quick_setup_test.py             # Setup validation
│   └── logging_examples.py             # Usage examples
├── crash_tests/                    # Crash detection scenarios
│   ├── realistic_angle_crash_test.py   # Realistic experiment crashes
│   ├── quick_crash_test.py             # Fast crash detection tests
│   ├── comprehensive_crash_tests.py    # Extended crash scenarios
│   ├── cluster_specific_crash_tests.py # Cluster environment tests
│   ├── run_all_crash_tests.py          # Test runner
│   └── cluster_diagnostics.sh          # System diagnostics script
├── setup_and_validation/           # Installation and validation
│   ├── final_validation.py             # Comprehensive validation
│   ├── test_enhanced_logging.py        # Enhanced feature tests
│   └── install_enhanced_logging.py     # Installation script
└── documentation/                  # Documentation and reports
    ├── README.md                       # Main documentation
    ├── CRASH_TESTS_README.md           # Crash test details
    ├── LOG_ANALYSIS_REPORT.md          # Analysis results
    └── CRASH_TEST_README2.md           # Additional documentation
```

## Quick Start

### 1. Basic Functionality Test (30 seconds)
```batch
# From root directory:
.\run_quick_tests.bat
```
This runs `crash_tests/quick_crash_test.py` which validates core functionality.

### 2. Realistic Crash Scenarios (2 minutes)
```batch
# From root directory:
.\run_crash_tests.bat
```
This runs `crash_tests/realistic_angle_crash_test.py` with actual experiment failures.

### 3. Simple Smoke Test
```batch
cd logging_tests\basic_tests
python simple_logging_test.py
```

## Test Categories

### 🔧 Basic Tests
**Purpose**: Validate basic logging functionality and provide usage examples.
- **simple_logging_test.py**: Minimal test that should complete in seconds
- **quick_setup_test.py**: Validates installation and setup
- **logging_examples.py**: Demonstrates various usage patterns

### 💥 Crash Tests
**Purpose**: Test crash detection and recovery in various failure scenarios.
- **realistic_angle_crash_test.py**: Simulates failures during quantum walk experiments
- **quick_crash_test.py**: Fast validation of crash detection
- **comprehensive_crash_tests.py**: Extended crash scenarios
- **cluster_specific_crash_tests.py**: Cluster environment specific tests

### ✅ Setup and Validation
**Purpose**: Installation, configuration, and comprehensive validation.
- **final_validation.py**: Complete system validation
- **test_enhanced_logging.py**: Tests enhanced features
- **install_enhanced_logging.py**: Automated installation

### 📚 Documentation
**Purpose**: Documentation, reports, and analysis results.
- Various README files with detailed documentation
- Analysis reports from test runs

## Running Tests by Category

### Basic Tests
```batch
cd logging_tests\basic_tests
python simple_logging_test.py      # Quick smoke test
python logging_examples.py         # Usage examples (may hang - use Ctrl+C)
```

### Crash Tests
```batch
cd logging_tests\crash_tests
python quick_crash_test.py                    # Fast crash detection
python realistic_angle_crash_test.py          # Realistic scenarios
python comprehensive_crash_tests.py           # Extended testing
python cluster_specific_crash_tests.py        # Cluster environments
```

### Validation Tests
```batch
cd logging_tests\setup_and_validation
python final_validation.py                    # Complete validation
python test_enhanced_logging.py               # Enhanced features
```

## Key Features Tested

✅ **Mid-experiment failure detection** - Catches errors between deviation runs  
✅ **Heartbeat monitoring** - Regular process alive confirmations  
✅ **Resource tracking** - Memory and CPU usage monitoring  
✅ **Cross-platform compatibility** - Windows/conda environment support  
✅ **Process isolation** - Each test runs in separate logging process  
✅ **Comprehensive error capture** - Full tracebacks with context  
✅ **Cluster environment detection** - SLURM, PBS, Torque support  
✅ **Signal handling** - SIGTERM, SIGINT, cluster-specific signals  

## Test Output

All tests generate detailed log files in the `../logs/YYYY-MM-DD/` directory:
- Real-time heartbeat monitoring
- Complete error tracebacks  
- Resource usage tracking
- Process state information
- Cluster environment details

## Requirements

- Python with numpy, networkx
- QWAK2 conda environment (recommended)
- psutil package (for enhanced monitoring)
- Windows or Linux environment

## Troubleshooting

1. **Import Errors**: Ensure you're running from the correct directory
2. **Module Not Found**: Check that logging_module is in the parent directory
3. **Test Hangs**: Use Ctrl+C to interrupt, check log files for details
4. **Conda Issues**: Use QWAK2 environment or update batch files for your setup

The enhanced logging system ensures your quantum walk experiments will never disappear silently on clusters again!
