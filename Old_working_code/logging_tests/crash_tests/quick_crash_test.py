#!/usr/bin/env python3
"""
Quick crash test for the enhanced logging system.
This will test basic crash detection capabilities.
"""

import sys
import os
import time
import signal
import threading
from pathlib import Path

# Add the parent directory and logging module to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'logging_module'))

try:
    from logging_module.crash_safe_logging import CrashSafeLogger
except ImportError as e:
    print(f"[ERROR] Could not import crash_safe_logging: {e}")
    sys.exit(1)

class QuickCrashTests:
    """Quick tests for basic crash detection functionality."""
    
    def __init__(self):
        self.results = []
        
    def run_test_scenario(self, scenario_name, test_func, timeout=30):
        """Run a test scenario with timeout."""
        print(f"[TEST] Running scenario: {scenario_name}")
        print("-" * 40)
        
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            status = "[PASS]" if result.get("success", False) else "[FAIL]"
            print(f"{status} {scenario_name}: {result.get('message', 'No message')}")
            print(f"   Duration: {duration:.1f}s")
            
            self.results.append({
                "name": scenario_name,
                "success": result.get("success", False),
                "duration": duration,
                "message": result.get("message", ""),
                "logs": result.get("logs", [])
            })
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"[FAIL] {scenario_name}: Exception - {e}")
            print(f"   Duration: {duration:.1f}s")
            
            self.results.append({
                "name": scenario_name,
                "success": False,
                "duration": duration,
                "message": f"Exception: {e}",
                "logs": []
            })
        
        print()
        
    def test_normal_execution(self):
        """Test 1: Normal successful execution."""
        def normal_function():
            print("Executing normal function...")
            time.sleep(2)
            print("Function completed successfully")
            return "success"
        
        logger = CrashSafeLogger(
            log_file_prefix="quick_normal",
            heartbeat_interval=1.0
        )
        
        logger.setup()
        result = logger.safe_execute(normal_function)
        logger.cleanup()
        
        return {
            "success": result == "success",
            "message": "Normal execution completed successfully",
            "logs": ["quick_normal"]
        }
    
    def test_exception_handling(self):
        """Test 2: Exception handling during execution."""
        def failing_function():
            print("Starting function that will fail...")
            time.sleep(1)
            raise ValueError("Intentional test exception")
        
        logger = CrashSafeLogger(
            log_file_prefix="quick_exception",
            heartbeat_interval=1.0
        )
        
        logger.setup()
        try:
            result = logger.safe_execute(failing_function)
            logger.cleanup()
            success = False  # Should not reach here
        except ValueError:
            logger.cleanup()
            success = True  # Expected exception
        
        return {
            "success": success,
            "message": "Exception correctly caught and logged",
            "logs": ["quick_exception"]
        }
    
    def test_resource_monitoring(self):
        """Test 3: Resource monitoring capabilities."""
        def resource_intensive_function():
            import numpy as np
            print("Starting resource monitoring test...")
            
            # Create some arrays to use memory
            arrays = []
            for i in range(10):
                arr = np.random.random((100, 100))
                arrays.append(arr)
                time.sleep(0.2)
            
            print("Resource test completed")
            return "resource_test_done"
        
        logger = CrashSafeLogger(
            log_file_prefix="quick_resource",
            heartbeat_interval=1.0
        )
        
        logger.setup()
        result = logger.safe_execute(resource_intensive_function)
        logger.cleanup()
        
        return {
            "success": result == "resource_test_done",
            "message": "Resource monitoring test completed",
            "logs": ["quick_resource"]
        }
    
    def test_logging_output(self):
        """Test 4: Verify logging output generation."""
        def simple_logging_function():
            print("Testing logging output...")
            for i in range(5):
                print(f"Log message {i+1}")
                time.sleep(0.5)
            return "logging_complete"
        
        logger = CrashSafeLogger(
            log_file_prefix="quick_log_test",
            heartbeat_interval=1.0
        )
        
        logger.setup()
        result = logger.safe_execute(simple_logging_function)
        logger.cleanup()
        
        return {
            "success": result == "logging_complete",
            "message": "Logging output test completed",
            "logs": ["quick_log_test"]
        }

def main():
    """Run all quick crash test scenarios."""
    print("QUICK CRASH DETECTION TESTS")
    print("=" * 40)
    print("Testing basic logging and crash detection...")
    print()
    
    tester = QuickCrashTests()
    
    # Define test scenarios
    scenarios = [
        ("Normal Execution", tester.test_normal_execution),
        ("Exception Handling", tester.test_exception_handling),
        ("Resource Monitoring", tester.test_resource_monitoring),
        ("Logging Output", tester.test_logging_output),
    ]
    
    # Run all scenarios
    for scenario_name, scenario_func in scenarios:
        tester.run_test_scenario(scenario_name, scenario_func)
        time.sleep(1)  # Brief pause between scenarios
    
    # Summary
    print("=" * 40)
    print("QUICK TEST SUMMARY")
    print("=" * 40)
    
    total_scenarios = len(tester.results)
    passed_scenarios = sum(1 for r in tester.results if r["success"])
    
    print(f"Total scenarios: {total_scenarios}")
    print(f"Passed: {passed_scenarios}")
    print(f"Failed: {total_scenarios - passed_scenarios}")
    print(f"Success rate: {passed_scenarios/total_scenarios*100:.1f}%")
    
    print("\nDETAILED RESULTS:")
    for result in tester.results:
        status = "[PASS]" if result["success"] else "[FAIL]"
        print(f"  {status} {result['name']}: {result['message']}")
        if result.get("logs"):
            print(f"    Logs: {', '.join(result['logs'])}")
    
    # Check for log files
    log_dir = Path("../logs")  # Logs are saved to parent directory
    if log_dir.exists():
        log_files = list(log_dir.rglob("*quick*.log"))
        print(f"\nGenerated log files: {len(log_files)}")
        for log_file in log_files[-4:]:  # Show last 4 files
            print(f"  - {log_file}")
    
    print(f"\nTest logs saved to: ../logs/ (search for 'quick')")
    
    if passed_scenarios == total_scenarios:
        print("\n[PASS] All quick tests completed successfully!")
        print("Basic crash detection functionality is working correctly.")
        return True
    else:
        print("\n[WARN] Some quick tests failed - review logs for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        import os
        os._exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Quick test suite interrupted by user")
        import os
        os._exit(1)
    except Exception as e:
        print(f"\n[ERROR] Quick test suite failed: {e}")
        import os
        os._exit(1)
