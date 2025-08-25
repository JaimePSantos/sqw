#!/usr/bin/env python3
"""
Comprehensive test suite for crash-safe logging.
Tests all possible termination scenarios that could cause processes to disappear.
"""

import sys
import os
import time
import signal
import threading
import gc
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from logging_module import crash_safe_log
from logging_module.config import update_config

# Update config to use test log directory
update_config(logs_base_directory="logs_test")

class CrashTestSuite:
    """Test suite for various crash scenarios."""
    
    def __init__(self):
        self.test_results = []
    
    def log_test_result(self, test_name, result, details=""):
        """Log test results."""
        self.test_results.append({
            "test": test_name,
            "result": result,
            "details": details
        })
        print(f"{'[PASS]' if result == 'PASS' else '[FAIL]'} {test_name}: {result}")
        if details:
            print(f"   Details: {details}")

def test_1_normal_termination():
    """Test 1: Normal successful completion."""
    @crash_safe_log(
        log_file_prefix="test_1_normal",
        heartbeat_interval=2.0,
        log_system_info=True
    )
    def normal_execution():
        print("Test 1: Normal execution")
        for i in range(5):
            print(f"  Working... {i+1}/5")
            time.sleep(1)
        return "completed_normally"
    
    try:
        result = normal_execution()
        return "PASS", f"Returned: {result}"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_2_keyboard_interrupt():
    """Test 2: Keyboard interrupt (Ctrl+C) simulation."""
    @crash_safe_log(
        log_file_prefix="test_2_keyboard",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def keyboard_interrupt_test():
        print("Test 2: Keyboard interrupt simulation")
        print("  Simulating Ctrl+C after 3 seconds...")
        
        def send_interrupt():
            time.sleep(3)
            print("  Sending SIGINT...")
            os.kill(os.getpid(), signal.SIGINT)
        
        # Start interrupt thread
        interrupt_thread = threading.Thread(target=send_interrupt, daemon=True)
        interrupt_thread.start()
        
        # Long running task
        for i in range(10):
            print(f"  Working... {i+1}/10")
            time.sleep(1)
        
        return "should_not_reach_here"
    
    try:
        result = keyboard_interrupt_test()
        return "FAIL", f"Should have been interrupted but returned: {result}"
    except KeyboardInterrupt:
        return "PASS", "Caught KeyboardInterrupt correctly"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_3_sigterm_termination():
    """Test 3: SIGTERM signal (cluster job termination)."""
    @crash_safe_log(
        log_file_prefix="test_3_sigterm",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def sigterm_test():
        print("Test 3: SIGTERM simulation")
        print("  Simulating cluster job termination after 3 seconds...")
        
        def send_sigterm():
            time.sleep(3)
            print("  Sending SIGTERM...")
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Start termination thread
        term_thread = threading.Thread(target=send_sigterm, daemon=True)
        term_thread.start()
        
        # Long running task
        for i in range(10):
            print(f"  Working... {i+1}/10")
            time.sleep(1)
        
        return "should_not_reach_here"
    
    try:
        result = sigterm_test()
        return "FAIL", f"Should have been terminated but returned: {result}"
    except SystemExit as e:
        return "PASS", f"Caught SystemExit with code: {e.code}"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_4_memory_exhaustion():
    """Test 4: Memory exhaustion leading to potential OOM kill."""
    @crash_safe_log(
        log_file_prefix="test_4_memory",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def memory_exhaustion_test():
        print("Test 4: Memory exhaustion simulation")
        print("  Gradually allocating memory to trigger warnings...")
        
        data = []
        max_iterations = 100  # Limit to prevent actual system crash
        
        for i in range(max_iterations):
            try:
                # Allocate ~50MB per iteration
                chunk = [0] * (50 * 1024 * 1024 // 8)  # 50MB of integers
                data.append(chunk)
                
                allocated_mb = (i + 1) * 50
                print(f"  Allocated {allocated_mb} MB")
                
                # Stop if we reach 2GB to prevent actual system issues
                if allocated_mb >= 2000:
                    print(f"  Stopping at {allocated_mb} MB to prevent system crash")
                    break
                    
                time.sleep(0.5)
                
            except MemoryError:
                print(f"  MemoryError at {(i+1)*50} MB")
                break
        
        # Clean up
        del data
        gc.collect()
        return "memory_test_completed"
    
    try:
        result = memory_exhaustion_test()
        return "PASS", f"Completed: {result}"
    except MemoryError:
        return "PASS", "Caught MemoryError as expected"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_5_cpu_intensive():
    """Test 5: CPU-intensive task that might hit CPU limits."""
    @crash_safe_log(
        log_file_prefix="test_5_cpu",
        heartbeat_interval=2.0,
        log_system_info=True
    )
    def cpu_intensive_test():
        print("Test 5: CPU-intensive computation")
        print("  Running CPU-intensive calculation...")
        
        # CPU-intensive calculation
        result = 0
        iterations = 10000000  # 10 million iterations
        
        for i in range(iterations):
            result += i * i
            
            # Progress reporting
            if i % 1000000 == 0:
                progress = (i / iterations) * 100
                print(f"  Progress: {progress:.1f}%")
        
        return f"cpu_test_completed_result_{result}"
    
    try:
        result = cpu_intensive_test()
        return "PASS", f"Completed: CPU test finished"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_6_exception_handling():
    """Test 6: Unhandled exception scenarios."""
    @crash_safe_log(
        log_file_prefix="test_6_exception",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def exception_test():
        print("Test 6: Exception handling")
        print("  Creating various exceptions...")
        
        # Test different types of exceptions
        test_cases = [
            ("division_by_zero", lambda: 1/0),
            ("index_error", lambda: [1,2,3][10]),
            ("key_error", lambda: {"a": 1}["b"]),
            ("type_error", lambda: "string" + 5),
            ("attribute_error", lambda: "string".nonexistent_method()),
        ]
        
        for test_name, test_func in test_cases:
            try:
                print(f"  Testing {test_name}...")
                test_func()
            except Exception as e:
                print(f"    Caught {type(e).__name__}: {e}")
        
        # Final exception that should propagate
        print("  Triggering final exception...")
        raise ValueError("Intentional test exception")
    
    try:
        result = exception_test()
        return "FAIL", "Should have raised ValueError"
    except ValueError as e:
        return "PASS", f"Caught expected ValueError: {e}"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_7_file_system_stress():
    """Test 7: File system stress that might hit disk limits."""
    @crash_safe_log(
        log_file_prefix="test_7_filesystem",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def filesystem_stress_test():
        print("Test 7: File system stress test")
        print("  Creating and writing multiple files...")
        
        test_dir = Path("test_filesystem_stress")
        test_dir.mkdir(exist_ok=True)
        
        try:
            for i in range(50):  # Create 50 files
                file_path = test_dir / f"test_file_{i}.txt"
                
                # Write ~1MB per file
                with open(file_path, 'w') as f:
                    content = "x" * (1024 * 1024)  # 1MB of 'x'
                    f.write(content)
                
                if i % 10 == 0:
                    print(f"  Created {i+1} files")
                
                time.sleep(0.1)
            
            # Clean up
            for file_path in test_dir.glob("test_file_*.txt"):
                file_path.unlink()
            test_dir.rmdir()
            
            return "filesystem_test_completed"
            
        except Exception as e:
            # Clean up on error
            try:
                for file_path in test_dir.glob("test_file_*.txt"):
                    file_path.unlink()
                test_dir.rmdir()
            except:
                pass
            raise e
    
    try:
        result = filesystem_stress_test()
        return "PASS", f"Completed: {result}"
    except Exception as e:
        return "FAIL", f"Filesystem error: {e}"

def test_8_subprocess_termination():
    """Test 8: Subprocess that gets terminated externally."""
    @crash_safe_log(
        log_file_prefix="test_8_subprocess",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def subprocess_test():
        print("Test 8: Subprocess termination simulation")
        print("  Starting subprocess that will be terminated...")
        
        # Create a script that will be terminated
        script_content = '''
import time
import sys

print("Subprocess started")
for i in range(60):
    print(f"Subprocess working... {i+1}/60")
    time.sleep(1)
print("Subprocess completed")
'''
        
        script_path = Path("temp_subprocess_test.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        try:
            # Start subprocess
            process = subprocess.Popen([sys.executable, str(script_path)])
            print(f"  Started subprocess PID: {process.pid}")
            
            # Let it run for a few seconds then terminate
            time.sleep(5)
            print("  Terminating subprocess...")
            process.terminate()
            
            # Wait for termination
            return_code = process.wait(timeout=5)
            print(f"  Subprocess terminated with code: {return_code}")
            
            return "subprocess_test_completed"
            
        finally:
            # Clean up
            if script_path.exists():
                script_path.unlink()
    
    try:
        result = subprocess_test()
        return "PASS", f"Completed: {result}"
    except Exception as e:
        return "FAIL", f"Subprocess error: {e}"

def test_9_resource_limits():
    """Test 9: Testing resource limit scenarios."""
    @crash_safe_log(
        log_file_prefix="test_9_resources",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def resource_limits_test():
        print("Test 9: Resource limits test")
        print("  Testing various resource scenarios...")
        
        # Try to import resource module (not available on Windows)
        try:
            import resource
            RESOURCE_AVAILABLE = True
        except ImportError:
            RESOURCE_AVAILABLE = False
            print("  Resource module not available on this platform (Windows)")
        
        if RESOURCE_AVAILABLE:
            # Log current limits
            limits = [
                ("RLIMIT_AS", resource.RLIMIT_AS),
                ("RLIMIT_CPU", resource.RLIMIT_CPU),
                ("RLIMIT_FSIZE", resource.RLIMIT_FSIZE),
                ("RLIMIT_NPROC", resource.RLIMIT_NPROC),
            ]
            
            for name, limit_type in limits:
                try:
                    soft, hard = resource.getrlimit(limit_type)
                    soft_str = "unlimited" if soft == resource.RLIM_INFINITY else str(soft)
                    hard_str = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
                    print(f"  {name}: soft={soft_str}, hard={hard_str}")
                except (OSError, AttributeError):
                    print(f"  {name}: not available on this system")
        else:
            print("  Simulating resource limits check on Windows...")
        
        # Test creating many threads (but not too many)
        threads = []
        try:
            for i in range(10):
                def worker():
                    time.sleep(2)
                
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)
                print(f"  Created thread {i+1}/10")
                time.sleep(0.1)
            
            # Wait for all threads
            for thread in threads:
                thread.join()
                
        except Exception as e:
            print(f"  Thread creation error: {e}")
        
        return "resource_limits_test_completed"
    
    try:
        result = resource_limits_test()
        return "PASS", f"Completed: {result}"
    except Exception as e:
        return "FAIL", f"Resource error: {e}"

def test_10_deadman_switch():
    """Test 10: Test deadman's switch detection."""
    @crash_safe_log(
        log_file_prefix="test_10_deadman",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def deadman_switch_test():
        print("Test 10: Deadman's switch test")
        print("  Running for a short time to test deadman's switch...")
        
        for i in range(5):
            print(f"  Working... {i+1}/5")
            time.sleep(1)
        
        print("  Creating artificial deadman file for testing...")
        
        # Create a test deadman file as if from a crashed process
        from datetime import datetime
        logs_dir = Path("logs_test") / datetime.now().strftime("%Y-%m-%d")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        deadman_file = logs_dir / "process_alive.txt"
        with open(deadman_file, 'w') as f:
            # Write timestamp from 5 minutes ago and fake PID
            old_timestamp = time.time() - 300  # 5 minutes ago
            f.write(f"{old_timestamp}\n99999\n")  # Fake PID that doesn't exist
        
        print(f"  Created test deadman file: {deadman_file}")
        
        return "deadman_switch_test_completed"
    
    try:
        result = deadman_switch_test()
        return "PASS", f"Completed: {result}"
    except Exception as e:
        return "FAIL", f"Deadman error: {e}"

def main():
    """Run all crash tests."""
    print("COMPREHENSIVE CRASH-SAFE LOGGING TEST SUITE")
    print("=" * 60)
    print("Testing all possible termination scenarios...\n")
    
    test_suite = CrashTestSuite()
    
    # Define all tests
    tests = [
        ("Normal Termination", test_1_normal_termination),
        ("Keyboard Interrupt (Ctrl+C)", test_2_keyboard_interrupt),
        ("SIGTERM Signal", test_3_sigterm_termination),
        ("Memory Exhaustion", test_4_memory_exhaustion),
        ("CPU Intensive", test_5_cpu_intensive),
        ("Exception Handling", test_6_exception_handling),
        ("File System Stress", test_7_file_system_stress),
        ("Subprocess Termination", test_8_subprocess_termination),
        ("Resource Limits", test_9_resource_limits),
        ("Deadman's Switch", test_10_deadman_switch),
    ]
    
    # Run each test
    for test_name, test_func in tests:
        print(f"\n[SCOPE] Running: {test_name}")
        print("-" * 40)
        
        try:
            result, details = test_func()
            test_suite.log_test_result(test_name, result, details)
        except Exception as e:
            test_suite.log_test_result(test_name, "ERROR", f"Test framework error: {e}")
        
        print("Waiting 2 seconds before next test...")
        time.sleep(2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("[FINISH] TEST SUITE SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in test_suite.test_results if r["result"] == "PASS")
    failed = sum(1 for r in test_suite.test_results if r["result"] in ["FAIL", "ERROR"])
    total = len(test_suite.test_results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    print("\n📋 DETAILED RESULTS:")
    for result in test_suite.test_results:
        status = "[PASS]" if result["result"] == "PASS" else "[FAIL]"
        print(f"{status} {result['test']}: {result['result']}")
        if result["details"]:
            print(f"   {result['details']}")
    
    print(f"\n[FOLDER] Check logs in: logs_test/ directory")
    print("[SEARCH] Run crash detection: python -m logging_module.crash_safe_logging --check-crashes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
