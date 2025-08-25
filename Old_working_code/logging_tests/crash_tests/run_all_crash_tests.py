#!/usr/bin/env python3
"""
Master test runner for all crash-safe logging tests.
Runs comprehensive tests and analyzes results across all test categories.
"""

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory and parent directories to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sqw_dir = os.path.dirname(parent_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, sqw_dir)

from logging_module import crash_safe_log
from logging_module.crash_safe_logging import check_for_crashed_processes, generate_cluster_diagnostic_script

class TestRunner:
    """Master test runner for all crash scenarios across organized test structure."""
    
    def __init__(self):
        self.test_logs_base = Path("logs_test_master")
        self.test_logs_base.mkdir(exist_ok=True)
        
        # Determine the correct base path for test files
        current_file_dir = Path(__file__).parent.absolute()
        logging_tests_dir = current_file_dir.parent
        
        # Test categories based on current directory structure
        self.test_categories = {
            "basic": {
                "path": logging_tests_dir / "basic_tests",
                "tests": ["simple_logging_test.py", "quick_setup_test.py", "integration_test.py"],
                "passed": 0, "failed": 0, "total": 0
            },
            "crash": {
                "path": current_file_dir,  # Current directory (crash_tests)
                "tests": ["comprehensive_crash_tests.py", "cluster_specific_crash_tests.py", 
                         "quick_crash_test.py", "realistic_angle_crash_test.py"],
                "passed": 0, "failed": 0, "total": 0
            },
            "validation": {
                "path": logging_tests_dir / "setup_and_validation", 
                "tests": ["final_validation.py", "test_enhanced_logging.py", "install_enhanced_logging.py"],
                "passed": 0, "failed": 0, "total": 0
            }
        }
        
        self.results = {
            "analysis": {"crashes_detected": 0, "deadman_triggers": 0}
        }
    
    def run_test_script(self, test_script_path, test_name):
        """Run a single test script and capture results."""
        print(f"\n[RUN] RUNNING TEST: {test_name}")
        print("-" * 40)
        
        try:
            # Run the test script
            result = subprocess.run([
                sys.executable, test_script_path
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout per test
            
            print(f"[DATA] {test_name} Output:")
            print("-" * 20)
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print(f"[WARN] {test_name} Errors:")
                print(result.stderr)
            
            # Consider test passed if it completed without error exit code
            success = result.returncode == 0
            print(f"[STATUS] {test_name}: {'PASS' if success else 'FAIL'} (exit code: {result.returncode})")
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {test_name} timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"[FAIL] Error running {test_name}: {e}")
            return False
    
    def run_test_category(self, category_name, skip_on_error=False):
        """Run all tests in a category."""
        category = self.test_categories[category_name]
        category_path = Path(category["path"])
        
        print(f"\n[CATEGORY] RUNNING {category_name.upper()} TESTS")
        print("=" * 60)
        print(f"[PATH] Looking for tests in: {category_path.absolute()}")
        
        passed = 0
        failed = 0
        total = 0
        
        for test_script in category["tests"]:
            test_path = category_path / test_script
            
            # Check if test file exists
            if not test_path.exists():
                print(f"[SKIP] Test file not found: {test_path}")
                print(f"       Absolute path: {test_path.absolute()}")
                continue
            
            total += 1
            test_name = f"{category_name}_{test_script.replace('.py', '')}"
            
            print(f"[FOUND] Running test: {test_path}")
            
            if self.run_test_script(str(test_path), test_name):
                passed += 1
            else:
                failed += 1
                if skip_on_error:
                    print(f"[ABORT] Stopping {category_name} tests due to failure")
                    break
            
            # Small delay between tests
            time.sleep(1)
        
        # Update category results
        category["passed"] = passed
        category["failed"] = failed
        category["total"] = total
        
        print(f"\n[SUMMARY] {category_name.upper()} CATEGORY RESULTS:")
        print(f"   Total: {total}, Passed: {passed}, Failed: {failed}")
        print(f"   Success Rate: {(passed/max(total,1))*100:.1f}%")
        
        return failed == 0
    
    def analyze_crash_evidence(self):
        """Analyze logs for crash evidence across all test directories."""
        print("\n[SEARCH] ANALYZING CRASH EVIDENCE")
        print("=" * 60)
        
        # Check for crashed processes in all test log directories
        test_dirs = [
            "logs_test",           # Basic crash tests
            "logs_test_cluster",   # Cluster tests  
            "logs_test_basic",     # Basic tests
            "logs_test_validation" # Validation tests
        ]
        total_crashes = 0
        deadman_triggers = 0
        
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                print(f"\n[FOLDER] Checking {test_dir}/...")
                
                # Look for deadman switch files
                deadman_files = list(Path(test_dir).rglob("process_alive.txt"))
                if deadman_files:
                    print(f"   Found {len(deadman_files)} deadman switch files")
                    deadman_triggers += len(deadman_files)
                
                # Look for log files with crash evidence
                log_files = list(Path(test_dir).rglob("*.log"))
                crash_indicators = 0
                
                for log_file in log_files:
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Look for crash indicators
                        indicators = [
                            "SIGNAL RECEIVED",
                            "KEYBOARD INTERRUPT", 
                            "SYSTEM EXIT",
                            "HIGH MEMORY USAGE",
                            "SYSTEM MEMORY CRITICAL",
                            "DEADMAN'S SWITCH TRIGGERED",
                            "OOM",
                            "CPU time limit exceeded",
                            "SIMULATED CRASH",
                            "FORCED TERMINATION",
                            "IMPORT ERROR SIMULATION",
                            "MEMORY ERROR SIMULATION"
                        ]
                        
                        found_indicators = []
                        for indicator in indicators:
                            if indicator in content:
                                found_indicators.append(indicator)
                        
                        if found_indicators:
                            crash_indicators += 1
                            print(f"   [FILE] {log_file.name}: {', '.join(found_indicators)}")
                    
                    except Exception as e:
                        print(f"   [WARN] Could not analyze {log_file.name}: {e}")
                
                total_crashes += crash_indicators
                print(f"   [DATA] {test_dir}: {crash_indicators} files with crash evidence")
        
        self.results["analysis"] = {
            "crashes_detected": total_crashes,
            "deadman_triggers": deadman_triggers
        }
        
        print(f"\n[DATA] CRASH ANALYSIS SUMMARY:")
        print(f"   Total crash evidence found: {total_crashes}")
        print(f"   Deadman switch triggers: {deadman_triggers}")
        
        return total_crashes, deadman_triggers
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive test report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_logs_base / f"test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("CRASH-SAFE LOGGING COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test category results
            total_tests = 0
            total_passed = 0
            
            for category_name, category in self.test_categories.items():
                f.write(f"{category_name.upper()} TESTS:\n")
                f.write(f"  Total: {category.get('total', 0)}\n")
                f.write(f"  Passed: {category.get('passed', 0)}\n")
                f.write(f"  Failed: {category.get('failed', 0)}\n")
                f.write(f"  Success Rate: {(category.get('passed', 0) / max(category.get('total', 1), 1)) * 100:.1f}%\n")
                f.write(f"  Tests Included: {', '.join(category['tests'])}\n\n")
                
                total_tests += category.get('total', 0)
                total_passed += category.get('passed', 0)
            
            # Analysis results
            analysis = self.results.get("analysis", {})
            f.write("CRASH EVIDENCE ANALYSIS:\n")
            f.write(f"  Crash evidence detected: {analysis.get('crashes_detected', 0)}\n")
            f.write(f"  Deadman switch triggers: {analysis.get('deadman_triggers', 0)}\n\n")
            
            # Overall assessment
            overall_success = (total_passed / max(total_tests, 1)) * 100
            
            f.write("OVERALL ASSESSMENT:\n")
            f.write(f"  Total tests executed: {total_tests}\n")
            f.write(f"  Total tests passed: {total_passed}\n")
            f.write(f"  Overall success rate: {overall_success:.1f}%\n")
            f.write(f"  Crash detection capability: {'PASS' if analysis.get('crashes_detected', 0) > 0 else 'FAIL'}\n")
            f.write(f"  Deadman switch functionality: {'PASS' if analysis.get('deadman_triggers', 0) > 0 else 'FAIL'}\n\n")
            
            # Category-specific recommendations
            f.write("CATEGORY-SPECIFIC RECOMMENDATIONS:\n")
            for category_name, category in self.test_categories.items():
                success_rate = (category.get('passed', 0) / max(category.get('total', 1), 1)) * 100
                if success_rate >= 80:
                    f.write(f"  [PASS] {category_name.capitalize()} tests: Working well\n")
                else:
                    f.write(f"  [WARN] {category_name.capitalize()} tests: {category.get('failed', 0)} failures need review\n")
            
            # Overall recommendations
            f.write("\nOVERALL RECOMMENDATIONS:\n")
            if overall_success >= 80:
                f.write("  [PASS] Crash-safe logging system is working well\n")
            else:
                f.write("  [WARN] Some tests failed - review implementation\n")
            
            if analysis.get('crashes_detected', 0) > 0:
                f.write("  [PASS] System successfully detects and logs crashes\n")
            else:
                f.write("  [WARN] Crash detection may need improvement\n")
            
            if analysis.get('deadman_triggers', 0) > 0:
                f.write("  [PASS] Deadman switch is functioning\n")
            else:
                f.write("  [WARN] Deadman switch may need verification\n")
            
            f.write("\nNEXT STEPS FOR CLUSTER DEPLOYMENT:\n")
            f.write("  1. Deploy enhanced logging to cluster environment\n")
            f.write("  2. Monitor logs during actual quantum walk experiments\n")
            f.write("  3. Use diagnostic tools to identify termination causes\n")
            f.write("  4. Implement checkpointing based on signal patterns\n")
            
            f.write("\nTEST DIRECTORY STRUCTURE:\n")
            f.write("  basic_tests/     - Basic functionality validation\n")
            f.write("  crash_tests/     - Comprehensive crash simulation\n")
            f.write("  setup_and_validation/ - Installation and validation\n")
            f.write("  documentation/   - README files and reports\n")
        
        print(f"\n[FILE] Comprehensive report saved: {report_file}")
        return report_file

@crash_safe_log(
    log_file_prefix="test_runner_master",
    heartbeat_interval=10.0,
    log_system_info=True
)
def run_all_tests(args):
    """Main test execution function."""
    runner = TestRunner()
    
    print("[TEST] CRASH-SAFE LOGGING MASTER TEST SUITE")
    print("=" * 60)
    print("This will run comprehensive tests of all crash scenarios")
    print("that could cause your quantum walk experiments to disappear.")
    print("\nOrganized Test Structure:")
    print("  [FOLDER] basic_tests/     - Basic functionality validation")
    print("  [FOLDER] crash_tests/     - Comprehensive crash simulation") 
    print("  [FOLDER] setup_and_validation/ - Installation and validation")
    print("  [FOLDER] documentation/   - README files and reports\n")
    
    # Display current working directory and expected paths
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent.absolute()
    print(f"[INFO] Current working directory: {current_dir}")
    print(f"[INFO] Script location: {script_dir}")
    print(f"[INFO] Test runner can be run from any directory - using absolute paths\n")
    
    success = True
    
    # Run basic tests first
    if not args.skip_basic:
        print("Phase 1: Running basic functionality tests...")
        if not runner.run_test_category("basic", skip_on_error=args.fail_fast):
            success = False
        time.sleep(2)
    
    # Run comprehensive crash tests
    if not args.skip_crash:
        print("Phase 2: Running comprehensive crash tests...")
        if not runner.run_test_category("crash", skip_on_error=args.fail_fast):
            success = False
        time.sleep(2)
    
    # Run validation tests
    if not args.skip_validation:
        print("Phase 3: Running setup and validation tests...")
        if not runner.run_test_category("validation", skip_on_error=args.fail_fast):
            success = False
        time.sleep(2)
    
    # Analyze crash evidence
    if not args.skip_analysis:
        print("Phase 4: Analyzing crash evidence...")
        crashes, deadman = runner.analyze_crash_evidence()
        time.sleep(1)
    
    # Generate diagnostic script
    if not args.skip_diagnostics:
        print("Phase 5: Generating cluster diagnostic tools...")
        generate_cluster_diagnostic_script("cluster_diagnostics.sh")
        print("[PASS] Cluster diagnostic script generated")
    
    # Generate comprehensive report
    report_file = runner.generate_comprehensive_report()
    
    # Final summary
    print("\n" + "=" * 60)
    print("[FINISH] MASTER TEST SUITE COMPLETE")
    print("=" * 60)
    
    # Calculate totals across all categories
    total_tests = sum(cat["total"] for cat in runner.test_categories.values())
    total_passed = sum(cat["passed"] for cat in runner.test_categories.values())
    analysis = runner.results.get("analysis", {})
    
    print(f"[DATA] FINAL RESULTS:")
    for category_name, category in runner.test_categories.items():
        print(f"   {category_name.capitalize()} tests: {category['passed']}/{category['total']} passed")
    print(f"   Overall: {total_passed}/{total_tests} ({(total_passed/max(total_tests,1))*100:.1f}%)")
    print(f"   Crash evidence: {analysis.get('crashes_detected', 0)} instances")
    print(f"   Deadman triggers: {analysis.get('deadman_triggers', 0)} instances")
    
    print(f"\n[FOLDER] TEST ARTIFACTS:")
    print(f"   Test logs: logs_test_*, logs_test_basic/, logs_test_validation/")
    print(f"   Master logs: {runner.test_logs_base}/")
    print(f"   Report: {report_file}")
    print(f"   Diagnostics: cluster_diagnostics.sh")
    
    print(f"\n[TARGET] NEXT STEPS:")
    print("   1. Review the comprehensive report")
    print("   2. Check specific test logs for details")
    print("   3. Run cluster diagnostics on your actual cluster")
    print("   4. Deploy enhanced logging to your quantum walk experiments")
    
    if success and total_passed >= total_tests * 0.8:  # 80% success threshold
        print("\n[SUCCESS] Crash-safe logging system is ready for cluster deployment!")
    else:
        print("\n[WARN] WARNING: Some tests failed - review before cluster deployment")
    
    return success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Master test runner for crash-safe logging")
    parser.add_argument("--skip-basic", action="store_true", 
                       help="Skip basic functionality tests")
    parser.add_argument("--skip-crash", action="store_true",
                       help="Skip comprehensive crash tests")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip setup and validation tests")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip crash evidence analysis")
    parser.add_argument("--skip-diagnostics", action="store_true",
                       help="Skip diagnostic script generation")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop category testing on first failure")
    parser.add_argument("--quick", action="store_true",
                       help="Run only essential tests (basic + validation)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.skip_crash = True  # Skip the time-consuming crash tests
        print("[RUN] Quick mode: Running basic and validation tests only...")
    
    try:
        success = run_all_tests(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[WARN] Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
