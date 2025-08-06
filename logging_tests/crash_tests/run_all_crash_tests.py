#!/usr/bin/env python3
"""
Master test runner for all crash-safe logging tests.
Runs comprehensive tests and analyzes results.
"""

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from logging_module import crash_safe_log
from logging_module.crash_safe_logging import check_for_crashed_processes, generate_cluster_diagnostic_script

class TestRunner:
    """Master test runner for all crash scenarios."""
    
    def __init__(self):
        self.test_logs_base = Path("logs_test_master")
        self.test_logs_base.mkdir(exist_ok=True)
        
        self.results = {
            "comprehensive": {"passed": 0, "failed": 0, "total": 0},
            "cluster": {"passed": 0, "failed": 0, "total": 0},
            "analysis": {"crashes_detected": 0, "deadman_triggers": 0}
        }
    
    def run_test_suite(self, test_script, suite_name):
        """Run a test suite and capture results."""
        print(f"\n[RUN] RUNNING {suite_name.upper()} TEST SUITE")
        print("=" * 60)
        
        try:
            # Run the test script
            result = subprocess.run([
                sys.executable, test_script
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            print(f"[DATA] {suite_name} Test Output:")
            print("-" * 30)
            print(result.stdout)
            
            if result.stderr:
                print(f"[WARN]  {suite_name} Test Errors:")
                print(result.stderr)
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            passed = 0
            failed = 0
            total = 0
            
            for line in output_lines:
                if "Total tests:" in line or "Total cluster tests:" in line:
                    total = int(line.split(':')[1].strip())
                elif "Passed:" in line:
                    passed = int(line.split(':')[1].strip())
                elif "Failed:" in line:
                    failed = int(line.split(':')[1].strip())
            
            self.results[suite_name.lower()] = {
                "passed": passed,
                "failed": failed, 
                "total": total,
                "exit_code": result.returncode
            }
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {suite_name} test suite timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"[FAIL] Error running {suite_name} test suite: {e}")
            return False
    
    def analyze_crash_evidence(self):
        """Analyze logs for crash evidence."""
        print("\n[SEARCH] ANALYZING CRASH EVIDENCE")
        print("=" * 60)
        
        # Check for crashed processes in all test log directories
        test_dirs = ["logs_test", "logs_test_cluster"]
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
                        with open(log_file, 'r') as f:
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
                            "CPU time limit exceeded"
                        ]
                        
                        found_indicators = []
                        for indicator in indicators:
                            if indicator in content:
                                found_indicators.append(indicator)
                        
                        if found_indicators:
                            crash_indicators += 1
                            print(f"   [FILE] {log_file.name}: {', '.join(found_indicators)}")
                    
                    except Exception as e:
                        print(f"   [WARN]  Could not analyze {log_file.name}: {e}")
                
                total_crashes += crash_indicators
        
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
            
            # Comprehensive tests results
            comp = self.results.get("comprehensive", {})
            f.write("COMPREHENSIVE TESTS:\n")
            f.write(f"  Total: {comp.get('total', 0)}\n")
            f.write(f"  Passed: {comp.get('passed', 0)}\n")
            f.write(f"  Failed: {comp.get('failed', 0)}\n")
            f.write(f"  Success Rate: {(comp.get('passed', 0) / max(comp.get('total', 1), 1)) * 100:.1f}%\n\n")
            
            # Cluster tests results
            cluster = self.results.get("cluster", {})
            f.write("CLUSTER-SPECIFIC TESTS:\n")
            f.write(f"  Total: {cluster.get('total', 0)}\n")
            f.write(f"  Passed: {cluster.get('passed', 0)}\n")
            f.write(f"  Failed: {cluster.get('failed', 0)}\n")
            f.write(f"  Success Rate: {(cluster.get('passed', 0) / max(cluster.get('total', 1), 1)) * 100:.1f}%\n\n")
            
            # Analysis results
            analysis = self.results.get("analysis", {})
            f.write("CRASH EVIDENCE ANALYSIS:\n")
            f.write(f"  Crash evidence detected: {analysis.get('crashes_detected', 0)}\n")
            f.write(f"  Deadman switch triggers: {analysis.get('deadman_triggers', 0)}\n\n")
            
            # Overall assessment
            total_tests = comp.get('total', 0) + cluster.get('total', 0)
            total_passed = comp.get('passed', 0) + cluster.get('passed', 0)
            overall_success = (total_passed / max(total_tests, 1)) * 100
            
            f.write("OVERALL ASSESSMENT:\n")
            f.write(f"  Total tests executed: {total_tests}\n")
            f.write(f"  Total tests passed: {total_passed}\n")
            f.write(f"  Overall success rate: {overall_success:.1f}%\n")
            f.write(f"  Crash detection capability: {'PASS' if analysis.get('crashes_detected', 0) > 0 else 'FAIL'}\n")
            f.write(f"  Deadman switch functionality: {'PASS' if analysis.get('deadman_triggers', 0) > 0 else 'FAIL'}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
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
    print("that could cause your quantum walk experiments to disappear.\n")
    
    success = True
    
    # Run comprehensive tests
    if not args.skip_comprehensive:
        print("Phase 1: Running comprehensive crash tests...")
        if not runner.run_test_suite("comprehensive_crash_tests.py", "Comprehensive"):
            success = False
        time.sleep(2)
    
    # Run cluster-specific tests
    if not args.skip_cluster:
        print("Phase 2: Running cluster-specific tests...")
        if not runner.run_test_suite("cluster_specific_crash_tests.py", "Cluster"):
            success = False
        time.sleep(2)
    
    # Analyze crash evidence
    if not args.skip_analysis:
        print("Phase 3: Analyzing crash evidence...")
        crashes, deadman = runner.analyze_crash_evidence()
        time.sleep(1)
    
    # Generate diagnostic script
    if not args.skip_diagnostics:
        print("Phase 4: Generating cluster diagnostic tools...")
        generate_cluster_diagnostic_script("cluster_diagnostics.sh")
        print("[PASS] Cluster diagnostic script generated")
    
    # Generate comprehensive report
    report_file = runner.generate_comprehensive_report()
    
    # Final summary
    print("\n" + "=" * 60)
    print("[FINISH] MASTER TEST SUITE COMPLETE")
    print("=" * 60)
    
    comp = runner.results.get("comprehensive", {})
    cluster = runner.results.get("cluster", {})
    analysis = runner.results.get("analysis", {})
    
    total_tests = comp.get('total', 0) + cluster.get('total', 0)
    total_passed = comp.get('passed', 0) + cluster.get('passed', 0)
    
    print(f"[DATA] FINAL RESULTS:")
    print(f"   Comprehensive tests: {comp.get('passed', 0)}/{comp.get('total', 0)} passed")
    print(f"   Cluster tests: {cluster.get('passed', 0)}/{cluster.get('total', 0)} passed")
    print(f"   Overall: {total_passed}/{total_tests} ({(total_passed/max(total_tests,1))*100:.1f}%)")
    print(f"   Crash evidence: {analysis.get('crashes_detected', 0)} instances")
    print(f"   Deadman triggers: {analysis.get('deadman_triggers', 0)} instances")
    
    print(f"\n[FOLDER] TEST ARTIFACTS:")
    print(f"   Test logs: logs_test/ and logs_test_cluster/")
    print(f"   Master logs: {runner.test_logs_base}/")
    print(f"   Report: {report_file}")
    print(f"   Diagnostics: cluster_diagnostics.sh")
    
    print(f"\n[TARGET] NEXT STEPS:")
    print("   1. Review the comprehensive report")
    print("   2. Check specific test logs for details")
    print("   3. Run cluster diagnostics on your actual cluster")
    print("   4. Deploy enhanced logging to your quantum walk experiments")
    
    if success and total_passed >= total_tests * 0.8:  # 80% success threshold
        print("\n🎉 SUCCESS: Crash-safe logging system is ready for cluster deployment!")
    else:
        print("\n[WARN]  WARNING: Some tests failed - review before cluster deployment")
    
    return success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Master test runner for crash-safe logging")
    parser.add_argument("--skip-comprehensive", action="store_true", 
                       help="Skip comprehensive crash tests")
    parser.add_argument("--skip-cluster", action="store_true",
                       help="Skip cluster-specific tests")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip crash evidence analysis")
    parser.add_argument("--skip-diagnostics", action="store_true",
                       help="Skip diagnostic script generation")
    parser.add_argument("--quick", action="store_true",
                       help="Run only essential tests (skips some time-consuming tests)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.skip_comprehensive = False  # Keep comprehensive as they're essential
        print("[RUN] Quick mode: Running essential tests only...")
    
    try:
        success = run_all_tests(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[WARN]  Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
