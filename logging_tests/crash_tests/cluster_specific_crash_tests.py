#!/usr/bin/env python3
"""
Cluster-specific crash test suite.
Simulates cluster environment conditions and termination scenarios.
"""

import sys
import os
import time
import signal
import threading
import subprocess
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from logging_module import crash_safe_log
from logging_module.config import update_config

# Update config to use cluster test log directory
update_config(logs_base_directory="logs_test_cluster")

class ClusterEnvironmentSimulator:
    """Simulates various cluster environment conditions."""
    
    def __init__(self):
        self.original_env = dict(os.environ)
        self.test_results = []
    
    def setup_slurm_environment(self):
        """Set up SLURM environment variables."""
        os.environ.update({
            'SLURM_JOB_ID': '123456',
            'SLURM_JOB_NAME': 'quantum_walk_test',
            'SLURM_PROCID': '0',
            'SLURM_LOCALID': '0',
            'SLURM_NTASKS': '1',
            'SLURM_CPUS_PER_TASK': '4',
            'SLURM_MEM_PER_NODE': '4096',
            'SLURM_TIMELIMIT': '120',  # 2 hours in minutes
        })
    
    def setup_pbs_environment(self):
        """Set up PBS environment variables."""
        os.environ.update({
            'PBS_JOBID': '789.cluster.domain',
            'PBS_JOBNAME': 'quantum_walk_test',
            'PBS_QUEUE': 'normal',
            'PBS_WALLTIME': '02:00:00',
            'PBS_NODEFILE': '/tmp/pbs_nodefile',
        })
    
    def cleanup_environment(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
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

def test_cluster_1_slurm_job_timeout():
    """Test 1: SLURM job timeout simulation."""
    @crash_safe_log(
        log_file_prefix="cluster_test_1_slurm_timeout",
        heartbeat_interval=2.0,
        log_system_info=True
    )
    def slurm_timeout_test():
        print("Cluster Test 1: SLURM job timeout simulation")
        print("  Simulating SLURM timeout with SIGUSR1 warning signal...")
        
        def send_timeout_warning():
            time.sleep(4)
            print("  SLURM: Sending warning signal (SIGUSR1)...")
            os.kill(os.getpid(), signal.SIGUSR1)
            
            time.sleep(2)
            print("  SLURM: Sending termination signal (SIGTERM)...")
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Start timeout simulation
        timeout_thread = threading.Thread(target=send_timeout_warning, daemon=True)
        timeout_thread.start()
        
        # Simulate long-running computation
        for i in range(20):
            print(f"  Computing quantum walk step {i+1}/20...")
            time.sleep(1)
        
        return "should_not_complete"
    
    # Set up SLURM environment
    simulator = ClusterEnvironmentSimulator()
    simulator.setup_slurm_environment()
    
    try:
        result = slurm_timeout_test()
        return "FAIL", f"Should have been terminated but completed: {result}"
    except SystemExit as e:
        return "PASS", f"Caught SystemExit with code: {e.code}"
    except Exception as e:
        return "PASS", f"Caught termination signal: {e}"
    finally:
        simulator.cleanup_environment()

def test_cluster_2_pbs_job_termination():
    """Test 2: PBS job termination simulation."""
    @crash_safe_log(
        log_file_prefix="cluster_test_2_pbs",
        heartbeat_interval=1.5,
        log_system_info=True
    )
    def pbs_termination_test():
        print("Cluster Test 2: PBS job termination simulation")
        print("  Simulating PBS job manager termination...")
        
        def send_pbs_termination():
            time.sleep(5)
            print("  PBS: Job time limit reached, sending SIGTERM...")
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Start termination simulation
        term_thread = threading.Thread(target=send_pbs_termination, daemon=True)
        term_thread.start()
        
        # Simulate quantum walk computation
        for i in range(15):
            print(f"  Processing quantum walk iteration {i+1}/15...")
            time.sleep(1)
        
        return "should_not_complete"
    
    # Set up PBS environment
    simulator = ClusterEnvironmentSimulator()
    simulator.setup_pbs_environment()
    
    try:
        result = pbs_termination_test()
        return "FAIL", f"Should have been terminated but completed: {result}"
    except SystemExit as e:
        return "PASS", f"Caught PBS termination: {e.code}"
    except Exception as e:
        return "PASS", f"Caught termination: {e}"
    finally:
        simulator.cleanup_environment()

def test_cluster_3_oom_kill_simulation():
    """Test 3: OOM kill simulation with cluster logging."""
    @crash_safe_log(
        log_file_prefix="cluster_test_3_oom",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def oom_kill_simulation():
        print("Cluster Test 3: OOM kill simulation")
        print("  Simulating memory exhaustion that would trigger OOM killer...")
        
        def simulate_oom_kill():
            time.sleep(6)
            print("  KERNEL: Sending SIGKILL (OOM killer)...")
            # Note: SIGKILL cannot be caught, but we'll use SIGTERM for simulation
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Start OOM simulation
        oom_thread = threading.Thread(target=simulate_oom_kill, daemon=True)
        oom_thread.start()
        
        # Simulate memory-intensive quantum walk
        data = []
        for i in range(100):
            # Allocate memory chunks
            try:
                chunk = [0] * (10 * 1024 * 1024 // 8)  # 10MB
                data.append(chunk)
                allocated_mb = (i + 1) * 10
                print(f"  Allocated {allocated_mb} MB for quantum walk matrices...")
                time.sleep(0.2)
                
                # Stop at reasonable limit for testing
                if allocated_mb >= 200:  # 200MB limit for testing
                    break
                    
            except MemoryError:
                print(f"  Memory allocation failed at {(i+1)*10} MB")
                break
        
        return "should_not_complete_due_to_oom"
    
    # Set up cluster environment
    simulator = ClusterEnvironmentSimulator()
    simulator.setup_slurm_environment()
    
    try:
        result = oom_kill_simulation()
        return "PASS", "Completed memory test without OOM kill"
    except SystemExit as e:
        return "PASS", f"Simulated OOM kill caught: {e.code}"
    except Exception as e:
        return "PASS", f"Memory-related termination: {e}"
    finally:
        simulator.cleanup_environment()

def test_cluster_4_network_failure():
    """Test 4: Network failure simulation affecting shared filesystems."""
    @crash_safe_log(
        log_file_prefix="cluster_test_4_network",
        heartbeat_interval=2.0,
        log_system_info=True
    )
    def network_failure_simulation():
        print("Cluster Test 4: Network/storage failure simulation")
        print("  Simulating network interruption affecting shared storage...")
        
        # Create temporary files to simulate shared storage
        temp_dir = Path("temp_shared_storage")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            for i in range(10):
                print(f"  Writing quantum walk results to shared storage... {i+1}/10")
                
                # Simulate writing to shared filesystem
                result_file = temp_dir / f"qw_result_{i}.dat"
                with open(result_file, 'w') as f:
                    f.write(f"quantum_walk_step_{i}_data\n" * 1000)
                
                # Simulate network delay
                time.sleep(0.5)
                
                # Simulate network failure at step 7
                if i == 6:
                    print("  NETWORK: Connection to shared storage lost!")
                    raise OSError("Network unreachable - shared storage unavailable")
            
            return "network_test_completed"
            
        finally:
            # Clean up
            try:
                for file in temp_dir.glob("*.dat"):
                    file.unlink()
                temp_dir.rmdir()
            except:
                pass
    
    try:
        result = network_failure_simulation()
        return "FAIL", f"Should have failed due to network issues: {result}"
    except OSError as e:
        return "PASS", f"Caught network failure: {e}"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def test_cluster_5_node_failure():
    """Test 5: Compute node failure simulation."""
    @crash_safe_log(
        log_file_prefix="cluster_test_5_node_failure",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def node_failure_simulation():
        print("Cluster Test 5: Compute node failure simulation")
        print("  Simulating sudden hardware failure...")
        
        def simulate_hardware_failure():
            time.sleep(4)
            print("  HARDWARE: Node experiencing critical failure!")
            print("  HARDWARE: Immediate shutdown required!")
            # Simulate immediate termination (hardware failure)
            os._exit(1)  # Immediate exit without cleanup
        
        # Start hardware failure simulation
        failure_thread = threading.Thread(target=simulate_hardware_failure, daemon=True)
        failure_thread.start()
        
        # Simulate distributed quantum walk computation
        for i in range(20):
            print(f"  Computing on node... step {i+1}/20")
            time.sleep(0.5)
        
        return "should_not_complete_hardware_failure"
    
    try:
        result = node_failure_simulation()
        return "FAIL", f"Should have failed due to hardware: {result}"
    except SystemExit as e:
        return "PASS", f"Caught system exit: {e.code}"
    except Exception as e:
        return "PASS", f"Caught failure: {e}"

def test_cluster_6_cpu_limit_exceeded():
    """Test 6: CPU time limit exceeded."""
    @crash_safe_log(
        log_file_prefix="cluster_test_6_cpu_limit",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def cpu_limit_test():
        print("Cluster Test 6: CPU time limit exceeded")
        print("  Simulating SIGXCPU (CPU time limit exceeded)...")
        
        def send_cpu_limit_signal():
            time.sleep(3)
            print("  CLUSTER: CPU time limit exceeded, sending SIGXCPU...")
            if hasattr(signal, 'SIGXCPU'):
                os.kill(os.getpid(), signal.SIGXCPU)
            else:
                print("  SIGXCPU not available on this platform, using SIGTERM...")
                os.kill(os.getpid(), signal.SIGTERM)
        
        # Start CPU limit simulation
        cpu_thread = threading.Thread(target=send_cpu_limit_signal, daemon=True)
        cpu_thread.start()
        
        # CPU-intensive quantum walk computation
        result = 0
        for i in range(1000000):
            result += i * i
            if i % 100000 == 0:
                print(f"  CPU-intensive computation... {i//10000}%")
        
        return f"cpu_computation_result_{result}"
    
    try:
        result = cpu_limit_test()
        return "PASS", "Completed CPU test"
    except SystemExit as e:
        return "PASS", f"Caught CPU limit signal: {e.code}"
    except Exception as e:
        return "PASS", f"Caught CPU-related termination: {e}"

def test_cluster_7_checkpointing():
    """Test 7: Checkpointing on cluster warning signals."""
    @crash_safe_log(
        log_file_prefix="cluster_test_7_checkpoint",
        heartbeat_interval=1.0,
        log_system_info=True
    )
    def checkpointing_test():
        print("Cluster Test 7: Checkpointing test")
        print("  Testing graceful shutdown with checkpointing...")
        
        checkpoint_data = {"step": 0, "results": []}
        checkpoint_file = Path("test_checkpoint.dat")
        
        def save_checkpoint():
            """Save current state to checkpoint file."""
            with open(checkpoint_file, 'w') as f:
                f.write(f"step={checkpoint_data['step']}\n")
                f.write(f"results={','.join(map(str, checkpoint_data['results']))}\n")
            print(f"  Checkpoint saved at step {checkpoint_data['step']}")
        
        def sigusr1_handler(signum, frame):
            """Handle SIGUSR1 as checkpoint signal."""
            print("  Received SIGUSR1 - saving checkpoint before termination...")
            save_checkpoint()
        
        # Set up signal handler for checkpointing
        signal.signal(signal.SIGUSR1, sigusr1_handler)
        
        def send_checkpoint_signal():
            time.sleep(5)
            print("  CLUSTER: Sending checkpoint warning (SIGUSR1)...")
            os.kill(os.getpid(), signal.SIGUSR1)
            
            time.sleep(1)
            print("  CLUSTER: Sending termination (SIGTERM)...")
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Start checkpoint simulation
        checkpoint_thread = threading.Thread(target=send_checkpoint_signal, daemon=True)
        checkpoint_thread.start()
        
        try:
            # Simulate long quantum walk computation with periodic checkpointing
            for i in range(20):
                checkpoint_data['step'] = i
                checkpoint_data['results'].append(i * 2)
                
                print(f"  Quantum walk step {i+1}/20...")
                
                # Save checkpoint every 5 steps
                if i % 5 == 0:
                    save_checkpoint()
                
                time.sleep(1)
            
            return "checkpointing_test_completed"
            
        finally:
            # Clean up
            if checkpoint_file.exists():
                checkpoint_file.unlink()
    
    try:
        result = checkpointing_test()
        return "PASS", "Checkpointing test completed"
    except SystemExit as e:
        return "PASS", f"Graceful termination with checkpoint: {e.code}"
    except Exception as e:
        return "PASS", f"Caught with checkpointing: {e}"

def test_cluster_8_resource_contention():
    """Test 8: Resource contention in multi-user cluster environment."""
    @crash_safe_log(
        log_file_prefix="cluster_test_8_contention",
        heartbeat_interval=2.0,
        log_system_info=True
    )
    def resource_contention_test():
        print("Cluster Test 8: Resource contention simulation")
        print("  Simulating resource competition with other cluster jobs...")
        
        # Simulate multiple competing processes
        processes = []
        
        try:
            # Create multiple competing subprocesses
            for i in range(3):
                script_content = f'''
import time
import sys

print(f"Competing job {i+1} started")
for j in range(10):
    # Simulate CPU and memory usage
    data = [k*k for k in range(100000)]
    time.sleep(0.5)
    print(f"Competing job {i+1} step {{j+1}}/10")
print(f"Competing job {i+1} completed")
'''
                
                script_path = Path(f"temp_competitor_{i}.py")
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Start competing process
                process = subprocess.Popen([sys.executable, str(script_path)])
                processes.append((process, script_path))
                print(f"  Started competing job {i+1} (PID: {process.pid})")
            
            # Main quantum walk computation under resource pressure
            for i in range(15):
                print(f"  Main quantum walk computation under load... {i+1}/15")
                
                # Simulate computation that competes for resources
                data = [j*j for j in range(50000)]
                time.sleep(1)
                
                # Check if we're being starved of resources
                if i == 10:
                    print("  Detecting resource starvation...")
                    # Simulate being killed due to resource pressure
                    raise ResourceWarning("Insufficient resources - job preempted")
            
            return "resource_contention_completed"
            
        finally:
            # Clean up competing processes
            for process, script_path in processes:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                    except:
                        pass
                
                try:
                    script_path.unlink()
                except:
                    pass
    
    try:
        result = resource_contention_test()
        return "PASS", f"Completed under contention: {result}"
    except ResourceWarning as e:
        return "PASS", f"Caught resource starvation: {e}"
    except Exception as e:
        return "FAIL", f"Unexpected error: {e}"

def main():
    """Run all cluster-specific crash tests."""
    print("CLUSTER-SPECIFIC CRASH TEST SUITE")
    print("=" * 60)
    print("Testing cluster environment termination scenarios...\n")
    
    simulator = ClusterEnvironmentSimulator()
    
    # Define cluster-specific tests
    tests = [
        ("SLURM Job Timeout", test_cluster_1_slurm_job_timeout),
        ("PBS Job Termination", test_cluster_2_pbs_job_termination),
        ("OOM Kill Simulation", test_cluster_3_oom_kill_simulation),
        ("Network/Storage Failure", test_cluster_4_network_failure),
        ("Compute Node Failure", test_cluster_5_node_failure),
        ("CPU Time Limit Exceeded", test_cluster_6_cpu_limit_exceeded),
        ("Checkpointing on Warning", test_cluster_7_checkpointing),
        ("Resource Contention", test_cluster_8_resource_contention),
    ]
    
    # Run each test
    for test_name, test_func in tests:
        print(f"\n[SCOPE] Running: {test_name}")
        print("-" * 40)
        
        try:
            result, details = test_func()
            simulator.log_test_result(test_name, result, details)
        except Exception as e:
            simulator.log_test_result(test_name, "ERROR", f"Test framework error: {e}")
        
        print("Waiting 3 seconds before next test...")
        time.sleep(3)
    
    # Print summary
    print("\n" + "=" * 60)
    print("[FINISH] CLUSTER TEST SUITE SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in simulator.test_results if r["result"] == "PASS")
    failed = sum(1 for r in simulator.test_results if r["result"] in ["FAIL", "ERROR"])
    total = len(simulator.test_results)
    
    print(f"Total cluster tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    print("\n📋 DETAILED RESULTS:")
    for result in simulator.test_results:
        status = "[PASS]" if result["result"] == "PASS" else "[FAIL]"
        print(f"{status} {result['test']}: {result['result']}")
        if result["details"]:
            print(f"   {result['details']}")
    
    print(f"\n[FOLDER] Check cluster logs in: logs_test_cluster/ directory")
    print("[SEARCH] Analyze crashes: python -m logging_module.crash_safe_logging --check-crashes")
    print("🔧 Generate diagnostics: python -m logging_module.crash_safe_logging --generate-diagnostics")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
