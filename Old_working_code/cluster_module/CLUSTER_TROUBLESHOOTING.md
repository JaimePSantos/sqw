# Diagnosing Sudden Process Termination in Cluster Environments

When running long quantum walk experiments on cluster systems, processes may suddenly disappear without leaving termination logs. This enhanced logging system helps identify the root cause of such terminations.

## Common Causes of Sudden Process Termination

### 1. OOM (Out of Memory) Kills
- **Symptom**: Process disappears, no termination logs, high memory usage in logs
- **Cause**: Kernel kills process when memory usage exceeds limits
- **Detection**: Look for "HIGH MEMORY USAGE" or "SYSTEM MEMORY CRITICAL" in logs

### 2. Resource Limit Violations
- **Symptom**: Process stops at specific resource thresholds
- **Cause**: Exceeding CPU time, memory, file size, or process limits
- **Detection**: System logs RLIMIT information, check for warnings

### 3. Cluster Job Scheduler Termination
- **Symptom**: Process ends at predictable times (job time limits)
- **Cause**: SLURM, PBS, or other job scheduler enforcing limits
- **Detection**: Check cluster environment variables and job status

### 4. Hardware/Network Failures
- **Symptom**: Sudden disappearance with no resource warnings
- **Cause**: Node crashes, network disconnections, storage failures
- **Detection**: Check system logs, cluster node status

### 5. SIGKILL (Forced Termination)
- **Symptom**: Immediate termination, cannot be caught by signal handlers
- **Cause**: System-level forced termination
- **Detection**: Deadman's switch triggers, no signal logs

## Enhanced Logging Features

### 1. Resource Monitoring
- Tracks memory usage, CPU usage, system load
- Warns when approaching dangerous levels
- Logs resource limits and current usage

### 2. Cluster Environment Detection
- Identifies SLURM, PBS, Torque, SGE environments
- Logs job IDs, node information, environment variables
- Monitors for cluster-specific warning signals

### 3. Deadman's Switch
- Creates timestamp file that's updated regularly
- Detects when main process stops updating file
- Provides evidence of sudden termination

### 4. Enhanced Signal Handling
- Catches standard signals (SIGTERM, SIGINT)
- Attempts to catch cluster-specific signals (SIGUSR1, SIGUSR2, SIGXCPU)
- Logs signal source and context

## Usage

### Basic Usage with Enhanced Monitoring

```python
from logging_module import crash_safe_log

@crash_safe_log(
    log_file_prefix="quantum_walk_experiment",
    heartbeat_interval=30.0,  # Longer intervals for cluster runs
    log_system_info=True
)
def run_quantum_walk_experiment():
    # Your experiment code here
    pass
```

### Crash Detection Analysis

```bash
# Check for evidence of crashed processes
python -m logging_module.crash_safe_logging --check-crashes

# Generate cluster diagnostic script
python -m logging_module.crash_safe_logging --generate-diagnostics
```

### Running Cluster Diagnostics

```bash
# Generate and run diagnostic script on cluster
python -m logging_module.crash_safe_logging --generate-diagnostics
bash cluster_diagnostics.sh > cluster_info.txt
```

## Interpreting Log Files

### Normal Termination Indicators
- "SHUTDOWN SEQUENCE" messages
- "SIGNAL RECEIVED" with proper signal handling
- "KEYBOARD INTERRUPT" for manual stops

### Abnormal Termination Indicators
- Last log entry is a HEARTBEAT with no shutdown messages
- "DEADMAN'S SWITCH TRIGGERED" in subsequent analysis
- Resource warnings followed by sudden silence
- High memory usage without resolution

### Example Log Analysis

```
# Normal termination
HEARTBEAT #45 - Elapsed: 450.2s - Process alive
=== SIGNAL RECEIVED: SIGTERM (15) ===
SIGTERM received - likely cluster job termination or timeout
=== BEGINNING SHUTDOWN SEQUENCE ===

# Abnormal termination (OOM kill)
HEARTBEAT #67 - Elapsed: 670.8s - Process alive  
HIGH MEMORY USAGE: 1847.3 MB
SYSTEM MEMORY CRITICAL: 94.2% used
HEARTBEAT #68 - Elapsed: 680.9s - Process alive
[No further logs - process killed by kernel]
```

## Preventing Process Termination

### 1. Memory Management
- Monitor memory usage patterns in logs
- Implement checkpointing for large computations
- Consider splitting experiments into smaller chunks
- Use memory-efficient algorithms and data structures

### 2. Resource Monitoring
- Set conservative resource requests in job scripts
- Monitor heartbeat logs for resource warnings
- Implement graceful degradation when approaching limits

### 3. Checkpointing
- Save intermediate results regularly
- Use the logging system to coordinate checkpoints
- Implement restart capability from checkpoints

### 4. Job Configuration
- Request appropriate time limits for cluster jobs
- Include buffer time for shutdown procedures
- Use job arrays for parallel processing

## Cluster-Specific Configuration

### SLURM Example
```bash
#SBATCH --time=02:00:00          # 2 hour time limit
#SBATCH --mem=4G                 # 4GB memory limit
#SBATCH --signal=SIGUSR1@90      # Send warning 90 seconds before timeout
```

### Handling Pre-termination Signals
```python
@crash_safe_log(heartbeat_interval=10.0)
def experiment_with_checkpointing():
    # Set up signal handler for SIGUSR1 (SLURM warning)
    import signal
    
    def checkpoint_handler(signum, frame):
        logger.warning("Received pre-termination signal - saving checkpoint")
        save_checkpoint()
        
    signal.signal(signal.SIGUSR1, checkpoint_handler)
    
    # Your experiment code with periodic checkpointing
    for i in range(num_iterations):
        run_iteration(i)
        if i % checkpoint_interval == 0:
            save_checkpoint()
```

## Troubleshooting Steps

1. **Check Recent Crashes**:
   ```bash
   python -m logging_module.crash_safe_logging --check-crashes
   ```

2. **Generate System Diagnostics**:
   ```bash
   python -m logging_module.crash_safe_logging --generate-diagnostics
   bash cluster_diagnostics.sh
   ```

3. **Analyze Resource Usage**:
   - Look for memory warnings in logs
   - Check CPU time usage patterns
   - Monitor disk space availability

4. **Check Cluster Status**:
   ```bash
   # SLURM
   squeue -u $USER
   sacct -j <job_id> --format=JobID,JobName,State,ExitCode,DerivedExitCode
   
   # PBS/Torque
   qstat -u $USER
   ```

5. **Review System Logs**:
   ```bash
   # Check for OOM kills
   sudo dmesg | grep -i "killed process"
   sudo journalctl | grep -i "out of memory"
   ```

## Contact and Support

For cluster-specific issues:
- Contact your cluster administrator
- Provide log files and diagnostic output
- Include job IDs and timestamps

For logging system issues:
- Check the log files in the `logs/` directory
- Run crash detection analysis
- Review resource usage patterns
