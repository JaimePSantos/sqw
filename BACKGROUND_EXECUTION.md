# Background Execution for Cluster Environments

This directory contains enhanced signal handling and background execution capabilities for running quantum walk experiments on Linux clusters.

## Quick Start for Cluster Usage

### Option 1: Using the Background Launcher (Recommended)
```bash
# Launch in background mode (survives terminal disconnection)
python background_launcher.py

# Check if the process is running
python background_launcher.py status

# Kill the background process
python background_launcher.py kill

# Monitor progress
tail -f logs/master.log
```

### Option 2: Manual Background Execution
```bash
# Traditional nohup approach
nohup python static_cluster_logged_mp.py > output.log 2>&1 &

# Or using screen/tmux
screen -S quantum_walk
python static_cluster_logged_mp.py
# Press Ctrl+A, D to detach

# Reattach later
screen -r quantum_walk
```

### Option 3: Direct Execution with Enhanced Signal Handling
```bash
# Start normally
python static_cluster_logged_mp.py

# Background with Ctrl+Z, bg (now properly handled)
# Press Ctrl+Z
bg
exit  # Terminal can now be closed safely
```

## Signal Handling Features

The script now includes enhanced signal handling:

- **SIGINT (Ctrl+C)**: Graceful shutdown with cleanup
- **SIGTERM**: Graceful shutdown for system termination
- **SIGHUP**: Ignored to prevent termination on terminal disconnect
- **SIGPIPE**: Ignored to prevent crashes when terminal disconnects

## Logging Behavior

### Normal Execution
- Console output shows clean progress messages
- Detailed logs written to `logs/master.log` and `logs/process_dev_*.log`

### Background Execution
- Output redirected to `background_execution.log`
- All experiment logs still written to `logs/` directory
- Process PID tracked in `background_process.pid`

## Monitoring Background Processes

```bash
# Check if process is running
ps aux | grep static_cluster

# Monitor master log
tail -f logs/master.log

# Monitor specific process logs
tail -f logs/process_dev_samples.log

# Check background execution log
tail -f background_execution.log

# Monitor system resources
top -p $(cat background_process.pid)
```

## Troubleshooting

### Process Won't Start in Background
- Check permissions: `chmod +x background_launcher.py`
- Verify Python path: `which python3`
- Check for missing dependencies

### Process Dies After Terminal Disconnect
- Use the background launcher instead of manual backgrounding
- Check cluster-specific policies for long-running jobs
- Consider using the cluster's job scheduler if available

### Logs Not Updating
- Verify the crash_safe_log decorator is enabled
- Check disk space: `df -h`
- Ensure write permissions: `ls -la logs/`

### High Memory/CPU Usage
- Monitor with: `top -p $(cat background_process.pid)`
- Adjust MAX_PROCESSES in the script if needed
- Check for swap usage: `free -h`

## Files Created During Execution

- `logs/master.log` - Main experiment log
- `logs/process_dev_*.log` - Individual process logs  
- `background_execution.log` - Background mode output
- `background_process.pid` - Process ID tracking
- `experiments_data_*.tar.gz` - Archived results

## Cluster-Specific Notes

### SLURM Clusters
```bash
# Submit as job
sbatch --wrap="python background_launcher.py"

# Interactive with background
srun --pty bash
python background_launcher.py
```

### PBS/Torque Clusters
```bash
# Submit as job
echo "python background_launcher.py" | qsub

# Interactive
qsub -I
python background_launcher.py
```

## Environment Variables

- `BACKGROUND_MODE=1` - Force background execution mode
- `KEEP_TERMINAL_OUTPUT=1` - Keep console output (used by launcher)

## Safety Features

- Automatic process cleanup on errors
- PID file management
- Signal handling for graceful shutdown
- Log rotation and archiving
- Error recovery and logging
