#!/bin/bash
# Cluster Diagnostics Script
# Run this on your cluster to gather system information

echo "=== CLUSTER DIAGNOSTIC INFORMATION ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo

echo "=== CLUSTER ENVIRONMENT ==="
env | grep -E "(SLURM|PBS|TORQUE|SGE|LSB)" || echo "No cluster environment variables found"
echo

echo "=== RESOURCE LIMITS ==="
ulimit -a
echo

echo "=== MEMORY INFORMATION ==="
free -h
echo

echo "=== CPU INFORMATION ==="
lscpu | head -20
echo

echo "=== DISK SPACE ==="
df -h
echo

echo "=== SYSTEM LOAD ==="
uptime
echo

echo "=== PROCESS LIMITS ==="
cat /proc/sys/kernel/pid_max 2>/dev/null || echo "Cannot read pid_max"
echo

echo "=== CGROUP INFORMATION ==="
cat /proc/1/cgroup 2>/dev/null || echo "Cannot read cgroup info"
echo

echo "=== CLUSTER SPECIFIC FILES ==="
for dir in /var/spool/slurm /var/spool/pbs /opt/sge /etc/slurm /etc/pbs /etc/torque; do
    if [ -d "$dir" ]; then
        echo "Found cluster directory: $dir"
        ls -la "$dir" 2>/dev/null | head -10
    fi
done
echo

echo "=== SYSTEM MESSAGES (last 50 lines) ==="
sudo tail -50 /var/log/messages 2>/dev/null || \
sudo tail -50 /var/log/syslog 2>/dev/null || \
echo "Cannot access system logs (try: sudo journalctl -n 50)"
echo

echo "=== MEMORY PRESSURE INFORMATION ==="
cat /proc/pressure/memory 2>/dev/null || echo "Memory pressure info not available"
echo

echo "=== OOM KILLER LOGS ==="
sudo dmesg | grep -i "killed process\|out of memory\|oom" | tail -10 || echo "Cannot access dmesg"
echo

echo "=== END DIAGNOSTICS ==="
