#!/bin/bash
# Kill any running reporter.py processes and clean up the PID file.
# Safe to run at any time, including before re-starting run_experiments.sh.

echo "Finding reporter processes..."
ps aux | grep "[r]eporter.py"

pkill -f "reporter.py" 2>/dev/null && echo "Killed." || echo "No reporter processes found."

rm -f checkpoints/.reporter.pid
echo "PID file removed."

echo "Verifying..."
ps aux | grep "[r]eporter.py" && echo "WARNING: processes still running" || echo "All clear."
