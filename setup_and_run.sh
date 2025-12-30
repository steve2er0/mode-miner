#!/bin/bash
# Mode Miner - Setup and Run Script for Linux
# Usage: ./setup_and_run.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Mode Miner Setup ==="

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please load a Python module or install Python 3."
    echo "On HPC, try: module load python/3.x"
    exit 1
fi

echo "Using Python: $PYTHON"
$PYTHON --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""

# Check if display is available (for GUI)
if [ -z "$DISPLAY" ]; then
    echo "WARNING: No display detected (\$DISPLAY is not set)."
    echo "GUI applications require X11 forwarding or a display."
    echo ""
    echo "Options:"
    echo "  1. SSH with X11 forwarding: ssh -X user@hpc"
    echo "  2. Use VNC or remote desktop"
    echo "  3. Set DISPLAY manually if using a virtual display"
    echo ""
    read -p "Try to run anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Activate the environment later with:"
        echo "  source venv/bin/activate"
        echo "  python run_test.py"
        exit 0
    fi
fi

# Run the application
echo "Starting Mode Miner..."
python run_test.py

