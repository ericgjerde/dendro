#!/bin/bash
# Setup script for dendrochronology dating tool
# Run: ./scripts/setup_env.sh

set -e

echo "Setting up Python environment for dendrochronology..."

# Check for Python 3.10+
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

echo ""
echo "Setup complete! To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  dendro --help       # Show CLI help"
echo "  dendro download     # Download reference chronologies"
echo "  dendro info         # Show available references"
echo "  pytest tests/       # Run tests"
