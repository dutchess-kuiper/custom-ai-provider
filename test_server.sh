#!/bin/bash

echo "Testing OpenAI-compatible API server..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Run the test client
echo "Running test client..."
python test_client.py 