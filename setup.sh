#!/bin/bash

echo "Setting up OpenAI-compatible API server with Haystack..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session if it was installed to ~/.cargo/bin
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

# Create a virtual environment using uv
echo "Creating a virtual environment with uv..."
uv venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "✅ Setup complete! You can now run the server with:"
echo "  ./start_server.sh"
echo ""
echo "To activate the virtual environment manually:"
echo "  source .venv/bin/activate" 