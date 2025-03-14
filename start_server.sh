#!/bin/bash

echo "Starting OpenAI-compatible API server on port 8001..."
echo "This server will be accessible at http://localhost:8001/v1"

# Kill any existing process on port 8001
lsof -ti:8001 | xargs kill -9 2>/dev/null

# Start the server using our main.py directly (which has IPv6 support)
python main.py 