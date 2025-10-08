#!/bin/bash

# ML Image Prediction API Startup Script

echo "Starting ML Image Prediction API..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in a virtual environment (recommended)
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: No virtual environment detected. Consider using 'python -m venv venv && source venv/bin/activate'"
fi

# Install dependencies if requirements.txt is newer than last install
if [[ ! -f .last_install ]] || [[ requirements.txt -nt .last_install ]]; then
    echo "Installing/updating dependencies..."
    pip install -r requirements.txt
    if [[ $? -eq 0 ]]; then
        touch .last_install
        echo "Dependencies installed successfully"
    else
        echo "Failed to install dependencies"
        exit 1
    fi
fi

# Set default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "Starting server on http://$HOST:$PORT"
echo "API Documentation: http://localhost:$PORT/docs"
echo "Health Check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI server with uvicorn
uvicorn main:app \
    --host $HOST \
    --port $PORT \
    --log-level $LOG_LEVEL \
    --reload