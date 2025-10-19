#!/bin/bash

echo "Starting ML Image Prediction API..."

if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in PATH"
    exit 1
fi

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: No virtual environment detected. Consider using 'python -m venv venv && source venv/bin/activate'"
fi

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

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "Starting server on http://$HOST:$PORT"
echo "API Documentation: http://localhost:$PORT/docs"
echo "Health Check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn main:app \
    --host $HOST \
    --port $PORT \
    --log-level $LOG_LEVEL \
    --reload