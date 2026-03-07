#!/bin/bash
# Start the Voice Training App

APP_DIR="$(cd "$(dirname "$0")/root/voice_app" && pwd)"
VENV="$(cd "$(dirname "$0")/venv" && pwd)"
PYTHON="$VENV/bin/python"

# Kill any existing instance on port 5000
fuser -k 5000/tcp 2>/dev/null || true

echo "Starting Voice Training App..."
echo "Open your browser at: http://localhost:5000"
echo "Press Ctrl+C to stop."
echo ""

cd "$APP_DIR"
"$PYTHON" app.py
