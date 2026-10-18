#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="$ROOT/root/voice_app/venv/bin/python"
export PYTHONPATH="$ROOT/root/voice_app:$ROOT/development${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/voice-trainer-mpl}"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'
for test in tests/test_*.js; do
    node "$test"
done
