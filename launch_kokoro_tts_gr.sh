#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/activate_venv.sh"

exec "$SCRIPT_DIR/venvMyCoding/bin/python" "$SCRIPT_DIR/kokoro_tts_gr.py"
