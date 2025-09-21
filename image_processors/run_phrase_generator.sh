#!/bin/bash
echo "Starting Human Situation Phrase Generator..."
cd "$(dirname "$0")"
source ".venv/bin/activate"
python phrase_generator.py
