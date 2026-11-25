#!/bin/bash
# NLI Demo — quick launch script
# Usage: bash run.sh
cd "$(dirname "$0")"
source .venv/bin/activate
python3 demo.py
