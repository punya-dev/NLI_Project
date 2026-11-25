#!/bin/bash
# NLI Demo — first-time setup script
# Usage: bash setup.sh
# Run this once on a new machine to create the venv and install dependencies.

set -e  # exit on error

cd "$(dirname "$0")"

echo "🔧 Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "⚡ Activating venv..."
source .venv/bin/activate

echo "📦 Installing dependencies from requirements.txt..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the demo:"
echo "  source .venv/bin/activate"
echo "  python demo.py"
echo ""
echo "Or simply:"
echo "  bash run.sh"
