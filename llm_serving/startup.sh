#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap for a fresh GPU instance: create a venv, install vLLM,
# pre-download the model, then start serving. Assumes an image with the
# NVIDIA driver + CUDA already installed (standard on rented A10G/L4 images
# from Lambda, RunPod, vast.ai, etc.) -- this script only handles the Python
# side.

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

bash download_model.sh
exec bash serve.sh
