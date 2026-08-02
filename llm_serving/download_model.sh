#!/usr/bin/env bash
set -euo pipefail

# Downloads the already-quantized AWQ checkpoint into the local HF cache so
# serve.sh's first request doesn't pay the download latency. Safe to re-run
# (huggingface_hub skips files already present).

MODEL_ID="${VLLM_MODEL:-TheBloke/Mistral-7B-Instruct-v0.2-AWQ}"

echo "Downloading ${MODEL_ID} into ${HF_HOME:-$HOME/.cache/huggingface} ..."
python3 - "$MODEL_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1])
PY
echo "Done."
