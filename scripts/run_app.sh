#!/usr/bin/env bash
# Run SplitSnap with the trained model.   bash scripts/run_app.sh [port]
# CKPT: fp16 export if present (411 MB, identical outputs), else the full checkpoint. DEVICE and flags can be overridden from the environment.
set -e
cd "$(dirname "$0")/.."
PORT="${1:-8000}"
if [ -z "$CKPT" ]; then
  for c in ckpt/best_fp16 ckpt/b2/runs/run1/best; do [ -d "$c" ] && CKPT="$c" && break; done
fi
[ -n "$CKPT" ] || { echo "No checkpoint found: set CKPT=path/to/best"; exit 1; }
export CKPT PRELOAD="${PRELOAD:-1}" FAST_PREP="${FAST_PREP:-0}"
echo "model: $CKPT   preload: $PRELOAD   port: $PORT"
exec python -m uvicorn splitsnap.app:app --host "${HOST:-127.0.0.1}" --port "$PORT"
