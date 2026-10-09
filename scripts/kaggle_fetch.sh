#!/usr/bin/env bash
# Usage: scripts/kaggle_fetch.sh splitsnap-models-qwen3   -> downloads output + log to out/<slug>/ and prints key lines
set -euo pipefail
SLUG="${1:?kernel slug}"; cd "$(dirname "$0")/.."; source "${NAALA_VENV:-$HOME/venvs/naala}/bin/activate"
U=$(kaggle config view | awk '/username/{print $3}'); mkdir -p "out/$SLUG"
kaggle kernels status "$U/$SLUG" | tail -1
kaggle kernels output "$U/$SLUG" -p "out/$SLUG" 2>&1 | tail -2
grep -hE "^(BASELINE|BENCH|VLM|PREP|TOKLEN)|\{\"epoch\"|FAILED|Traceback|Error" "out/$SLUG/$SLUG.log" 2>/dev/null | cut -c1-600 | head -40 || true
