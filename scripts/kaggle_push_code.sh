#!/usr/bin/env bash
# Upload ./splitsnap as a PRIVATE Kaggle dataset (atrijopal/splitsnap-code). Run again after every code change.
set -euo pipefail
cd "$(dirname "$0")/.."
source "${NAALA_VENV:-$HOME/venvs/naala}/bin/activate"
USER_NAME=$(kaggle config view | awk '/username/{print $3}')
ID="$USER_NAME/splitsnap-code"
STAGE=build/code_ds; rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -r splitsnap "$STAGE/splitsnap"; find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "splitsnap-code", "id": "$ID", "licenses": [{"name": "CC0-1.0"}]}
JSON
if kaggle datasets status "$ID" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "code $(date +%F-%H%M)" --dir-mode zip
else
  kaggle datasets create -p "$STAGE" --dir-mode zip          # private by default
fi
