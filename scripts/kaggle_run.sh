#!/usr/bin/env bash
# Usage: scripts/kaggle_run.sh MODE [TAG]
#   MODE: baseline|bench|models|smoke|overfit|full      TAG: optional suffix => separate kernel (splitsnap-MODE-TAG)
# Env knobs:  TIMEOUT=27000                        (hard kernel time limit in seconds; default 27000 for full, none otherwise)
#              GPU=0                                (CPU-only kernel, uses no GPU quota; e.g. eda)
#             EXTRA_DATASETS="a/b,c/d"             (extra Kaggle datasets to attach, e.g. SROIE copies)
#              ONLY="Qwen/Qwen3-VL-2B-Instruct"  (models mode: "donut" or comma list of HF ids)
#             CHUNK=2                            (full mode: epochs per kernel run; chain runs with RESUME_KERNEL)
#             REAL_DATASET=atrijopal/splitsnap-real   RESUME_KERNEL=atrijopal/splitsnap-full-a
# Kaggle shows NOTHING mid-run, so split big jobs into several small kernels (one model / a few epochs each).
set -euo pipefail
MODE="${1:?mode: bench|baseline|models|smoke|overfit|full}"; TAG="${2:-}"
cd "$(dirname "$0")/.."
source "${NAALA_VENV:-$HOME/venvs/naala}/bin/activate"
USER_NAME=$(kaggle config view | awk '/username/{print $3}')
SLUG="splitsnap-$MODE${TAG:+-$TAG}"
B=build/kernel_$MODE${TAG:+_$TAG}; rm -rf "$B"; mkdir -p "$B"
cp kaggle/run_kaggle.py "$B/run_kaggle.py"
sed -i "s/^MODE = \".*\"$/MODE = \"$MODE\"/; s#^ONLY = \"\"#ONLY = \"${ONLY:-}\"#; s/^CHUNK = 0 /CHUNK = ${CHUNK:-0} /" "$B/run_kaggle.py"
SOURCES="\"$USER_NAME/splitsnap-code\""
if [ -z "${EXTRA_DATASETS:-}" ] && { [ "$MODE" = full ] || [ "$MODE" = smoke ] || [ "$MODE" = eval ] || [ "$MODE" = benchinfer ]; }; then EXTRA_DATASETS="urbikn/sroie-datasetv2"; fi   # SROIE is required for full/smoke
for d in $(echo "${EXTRA_DATASETS:-}" | tr "," " "); do SOURCES="$SOURCES, \"$d\""; done
if [ -n "${REAL_DATASET:-}" ]; then SOURCES="$SOURCES, \"$REAL_DATASET\""; fi
if [ -n "${RESUME_KERNEL:-}" ]; then KS="\"$RESUME_KERNEL\""; else KS=""; fi
cat > "$B/kernel-metadata.json" <<JSON
{"id": "$USER_NAME/$SLUG", "title": "$SLUG", "code_file": "run_kaggle.py",
 "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "$([ "${GPU:-1}" = 1 ] && echo true || echo false)",
 "enable_tpu": "false", "enable_internet": "true", "machine_shape": "$([ "${GPU:-1}" = 1 ] && echo NvidiaTeslaT4 || echo "")",
 "dataset_sources": [$SOURCES], "competition_sources": [], "kernel_sources": [$KS], "model_sources": []}
JSON
TIMEOUT="${TIMEOUT:-$([ "$MODE" = full ] && echo 27000 || { [ "$MODE" = smoke ] && echo 3600 || { { [ "$MODE" = eval ] || [ "$MODE" = benchinfer ]; } && echo 5400 || echo 0; }; })}"   # seconds; Kaggle kills the kernel at this limit (full: 7.5 h)
if [ "$TIMEOUT" != 0 ]; then kaggle kernels push -p "$B" -t "$TIMEOUT"; else kaggle kernels push -p "$B"; fi
echo "status: kaggle kernels status $USER_NAME/$SLUG   |   fetch when done: scripts/kaggle_fetch.sh $SLUG"
