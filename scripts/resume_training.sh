#!/usr/bin/env bash
# Resume both MAPPO variants. Run from any directory with an activated environment.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/resume_training.sh SHEPHERD_MODEL_DIR FLAT_MODEL_DIR" >&2
    exit 2
fi
python -m training.train_traffic_shepherd --model-dir "$1"
python -m training.train_flat_marl --model-dir "$2"
