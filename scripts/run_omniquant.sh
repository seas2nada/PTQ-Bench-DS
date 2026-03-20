#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=${1:?usage: run_omniquant.sh GPU CONFIG}
CONFIG=${2:?usage: run_omniquant.sh GPU CONFIG}
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_quant.py --method omniquant --config "$CONFIG"
