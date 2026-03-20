#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=${1:?usage: eval_ppl.sh GPU MODEL [EXTRA_ARGS...]}
MODEL=${2:?usage: eval_ppl.sh GPU MODEL [EXTRA_ARGS...]}
shift 2
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_ppl.py --model "$MODEL" "$@"
