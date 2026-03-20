#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=${1:?usage: run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]}
CONFIG=${2:?usage: run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]}
DATASET=${3:?usage: run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]}
BITS=${4:?usage: run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]}
GROUP_SIZE=${5:?usage: run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]}
NSAMPLES=${6:?usage: run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]}
SAVE_PATH=${7:-./output/llama-2-7b-AWQ-${DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}}
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_quant.py --method awq --config "$CONFIG" --dataset "$DATASET" --bits "$BITS" --group_size "$GROUP_SIZE" --nsamples "$NSAMPLES" --save_path "$SAVE_PATH"
