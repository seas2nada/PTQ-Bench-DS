#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=${1:?usage: run_cgptq2.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES}
CONFIG=${2:?usage: run_cgptq2.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES}
BITS=${3:?usage: run_cgptq2.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES}
GROUP_SIZE=${4:?usage: run_cgptq2.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES}
NSAMPLES=${5:?usage: run_cgptq2.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES}
DATASET=wikitext2
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_quant.py --method c-gptq --config "$CONFIG" --dataset "$DATASET" --bits "$BITS" --group_size "$GROUP_SIZE" --nsamples "$NSAMPLES" --h-pi 1.0 --save_path "./output/llama-2-13b-GPTQ-${DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}"
PREV_DATASET=$DATASET
for DATASET in c4 boolq piqa winogrande; do
  H_IN="./output/llama-2-13b-GPTQ-${PREV_DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}/h_out.pt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_quant.py --method c-gptq --config "$CONFIG" --dataset "$DATASET" --bits "$BITS" --group_size "$GROUP_SIZE" --nsamples "$NSAMPLES" --h-in "$H_IN" --h-pi 1.0 --save_path "./output/llama-2-13b-GPTQ-${PREV_DATASET}_${DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}"
  PREV_DATASET="${PREV_DATASET}_${DATASET}"
done
