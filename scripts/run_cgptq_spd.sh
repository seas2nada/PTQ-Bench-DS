#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=${1:?usage: run_cgptq_spd.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES SPDMODE}
CONFIG=${2:?usage: run_cgptq_spd.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES SPDMODE}
BITS=${3:?usage: run_cgptq_spd.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES SPDMODE}
GROUP_SIZE=${4:?usage: run_cgptq_spd.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES SPDMODE}
NSAMPLES=${5:?usage: run_cgptq_spd.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES SPDMODE}
SPDMODE=${6:?usage: run_cgptq_spd.sh GPU CONFIG BITS GROUP_SIZE NSAMPLES SPDMODE}
DATASET=wikitext2
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_quant.py --method c-gptq --config "$CONFIG" --dataset "$DATASET" --bits "$BITS" --group_size "$GROUP_SIZE" --nsamples "$NSAMPLES" --save_path "./output/llama-2-7b-SPD-${DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}"
PREV_DATASET=$DATASET
for DATASET in boolq piqa winogrande; do
  H_IN="./output/llama-2-7b-SPD-${PREV_DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}/h_out.pt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_quant.py --method c-gptq --config "$CONFIG" --dataset "$DATASET" --bits "$BITS" --group_size "$GROUP_SIZE" --nsamples "$NSAMPLES" --h-in "$H_IN" --use_spd --spdmode "$SPDMODE" --save_path "./output/llama-2-7b-SPD-${PREV_DATASET}_${DATASET}-${BITS}bit-g${GROUP_SIZE}-nsamples${NSAMPLES}"
  PREV_DATASET="${PREV_DATASET}_${DATASET}"
done
