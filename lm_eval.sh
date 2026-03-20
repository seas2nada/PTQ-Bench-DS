#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTHONPATH="$ROOT_DIR/lm-evaluation-harness${PYTHONPATH:+:$PYTHONPATH}"
GPU=${1:?usage: lm_eval.sh GPU MODEL [TASKS] [BATCH_SIZE] [OUTPUT_DIR]}
MODEL=${2:?usage: lm_eval.sh GPU MODEL [TASKS] [BATCH_SIZE] [OUTPUT_DIR]}
TASKS=${3:-boolq,piqa,winogrande,hellaswag,arc_easy,arc_challenge}
BATCH_SIZE=${4:-1}
OUTPUT_DIR=${5:-./results/$(basename "$MODEL")}
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m lm_eval --model hf --model_args "pretrained=$MODEL" --tasks "$TASKS" --batch_size "$BATCH_SIZE" --output "$OUTPUT_DIR" --confirm_run_unsafe_code
