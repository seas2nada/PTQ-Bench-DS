#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=${1:?usage: run_quip.sh GPU CONFIG [DATASET] [BITS] [NSAMPLES] [SAVE_PATH]}
CONFIG=${2:?usage: run_quip.sh GPU CONFIG [DATASET] [BITS] [NSAMPLES] [SAVE_PATH]}
DATASET=${3:-}
BITS=${4:-}
NSAMPLES=${5:-}
SAVE_PATH=${6:-}
ARGS=(run_quant.py --method quip --config "$CONFIG")
if [[ -n "$DATASET" ]]; then ARGS+=(--dataset "$DATASET"); fi
if [[ -n "$BITS" ]]; then ARGS+=(--bits "$BITS"); fi
if [[ -n "$NSAMPLES" ]]; then ARGS+=(--nsamples "$NSAMPLES"); fi
if [[ -n "$SAVE_PATH" ]]; then ARGS+=(--save_path "$SAVE_PATH"); fi
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "${ARGS[@]}"
