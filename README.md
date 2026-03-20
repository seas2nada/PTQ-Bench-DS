# PTQ-Bench-DS

PTQ-Bench-DS is a distilled subset of PTQ-Bench that keeps only the following method implementations and evaluation utilities:

- GPTQ
- C-GPTQ
- AWQ
- OmniQuant
- QuIP
- Perplexity evaluation (`eval_ppl.py`)
- Zero-shot evaluation via `lm-evaluation-harness`

This directory is intended to be a smaller, easier-to-share codebase focused on PTQ method runs and evaluation.

## Current support status

- `gptq`: standard GPTQ quantization
- `c-gptq`: GPTQ-based continual quantization with Hessian carry-over (`h_in`, `h_out`, `h_pi`, optional SPD controls)
- `awq`: standard AWQ flow
- `omniquant`: standard OmniQuant flow
- `quip`: standard QuIP flow

Important limitation:

- Continual PTQ is currently implemented only for the GPTQ-based continual path (`c-gptq`).
- AWQ, OmniQuant, and QuIP are included as single-task baselines and do not currently have continual extensions in this repository.

## Repository layout

```text
PTQ-Bench-DS/
├── OmniQuant/
├── QuIP/
├── awq/
├── gptq/
├── c-gptq/
├── configs/
├── scripts/
├── lm-evaluation-harness/
├── run_quant.py
├── eval_ppl.py
├── test_ppl.bash
└── lm_eval.sh
```

## Method configs

The launcher uses the YAML files in `configs/`:

- `configs/gptq.yaml`
- `configs/c-gptq.yaml`
- `configs/awq.yaml`
- `configs/omniquant.yaml`
- `configs/quip.yaml`

List the registered methods:

```bash
python3 run_quant.py --method gptq --config configs/gptq.yaml --list
```

## Running quantization

Use the common launcher directly:

```bash
python3 run_quant.py --method gptq --config configs/gptq.yaml
python3 run_quant.py --method c-gptq --config configs/c-gptq.yaml
python3 run_quant.py --method awq --config configs/awq.yaml
python3 run_quant.py --method omniquant --config configs/omniquant.yaml
python3 run_quant.py --method quip --config configs/quip.yaml
```

## Bash wrappers

Convenience wrappers are provided in `scripts/`.

### GPTQ

```bash
bash scripts/run_gptq.sh 0 configs/gptq.yaml wikitext2 3 128 128
```

Arguments:

```text
run_gptq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]
```

### AWQ

```bash
bash scripts/run_awq.sh 0 configs/awq.yaml wikitext2 4 128 128
```

Arguments:

```text
run_awq.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [SAVE_PATH]
```

### QuIP

```bash
bash scripts/run_quip.sh 0 configs/quip.yaml
```

Arguments:

```text
run_quip.sh GPU CONFIG [DATASET] [BITS] [NSAMPLES] [SAVE_PATH]
```

### OmniQuant

```bash
bash scripts/run_omniquant.sh 0 configs/omniquant.yaml
```

Arguments:

```text
run_omniquant.sh GPU CONFIG
```

### Continual GPTQ

Single step:

```bash
bash scripts/run_cgpt.sh 0 configs/c-gptq.yaml wikitext2 3 128 128
```

Arguments:

```text
run_cgpt.sh GPU CONFIG DATASET BITS GROUP_SIZE NSAMPLES [H_IN] [SAVE_PATH]
```

Multi-step examples:

```bash
bash scripts/run_cgptq2.sh 0 configs/c-gptq.yaml 3 128 128
bash scripts/run_cgptq_spd.sh 0 configs/c-gptq.yaml 3 128 128 logeuc
```

## Evaluation

### Perplexity

```bash
bash scripts/eval_ppl.sh 0 /path/to/model
```

Root-level compatibility wrapper:

```bash
bash test_ppl.bash 0 /path/to/model
```

### Zero-shot tasks

```bash
bash scripts/lm_eval.sh 0 /path/to/model
```

Optional arguments:

```text
lm_eval.sh GPU MODEL [TASKS] [BATCH_SIZE] [OUTPUT_DIR]
```

Example:

```bash
bash scripts/lm_eval.sh 0 /path/to/model boolq,piqa,winogrande 1 ./results/demo
```

The wrapper sets `PYTHONPATH` to the bundled `lm-evaluation-harness` directory before calling `python3 -m lm_eval`.

## Environment notes

- The wrappers in this distilled repo use `python3`.
- Some method subdirectories keep the dependency assumptions of their original upstream implementations.
- AWQ usually requires its package install and kernel build steps.
- OmniQuant and QuIP may also require their original dependency stacks depending on the target model.

## Origin

This repository was extracted from a larger PTQ-Bench working tree to keep only the PTQ methods and evaluation components relevant to:

- OmniQuant
- QuIP
- AWQ
- GPTQ
- C-GPTQ
- evaluation
