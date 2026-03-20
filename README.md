# PTQ-Bench-DS

PTQ-Bench-DS는 DS 프로젝트에서 사용하기 위해 정리한 PTQ-Bench 기반 저장소입니다.

이 저장소는 [zjq0455/PTQ-Bench](https://github.com/zjq0455/PTQ-Bench)를 기반으로 필요한 메서드와 평가 코드 중심으로 구성했습니다.

포함된 주요 구성은 다음과 같습니다.

- GPTQ
- C-GPTQ
- AWQ
- OmniQuant
- QuIP
- perplexity 평가 코드 (`eval_ppl.py`)
- zero-shot 평가 코드 (`lm-evaluation-harness`, `lm_eval.sh`)

## 현재 지원 상태

- `gptq`: 일반 GPTQ 양자화
- `c-gptq`: Hessian 누적 기반 continual GPTQ
- `awq`: 일반 AWQ
- `omniquant`: 일반 OmniQuant
- `quip`: 일반 QuIP

중요한 제한사항:

- 현재 continual PTQ는 GPTQ 계열의 `c-gptq` 경로에만 구현되어 있습니다.
- `awq`, `omniquant`, `quip` 에는 continual 확장이 아직 들어가 있지 않습니다.
- 즉, 이 세 방법은 현재 단일 태스크 기준 baseline 용도로만 포함되어 있습니다.

## 디렉토리 구성

```text
PTQ-Bench-DS/
├── OmniQuant/
├── QuIP/
├── awq/
├── gptq/
├── c-gptq/
├── configs/
├── lm-evaluation-harness/
├── run_quant.py
├── run_gptq.sh
├── run_awq.sh
├── run_cgpt.sh
├── run_cgptq2.sh
├── run_cgptq_spd.sh
├── eval_ppl.py
├── test_ppl.bash
└── lm_eval.sh
```

## Docker 기반 설치 가이드

실험은 Docker Hub의 [`seas2nada/ptq_docker`](https://hub.docker.com/repository/docker/seas2nada/ptq_docker/general) 이미지를 기반으로 진행했습니다.

2026년 3월 20일 기준 Docker Hub 공개 태그는 `26.02.02` 입니다.

### 1. Docker 이미지 pull

```bash
docker pull seas2nada/ptq_docker:26.02.02
```

### 2. 컨테이너 실행

아래 예시는 현재 실험 환경과 동일하게 `/home` 과 `/DB` 를 bind mount 하는 방식입니다.

```bash
docker run --gpus all -it \
  --name ptq_docker \
  --shm-size 64g \
  -v /home:/home \
  -v /DB:/DB \
  -w /home/ptq_docker/Workspace \
  seas2nada/ptq_docker:26.02.02 \
  bash
```

### 3. 레포 준비

컨테이너 내부에서 작업합니다.

```bash
cd /home/ptq_docker/Workspace
git clone https://github.com/seas2nada/PTQ-Bench-DS.git
cd PTQ-Bench-DS
```

### 4. 설치 확인

```bash
python run_quant.py --method gptq --config configs/gptq.yaml --list
```

참고:

- root bash script는 `python` 명령을 사용합니다.
- 따라서 README의 실행 예시는 컨테이너 내부 환경을 기준으로 작성했습니다.
- 호스트에서 직접 실행할 경우에는 Python alias 및 의존성 구성을 별도로 맞춰야 합니다.

## 설정 파일

공통 launcher는 `configs/` 아래 YAML을 사용합니다.

- `configs/gptq.yaml`
- `configs/c-gptq.yaml`
- `configs/awq.yaml`
- `configs/omniquant.yaml`
- `configs/quip.yaml`

등록된 메서드 확인:

```bash
python run_quant.py --method gptq --config configs/gptq.yaml --list
```

## 실행 방식

### 1. 공통 launcher 사용

```bash
python run_quant.py --method gptq --config configs/gptq.yaml
python run_quant.py --method c-gptq --config configs/c-gptq.yaml
python run_quant.py --method awq --config configs/awq.yaml
python run_quant.py --method omniquant --config configs/omniquant.yaml
python run_quant.py --method quip --config configs/quip.yaml
```

### 2. 원본 PTQ-Bench의 bash 스크립트 그대로 사용

아래 스크립트들은 `/home/ptq_docker/Workspace/PTQ-Bench` 에 있던 root bash script를 그대로 복사한 것입니다.

- `run_gptq.sh`
- `run_awq.sh`
- `run_cgpt.sh`
- `run_cgptq2.sh`
- `run_cgptq_spd.sh`
- `test_ppl.bash`
- `lm_eval.sh`

예시:

```bash
bash run_gptq.sh 0 configs/gptq.yaml wikitext2 3 128 128
bash run_awq.sh 0 configs/awq.yaml wikitext2 4 128 128
bash run_cgpt.sh 0 configs/c-gptq.yaml wikitext2 3 128 128
bash test_ppl.bash 0 /path/to/model
bash lm_eval.sh 0 /path/to/model
```

주의:

- 위 bash 스크립트는 원본과 동일하게 `python` 을 호출합니다.
- 따라서 실행 환경에도 원본 PTQ-Bench 와 동일한 Python/패키지 구성이 필요합니다.
- `OmniQuant` 와 `QuIP` 는 원본 root 기준 별도 bash wrapper가 없어서 `run_quant.py` 로 실행하면 됩니다.

## 평가

### Perplexity 평가

```bash
python eval_ppl.py --model /path/to/model
```

또는

```bash
bash test_ppl.bash 0 /path/to/model
```

### Zero-shot 평가

```bash
bash lm_eval.sh 0 /path/to/model
```

기본 task set은 다음과 같습니다.

- `boolq`
- `piqa`
- `winogrande`
- `hellaswag`
- `arc_easy`
- `arc_challenge`

## 메모

- 이 레포는 DS 프로젝트에서 바로 참고할 수 있도록 필요한 메서드와 평가 코드만 남긴 버전입니다.
- 원본 PTQ-Bench의 전체 실험 코드나 다른 방법들은 의도적으로 제외했습니다.
