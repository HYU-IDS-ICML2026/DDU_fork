# CIFAR-10 ViT: AdamW 대 SAM(AdamW), 3-seed sweep

이 디렉터리는 단일 GPU에서 아래 9개 학습을 순차적으로 실행하기 위한 재현 패키지입니다.

1. AdamW: seed 0, 1, 2
2. SAM(AdamW), `rho=0.05`: seed 0, 1, 2
3. SAM(AdamW), `rho=0.5`: seed 0, 1, 2

## 고정 학습 설정

```text
model: vit_tiny_patch4_32
dataset: cifar10
epochs: 200
batch_size: 64
learning_rate: 1e-4
weight_decay: 5e-4
betas: [0.9, 0.999]
adam_eps: 1e-8
label_smoothing: 0.1
scheduler: cosine
augmentation: RandomCrop + RandomHorizontalFlip + CIFAR10 AutoAugment
pretrained: false
spectral_normalization: false
architectural_modification: false
temperature: 1.0
save_interval: 50 epochs
log_interval: 100 batches
visible_gpu: 0
```

[`manifest.json`](manifest.json)은 위 설정과 각 run을 기계가 읽을 수 있는 JSON으로 기록합니다. 각 run은 다음 파일을 생성합니다.

- `runs/<run_id>/command.txt`: shell이 실제 실행한 명령
- `runs/<run_id>/training_args.json`: Python이 파싱한 전체 인자
- `runs/<run_id>/environment.txt`: branch, commit, 라이브러리 및 GPU 정보
- `logs/<run_id>.log`: 해당 run의 표준 출력과 오류

실행 시작 시 학습 핵심 파일을 `source_snapshot/`에 복사하고 `code_sha256.txt`에 hash를 저장합니다. 다음 run을 시작하기 전에 hash를 검사하므로 sweep 도중 학습 코드가 달라지면 실행을 중단합니다.

## 실행과 확인

GNU Screen detached session으로 시작합니다.

```bash
bash experiments/vit_adamw_sam_3seed_20260722/start_screen.sh
```

현재 상태와 compact log를 확인합니다.

```bash
bash experiments/vit_adamw_sam_3seed_20260722/monitor.sh
```

session에 직접 접속하려면 다음을 사용합니다.

```bash
screen -r vit_adamw_sam_3seed
```

`Ctrl-a` 다음 `d`를 누르면 학습을 유지한 채 detach합니다. `current.log`는 현재 실행 중인 run의 로그를 가리키며, 첫 실패가 발생하면 launcher는 다음 조건으로 넘어가지 않고 종료합니다.

## 재실행 주의사항

launcher는 기존 결과를 덮어쓰지 않습니다. `status.tsv`, `runs/`, `source_snapshot/` 등이 이미 있는 동일 디렉터리에서는 재시작이 거부됩니다. 이전 실험을 보존한 뒤 새 실험 디렉터리를 만들거나, 기존 산출물의 의미를 확인하고 별도로 정리해야 합니다.
