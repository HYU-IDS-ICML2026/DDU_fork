# CIFAR-10 ViT: AdamW와 SAM(AdamW) 비교 실험

이 브랜치는 [DDU(Deep Deterministic Uncertainty)](https://arxiv.org/abs/2102.11582) 코드베이스에 CIFAR-10용 Vision Transformer와 SAM 학습 경로를 연결한 연구용 구현입니다. 현재 핵심 실험은 동일한 ViT와 학습 설정을 유지하면서 아래 세 조건을 비교하는 것입니다.

- AdamW
- SAM(base optimizer=AdamW), `rho=0.05`
- SAM(base optimizer=AdamW), `rho=0.5`

각 조건을 seed `0`, `1`, `2`로 학습하며, 최종 평가는 세 seed의 평균과 표준편차로 보고하는 것을 전제로 합니다. 모델은 pretrained weight 없이 CIFAR-10에서 처음부터 학습합니다.

## 1. 모델: `vit_tiny_patch4_32`

구현 파일은 [`net/vit.py`](net/vit.py)입니다. `timm.models.vision_transformer.VisionTransformer`를 얇게 감싸며, `timm` 소스를 저장소에 복사하지 않습니다.

| 항목 | 설정 |
|---|---:|
| 입력 | RGB `32 x 32` |
| patch 크기 | `4 x 4` |
| patch token 수 | `8 x 8 = 64` |
| class token | 사용 |
| embedding 차원 | `192` |
| Transformer depth | `12` |
| attention heads | `3` |
| MLP ratio | `4.0` |
| dropout / stochastic depth | `0.0` |
| classifier 출력 | CIFAR-10의 10 classes |
| pretrained | 사용하지 않음 |

`ViT-Tiny/4`라는 표현에서 `Tiny`는 `embed_dim=192`, `depth=12`, `num_heads=3`인 작은 ViT 구성을, `/4`는 한 patch의 한 변이 4 pixel임을 뜻합니다. CIFAR-10처럼 32x32인 작은 영상에 ImageNet에서 흔히 사용하는 16x16 patch를 적용하면 token이 4개뿐이므로, 여기서는 4x4 patch로 공간 정보를 더 세밀하게 유지합니다.

DDU 평가 코드와 호환되도록 wrapper는 다음 인터페이스를 제공합니다.

- `net.feature`: classifier 직전의 최종 LayerNorm 이후 CLS representation, shape `[batch, 192]`
- `net.fc`: `Linear(192, 10)` classifier
- `net.feature_dim`: `192`
- logits: `net.fc(feature) / temperature`

현재 이 ViT wrapper는 spectral normalization(`-sn`), CNN 구조 변경(`-mod`), MNIST 입력을 지원하지 않으며 해당 옵션을 사용하면 명시적으로 실패합니다.

## 2. 학습 설정

현재 9-run 비교 실험의 공통 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| dataset split | CIFAR-10 train 45,000 / validation 5,000 |
| epochs | `200` |
| batch size | `64` |
| learning rate | `1e-4` |
| weight decay | `5e-4` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| label smoothing | `0.1` |
| scheduler | cosine annealing |
| augmentation | RandomCrop + HorizontalFlip + CIFAR-10 AutoAugment |
| temperature | `1.0` |
| checkpoint interval | 50 epochs |

AdamW와 SAM(AdamW)은 동일한 parameter list, learning rate, betas, epsilon, weight decay를 사용합니다. 유일한 optimizer 차이는 SAM이 각 minibatch에서 다음 두 단계를 수행한다는 점입니다.

1. 원래 parameter에서 gradient를 계산하고 반경 `rho`의 perturbation을 적용합니다.
2. perturbation된 parameter에서 다시 gradient를 계산한 뒤 원래 parameter로 복원하고 AdamW update를 수행합니다.

학습 로그의 loss와 accuracy는 두 조건을 공정하게 비교할 수 있도록 첫 번째, 즉 perturbation 전 forward에서 기록합니다.

## 3. 환경 설치

권장 환경은 [`environment.yml`](environment.yml)에 기록되어 있으며 `timm==1.0.27`을 고정합니다.

```bash
conda env create -f environment.yml
conda activate environment
```

CIFAR-10은 첫 실행 시 `torchvision`이 `./data` 아래로 내려받습니다. 공유 서버에서는 기존 데이터가 있다면 `data/cifar-10-batches-py`가 보이도록 준비하면 됩니다.

## 4. 단일 run 실행

아래 명령에서 `<output-dir>`은 조건과 seed마다 서로 다른 디렉터리를 사용해야 합니다.

### AdamW

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --seed 0 \
  --dataset cifar10 \
  --model vit_tiny_patch4_32 \
  --opt adamw \
  --rho 0.0 \
  -e 200 -b 64 \
  --lr 1e-4 --decay 5e-4 \
  --beta1 0.9 --beta2 0.999 --adam-eps 1e-8 \
  --label-smoothing 0.1 \
  --scheduler cosine \
  --data-aug --autoaugment \
  --save-interval 50 --log-interval 100 \
  --save-path <output-dir>
```

### SAM(AdamW)

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --seed 0 \
  --dataset cifar10 \
  --model vit_tiny_patch4_32 \
  --opt sam_adamw \
  --rho 0.05 \
  -e 200 -b 64 \
  --lr 1e-4 --decay 5e-4 \
  --beta1 0.9 --beta2 0.999 --adam-eps 1e-8 \
  --label-smoothing 0.1 \
  --scheduler cosine \
  --data-aug --autoaugment \
  --save-interval 50 --log-interval 100 \
  --save-path <output-dir>
```

`rho=0.5` 실험은 위 명령의 `--rho`만 `0.5`로 변경합니다. seed 1과 2도 `--seed`와 출력 디렉터리만 변경하며 나머지 인자는 고정합니다.

각 출력 디렉터리에는 다음 자료가 저장됩니다.

- `training_args.json`: 프로그램이 실제로 해석한 전체 인자
- `*.model`: checkpoint
- `*_train_loss.json`, `*_train_accuracy.json`: epoch별 수치
- `stats_logging/`: TensorBoard event

## 5. 9-run sweep 실행

재현용 정의와 실행기는 [`experiments/vit_adamw_sam_3seed_20260722`](experiments/vit_adamw_sam_3seed_20260722)에 있습니다.

- `manifest.json`: 공통 설정과 9개 run의 machine-readable 정의
- `preflight.sh`: CUDA, dependency, 모델 forward, AdamW/SAM step 검사
- `launch.sh`: 한 GPU에서 9개 run을 순차 실행
- `start_screen.sh`: SSH 연결이 끊겨도 실행이 유지되도록 GNU Screen session 시작
- `monitor.sh`: 상태와 현재 로그의 마지막 40줄 출력

```bash
bash experiments/vit_adamw_sam_3seed_20260722/start_screen.sh
bash experiments/vit_adamw_sam_3seed_20260722/monitor.sh
```

직접 session에 들어가려면 다음을 사용합니다.

```bash
screen -r vit_adamw_sam_3seed
```

`Ctrl-a` 다음 `d`를 누르면 학습을 종료하지 않고 다시 detach합니다. 실행기는 시작 시 학습 핵심 코드의 snapshot과 SHA-256을 기록하며, sweep 도중 코드가 바뀌면 서로 다른 구현이 한 실험에 섞이지 않도록 다음 run 시작 전에 중단합니다.

기본 launcher는 `CUDA_VISIBLE_DEVICES=0`을 사용합니다. 다른 GPU를 사용하려면 실행 전에 [`launch.sh`](experiments/vit_adamw_sam_3seed_20260722/launch.sh)의 GPU 번호를 명시적으로 변경하십시오.

## 6. 단일 checkpoint 평가

`evaluate_v2.py`는 accuracy, calibration, 여러 OOD score, DDU/GMM 및 geometry 경로를 평가합니다. checkpoint에 `module.` prefix가 있어도 제거하여 읽습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_v2.py \
  --checkpoint_path <checkpoint.model> \
  --dataset cifar10 \
  --ood_dataset svhn \
  --model vit_tiny_patch4_32 \
  --batch_size 128 \
  --seed 0 \
  --gpu \
  --output_dir <evaluation-output-dir>
```

DDU 또는 geometry 계산이 실패하면 `0.0`이라는 정상 AUROC처럼 숨기지 않고 JSON에 `null`과 오류 메시지를 기록합니다. 세 seed의 평균·표준편차 집계는 평가 JSON을 모두 만든 뒤 별도 집계 단계에서 수행해야 하며, 이 브랜치에는 아직 전용 집계 스크립트를 추가하지 않았습니다.

## 7. 주요 변경 파일

- [`net/vit.py`](net/vit.py): CIFAR-10 ViT와 DDU feature interface
- [`train.py`](train.py): ViT registry, AdamW/SAM(AdamW), cosine scheduler, 재현성 및 인자 기록
- [`utils/train_utils.py`](utils/train_utils.py): SAM two-pass, label smoothing, loss/accuracy logging
- [`utils/args.py`](utils/args.py): 새 optimizer 관련 CLI 인자
- [`data/ood_detection/cifar10.py`](data/ood_detection/cifar10.py): optional CIFAR-10 AutoAugment
- [`evaluate_v2.py`](evaluate_v2.py), [`evaluate_ddu_pca.py`](evaluate_ddu_pca.py): ViT feature dimension과 평가 registry

기존 [`utils/sam.py`](utils/sam.py)의 SAM 구현은 수정하지 않고 base optimizer에 `torch.optim.AdamW`를 전달하여 재사용합니다.

## 8. 구현 근거

- Vision Transformer: [Dosovitskiy et al., *An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929)
- AdamW: [Loshchilov and Hutter, *Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101)
- SAM: [Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412)
- ViT implementation: [Hugging Face pytorch-image-models (`timm`)](https://github.com/huggingface/pytorch-image-models)
- DDU: [Mukhoti et al., *Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty*](https://arxiv.org/abs/2102.11582)

## 9. 현재 범위와 주의사항

- ViT는 CIFAR-10 scratch 학습만 검증 대상으로 합니다.
- ViT에 spectral normalization이나 DDU 논문의 CNN architectural modification을 억지로 적용하지 않습니다.
- AdamW와 SAM(AdamW)의 비교에서는 `rho` 외의 공통 optimizer 설정을 변경하지 마십시오.
- SAM은 minibatch당 forward/backward를 두 번 수행하므로 AdamW보다 학습 시간이 길고 메모리보다는 계산량 증가가 큽니다.
- 장기 sweep을 다시 시작할 때는 기존 `runs/`, `status.tsv`, `source_snapshot/`을 덮어쓰지 않습니다. 보존하거나 실험 디렉터리를 새 이름으로 복제한 뒤 실행하십시오.

## License

원본 DDU 저장소와 동일하게 MIT License를 따릅니다. 자세한 내용은 [`LICENSE`](LICENSE)를 확인하십시오.
