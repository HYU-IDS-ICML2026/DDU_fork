# ViM·kNN 전용 OOD 평가 파이프라인 작업 지시서

## 0. 목적

학과 서버의 기존 코드를 먼저 읽기 전용으로 조사한 뒤, 사용자와 구현안을 협의하고 승인받아 **ViM과 kNN만 별도로 평가하는 전용 파이프라인**을 구현한다.

첫 실행에서는 반드시 **1단계 읽기 전용 감사만 수행**한다. 코드 수정, 파일 생성, symlink 생성, dependency 설치, dataset 다운로드, 모델 평가 실행은 사용자 승인 이후에만 수행한다.

---

## 1. 고정된 실험 범위

```text
ID dataset: CIFAR-10
Backbone: WRN-28-10
ViM D: 256, 320
kNN k: 기본값 50
OOD: CIFAR-100, Tiny-ImageNet, SVHN, MNIST
평가 metric: AUROC, FPR@95
학습 seed: 0, 1, 2
최종 보고: 동일 실험군 seed 0/1/2의 mean ± standard deviation
```

ViM과 kNN은 같은 penultimate feature를 재사용하지만 서로 독립적인 detector로 계산하고 저장한다.

이 작업은 다음을 하지 않는다.

- 모델 재학습
- ViM과 kNN score 결합
- OOD test set을 이용한 D 또는 k 선택
- Energy/Residual의 개선·악화 자동 판정

---

## 2. 서버 기준 경로

### 작업 repository

```text
/home/ghjin/DDU_fork_0724VIM
```

우선 확인할 파일:

```text
/home/ghjin/DDU_fork_0724VIM/run_eval.py
/home/ghjin/DDU_fork_0724VIM/evaluate_v2.py
```

이 두 파일과 import 경로에서 다음 정의를 최대한 재사용한다.

- checkpoint loading
- WRN-28-10 생성
- Spectral Normalization patch
- `mod`, `coeff`, `temp` 처리
- feature extraction
- dataset loader
- 기존 kNN
- AUROC 계산
- JSON 직렬화

### 공유 dataset

```text
/home/ghjin/ICML/DDU_fork/data
```

현재 loader가 `./data`를 하드코딩했을 수 있으므로 실제 directory 구조를 먼저 확인한다. 사용자 승인 전 symlink를 만들거나 dataset을 다시 다운로드하지 않는다.

### checkpoint root

```text
/home/ghjin/all_models_260328/models_260328/CIFAR-10/WideResNet
```

대표 파일:

```text
/home/ghjin/all_models_260328/models_260328/CIFAR-10/WideResNet/cifar10_sam_0.5wide_resnet_mod_0_350.model
```

파일명에서 `mod_<숫자>`의 숫자가 학습 seed 0/1/2인 것으로 예상되지만, 전체 파일 목록을 확인하기 전에는 parser를 확정하지 않는다.

---

# 3. 1단계: 읽기 전용 서버 감사

## 3.1 허용

- 파일 읽기
- directory listing
- import 경로 추적
- git 상태 확인
- checkpoint 파일명 manifest 작성
- dependency 존재 여부 확인
- 구현안 작성
- 사용자에게 질문

## 3.2 금지

- 코드 수정
- 새 파일 생성
- symlink 생성
- dependency 설치
- dataset 다운로드
- 모델 평가 실행
- 장시간 GPU 사용
- commit, push
- checkpoint 이동 또는 삭제

## 3.3 repository 상태

```bash
cd /home/ghjin/DDU_fork_0724VIM
git branch --show-current
git status --short
git log -1 --oneline
```

기존 미커밋 변경이 있으면 사용자 승인 없이 덮어쓰지 않는다.

## 3.4 코드 감사 항목

`run_eval.py`, `evaluate_v2.py`와 실제 import 파일에서 다음을 근거 위치와 함께 확인한다.

1. WRN-28-10 생성 방식
2. 실제 penultimate feature 위치
3. 실제 feature dimension
4. classifier layer 이름
5. checkpoint 구조와 `module.` prefix 처리
6. SN, `mod`, `coeff`, `temp` 복원 방식
7. CIFAR-10 fitting train loader
8. train/validation split 생성 방식
9. split seed 전달 위치
10. OOD loader별 data root 처리
11. 기존 kNN score 정의와 batching
12. 높은 score가 ID인지 여부
13. AUROC 구현
14. FPR@95 구현 존재 여부
15. feature 저장 device와 dtype
16. 기존 결과 파일 naming과 schema

예상 관련 파일은 다음과 같지만 실제 repository 구조를 따른다.

```text
utils/ood_scores.py
utils/gmm_utils.py
metrics/ood_metrics.py
net/wide_resnet.py
data/ood_detection/cifar10.py
data/ood_detection/cifar100.py
data/ood_detection/tiny_imagenet.py
data/ood_detection/svhn.py
data/ood_detection/mnist_ood.py
```

## 3.5 dataset 감사

다음을 읽기 전용으로 조사한다.

```bash
find /home/ghjin/ICML/DDU_fork/data -maxdepth 3 -type d | sort
```

확인 대상:

```text
CIFAR-10
CIFAR-100
Tiny-ImageNet
SVHN
MNIST
```

다음 중 어떤 방식이 더 안전한지 제안한다.

- `--data_root /home/ghjin/ICML/DDU_fork/data` 추가
- repository의 `data`에 symlink 사용

symlink는 사용자 승인 전 생성하지 않는다.

## 3.6 checkpoint manifest 감사

```bash
find /home/ghjin/all_models_260328/models_260328/CIFAR-10/WideResNet   -maxdepth 1 -type f -name '*.model' | sort
```

다음 표를 만든다.

```text
experiment_group | optimizer/method | hyperparameter | seed | epoch | checkpoint_path
```

확인:

- 각 실험군에 seed 0, 1, 2가 모두 존재하는가
- 중복 checkpoint가 있는가
- 350 epoch 이외 파일이 섞여 있는가
- filename parser가 모든 파일에 일관되게 적용되는가

예상 regex는 참고용일 뿐이다.

```python
r"_mod_(?P<seed>[012])_(?P<epochs>\d+)\.model$"
```

## 3.7 seed 처리 설계

기존 평가 코드의 `seed=0` 하드코딩을 제거해야 한다.

권장 interface:

```text
--seed auto
```

동작:

```text
checkpoint filename의 학습 seed
= evaluation seed
= CIFAR-10 split seed
```

가능하면 다음에도 같은 seed를 명시적으로 전달한다.

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
DataLoader split
```

다음 경우에는 조용히 seed 0을 쓰지 말고 실패한다.

- filename에서 seed를 찾을 수 없음
- seed token이 모호함
- filename seed와 명시적 `--seed`가 다름
- seed가 0, 1, 2가 아님

모든 결과 metadata에 저장:

```text
checkpoint_path
parsed_training_seed
evaluation_seed
split_seed
```

## 3.8 1단계 완료 보고

다음을 한국어로 보고한다.

### A. 서버 상태

```text
branch
working tree
Python 환경
GPU 사용 가능 여부
필수 dependency 존재 여부
```

### B. 기존 코드 재사용 계획

```text
run_eval.py에서 재사용할 부분
evaluate_v2.py에서 재사용할 부분
새로 구현할 부분
```

### C. dataset 연결 계획

```text
실제 dataset 경로
loader 호환 여부
--data_root 또는 symlink 권고
```

### D. checkpoint manifest

```text
실험군
seed triplet
누락 seed
중복
parser 제안
```

### E. 구현 파일 제안

예상 후보:

```text
evaluate_vim_knn.py
run_vim_knn_batch.py
aggregate_vim_knn_results.py
tests/test_vim_knn.py
```

더 적은 파일이 적절하면 이유를 설명한다.

### F. 계산 비용과 위험

```text
checkpoint 하나당 feature extraction 횟수
feature memory
exact kNN 계산 비용
전체 checkpoint 수
예상 실행 시간
```

### G. 사용자 결정이 필요한 질문

최소한 다음을 확인한다.

- 최종 output root
- 표준편차 `ddof=0` 또는 `ddof=1`
- 누락 seed 처리
- 350 epoch checkpoint만 평가할지
- SN/mod 설정 식별 방식

마지막에는 반드시 멈춘다.

```text
읽기 전용 감사를 완료했습니다. 아직 파일을 수정하거나 평가를 시작하지 않았습니다. 위 구현안을 승인하거나 수정해 주세요.
```

---

# 4. 2단계: 승인 후 구현할 목표

사용자 승인 후에만 수행한다.

## 4.1 단일 checkpoint evaluator

권장 파일:

```text
evaluate_vim_knn.py
```

한 checkpoint에서 다음 순서로 처리한다.

1. checkpoint seed 파싱
2. 모델 생성과 checkpoint load
3. CIFAR-10 fitting train feature 한 번 추출
4. CIFAR-10 test feature 한 번 추출
5. OOD 4종 feature 각각 한 번 추출
6. ViM D=256, D=320 계산
7. kNN k=50 계산
8. raw score 저장
9. AUROC와 FPR@95 저장

## 4.2 CLI

```text
--checkpoint_path PATH
--data_root PATH
--ood_datasets NAME [NAME ...]
--vim_dims D [D ...]
--knn_k K
--seed auto|INT
--output_dir PATH
--batch_size INT
--gpu
```

오늘 실험 기본값:

```text
--data_root /home/ghjin/ICML/DDU_fork/data
--ood_datasets cifar100 tiny_imagenet svhn mnist
--vim_dims 256 320
--knn_k 50
--seed auto
--batch_size 128
```

`--vim_dims`는 하나 또는 여러 값을 받는다.

```bash
--vim_dims 256
--vim_dims 256 320
```

검사:

```text
0 < D < feature_dim
1 <= k <= ID fitting sample 수
```

ViM D와 kNN k를 함수 내부에 하드코딩하지 않는다.

---

# 5. ViM 정의

ViM fitting에는 다음만 사용한다.

```text
CIFAR-10 fitting train feature
classifier W
classifier b
```

OOD test feature나 OOD AUROC로 D를 선택하지 않는다.

계산:

```text
u = -pinv(W) @ b
shifted = feature - u

ID train shifted feature로 second moment/covariance 계산

상위 D eigenvectors = principal space
나머지 eigenvectors = residual subspace NS

residual = ||(feature-u) @ NS||_2

alpha =
    mean_ID_train(max_raw_logit)
    / mean_ID_train(residual)

virtual_logit = alpha * residual
energy = logsumexp(raw_logits)
vim_score = energy - virtual_logit
```

score convention:

```text
vim_score가 클수록 ID
```

raw logits:

```text
feature @ W.T + b
```

Temperature-scaled logits은 사용하지 않는다.

수치 조건:

- NumPy float64
- `np.linalg.pinv`
- 대칭 covariance에 `np.linalg.eigh`
- residual mean과 alpha의 finite 검사
- 실패를 `metric=0`으로 저장하지 않음

---

# 6. kNN 정의

기존 `KNNScorer`를 확인하고 가능한 한 그대로 재사용한다.

의도한 score:

```text
각 sample의 ID fitting train feature에 대한 k번째 최근접 Euclidean distance
knn_score = -k번째 거리
```

```text
높을수록 ID
```

default:

```text
k = 50
```

전체 거리 행렬을 한 번에 만들지 말고 안전한 batching/chunking을 사용한다.

---

# 7. metric과 diagnostic score

필수:

```text
AUROC
FPR@95
```

모든 metric 계산에서 높은 값이 ID가 되도록 한다.

```text
ViM: vim_score
kNN: -kth_distance
Energy diagnostic: energy
Residual diagnostic: -residual
Virtual Logit diagnostic: -virtual_logit
```

raw 파일에는 residual과 virtual logit을 양수 원값으로 저장한다.

---

# 8. Energy·Residual 원본 저장

이번 구현은 자동으로 “개선/악화”를 판정하지 않는다. 이후 분석을 위해 원본과 요약만 저장한다.

ViM sample별:

```text
energy
residual
virtual_logit
vim_score
sample_index
label(ID인 경우)
D
alpha
feature_dim
residual_dim
checkpoint
seed
dataset
```

kNN sample별:

```text
knn_score
sample_index
label(ID인 경우)
k
checkpoint
seed
dataset
```

권장 `.npz` 구조:

```text
OUTPUT_ROOT/
└── EXPERIMENT_GROUP/
    └── seed_0/
        ├── metadata.json
        ├── summary.json
        ├── summary.csv
        └── raw/
            ├── id/
            │   ├── vim_D256_components.npz
            │   ├── vim_D320_components.npz
            │   └── knn_k50_scores.npz
            └── ood/
                ├── cifar100/
                ├── tiny_imagenet/
                ├── svhn/
                └── mnist/
```

요약 통계:

```text
count
mean
standard deviation
median
minimum
maximum
```

각 OOD에 대해:

```text
ViM D별 AUROC, FPR@95
kNN AUROC, FPR@95
Energy-only AUROC, FPR@95
Residual-only AUROC, FPR@95
Virtual-logit-only AUROC, FPR@95
```

---

# 9. 여러 checkpoint 자동 실행

권장 파일:

```text
run_vim_knn_batch.py
```

역할:

- checkpoint 자동 검색
- manifest 생성
- seed 파싱
- 실험군 grouping
- checkpoint당 evaluator 한 번 실행
- 완료 결과 skip
- 실패 기록
- 중단 후 재시작

예상 단일 실행:

```bash
python evaluate_vim_knn.py   --checkpoint_path /ABSOLUTE/PATH/model.model   --data_root /home/ghjin/ICML/DDU_fork/data   --ood_datasets cifar100 tiny_imagenet svhn mnist   --vim_dims 256 320   --knn_k 50   --seed auto   --batch_size 128   --gpu   --output_dir /ABSOLUTE/PATH/output
```

---

# 10. seed 0/1/2 집계

같은 실험군의 seed 0, 1, 2를 묶는다.

저장:

```text
seed0
seed1
seed2
mean
standard deviation
n_success
missing_seeds
```

최종 표기:

```text
mean ± standard deviation
```

`ddof`는 기존 프로젝트 관행을 확인하고, 불명확하면 사용자에게 질문한다.

---

# 11. 필수 테스트

## 정적 검사

```bash
python -m py_compile evaluate_vim_knn.py
```

생성된 batch/aggregate 파일도 검사한다.

## synthetic ViM test

- shape
- finite
- D 경계
- fit 전 score 오류
- reference 계산과 `allclose`

## synthetic kNN test

- 직접 계산한 k번째 거리와 일치
- score sign
- k 경계
- batching 일치

## 실제 checkpoint smoke test

전체 batch 전에 대표 checkpoint 하나와 OOD 하나로 검증한다.

```text
/home/ghjin/all_models_260328/models_260328/CIFAR-10/WideResNet/cifar10_sam_0.5wide_resnet_mod_0_350.model
```

확인:

- parsed seed=0
- split seed=0
- feature dimension
- D=256, 320
- k=50
- AUROC/FPR@95 finite
- raw `.npz`
- summary 파일

## logit parity

```text
model forward logits
vs.
feature @ W.T + b
```

권장:

```text
max_abs_diff < 1e-4
```

---

# 12. 금지 사항

- 학습 코드 수정
- checkpoint 수정·이동·삭제
- ViM D 하드코딩
- kNN k 하드코딩
- OOD test로 D/k 선택
- Temperature-scaled logits 혼합
- 개선/악화 자동 판정
- 실패를 0 metric으로 저장
- 사용자 승인 전 symlink 생성
- 사용자 승인 전 장기 평가
- 대규모 dependency 설치
- commit, push

---

# 13. Codex CLI 시작 프롬프트

아래 내용을 repository 최상위에서 입력한다.

```text
`VIM_KNN_Codex_작업지시서.md`를 처음부터 끝까지 읽어 주세요.

이번 실행에서는 반드시 문서의 “1단계: 읽기 전용 서버 감사”만 수행하세요.
아직 구현하거나 평가를 시작하지 마세요.

다음 경로를 우선 조사하세요.

- /home/ghjin/DDU_fork_0724VIM/run_eval.py
- /home/ghjin/DDU_fork_0724VIM/evaluate_v2.py
- 위 파일들이 import하는 모델, feature extraction, dataset loader, OOD score, metric 관련 코드
- /home/ghjin/ICML/DDU_fork/data
- /home/ghjin/all_models_260328/models_260328/CIFAR-10/WideResNet

읽기 전용 감사에서 다음을 확인해 주세요.

1. 기존 model loading과 feature extraction 정의의 재사용 방법
2. WRN-28-10의 실제 penultimate feature 위치와 차원
3. 공유 dataset의 directory 구조와 loader 호환성
4. checkpoint 파일명 규칙과 seed 0/1/2 parser
5. 학습 seed를 평가 seed와 CIFAR-10 split seed에 전달하는 방법
6. 같은 실험의 seed checkpoint grouping 규칙
7. ViM과 kNN이 feature extraction을 공유하면서 독립적으로 계산·저장되는 최소 구현안
8. `--vim_dims`가 한 개 또는 여러 D를 받는 방법
9. `--knn_k` 기본값 50과 입력 인자 처리
10. OOD 4종을 한 실행에서 평가하는 방법
11. AUROC와 FPR@95 convention
12. Energy, Residual, Virtual Logit, ViM score와 kNN raw score 저장 구조
13. 여러 checkpoint 자동 실행과 seed mean ± standard deviation 집계 구조
14. exact kNN의 시간·메모리 위험
15. 필요한 최소 파일과 테스트 계획

이번 단계에서는 다음을 하지 마세요.

- 파일 수정 또는 생성
- symlink 생성
- dependency 설치
- dataset 다운로드
- 모델 평가 실행
- 장시간 GPU 작업
- commit 또는 push

감사 후에는 다음을 포함한 구체적인 구현안을 한국어로 제시하세요.

- 실제 코드 구조
- 재사용 가능한 함수와 새로 필요한 코드
- dataset 연결 방식 권고
- checkpoint manifest와 seed parser
- 제안 파일 목록
- output directory 구조
- 계산 비용과 위험
- 아직 사용자 결정이 필요한 질문

마지막에는 반드시 아래 문장으로 멈추세요.

“읽기 전용 감사를 완료했습니다. 아직 파일을 수정하거나 평가를 시작하지 않았습니다. 위 구현안을 승인하거나 수정해 주세요.”
```
