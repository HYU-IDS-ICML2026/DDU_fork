import os
import glob
import time

# =========================================================
# [설정 수정 영역]
# 1. 모델이 저장된 실제 절대 경로로 변경했습니다.
MODEL_DIR = "/home/ghjin/ICML/DDU_fork/models/wrn_cifar10"

# 2. 평가할 OOD 데이터셋 리스트
OOD_LIST = ["svhn", "tiny_imagenet", "mnist"] 

# 3. 로그 파일 이름
LOG_FILE = "evaluation_fpr_results.log"
# =========================================================

# 모델 파일 검색 (.model 확장자)
# 해당 경로의 모든 .model 파일을 찾습니다.
model_paths = glob.glob(os.path.join(MODEL_DIR, "*.model"))
model_paths.sort()

print(f"Target Directory: {MODEL_DIR}")
print(f"Found {len(model_paths)} models.")

# 로그 파일에 시작 시간 기록
with open(LOG_FILE, "a") as f:
    f.write(f"\n{'='*20} New Evaluation Started at {time.ctime()} {'='*20}\n")

for model in model_paths:
    print(f"Evaluating model: {os.path.basename(model)}")
    for ood in OOD_LIST:
        print(f"  - OOD Dataset: {ood}")
        
        # [명령어 생성]
        # 모델 폴더명이 'wrn_cifar10'이므로 --dataset은 cifar10으로 설정했습니다.
        # DDU 학습 설정에 맞춰 --sn, --mod, --coeff 3.0 옵션을 포함했습니다.
        command = f"python evaluate_v2_FPR.py \
                    --checkpoint_path {model} \
                    --dataset cifar10 \
                    --ood_dataset {ood} \
                    --model wide_resnet \
                    --sn \
                    --coeff 3.0 \
                    --mod \
                    --seed 0 \
                    --batch_size 128 \
                    --gpu >> {LOG_FILE} 2>&1"
        
        # 명령어 실행
        exit_code = os.system(command)
        
        if exit_code != 0:
            print(f"    [Error] Failed to evaluate {os.path.basename(model)} on {ood}")
        else:
            print(f"    [Done] Logged to {LOG_FILE}")

print(f"\nAll evaluations finished. Results are saved in {LOG_FILE}")