import os
import glob

MODEL_DIR = "models/wide_resnet"
OOD_LIST = ["cifar100", "svhn", "mnist", "tiny_imagenet"]
LOG_FILE = "evaluation_all.log"

# 모델 파일 검색
model_paths = glob.glob(os.path.join(MODEL_DIR, "*.model"))
model_paths.sort()



for model in model_paths:
    for ood in OOD_LIST:
        command = f"python evaluate_v2.py \
                    --checkpoint_path {model} \
                    --dataset cifar10 \
                    --ood_dataset {ood} \
                    --model wide_resnet \
                    --sn \
                    --mod \
                    --coeff 3.0 \
                    --seed 0 \
                    --batch_size 128\
                    --gpu >> {LOG_FILE} 2>&1"
        
        os.system(command)