import os
import json
import csv
import re

# ==========================================
# 설정 (여기서 경로 및 파라미터 리스트 수정)
# ==========================================
target_folder = "logs/eval/cifar10_tide"  # json 파일들이 있는 폴더 경로
output_file = "result_summary.csv"

# 파싱에 사용할 리스트 정의
VALID_RHOS = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5']
VALID_BACKBONES = ['wide_resnet', 'lenet', 'resnet18', 'resnet50', 'vgg16']

# CSV 헤더 정의
headers = [
    "optimizer", "rho", "backbone", "seed", "epoch", 
    "accuracy", "ece", "t_ece", 
    "Within-class Variance", "Inter-class Distance", "Anisotropy", "Effective Rank", 
    "NC1 (Variability)", "NC2 (Mean Sim)", "NC2 (ETF Dist)", "NC3 (Alignment)", "NC4 (NCC Acc)", 
    "ood_data", 
    "msp_auroc", "entropy_auroc", "energy_auroc", "ddu_auroc", "maha_auroc", "knn_auroc"
]

# ==========================================
# 실행 코드
# ==========================================

# 파일명 파싱용 정규표현식
# 예: res__sam_0.01wide_resnet_sn_3.0_mod_0_350.model_cifar100.json
# 그룹: 1=optimizer, 2=rho+backbone, 3=seed, 4=epoch, 5=ood_data
filename_pattern = re.compile(r"res__([a-zA-Z0-9]+)_(.+)_sn_3\.0_mod_(\d+)_(\d+)\.model_(.+)\.json")

rows = []

if not os.path.exists(target_folder):
    print(f"Error: Folder '{target_folder}' not found.")
    exit()

files = [f for f in os.listdir(target_folder) if f.endswith('.json')]
print(f"Found {len(files)} json files.")

for filename in files:
    match = filename_pattern.match(filename)
    if not match:
        print(f"Skipping (pattern mismatch): {filename}")
        continue

    opt, middle_part, seed, epoch, ood_data = match.groups()
    
    # rho와 backbone 분리 로직
    rho = "0" # sgd인 경우 기본값
    backbone = middle_part

    if opt == "sam":
        for r in VALID_RHOS:
            if middle_part.startswith(r):
                # rho값과 일치하는 부분이 앞에 있으면 분리
                check_backbone = middle_part[len(r):]
                if check_backbone in VALID_BACKBONES:
                    rho = r
                    backbone = check_backbone
                    break
    
    try:
        with open(os.path.join(target_folder, filename), 'r') as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {})
        geometry = data.get("geometry", {})

        row = [
            opt,
            rho,
            backbone,
            int(seed),
            int(epoch),
            data.get("accuracy"),
            data.get("ece"),
            data.get("t_ece"),
            geometry.get("Within-class Variance"),
            geometry.get("Inter-class Distance"),
            geometry.get("Anisotropy"),
            geometry.get("Effective Rank"),
            geometry.get("NC1 (Variability)"),
            geometry.get("NC2 (Mean Sim)"),
            geometry.get("NC2 (ETF Dist)"),
            geometry.get("NC3 (Alignment)"),
            geometry.get("NC4 (NCC Acc)"),
            ood_data,
            metrics.get("msp_auroc"),
            metrics.get("entropy_auroc"),
            metrics.get("energy_auroc"),
            metrics.get("ddu_auroc"),
            metrics.get("maha_auroc"),
            metrics.get("knn_auroc")
        ]
        rows.append(row)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# CSV 작성
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"Successfully saved to {output_file}")