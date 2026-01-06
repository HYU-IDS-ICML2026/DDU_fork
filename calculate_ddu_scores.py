import os
import glob
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================================================
# [설정 영역] 사용자 정의 변수
# =============================================================================
MODEL_DIR = "./models/cifar10"   # 모델 파일 경로
MODEL_ARCH = "wide_resnet"        # wide_resnet, resnet50, vgg16

# 모델 하이퍼파라미터 (파일 이름 파싱 X, 직접 지정)
USE_SN = True
USE_MOD = True
COEFF = 3.0

# 데이터셋 설정
ID_DATASET = "cifar10"
NEAR_OOD_LIST = ["cifar100", "tiny_imagenet"]
FAR_OOD_LIST = ["mnist", "svhn"]

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================

# 1. Import Packages & Patching (evaluate_v2.py 참조)
try:
    from net.resnet import resnet50, resnet18
    from net.wide_resnet import wrn
    from net.vgg import vgg16
    import net.spectral_normalization.spectral_norm_conv_inplace as sn_lib
    
    # Dataloaders
    import data.ood_detection.cifar10 as cifar10
    import data.ood_detection.cifar100 as cifar100
    import data.ood_detection.svhn as svhn
    import data.ood_detection.mnist_ood as mnist_ood
    import data.ood_detection.tiny_imagenet as tiny_imagenet
    
    # Utils
    from utils.gmm_utils import get_embeddings, gmm_fit, gmm_get_logits

except ImportError as e:
    print(f"Error: 프로젝트 루트에서 실행해야 합니다. {e}")
    exit(1)

# -----------------------------------------------------------------------------
# [Monkey Patch] evaluate_v2.py의 핵심 로직: SN Hook 에러 방지
# -----------------------------------------------------------------------------
original_load_hook = sn_lib.SpectralNormConvLoadStateDictPreHook.__call__

def patched_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    fn = self.fn
    version = local_metadata.get("spectral_norm_conv", {}).get(fn.name + ".version", None)
    # state_dict에 'weight'가 없고 'weight_orig'만 있으면 훅 실행을 건너뜀 (에러 방지)
    if (version is None or version < 1) and (prefix + fn.name) not in state_dict:
        if (prefix + fn.name + "_orig") in state_dict:
            return
    return original_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

# 패치 적용
sn_lib.SpectralNormConvLoadStateDictPreHook.__call__ = patched_load_hook
# -----------------------------------------------------------------------------

# Configuration Maps
dataset_loader_map = {
    "cifar10": cifar10, "cifar100": cifar100, "svhn": svhn,
    "mnist": mnist_ood, "tiny_imagenet": tiny_imagenet
}
dataset_num_classes = {
    "cifar10": 10, "cifar100": 100, "svhn": 10, "mnist": 10, "tiny_imagenet": 200
}
models_map = {
    "resnet50": resnet50, "resnet18": resnet18, 
    "wide_resnet": wrn, "vgg16": vgg16
}
model_dim_map = {
    "resnet50": 2048, "resnet18": 512, 
    "wide_resnet": 640, "vgg16": 512
}

def load_model(arch, path, num_classes, sn, mod, coeff, device):
    try:
        # 모델 생성
        net = models_map[arch](
            spectral_normalization=sn, 
            mod=mod, 
            coeff=coeff, 
            num_classes=num_classes, 
            temp=1.0
        )
        
        # 가중치 로드
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # DataParallel의 'module.' 접두어 제거
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # 패치 덕분에 strict=True여도(기본값) 에러 없이 로드됨
        net.load_state_dict(new_state_dict)
        net.to(device)
        net.eval()
        return net
    except Exception as e:
        print(f"[-] Model Load Failed ({os.path.basename(path)}): {e}")
        return None

def main():
    print(">>> Starting DDU Log-Likelihood Calculation (Bulk Processing) <<<")
    
    # 1. 모델 파일 리스트
    model_paths = glob.glob(os.path.join(MODEL_DIR, "*.model"))
    model_paths.sort()
    
    if not model_paths:
        print(f"No .model files found in {MODEL_DIR}")
        return

    # 2. 데이터셋 로더 준비
    # 2.1 ID Train Loader (GMM Fitting용)
    print(f"Loading ID Train Data ({ID_DATASET})...")
    train_loader, _ = dataset_loader_map[ID_DATASET].get_train_valid_loader(
        batch_size=BATCH_SIZE, augment=False, val_seed=0, val_size=0.1, pin_memory=True
    )
    
    # 2.2 Test Loaders (ID + OOD)
    test_loaders = {}
    target_datasets = [ID_DATASET] + NEAR_OOD_LIST + FAR_OOD_LIST
    
    for ds_name in target_datasets:
        print(f"Loading Test Data ({ds_name})...")
        if ds_name in ["mnist", "tiny_imagenet"]:
             # evaluate_v2.py 참고: 일부 데이터셋은 root 인자 필요할 수 있음 (환경에 따라 조정)
             test_loaders[ds_name] = dataset_loader_map[ds_name].get_test_loader(batch_size=BATCH_SIZE, pin_memory=True)
        else:
             test_loaders[ds_name] = dataset_loader_map[ds_name].get_test_loader(batch_size=BATCH_SIZE, pin_memory=True)

    num_classes = dataset_num_classes.get(ID_DATASET, 100)
    feat_dim = model_dim_map[MODEL_ARCH]

    # 3. 모델별 반복 수행
    for model_path in tqdm(model_paths, desc="Processing Models"):
        model_name = os.path.basename(model_path)
        
        # 모델 로드
        net = load_model(MODEL_ARCH, model_path, num_classes, USE_SN, USE_MOD, COEFF, DEVICE)
        if net is None: continue

        # 3.1 GMM 피팅 (ID Train Features)
        try:
            # evaluate_v2.py에서는 device=device, storage_device='cpu'로 메모리 절약
            embeddings, labels = get_embeddings(net, train_loader, feat_dim, torch.double, DEVICE, torch.device('cpu'))
            
            # Embeddings를 GPU로 옮겨서 피팅 (속도 향상)
            embeddings = embeddings.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # GMM 피팅 (utils/gmm_utils.py가 수정된 상태라면 대각 공분산 등 적용됨)
            gmm, _ = gmm_fit(embeddings, labels, num_classes)
            
            # 메모리 정리
            del embeddings, labels
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [Error] GMM Fit failed for {model_name}: {e}")
            continue

        # 3.2 각 데이터셋별 Log-Likelihood 계산
        model_results = []
        
        for ds_name in target_datasets:
            if ds_name not in test_loaders: continue
            
            loader = test_loaders[ds_name]
            try:
                # Feature 추출 (CPU 저장)
                feats, _ = get_embeddings(net, loader, feat_dim, torch.double, DEVICE, torch.device('cpu'))
                feats = feats.to(DEVICE) # 계산할 땐 GPU로
                
                # Logits 계산 (log pi_c + l_c(z))
                logits = gmm_get_logits(gmm, feats)  # [N, C]

                logits = logits / (feat_dim**0.5)

                # 1. Term A: c_max(...) -> 가장 높은 클래스의 로짓값
                max_logits, _ = torch.max(logits, dim=1) 
                
                # 2. Term B: log ∑ exp(...) -> 전체 에비던스(정규화 상수)
                log_sum_exp = torch.logsumexp(logits, dim=1)
                
                # 3. 최종 스코어 계산
                # (이 값은 0 이하의 음수이며, 0에 가까울수록 ID일 확률이 높음)
                scores = (max_logits - log_sum_exp).detach().cpu().numpy()

                for idx, score in enumerate(scores):
                    model_results.append({
                        "TrainDataset": ID_DATASET,
                        "OODDataset": ds_name,
                        "DataIndex": idx,
                        "LikelihoodGap": score  # 컬럼명은 호환성을 위해 유지 (실제값은 LogPosterior)
                    })
                
                del feats, logits
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [Error] Eval failed for {ds_name} on {model_name}: {e}")

        # 4. 모델별 CSV 저장
        if model_results:
            safe_name = os.path.splitext(model_name)[0]
            csv_filename = f"ddu_scores_{safe_name}.csv"
            
            df = pd.DataFrame(model_results)
            df = df[["TrainDataset", "OODDataset", "DataIndex", "LikelihoodGap"]]
            df.to_csv(csv_filename, index=False)

    print(">>> All Processing Done <<<")

if __name__ == "__main__":
    main()