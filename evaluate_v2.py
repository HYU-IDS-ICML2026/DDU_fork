import os
import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# 데이터셋 및 모델 Import (기존 유지)
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
from net.resnet import resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# 유틸 (기존 + 신규)
from utils.args import eval_args
from utils.train_utils import model_save_name
from utils.ood_scores import get_energy_score, MahalanobisScorer, KNNScorer
from utils.gmm_utils import gmm_fit # DDU용

dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}
dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn}
models = {"resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16}

# 간단한 평가 지표 출력 함수
def evaluate_metric(name, id_scores, ood_scores):
    # 점수가 높을수록 ID(1), 낮을수록 OOD(0)라고 가정
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores.cpu().numpy(), ood_scores.cpu().numpy()])
    
    auroc = roc_auc_score(y_true, y_scores)
    # FPR95 구하기
    # ... (생략 가능하거나 sklearn 등으로 구현. 여기선 AUROC만 예시로 출력)
    print(f"[{name}] AUROC: {auroc:.4f}")

# Feature 추출 헬퍼 함수
def extract_features(model, loader, device):
    model.eval()
    feats, logits, labels = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            out = model(data) # Logit 계산
            
            # 모델별 Feature 가져오기 (이미 구현되어 있음)
            feat = model.feature 
            
            feats.append(feat.cpu())
            logits.append(out.cpu())
            labels.append(target)
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)

if __name__ == "__main__":
    args = eval_args().parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    num_classes = dataset_num_classes[args.dataset]

    # 1. 데이터 로더 준비 (Train, Test, OOD)
    print("Loading Data...")
    train_loader, test_loader = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root, batch_size=args.batch_size, augment=False, val_size=0.1, val_seed=args.seed
    )
    _, ood_loader = dataset_loader[args.ood_dataset].get_train_valid_loader(
        root=args.dataset_root, batch_size=args.batch_size, augment=False, val_size=0.1, val_seed=args.seed
    )

    # 2. 모델 로드
    # train.py 저장 규칙에 맞춰 파일명 생성
    saved_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed) + "_350.model"
    # 경로 수정: save_loc 아래 RunX 폴더가 있다면 그 경로까지 맞춰주어야 함
    # 예시: args.load_loc/Run1/파일명 (사용자 환경에 맞게 조정 필요)
    load_path = os.path.join(args.load_loc, "Run1", saved_name) 
    
    print(f"Loading Model from {load_path}")
    net = models[args.model](
        spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes
    )
    net.load_state_dict(torch.load(load_path, map_location=device))
    net.to(device)

    # 3. Feature 추출
    print("Extracting Features...")
    train_feats, train_logits, train_lbls = extract_features(net, train_loader, device)
    test_feats, test_logits, _ = extract_features(net, test_loader, device)
    ood_feats, ood_logits, _ = extract_features(net, ood_loader, device)

    print(f"Train: {train_feats.shape}, Test: {test_feats.shape}, OOD: {ood_feats.shape}")

    # ==========================
    # 4. OOD 스코어 계산 및 평가
    # ==========================

    # (1) MSP (Softmax)
    print("\n--- MSP ---")
    msp_id = torch.softmax(test_logits, dim=1).max(1)[0]
    msp_ood = torch.softmax(ood_logits, dim=1).max(1)[0]
    evaluate_metric("MSP", msp_id, msp_ood)

    # (2) Energy
    print("\n--- Energy ---")
    energy_id = get_energy_score(test_logits, T=1.0)
    energy_ood = get_energy_score(ood_logits, T=1.0)
    evaluate_metric("Energy", energy_id, energy_ood)

    # (3) kNN
    print("\n--- kNN ---")
    knn = KNNScorer(k=50)
    knn.fit(train_feats) # 학습 데이터로 피팅
    knn_id = knn.score(test_feats)
    knn_ood = knn.score(ood_feats)
    evaluate_metric("kNN", knn_id, knn_ood)

    # (4) Mahalanobis
    print("\n--- Mahalanobis ---")
    maha = MahalanobisScorer()
    maha.fit(train_feats, train_lbls, num_classes)
    maha_id = maha.score(test_feats)
    maha_ood = maha.score(ood_feats)
    evaluate_metric("Mahalanobis", maha_id, maha_ood)

    # (5) DDU (GMM)
    print("\n--- DDU (GMM) ---")
    # utils/gmm_utils.py의 gmm_fit 활용
    # Feature를 GPU로 옮겨서 피팅 (DDU 코드는 GPU 텐서를 입력으로 받음)
    gmm, _ = gmm_fit(train_feats.to(device), train_lbls.to(device), num_classes)
    
    # GMM Score (Log Likelihood) 계산
    # sklearn GMM 객체가 아니라면 gmm_utils의 evaluate 함수를 쓰거나,
    # gmm이 sklearn 객체라면 아래처럼 score_samples 사용
    from sklearn.mixture import GaussianMixture
    if isinstance(gmm, GaussianMixture):
        ddu_id = torch.tensor(gmm.score_samples(test_feats.numpy()))
        ddu_ood = torch.tensor(gmm.score_samples(ood_feats.numpy()))
    else:
        # gmm_utils가 커스텀 구현체인 경우 해당 evaluate 함수 사용 필요
        # 현재 파일 구조상 sklearn GMM을 반환하는 것으로 보임
        ddu_id = torch.tensor(gmm.score_samples(test_feats.cpu().numpy()))
        ddu_ood = torch.tensor(gmm.score_samples(ood_feats.cpu().numpy()))
        
    evaluate_metric("DDU", ddu_id, ddu_ood)