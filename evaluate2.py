"""
Script to evaluate a single model.

[DDU 공식 evaluate.py 기반 수정본]
- 기존 기능 유지: ID Accuracy / ECE / Temperature scaling
- 추가 기능: MSP, Energy(EBO), kNN, Mahalanobis(shared covariance), DDU(m1) OOD 탐지 (AUROC/AUPRC)
- 핵심 아이디어: penultimate feature를 1회만 추출하여 (kNN / Mahalanobis / DDU)에서 재사용

[참고/출처]
- DDU 공식 구현: https://github.com/omegafragger/DDU  (evaluate.py, gmm_utils.py, temperature_scaling.py)
- OpenOOD v1.5 구현 참고: https://github.com/Jingkang50/OpenOOD
  - Energy(EBO) 정의 참고: OpenOOD의 ebo_postprocessor.py
  - kNN 정의 참고: OpenOOD의 knn_postprocessor.py (L2 normalize + kth distance)
  - Mahalanobis(MDS) 정의 참고: OpenOOD의 mds_postprocessor.py (shared covariance 기반 score=-d^2)
- Temperature scaling은 DDU에서 쓰는 ModelWithTemperature를 그대로 사용 (gpleiss/temperature_scaling 기반)

※ 팀 공유를 위해, 변경/추가/제거 지점을 [추가], [변경], [제거] 주석으로 표시함.
"""
import os
import json
import math
import torch
import argparse
import numpy as np
#추가
from datetime import datetime
import re
# =========================
# [추가] FAISS (GPU kNN 가속)
# - 목적: kNN OOD score 계산을 torch.mm (O(N*D) per batch) 대신 FAISS로 가속
# - 구현 참고: OpenOOD (KNN postprocessor)에서의 FAISS 사용 방식
# =========================
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn

# Import network models
# [변경] resnet18을 추가로 지원 (팀 실험에서 ResNet-18 사용)
from net.resnet import resnet50
try:
    from net.resnet import resnet18  # DDU 레포에 없으면 사용자가 추가했을 수 있음
except Exception:
    resnet18 = None
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import metrics to compute (ID accuracy/ECE는 기존 그대로 사용)
from metrics.classification_metrics import (
    test_classification_net,
    test_classification_net_ensemble,
)
from metrics.calibration_metrics import expected_calibration_error

# Import GMM utils (DDU)
# [변경] gmm_get_logits를 추가로 사용(이미 추출한 embedding에서 바로 GMM logits 계산)
from utils.gmm_utils import get_embeddings, gmm_fit, gmm_get_logits

from utils.ensemble_utils import load_ensemble
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name
from utils.args import eval_args

# Temperature scaling (DDU 그대로)
from utils.temperature_scaling import ModelWithTemperature

# Dataset params
dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}
dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn}

# Mapping model name to model function
models = {"resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16}
if resnet18 is not None:
    models["resnet18"] = resnet18

# penultimate dimension mapping
# [변경] resnet18 추가 (일반적으로 ResNet-18 penultimate dim=512)
model_to_num_dim = {"resnet50": 2048, "wide_resnet": 640, "vgg16": 512}
if resnet18 is not None:
    model_to_num_dim["resnet18"] = 512


# =========================
# [추가] 공통 유틸: AUROC/AUPRC
# =========================
def compute_auroc_auprc_from_ood_scores(id_scores: torch.Tensor, ood_scores: torch.Tensor):
    """
    모든 method에서 최종 score는 'score ↑ = OOD' 방향으로 들어온다고 가정.
    positive label = OOD 로 두고 AUROC/AUPRC 계산.
    """
    id_scores = id_scores.detach().float().cpu().numpy()
    ood_scores = ood_scores.detach().float().cpu().numpy()
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    s = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))
    return auroc, auprc


# =========================
# [추가] logits 수집 + MSP/Energy
# =========================
@torch.no_grad()
def collect_logits(model, loader, device):
    """모델 출력(logits)을 전부 모아 반환."""
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)  # temp_scaled_net이면 이미 logits/T
        logits_list.append(out.detach().cpu())
        labels_list.append(y.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def msp_ood_score_from_scaled_logits(logits_scaled: torch.Tensor):
    """
    MSP(Max Softmax Probability):
      MSP(x)=max softmax(logits_scaled)
    최종 통일 규칙: OOD-score↑=OOD 이므로 1-MSP를 사용.
    """
    probs = F.softmax(logits_scaled, dim=1)
    msp = probs.max(dim=1).values
    return 1.0 - msp  # OOD-score


def energy_ood_score_from_scaled_logits(logits_scaled: torch.Tensor, T: float):
    """
    Energy-based score (EBO 관련):
    OpenOOD ebo_postprocessor.py의 'conf = T * logsumexp(logits/T)' 형태를 참고.
    - 여기서 logits_scaled = logits/T 이므로 conf = T * logsumexp(logits_scaled)
    최종 통일 규칙: OOD-score↑=OOD 이므로 Energy = -conf 를 사용.
    """
    conf = float(T) * torch.logsumexp(logits_scaled, dim=1)
    energy = -conf  # OOD-score
    return energy


# =========================
# [추가] kNN (L2 normalize + kth distance)
# =========================
def knn_kth_distance_ood_score(
    query_z: torch.Tensor,
    ref_z: torch.Tensor,
    k: int = 50,
    batch_size: int = 4096,
    backend: str = "auto",
    use_faiss_gpu: bool = True,
    faiss_gpu_id: int = 0,
) -> torch.Tensor:
    """Compute kNN OOD score using penultimate features.

    Score definition (direction): **distance ↑ => OOD**.
    - We L2-normalize embeddings first, then use Euclidean distance in that space.

    Backends:
    - backend='faiss' : FAISS IndexFlatL2 (GPU if available & use_faiss_gpu=True)
    - backend='torch' : torch.mm 기반(기존 구현). (fallback)
    - backend='auto'  : FAISS 가능하면 FAISS, 아니면 torch

    Notes:
    - FAISS IndexFlatL2는 **squared L2 distance**를 반환합니다.
      본 구현은 최종 score를 sqrt하여 Euclidean distance로 맞춥니다.
    """

    if backend not in {"auto", "faiss", "torch"}:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "auto":
        backend = "faiss" if _FAISS_AVAILABLE else "torch"

    q = query_z.detach().float().cpu()
    r = ref_z.detach().float().cpu()
    q = F.normalize(q, dim=1)
    r = F.normalize(r, dim=1)

    if k <= 0:
        raise ValueError("k must be >= 1")
    k_eff = min(k, r.shape[0])

    if backend == "faiss":
        if not _FAISS_AVAILABLE:
            raise RuntimeError("backend='faiss' requested but faiss is not available in this env.")

        r_np = np.ascontiguousarray(r.numpy().astype(np.float32))
        q_np = np.ascontiguousarray(q.numpy().astype(np.float32))
        d = r_np.shape[1]

        index = faiss.IndexFlatL2(d)
        if use_faiss_gpu and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_id, index)

        index.add(r_np)

        kth_list = []
        for i in range(0, q_np.shape[0], batch_size):
            q_chunk = q_np[i : i + batch_size]
            D, _ = index.search(q_chunk, k_eff)
            kth_sq = D[:, k_eff - 1]
            kth_list.append(kth_sq)

        kth_sq_all = np.concatenate(kth_list, axis=0)
        kth_dist = np.sqrt(np.maximum(kth_sq_all, 0.0))
        return torch.from_numpy(kth_dist).float()

    scores = []
    for i in range(0, q.shape[0], batch_size):
        batch = q[i : i + batch_size]
        sim = torch.mm(batch, r.t())
        kth_sim, _ = torch.topk(sim, k_eff, dim=1, largest=True, sorted=True)
        kth_dist_sq = 2.0 - 2.0 * kth_sim[:, -1]
        kth_dist = torch.sqrt(torch.clamp(kth_dist_sq, min=0.0))
        scores.append(kth_dist.cpu())
    return torch.cat(scores, dim=0)

def fit_mahalanobis_shared(emb_train: torch.Tensor, y_train: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    shared covariance Σ: within-class centered features 사용
    Σ = (1/(N-C)) * sum_i (z_i - μ_{y_i})(z_i - μ_{y_i})^T
    precision = Σ^{-1}

    OpenOOD mds_postprocessor.py는 score=-d^2 형태(conf=ID)로 구현.
    우리는 OOD-score↑=OOD로 통일하기 위해 min_c d^2 를 그대로 사용.
    """
    d = emb_train.shape[1]
    mu = []
    for c in range(num_classes):
        mu.append(emb_train[y_train == c].mean(dim=0))
    mu = torch.stack(mu, dim=0)  # (C,d)

    centered = emb_train - mu[y_train]  # (N,d)
    denom = max(int(centered.shape[0] - num_classes), 1)
    cov = (centered.t() @ centered) / float(denom)
    cov = cov + eps * torch.eye(d, device=emb_train.device, dtype=emb_train.dtype)

    precision = torch.linalg.inv(cov)
    return mu, precision


def mahalanobis_min_sqdist_ood_score(
    query_Z: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor, batch_size: int = 512
):
    """
    OOD-score = min_c (z-μ_c)^T P (z-μ_c)
    """
    C = mu.shape[0]
    scores = torch.empty(query_Z.shape[0], device=query_Z.device, dtype=torch.float32)
    for start in range(0, query_Z.shape[0], batch_size):
        end = min(start + batch_size, query_Z.shape[0])
        z = query_Z[start:end]  # (B,d)
        diff = z[:, None, :] - mu[None, :, :]  # (B,C,d)
        diff_flat = diff.reshape(-1, diff.shape[-1])  # (B*C,d)
        md2_flat = (diff_flat @ precision) * diff_flat
        md2 = md2_flat.sum(dim=1).reshape(-1, C)  # (B,C)
        scores[start:end] = md2.min(dim=1).values.float()
    return scores  # OOD-score↑=OOD


# =========================
# [추가] DDU(m1) from embeddings
# =========================
def ddu_m1_ood_score_from_embeddings(emb_test: torch.Tensor, emb_ood: torch.Tensor, gmm):
    """
    DDU m1 score:
      m1(x) = logsumexp( log p(z|c) )  (uniform prior 가정)
    DDU 공식 evaluate.py는 get_roc_auc_logits(..., logsumexp, confidence=True)로 ID confidence를 계산.
    우리는 OOD-score↑=OOD로 통일하기 위해 최종 score = -m1 을 사용.
    """
    logits_test = gmm_get_logits(gmm, emb_test)  # (N_id, C)
    logits_ood = gmm_get_logits(gmm, emb_ood)    # (N_ood, C)
    m1_test = torch.logsumexp(logits_test, dim=1)
    m1_ood = torch.logsumexp(logits_ood, dim=1)
    return -m1_test, -m1_ood  # OOD-score↑=OOD


# =========================
# [추가] 체크포인트 로딩 유틸 (DataParallel prefix 처리)
# =========================
def load_state_dict_flexible(net, ckpt_path: str, device: torch.device):
    """
    - 학습 시 DataParallel을 썼든 안 썼든 모두 로딩되도록 'module.' prefix를 유연하게 처리.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    net_state_keys = net.state_dict().keys()
    state_keys = state.keys()

    if any(k.startswith("module.") for k in state_keys) and not any(k.startswith("module.") for k in net_state_keys):
        new_state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state = new_state
    elif (not any(k.startswith("module.") for k in state_keys)) and any(k.startswith("module.") for k in net_state_keys):
        new_state = {"module." + k: v for k, v in state.items()}
        state = new_state

    net.load_state_dict(state)
    net.to(device)
    return net


# =========================
# [추가] 학습 러닝 폴더명 파싱 (train_results 하위 폴더 규칙)
# =========================
def parse_training_run_from_path(path: str):
    """
    [변경] folder.split("_") 방식은 rho0_2, sn3_0처럼 "소수점 표현"에 underscore('_')를 쓰는 경우
           토큰이 쪼개져서 파싱이 깨질 수 있습니다.

    그래서 train.py가 만드는 "run 폴더명" 전체를 정규식(regex)으로 한 번에 파싱합니다.

    Expected folder name pattern (train.py 기준):
      model_dataset_(sam|sgd)[_rhoX]_(nosn|snY)_seedZ_MMDD_HHMMSS

    예시:
      resnet50_cifar100_sam_rho0_2_sn3_0_seed1_1230_192623
      resnet50_cifar100_sgd_sn3_0_seed1_1230_192541
    """
    abs_path = os.path.abspath(path)
    run_dir = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)
    folder = os.path.basename(run_dir)

    pattern = re.compile(
        r"^(?P<model>[^_]+)_(?P<dataset>[^_]+)_(?P<opt>sam|sgd)"
        r"(?:_rho(?P<rho>[0-9]+(?:_[0-9]+)*))?"
        r"_(?P<snseg>nosn|sn(?P<coeff>[0-9]+(?:_[0-9]+)*))"
        r"_seed(?P<seed>[0-9]+)_(?P<ts>[0-9]{4}_[0-9]{6})$"
    )

    m = pattern.match(folder)
    if m is None:
        raise ValueError(
            f"Folder name '{folder}' does not match the expected pattern. "
            "Pass a checkpoint under train_results/<run_folder>/... ."
        )

    model_name = m.group("model")
    dataset_name = m.group("dataset")
    optimizer = m.group("opt")

    rho_str = m.group("rho")
    sam_rho = float(rho_str.replace("_", ".")) if (optimizer == "sam" and rho_str is not None) else None

    snseg = m.group("snseg")
    sn_enabled = snseg.startswith("sn")
    coeff_str = m.group("coeff")
    coeff = float(coeff_str.replace("_", ".")) if (sn_enabled and coeff_str is not None) else None

    seed = int(m.group("seed"))
    ts = m.group("ts")

    return {
        "model": model_name,
        "dataset": dataset_name,
        "optimizer": optimizer,
        "sam_rho": sam_rho,
        "sn": sn_enabled,
        "coeff": coeff,
        "seed": seed,
        "timestamp": ts,
    }
#==============================



if __name__ == "__main__":

    # =========================
    # [변경] eval_args() 기반 parser에 추가 인자 주입
    # =========================
    parser = eval_args()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        #default=None,
        required=True,
        help=(
            "사용자가 지정한 체크포인트 파일 경로. 지정 시 DDU의 기본 저장 규칙(model_load_name..._350.model)을 무시하고 "
            "이 파일을 직접 로드함. 예: ./Models/resnet18_cifar10_sgd_seed1_last.model"
        ),
    )
    parser.add_argument("--knn-k", type=int, default=50, help="kNN OOD 탐지에서 사용할 K (default: 50)")
    # =========================
    # [추가] kNN backend 선택 (default: faiss if available)
    # =========================
    parser.add_argument(
        "--knn-backend",
        type=str,
        default="auto",
        choices=["auto", "faiss", "torch"],
        help="kNN 계산 backend 선택. auto=FAISS 사용 가능하면 FAISS, 아니면 torch.mm (default: auto)",
    )
    parser.add_argument(
        "--faiss-no-gpu",
        action="store_true",
        help="(선택) FAISS를 사용하되 GPU를 쓰지 않음(=CPU index). 디버깅/호환성용.",
    )
    parser.add_argument(
        "--knn-batch-size",
        type=int,
        default=4096,
        help="kNN query chunk batch size (FAISS search/torch fallback 공통) (default: 4096)",
    )

    parser.add_argument("--maha-eps", type=float, default=1e-6, help="Mahalanobis shared covariance inverse 안정화 eps")
    parser.add_argument(
        "--use-data-parallel",
        action="store_true",
        help="(선택) DataParallel로 평가. 기본은 단일 GPU. (CUDA_VISIBLE_DEVICES와 병행 권장)",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="결과를 저장할 json 파일명(또는 경로). 미지정 시 기존 규칙 res_{...}.json 사용.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluate_results",
        help="평가 결과(json)를 저장할 디렉터리(없으면 생성)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="학습 시 사용한 optimizer (폴더 파싱 시 자동 채워짐)",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=None,
        help="SAM 사용 시 rho 값 (폴더 파싱 시 자동 채워짐)",
    )
    parser.add_argument(
        "--no-infer-from-path",
        dest="infer_from_path",
        action="store_false",
        help="checkpoint 경로에서 학습 설정을 자동 파싱하지 않음",
    )
    parser.set_defaults(infer_from_path=True)
#===============================================
    args = parser.parse_args()
    
    #추가
    # [추가] --checkpoint-path를 주면 train_results 폴더명에서 학습 설정(model/dataset/opt/sn/seed)을 자동 파싱합니다.
    #        (사용자가 --model/--dataset/--seed 등을 직접 넣지 않아도 되도록)
    run_info = None
    if args.checkpoint_path is not None and args.infer_from_path:
        run_info = parse_training_run_from_path(args.checkpoint_path)

        # [변경] 학습 설정 자동 주입
        args.model = run_info.get("model", args.model)
        args.dataset = run_info.get("dataset", args.dataset)
        args.sn = run_info.get("sn", args.sn)
        if run_info.get("coeff") is not None:
            args.coeff = run_info["coeff"]
        args.seed = run_info.get("seed", args.seed)

        # [추가] 결과 파일명 생성 등에 사용
        args.train_optimizer = run_info.get("optimizer", None)
        args.train_sam_rho = run_info.get("sam_rho", None)

        print(f"[AUTO-INFER] Parsed training run info from path: {run_info}")

    # [추가] seed가 여전히 None이면(파싱 실패/미지정) 안전하게 중단
    if args.seed is None:
        raise ValueError(
            "Seed is not set. Provide --checkpoint-path in train_results format or pass --seed explicitly."
        )

    #==========
    cuda = torch.cuda.is_available()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if (cuda and args.gpu) else "cpu")

    num_classes = dataset_num_classes[args.dataset]

    test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)

    accuracies = []
    eces = []
    t_eces = []
    topts = []

    msp_aurocs, msp_auprcs = [], []
    energy_aurocs, energy_auprcs = [], []
    knn_aurocs, knn_auprcs = [], []
    maha_aurocs, maha_auprcs = [], []
    ddu_aurocs, ddu_auprcs = [], []

    for i in range(args.runs):
        print(f"Evaluating run: {(i + 1)}")

        if args.model_type == "ensemble":
            # (기존 로직 유지, 신규 OOD method는 nan 처리)
            val_loaders = []
            for j in range(args.ensemble):
                train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                    batch_size=args.batch_size,
                    augment=args.data_aug,
                    val_seed=(args.seed + (5 * i) + j),
                    val_size=0.1,
                    pin_memory=args.gpu,
                )
                val_loaders.append(val_loader)

            ensemble_loc = os.path.join(args.load_loc, ("Run" + str(i + 1)))
            net_ensemble = load_ensemble(
                ensemble_loc=ensemble_loc,
                model_name=args.model,
                device=device,
                num_classes=num_classes,
                spectral_normalization=args.sn,
                mod=args.mod,
                coeff=args.coeff,
                seed=(5 * i + 1),
            )

            (_, accuracy, labels_list, predictions, confidences,) = test_classification_net_ensemble(
                net_ensemble, test_loader, device
            )
            ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

            t_ensemble = []
            for model, vloader in zip(net_ensemble, val_loaders):
                t_model = ModelWithTemperature(model)
                t_model.set_temperature(vloader)
                t_ensemble.append(t_model)

            (_, t_accuracy, t_labels_list, t_predictions, t_confidences,) = test_classification_net_ensemble(
                t_ensemble, test_loader, device
            )
            t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

            accuracies.append(accuracy)
            eces.append(ece)
            t_eces.append(t_ece)
            topts.append(float("nan"))

            msp_aurocs.append(float("nan")); msp_auprcs.append(float("nan"))
            energy_aurocs.append(float("nan")); energy_auprcs.append(float("nan"))
            knn_aurocs.append(float("nan")); knn_auprcs.append(float("nan"))
            maha_aurocs.append(float("nan")); maha_auprcs.append(float("nan"))
            ddu_aurocs.append(float("nan")); ddu_auprcs.append(float("nan"))
            continue

        # Deterministic(single) model
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.batch_size, augment=args.data_aug, val_seed=(args.seed + i), val_size=0.1, pin_memory=args.gpu,
        )

        '''if args.checkpoint_path is not None:
            ckpt_path = args.checkpoint_path.format(run=(i + 1), seed=(args.seed + i))
        else:
            ckpt_path = os.path.join(
                args.load_loc,
                "Run" + str(i + 1),
                model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350.model",
            )'''
        #추가
        ckpt_path = args.checkpoint_path.format(run=(i + 1), seed=(args.seed + i))
        if not ckpt_path.endswith(".model"):
            raise ValueError(f"Checkpoint 파일 확장자는 .model 이어야 합니다: {ckpt_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint 파일을 찾을 수 없습니다: {ckpt_path}")
        #=========
        if args.model not in models:
            raise ValueError(f"Unknown model '{args.model}'. Available: {list(models.keys())}")

        net = models[args.model](
            spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes, temp=1.0,
        )

        if args.gpu:
            net.to(device)
            if args.use_data_parallel and torch.cuda.device_count() > 1:
                net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                cudnn.benchmark = True

        net = load_state_dict_flexible(net, ckpt_path, device)
        net.eval()

        # (유지) ID accuracy / ECE
        (_, accuracy, labels_list, predictions, confidences,) = test_classification_net(net, test_loader, device)
        ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

        # (유지) Temperature scaling
        temp_scaled_net = ModelWithTemperature(net)
        temp_scaled_net.set_temperature(val_loader)
        topt = float(temp_scaled_net.temperature)

        (_, t_accuracy, t_labels_list, t_predictions, t_confidences,) = test_classification_net(
            temp_scaled_net, test_loader, device
        )
        t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

        # [추가] MSP / Energy
        id_logits_scaled, _ = collect_logits(temp_scaled_net, test_loader, device)
        ood_logits_scaled, _ = collect_logits(temp_scaled_net, ood_test_loader, device)

        msp_id = msp_ood_score_from_scaled_logits(id_logits_scaled)
        msp_ood = msp_ood_score_from_scaled_logits(ood_logits_scaled)
        msp_auroc, msp_auprc = compute_auroc_auprc_from_ood_scores(msp_id, msp_ood)

        energy_id = energy_ood_score_from_scaled_logits(id_logits_scaled, T=topt)
        energy_ood = energy_ood_score_from_scaled_logits(ood_logits_scaled, T=topt)
        energy_auroc, energy_auprc = compute_auroc_auprc_from_ood_scores(energy_id, energy_ood)

        # [추가] Penultimate embedding 1회 추출
        num_dim = model_to_num_dim[args.model]
        emb_train_f, y_train = get_embeddings(
            net, train_loader, num_dim=num_dim, dtype=torch.float, device=device, storage_device=device
        )
        emb_test_f, _ = get_embeddings(
            net, test_loader, num_dim=num_dim, dtype=torch.float, device=device, storage_device=device
        )
        emb_ood_f, _ = get_embeddings(
            net, ood_test_loader, num_dim=num_dim, dtype=torch.float, device=device, storage_device=device
        )

        # [추가] kNN
        knn_id = knn_kth_distance_ood_score(
            emb_test_f,
            emb_train_f,
            k=args.knn_k,
            batch_size=args.knn_batch_size,
            backend=args.knn_backend,
            use_faiss_gpu=(not args.faiss_no_gpu),
            faiss_gpu_id=0,
        )  # [변경] FAISS backend 지원
        knn_ood = knn_kth_distance_ood_score(
            emb_ood_f,
            emb_train_f,
            k=args.knn_k,
            batch_size=args.knn_batch_size,
            backend=args.knn_backend,
            use_faiss_gpu=(not args.faiss_no_gpu),
            faiss_gpu_id=0,
        )  # [변경] FAISS backend 지원
        knn_auroc, knn_auprc = compute_auroc_auprc_from_ood_scores(knn_id, knn_ood)

        # [추가] Mahalanobis(shared cov)
        mu, precision = fit_mahalanobis_shared(emb_train_f, y_train, num_classes=num_classes, eps=args.maha_eps)
        maha_id = mahalanobis_min_sqdist_ood_score(emb_test_f, mu, precision)
        maha_ood = mahalanobis_min_sqdist_ood_score(emb_ood_f, mu, precision)
        maha_auroc, maha_auprc = compute_auroc_auprc_from_ood_scores(maha_id, maha_ood)

        # (유지/활용) DDU(m1)
        emb_train_d = emb_train_f.double()
        emb_test_d = emb_test_f.double()
        emb_ood_d = emb_ood_f.double()

        try:
            gmm, jitter_eps = gmm_fit(embeddings=emb_train_d, labels=y_train, num_classes=num_classes)
            ddu_id, ddu_ood = ddu_m1_ood_score_from_embeddings(emb_test_d, emb_ood_d, gmm)
            ddu_auroc, ddu_auprc = compute_auroc_auprc_from_ood_scores(ddu_id, ddu_ood)
        except RuntimeError as e:
            print("[DDU] Runtime Error caught: " + str(e))
            ddu_auroc, ddu_auprc = float("nan"), float("nan")

        accuracies.append(float(accuracy))
        eces.append(float(ece))
        t_eces.append(float(t_ece))
        topts.append(float(topt))

        msp_aurocs.append(float(msp_auroc)); msp_auprcs.append(float(msp_auprc))
        energy_aurocs.append(float(energy_auroc)); energy_auprcs.append(float(energy_auprc))
        knn_aurocs.append(float(knn_auroc)); knn_auprcs.append(float(knn_auprc))
        maha_aurocs.append(float(maha_auroc)); maha_auprcs.append(float(maha_auprc))
        ddu_aurocs.append(float(ddu_auroc)); ddu_auprcs.append(float(ddu_auprc))

    def mean_se(x_list):
        t = torch.tensor(x_list, dtype=torch.float32)
        mean = float(torch.mean(t))
        se = float(torch.std(t) / math.sqrt(max(t.shape[0], 1)))
        return mean, se

    #[제거]res_dict = {"mean": {}, "std": {}, "values": {}, "info": vars(args)}
    #[추가]
    res_dict = {"mean": {}, "std": {}, "values": {}, "info": vars(args), "train_run_info": run_info}

    res_dict["mean"]["accuracy"], res_dict["std"]["accuracy"] = mean_se(accuracies)
    res_dict["mean"]["ece"], res_dict["std"]["ece"] = mean_se(eces)
    res_dict["mean"]["t_ece"], res_dict["std"]["t_ece"] = mean_se(t_eces)
    res_dict["mean"]["topt"], res_dict["std"]["topt"] = mean_se(topts)

    res_dict["values"]["accuracy"] = accuracies
    res_dict["values"]["ece"] = eces
    res_dict["values"]["t_ece"] = t_eces
    res_dict["values"]["topt"] = topts

    for name, aurocs, auprcs in [
        ("msp", msp_aurocs, msp_auprcs),
        ("energy", energy_aurocs, energy_auprcs),
        ("knn", knn_aurocs, knn_auprcs),
        ("maha", maha_aurocs, maha_auprcs),
        ("ddu", ddu_aurocs, ddu_auprcs),
    ]:
        res_dict["mean"][f"{name}_auroc"], res_dict["std"][f"{name}_auroc"] = mean_se(aurocs)
        res_dict["mean"][f"{name}_auprc"], res_dict["std"][f"{name}_auprc"] = mean_se(auprcs)
        res_dict["values"][f"{name}_auroc"] = aurocs
        res_dict["values"][f"{name}_auprc"] = auprcs

    #[추가]
    os.makedirs(args.results_dir, exist_ok=True)
    eval_timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    #====================================
    if args.result_json is not None:
        out_json = args.result_json
        #[추가]
        if not os.path.isabs(out_json):
            out_json = os.path.join(args.results_dir, out_json)
           
    else:
        train_model = run_info.get("model") if run_info else args.model
        train_dataset = run_info.get("dataset") if run_info else args.dataset

        optimizer = None
        if run_info is not None:
            optimizer = run_info.get("optimizer")
        if optimizer is None:
            optimizer = getattr(args, "optimizer", None)
        if optimizer is None and hasattr(args, "optimiser"):
            optimizer = getattr(args, "optimiser")
        optimizer = optimizer or "opt"

        sam_rho_str = None
        if run_info is not None:
            sam_rho_str = run_info.get("sam_rho_str")
        if sam_rho_str is None and optimizer == "sam":
            sam_rho_val = getattr(args, "sam_rho", None)
            if sam_rho_val is not None:
                sam_rho_str = str(sam_rho_val).replace(".", "_")

        sn_enabled = run_info.get("sn") if run_info is not None else args.sn
        coeff_str = None
        if run_info is not None:
            coeff_str = run_info.get("coeff_str")
        if coeff_str is None and sn_enabled:
            coeff_val = getattr(args, "coeff", None)
            if coeff_val is not None:
                coeff_str = str(coeff_val).replace(".", "_")

        seed_val = run_info.get("seed") if run_info is not None else args.seed

        train_tag = f"{train_model}_{train_dataset}_{optimizer}"
        if optimizer == "sam" and sam_rho_str is not None:
            train_tag += f"_rho{sam_rho_str}"
        if sn_enabled:
            coeff_seg = coeff_str or ""
            train_tag += f"_sn{coeff_seg}"
        else:
            train_tag += "_nosn"
        train_tag += f"_seed{seed_val}"

        out_json = os.path.join(
            args.results_dir,
            f"{train_tag}_ood_{args.ood_dataset}_{eval_timestamp}.json",
        )

    with open(out_json, "w") as f:
        json.dump(res_dict, f, indent=2)

    print(f"[DONE] Saved results to: {out_json}")