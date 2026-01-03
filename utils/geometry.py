import torch
import numpy as np
import math

# ---------------------------------------------------------
# Helper Functions (JIN Branch Logic)
# ---------------------------------------------------------
def fro_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x, p="fro")

def make_etf_gram(K: int, device, dtype) -> torch.Tensor:
    I = torch.eye(K, device=device, dtype=dtype)
    ones = torch.ones((K, K), device=device, dtype=dtype)
    return (I - ones / K) / math.sqrt(K - 1)

def pairwise_mean_dist(M: torch.Tensor) -> torch.Tensor:
    K = M.shape[0]
    D = torch.cdist(M, M, p=2)
    idx = torch.triu_indices(K, K, offset=1)
    return D[idx[0], idx[1]].mean()

def effective_rank_from_cov(cov: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    evals = torch.linalg.eigvalsh(cov)
    evals = torch.clamp(evals, min=0.0)
    s = evals.sum()
    if s < eps:
        return torch.tensor(0.0, device=cov.device, dtype=cov.dtype)
    p = evals / (s + eps)
    ent = -(p * torch.log(p + eps)).sum()
    return torch.exp(ent)

def anisotropy_lambda1_over_trace(cov: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    evals = torch.linalg.eigvalsh(cov)
    evals = torch.clamp(evals, min=0.0)
    tr = evals.sum()
    if tr < eps:
        return torch.tensor(0.0, device=cov.device, dtype=cov.dtype)
    return evals[-1] / (tr + eps)

# ---------------------------------------------------------
# Main Calculation Function
# ---------------------------------------------------------
def get_geometry_stats(model, loader, device, num_classes):
    """
    Main Branch의 구조(Feature List)를 유지하여 NC4 계산을 지원하되,
    NC1~NC3 및 기타 지표 계산은 JIN Branch의 정확한 수식을 적용합니다.
    """
    model.eval()
    model.to(device)
    dtype = torch.float64  # 정밀도를 위해 float64 사용

    # 1. Feature Extraction (Main Branch Style)
    features_list = []
    labels_list = []
    
    # Weight 추출
    if hasattr(model, 'module'):
        fc_layer = model.module.fc if hasattr(model.module, 'fc') else model.module.linear
    else:
        fc_layer = model.fc if hasattr(model, 'fc') else model.linear
    
    # Hook 설정
    feats_buf = {}
    def fc_prehook(module, inputs):
        feats_buf["h"] = inputs[0].detach()
    handle = fc_layer.register_forward_pre_hook(fc_prehook)

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            _ = model(data)
            features_list.append(feats_buf["h"].cpu()) # CPU로 이동하여 저장
            labels_list.append(target.cpu())
            
    handle.remove()

    if not features_list:
        return {}

    features = torch.cat(features_list).to(dtype=dtype) # (N, D)
    labels = torch.cat(labels_list)
    
    N, D = features.shape
    K = num_classes

    # 2. Statistics Calculation (Centering 적용)
    # Global Mean
    global_mean = features.mean(dim=0) # (D,)
    
    # Class Means & Covariances
    class_means = torch.zeros((K, D), dtype=dtype)
    within_class_scatter = torch.zeros((D, D), dtype=dtype)
    anisotropies = []
    eff_ranks = []
    
    # NC4 계산을 위한 Class Means (Raw)
    # JIN 로직은 Centering이 핵심이지만, NC4 거리 계산은 Raw Mean으로 해도 무방
    # (어차피 Global Mean 빼도 거리는 동일)
    
    for c in range(K):
        idxs = (labels == c)
        if idxs.sum() == 0: continue
        
        feats_c = features[idxs]
        mu_c = feats_c.mean(dim=0)
        class_means[c] = mu_c
        
        # Within-Class Scatter (Sigma_W의 구성 요소)
        centered_c = feats_c - mu_c
        cov_c = torch.matmul(centered_c.T, centered_c) # Sum of squares
        within_class_scatter += cov_c
        
        # Anisotropy & Effective Rank (JIN Logic)
        # Covariance matrix for this class
        actual_cov_c = cov_c / (feats_c.shape[0]) # biased or unbiased choice (JIN uses 1/n)
        actual_cov_c = 0.5 * (actual_cov_c + actual_cov_c.T) # Symmetrize
        
        anisotropies.append(anisotropy_lambda1_over_trace(actual_cov_c).item())
        eff_ranks.append(effective_rank_from_cov(actual_cov_c).item())

    # Sigma_W & Sigma_B
    Sigma_W = within_class_scatter / N
    
    Sigma_B = torch.zeros((D, D), dtype=dtype)
    for c in range(K):
        # Centering 적용 (mu_c - mu_G)
        dc = (class_means[c] - global_mean).view(D, 1)
        Sigma_B += dc @ dc.t()
    Sigma_B /= K

    # 3. Metrics Calculation (JIN Formula)

    # [NC1] Variability Collapse
    # JIN Formula: trace(Sigma_W @ pinv(Sigma_B)) / K
    SigmaB_pinv = torch.linalg.pinv(Sigma_B)
    nc1_val = (torch.trace(Sigma_W @ SigmaB_pinv) / K).item()

    # [NC2] Simplex ETF (JIN Formula: Norm Difference)
    W = fc_layer.weight.detach().to(dtype=dtype).cpu()
    WWt = W @ W.t()
    G_W = WWt / (fro_norm(WWt) + 1e-12)
    G_ETF = make_etf_gram(K, device=G_W.device, dtype=dtype)
    nc2_etf_dist = fro_norm(G_W - G_ETF).item()

    # [NC2 - Compatibility] Mean Cosine Sim (Centered)
    # 기존 코드의 출력 포맷을 유지하되, Centering을 적용하여 올바르게 계산
    M_centered = class_means - global_mean
    M_norm = M_centered / (M_centered.norm(dim=1, keepdim=True) + 1e-12)
    cos_sim = torch.mm(M_norm, M_norm.t())
    mask = ~torch.eye(K, dtype=torch.bool)
    nc2_mean_sim = cos_sim[mask].mean().item()

    # [NC3] Alignment (JIN Formula: Norm Difference)
    H = (class_means - global_mean).t()
    WH = W @ H
    G_WH = WH / (fro_norm(WH) + 1e-12)
    nc3_val = fro_norm(G_WH - G_ETF).item()

    # [NC4] NCC Accuracy (Main Branch Feature)
    # Features와 Class Means 사이의 거리를 계산
    # (둘 다 Raw 값이므로 거리 계산에 문제 없음)
    dists = torch.cdist(features.to(device).float(), class_means.to(device).float(), p=2)
    preds = torch.argmin(dists, dim=1)
    nc4_acc = (preds == labels.to(device)).float().mean().item()

    # 기타 지표
    mean_within_var = torch.trace(Sigma_W).item()
    inter_dist = pairwise_mean_dist(class_means).item()
    mean_anisotropy = float(np.mean(anisotropies)) if anisotropies else 0.0
    mean_eff_rank = float(np.mean(eff_ranks)) if eff_ranks else 0.0

    return {
        "Within-class Variance": mean_within_var,
        "Inter-class Distance": inter_dist,
        "Anisotropy": mean_anisotropy,
        "Effective Rank": mean_eff_rank,
        
        "NC1 (Variability)": nc1_val,
        
        # 기존 호환성 유지 (수정된 Centered Mean Sim)
        "NC2 (Mean Sim)": nc2_mean_sim,
        # JIN 브랜치의 정확한 지표 (ETF 거리)
        "NC2 (ETF Dist)": nc2_etf_dist,
        
        "NC3 (Alignment)": nc3_val,
        "NC4 (NCC Acc)": nc4_acc
    }