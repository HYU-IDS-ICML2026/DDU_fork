import torch
from torch import nn
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 2, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):

    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):

    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


# def gmm_fit(embeddings, labels, num_classes):
#     with torch.no_grad():
#         classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
#         classwise_cov_features = torch.stack(
#             [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
#         )

#     with torch.no_grad():
#         for jitter_eps in JITTERS:
#             try:
#                 jitter = jitter_eps * torch.eye(
#                     classwise_cov_features.shape[1], device=classwise_cov_features.device,
#                 ).unsqueeze(0)
#                 gmm = torch.distributions.MultivariateNormal(
#                     loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
#                 )
#             except RuntimeError as e:
#                 if "cholesky" in str(e):
#                     continue
#             except ValueError as e:
#                 if "The parameter covariance_matrix has invalid values" in str(e):
#                     continue
#             break

#     return gmm, jitter_eps


# def gmm_fit(embeddings, labels, num_classes):
#     with torch.no_grad():
#         # 1. 각 클래스별 평균 계산 (기존과 동일)
#         classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        
#         # 2. Tied Covariance 계산: 모든 샘플에서 해당 클래스의 평균을 빼서 '잔차'를 구함
#         # 각 데이터 포인트에서 자기 클래스의 평균을 뺀 전체 잔차 행렬 생성
#         residuals = embeddings - classwise_mean_features[labels.long()]
        
#         # 3. 모든 클래스의 데이터를 통합하여 단 하나의 공분산 행렬 계산
#         # 샘플 수가 N(50,000)으로 늘어나 Singularity 문제가 해결됨
#         shared_cov = centered_cov_torch(residuals)
        
#         # 4. MultivariateNormal 입력을 위해 클래스 수만큼 행렬 복제 [num_classes, dim, dim]
#         classwise_cov_features = shared_cov.unsqueeze(0).expand(num_classes, -1, -1)

#     with torch.no_grad():
#         for jitter_eps in JITTERS:
#             try:
#                 jitter = jitter_eps * torch.eye(
#                     classwise_cov_features.shape[1], device=classwise_cov_features.device,
#                 ).unsqueeze(0)
                
#                 # 모든 클래스가 동일한 공분산(shared_cov + jitter)을 공유하게 됨
#                 gmm = torch.distributions.MultivariateNormal(
#                     loc=classwise_mean_features, 
#                     covariance_matrix=(classwise_cov_features + jitter),
#                 )
#             except (RuntimeError, ValueError) as e:
#                 continue
#             break

#     return gmm, jitter_eps


# L2

def gmm_fit(embeddings, labels, num_classes):
    # L2 정규화 강도 (Hyperparameter)
    # 0.1 정도면 N < D 상황에서도 매우 안정적입니다. (필요 시 조절: 0.01 ~ 0.5)
    # 값이 클수록 '대각 공분산'이나 '단위 행렬'에 가까워지고, 작을수록 '원본(Full)'에 가까워집니다.
    alpha = 0.1 
    
    with torch.no_grad():
        # 1. 각 클래스별 평균 계산
        classwise_mean_features = torch.stack([
            torch.mean(embeddings[labels == c], dim=0) 
            for c in range(num_classes)
        ])
        
        # 2. 각 클래스별 Full Covariance 계산 (Singular 상태)
        classwise_cov_features = torch.stack([
            centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) 
            for c in range(num_classes)
        ])

        # 3. L2 Regularization (Shrinkage) 적용
        # 공식: Sigma_new = (1 - alpha) * Sigma_old + alpha * Identity
        # 이렇게 하면 모든 고유값(Eigenvalue)이 최소 'alpha' 이상이 되어 역행렬이 무조건 존재합니다.
        num_features = classwise_cov_features.shape[1]
        device = classwise_cov_features.device
        identity = torch.eye(num_features, device=device).unsqueeze(0)
        
        # 정규화된 공분산 행렬 계산
        classwise_cov_features = (1 - alpha) * classwise_cov_features + alpha * identity

    # 4. GMM 생성 (이제 Jitter 루프 없이도 한 번에 성공합니다)
    try:
        gmm = torch.distributions.MultivariateNormal(
            loc=classwise_mean_features, 
            covariance_matrix=classwise_cov_features
        )
    except Exception as e:
        print(f"GMM Fit Failed even with Shrinkage: {e}")
        # 만약 alpha=0.1로도 안 되면 데이터 자체(NaN 등)에 문제가 있는 것입니다.
        raise e

    # jitter_eps는 alpha로 대체하여 반환하거나 0으로 둠
    return gmm, alpha