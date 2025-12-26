import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance

# 1. Energy-Based Score
def get_energy_score(logits, T=1.0):
    # Energy = -T * logsumexp(logits / T)
    # 점수가 높을수록 ID(In-Distribution)여야 하므로, 일반적으로 사용하는 음수(-) 부호 대신
    # DDU 코드 관례(점수 높음=ID)에 맞춰 양수로 변환하거나, 후처리에서 통일해야 합니다.
    # 여기서는 "에너지가 낮을수록 ID"라는 원래 정의를 뒤집어 "점수가 높을수록 ID"가 되도록 구현합니다.
    return T * torch.logsumexp(logits / T, dim=1)

# 2. Mahalanobis Distance
class MahalanobisScorer:
    def __init__(self):
        self.class_means = []
        self.precision = None

    def fit(self, features, labels, num_classes):
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        self.class_means = []
        
        # 클래스별 평균 계산
        for c in range(num_classes):
            self.class_means.append(np.mean(features[labels == c], axis=0))
        
        # 공분산 계산 (sklearn 활용)
        X_centered = []
        for i in range(len(features)):
            X_centered.append(features[i] - self.class_means[labels[i]])
        
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(X_centered))
        self.precision = ec.precision_

    def score(self, features):
        features = features.cpu().numpy()
        scores = []
        for i in range(len(features)):
            dists = []
            for c in range(len(self.class_means)):
                diff = features[i] - self.class_means[c]
                # 거리 계산: (x-u)^T * Sigma^-1 * (x-u)
                dist = np.dot(np.dot(diff, self.precision), diff)
                dists.append(dist)
            # 거리가 가까울수록(작을수록) ID이므로, 마이너스를 붙여 점수화
            scores.append(-np.min(dists))
        return torch.tensor(scores)

# 3. kNN Score
class KNNScorer:
    def __init__(self, k=50):
        self.k = k
        self.train_features = None

    def fit(self, features):
        # 학습 데이터 Feature 저장 (GPU 메모리 절약을 위해 CPU로 이동 권장)
        self.train_features = features.cpu()

    def score(self, features):
        features = features.cpu()
        batch_size = 100
        scores = []
        
        # 배치 단위로 거리 계산 (메모리 이슈 방지)
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            dist_matrix = torch.cdist(batch, self.train_features, p=2)
            
            # k번째로 가까운 거리 찾기
            kth_dists, _ = torch.topk(dist_matrix, k=self.k, dim=1, largest=False)
            # 거리가 가까울수록 ID -> 마이너스 부호
            scores.append(-kth_dists[:, -1])
            
        return torch.cat(scores)