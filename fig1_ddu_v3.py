"""
fig1_ddu.py: Reproducing DDU Figure 1 with SGD vs SAM models.
Compares Softmax Entropy and Gaussian Log-Density on:
1. CIFAR-10 (ID: Low Entropy) / Ambiguous (ID: High Entropy via CIFAR-10H) / SVHN (OOD)
2. MNIST (ID) / Dirty-MNIST (Ambiguous) / Fashion-MNIST (OOD)

New Features:
- Overlap Area Calculation
- FPR95 Calculation
- PCA Visualization
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import json
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

# Import project modules
from net.resnet import resnet18
from utils.gmm_utils import get_embeddings, gmm_fit

# Import Data Loaders
try:
    from data.fast_mnist import FastMNIST
    from data.ambiguous_mnist.ambiguous_mnist_dataset import AmbiguousMNIST
    import data.dirty_mnist as dirty_mnist
except ImportError:
    print("Warning: MNIST specific modules not found. Ensure you are in the project root.")

# --- Configuration & Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="DDU Figure 1 Reproduction: SGD vs SAM")
    
    # Dataset Selection
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'mnist'],
                        help='Choose dataset: cifar10 (with CIFAR-10H) or mnist')

    # Model Paths
    parser.add_argument('--model_sgd', type=str, required=True, help='Path to the SGD trained model')
    parser.add_argument('--model_sam', type=str, required=True, help='Path to the SAM trained model')
    
    # Dataset Settings
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Execution Settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')

    # [추가] 실험 결과 파일명 구분을 위한 접미사    
    parser.add_argument('--suffix', type=str, default='', help='Suffix for saved files (e.g., _rho0.1)')
    
    return parser.parse_args()

# --- 1. Dataset Helpers (CIFAR-10H Split) ---

def get_cifar10h_splits(data_root, test_dataset, clean_percentile=60, ambiguous_percentile=80):
    probs_path = os.path.join(data_root, 'cifar10h-probs.npy')
    
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"CIFAR-10H probabilities not found at {probs_path}. Please download it.")
        
    print(f"  -> Loading CIFAR-10H from {probs_path}...")
    probs = np.load(probs_path)
    human_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    thresh_clean = np.percentile(human_entropy, clean_percentile)
    thresh_ambiguous = np.percentile(human_entropy, ambiguous_percentile)
    
    clean_indices = np.where(human_entropy <= thresh_clean)[0]
    ambiguous_indices = np.where(human_entropy >= thresh_ambiguous)[0]
    
    print(f"  -> CIFAR-10H Split Stats: Clean({len(clean_indices)}) / Ambiguous({len(ambiguous_indices)})")
    
    return Subset(test_dataset, clean_indices), Subset(test_dataset, ambiguous_indices)

def get_dataloaders(args, device):
    print(f"Preparing Dataloaders for {args.dataset}...")
    
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        train_set = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
        base_test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
        
        try:
            test_set, ambiguous_set = get_cifar10h_splits(args.data_root, base_test_set, 
                                                          clean_percentile=60, ambiguous_percentile=80)
        except FileNotFoundError as e:
            print(e)
            return None, None, None, None

        svhn_set = datasets.SVHN(root=args.data_root, split='test', download=True, transform=transform)
        ood_set = svhn_set
        nw = args.num_workers

    elif args.dataset == 'mnist':
        print("  -> Loading Dirty-MNIST for GMM fitting...")
        train_loader, _ = dirty_mnist.get_train_valid_loader(
            root=args.data_root, batch_size=args.batch_size, augment=False, val_size=0.1, val_seed=args.seed, pin_memory=args.gpu
        )
        test_set = FastMNIST(args.data_root, train=False, download=True)
        try:
            ambiguous_set = AmbiguousMNIST(root=args.data_root, train=False, device=device)
        except FileNotFoundError:
            print("Error: 'amnist_samples.pt' not found via AmbiguousMNIST.")
            raise

        fmnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ood_set = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=fmnist_transform)
        nw = 0 

    if args.dataset != 'mnist':
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)
        
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    ambiguous_loader = DataLoader(ambiguous_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    ood_loader = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    return train_loader, test_loader, ambiguous_loader, ood_loader

# --- 2. Smart Model Loading ---

def inspect_checkpoint(checkpoint):
    keys = list(checkpoint.keys())
    has_module = keys[0].startswith("module.")
    clean_keys = [k.replace("module.", "") for k in keys] if has_module else keys
    has_shortcut_weights = any("layer2.0.shortcut.0.weight" in k for k in clean_keys)
    detected_mod = not has_shortcut_weights
    has_sn = any("weight_orig" in k for k in clean_keys)
    return has_module, detected_mod, has_sn

def load_model(path, device, dataset_name, default_sn_coeff=3.0):
    print(f"Loading model from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location='cpu')
    has_module_prefix, detected_mod, detected_sn = inspect_checkpoint(checkpoint)
    print(f"  -> Detected Config: DataParallel={has_module_prefix}, mod={detected_mod}, SN={detected_sn}")

    is_mnist = (dataset_name == 'mnist')
    net = resnet18(spectral_normalization=detected_sn, mod=detected_mod, coeff=default_sn_coeff, num_classes=10, mnist=is_mnist)
    
    if has_module_prefix: net = torch.nn.DataParallel(net)
    try: net.load_state_dict(checkpoint)
    except RuntimeError:
        print(f"  -> Strict load failed. Retrying with strict=False...")
        net.load_state_dict(checkpoint, strict=False)
    if has_module_prefix: net = net.module
        
    net.to(device)
    net.eval()
    return net

# --- 3. Metrics & Features ---

def get_features_and_metrics(net, gmm, loader, device):
    """
    Returns: (features, entropy, log_density)
    """
    features_list, entropy_list, density_list = [], [], []
    with torch.no_grad():
        for data, _ in tqdm(loader, desc="Computing Metrics"):
            data = data.to(device)
            
            # Features
            if isinstance(net, nn.DataParallel): _ = net.module(data); feats = net.module.feature
            else: _ = net(data); feats = net.feature
            
            # Entropy
            logits = net(data)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
            # Log-Density (Modified to match official DDU evaluate.py)
            # DDU 공식 구현에서는 GMM의 confidence score(m1)로 LogSumExp를 사용합니다.
            # 이는 모든 클래스에 대한 우도의 합(marginal likelihood)을 로그 스케일로 구하는 것입니다.
            log_probs = gmm.log_prob(feats.unsqueeze(1))
            
            # [수정 전] 최댓값 사용
            # max_log_density, _ = torch.max(log_probs, dim=1) 
            
            # [수정 후] LogSumExp 사용 (evaluate.py의 m1 metric 정의 준수)
            log_density = torch.logsumexp(log_probs, dim=1)
            
            features_list.append(feats.cpu())
            entropy_list.append(entropy.cpu())
            density_list.append(log_density.cpu())
            
    return (torch.cat(features_list).numpy(), torch.cat(entropy_list).numpy(), torch.cat(density_list).numpy())

# --- 4. Advanced Metrics: FPR95 & Overlap ---

def calculate_fpr95(id_scores, ood_scores):
    """
    Calculates False Positive Rate at 95% True Positive Rate.
    Lower is better.
    """
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find FPR when TPR >= 0.95
    for i, t in enumerate(tpr):
        if t >= 0.95:
            return fpr[i]
    return 1.0

def calculate_overlap_area(dist1, dist2, bins=50):
    """
    Calculates the overlapping area between two histograms (0.0 ~ 1.0).
    Higher means more confusion.
    """
    # Define common range
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    hist1, _ = np.histogram(dist1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), density=True)
    
    # Normalize to sum to 1 (Probability Mass)
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate overlap: sum of min heights
    overlap = np.sum(np.minimum(hist1, hist2))
    return overlap

# --- 5. Visualization: Plot & PCA ---

# --- 5. Visualization: Plot & PCA ---

# [수정] suffix 인자 추가 (기존: def plot_distributions(results, save_dir, dataset_name):)
def plot_distributions(results, save_dir, dataset_name, suffix=''):
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")
    
    if dataset_name == 'cifar10':
        dataset_names = ['ID (Clean)', 'Ambiguous (Human)', 'OOD (SVHN)']
    else:
        dataset_names = ['ID (MNIST)', 'Ambiguous (Dirty)', 'OOD (Fashion)']

    colors = ['#1f77b4', '#7f7f7f', '#ff7f0e'] 
    metrics = [('Entropy', 'Softmax Entropy'), ('LogDensity', 'Gaussian Log-Density')]
    
    for model_name in results.keys():
        if not results[model_name]['Entropy']: continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[idx]
            data_map = results[model_name][metric_key]
            data_list = [data_map.get('ID'), data_map.get('Ambiguous'), data_map.get('OOD')]
            
            for data, name, color in zip(data_list, dataset_names, colors):
                if data is None or len(data) == 0: continue
                sns.histplot(data, stat='probability', element="step", fill=True,
                             label=name, color=color, alpha=0.3, ax=ax, bins=50, common_norm=False)
            
            ax.set_title(metric_title)
            ax.set_xlabel(metric_title)
            ax.set_ylabel("Fraction") 
            if idx == 0: ax.legend(loc='upper right')
        
        plt.suptitle(f"Model: {model_name} on {dataset_name.upper()} (OOD Detection)", fontsize=16)
        plt.tight_layout()
        
        # [수정] 저장 파일명에 suffix 추가
        filename = f"fig1_ddu_{dataset_name}_{model_name}{suffix}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

# [수정] suffix 인자 추가 (기존: def plot_pca(results, save_dir, dataset_name):)
def plot_pca(results, save_dir, dataset_name, suffix=''):
    """
    Visualizes Feature Space using PCA (2 Components).
    Shows: Clean ID (Blue), Ambiguous (Grey), OOD (Orange)
    """
    print("Generating PCA Plots...")
    sns.set_style("white")
    n_sample = 500 # Subsample for clarity
    
    for model_name in results.keys():
        if not results[model_name]['Features']: continue
        
        # Prepare Data
        X_clean = results[model_name]['Features']['ID'][:n_sample]
        X_ambig = results[model_name]['Features']['Ambiguous'][:n_sample]
        X_ood   = results[model_name]['Features']['OOD'][:n_sample]
        
        X_all = np.concatenate([X_clean, X_ambig, X_ood])
        y_all = ['ID (Clean)']*len(X_clean) + ['Ambiguous']*len(X_ambig) + ['OOD']*len(X_ood)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_all, 
                        palette={'ID (Clean)':'#1f77b4', 'Ambiguous':'#7f7f7f', 'OOD':'#ff7f0e'},
                        alpha=0.6, s=50)
        plt.title(f"{model_name} Feature Space (PCA) - {dataset_name.upper()}")
        plt.tight_layout()
        
        # [수정] 저장 파일명에 suffix 추가
        filename = f"pca_{dataset_name}_{model_name}{suffix}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

# --- Main Execution ---

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    try:
        train_loader, test_loader, ambiguous_loader, ood_loader = get_dataloaders(args, device)
        if train_loader is None: return 
        loaders = {'ID': test_loader, 'Ambiguous': ambiguous_loader, 'OOD': ood_loader}
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return

    results = {'SGD': {'Entropy': {}, 'LogDensity': {}, 'Features': {}}, 
               'SAM': {'Entropy': {}, 'LogDensity': {}, 'Features': {}}}
    
    models_to_run = [('SGD', args.model_sgd), ('SAM', args.model_sam)]
    
    for model_name, model_path in models_to_run:
        print(f"\n[{model_name}] Processing...")
        
        try:
            net = load_model(model_path, device, args.dataset, default_sn_coeff=3.0)
        except Exception as e:
            print(f"[{model_name}] CRITICAL ERROR loading model: {e}")
            continue 

        # Fit GMM
        print(f"[{model_name}] Fitting GMM...")
        embeddings, labels = get_embeddings(net, train_loader, num_dim=512, dtype=torch.float, 
                                            device=device, storage_device=device)
        try:
            gmm, jitter = gmm_fit(embeddings, labels, num_classes=10)
        except Exception as e:
            print(f"[{model_name}] GMM fitting failed: {e}")
            continue
            
        # Compute Metrics
        for data_name, loader in loaders.items():
            print(f"[{model_name}] Evaluating on {data_name}...")
            feats, ent, den = get_features_and_metrics(net, gmm, loader, device)
            
            results[model_name]['Features'][data_name] = feats
            results[model_name]['Entropy'][data_name] = ent
            results[model_name]['LogDensity'][data_name] = den

# Quantitative Analysis
    print("\n" + "="*60)
    print(" >>> Quantitative Results (Detailed Analysis) <<<")
    print("="*60)
    
    for model_name in ['SGD', 'SAM']:
        if not results[model_name]['LogDensity']: continue
        
        # 1. 데이터 준비 (Data Preparation)
        clean_id = results[model_name]['LogDensity']['ID']
        ambiguous_id = results[model_name]['LogDensity']['Ambiguous']
        ood = results[model_name]['LogDensity']['OOD']
        
        # Total ID = Clean + Ambiguous (우리가 정의한 전체 ID)
        total_id = np.concatenate([clean_id, ambiguous_id])
        
        # 2. 메트릭 계산 헬퍼 함수 (AUROC, FPR95, Overlap)
        def get_all_metrics(pos_scores, neg_scores):
            # Label: ID(pos)=1, OOD(neg)=0
            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_scores = np.concatenate([pos_scores, neg_scores])
            
            auroc = roc_auc_score(y_true, y_scores)
            fpr95 = calculate_fpr95(pos_scores, neg_scores)     # 기존 함수 사용
            overlap = calculate_overlap_area(pos_scores, neg_scores) # 기존 함수 사용
            return auroc, fpr95, overlap

        # 3. [Total ID vs OOD] 성능 측정 (기존 방식)
        auroc_total, fpr_total, overlap_total = get_all_metrics(total_id, ood)
        
        # 4. [Ambiguous ID vs OOD] 성능 측정 (추가된 분석)
        # -> SAM이 애매한 데이터를 OOD와 얼마나 헷갈려하는지 확인
        auroc_ambig, fpr_ambig, overlap_ambig = get_all_metrics(ambiguous_id, ood)

        print(f"Model: {model_name}")
        print(f"  [Total ID (Clean+Ambiguous) vs OOD]")
        print(f"    - AUROC:       {auroc_total:.4f} (Higher is better)")
        print(f"    - FPR@95%TPR:  {fpr_total:.4f} (Lower is better)")
        print(f"    - Overlap:     {overlap_total:.4f} (Lower is better)")
        print(f"  [Ambiguous ID vs OOD] (Check Collapse Effect)")
        print(f"    - AUROC:       {auroc_ambig:.4f}")
        print(f"    - FPR@95%TPR:  {fpr_ambig:.4f}")
        print(f"    - Overlap:     {overlap_ambig:.4f}")
        print("-" * 40)

    # Visualization
    print("\nGenerating Plots...")
    try:
        plot_distributions(results, args.save_dir, args.dataset, suffix=args.suffix)
        plot_pca(results, args.save_dir, args.dataset, suffix=args.suffix)
        print("Done! Check results directory.")
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
