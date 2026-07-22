"""
Fast Evaluation Script for DDU (GMM) with PCA.
Skips MSP, Energy, Mahalanobis, kNN, and Geometry stats.
Focuses solely on extracting features, applying PCA, and computing DDU AUROC.
"""
import os
import json
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# 1. Import Dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.ood_detection.mnist_ood as mnist_ood
import data.ood_detection.tiny_imagenet as tiny_imagenet

# 2. Import Networks
from net.resnet import resnet50, resnet18
from net.wide_resnet import wrn
from net.vgg import vgg16
from net.vit import vit_tiny_patch4_32
import net.spectral_normalization.spectral_norm_conv_inplace as sn_lib

# =============================================================================
# [Monkey Patch] Fix KeyError: 'weight' in SpectralNormConvLoadStateDictPreHook
# =============================================================================
original_load_hook = sn_lib.SpectralNormConvLoadStateDictPreHook.__call__

def patched_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    fn = self.fn
    version = local_metadata.get("spectral_norm_conv", {}).get(fn.name + ".version", None)
    if (version is None or version < 1) and (prefix + fn.name) not in state_dict:
        if (prefix + fn.name + "_orig") in state_dict:
            return
    return original_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

sn_lib.SpectralNormConvLoadStateDictPreHook.__call__ = patched_load_hook
# =============================================================================

# 3. Import Utils
from utils.gmm_utils import get_embeddings, gmm_fit, gmm_get_logits
from metrics.uncertainty_confidence import logsumexp

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif torch.is_tensor(obj): return obj.item() if obj.numel() == 1 else obj.tolist()
        return super(NumpyEncoder, self).default(obj)

dataset_loader = {
    "cifar10": cifar10, "cifar100": cifar100, "svhn": svhn,
    "mnist": mnist_ood, "tiny_imagenet": tiny_imagenet
}
dataset_num_classes = {
    "cifar10": 10, "cifar100": 100, "svhn": 10, "mnist": 10, "tiny_imagenet": 200
}
models = {
    "resnet50": resnet50, "resnet18": resnet18, 
    "wide_resnet": wrn, "vgg16": vgg16,
    "vit_tiny_patch4_32": vit_tiny_patch4_32,
}
model_to_num_dim = {
    "resnet50": 2048, "resnet18": 512, 
    "wide_resnet": 640, "vgg16": 512,
    "vit_tiny_patch4_32": 192,
}

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate DDU Only", allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=list(dataset_loader.keys()))
    parser.add_argument("--ood_dataset", type=str, default="svhn", choices=list(dataset_loader.keys()))
    parser.add_argument("--model", type=str, default="wide_resnet", choices=list(models.keys()))
    parser.add_argument("--sn", action="store_true")
    parser.add_argument("--mod", action="store_true")
    parser.add_argument("--coeff", type=float, default=3.0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.add_argument("--pca_dim", type=int, default=128)
    return parser.parse_args()

def compute_auroc(id_scores, ood_scores):
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_scores)

def apply_pca(train_feats, test_feats, ood_feats, n_components):
    print(f"Applying PCA: Reducing dimension from {train_feats.shape[1]} to {n_components}...")
    train_np = train_feats.cpu().numpy()
    
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_np)
    
    print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    train_pca = torch.from_numpy(pca.transform(train_np)).to(train_feats.device)
    test_pca = torch.from_numpy(pca.transform(test_feats.cpu().numpy())).to(test_feats.device)
    ood_pca = torch.from_numpy(pca.transform(ood_feats.cpu().numpy())).to(ood_feats.device)
    
    return train_pca, test_pca, ood_pca

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    # 1. Load Data
    num_classes = dataset_num_classes[args.dataset]
    test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, pin_memory=cuda)
    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.batch_size, augment=args.data_aug, val_seed=args.seed, val_size=0.1, pin_memory=cuda
    )
    if args.ood_dataset in ["mnist", "tiny_imagenet"]:
        ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, root="./data")
    else:
        ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, pin_memory=cuda)

    # 2. Build Model
    net = models[args.model](spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes, temp=args.temp)
    
    # 3. Load Weights
    print(f"Loading Weights from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.to(device)
    net.eval()

    # 4. Extract Features (DDU Only)
    print("\n--- Extracting Features (Double Precision, GPU) ---")
    dim = model_to_num_dim[args.model]
    
    train_feats, train_lbls = get_embeddings(net, train_loader, num_dim=dim, dtype=torch.double, device=device, storage_device=device)
    test_feats, _ = get_embeddings(net, test_loader, num_dim=dim, dtype=torch.double, device=device, storage_device=device)
    ood_feats, _ = get_embeddings(net, ood_test_loader, num_dim=dim, dtype=torch.double, device=device, storage_device=device)

    # 5. DDU with PCA
    print("\n--- Computing DDU Score (PCA) ---")
    results = {"metrics": {}}
    
    try:
        # [PCA]
        train_pca, test_pca, ood_pca = apply_pca(train_feats, test_feats, ood_feats, n_components=args.pca_dim)
        
        # [Fit]
        gmm_model, _ = gmm_fit(train_pca, train_lbls, num_classes)
        
        # [Score]
        ddu_id_logits = gmm_get_logits(gmm_model, test_pca)
        ddu_ood_logits = gmm_get_logits(gmm_model, ood_pca)
        
        ddu_id_score = logsumexp(ddu_id_logits)
        ddu_ood_score = logsumexp(ddu_ood_logits)
        
        ddu_auc = compute_auroc(ddu_id_score.detach().cpu().numpy(), ddu_ood_score.detach().cpu().numpy())
        
        results["metrics"]["ddu_auroc"] = ddu_auc
        print(f"DDU (GMM) AUROC: {ddu_auc:.4f}")
    except Exception as e:
        print(f"Error computing DDU: {e}")
        results["metrics"]["ddu_auroc"] = None
        results["metrics"]["ddu_error"] = str(e)

    # 6. Save
    ckpt_name = os.path.basename(args.checkpoint_path)
    save_name = f"res_{ckpt_name}_{args.ood_dataset}_ddu_pca.json"
    save_path = os.path.join(args.output_dir, save_name)
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print(f"\nSaved to: {save_path}")

if __name__ == "__main__":
    main()
