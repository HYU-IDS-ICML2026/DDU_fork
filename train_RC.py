"""
Script for training a single model for OOD detection.
Ref: Modified from train.py to include Rank Consistency Loss and Variance Regularizer.
"""

import json
import torch
import argparse
import math
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.dirty_mnist as dirty_mnist

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50, ResNet
from net.wide_resnet import wrn, WideResNet
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

# SAM
from utils.sam import SAM


# =============================================================================
# [Monkey Patch] Enable return_feature=True for ResNet and WideResNet
# 기존 모델 파일(net/*.py)을 수정하지 않고 ipynb의 로직(return_feat=True)을 지원하기 위함
# =============================================================================
def resnet_forward_patch(self, x, return_feature=False):
    out = self.activation(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    # self.feature = out.clone().detach() # Original behavior
    feature = out
    logits = self.fc(out) / self.temp
    if return_feature:
        return logits, feature
    return logits

def wideresnet_forward_patch(self, x, return_feature=False):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.activation(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.flatten(1)
    # self.feature = out.clone().detach() # Original behavior
    feature = out
    if self.num_classes is not None:
        logits = self.linear(out) / self.temp
    else:
        logits = out
        
    if return_feature:
        return logits, feature
    return logits

ResNet.forward = resnet_forward_patch
WideResNet.forward = wideresnet_forward_patch
# =============================================================================


# =============================================================================
# [Added] Loss Functions from rank_consistent_ood_sam_sequential_0810.ipynb
# =============================================================================
def listmle_topk_loss(target_scores, pred_scores, topk=10):
    # target_scores, pred_scores: [B, B] similarities (higher is more similar)
    k = min(topk, target_scores.size(1)-1)
    idx_topk = torch.topk(target_scores, k=k, dim=1).indices
    gathered = torch.gather(pred_scores, 1, idx_topk)
    loss = 0.0
    for j in range(gathered.size(1)):
        tail = gathered[:, j:]
        loss += (torch.logsumexp(tail, dim=1) - tail[:,0]).mean()
    return loss

def compute_rank_loss(logits, feats, topk=10, sym=True):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
    z = F.normalize(feats, dim=1)
    sim_f = z @ z.t()
    sim_l = (probs @ probs.t())
    eye = torch.eye(sim_f.size(0), device=sim_f.device)
    sim_f = sim_f - 2*eye
    sim_l = sim_l - 2*eye
    loss = listmle_topk_loss(sim_f, sim_l, topk=topk)
    if sym:
        loss = 0.5*(loss + listmle_topk_loss(sim_l, sim_f, topk=topk))
    return loss

def variance_regularizer(feats, labels):
    loss = 0.0
    classes = labels.unique()
    for c in classes:
        idx = (labels==c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2: continue
        zc = feats[idx]
        mu = zc.mean(dim=0, keepdim=True)
        loss += ((zc - mu)**2).sum(dim=1).mean()
    return loss / max(len(classes), 1)

# =============================================================================
# [Added] Custom Training Loop for RC Loss (Replaces utils.train_utils.train_single_epoch)
# =============================================================================
def train_single_epoch_rc(
    epoch, model, train_loader, optimizer, device, loss_mean=False, optimiser_name="sgd",
    lambda_rank=0.5, lambda_var=0.1, topk=10 
):
    model.train()
    train_loss = 0
    num_samples = 0
    log_interval = 10

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        # SAM
        if optimiser_name == "sam":
            # 1. First Step
            logits, feats = model(data, return_feature=True)
            ce = F.cross_entropy(logits, labels)
            rc = compute_rank_loss(logits, feats, topk=topk) if lambda_rank > 0 else torch.tensor(0.0, device=device)
            var = variance_regularizer(feats, labels) if lambda_var > 0 else torch.tensor(0.0, device=device)
            loss = ce + lambda_rank * rc + lambda_var * var
            
            if loss_mean: loss = loss / len(data)
            
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 2. Second Step
            logits_2, feats_2 = model(data, return_feature=True)
            ce_2 = F.cross_entropy(logits_2, labels)
            rc_2 = compute_rank_loss(logits_2, feats_2, topk=topk) if lambda_rank > 0 else torch.tensor(0.0, device=device)
            var_2 = variance_regularizer(feats_2, labels) if lambda_var > 0 else torch.tensor(0.0, device=device)
            loss_2 = ce_2 + lambda_rank * rc_2 + lambda_var * var_2
            
            if loss_mean: loss_2 = loss_2 / len(data)
            
            loss_2.backward()
            optimizer.second_step(zero_grad=True)
            
            # 기록용 loss는 첫 번째 loss 사용 (혹은 두 번째)
            train_loss += loss.item()

        # Original SGD & Adam
        else:
            optimizer.zero_grad()

            logits, feats = model(data, return_feature=True)
            ce = F.cross_entropy(logits, labels)
            rc = compute_rank_loss(logits, feats, topk=topk) if lambda_rank > 0 else torch.tensor(0.0, device=device)
            var = variance_regularizer(feats, labels) if lambda_var > 0 else torch.tensor(0.0, device=device)
            loss = ce + lambda_rank * rc + lambda_var * var

            if loss_mean:
                loss = loss / len(data)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader) * len(data),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(), # Current batch loss
                )
            )

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))
    return train_loss / num_samples


dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
    )

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    opt_params = net.parameters()
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    

    # SAM
    elif args.optimiser == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            opt_params,
            base_optimizer,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rho=args.rho, # args.rho 추가
            nesterov=args.nesterov
        )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
    )

    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        val_size=0.1,
        val_seed=args.seed,
        pin_memory=args.gpu,
    )

    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")

    training_set_loss = {}

    save_name = model_save_name(
        args.model, 
        args.sn, 
        args.mod, 
        args.coeff, 
        args.seed, 
        args.optimiser, 
        args.rho
    )
    print("Model save name", save_name)

    # --- [Settings for RC Loss] ---
    # Notebook settings: lambda_rank=0.5, lambda_var=0.1, topk=10
    lambda_rank = 0.5
    lambda_var = 0.1
    topk = 10
    # ------------------------------

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        
        # --- [Modified] Call custom training function with RC loss ---
        # Replaces: train_loss = train_single_epoch(...)
        train_loss = train_single_epoch_rc(
            epoch, net, train_loader, optimizer, device, 
            loss_mean=args.loss_mean,
            optimiser_name=args.optimiser,
            lambda_rank=lambda_rank,
            lambda_var=lambda_var,
            topk=topk
        )
        # -------------------------------------------------------------

        training_set_loss[epoch] = train_loss
        writer.add_scalar(save_name + "_train_loss", train_loss, (epoch + 1))

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
            torch.save(net.state_dict(), saved_name)

    saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)

    writer.close()
    with open(saved_name[: saved_name.rfind("_")] + "_train_loss.json", "a") as f:
        json.dump(training_set_loss, f)