"""
Script for training a single model for OOD detection.

[변경 사항 요약]
1) 옵티마이저에 SAM(Sharpness-Aware Minimization) 추가
   - 참고 논문: Foret et al., ICLR 2021
   - 참고 구현: https://github.com/davda54/sam (MIT License)
   - [변경 사항] 기존 SGD의 단일 step을 SAM 알고리즘의 두 단계 업데이트(first_step/second_step)로 치환

2) GPU 사용 방식 개선
   - [변경 사항] 기존 torch.nn.DataParallel 기본 적용을 옵션(--use-data-parallel)으로 변경
   - 기본은 단일 GPU(cuda:{gpu_id}) 사용
   - CUDA_VISIBLE_DEVICES와 함께 쓰면 스크립트 내부에서는 보통 --gpu-id 0 사용

3) 체크포인트 파일명 커스터마이즈
   - --output-name으로 최종 저장 파일명을 지정 가능
   - 미지정 시 "{model}_{dataset}_{opt}_rho{rho}_last.model" 형태로 자동 생성
"""
#train.py저장폴더에 겹치지않게 시간정보추가
from datetime import datetime

import os
import json
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.dirty_mnist as dirty_mnist

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train args (DDU original)
from utils.args import training_args

# DDU utilities (name helper)
from utils.train_utils import model_save_name

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

# [추가] SAM 옵티마이저 + BN running stats 헬퍼
from utils.sam import SAM, enable_running_stats, disable_running_stats


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


def build_criterion(args) -> nn.Module:
    # DDU 기본 분류 학습은 cross-entropy를 사용한다고 가정(원본 train.py도 분류 목적) :contentReference[oaicite:6]{index=6}
    loss_name = getattr(args, "loss_function", "cross_entropy")
    if loss_name in ["cross_entropy", "ce", "xent"]:
        return nn.CrossEntropyLoss(reduction="mean")
    raise ValueError(f"Unsupported loss_function: {loss_name}")


def auto_output_name(args, optimizer_type: str, sam_rho: float) -> str:
    """
    [변경 사항] 서로 다른 실험 결과가 덮어써지지 않도록
    모델/데이터셋/옵티마이저/rho 정보를 포함해 기본 파일명을 자동 생성
    """
    if optimizer_type == "sam":
        rho_txt = str(sam_rho).replace(".", "_")
        return f"{args.model}_{args.dataset}_sam_rho{rho_txt}_last.model"
    return f"{args.model}_{args.dataset}_sgd_last.model"


# train.py 학습 결과 저장 폴더명 규칙 및 생성
def build_run_directory(args, optimizer_type: str, sam_rho: float) -> str:
    """
    Create a unique run directory under ./train_results with the pattern:
    model_dataset_optimizer[_rhoX]_sn{coeff|nosn}_seed{seed}_{MMDD_HHMMSS}
    """

    base_dir = os.path.join(args.save_loc, "")
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    parts = [args.model, args.dataset, optimizer_type]
    if optimizer_type == "sam":
        parts.append(f"rho{str(sam_rho).replace('.', '_')}")
    sn_part = f"sn{str(args.coeff).replace('.', '_')}" if args.sn else "nosn"
    parts.extend([sn_part, f"seed{args.seed}", timestamp])
    run_dir = os.path.join(base_dir, "_".join(parts))

    os.makedirs(run_dir, exist_ok=True)
    return run_dir



def train_one_epoch_sgd(net, loader, optimizer, device, criterion):
    net.train()
    total_loss = 0.0
    total_n = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def train_one_epoch_sam(net, loader, optimizer: SAM, device, criterion):
    """
    [변경 사항] SAM 학습 루프 구현
    - 논문 Algorithm 1 및 공식 구현 예시(davda54/sam)를 참고하여 작성 :contentReference[oaicite:7]{index=7}
    - 패턴:
        (1) enable_running_stats -> forward/backward -> first_step
        (2) disable_running_stats -> full forward/backward -> second_step
    """
    net.train()
    total_loss = 0.0
    total_n = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---------- 1st forward-backward ----------
        enable_running_stats(net)  # [BN] 1st pass에서만 running stats 업데이트
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # ---------- 2nd forward-backward ----------
        disable_running_stats(net)  # [BN] 2nd pass에서 running stats 업데이트 방지
        # "반드시 full forward" (same batch, perturbed weights)
        criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


if __name__ == "__main__":
    args = training_args().parse_args()
    print("Parsed args", args)

    # -------------------------
    # Reproducibility
    # -------------------------
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    if cuda:
        # [변경 사항] 단일 GPU 선택: CUDA_VISIBLE_DEVICES와 조합하면 보통 gpu-id=0 사용
        gpu_id = getattr(args, "gpu_id", 0)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print("CUDA set:", cuda, "| device:", device)

    # -------------------------
    # Model
    # -------------------------
    num_classes = dataset_num_classes[args.dataset]
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
    ).to(device)

    # [변경 사항] DataParallel 기본 적용 제거 -> 옵션으로만 사용
    use_dp = bool(getattr(args, "use_data_parallel", False))
    if cuda and use_dp and torch.cuda.device_count() > 1:
        # NOTE: 팀 실험 목적이 "옵티마이저만 교체"인 경우,
        # DataParallel은 또 다른 변수를 만들 수 있어 기본 False를 권장합니다.
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

    cudnn.benchmark = True

    # -------------------------
    # Optimizer (SGD vs SAM)
    # -------------------------
    # 기존 DDU args.optimiser(영국식)와 새 args.optimizer_type(미국식)를 병행 지원
    optimizer_type = getattr(args, "optimizer_type", None) or getattr(args, "optimiser", "sgd")
    optimizer_type = optimizer_type.lower()

    opt_params = net.parameters()

    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
        sam_rho = None

    elif optimizer_type == "sam":
        sam_rho = float(getattr(args, "sam_rho", 0.05))
        sam_adaptive = bool(getattr(args, "sam_adaptive", False))

        # [변경 사항] SAM 래퍼로 SGD를 감싼다 (Foret et al., ICLR 2021 / davda54/sam) :contentReference[oaicite:8]{index=8}
        optimizer = SAM(
            opt_params,
            base_optimizer=optim.SGD,
            rho=sam_rho,
            adaptive=sam_adaptive,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type} (use sgd|sam)")

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.first_milestone, args.second_milestone],
        gamma=0.1,
    )

    # -------------------------
    # Data
    # -------------------------
    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        val_size=0.1,
        val_seed=args.seed,
        pin_memory=args.gpu,
    )

    # -------------------------
    # Logging / Saving
    # -------------------------
    
    #기존 부분 제거
    """os.makedirs(args.save_loc, exist_ok=True)
    stats_dir = os.path.join(args.save_loc, "stats_logging")"""
    
    #[추가]
    run_dir = build_run_directory(args, optimizer_type, sam_rho if sam_rho is not None else 0.0)
    print(f"[SAVE] Run directory: {run_dir}")

    stats_dir = os.path.join(run_dir, "stats_logging")
    
    #[기존]
    os.makedirs(stats_dir, exist_ok=True)

    writer = SummaryWriter(stats_dir)

    training_set_loss = {}

    # DDU 원본 네이밍(호환성) + 옵티마이저 정보를 추가한 run tag
    base_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
    if optimizer_type == "sam":
        run_tag = f"{base_name}_sam_rho{str(sam_rho).replace('.', '_')}"
    else:
        run_tag = f"{base_name}_sgd"

    print("Run tag:", run_tag)

    # 최종 저장 파일명
    output_name = getattr(args, "output_name", None)
    if output_name is None:
        output_name = auto_output_name(args, optimizer_type, sam_rho if sam_rho is not None else 0.0)

    # -------------------------
    # Train
    # -------------------------
    criterion = build_criterion(args)

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)

        if optimizer_type == "sam":
            train_loss = train_one_epoch_sam(net, train_loader, optimizer, device, criterion)
        else:
            train_loss = train_one_epoch_sgd(net, train_loader, optimizer, device, criterion)

        training_set_loss[epoch] = train_loss
        writer.add_scalar(run_tag + "_train_loss", train_loss, (epoch + 1))

        scheduler.step()

        # 주기 저장(원본 기능 유지)
        if (epoch + 1) % args.save_interval == 0:
            ckpt_name = f"{run_tag}_ep{epoch+1}.model"
            '''ckpt_path = os.path.join(args.save_loc, ckpt_name)'''
            #[변경 사항] run_dir 사용
            ckpt_path = os.path.join(run_dir, ckpt_name)
            torch.save(net.state_dict(), ckpt_path)
            print("Checkpoint saved to:", ckpt_path)

    # 최종(last) 저장: output-name 사용
    '''last_path = os.path.join(args.save_loc, output_name)'''
    #[변경 사항] run_dir 사용
    last_path = os.path.join(run_dir, output_name)
    torch.save(net.state_dict(), last_path)
    print("Final model saved to:", last_path)

    writer.close()

    # loss json도 output-name 기반으로 저장
    '''loss_json_path = os.path.join(args.save_loc, output_name.replace(".model", "") + "_train_loss.json")'''
    #[변경 사항] run_dir 사용
    loss_json_path = os.path.join(run_dir, output_name.replace(".model", "") + "_train_loss.json")
    with open(loss_json_path, "w") as f:
        json.dump(training_set_loss, f)
    print("Train loss json saved to:", loss_json_path)
