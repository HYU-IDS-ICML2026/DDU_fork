import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# DDU_fork 프로젝트 모듈 임포트
from net.resnet import resnet18
from utils.sam import SAM

# [핵심] train_RC.py에서 정의들을 직접 가져옵니다.
# 이 import가 실행되는 순간 ResNet에 대한 Monkey Patch(Feature 추출 기능)도 자동으로 적용됩니다.
try:
    import train_RC
    from train_RC import compute_rank_loss, variance_regularizer
    print("Successfully imported definitions from train_RC.py")
except ImportError:
    raise ImportError("train_RC.py 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 with Mixup and RC Loss")
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_path', default='./models/dirty_cifar', type=str)
    parser.add_argument('--save_name', default=None, type=str, help='Custom file name')
    
    # Model Options
    parser.add_argument('--sn', action='store_true', help='Use Spectral Normalization')
    parser.add_argument('--coeff', default=3.0, type=float, help='SN coefficient')
    
    # Optimizer Options
    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'sam'])
    parser.add_argument('--rho', default=0.05, type=float, help='SAM rho parameter')
    
    # Mixup Options
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup interpolation coefficient')

    # [RC Loss Options]
    parser.add_argument('--use_rc', action='store_true', help='Enable Rank Consistency Loss from train_RC.py')
    parser.add_argument('--lambda_rank', default=0.5, type=float, help='Weight for Rank Loss')
    parser.add_argument('--lambda_var', default=0.1, type=float, help='Weight for Variance Regularizer')
    
    return parser.parse_args()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def compute_loss_with_rc(model, inputs, targets_a, targets_b, lam, criterion, args, device):
    """
    Mixup된 데이터에 대해 train_RC.py의 로스를 적용하는 함수
    """
    # 1. Forward Pass (return_feature=True는 train_RC 임포트 덕분에 가능해짐)
    logits, feats = model(inputs, return_feature=True)
    
    # 2. 기본 Mixup CE Loss
    ce_loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
    
    # 3. RC Loss 추가 (사용 시)
    extra_loss = 0.0
    if args.use_rc:
        # Rank Loss: Logit과 Feature 사이의 순위 일관성 (레이블 무관하므로 그대로 적용)
        if args.lambda_rank > 0:
            rc = compute_rank_loss(logits, feats)
            extra_loss += args.lambda_rank * rc
            
        # Variance Regularizer: 클래스 내 분산 억제
        # Mixup 상황이므로 target_a와 target_b에 대해 각각 계산 후 섞음
        if args.lambda_var > 0:
            var_a = variance_regularizer(feats, targets_a)
            var_b = variance_regularizer(feats, targets_b)
            var_loss = lam * var_a + (1 - lam) * var_b
            extra_loss += args.lambda_var * var_loss
            
    total_loss = ce_loss + extra_loss
    return total_loss, logits

def main():
    args = parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    print(f"Training with {args.opt.upper()} | Mixup Alpha: {args.alpha}")
    if args.use_rc:
        print(f" >>> RC Loss Enabled (via train_RC.py): Rank={args.lambda_rank}, Var={args.lambda_var}")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    model = resnet18(num_classes=10, spectral_normalization=args.sn, coeff=args.coeff, mod=True, mnist=False)
    model = model.to(device)

    # Optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'sam':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    # Training Loop
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup 적용
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda=True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            if args.opt == 'sgd':
                loss, outputs = compute_loss_with_rc(model, inputs, targets_a, targets_b, lam, criterion, args, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            elif args.opt == 'sam':
                # SAM Step 1
                loss, outputs = compute_loss_with_rc(model, inputs, targets_a, targets_b, lam, criterion, args, device)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # SAM Step 2
                loss_2, _ = compute_loss_with_rc(model, inputs, targets_a, targets_b, lam, criterion, args, device)
                loss_2.backward()
                optimizer.second_step(zero_grad=True)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
            pbar.set_postfix({'loss': train_loss / (total/args.batch_size), 'acc': 100.*correct/total})

        scheduler.step()

    # Save Model
    if args.save_name:
        model_name = args.save_name
        if not model_name.endswith('.model'): model_name += '.model'
    else:
        base = f"cifar10_{args.opt}"
        if args.opt == 'sam': base += f"_rho{args.rho}"
        if args.use_rc: base += "_RC"
        model_name = f"{base}_mixup_resnet18_sn_{args.coeff}.model"

    save_full_path = os.path.join(args.save_path, model_name)
    torch.save(model.state_dict(), save_full_path)
    print(f"Model saved to {save_full_path}")

if __name__ == "__main__":
    main()