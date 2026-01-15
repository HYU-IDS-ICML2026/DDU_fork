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

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 with Mixup for DDU Experiment")
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=100, type=int) # 논문 재현용으로는 100~150이면 충분 (350은 너무 김)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_path', default='./models/dirty_cifar', type=str)
    parser.add_argument('--save_name', default=None, type=str, help='Custom file name for the model')
    
    # Model Options
    parser.add_argument('--sn', action='store_true', help='Use Spectral Normalization')
    parser.add_argument('--coeff', default=3.0, type=float, help='SN coefficient')
    
    # Optimizer Options
    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'sam'])
    parser.add_argument('--rho', default=0.05, type=float, help='SAM rho parameter')
    
    # Mixup Options
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup interpolation coefficient (default: 1.0)')
    
    return parser.parse_args()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
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

def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    print(f"Training with {args.opt.upper()} | Mixup Alpha: {args.alpha} | SN: {args.sn}")

    # Data (CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model (ResNet18 + SN)
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
            
            # 1. Apply Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda=True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            # 2. Forward & Backward
            if args.opt == 'sgd':
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            elif args.opt == 'sam':
                # SAM First Step
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # SAM Second Step
                mixup_criterion(criterion, model(inputs), targets_a, targets_b, lam).backward()
                optimizer.second_step(zero_grad=True)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # Mixup Accuracy (approx)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
            pbar.set_postfix({'loss': train_loss / (total/args.batch_size), 'acc': 100.*correct/total})

        scheduler.step()


    # --- [수정된 부분] 파일 저장 로직 ---
    if args.save_name:
        # 1. 사용자가 직접 이름을 지정한 경우
        model_name = args.save_name
        if not model_name.endswith('.model'): model_name += '.model'
    else:
        # 2. 자동으로 이름을 생성하는 경우 (SAM이면 rho 포함)
        if args.opt == 'sam':
            model_name = f"cifar10_{args.opt}_rho{args.rho}_mixup_resnet18_sn_{args.coeff}.model"
        else:
            model_name = f"cifar10_{args.opt}_mixup_resnet18_sn_{args.coeff}.model"

    save_full_path = os.path.join(args.save_path, model_name)
    torch.save(model.state_dict(), save_full_path)
    print(f"Model saved to {save_full_path}")

if __name__ == "__main__":
    main()