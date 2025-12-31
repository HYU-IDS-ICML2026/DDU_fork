"""
Script for training a single model for OOD detection.
"""

import json
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

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

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

# SAM
from utils.sam import SAM



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
    if cuda:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU {args.gpu_id}")
    else:
        device = torch.device("cpu")
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
        net.to(device)
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

    train_loader, valid_loader = dataset_loader[args.dataset].get_train_valid_loader(
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
    validation_set_loss = {}
    validation_set_accuracy = {}

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

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss = train_single_epoch(
            epoch, net, train_loader, optimizer, device, loss_function=args.loss_function, loss_mean=args.loss_mean,
            optimiser_name=args.optimiser # SAM
        )

        training_set_loss[epoch] = train_loss
        writer.add_scalar(save_name + "_train_loss", train_loss, (epoch + 1))

        # Validation at the end of each epoch
        print("Validating epoch", epoch)
        val_loss, val_accuracy = test_single_epoch(epoch, net, valid_loader, device, loss_function=args.loss_function)
        validation_set_loss[epoch] = val_loss
        validation_set_accuracy[epoch] = val_accuracy
        
        writer.add_scalar(save_name + "_val_loss", val_loss, (epoch + 1))
        writer.add_scalar(save_name + "_val_accuracy", val_accuracy, (epoch + 1))
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
            torch.save(net.state_dict(), saved_name)

    saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)

    writer.close()

    # Save hyperparameters
    args_dict = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    with open(saved_name[: saved_name.rfind("_")] + "_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    with open(saved_name[: saved_name.rfind("_")] + "_train_loss.json", "w") as f:
        json.dump(training_set_loss, f, indent=4)
    
    with open(saved_name[: saved_name.rfind("_")] + "_val_loss.json", "w") as f:
        json.dump(validation_set_loss, f, indent=4)
    
    with open(saved_name[: saved_name.rfind("_")] + "_val_accuracy.json", "w") as f:
        json.dump(validation_set_accuracy, f, indent=4)
