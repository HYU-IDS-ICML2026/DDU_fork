"""
This module contains methods for training models.
"""

import torch
from torch.nn import functional as F
from torch import nn


loss_function_dict = {"cross_entropy": F.cross_entropy}


def train_single_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    device,
    loss_function="cross_entropy",
    loss_mean=False,
    optimiser_name="sgd",
    label_smoothing=0.0,
    return_accuracy=False,
    log_interval=10,
):
    """
    Util method for training a model for a single epoch.
    """
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    model.train()
    train_loss_sum = 0
    correct = 0
    num_samples = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        # SAM
        if optimiser_name in {"sam", "sam_adamw"}:
            # 1. First Step
            logits = model(data)
            raw_loss = loss_function_dict[loss_function](logits, labels, label_smoothing=label_smoothing)
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Non-finite unperturbed loss at epoch {epoch}, batch {batch_idx}")
            loss = raw_loss / len(data) if loss_mean else raw_loss
            
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 2. Second Step
            logits_2 = model(data)
            raw_loss_2 = loss_function_dict[loss_function](logits_2, labels, label_smoothing=label_smoothing)
            if not torch.isfinite(raw_loss_2):
                raise FloatingPointError(f"Non-finite perturbed loss at epoch {epoch}, batch {batch_idx}")
            loss_2 = raw_loss_2 / len(data) if loss_mean else raw_loss_2
            
            loss_2.backward()
            optimizer.second_step(zero_grad=True)
            
        # Original SGD, Adam, and AdamW
        else:
            optimizer.zero_grad()

            logits = model(data)
            raw_loss = loss_function_dict[loss_function](logits, labels, label_smoothing=label_smoothing)
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_idx}")
            loss = raw_loss / len(data) if loss_mean else raw_loss

            loss.backward()
            optimizer.step()

        # Record the unperturbed, unscaled first-forward metrics.
        train_loss_sum += raw_loss.item() * len(data)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader) * len(data),
                    100.0 * batch_idx / len(train_loader),
                    raw_loss.item(),
                )
            )

    avg_loss = train_loss_sum / num_samples
    accuracy = correct / num_samples
    print("====> Epoch: {} Average loss: {:.4f}, Accuracy: {:.4f}".format(epoch, avg_loss, accuracy))
    return (avg_loss, accuracy) if return_accuracy else avg_loss


def test_single_epoch(epoch, model, test_val_loader, device, loss_function="cross_entropy"):
    """
    Util method for testing a model for a single epoch.
    """
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for data, labels in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += loss_function_dict[loss_function](logits, labels).item()
            num_samples += len(data)

    print("======> Test set loss: {:.4f}".format(loss / num_samples))
    return loss / num_samples


def model_save_name(model_name, sn, mod, coeff, seed, optimizer="sgd", rho=0.0):
    if sn:
        if mod:
            strn = "_sn_" + str(coeff) + "_mod_"
        else:
            strn = "_sn_" + str(coeff) + "_"
    else:
        if mod:
            strn = "_mod_"
        else:
            strn = "_"

    # Optimizer & Rho config
    if optimizer == "sgd":
        opt_str = "_sgd_0"
    elif optimizer in {"sam", "sam_adamw"}:
        opt_str = f"_{optimizer}_{rho}"
    else:
        opt_str = f"_{optimizer}_0"


    return opt_str + str(model_name) + strn + str(seed)
