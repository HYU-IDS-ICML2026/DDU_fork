#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="0"

cd "$REPOSITORY"

echo "preflight_started=$(date --iso-8601=seconds)"
echo "repository=$REPOSITORY"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "screen=$(screen --version | head -n 1)"
python --version
python -m pip check
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

test -f "$REPOSITORY/data/cifar-10-batches-py/data_batch_1"

for run_id in \
    adamw_seed0 sam_adamw_rho0.05_seed0 sam_adamw_rho0.5_seed0 \
    adamw_seed1 sam_adamw_rho0.05_seed1 sam_adamw_rho0.5_seed1 \
    adamw_seed2 sam_adamw_rho0.05_seed2 sam_adamw_rho0.5_seed2; do
    if [[ -e "$EXPERIMENT_DIR/runs/$run_id" ]]; then
        echo "ERROR: existing run directory would be overwritten: $EXPERIMENT_DIR/runs/$run_id" >&2
        exit 1
    fi
done

python - <<'PY'
import copy
import math

import timm
import torch
import torch.nn.functional as F
import torchvision

from net.vit import CIFARViT
from utils.sam import SAM

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; refusing to start a CPU sweep")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")

device = torch.device("cuda")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
x = torch.randn(2, 3, 32, 32, device=device)
y = torch.tensor([0, 1], device=device)

base = CIFARViT(num_classes=10).to(device)
logits = base(x)
assert logits.shape == (2, 10)
assert base.feature.shape == (2, 192)
assert torch.isfinite(logits).all() and torch.isfinite(base.feature).all()

adam_model = copy.deepcopy(base)
adam = torch.optim.AdamW(
    adam_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4
)
adam_loss = F.cross_entropy(adam_model(x), y, label_smoothing=0.1)
adam_loss.backward()
adam.step()

sam_model = copy.deepcopy(base)
sam = SAM(
    sam_model.parameters(), torch.optim.AdamW, lr=1e-4, betas=(0.9, 0.999),
    eps=1e-8, weight_decay=5e-4, rho=0.5,
)
sam_loss_1 = F.cross_entropy(sam_model(x), y, label_smoothing=0.1)
sam_loss_1.backward()
sam.first_step(zero_grad=True)
sam_loss_2 = F.cross_entropy(sam_model(x), y, label_smoothing=0.1)
sam_loss_2.backward()
sam.second_step(zero_grad=True)
torch.cuda.synchronize()

assert math.isfinite(adam_loss.item())
assert math.isfinite(sam_loss_1.item()) and math.isfinite(sam_loss_2.item())
print("python_versions:", {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
})
print("cuda_device:", torch.cuda.get_device_name(0))
print("model_parameters:", sum(p.numel() for p in base.parameters()))
print("gpu_smoke:", {
    "forward": "ok",
    "adamw_step": "ok",
    "sam_adamw_rho_0.5_step": "ok",
})
PY

echo "preflight_completed=$(date --iso-8601=seconds)"
