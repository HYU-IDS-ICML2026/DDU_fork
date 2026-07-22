#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
LOG_DIR="$EXPERIMENT_DIR/logs"
RUNS_DIR="$EXPERIMENT_DIR/runs"
STATUS_FILE="$EXPERIMENT_DIR/status.tsv"
SNAPSHOT_DIR="$EXPERIMENT_DIR/source_snapshot"
CHECKSUM_FILE="$EXPERIMENT_DIR/code_sha256.txt"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"

mkdir -p "$LOG_DIR" "$RUNS_DIR"
cd "$REPOSITORY"

if [[ -e "$STATUS_FILE" ]]; then
    echo "ERROR: status file already exists; refusing to overwrite an existing sweep" >&2
    exit 1
fi

bash "$EXPERIMENT_DIR/preflight.sh" 2>&1 | tee "$LOG_DIR/preflight_at_launch.log"

source_files=(
    train.py
    net/vit.py
    utils/args.py
    utils/train_utils.py
    utils/sam.py
    data/ood_detection/cifar10.py
    environment.yml
)
if [[ -e "$SNAPSHOT_DIR" || -e "$CHECKSUM_FILE" ]]; then
    echo "ERROR: source snapshot already exists; refusing to overwrite it" >&2
    exit 1
fi
mkdir -p "$SNAPSHOT_DIR"
cp --parents "${source_files[@]}" "$SNAPSHOT_DIR"
sha256sum "${source_files[@]}" > "$CHECKSUM_FILE"
git diff --binary -- "${source_files[@]}" > "$EXPERIMENT_DIR/tracked_changes.patch"

printf 'run_id\tstatus\tstarted_at\tfinished_at\texit_code\n' > "$STATUS_FILE"

run_ids=(
    adamw_seed0 sam_adamw_rho0.05_seed0 sam_adamw_rho0.5_seed0
    adamw_seed1 sam_adamw_rho0.05_seed1 sam_adamw_rho0.5_seed1
    adamw_seed2 sam_adamw_rho0.05_seed2 sam_adamw_rho0.5_seed2
)
optimizers=(adamw sam_adamw sam_adamw adamw sam_adamw sam_adamw adamw sam_adamw sam_adamw)
rhos=(0.0 0.05 0.5 0.0 0.05 0.5 0.0 0.05 0.5)
seeds=(0 0 0 1 1 1 2 2 2)

for index in "${!run_ids[@]}"; do
    run_id="${run_ids[$index]}"
    optimizer="${optimizers[$index]}"
    rho="${rhos[$index]}"
    seed="${seeds[$index]}"
    run_dir="$RUNS_DIR/$run_id"
    log_file="$LOG_DIR/$run_id.log"

    if [[ -e "$run_dir" || -e "$log_file" ]]; then
        echo "ERROR: output already exists for $run_id; refusing to overwrite" >&2
        exit 1
    fi

    mkdir -p "$run_dir"
    ln -sfn "logs/$run_id.log" "$EXPERIMENT_DIR/current.log"
    sha256sum --check "$CHECKSUM_FILE"

    command=(
        python -u train.py
        --seed "$seed"
        --dataset cifar10
        --model vit_tiny_patch4_32
        --opt "$optimizer"
        --rho "$rho"
        -e 200
        -b 64
        --lr 1e-4
        --decay 5e-4
        --beta1 0.9
        --beta2 0.999
        --adam-eps 1e-8
        --label-smoothing 0.1
        --scheduler cosine
        --data-aug
        --autoaugment
        --save-interval 50
        --log-interval 100
        --save-path "$run_dir"
    )

    printf '%q ' "${command[@]}" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    {
        echo "recorded_at=$(date --iso-8601=seconds)"
        echo "branch=$(git branch --show-current)"
        echo "commit=$(git rev-parse HEAD)"
        echo "git_status_begin"
        git status --short
        echo "git_status_end"
        python --version
        python - <<'PY'
import timm
import torch
import torchvision
print("torch=" + torch.__version__)
print("torchvision=" + torchvision.__version__)
print("timm=" + timm.__version__)
print("cuda_runtime=" + str(torch.version.cuda))
print("cuda_device=" + torch.cuda.get_device_name(0))
PY
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    } > "$run_dir/environment.txt"

    started_at="$(date --iso-8601=seconds)"
    printf '%s\tRUNNING\t%s\t\t\n' "$run_id" "$started_at" >> "$STATUS_FILE"
    echo "[$started_at] START $run_id"

    set +e
    "${command[@]}" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    finished_at="$(date --iso-8601=seconds)"
    if [[ "$exit_code" -ne 0 ]]; then
        printf '%s\tFAILED\t%s\t%s\t%s\n' "$run_id" "$started_at" "$finished_at" "$exit_code" >> "$STATUS_FILE"
        echo "[$finished_at] FAILED $run_id exit_code=$exit_code"
        exit "$exit_code"
    fi

    printf '%s\tCOMPLETED\t%s\t%s\t0\n' "$run_id" "$started_at" "$finished_at" >> "$STATUS_FILE"
    echo "[$finished_at] COMPLETED $run_id"
done

touch "$EXPERIMENT_DIR/SWEEP_COMPLETED"
echo "sweep_completed=$(date --iso-8601=seconds)"
