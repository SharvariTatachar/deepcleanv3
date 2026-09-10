#!/bin/bash
set -euo pipefail

cd /usr/scratch/statachar3/deepcleanv3
source .venv_a6000_py311/bin/activate

mkdir -p logs runs_longtrain

GPUS=(0 1 2)
SEED=0

CONFIGS=(
  add60mixed
  add150low
  add400high
)

run_one () {
    gpu=$1
    config_name=$2

    chan_config="channelconfigs/${config_name}.ini"
    dataset="data/channel_ablation_npzs/${config_name}.npz"

    run_name="${config_name}_seed${SEED}_epoch40"
    run_dir="runs_longtrain/${run_name}"

    if [[ ! -f "$chan_config" ]]; then
        echo "Missing channel config: $chan_config"
        return 0
    fi

    if [[ ! -f "$dataset" ]]; then
        echo "Missing dataset: $dataset"
        return 0
    fi

    mkdir -p "$run_dir"

    echo "Starting ${run_name} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-train.py \
        --config configs/110train.ini \
        --chanslist "$chan_config" \
        --dataset-file "$dataset" \
        --seed "$SEED" \
        --train-dir "$run_dir" \
        --train-t0 0 \
        --log train.log \
        > "logs/${run_name}.out" \
        2> "logs/${run_name}.err"

    echo "Finished ${run_name}"
}

job_id=0

for config_name in "${CONFIGS[@]}"; do
    gpu=${GPUS[$((job_id % ${#GPUS[@]}))]}

    run_one "$gpu" "$config_name" &

    job_id=$((job_id + 1))

    if (( job_id % ${#GPUS[@]} == 0 )); then
        wait
    fi
done

wait
echo "All long-training runs complete."