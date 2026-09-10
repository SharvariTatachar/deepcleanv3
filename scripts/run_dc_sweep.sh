#!/bin/bash
set -euo pipefail

cd /usr/scratch/statachar3/deepcleanv3

source .venv_a6000_py311/bin/activate

mkdir -p logs runs_channel_sweep

GPUS=(0 1 2 3 4 5)
SEEDS=(0 1 2)

COUNTS=(60 80 150 200 400)
STRATEGIES=(high low mixed)

CHANNEL_SETS=()

for n in "${COUNTS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        CHANNEL_SETS+=("channelconfigs/add${n}${strategy}.ini")
    done
done

echo "Found channel configs:"
printf '%s\n' "${CHANNEL_SETS[@]}"
echo "Total configs: ${#CHANNEL_SETS[@]}"

run_one () {
    gpu=$1
    chan_config=$2
    seed=$3

    base=$(basename "$chan_config" .ini)
    dataset="data/channel_ablation_npzs/${base}.npz"

    run_name="${base}_seed${seed}"
    run_dir="runs_channel_sweep/${run_name}"

    if [[ ! -f "$chan_config" ]]; then
        echo "Missing channel config: $chan_config"
        return 1
    fi

    if [[ ! -f "$dataset" ]]; then
        echo "Missing dataset: $dataset"
        return 1
    fi

    mkdir -p "$run_dir"

    echo "Starting $run_name on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-train.py \
        --config configs/110train.ini \
        --chanslist "$chan_config" \
        --dataset-file "$dataset" \
        --seed "$seed" \
        --train-dir "$run_dir" \
        --train-t0 0 \
        --log train.log \
        > "logs/${run_name}.out" \
        2> "logs/${run_name}.err"

    echo "Finished $run_name on GPU $gpu"
}

job_id=0

for chan_config in "${CHANNEL_SETS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        gpu=${GPUS[$((job_id % ${#GPUS[@]}))]}

        run_one "$gpu" "$chan_config" "$seed" &

        job_id=$((job_id + 1))

        # wait after launching 6 jobs
        if (( job_id % ${#GPUS[@]} == 0 )); then
            wait
        fi
    done
done

wait

echo "All runs finished."