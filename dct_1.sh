#!/bin/bash

set -u

SEED_GPUS=(3 4 5)

TRAIN_SCRIPT="dc-transform-train.py"
CONFIG="configs/110dcv3train.ini"

exp="baseline"

RUN_ROOT="runs_dct_baseline_110-130_modularlr"
LOG_DIR="logs_dct_baseline_110-130_modularlr"

datafile="data/channel_ablation_npzs/ogchannels.npz"
chanfile="channelconfigs/110-130/ogchannels.ini"

mkdir -p "${RUN_ROOT}"
mkdir -p "${LOG_DIR}"

pids=()

for i in "${!SEED_GPUS[@]}"; do
    seed="${i}"
    gpu="${SEED_GPUS[$i]}"

    run_dir="${RUN_ROOT}/${exp}_seed${seed}"
    log_file="${LOG_DIR}/${exp}_seed${seed}.log"

    echo "Starting ${exp} seed ${seed} on GPU ${gpu}"
    echo "  run directory: ${run_dir}"
    echo "  log file: ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python3 "${TRAIN_SCRIPT}" \
        --config "${CONFIG}" \
        --dataset-file "${datafile}" \
        --chanslist "${chanfile}" \
        --train-t0 0 \
        --train-dir "${run_dir}" \
        --seed "${seed}" \
        --max-epochs 30 \
        > "${log_file}" 2>&1 &

    pids+=("$!")
done

failed=0

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    seed="${i}"
    gpu="${SEED_GPUS[$i]}"

    if wait "${pid}"; then
        echo "Finished: ${exp} seed ${seed} on GPU ${gpu}"
    else
        status=$?
        echo "FAILED: ${exp} seed ${seed} on GPU ${gpu} (exit code ${status})"
        failed=1
    fi
done

echo
echo "All DC-T baseline runs finished."
echo "Runs: ${RUN_ROOT}"
echo "Logs: ${LOG_DIR}"

exit "${failed}"