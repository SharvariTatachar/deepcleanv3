#!/bin/bash

set -uo pipefail

CONFIG="configs/110dcv3train.ini"
TRAIN_SCRIPT="dc-train.py"

DATAFILE="data/channel_ablation_npzs/ogchannels.npz"
CHANFILE="channelconfigs/110-130/ogchannels.ini"

RUN_ROOT="runs_dc_baseline_110-130"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

AVAILABLE_GPUS=(0 1 2)
pids=()

run_job() {
    local gpu=$1
    local seed=$2

    local exp="baseline"
    local run_dir="${RUN_ROOT}/${exp}_seed${seed}"
    local log_file="${LOG_DIR}/${exp}_seed${seed}.log"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "Missing config: ${CONFIG}"
        return 1
    fi

    if [[ ! -f "${CHANFILE}" ]]; then
        echo "Missing channel file: ${CHANFILE}"
        return 1
    fi

    if [[ ! -f "${DATAFILE}" ]]; then
        echo "Missing dataset file: ${DATAFILE}"
        return 1
    fi

    echo "=================================================="
    echo "[GPU ${gpu}] ${exp}, seed ${seed}"
    echo "Config:   ${CONFIG}"
    echo "Channels: ${CHANFILE}"
    echo "Dataset:  ${DATAFILE}"
    echo "Run dir:  ${run_dir}"
    echo "Log:      ${log_file}"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python3 "${TRAIN_SCRIPT}" \
        --config "${CONFIG}" \
        --dataset-file "${DATAFILE}" \
        --chanslist "${CHANFILE}" \
        --train-t0 0 \
        --max-epochs 30 \
        --train-dir "${run_dir}" \
        --seed "${seed}" \
        > "${log_file}" 2>&1

    local status=$?

    if [[ ${status} -eq 0 ]]; then
        echo "[GPU ${gpu}] Finished ${exp}, seed ${seed}"
    else
        echo "[GPU ${gpu}] FAILED ${exp}, seed ${seed}; see ${log_file}"
    fi

    return ${status}
}

for seed in 0 1 2; do
    gpu=${AVAILABLE_GPUS[$seed]}

    run_job "${gpu}" "${seed}" &
    pids+=("$!")
done

echo
echo "Waiting for all three runs..."
echo

for pid in "${pids[@]}"; do
    wait "${pid}" || true
done

echo
echo "All DeepClean baseline runs finished."
echo "Runs: ${RUN_ROOT}"
echo "Logs: ${LOG_DIR}"