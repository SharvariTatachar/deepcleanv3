#!/bin/bash

set -uo pipefail

if (( $# < 2 )); then
    echo "Usage: bash $0 <config.ini> <band>"
    echo "Example: bash $0 configs/110dcv3train.ini 110-130"
    exit 1
fi

CONFIG=$1
BAND=$2

TRAIN_SCRIPT=dc-transform-train.py

DATA_DIR="data/channel_ablation_npzs/${BAND}/granular"
CHANNEL_DIR="channelconfigs/${BAND}/granular"

RUN_ROOT="runs_pcc_ablation_${BAND}"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

SIZES=(
    add200low
    add400low
)

AVAILABLE_GPUS=(0 2 4)
NUM_GPUS=${#AVAILABLE_GPUS[@]}

job_index=0
pids=()


run_job() {
    local gpu=$1
    local exp=$2
    local seed=$3

    local chanfile
    local datafile
    local run_dir
    local log_file
    local status

    if [[ "${exp}" == "baseline" ]]; then
        chanfile="channelconfigs/${BAND}/ogchannels.ini"
        datafile="data/channel_ablation_npzs/${BAND}/ogchannels.npz"
    else
        chanfile="${CHANNEL_DIR}/${exp}.ini"
        datafile="${DATA_DIR}/${exp}.npz"
    fi

    run_dir="${RUN_ROOT}/${exp}_seed${seed}"
    log_file="${LOG_DIR}/${exp}_seed${seed}.log"

    if [[ ! -f "${chanfile}" ]]; then
        echo "[GPU ${gpu}] Missing channel file: ${chanfile}"
        return 1
    fi

    if [[ ! -f "${datafile}" ]]; then
        echo "[GPU ${gpu}] Missing dataset file: ${datafile}"
        return 1
    fi

    if [[ -f "${run_dir}/dc_pcc_run.json" ]]; then
        echo "[GPU ${gpu}] Skipping completed run: ${exp}, seed ${seed}"
        return 0
    fi

    echo "=================================================="
    echo "[GPU ${gpu}] ${exp}, seed ${seed}"
    echo "Channels: ${chanfile}"
    echo "Dataset:  ${datafile}"
    echo "Run dir:  ${run_dir}"
    echo "Log:      ${log_file}"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python3 "${TRAIN_SCRIPT}" \
        --config "${CONFIG}" \
        --dataset-file "${datafile}" \
        --chanslist "${chanfile}" \
        --train-t0 0 \
        --train-dir "${run_dir}" \
        --max-epochs 35 \
        --seed "${seed}" \
        > "${log_file}" 2>&1

    status=$?

    if [[ ${status} -eq 0 ]]; then
        echo "[GPU ${gpu}] Finished ${exp}, seed ${seed}"
    else
        echo "[GPU ${gpu}] FAILED ${exp}, seed ${seed}; see ${log_file}"
    fi

    return "${status}"
}


for seed in 0 1 2; do
    for exp in "${SIZES[@]}"; do
        gpu=${AVAILABLE_GPUS[$((job_index % NUM_GPUS))]}

        run_job "${gpu}" "${exp}" "${seed}" &
        pids+=("$!")

        job_index=$((job_index + 1))

        # Wait after launching one job per available GPU.
        if (( ${#pids[@]} == NUM_GPUS )); then
            echo
            echo "Waiting for current batch of ${NUM_GPUS} jobs..."
            echo

            for pid in "${pids[@]}"; do
                wait "${pid}" || true
            done

            pids=()
        fi
    done
done

if (( ${#pids[@]} > 0 )); then
    echo
    echo "Waiting for final batch of ${#pids[@]} jobs..."
    echo

    for pid in "${pids[@]}"; do
        wait "${pid}" || true
    done
fi

echo
echo "All PCC channel-ablation runs finished."
echo "Runs: ${RUN_ROOT}"
echo "Logs: ${LOG_DIR}"