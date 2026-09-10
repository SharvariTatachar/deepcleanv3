#!/bin/bash

for seed in 0 1 2; do
    gpu=$((seed))

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-train.py \
        --config configs/110train.ini \
        --chanslist channelconfigs/add150low.ini \
        --dataset-file data/channel_ablation_npzs/add150low.npz \
        --train-t0 0 \
        --seed $seed \
        --max-epochs 20 \
        --train-dir runs_dc_150_110-130_${seed} \
        --log train.log &
done

for seed in 0 1 2; do
    gpu=$((seed + 3))

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-train.py \
        --config configs/110train.ini \
        --chanslist channelconfigs/ogchannels.ini \
        --dataset-file data/channel_ablation_npzs/ogchannels.npz \
        --train-t0 0 \
        --seed $seed \
        --max-epochs 20 \
        --train-dir runs_dc_35_110-130_${seed} \
        --log train.log &
done

wait
echo "All runs complete."