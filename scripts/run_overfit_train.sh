#!/bin/bash

# -----------------------------------------
# Per-channel CNN / PCC (GPUs 0-2)
# -----------------------------------------
for seed in 0 1 2; do
    gpu=$((seed))

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-transform-train.py \
        --config configs/118dcv3train.ini \
        --chanslist channelconfigs/add150low.ini \
        --dataset-file data/channel_ablation_npzs/add150low.npz \
        --train-t0 0 \
        --seed $seed \
        --max-epochs 60 \
        --batch-size 8 \
        --lr 1e-3 \
        --weight-decay 0 \
        --train-dir runs_bottleneck_dc_150_seed${seed} \
        --log train.log &
done

# -----------------------------------------
# Per-channel CNN / PCC (GPUs 3-5)
# -----------------------------------------
for seed in 0 1 2; do
    gpu=$((seed+3))

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-transform-train.py \
        --config configs/118dcv3train.ini \
        --chanslist channelconfigs/add150low.ini \
        --dataset-file data/channel_ablation_npzs/add150low.npz \
        --train-t0 0 \
        --seed $seed \
        --max-epochs 100 \
        --batch-size 8 \
        --lr 1e-3 \
        --weight-decay 0 \
        --train-dir runs_pcc_8layers_150_seed${seed} \
        --log train.log &
done

wait

echo "All overfit experiments complete."