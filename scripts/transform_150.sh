#!/bin/bash

source .venv_a6000_py311/bin/activate

for seed in 0 1 2; do

    CUDA_VISIBLE_DEVICES=$seed \
    python3 dc-transform-train.py \
        --config configs/118dcv3train.ini \
        --chanslist channelconfigs/add150low.ini \
        --dataset-file data/channel_ablation_npzs/add150low.npz \
        --train-t0 0 \
        --seed $seed \
        --train-dir runs_transform_add150low_seed${seed} \
        --log train.log &

done

wait

echo "All 3 runs complete."