#!/bin/bash

configs=(
    "add80low channelconfigs/add40low.ini data/channel_ablation_npzs/add80low.npz"

)

for cfg in "${configs[@]}"; do
    read name chans npz <<< "$cfg"

    echo "===== $name ====="

    for seed in 0 1 2; do
        gpu=$((seed + 3))

        CUDA_VISIBLE_DEVICES=$gpu python3 dc-transform-train.py \
            --config configs/118dcv3train.ini \
            --chanslist "$chans" \
            --dataset-file "$npz" \
            --train-t0 0 \
            --seed $seed \
            --max-epochs 20 \
            --train-dir runs_channel_ablation_pcc/${name}_seed${seed} \
            --log train.log &

    done

    wait
done