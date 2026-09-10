#!/bin/bash

for seed in 0 1 2; do
    gpu=$((seed))

    CUDA_VISIBLE_DEVICES=$gpu python3 dc-transform-train.py \
        --config configs/118dcv3train.ini \
        --chanslist channelconfigs/ogchannels.ini \
        --dataset-file data/channel_ablation_npzs/ogchannels.npz \
        --train-t0 0 \
        --filt-fl 110 \
        --filt-fh 130 \
        --seed $seed \
        --max-epochs 50 \
        --cnn-lr 1e-3 \
        --transformer-lr 1e-3 \
        --train-dir runs_pcc_150_110-130_${seed} \
        --log train.log &
done

wait
echo "All runs complete."