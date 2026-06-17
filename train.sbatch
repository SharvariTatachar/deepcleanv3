#!/bin/bash
#SBATCH --job-name=dc_transform_4layers
#SBATCH --account=gts-pli77-ideas_l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/dc_transform_4layers_%j.out
#SBATCH --error=logs/dc_transform_4layers_%j.err

set -euo pipefail

cd ~/r-pli77-0/deepcleanv3 
mkdir -p logs 
source .venv/bin/activate 
srun python3 dc-transform-train.py 