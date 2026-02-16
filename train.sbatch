#!/bin/bash 
#SBATCH --job-name=train 
#SBATCH --account=gts-pli77-ideas_l40s
#SBATCH --partition=gpu
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=48G 
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out 
#SBATCH --error=logs/%x-%j.err 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=statachar3@gatech.edu

set -euo pipefail

cd ~/r-pli77-0/deepcleanv3 
mkdir -p logs 
source .venv/bin/activate 
srun python3 dc-transform-train.py 