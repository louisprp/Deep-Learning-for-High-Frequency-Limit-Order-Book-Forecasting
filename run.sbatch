#!/bin/bash
#SBATCH --job-name=deeplob
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=42-44
#SBATCH --output=logs/%x-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
source ../venv/bin/activate
python -c "
import os
from models import CONFIGS
from training import train
for name in CONFIGS:
    train(name, seeds=(int(os.environ['SLURM_ARRAY_TASK_ID']),))
"
