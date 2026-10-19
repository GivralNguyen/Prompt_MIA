#!/bin/bash
#SBATCH --job-name=pmia_fig2_3_18
#SBATCH --output=logs/fig2_3_18-%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Usage (from nips-prompt-mia-camera-ready/):  sbatch slurm/fig2_3_18.sbatch [extra train.py args]
set -euo pipefail
PY=${PY:-/blue/thai/quan.nguyen1/anaconda3/envs/ngq/bin/python}
nvidia-smi -L
$PY tests/check_core.py
$PY scripts/train.py "$@"
$PY scripts/figures.py
