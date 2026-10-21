#!/bin/bash
#SBATCH --job-name=pmia_keylr
#SBATCH --output=logs/sweep_key_lr-%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
# Key learning rate sweep (+ one lambda run). Baseline key_lr = lr = 1e-4, lam = 0.5 is runs/keylr_1e-4.
# Every config sets --key_lr explicitly, so the sweep does not depend on train.py's default (1e-3).
set -euo pipefail
PY=${PY:-/blue/thai/quan.nguyen1/anaconda3/envs/ngq/bin/python}
CONFIGS=(
  "--name keylr_3e-4 --key_lr 3e-4"
  "--name keylr_1e-3 --key_lr 1e-3"
  "--name keylr_3e-3 --key_lr 3e-3"
  "--name keylr_1e-2 --key_lr 1e-2"
  "--name lam_5 --lam 5 --key_lr 1e-4"
)
ARGS=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=$(echo $ARGS | awk '{print $2}')
$PY scripts/train.py $ARGS
$PY scripts/figures.py --run runs/$NAME
