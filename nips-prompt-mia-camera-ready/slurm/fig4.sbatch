#!/bin/bash
#SBATCH --job-name=pmia_fig4
#SBATCH --output=logs/fig4-%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-11%5
#SBATCH --exclude=c1009a-s25
# One (dataset, backbone) per task. Submit everything with slurm/fig4_submit.sh; extra args go to fig4_run.py.
set -euo pipefail
export HF_HUB_OFFLINE=1
PY=${PY:-/blue/thai/quan.nguyen1/anaconda3/envs/ngq/bin/python}
DATASETS=(cifar10 cifar100 tinyimagenet fourdataset)
BACKBONES=(vit_b32 deit_b16 convit_base)
D=${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}
B=${BACKBONES[$((SLURM_ARRAY_TASK_ID % 3))]}
$PY scripts/fig4_run.py --dataset $D --backbone $B "$@"
