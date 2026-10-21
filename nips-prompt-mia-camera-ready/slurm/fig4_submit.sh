#!/bin/bash
# checks -> 12 (dataset, backbone) runs, 5 GPUs at a time -> plot. Run from nips-prompt-mia-camera-ready/.
set -euo pipefail
PY=${PY:-/blue/thai/quan.nguyen1/anaconda3/envs/ngq/bin/python}
COMMON="--partition=hpg-b200 --gres=gpu:1 --cpus-per-task=8 --mem=48G --exclude=c1009a-s25"
mkdir -p logs
CHECK=$(sbatch --parsable $COMMON --job-name=pmia_fig4_check --output=logs/fig4_check-%j.out --time=01:00:00 \
        --wrap "export HF_HUB_OFFLINE=1; $PY tests/check_core.py && $PY tests/check_fig4.py")
RUNS=$(sbatch --parsable --dependency=afterok:$CHECK slurm/fig4.sbatch)
PLOT=$(sbatch --parsable --dependency=afterok:$RUNS --partition=hpg-default --cpus-per-task=2 --mem=8G --time=00:20:00 \
       --job-name=pmia_fig4_plot --output=logs/fig4_plot-%j.out --wrap "$PY scripts/fig4_plot.py")
echo "check $CHECK -> runs $RUNS -> plot $PLOT"
