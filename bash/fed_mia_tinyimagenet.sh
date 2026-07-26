#!/bin/bash
#SBATCH --job-name=fed_mia_tinyimagenet
#SBATCH --output=output/fed_mia_tinyimagenet-%A.out
#SBATCH --error=output/fed_mia_tinyimagenet-%A.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL
#SBATCH --mail-type=REQUEUE

#SBATCH --mail-user=cobainnguyen@gmail.com

python mia_loss_based.py \
    --dataset tinyimagenet --comms 60 --n_clients 80 --n_sampled_clients 10 \
    --data_distribution non_iid_dirichlet --alpha 0.5 \
    --target_client 0 --mia_record_interval 1