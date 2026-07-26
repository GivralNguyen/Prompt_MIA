#!/bin/bash
#SBATCH --job-name=acc_impact_cifar100_N16
#SBATCH --output=output/acc_impact_cifar100_N16-%A.out
#SBATCH --error=output/acc_impact_cifar100_N16-%A.err
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

python train_acc_impact.py --inject_rate 0.8 --save_model --dataset cifar100