#!/bin/bash
#SBATCH --job-name=cifar100_defense_sparse_08
#SBATCH --output=output/cifar100_defense_sparse_08-%A.out
#SBATCH --error=output/cifar100_defense_sparse_08-%A.err
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

 python train.py --dataset cifar100 --comms 120 --save_model --defense sparse --sparse_ratio 0.8