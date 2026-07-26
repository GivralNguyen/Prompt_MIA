#!/bin/bash
#SBATCH --job-name=cifar10_defense_pertube_003
#SBATCH --output=output/cifar10_defense_pertube_003-%A.out
#SBATCH --error=output/cifar10_defense_pertube_003-%A.err
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

 python train.py --save_model --defense perturb --perturb_sigma 0.03