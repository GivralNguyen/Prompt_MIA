#!/bin/bash
#SBATCH --job-name=cifar10_80clietns_cross_devices_modelvit_cls_non_iid_dirichlet_alpha0.05
#SBATCH --output=output/cifar10_80clietns_cross_devices_modelvit_cls_non_iid_dirichlet_alpha0.05-%A.out
#SBATCH --error=output/cifar10_80clietns_cross_devices_modelvit_cls_non_iid_dirichlet_alpha0.05-%A.err
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

python train.py --alpha 0.05