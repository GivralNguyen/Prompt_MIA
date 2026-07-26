"""
mia_loss_based.py – Loss-based Membership Inference Attacks for PromptMIA

Implements three attacks from the FedMIA paper adapted to prompt-based FL:
  - Blackbox-Loss : Single-round loss threshold attack (Yeom et al., 2018)
  - Loss-Series   : Multi-round averaged loss attack  (Gu et al., 2022)
  - FedMIA-I      : Multi-round, multi-client CDF-based attack using data loss

Usage:
  python mia_loss_based.py --dataset cifar10 --comms 60 --n_clients 80 \
      --n_sampled_clients 10 --data_distribution non_iid_dirichlet --alpha 0.5 \
      --target_client 0 --mia_record_interval 1
"""

import os
import json
import argparse
import random
import copy
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import torchvision
from tqdm._tqdm import trange
import scipy.stats
from sklearn import metrics
import pytz
from datetime import datetime

from Algo.get_algos import get_algorithm
from data_util.DataTransformBuilder import build_data_transform
from data_util.DataDistributer import DataPartitioner
from data_util.TinyImageNet_reader import TinyImageNet_reader
from data_util.PACS_reader import Pacs_reader
from data_util.Four_dataset_reader import four_dataset_reader
from Scenario import FL_scenario
from util.TOdevice import to_device
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_DeiT_B16, L2P_ConViT_Base
from util.saving_tools import save_eval_npy_file, save_model_trainable_part, load_model_trainable_part


# ===========================================================================
# Helper functions
# ===========================================================================

def apply_client_data_sampling(partitioner, sampling_proportion_data, seed=0):
    sampling_proportion_data = float(max(0.0, min(1.0, sampling_proportion_data)))
    if sampling_proportion_data >= 1.0:
        return
    rng = np.random.default_rng(seed)
    sampled_client_idcs = []
    for client_idcs in partitioner.client_idcs:
        client_idcs = np.array(client_idcs)
        if client_idcs.size == 0:
            sampled_client_idcs.append(client_idcs)
            continue
        keep_num = max(1, int(client_idcs.size * sampling_proportion_data))
        sampled_idcs = rng.choice(client_idcs, size=keep_num, replace=False)
        sampled_client_idcs.append(np.array(sampled_idcs, dtype=client_idcs.dtype))
    partitioner.client_idcs = sampled_client_idcs


@torch.no_grad()
def compute_per_sample_loss(model, dataloader, device):
    """Compute per-sample cross-entropy loss. Returns numpy array of shape (N,)."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='none')
    losses = []
    for data, target in dataloader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        logits = output[1] if isinstance(output, tuple) else output
        losses.append(ce(logits, target).cpu())
    return torch.cat(losses).numpy()


def compute_auc_and_tpr(member_scores, nonmember_scores):
    """
    Compute ROC-AUC and TPR at various FPR thresholds.
    Convention: higher score = more likely to be a member.
    """
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores))
    ])
    scores = np.concatenate([member_scores, nonmember_scores])
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc_val = metrics.auc(fpr, tpr)
    tprs = {}
    # Add more FPR thresholds as requested
    fpr_thresholds = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for fpr_thresh in fpr_thresholds:
        idx = np.sum(fpr < fpr_thresh)
        tprs[str(fpr_thresh)] = float(tpr[idx - 1]) if idx > 0 else 0.0
    # FPR at TPR=1: minimum FPR where all members are detected
    tpr1_indices = np.where(tpr >= 1.0)[0]
    tprs['fpr@tpr=1'] = float(fpr[tpr1_indices[0]]) if len(tpr1_indices) > 0 else 1.0
    return auc_val, tprs


def print_attack_result(name, auc_val, tprs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  AUC:              {auc_val:.4f}")
    # Print TPRs at specific FPRs in sorted order for clarity
    ordered_fprs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.001, 0.0001]
    for fpr in ordered_fprs:
        k = str(fpr)
        if k in tprs:
            print(f"  TPR@FPR={float(k):6.3f}:  {tprs[k]:.4f}")
    # Print FPR@TPR=1 if present
    if 'fpr@tpr=1' in tprs:
        print(f"  FPR@TPR= 1.000:  {tprs['fpr@tpr=1']:.4f}")
    print(f"{'='*60}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Loss-based MIA attacks (Blackbox-Loss, Loss-Series, FedMIA-I)')

    # ---- FL settings (same as mia_stealth.py) ----
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--alg', type=str, default='fedavg')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'PACS', 'tinyimagenet', 'fourdataset'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--comms', type=int, default=60)
    parser.add_argument('--local_eps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_clients', type=int, default=80)
    parser.add_argument('--n_sampled_clients', type=int, default=10)
    parser.add_argument('--data_distribution', type=str, default='non_iid_dirichlet',
                        choices=['non_iid_dirichlet', 'manual_extreme_heterogeneity'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--n_dominated_class', type=int, default=1)
    parser.add_argument('--dominated_ratio', type=float, default=0.99)

    # ---- Model settings ----
    parser.add_argument('--model_type', type=str, default='L2P')
    parser.add_argument('--prompt_method', type=str, default='shallow')
    parser.add_argument('--n_tokens', type=int, default=4)
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--batchwise_prompt', type=bool, default=False)
    parser.add_argument('--reduce_sim_scalar', type=float, default=0.5)
    parser.add_argument('--query_model_type', type=str, default='vit_cls',
                        choices=['vit_cls', 'deit_b16', 'convit_base'])
    parser.add_argument('--instance_label', type=int, default=1)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_model', action='store_true')

    # ---- DP settings ----
    parser.add_argument('--dp_mechanism', type=str, default='no_dp')
    parser.add_argument('--eps', type=float, default=1.0)

    # ---- Defense settings ----
    parser.add_argument('--defense', type=str, default='none',
                        choices=['none', 'perturb', 'sparse', 'mix_up'])
    parser.add_argument('--perturb_sigma', type=float, default=0.0)
    parser.add_argument('--sparse_ratio', type=float, default=0.0)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--data_augment', action='store_true')
    parser.add_argument('--sampling_proportion', type=float, default=1.0)
    parser.add_argument('--client_sampling_type', type=str, default='uniform',
                        choices=['uniform', 'poisson'])
    parser.add_argument('--sampling_proportion_data', type=float, default=1.0)

    # ---- Loss-based MIA settings ----
    parser.add_argument('--target_client', type=int, default=0,
                        help='Global client index whose membership to attack')
    parser.add_argument('--mia_record_interval', type=int, default=1,
                        help='Record losses every N communication rounds')
    parser.add_argument('--mia_save_dir', type=str, default='./output_record/mia_loss_based',
                        help='Directory to save MIA results')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    def _fmt(v):
        return str(v).replace('.', 'p')

    defense_tag = (
        f"def{args.defense}"
        f"_ptb{_fmt(args.perturb_sigma)}"
        f"_spr{_fmt(args.sparse_ratio)}"
        f"_aug{int(args.data_augment)}"
        f"_sprop{_fmt(args.sampling_proportion)}"
        f"_dsprop{_fmt(args.sampling_proportion_data)}"
    )

    if args.n_clients > args.n_sampled_clients:
        scenrio_type = 'cross_devices'
    else:
        scenrio_type = 'cross_silo'

    # ---- Transforms ----
    norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = 256
    centercrop_size = 224
    train_preprocess = build_data_transform(norm_stats, resize, centercrop_size,
                                            train_augment=args.data_augment)
    test_preprocess = build_data_transform(norm_stats, resize, centercrop_size,
                                           train_augment=False)

    # ---- Datasets ----
    class_mask = None
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=True, download=True, transform=train_preprocess)
        testset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=False, download=True, transform=test_preprocess)
        trainset_eval = torchvision.datasets.CIFAR10(
            root='./Dataset', train=True, download=False, transform=test_preprocess)
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=True, download=True, transform=train_preprocess)
        testset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=False, download=True, transform=test_preprocess)
        trainset_eval = torchvision.datasets.CIFAR100(
            root='./Dataset', train=True, download=False, transform=test_preprocess)
        num_classes = 100
    elif args.dataset == 'PACS':
        data_name = ['art_painting', 'cartoon', 'photo', 'sketch']
        trainset = [Pacs_reader('./Dataset/PACS', split=d, train=True,
                                transform=train_preprocess, random_seed=i)
                    for i, d in enumerate(data_name)]
        testset = [Pacs_reader('./Dataset/PACS', split=d, train=False,
                               transform=test_preprocess, random_seed=i)
                   for i, d in enumerate(data_name)]
        testset = Data.ConcatDataset(testset)
        trainset_eval = [Pacs_reader('./Dataset/PACS', split=d, train=True,
                                     transform=test_preprocess, random_seed=i)
                         for i, d in enumerate(data_name)]
        num_classes = 28
    elif args.dataset == 'tinyimagenet':
        trainset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=True,
                                       transform=train_preprocess)
        testset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=False,
                                      transform=test_preprocess)
        trainset_eval = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=True,
                                            transform=test_preprocess)
        num_classes = 200
    elif args.dataset == 'fourdataset':
        data_name = ['mnistm', 'fashion', 'cinic10', 'mmafedb']
        trainset, class_mask = four_dataset_reader([30000]*4, train=True,
                                                   transform=train_preprocess)
        testset, _ = four_dataset_reader([2500]*4, train=False, transform=test_preprocess)
        trainset_eval, _ = four_dataset_reader([30000]*4, train=True,
                                               transform=test_preprocess)
        num_classes = 37
    else:
        raise ValueError("Input dataset is not supported")

    print(class_mask)

    # ---- Test loader ----
    if args.dataset == 'fourdataset':
        testset_concat = Data.ConcatDataset(testset)
        testloader = [Data.DataLoader(testset[i], batch_size=args.batch,
                                      num_workers=2, shuffle=False)
                      for i in range(len(testset))]
    else:
        testset_concat = testset
        testloader = Data.DataLoader(testset, batch_size=args.batch,
                                     num_workers=2, shuffle=False)

    # ---- Partitioner ----
    if args.dataset == 'PACS' or args.dataset == 'fourdataset':
        data_partitioner = [DataPartitioner(trainset[i], args.n_clients // len(data_name))
                            for i in range(len(trainset))]
    else:
        data_partitioner = DataPartitioner(trainset, args.n_clients)

    # Seed selection (same as mia_stealth.py)
    seed_map = {0.01: 688, 0.05: 805, 0.1: 871, 0.2: 459, 0.3: 429, 0.4: 3760, 0.5: 448}
    seed = seed_map.get(args.alpha, 448)

    LATz = pytz.timezone("America/Los_Angeles")
    timeInLA = datetime.now(LATz)

    if args.data_distribution == 'non_iid_dirichlet':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                s = seed if i == 0 else data_partitioner[0].seed
                data_partitioner[i].dirichlet_split_noniid(alpha=args.alpha,
                                                           least_samples=32, manual_seed=s)
            for i in range(len(data_partitioner)):
                apply_client_data_sampling(data_partitioner[i],
                                           args.sampling_proportion_data, seed + i)
        else:
            data_partitioner.dirichlet_split_noniid(alpha=args.alpha,
                                                    least_samples=32, manual_seed=seed)
            apply_client_data_sampling(data_partitioner,
                                       args.sampling_proportion_data, seed)
    elif args.data_distribution == 'manual_extreme_heterogeneity':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                data_partitioner[i].manual_allocating_noniid(
                    args.n_dominated_class, args.dominated_ratio, 1.0)
            for i in range(len(data_partitioner)):
                apply_client_data_sampling(data_partitioner[i],
                                           args.sampling_proportion_data, seed + i)
        else:
            data_partitioner.manual_allocating_noniid(
                args.n_dominated_class, args.dominated_ratio, 1.0)
            apply_client_data_sampling(data_partitioner,
                                       args.sampling_proportion_data, seed)
    else:
        raise ValueError("Input data distribution is not supported")

    # ---- MIA eval loaders (member = target client's data, non-member = test set) ----
    if isinstance(data_partitioner, list):
        # For PACS/fourdataset: target_client indexes within the first sub-partitioner
        target_indices = data_partitioner[0].client_idcs[args.target_client]
        member_eval_dataset = Data.Subset(trainset_eval[0],
                                          torch.from_numpy(target_indices))
    else:
        target_indices = data_partitioner.client_idcs[args.target_client]
        member_eval_dataset = Data.Subset(trainset_eval,
                                          torch.from_numpy(target_indices))

    member_eval_loader = Data.DataLoader(member_eval_dataset, batch_size=args.batch,
                                         shuffle=False, num_workers=2)

    nonmember_dataset = testset_concat if args.dataset == 'fourdataset' else testset
    nonmember_eval_loader = Data.DataLoader(nonmember_dataset, batch_size=args.batch,
                                            shuffle=False, num_workers=2)

    n_members = len(member_eval_dataset)
    n_nonmembers = len(nonmember_dataset)
    print(f"\n[MIA] Target client: {args.target_client}  "
          f"Members: {n_members}  Non-members: {n_nonmembers}\n")

    # ---- FL scenario and model ----
    eval_loss_record = []
    eval_acc_record = []
    eval_pool_size = None
    eval_pool = None
    client_pool = None

    if args.alg in ['fedavg', 'fedavg_nonpara']:
        if args.dataset == 'PACS' or args.dataset == 'fourdataset':
            all_clients_weights = np.concatenate(
                [dp.get_all_client_weights() for dp in data_partitioner])
            distributed_dataloaders = [dp.get_distributed_data(batch_size=args.batch)
                                       for dp in data_partitioner]
            distributed_dataloaders = [dl for sub in distributed_dataloaders for dl in sub]
            fl_scen = FL_scenario(all_clients_weights, args.n_clients,
                                  args.n_sampled_clients, distributed_dataloaders)
        else:
            fl_scen = FL_scenario(data_partitioner.get_all_client_weights(),
                                  args.n_clients, args.n_sampled_clients,
                                  data_partitioner.get_distributed_data(batch_size=args.batch))

        print(args.model_type)
        if args.model_type == 'L2P':
            if args.query_model_type == 'vit_cls':
                server_model = to_device(L2P_ViT_B32(
                    prompt_method=args.prompt_method,
                    batchwise_prompt=args.batchwise_prompt,
                    pool_size=args.pool_size,
                    top_k=args.n_tokens,
                    num_classes=num_classes,
                    candidate_key=None), args.device)
            elif args.query_model_type == 'deit_b16':
                server_model = to_device(L2P_DeiT_B16(
                    prompt_method=args.prompt_method,
                    batchwise_prompt=args.batchwise_prompt,
                    pool_size=args.pool_size,
                    top_k=args.n_tokens,
                    num_classes=num_classes,
                    candidate_key=None, pretrained=True), args.device)
            elif args.query_model_type == 'convit_base':
                server_model = to_device(L2P_ConViT_Base(
                    prompt_method=args.prompt_method,
                    batchwise_prompt=args.batchwise_prompt,
                    pool_size=args.pool_size,
                    top_k=args.n_tokens,
                    num_classes=num_classes,
                    candidate_key=None, pretrained=True), args.device)
            else:
                raise ValueError("Model type is not supported")
            eval_pool_size = [args.pool_size]
            eval_pool = [server_model.pool.prompt.data.detach().cpu().numpy()]
            client_pool = []
        else:
            raise ValueError("Model type is not supported")

        server_model.build_trainable_keys()

        if '_' in args.alg:
            associate_algo = args.alg.split('_')[0]
            algclass = get_algorithm(associate_algo)
        else:
            algclass = get_algorithm(args.alg)

        if args.alg == 'fedavg':
            print('normal aggregation')
            algo = algclass(server_model=server_model, scenario=fl_scen,
                            loss_fun=nn.CrossEntropyLoss(), fed_method='aggregation',
                            class_mask=class_mask, device=args.device)
        else:
            raise ValueError("Algorithm is not supported")

        # ============================================================
        # MIA loss storage
        # ============================================================
        # For Blackbox-Loss and Loss-Series (global model after aggregation)
        global_member_losses = []       # list of (n_members,) arrays
        global_nonmember_losses = []    # list of (n_nonmembers,) arrays
        recorded_rounds = []

        # For FedMIA-I (local client models before aggregation)
        fedmia_target_member_losses = []     # (n_members,) per round
        fedmia_target_nonmember_losses = []  # (n_nonmembers,) per round
        fedmia_shadow_member_losses = []     # (n_shadow, n_members) per round
        fedmia_shadow_nonmember_losses = []  # (n_shadow, n_nonmembers) per round
        fedmia_rounds = []

        # ============================================================
        # Training loop
        # ============================================================
        if args.load_model is not None:
            if args.load_model == "random":
                print("Randomly initialized model loaded\n")
            else:
                load_model_trainable_part(algo.server_model, args.load_model)
                print(f"Model loaded from {args.load_model}\n")

            # For pre-trained model: only Blackbox-Loss is applicable
            print("[MIA] Computing Blackbox-Loss on loaded model...")
            gmem = compute_per_sample_loss(algo.server_model, member_eval_loader, args.device)
            gnonmem = compute_per_sample_loss(algo.server_model, nonmember_eval_loader, args.device)
            global_member_losses.append(gmem)
            global_nonmember_losses.append(gnonmem)
            recorded_rounds.append(0)
        else:
            for comm_round in trange(args.comms,
                                     desc="FL Training + MIA Recording"):
                # ---- Local training ----
                algo.client_train(
                    comm_round=comm_round, epochs=args.local_eps, lr=args.lr,
                    dp_mechanism=args.dp_mechanism, eps=args.eps,
                    frac=args.n_sampled_clients / args.n_clients,
                    reduce_sim_scalar=args.reduce_sim_scalar, print_output=True,
                    defense=args.defense, mixup_alpha=args.mixup_alpha,
                    sparse_ratio=args.sparse_ratio, perturb_sigma=args.perturb_sigma,
                    sampling_proportion=args.sampling_proportion,
                    client_sampling_type=args.client_sampling_type)

                # Record client prompts
                if args.model_type == 'L2P':
                    for i in range(len(algo.selected_client_index)):
                        client_pool.append(
                            algo.client_model[i].pool.prompt.data.detach().cpu().numpy())

                should_record = ((comm_round + 1) % args.mia_record_interval == 0
                                 or comm_round == args.comms - 1)

                # ---- FedMIA-I: record local model losses BEFORE aggregation ----
                if should_record:
                    selected = algo.selected_client_index
                    target_local = None
                    if args.target_client in selected:
                        target_local = int(
                            np.where(selected == args.target_client)[0][0])

                    if target_local is not None:
                        # Target model losses
                        tmem = compute_per_sample_loss(
                            algo.client_model[target_local],
                            member_eval_loader, args.device)
                        tnonmem = compute_per_sample_loss(
                            algo.client_model[target_local],
                            nonmember_eval_loader, args.device)
                        fedmia_target_member_losses.append(tmem)
                        fedmia_target_nonmember_losses.append(tnonmem)

                        # Shadow model losses (all other sampled clients)
                        shadow_mem_list = []
                        shadow_nonmem_list = []
                        for i in range(len(selected)):
                            if i == target_local:
                                continue
                            smem = compute_per_sample_loss(
                                algo.client_model[i],
                                member_eval_loader, args.device)
                            snonmem = compute_per_sample_loss(
                                algo.client_model[i],
                                nonmember_eval_loader, args.device)
                            shadow_mem_list.append(smem)
                            shadow_nonmem_list.append(snonmem)

                        fedmia_shadow_member_losses.append(
                            np.vstack(shadow_mem_list))
                        fedmia_shadow_nonmember_losses.append(
                            np.vstack(shadow_nonmem_list))
                        fedmia_rounds.append(comm_round)

                        print(f"  [MIA] Round {comm_round+1}: target selected "
                              f"(local_idx={target_local}), "
                              f"{len(shadow_mem_list)} shadow clients recorded")
                    else:
                        print(f"  [MIA] Round {comm_round+1}: target NOT selected, "
                              f"skipping FedMIA-I for this round")

                # ---- Server aggregation ----
                algo.server_aggre()

                # ---- Global model losses AFTER aggregation ----
                if should_record:
                    gmem = compute_per_sample_loss(
                        algo.server_model, member_eval_loader, args.device)
                    gnonmem = compute_per_sample_loss(
                        algo.server_model, nonmember_eval_loader, args.device)
                    global_member_losses.append(gmem)
                    global_nonmember_losses.append(gnonmem)
                    recorded_rounds.append(comm_round)

                # ---- Regular evaluation ----
                if args.model_type == 'L2P':
                    eval_loss, eval_acc, current_pool_size = algo.server_eval(
                        testloader, comm_round)
                    eval_pool_size.append(current_pool_size)
                    eval_pool.append(
                        algo.server_model.pool.prompt.data.detach().cpu().numpy())
                else:
                    eval_loss, eval_acc = algo.server_eval(testloader, comm_round)
                eval_loss_record.append(eval_loss)
                eval_acc_record.append(eval_acc)

            # Save model if requested
            if args.save_model:
                best_model_path = (
                    f"./output_record/saved_model/saved_model_{args.model_type}_"
                    f"{args.alg}_{args.dataset}_{args.n_clients}clietns_"
                    f"{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_"
                    f"{args.instance_label}_{defense_tag}_round_{args.comms}_"
                    f"acc_{eval_acc:.4f}.pt"
                )
                save_model_trainable_part(algo.server_model, best_model_path)
                print(f"\n Model saved to {best_model_path}\n")
    else:
        raise ValueError("Algorithm is not supported")

    # ====================================================================
    #  ATTACK PHASE
    # ====================================================================

    results = {}
    n_recorded = len(global_member_losses)
    print(f"\n{'#'*60}")
    print(f"  LOSS-BASED MIA ATTACKS  ({n_recorded} recorded rounds)")
    print(f"  FedMIA-I available rounds: {len(fedmia_target_member_losses)}")
    print(f"{'#'*60}")

    # ------------------------------------------------------------------
    # 1. Blackbox-Loss: last recorded round, score = -CE_loss
    # ------------------------------------------------------------------
    bb_member = -global_member_losses[-1]
    bb_nonmember = -global_nonmember_losses[-1]
    bb_auc, bb_tprs = compute_auc_and_tpr(bb_member, bb_nonmember)
    print_attack_result("Blackbox-Loss (single round)", bb_auc, bb_tprs)
    results['blackbox_loss'] = {'auc': bb_auc, 'tprs': bb_tprs,
                                'round': int(recorded_rounds[-1])}

    # Also compute Blackbox-Loss per recorded round for analysis
    bb_per_round = []
    for r_idx, r in enumerate(recorded_rounds):
        mem_s = -global_member_losses[r_idx]
        nonmem_s = -global_nonmember_losses[r_idx]
        a, t = compute_auc_and_tpr(mem_s, nonmem_s)
        bb_per_round.append({'round': int(r), 'auc': a, 'tpr@0.001': t['0.001']})
    results['blackbox_loss_per_round'] = bb_per_round

    # ------------------------------------------------------------------
    # 2. Loss-Series: average -CE_loss across all recorded rounds
    # ------------------------------------------------------------------
    if n_recorded >= 2:
        ls_member = -np.vstack(global_member_losses).mean(axis=0)
        ls_nonmember = -np.vstack(global_nonmember_losses).mean(axis=0)
        ls_auc, ls_tprs = compute_auc_and_tpr(ls_member, ls_nonmember)
        print_attack_result("Loss-Series (multi-round average)", ls_auc, ls_tprs)
        results['loss_series'] = {'auc': ls_auc, 'tprs': ls_tprs,
                                  'n_rounds': n_recorded}
    else:
        print("\n[Loss-Series] Skipped: need >= 2 recorded rounds")

    # ------------------------------------------------------------------
    # 3. FedMIA-I: CDF from shadow clients, averaged across rounds
    # ------------------------------------------------------------------
    n_fedmia_rounds = len(fedmia_target_member_losses)
    if n_fedmia_rounds >= 1:
        cdf_member_per_round = []
        cdf_nonmember_per_round = []

        for r in range(n_fedmia_rounds):
            # Target model losses (negated: higher = lower CE = more member-like)
            target_mem = -fedmia_target_member_losses[r]
            target_nonmem = -fedmia_target_nonmember_losses[r]

            # Shadow model losses (negated)
            shadow_mem = -fedmia_shadow_member_losses[r]    # (n_shadow, n_members)
            shadow_nonmem = -fedmia_shadow_nonmember_losses[r]  # (n_shadow, n_nonmem)

            # Fit Gaussian per sample from shadow models
            mem_mu = shadow_mem.mean(axis=0)
            mem_var = shadow_mem.var(axis=0) + 1e-8
            nonmem_mu = shadow_nonmem.mean(axis=0)
            nonmem_var = shadow_nonmem.var(axis=0) + 1e-8

            # CDF: P(shadow_score <= target_score)
            # For members: target trained on this data → lower CE → higher -CE
            #   → higher CDF (target is in the upper tail of shadow distribution)
            cdf_mem = scipy.stats.norm.cdf(
                target_mem, mem_mu, np.sqrt(mem_var))
            cdf_nonmem = scipy.stats.norm.cdf(
                target_nonmem, nonmem_mu, np.sqrt(nonmem_var))

            cdf_member_per_round.append(cdf_mem)
            cdf_nonmember_per_round.append(cdf_nonmem)

        # Average CDF scores across all rounds (temporal aggregation)
        fi_member = np.vstack(cdf_member_per_round).mean(axis=0)
        fi_nonmember = np.vstack(cdf_nonmember_per_round).mean(axis=0)
        fi_auc, fi_tprs = compute_auc_and_tpr(fi_member, fi_nonmember)
        print_attack_result(
            f"FedMIA-I (CDF + temporal avg, {n_fedmia_rounds} rounds)",
            fi_auc, fi_tprs)
        results['fedmia_i'] = {'auc': fi_auc, 'tprs': fi_tprs,
                                'n_rounds': n_fedmia_rounds}

        # Also report single best round for FedMIA-I
        best_fi_auc = 0
        best_fi_tprs = None
        best_fi_round = -1
        for r in range(n_fedmia_rounds):
            a, t = compute_auc_and_tpr(
                cdf_member_per_round[r], cdf_nonmember_per_round[r])
            if a > best_fi_auc:
                best_fi_auc = a
                best_fi_tprs = t
                best_fi_round = fedmia_rounds[r]
        if best_fi_tprs is not None:
            print_attack_result(
                f"FedMIA-I best single round (round {best_fi_round+1})",
                best_fi_auc, best_fi_tprs)
            results['fedmia_i_best_round'] = {
                'auc': best_fi_auc, 'tprs': best_fi_tprs,
                'round': int(best_fi_round)}
    else:
        print("\n[FedMIA-I] Skipped: target client was never selected during "
              "recorded rounds. Try increasing --comms or decreasing "
              "--mia_record_interval.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(args.mia_save_dir, exist_ok=True)
    result_path = os.path.join(
        args.mia_save_dir,
        f"mia_results_{args.dataset}_{args.n_clients}clients_"
        f"alpha{args.alpha}_{defense_tag}_target{args.target_client}.json"
    )
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    # Expanded summary table with all requested TPR@FPRs
    summary_fprs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.001, 0.0001]
    summary_headers = [f"TPR@{fpr:.3f}" for fpr in summary_fprs] + ["FPR@TPR=1"]
    header_line = f"  {'Attack':<35s} {'AUC':>8s}  " + "  ".join([f"{h:>9s}" for h in summary_headers])
    sep_line = f"  {'-'*35} {'-'*8}  " + "  ".join([f"{'-'*9}" for _ in summary_headers])
    print(f"\n{'='*160}")
    print(header_line)
    print(sep_line)
    def print_row(name, r):
        row = f"  {name:<35s} {r['auc']:>8.4f}  "
        for fpr in summary_fprs:
            val = r['tprs'].get(str(fpr), 0.0)
            row += f"{val:>9.4f}  "
        row += f"{r['tprs'].get('fpr@tpr=1', 0.0):>9.4f}"
        print(row)
    if 'blackbox_loss' in results:
        print_row('Blackbox-Loss', results['blackbox_loss'])
    if 'loss_series' in results:
        print_row('Loss-Series', results['loss_series'])
    if 'fedmia_i' in results:
        print_row('FedMIA-I', results['fedmia_i'])
    if 'fedmia_i_best_round' in results:
        print_row('FedMIA-I (best single round)', results['fedmia_i_best_round'])
    print(f"{'='*160}")
