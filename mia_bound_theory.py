"""
mia_bound2.py — Compute the FPR bound from Theorem 3.3 of PromptMIA.

Theorem 3.3 (False Positive Rate):
    FPR <= max_{1<=i<=K} ( min_{1<=j<=N, 1<=l<=K} Phi(z_{ijl}) )

    where K = M - N (number of benign keys),
          z_{ijl} = (k_{a_j} - k_{b_l})^T k_{b_i} / (sigma_i * ||k_{a_j} - k_{b_l}||)
          sigma_i = std of non-member queries assigned to benign cluster i,
          Phi is the standard normal CDF.

This script:
    1. Loads a trained L2P model with its benign key pool.
    2. Extracts CLS query features from the real dataset.
    3. For each target sample, generates adversarial keys using the same
       procedure as mia_stealth.py (GenAdvKeySet).
    4. Assigns each query to its nearest benign key cluster to estimate sigma_i.
    5. Computes z_{ijl} and the FPR bound from Theorem 3.3.

Usage:
    python mia_bound_theory.py --gpus 0 --dataset cifar10 --query_model_type vit_cls --pool_size 20 --n_tokens 4 --delta_min 0.02 --delta_max 0.05 --n_targets 100 --max_samples 50000 --load_model /home/quannm/code/nips-promptmia/output_record/saved_model/saved_model_L2P_fedavg_cifar10_80clietns_cross_devices_modelvit_cls_non_iid_dirichlet_alpha0.5_1_round_61_acc_0.9442.pt
    REMEMBER TO CHANGE POOL_SIZE TO MATCH THE TRAINED MODEL CONFIG!
"""

import os
import argparse
import numpy as np
import torch
import torch.utils.data as Data
import torchvision
from scipy.stats import norm as scipy_norm

from data_util.DataTransformBuilder import build_data_transform
from data_util.TinyImageNet_reader import TinyImageNet_reader
from data_util.Four_dataset_reader import four_dataset_reader
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_DeiT_B16, L2P_ConViT_Base
from util.TOdevice import to_device
from util.saving_tools import load_model_trainable_part


def l2_normalize(x, dim=1, eps=1e-12):
    """L2-normalize tensor along given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def generate_adversarial_key(target_query_np, similarity, benign_key_np, beta=0.0):
    """
    Generate a single adversarial key with a specified cosine similarity to the
    target query, matching the procedure in mia_stealth.py / Algorithm 1.

    Parameters
    ----------
    target_query_np : ndarray, shape (D,) — q(T), the target query vector.
    similarity      : float — desired cosine similarity s in (0, 1).
    benign_key_np   : ndarray, shape (D,) — a benign key (used when beta > 0).
    beta            : float — mixing factor between random and benign direction.

    Returns
    -------
    ndarray, shape (D,) — the adversarial key k_a.
    """
    target_vec = target_query_np.reshape(-1)
    benign_key = benign_key_np.reshape(-1)

    target_norm = target_vec / np.linalg.norm(target_vec)
    benign_norm = benign_key / np.linalg.norm(benign_key)

    rand_vec = np.random.rand(target_vec.shape[0])
    rand_vec = rand_vec / np.linalg.norm(rand_vec)

    final_vec = beta * benign_norm + (1 - beta) * rand_vec
    final_vec /= np.linalg.norm(final_vec)

    orthogonal_vec = final_vec - np.dot(final_vec, target_norm) * target_norm
    orthogonal_norm = orthogonal_vec / (np.linalg.norm(orthogonal_vec) + 1e-8)

    new_vec_norm = similarity * target_norm + np.sqrt(1 - similarity ** 2) * orthogonal_norm
    new_vec_norm /= np.linalg.norm(new_vec_norm)

    new_vec = new_vec_norm * np.linalg.norm(target_vec)
    return new_vec


def generate_adversarial_key_set(target_query_np, benign_keys_np, N,
                                  delta_min, delta_max, beta=0.0):
    """
    Generate a set of N adversarial keys following the GenAdvKeySet algorithm,
    matching mia_stealth.py logic.

    Parameters
    ----------
    target_query_np : ndarray, shape (D,)
    benign_keys_np  : ndarray, shape (K, D) — all benign keys (after L2 norm).
    N               : int — number of adversarial keys.
    delta_min       : float
    delta_max       : float — the delta_max arg (total range = delta_min + delta_max).
    beta            : float

    Returns
    -------
    adv_keys : ndarray, shape (N, D) — adversarial keys.
    chosen_idx : list[int] — indices of benign keys replaced.
    """
    target_norm = target_query_np / np.linalg.norm(target_query_np)

    # Cosine similarity between normalized target and all benign keys (already normalized)
    sims = benign_keys_np @ target_norm  # (K,)

    # Choose top-N keys (these will be replaced by adversarial keys)
    topk_idx = np.argsort(sims)[-N:][::-1].tolist()

    # Max benign similarity excluding chosen keys
    mask = np.ones(len(sims), dtype=bool)
    mask[topk_idx] = False
    max_benign_sim = sims[mask].max()

    s_start = min(max_benign_sim + delta_min, 1.0)
    s_end = min(max_benign_sim + delta_min + delta_max, 1.0)

    varied_sims = np.random.uniform(s_start, s_end, N)

    non_chosen = np.where(mask)[0]
    anchor_idx = np.random.choice(non_chosen, size=N, replace=False)

    adv_keys = []
    for s, a_idx in zip(varied_sims, anchor_idx):
        k_a = generate_adversarial_key(target_query_np, s, benign_keys_np[a_idx], beta)
        adv_keys.append(k_a)

    return np.array(adv_keys), topk_idx


def estimate_cluster_sigmas(query_features_norm, benign_keys_norm):
    """
    Assign each query to its nearest benign key and estimate per-cluster sigma_i.

    Under Assumption 2: q(x) ~ N(k_{b_i}, sigma_i^2 I), so
       sigma_i^2 = E[||q(x) - k_{b_i}||^2] / D
    for queries assigned to cluster i.

    Parameters
    ----------
    query_features_norm : ndarray, shape (n, D) — L2-normalized query vectors.
    benign_keys_norm    : ndarray, shape (K, D) — L2-normalized benign keys.

    Returns
    -------
    sigmas     : ndarray, shape (K,) — estimated sigma_i per cluster.
    counts     : ndarray, shape (K,) — number of queries per cluster.
    """
    K, D = benign_keys_norm.shape
    # Cosine similarity = dot product of normalized vectors
    sims = query_features_norm @ benign_keys_norm.T  # (n, K)
    assignments = np.argmax(sims, axis=1)  # (n,)

    sigmas = np.zeros(K)
    counts = np.zeros(K, dtype=int)

    for i in range(K):
        members = query_features_norm[assignments == i]
        counts[i] = len(members)
        if len(members) < 2:
            sigmas[i] = 1e-6  # fallback for empty/singleton clusters
        else:
            # ||q(x) - k_{b_i}||^2  for each member
            diffs = members - benign_keys_norm[i]
            sq_dists = np.sum(diffs ** 2, axis=1)
            sigmas[i] = np.sqrt(np.mean(sq_dists) / D)

    return sigmas, counts


def compute_fpr_bound_theorem33(benign_keys, adv_keys, sigmas):
    """
    Compute the FPR bound from Theorem 3.3.

    Per-sample FPR <= max_{i} ( min_{j,l} Phi(z_{ijl}) )

    z_{ijl} = (k_{a_j} - k_{b_l})^T k_{b_i} / (sigma_i * ||k_{a_j} - k_{b_l}||)

    Also computes per-key bounds for the batch FPR formula:
        p_j = max_i min_l Phi(z_{ijl})  (per adversarial key j)

    Parameters
    ----------
    benign_keys : ndarray, shape (K, D)
    adv_keys    : ndarray, shape (N, D)
    sigmas      : ndarray, shape (K,) — per-cluster sigma_i.

    Returns
    -------
    fpr_bound       : float — upper bound on the per-sample FPR.
    per_cluster_phi : ndarray, shape (K,) — per-cluster min_j,l Phi(z_{ijl}).
    per_key_phi     : ndarray, shape (N,) — per-key bound p_j = max_i min_l Phi(z_{ijl}).
    """
    K, D = benign_keys.shape
    N = adv_keys.shape[0]

    # z_phi[i, j, l] = Phi(z_{ijl})
    z_phi = np.zeros((K, N, K))

    for i in range(K):
        k_bi = benign_keys[i]
        sigma_i = sigmas[i]
        if sigma_i < 1e-4:
            # Empty/singleton cluster (fallback sigma) or extremely tight => skip
            continue
        for j in range(N):
            for l in range(K):
                diff = adv_keys[j] - benign_keys[l]
                diff_norm = np.linalg.norm(diff)
                if diff_norm < 1e-12:
                    continue
                z = np.dot(diff, k_bi) / (sigma_i * diff_norm)
                z_phi[i, j, l] = scipy_norm.cdf(z)

    # Per-cluster bound: min over (j, l), then max over i
    per_cluster_phi = np.min(z_phi, axis=(1, 2))  # shape (K,)
    fpr_bound = np.max(per_cluster_phi)

    # Per-key bound: for each j, min over l, then max over i
    per_key_per_cluster = np.min(z_phi, axis=2)    # shape (K, N) — min over l
    per_key_phi = np.max(per_key_per_cluster, axis=0)  # shape (N,) — max over i

    return fpr_bound, per_cluster_phi, per_key_phi


def extract_cls_features(model, dataset, device, max_samples=None, batch_size=64):
    """Extract CLS features from the model for a dataset."""
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats = []
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = to_device(x, device).float()
            cls = model.get_cls_keys(x)  # (B, D)
            feats.append(cls.cpu())
            count += cls.shape[0]
            if max_samples and count >= max_samples:
                break
    feats = torch.cat(feats, dim=0)
    if max_samples:
        feats = feats[:max_samples]
    return feats


def main():
    parser = argparse.ArgumentParser(
        description="Compute the FPR bound from Theorem 3.3 of PromptMIA."
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimagenet", "fourdataset"])
    parser.add_argument("--query_model_type", type=str, default="vit_cls",
                        choices=["vit_cls", "deit_b16", "convit_base"])
    parser.add_argument("--pool_size", type=int, default=20,
                        help="Total prompt pool size M")
    parser.add_argument("--n_tokens", type=int, default=4,
                        help="Number of adversarial keys N (top_k)")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to saved model (or 'random' for untrained)")
    parser.add_argument("--delta_min", type=float, default=0.02)
    parser.add_argument("--delta_max", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Mixing factor for adversarial key generation")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max samples for query feature extraction")
    parser.add_argument("--n_targets", type=int, default=100,
                        help="Number of target samples to evaluate FPR bound")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prompt_method", type=str, default="shallow")
    parser.add_argument("--batchwise_prompt", type=bool, default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = args.device

    # ---- Build data transforms & load dataset ----
    norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_preprocess = build_data_transform(norm_stats, 256, 224, train_augment=False)

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="./Dataset", train=True,
                                                download=True, transform=train_preprocess)
        testset = torchvision.datasets.CIFAR10(root="./Dataset", train=False,
                                               download=True, transform=train_preprocess)
        num_classes = 10
    elif args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root="./Dataset", train=True,
                                                 download=True, transform=train_preprocess)
        testset = torchvision.datasets.CIFAR100(root="./Dataset", train=False,
                                                download=True, transform=train_preprocess)
        num_classes = 100
    elif args.dataset == "tinyimagenet":
        trainset = TinyImageNet_reader("./Dataset/tiny-imagenet-200/", train=True,
                                       transform=train_preprocess)
        testset = TinyImageNet_reader("./Dataset/tiny-imagenet-200/", train=False,
                                      transform=train_preprocess)
        num_classes = 200
    elif args.dataset == "fourdataset":
        trainset_list, _ = four_dataset_reader([30000] * 4, train=True,
                                               transform=train_preprocess)
        testset_list, _ = four_dataset_reader([2500] * 4, train=False,
                                              transform=train_preprocess)
        trainset = Data.ConcatDataset(trainset_list)
        testset = Data.ConcatDataset(testset_list)
        num_classes = 37
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # ---- Build model ----
    if args.query_model_type == "vit_cls":
        model = L2P_ViT_B32(
            prompt_method=args.prompt_method,
            batchwise_prompt=args.batchwise_prompt,
            pool_size=args.pool_size, top_k=args.n_tokens,
            num_classes=num_classes, candidate_key=None,
        )
    elif args.query_model_type == "deit_b16":
        model = L2P_DeiT_B16(
            prompt_method=args.prompt_method,
            batchwise_prompt=args.batchwise_prompt,
            pool_size=args.pool_size, top_k=args.n_tokens,
            num_classes=num_classes, candidate_key=None, pretrained=True,
        )
    elif args.query_model_type == "convit_base":
        model = L2P_ConViT_Base(
            prompt_method=args.prompt_method,
            batchwise_prompt=args.batchwise_prompt,
            pool_size=args.pool_size, top_k=args.n_tokens,
            num_classes=num_classes, candidate_key=None, pretrained=True,
        )
    else:
        raise ValueError(f"Unsupported query model type: {args.query_model_type}")

    model = to_device(model, device)
    model.build_trainable_keys()
    if args.load_model and args.load_model != "random":
        load_model_trainable_part(model, args.load_model)
        print(f"Model loaded from {args.load_model}")
    else:
        print("Using randomly initialized model (no checkpoint loaded)")

    M = args.pool_size
    N = args.n_tokens
    K = M - N  # number of benign keys
    D = model.pool.key.shape[1]

    print(f"M={M}, N={N}, K={K}, D={D}")
    print(f"delta_min={args.delta_min}, delta_max={args.delta_max}, beta={args.beta}")

    # ---- Extract CLS features from train set (non-member queries for sigma estimation) ----
    print(f"Extracting CLS features (max_samples={args.max_samples})...")
    train_features = extract_cls_features(model, trainset, device, args.max_samples)
    test_features = extract_cls_features(model, testset, device, args.max_samples)
    print(f"  train features: {train_features.shape}, test features: {test_features.shape}")

    # L2-normalize features
    train_norm = l2_normalize(train_features).numpy()  # (n_train, D)
    test_norm = l2_normalize(test_features).numpy()     # (n_test, D)

    # ---- Get benign key pool ----
    all_keys = model.pool.key.detach().cpu()             # (M, D)
    all_keys_norm = l2_normalize(all_keys).numpy()       # (M, D)

    # ---- Estimate sigma_i using train queries and ALL keys as benign ----
    # (Before adversarial injection, all keys are benign)
    sigmas_all, counts_all = estimate_cluster_sigmas(train_norm, all_keys_norm)
    print(f"\nCluster sigma statistics (all {M} keys as benign):")
    print(f"  mean sigma = {np.mean(sigmas_all):.6f}")
    print(f"  min sigma  = {np.min(sigmas_all):.6f}")
    print(f"  max sigma  = {np.max(sigmas_all):.6f}")
    print(f"  cluster sizes: {counts_all}")

    # ---- For each target, generate adversarial keys and compute FPR bound ----
    n_targets = min(args.n_targets, test_norm.shape[0])
    np.random.seed(42)
    target_indices = np.random.choice(test_norm.shape[0], n_targets, replace=False)

    print(f"\nComputing FPR bounds for {n_targets} targets...")

    all_fpr_bounds = []
    all_per_cluster_phi = []

    for t_i, t_idx in enumerate(target_indices):
        target_query = test_norm[t_idx]  # (D,) — L2-normalized target query

        # 1. Generate adversarial keys using the same method as mia_stealth.py
        #    The adversarial key generation operates on normalized keys/queries
        adv_keys, chosen_idx = generate_adversarial_key_set(
            target_query, all_keys_norm, N,
            args.delta_min, args.delta_max, args.beta
        )
        # adv_keys: (N, D), chosen_idx: list of N indices that were replaced

        # 2. Determine benign keys (the ones NOT replaced)
        benign_mask = np.ones(M, dtype=bool)
        benign_mask[chosen_idx] = False
        benign_keys = all_keys_norm[benign_mask]  # (K, D)

        # 3. Estimate sigma_i for benign clusters using train queries
        #    Re-estimate with only benign keys (after adversarial injection,
        #    only benign keys are cluster centroids for non-member queries)
        sigmas, counts = estimate_cluster_sigmas(train_norm, benign_keys)

        # 4. Compute FPR bound from Theorem 3.3
        fpr_bound, per_cluster_phi, per_key_phi = compute_fpr_bound_theorem33(
            benign_keys, adv_keys, sigmas
        )

        all_fpr_bounds.append(fpr_bound)
        all_per_cluster_phi.append(per_cluster_phi)

        if args.verbose and t_i < 10:
            print(f"\n  Target {t_i} (idx={t_idx}):")
            print(f"    sigma range: [{sigmas.min():.6f}, {sigmas.max():.6f}]")
            print(f"    per-cluster Phi(z): {per_cluster_phi}")
            print(f"    FPR bound (per-sample): {fpr_bound:.6e}")

    all_fpr_bounds = np.array(all_fpr_bounds)

    # ---- Report results ----
    print("\n" + "=" * 90)
    print("PromptMIA Theorem 3.3 FPR Bound (Empirical)")
    print(f"  D={D}, M={M}, N={N}, K={K}")
    print(f"  delta_min={args.delta_min}, delta_max={args.delta_max}, beta={args.beta}")
    print(f"  n_targets={n_targets}, n_train_queries={train_norm.shape[0]}")
    print("=" * 90)

    print(f"\nPer-sample FPR bound statistics across {n_targets} targets:")
    print(f"  mean = {np.mean(all_fpr_bounds):.6e}")
    print(f"  std  = {np.std(all_fpr_bounds):.6e}")
    print(f"  min  = {np.min(all_fpr_bounds):.6e}")
    print(f"  max  = {np.max(all_fpr_bounds):.6e}")
    print(f"  median = {np.median(all_fpr_bounds):.6e}")

    advantage = 1.0 - np.mean(all_fpr_bounds)  # TPR=1 by Theorem 3.1
    asr = (1.0 + advantage) / 2.0
    print(f"\nAdvantage = {advantage:.6f}, ASR = {asr:.6f}")

    # ---- Per-cluster sigma analysis ----
    print(f"\nCluster sigma_i (from last target, {K} benign clusters):")
    last_sigmas, last_counts = estimate_cluster_sigmas(train_norm, all_keys_norm[np.ones(M, dtype=bool)])
    for i in range(M):
        print(f"  cluster {i:2d}: sigma={last_sigmas[i]:.6f}, n_queries={last_counts[i]}")


if __name__ == "__main__":
    main()
