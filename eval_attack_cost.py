"""
eval_attack_cost.py — Evaluate the computational cost, sparsity, budget, and
scalability of the PromptMIA stealth attack.

Metrics computed:
  1. Attack Sparsity: fraction of model parameters modified by the attack.
  2. Attack Budget: number of adversarial keys injected, prompts affected,
     and communication overhead per attack round.
  3. Adversarial Key Generation Cost: FLOPs, wall-clock time, and throughput
     (keys per second) for generating adversarial keys.
  4. Inference Latency: time to run the MIA membership decision for a single
     target (key generation + similarity + decision), with FPS.
  5. Scalability: how costs scale with pool size M, top-k N, embedding dim D,
     and number of targets.

Usage:
    python eval_attack_cost.py \\
        --gpus 0 --dataset cifar10 --query_model_type vit_cls \\
        --pool_size 20 --n_tokens 4 \\
        --load_model <path_to_saved_model> \\
        --delta_min 0.02 --delta_max 0.05 --beta 0 \\
        --n_targets 200 --max_samples 5000

    # Evaluate attack sparsity across multiple model families
    python eval_attack_cost.py \\
        --gpus 0 --dataset cifar10 \\
        --query_model_types vit_cls deit_b16 convit_base \\
        --pool_size 20 --n_tokens 4
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.utils.data as Data
import torchvision

from data_util.DataTransformBuilder import build_data_transform
from data_util.TinyImageNet_reader import TinyImageNet_reader
from data_util.Four_dataset_reader import four_dataset_reader
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_DeiT_B16, L2P_ConViT_Base
from util.TOdevice import to_device
from util.saving_tools import load_model_trainable_part


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions (matching mia_stealth.py logic)
# ──────────────────────────────────────────────────────────────────────────────

def l2_normalize(x, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def generate_point_with_similarity(target_vec, similarity, benign_key, beta=0.0):
    """Generate one adversarial key (same as mia_stealth.py)."""
    target_vec = target_vec.reshape(-1)
    benign_key = benign_key.reshape(-1)
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


def generate_adversarial_key_set(query_np, keys_norm_np, N, delta_min, delta_max, beta):
    """Generate N adversarial keys for a target query. Returns keys + indices."""
    target_norm = query_np / np.linalg.norm(query_np)
    sims = keys_norm_np @ target_norm
    topk_idx = np.argsort(sims)[-N:][::-1].tolist()

    mask = np.ones(len(sims), dtype=bool)
    mask[topk_idx] = False
    max_benign_sim = sims[mask].max()

    s_start = min(max_benign_sim + delta_min, 1.0)
    s_end = min(max_benign_sim + delta_min + delta_max, 1.0)
    varied_sims = np.random.uniform(s_start, s_end, N)

    non_chosen = np.where(mask)[0]
    anchor_idx = np.random.choice(non_chosen, size=N, replace=(N > len(non_chosen)))

    adv_keys = []
    for s, a_idx in zip(varied_sims, anchor_idx):
        k_a = generate_point_with_similarity(query_np, s, keys_norm_np[a_idx], beta)
        adv_keys.append(k_a)
    return np.array(adv_keys), topk_idx


def mia_decision(query_norms, key_pool_norm, adv_indices, N):
    """
    Run the MIA decision for a batch of queries.
    Returns True if attack predicts 'member' (all adv keys in top-N for at least one sample).
    """
    sims = query_norms @ key_pool_norm.T  # (B, M)
    topk = np.argsort(sims, axis=1)[:, -N:]  # (B, N)
    for adv_idx in adv_indices:
        if not np.any(topk == adv_idx):
            return False
    return True


def extract_cls_features(model, dataset, device, max_samples=None, batch_size=64):
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats = []
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = to_device(x, device).float()
            cls = model.get_cls_keys(x)
            feats.append(cls.cpu())
            count += cls.shape[0]
            if max_samples and count >= max_samples:
                break
    feats = torch.cat(feats, dim=0)
    if max_samples:
        feats = feats[:max_samples]
    return feats


def count_flops_keygen(D, N):
    """
    Count FLOPs for generating N adversarial keys in D dimensions.

    Per key:
      - normalize target:  D mul + D add + 1 sqrt + D div ≈ 3D
      - normalize benign:  3D
      - random vec + norm: D + 3D = 4D
      - final_vec blend:   2D + 3D = 5D  (scale + add + normalize)
      - orthogonalize:     D (dot) + D (scale) + D (sub) + 3D (norm) = 6D
      - combine:           D (scale target) + D (sqrt+scale orth) + D (add) + 3D (norm) = 6D
      - scale to norm:     D mul
    Total per key ≈ 28D FLOPs

    Returns total FLOPs for N keys.
    """
    flops_per_key = 28 * D
    return N * flops_per_key


def count_flops_similarity(B, M, D):
    """FLOPs for computing cosine similarity: matmul(B,D) x (M,D)^T = 2*B*M*D."""
    return 2 * B * M * D


def count_flops_topk(B, M, N):
    """FLOPs for top-k selection: ≈ B * M * log2(N) comparisons."""
    import math
    return int(B * M * math.log2(max(N, 2)))


def count_flops_decision(B, N_adv, topk_size):
    """FLOPs for checking if all adv keys are in top-k: B * N_adv * topk_size comparisons."""
    return B * N_adv * topk_size


def build_query_model(query_model_type, prompt_method, batchwise_prompt, M, N, num_classes):
    """Build one query model instance for the requested architecture."""
    if query_model_type == "vit_cls":
        return L2P_ViT_B32(
            prompt_method=prompt_method,
            batchwise_prompt=batchwise_prompt,
            pool_size=M,
            top_k=N,
            num_classes=num_classes,
            candidate_key=None,
        )
    if query_model_type == "deit_b16":
        return L2P_DeiT_B16(
            prompt_method=prompt_method,
            batchwise_prompt=batchwise_prompt,
            pool_size=M,
            top_k=N,
            num_classes=num_classes,
            candidate_key=None,
            pretrained=True,
        )
    if query_model_type == "convit_base":
        return L2P_ConViT_Base(
            prompt_method=prompt_method,
            batchwise_prompt=batchwise_prompt,
            pool_size=M,
            top_k=N,
            num_classes=num_classes,
            candidate_key=None,
            pretrained=True,
        )
    raise ValueError(f"Unsupported query_model_type: {query_model_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate attack cost, sparsity, budget, and scalability of PromptMIA."
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimagenet", "fourdataset"])
    parser.add_argument("--query_model_type", type=str, default="vit_cls",
                        choices=["vit_cls", "deit_b16", "convit_base"])
    parser.add_argument("--query_model_types", nargs="+", default=None,
                        choices=["vit_cls", "deit_b16", "convit_base"],
                        help="Optional list of model types for multi-model sparsity comparison.")
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--n_tokens", type=int, default=4)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--delta_min", type=float, default=0.02)
    parser.add_argument("--delta_max", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--n_targets", type=int, default=200)
    parser.add_argument("--prompt_method", type=str, default="shallow")
    parser.add_argument("--batchwise_prompt", type=bool, default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = args.device

    M = args.pool_size
    N = args.n_tokens
    D = 768  # ViT-B embed dim

    if args.query_model_types is None:
        query_model_types = [args.query_model_type]
    else:
        # Preserve user order and remove duplicates.
        query_model_types = list(dict.fromkeys(args.query_model_types))

    # ── Build model ──
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

    if len(query_model_types) > 1:
        print("\n" + "=" * 80)
        print("ATTACK SPARSITY ACROSS MULTIPLE query_model_type")
        print("=" * 80)
        print(f"  Dataset: {args.dataset}")
        print(f"  Pool setup: M={M}, N={N}")
        print(f"  Models: {', '.join(query_model_types)}")

        print("\n  "
              f"{'Model':<12s} | {'D':>5s} | {'Modified':>12s} | {'Total params':>14s} | "
              f"{'Trainable':>14s} | {'Sparsity total (%)':>18s} | "
              f"{'Sparsity trainable (%)':>22s} | {'Sparsity key-pool (%)':>21s}")
        print("  " + "-" * 145)

        for model_type in query_model_types:
            model = build_query_model(
                model_type,
                args.prompt_method,
                args.batchwise_prompt,
                M,
                N,
                num_classes,
            )
            model = to_device(model, device)
            model.build_trainable_keys()

            if args.load_model and args.load_model != "random":
                try:
                    load_model_trainable_part(model, args.load_model)
                    print(f"  [info] Loaded model weights for {model_type} from {args.load_model}")
                except Exception as e:
                    print(f"  [warn] Failed to load weights for {model_type}: {e}")

            model.eval()

            D_model = model.pool.key.shape[1]
            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            attack_modified_params = N * D_model
            pool_key_params = model.pool.key.numel()

            print(
                "  "
                f"{model_type:<12s} | {D_model:>5d} | {attack_modified_params:>12,d} | "
                f"{total_params:>14,d} | {total_trainable:>14,d} | "
                f"{attack_modified_params / total_params * 100:>18.4f} | "
                f"{attack_modified_params / total_trainable * 100:>22.4f} | "
                f"{attack_modified_params / pool_key_params * 100:>21.2f}"
            )

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        return

    model = build_query_model(
        query_model_types[0],
        args.prompt_method,
        args.batchwise_prompt,
        M,
        N,
        num_classes,
    )

    model = to_device(model, device)
    model.build_trainable_keys()
    if args.load_model and args.load_model != "random":
        load_model_trainable_part(model, args.load_model)
        print(f"Model loaded from {args.load_model}")
    model.eval()

    D = model.pool.key.shape[1]

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. ATTACK SPARSITY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. ATTACK SPARSITY")
    print("=" * 80)

    # Count all trainable parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Count total model parameters (including frozen backbone)
    total_params = sum(p.numel() for p in model.parameters())
    # Parameters modified by the attack: only the N key vectors that get replaced
    attack_modified_params = N * D
    # Prompt pool parameters
    pool_key_params = model.pool.key.numel()      # M * D
    pool_prompt_params = model.pool.prompt.numel()  # M * D

    print(f"  Total model parameters:       {total_params:>12,d}")
    print(f"  Trainable parameters:         {total_trainable:>12,d}")
    print(f"  Pool key parameters (M×D):    {pool_key_params:>12,d}  ({M}×{D})")
    print(f"  Pool prompt parameters (M×D): {pool_prompt_params:>12,d}  ({M}×{D})")
    print(f"  Parameters modified by attack:{attack_modified_params:>12,d}  ({N}×{D})")
    print(f"  Attack sparsity (vs total):   {attack_modified_params/total_params*100:>12.4f}%")
    print(f"  Attack sparsity (vs trainable):{attack_modified_params/total_trainable*100:>11.4f}%")
    print(f"  Attack sparsity (vs key pool):{attack_modified_params/pool_key_params*100:>12.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. ATTACK BUDGET
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. ATTACK BUDGET")
    print("=" * 80)

    # Per target: N adversarial keys generated, N keys replaced in pool
    # Communication: attacker sends modified key pool (M×D floats) per round
    bytes_per_float = 4  # float32
    key_pool_bytes = M * D * bytes_per_float
    prompt_pool_bytes = M * D * bytes_per_float
    total_pool_bytes = key_pool_bytes + prompt_pool_bytes
    attack_payload_bytes = N * D * bytes_per_float  # only the modified keys

    print(f"  Number of adversarial keys per target:  {N}")
    print(f"  Keys replaced per target:               {N} out of {M} ({N/M*100:.1f}%)")
    print(f"  Prompts affected per target:            {N} (associated with replaced keys)")
    print(f"  Key pool size (communicated per round):   {key_pool_bytes:>10,d} bytes ({key_pool_bytes/1024:.1f} KB)")
    print(f"  Prompt pool size (communicated per round): {prompt_pool_bytes:>10,d} bytes ({prompt_pool_bytes/1024:.1f} KB)")
    print(f"  Total pool communication per round:       {total_pool_bytes:>10,d} bytes ({total_pool_bytes/1024:.1f} KB)")
    print(f"  Attack-specific payload (N×D floats):     {attack_payload_bytes:>10,d} bytes ({attack_payload_bytes/1024:.1f} KB)")
    print(f"  Attack overhead vs normal update:         {attack_payload_bytes/total_pool_bytes*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. ADVERSARIAL KEY GENERATION COST
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. ADVERSARIAL KEY GENERATION COST")
    print("=" * 80)

    # FLOPs analysis
    flops_keygen = count_flops_keygen(D, N)
    print(f"  Key generation FLOPs (N={N} keys, D={D}): {flops_keygen:>10,d}  ({flops_keygen/1e3:.1f} KFLOPs)")

    # Wall-clock timing: generate adversarial keys for many targets
    all_keys = model.pool.key.detach().cpu()
    all_keys_norm = l2_normalize(all_keys).numpy()

    print("\n  Extracting test features for timing...")
    test_features = extract_cls_features(model, testset, device, args.max_samples)
    test_norm = l2_normalize(test_features).numpy()

    n_targets = min(args.n_targets, test_norm.shape[0])
    target_indices = np.random.choice(test_norm.shape[0], n_targets, replace=False)

    # Warm up
    for i in range(min(10, n_targets)):
        _ = generate_adversarial_key_set(
            test_norm[target_indices[i]], all_keys_norm, N,
            args.delta_min, args.delta_max, args.beta
        )

    # Timed keygen
    t_start = time.perf_counter()
    for i in range(n_targets):
        _ = generate_adversarial_key_set(
            test_norm[target_indices[i]], all_keys_norm, N,
            args.delta_min, args.delta_max, args.beta
        )
    t_keygen = time.perf_counter() - t_start

    keygen_per_target = t_keygen / n_targets
    keygen_fps = n_targets / t_keygen

    print(f"  Targets evaluated:            {n_targets}")
    print(f"  Total keygen time:            {t_keygen*1000:.2f} ms")
    print(f"  Time per target (N={N} keys): {keygen_per_target*1e6:.1f} µs")
    print(f"  Keygen throughput:            {keygen_fps:.0f} targets/sec")
    print(f"  Keys per second:              {keygen_fps * N:.0f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. INFERENCE LATENCY (full MIA decision pipeline)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. MIA INFERENCE LATENCY (end-to-end per target)")
    print("=" * 80)

    # Phase breakdown: (a) query extraction, (b) keygen, (c) similarity + decision
    # (a) Query extraction (GPU)
    single_loader = Data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    sample_x, _ = next(iter(single_loader))
    sample_x = to_device(sample_x, device).float()

    # Warm up GPU
    for _ in range(10):
        with torch.no_grad():
            _ = model.get_cls_keys(sample_x)
    if device == "cuda":
        torch.cuda.synchronize()

    n_query_trials = 100
    t_start = time.perf_counter()
    for _ in range(n_query_trials):
        with torch.no_grad():
            q = model.get_cls_keys(sample_x)
    if device == "cuda":
        torch.cuda.synchronize()
    t_query = (time.perf_counter() - t_start) / n_query_trials

    # (b) Keygen: already measured above
    # (c) Similarity computation + decision (CPU, numpy)
    query_np = l2_normalize(q.cpu()).numpy().reshape(-1)

    n_sim_trials = 1000
    t_start = time.perf_counter()
    for _ in range(n_sim_trials):
        adv_keys, chosen_idx = generate_adversarial_key_set(
            query_np, all_keys_norm, N, args.delta_min, args.delta_max, args.beta
        )
        # Build modified key pool
        modified_keys = all_keys_norm.copy()
        modified_keys[chosen_idx] = adv_keys / (np.linalg.norm(adv_keys, axis=1, keepdims=True) + 1e-12)
        # Similarity + decision for batch size 1
        mia_decision(query_np.reshape(1, -1), modified_keys, chosen_idx, N)
    t_sim = (time.perf_counter() - t_start) / n_sim_trials

    # FLOPs for similarity + decision
    B = 1
    flops_sim = count_flops_similarity(B, M, D)
    flops_topk = count_flops_topk(B, M, N)
    flops_dec = count_flops_decision(B, N, N)
    flops_total = flops_keygen + flops_sim + flops_topk + flops_dec

    t_total = t_query + keygen_per_target + t_sim
    fps = 1.0 / t_total

    print(f"  (a) Query extraction (GPU):   {t_query*1e6:>10.1f} µs")
    print(f"  (b) Adversarial keygen (CPU): {keygen_per_target*1e6:>10.1f} µs")
    print(f"  (c) Similarity + decision:    {t_sim*1e6:>10.1f} µs")
    print("  ─────────────────────────────────────────")
    print(f"  Total per target (B=1):       {t_total*1e6:>10.1f} µs  ({t_total*1000:.3f} ms)")
    print(f"  FPS (targets/sec):            {fps:>10.0f}")

    print("\n  FLOPs breakdown (B=1):")
    print(f"    Key generation:             {flops_keygen:>12,d}")
    print(f"    Cosine similarity (B×M×D):  {flops_sim:>12,d}")
    print(f"    Top-k selection:            {flops_topk:>12,d}")
    print(f"    Decision check:             {flops_dec:>12,d}")
    print("    ───────────────────────────────────────")
    print(f"    Total per target:           {flops_total:>12,d}  ({flops_total/1e6:.2f} MFLOPs)")

    # Batch size sweep
    print("\n  Batch size sweep (similarity + decision only):")
    print(f"  {'B':>6s} | {'FLOPs':>12s} | {'Time (µs)':>12s} | {'FPS':>10s}")
    print(f"  {'-'*48}")
    for B in [1, 4, 8, 16, 32, 64, 128, 256]:
        f = count_flops_similarity(B, M, D) + count_flops_topk(B, M, N) + count_flops_decision(B, N, N)
        # Time similarity for this batch size
        batch_queries = test_norm[:B]
        n_batch_trials = 500
        t0 = time.perf_counter()
        for _ in range(n_batch_trials):
            sims = batch_queries @ all_keys_norm.T
            topk = np.argsort(sims, axis=1)[:, -N:]
            for idx in range(N):
                np.any(topk == idx)
        t_batch = (time.perf_counter() - t0) / n_batch_trials
        total_t = t_query + keygen_per_target + t_batch
        print(f"  {B:>6d} | {f:>12,d} | {total_t*1e6:>12.1f} | {1.0/total_t:>10.0f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. SCALABILITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. SCALABILITY ANALYSIS")
    print("=" * 80)

    # (a) Scaling with pool size M
    print(f"\n  (a) Scaling with pool size M (N={N}, D={D}):")
    print(f"  {'M':>6s} | {'Modified params':>16s} | {'Sparsity (%)':>13s} | {'KeyGen FLOPs':>13s} | {'Sim FLOPs (B=1)':>16s} | {'Total FLOPs':>12s}")
    print(f"  {'-'*90}")
    for M_test in [5, 10, 20, 50, 100]:
        mod_p = N * D
        sp = mod_p / (total_params) * 100  # sparsity stays same since backbone dominates
        fg = count_flops_keygen(D, N)
        fs = count_flops_similarity(1, M_test, D)
        ft = fg + fs + count_flops_topk(1, M_test, N) + count_flops_decision(1, N, N)
        print(f"  {M_test:>6d} | {mod_p:>16,d} | {sp:>13.4f} | {fg:>13,d} | {fs:>16,d} | {ft:>12,d}")

    # (b) Scaling with top-k N (adversarial keys)
    print(f"\n  (b) Scaling with top-k N (M={M}, D={D}):")
    print(f"  {'N':>6s} | {'Modified params':>16s} | {'Sparsity (%)':>13s} | {'KeyGen FLOPs':>13s} | {'KeyGen time (µs)':>18s}")
    print(f"  {'-'*75}")
    for N_test in [1, 2, 4, 8, 16]:
        if N_test >= M:
            continue
        mod_p = N_test * D
        sp = mod_p / total_params * 100
        fg = count_flops_keygen(D, N_test)
        # Time it
        t0 = time.perf_counter()
        for i in range(200):
            _ = generate_adversarial_key_set(
                test_norm[target_indices[i % n_targets]], all_keys_norm, N_test,
                args.delta_min, args.delta_max, args.beta
            )
        t_n = (time.perf_counter() - t0) / 200
        print(f"  {N_test:>6d} | {mod_p:>16,d} | {sp:>13.4f} | {fg:>13,d} | {t_n*1e6:>18.1f}")

    # (c) Scaling with embedding dimension D
    print(f"\n  (c) Scaling with embedding dimension D (M={M}, N={N}):")
    print(f"  {'D':>6s} | {'KeyGen FLOPs':>13s} | {'Sim FLOPs (B=1)':>16s} | {'Modified params':>16s}")
    print(f"  {'-'*60}")
    for D_test in [192, 384, 768, 1024, 2048]:
        fg = count_flops_keygen(D_test, N)
        fs = count_flops_similarity(1, M, D_test)
        mp = N * D_test
        print(f"  {D_test:>6d} | {fg:>13,d} | {fs:>16,d} | {mp:>16,d}")

    # (d) Scaling with number of targets
    print("\n  (d) Scaling with number of targets:")
    print(f"  {'n_targets':>10s} | {'Total keygen time':>18s} | {'Total MIA time':>15s}")
    print(f"  {'-'*50}")
    for nt in [1, 10, 100, 1000, 10000]:
        tkg = keygen_per_target * nt
        ttotal = t_total * nt
        print(f"  {nt:>10d} | {tkg*1000:>15.2f} ms | {ttotal*1000:>12.2f} ms")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Model:                {query_model_types[0]}")
    print(f"  Dataset:              {args.dataset}")
    print(f"  Pool: M={M}, N={N}, D={D}")
    print(f"  Total model params:   {total_params:,d}")
    print(f"  Trainable params:     {total_trainable:,d}")
    print(f"  Attack modifies:      {attack_modified_params:,d} params ({attack_modified_params/total_params*100:.4f}% of total)")
    print(f"  Attack FLOPs/target:  {flops_total:,d} ({flops_total/1e6:.2f} MFLOPs)")
    print(f"  Attack latency:       {t_total*1000:.3f} ms/target")
    print(f"  Attack throughput:    {fps:.0f} targets/sec")
    print(f"  Communication:        {total_pool_bytes/1024:.1f} KB/round (pool), {attack_payload_bytes/1024:.1f} KB attack payload")

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. DISCUSSION-READY TAKEAWAYS
    # ═══════════════════════════════════════════════════════════════════════════
    keygen_share = (keygen_per_target / t_total) * 100
    query_share = (t_query / t_total) * 100
    sim_share = (t_sim / t_total) * 100

    print("\n" + "=" * 80)
    print("6. DISCUSSION-READY TAKEAWAYS")
    print("=" * 80)
    print("  • Computational overhead is explicitly quantified via FLOPs, wall-clock latency, and throughput.")
    print(f"    - End-to-end attack cost: {flops_total/1e6:.2f} MFLOPs and {t_total*1000:.3f} ms per target ({fps:.0f} targets/s).")
    print(f"    - Runtime composition: query extraction {query_share:.1f}%, adversarial key generation {keygen_share:.1f}%, similarity+decision {sim_share:.1f}%.")
    print("  • Adversarial prompt generation overhead is measured in isolation.")
    print(f"    - Key generation: {flops_keygen/1e3:.1f} KFLOPs per target, {keygen_per_target*1e6:.1f} us/target, {keygen_fps*N:.0f} keys/s.")
    print("  • Attack scalability is characterized along all main dimensions (M, N, D, number of targets).")
    print("    - Similarity FLOPs scale linearly with pool size and embedding dimension: O(MD).")
    print("    - Key generation cost and modified parameters scale linearly with top-k tokens: O(ND).")
    print("    - Aggregate runtime scales linearly with the number of attacked targets.")
    print("  • Communication overhead is bounded by sparse key replacement.")
    print(f"    - Attack payload is {attack_payload_bytes/1024:.1f} KB vs full pool transfer {total_pool_bytes/1024:.1f} KB per round.")


if __name__ == "__main__":
    main()
