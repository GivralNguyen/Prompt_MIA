import numpy as np, torch, torchvision, os
import torch.utils.data as Data
from data_util.DataTransformBuilder import build_data_transform
from Models.L2P_heuristic_model import L2P_ViT_B32
from util.TOdevice import to_device
from util.saving_tools import load_model_trainable_part

def l2_normalize(x, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def estimate_sigmas(feats_norm, keys_norm):
    K, D = keys_norm.shape
    sims = feats_norm @ keys_norm.T
    assignments = np.argmax(sims, axis=1)
    sigmas = np.zeros(K)
    counts = np.zeros(K, dtype=int)
    for i in range(K):
        members = feats_norm[assignments == i]
        counts[i] = len(members)
        if len(members) < 2:
            sigmas[i] = np.nan
        else:
            diffs = members - keys_norm[i]
            sq_dists = np.sum(diffs**2, axis=1)
            sigmas[i] = np.sqrt(np.mean(sq_dists) / D)
    return sigmas, counts

norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
tp = build_data_transform(norm_stats, 256, 224, train_augment=False)
trainset = torchvision.datasets.CIFAR10(root='./Dataset', train=True, download=False, transform=tp)

model = L2P_ViT_B32(prompt_method='shallow', batchwise_prompt=False, pool_size=20, top_k=4, num_classes=10, candidate_key=None)
model = to_device(model, 'cuda')
model.build_trainable_keys()
load_model_trainable_part(model, 'output_record/saved_model/saved_model_L2P_fedavg_cifar10_80clietns_cross_devices_modelvit_cls_non_iid_dirichlet_alpha0.5_1_round_61_acc_0.9442.pt')

# Extract ALL 50000 training features
loader = Data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
all_feats = []
with torch.no_grad():
    for x, _ in loader:
        x = to_device(x, 'cuda').float()
        all_feats.append(model.get_cls_keys(x).cpu())
all_feats = torch.cat(all_feats, dim=0)
all_feats_norm = l2_normalize(all_feats).numpy()
print(f"Total features extracted: {all_feats_norm.shape[0]}")

keys = model.pool.key.detach().cpu()
keys_norm = l2_normalize(keys).numpy()
D = 768
M = 20

# Ground truth: sigma from ALL 50000 samples
gt_sigmas, gt_counts = estimate_sigmas(all_feats_norm, keys_norm)
print(f"\n=== Ground truth sigma (n=50000) ===")
for i in range(M):
    print(f"  cluster {i:2d}: sigma={gt_sigmas[i]:.6f}, n={gt_counts[i]}")

# Test different sample sizes
sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
n_trials = 10  # repeat each to get error statistics

print(f"\n{'n_samples':>10s} | {'mean_abs_err':>13s} | {'max_abs_err':>13s} | {'mean_rel_err%':>14s} | {'max_rel_err%':>13s} | {'min_cluster_n':>14s}")
print("-" * 95)

for n in sample_sizes:
    abs_errs = []
    rel_errs = []
    min_cluster_ns = []
    for trial in range(n_trials):
        idx = np.random.choice(all_feats_norm.shape[0], min(n, all_feats_norm.shape[0]), replace=False)
        subset = all_feats_norm[idx]
        est_sigmas, est_counts = estimate_sigmas(subset, keys_norm)
        
        # Only compare clusters that have enough samples in both
        valid = (~np.isnan(est_sigmas)) & (~np.isnan(gt_sigmas))
        if valid.sum() == 0:
            continue
        ae = np.abs(est_sigmas[valid] - gt_sigmas[valid])
        re = ae / gt_sigmas[valid] * 100
        abs_errs.append(ae)
        rel_errs.append(re)
        min_cluster_ns.append(est_counts[valid].min())
    
    if abs_errs:
        all_ae = np.concatenate(abs_errs)
        all_re = np.concatenate(rel_errs)
        mcn = min(min_cluster_ns)
        print(f"{n:>10d} | {np.mean(all_ae):>13.6f} | {np.max(all_ae):>13.6f} | {np.mean(all_re):>13.2f}% | {np.max(all_re):>12.2f}% | {mcn:>14d}")