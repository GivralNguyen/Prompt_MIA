"""Figures 2, 3 and 18 from a training run, plus 768-D diagnostics behind them.

Plot method follows the paper's notebook: L2-normalised keys, the first 200
training queries, t-SNE(perplexity=10, random_state=42); Fig. 3 regions are
K-Means(5) over the key embeddings and a per-key 2-D Gaussian (MAD scale).
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from promptmia.data import load_cifar10, ImageCache
from promptmia.model import L2PViT
from promptmia.attack import promptmia_inject, naive_inject

p = argparse.ArgumentParser()
p.add_argument('--run', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'cifar10_vit_b32'))
p.add_argument('--round', type=int, default=60, help='round of the keys used in Fig. 2 and 3')
p.add_argument('--rounds18', default='0,1,20,60')
p.add_argument('--target', type=int, default=0, help='training-set index of the target sample T (Fig. 2)')
p.add_argument('--n_queries', type=int, default=200)
p.add_argument('--delta_min', type=float, default=0.02)
p.add_argument('--delta', type=float, default=0.05)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default=None, help='output folder (default: <run>/figures)')
args = p.parse_args()

cfg = json.load(open(os.path.join(args.run, 'config.json')))
keys_by_round = torch.load(os.path.join(args.run, 'keys.pt'))
out = args.out or os.path.join(args.run, 'figures')
os.makedirs(out, exist_ok=True)
dev = 'cuda'
N = cfg['N']

# queries of the first n_queries training images, plus the target
train_set, _, n_classes = load_cifar10()
ids = sorted(set(range(args.n_queries)) | {args.target})
cache = ImageCache(Data.Subset(train_set, ids), dev)
model = L2PViT(n_classes, cfg['M'], N, cfg['head']).to(dev).eval()
with torch.no_grad():
    Q = model.query(cache.get(torch.arange(len(cache), device=dev))[0]).cpu()
q_target = Q[ids.index(args.target)]
Q = Q[:args.n_queries]
Qn = F.normalize(Q, dim=1).numpy()

def tsne(x):
    return TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(x)

def style(ax, title):
    ax.set_title(title, fontsize=13); ax.grid(True); ax.legend(fontsize=9)

def cos_offdiag(k):
    c = F.normalize(k, dim=1) @ F.normalize(k, dim=1).t()
    return c[~torch.eye(len(k), dtype=torch.bool)]

def n_distinct(k, thr=0.99):
    """Number of groups of keys whose pairwise cosine exceeds thr (connected components)."""
    c = (F.normalize(k, dim=1) @ F.normalize(k, dim=1).t()) > thr
    seen, groups = set(), 0
    for i in range(len(k)):
        if i in seen:
            continue
        groups += 1
        stack = [i]
        while stack:
            j = stack.pop()
            if j not in seen:
                seen.add(j); stack.extend(torch.nonzero(c[j]).flatten().tolist())
    return groups

diag = {}
keys = keys_by_round[args.round].double()

# ---------------- Fig. 2: keys without injection, naive injection, PromptMIA injection
gen = torch.Generator().manual_seed(args.seed)
k_naive, S_naive = naive_inject(keys, q_target.double(), N)
k_pmia, S, (lo, hi) = promptmia_inject(keys, q_target.double(), N, args.delta_min, args.delta, gen)
mask = torch.ones(len(keys), dtype=torch.bool); mask[S] = False
sim_t = F.normalize(k_pmia, dim=1) @ F.normalize(q_target.double(), dim=0)
top_sel = set(sim_t.topk(N).indices.tolist())
diag['fig2'] = {
    'target_index': args.target, 'injected_slots': S.tolist(), 'eq7_interval': [lo, hi],
    'adv_similarity_to_qT': sim_t[S].tolist(), 'max_benign_similarity_to_qT': sim_t[mask].max().item(),
    'target_selects_exactly_adv_keys': top_sel == set(S.tolist()),
    'cos_adv_adv_mean': cos_offdiag(k_pmia[S]).mean().item(),
    'cos_benign_benign_mean': cos_offdiag(k_pmia[mask]).mean().item(),
    'cos_benign_benign_max': cos_offdiag(k_pmia[mask]).max().item(),
}
fig, axs = plt.subplots(1, 3, figsize=(21, 6))
for ax, (k, adv, title) in zip(axs, [(keys, [], 'Unmodified'), (k_naive, S_naive.tolist(), 'with Naive Injected Keys Highlighted'),
                                     (k_pmia, S.tolist(), 'with PromptMIA Injected Keys Highlighted')]):
    e = tsne(F.normalize(k, dim=1).float().numpy())
    ben = [i for i in range(len(k)) if i not in adv]
    ax.scatter(e[ben, 0], e[ben, 1], c='#9999ff', s=80, label='Benign Keys')
    if adv:
        ax.scatter(e[adv, 0], e[adv, 1], c='red', edgecolors='black', s=80, label='Adversarial Keys', zorder=5)
    for i in range(len(k)):
        ax.annotate(str(i), e[i], textcoords='offset points', xytext=(0, 5), ha='center', fontsize=9,
                    color='darkred' if i in adv else 'navy')
    style(ax, f't-SNE of Global Keys ({title})')
fig.tight_layout(); fig.savefig(os.path.join(out, 'fig2.pdf')); fig.savefig(os.path.join(out, 'fig2.png'), dpi=150)

# ---------------- Fig. 3: keys + queries, K-Means regions, Gaussian query model
def joint_tsne(k):
    e = tsne(np.concatenate([F.normalize(k.float(), dim=1).numpy(), Qn]))
    return e[:len(k)], e[len(k):]

ke, qe = joint_tsne(keys)
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(ke)
centers = km.cluster_centers_
q_region = ((qe[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
lo_xy, hi_xy = np.vstack([ke, qe]).min(0) - 5, np.vstack([ke, qe]).max(0) + 5
xx, yy = np.meshgrid(np.linspace(lo_xy[0], hi_xy[0], 400), np.linspace(lo_xy[1], hi_xy[1], 400))
grid = np.c_[xx.ravel(), yy.ravel()]
cmap = ListedColormap(['#ffb347', '#c9e4ff', '#fff3c4', '#d6d6d6', '#e6d6ff'])

d_qk = ((qe[:, None] - ke[None]) ** 2).sum(-1) ** 0.5
nearest = d_qk.argmin(1)
sig = np.zeros(len(ke))
for i in range(len(ke)):
    d = d_qk[nearest == i, i]
    if len(d):
        mad = np.median(np.abs(d - np.median(d)))
        sig[i] = max(mad / 0.6745 if mad > 0 else d.std(), 1e-3)
logpdf = np.full(len(grid), -np.inf)
for (mx, my), s in zip(ke, sig):
    if s > 0:
        logpdf = np.logaddexp(logpdf, -np.log(2 * np.pi * s * s) - 0.5 * ((grid[:, 0] - mx) ** 2 + (grid[:, 1] - my) ** 2) / (s * s))

fig, axs = plt.subplots(1, 3, figsize=(21, 6))
axs[0].scatter(qe[:, 0], qe[:, 1], c='green', edgecolors='black', s=40, label='Query Vectors')
axs[0].scatter(ke[:, 0], ke[:, 1], c='blue', alpha=0.5, s=60, label='Benign Keys', zorder=5)
style(axs[0], 't-SNE of Benign Keys vs. Query Vectors in Trainset')
axs[1].pcolormesh(xx, yy, ((grid[:, None] - centers[None]) ** 2).sum(-1).argmin(1).reshape(xx.shape), cmap=cmap, alpha=0.25, shading='auto')
axs[2].contourf(xx, yy, logpdf.reshape(xx.shape), levels=18, alpha=0.2)
for ax in axs[1:]:
    for r in range(5):
        m = q_region == r
        ax.scatter(qe[m, 0], qe[m, 1], color=cmap.colors[r], edgecolors='black', s=40, label=f'Cluster {r} queries')
    ax.scatter(ke[:, 0], ke[:, 1], c='blue', edgecolors='black', s=60, label='Benign Keys', zorder=10)
axs[1].scatter(centers[:, 0], centers[:, 1], facecolors='none', edgecolors='black', marker='D', s=200, zorder=12)
style(axs[1], 'K-Means Clustering of Query Vectors and Keys'); style(axs[2], 'Gaussian Query Model')
fig.tight_layout(); fig.savefig(os.path.join(out, 'fig3.pdf')); fig.savefig(os.path.join(out, 'fig3.png'), dpi=150)

# 768-D facts the 2-D panels cannot show
nearest_768 = (F.normalize(Q.double(), dim=1) @ F.normalize(keys, dim=1).t()).argmax(1)
diag['fig3'] = {
    'round': args.round, 'n_keys': len(keys), 'distinct_keys_cos>0.99': n_distinct(keys),
    'key_cos_offdiag_max': cos_offdiag(keys).max().item(), 'key_cos_offdiag_mean': cos_offdiag(keys).mean().item(),
    'queries_per_nearest_key_768d': np.bincount(nearest_768.numpy(), minlength=len(keys)).tolist(),
    'keys_never_nearest_768d': int((np.bincount(nearest_768.numpy(), minlength=len(keys)) == 0).sum()),
}

# ---------------- Fig. 18: the same keys/queries view across training rounds
rounds18 = [int(r) for r in args.rounds18.split(',')]
fig, axs = plt.subplots(1, len(rounds18), figsize=(6 * len(rounds18), 5))
diag['fig18'] = {}
for ax, r in zip(axs, rounds18):
    k = keys_by_round[r]
    ke, qe = joint_tsne(k)
    ax.scatter(qe[:, 0], qe[:, 1], c='darkgreen', s=20, label='Query Vectors')
    ax.scatter(ke[:, 0], ke[:, 1], c='blue', alpha=0.5, s=40, label='Benign Keys', zorder=5)
    style(ax, f't-SNE of Benign Keys vs. Query Vectors (Round {r})')
    diag['fig18'][r] = {'distinct_keys_cos>0.99': n_distinct(k), 'key_cos_offdiag_max': cos_offdiag(k).max().item()}
fig.tight_layout(); fig.savefig(os.path.join(out, 'fig18.pdf')); fig.savefig(os.path.join(out, 'fig18.png'), dpi=150)

json.dump(diag, open(os.path.join(out, 'diagnostics.json'), 'w'), indent=1)
print(json.dumps(diag, indent=1))
