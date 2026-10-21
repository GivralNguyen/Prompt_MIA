# PromptMIA — camera-ready reproduction

Minimal, from-scratch code for the paper's federated prompt tuning setup and attack.
Currently covers **Fig. 2, Fig. 3, Fig. 18** (ViT-B/32, CIFAR-10) and **Fig. 4** (4 datasets x 3 backbones).

```
promptmia/data.py     datasets, Dirichlet split, GPU image cache, frozen-query cache
promptmia/backbones.py  frozen query models: ViT-B/32, DeiT-B/16, ConViT-Base
promptmia/model.py    L2P prompt pool on frozen ViT-B/32 (Eq. 1-2)
promptmia/fl.py       FedAvg over clients (App. A.1, A.6)
promptmia/attack.py   Algorithms 1-2 (Eq. 7), naive injection (Eq. 5-6), batched AMI security game
scripts/train.py      training; saves keys every round + checkpoints
scripts/figures.py    Fig. 2, 3, 18 + diagnostics.json
scripts/fig4_run.py   Fig. 4 data for one (dataset, backbone)
scripts/fig4_plot.py  Fig. 4 + figures/fig4_results.json
tests/check_core.py   checks for the properties the original code got wrong
tests/check_fig4.py   key-only training == full training; queries == original code; measured TPR == 1
slurm/fig2_3_18.sbatch  checks -> train -> figures
slurm/fig4_submit.sh    checks -> 12 runs (5 GPUs) -> plot
```

Run (from this folder): `mkdir -p logs && sbatch slurm/fig2_3_18.sbatch`
Outputs: `runs/cifar10_vit_b32/{metrics.json,keys.pt,ckpt_round*.pt,figures/}`.
Datasets are read from `../Dataset`.

Fig. 4: `bash slurm/fig4_submit.sh` (about 2 min per run once queries are cached).
Outputs: `runs/fig4/<dataset>_<backbone>/{keys.pt,attack.json}`, `figures/fig4.{pdf,png}`, `figures/fig4_results.json`.

## Settings (App. A.6)
80 clients, 10 per round, Dirichlet alpha = 0.5, 60 rounds, 5 local epochs, batch 64,
Adam lr 1e-4 for prompts and classifier, **1e-3 for keys** (betas 0.9/0.98, eps 1e-6 as in the original code),
lambda = 0.5, M = 20, N = 4, delta_min = 0.02, Delta = 0.05.

Key learning rate: the keys' only gradient is lambda * key-alignment loss, and Adam ignores gradient scale, so
lambda barely matters (lambda = 5 gives the same keys as 0.5) and the key lr sets how far keys converge in 60
rounds. Seed 0 sweep, distinct keys (pairwise cosine > 0.99) at rounds 1/20/60: 20/18/11 for 1e-4,
20/8/5 for 3e-4, 20/5/5 for 1e-3, 3e-3 and 1e-2 (all converge to the same 5 positions, one per query cluster).
Over seeds 0-4, 3e-4 ends at 5-8 distinct keys; 1e-3 ends at 5 on every seed (merging during rounds 2-15).
A.6 must state the key lr.

## Results so far
| | test acc, round 60 | paper (Table 5) |
|---|---|---|
| defaults (`--head cls`, key lr 1e-3), seeds 0-4 | 0.933 +- 0.002 | 0.95 +- 0.01 |
| `--head prompt` (key lr 1e-4, seed 0) | 0.857 | |

The keys are trained only by the key-alignment term of Eq. 2, so they are identical under
both heads: Fig. 2, 3, 18 and every attack result do not depend on the head.
With the defaults, every seed ends with 5 distinct keys and the target selects exactly the injected keys.

## Fig. 4 (key lr 1e-3, 5 game repeats, averaged over 3 backbones)
Game (as in the original code): TPR over every training image T, the client's batch being T plus B-1 other random
training images; FPR over every test image T, the batch being B random training images. Member = the union of the
batch's top-N selections contains every injected key.

Advantage at B = 256 (paper values read off the figure):

| | PromptMIA (ours / paper) | naive (ours / paper) |
|---|---|---|
| CIFAR-10 | 0.89 / ~0.96 | 0.39 / ~0.41 |
| CIFAR-100 | 0.94 / ~0.97 | 0.45 / ~0.45 |
| TinyImageNet | 0.95 / ~0.97 | 0.46 / ~0.46 |
| FourDataset | 0.93 / ~0.90 | 0.17 / ~0.23 |

Measured TPR = 1 in every run. Key lr 1e-4 gives the same picture (`figures/fig4_keylr1e-4.png`).
Naive's slot choice matters a lot: the paper does not specify S, and overwriting the N keys nearest q(T)
removes T's neighbouring benign keys, dropping naive to 0.05-0.15 at B = 256 (`figures/fig4_naive-nearest.png`).

## Differences from the original code
| | original code | here |
|---|---|---|
| Eq. 7 threshold | from a random training sample's query | from q(T) (Algorithm 2) |
| client state after aggregation | server and all clients share one tensor, so clients train sequentially | each client works on cloned tensors (checked) |
| classifier (Eq. 2) | CLS -> frozen ImageNet head -> MLP | same (`--head cls`, default); Eq. 2's prompt-token head via `--head prompt` |
| FedAvg weights | prompts/keys averaged uniformly, head by client size | every trainable tensor weighted by client size |
| per-step cost | PIL resize every sample every epoch; q(x) recomputed every step | images cached once on GPU (bit-identical); q(x) computed once |
| client models | 10 deep copies of ViT-B/32 | one shared frozen backbone |
| TF32 | cuDNN TF32 on by default (batch-size-dependent queries) | off unless `--tf32` |
| seeds | unseeded client sampling | everything seeded (`--seed`) |
| injection slots S | last N keys (naive) / top-N of a random query (PromptMIA) | PromptMIA: top-N keys of q(T); naive: last N as in the original (`--naive_slots nearest` for top-N of q(T)) |
| MIA game | PromptMIA TPR hardcoded to 1; member batches in dataset order; non-member fillers drawn with replacement, shared within a test batch | TPR and FPR measured; same target sets (train for TPR, test for FPR); fillers drawn per target without replacement; batched on GPU |
| Fig. 4 training | full FL training per model | keys only (the attack depends only on the keys; identical keys, checked) |
