# PromptMIA — camera-ready reproduction

Minimal, from-scratch code for the paper's federated prompt tuning setup and attack.
Currently covers **Fig. 2, Fig. 3 and Fig. 18** (ViT-B/32, CIFAR-10).

```
promptmia/data.py     CIFAR-10, Dirichlet split, GPU image cache, frozen-query cache
promptmia/model.py    L2P prompt pool on frozen ViT-B/32 (Eq. 1-2)
promptmia/fl.py       FedAvg over clients (App. A.1, A.6)
promptmia/attack.py   Algorithms 1-2 (Eq. 7), naive injection (Eq. 5-6)
scripts/train.py      training; saves keys every round + checkpoints
scripts/figures.py    Fig. 2, 3, 18 + diagnostics.json
tests/check_core.py   checks for the properties the original code got wrong
slurm/fig2_3_18.sbatch  checks -> train -> figures
```

Run (from this folder): `mkdir -p logs && sbatch slurm/fig2_3_18.sbatch`
Outputs: `runs/cifar10_vit_b32/{metrics.json,keys.pt,ckpt_round*.pt,figures/}`.
Datasets are read from `../Dataset`.

## Settings (App. A.6)
80 clients, 10 per round, Dirichlet alpha = 0.5, 60 rounds, 5 local epochs, batch 64,
Adam lr 1e-4 (betas 0.9/0.98, eps 1e-6 as in the original code), lambda = 0.5, M = 20, N = 4,
delta_min = 0.02, Delta = 0.05.

## Results so far (seed 0)
| | test acc, round 60 | paper (Table 5) |
|---|---|---|
| `--head cls` (default) | 0.935 | 0.95 |
| `--head prompt` | 0.857 | |

The keys are trained only by the key-alignment term of Eq. 2, so they are identical under
both heads: Fig. 2, 3, 18 and every attack result do not depend on the head.
Distinct keys (pairwise cosine > 0.99) at rounds 0 / 1 / 20 / 60: 20 / 20 / 18 / 11.

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
| injection slots S | last N keys (naive) / top-N of a random query (PromptMIA) | top-N keys of q(T) for both |
