# PromptMIA

Membership Inference Attack (MIA) on federated learning via prompt injection, targeting **Learning to Prompt (L2P)**-based federated models. The attack exploits the prompt pool key-query matching mechanism to infer whether a sample was used in a client's training data, without modifying model weights.

---

## Step 1: Set Up Datasets

Place datasets under `./Dataset/`. See [Dataset/readme.md](Dataset/readme.md) for the expected folder structure.

Supported datasets:

| Argument value | Dataset | Notes |
|---|---|---|
| `cifar10` | CIFAR-10 | Auto-downloaded to `./Dataset/cifar-10-batches-py/` |
| `cifar100` | CIFAR-100 | Auto-downloaded to `./Dataset/cifar-100-python/` |
| `tinyimagenet` | Tiny ImageNet | Place under `./Dataset/tiny-imagenet-200/` |
| `fourdataset` | MNIST-M + FashionMNIST + CINIC-10 + MMAFEDB | Place each under `./Dataset/` |
| `PACS` | PACS | Place under `./Dataset/PACS/<domain>/` |

---

## Step 2: Train the Federated Model

`train.py` runs federated training and saves the global model checkpoint.

```bash
python train.py \
  --gpus 0 \
  --alg fedavg \
  --dataset cifar10 \
  --device cuda \
  --batch 64 \
  --comms 60 \
  --local_eps 5 \
  --lr 1e-4 \
  --n_clients 80 \
  --n_sampled_clients 10 \
  --data_distribution non_iid_dirichlet \
  --alpha 0.5 \
  --n_tokens 4 \
  --pool_size 20 \
  --save_model
```

`--save_model` saves a checkpoint at the final communication round under `./output_record/saved_model/`.

**Algorithms** (`--alg`): `fedavg`, `fedavg_nonpara`

**Data heterogeneity settings:**

- Dirichlet non-IID (default): `--data_distribution non_iid_dirichlet --alpha 0.5`
- Extreme heterogeneity: `--data_distribution manual_extreme_heterogeneity --n_dominated_class 1 --dominated_ratio 0.99`

**Training-time defenses** (`--defense`):

| Defense | Arguments |
|---|---|
| Mixup | `--defense mix_up --mixup_alpha 0.2` |
| Gradient sparsification | `--defense sparse --sparse_ratio 0.8` |
| Gaussian gradient perturbation | `--defense perturb --perturb_sigma 0.01` |
| Data augmentation | `--data_augment` |

---

## Step 3: Run Membership Inference Attacks

All attack scripts require `--load_model` pointing to a saved checkpoint. Match `--pool_size` and `--n_tokens` to the trained model's configuration.

### PromptMIA (proposed attack)

`mia_stealth.py` injects adversarial prompt keys crafted to match only the target sample's query embedding, enabling stealthy membership inference.

```bash
python mia_stealth.py \
  --gpus 0 \
  --dataset cifar10 \
  --n_clients 80 \
  --n_sampled_clients 10 \
  --data_distribution non_iid_dirichlet \
  --alpha 0.5 \
  --pool_size 20 \
  --n_tokens 4 \
  --query_model_type vit_cls \
  --delta_min 0.02 \
  --delta_max 0.05 \
  --load_model <path_to_saved_model>
```

Key attack parameters:
- `--delta_min` / `--delta_max`: cosine similarity range for adversarial key generation
- `--query_model_type`: query encoder backbone — `vit_cls`, `deit_b16`, `convit_base`
- `--beta`: direction control for adversarial key (default `0`; do not change unless you understand the geometry)

### PromptMIA under Input Perturbation Defense

`mia_stealth_noise.py` evaluates PromptMIA when the server applies input-level noise as a defense.

```bash
python mia_stealth_noise.py \
  --gpus 0 \
  --dataset cifar10 \
  --pool_size 20 \
  --n_tokens 4 \
  --query_model_type vit_cls \
  --delta_min 0.02 \
  --delta_max 0.05 \
  --sigma 0.5 \
  --load_model <path_to_saved_model>
```

### Naive Prompt Injection (baseline)

`mia_naive.py` performs a simpler prompt injection attack without adversarial key crafting.

```bash
python mia_naive.py \
  --gpus 0 \
  --dataset cifar10 \
  --pool_size 20 \
  --n_tokens 4 \
  --load_model <path_to_saved_model>
```

### Loss-Based MIA Baselines

`mia_loss_based.py` implements three loss-based baselines for comparison:
- **Blackbox-Loss**: single-round loss threshold (Yeom et al., 2018)
- **Loss-Series**: multi-round averaged loss (Gu et al., 2022)
- **FedMIA-I**: multi-round, multi-client CDF-based attack

```bash
python mia_loss_based.py \
  --dataset cifar10 \
  --comms 60 \
  --n_clients 80 \
  --n_sampled_clients 10 \
  --data_distribution non_iid_dirichlet \
  --alpha 0.5 \
  --gpus 0 \
  --target_client 0 \
  --mia_record_interval 1
```

---

## Theoretical FPR Bound

`mia_bound_theory.py` computes the per-sample false positive rate (FPR) upper bound from Theorem 3.3 of the paper.

```bash
python mia_bound_theory.py \
  --gpus 0 \
  --dataset cifar10 \
  --query_model_type vit_cls \
  --pool_size 20 \
  --n_tokens 4 \
  --delta_min 0.02 \
  --delta_max 0.05 \
  --n_targets 100 \
  --max_samples 50000 \
  --load_model <path_to_saved_model>
```

> **Note:** `--pool_size` must match the trained model's configuration exactly.

---

## Visualization

| Notebook | Description |
|---|---|
| `plot_embedding.ipynb` | Visualize the Gaussian Mixture query model and prompt pool key clusters |
| `plot_outlier.ipynb` | Visualize and measure the performance of outlier detection defenses |

---

## Output

- Trained model checkpoints: `./output_record/saved_model/`
- Attack results and logs: `./result_log/`

