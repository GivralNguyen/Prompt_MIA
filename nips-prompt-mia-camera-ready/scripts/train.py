"""Federated prompt tuning of ViT-B/32 on CIFAR-10 (defaults = paper App. A.6).

Writes to runs/<name>/: config.json, metrics.json, keys.pt (global keys after every
round, round 0 = initialisation) and ckpt_round<r>.pt for each --save_rounds.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from promptmia.data import load_cifar10, ImageCache, compute_queries, dirichlet_split
from promptmia.model import L2PViT
from promptmia.fl import run_fedavg

p = argparse.ArgumentParser()
p.add_argument('--name', default='cifar10_vit_b32')
p.add_argument('--clients', type=int, default=80)
p.add_argument('--clients_per_round', type=int, default=10)
p.add_argument('--rounds', type=int, default=60)
p.add_argument('--local_epochs', type=int, default=5)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--lam', type=float, default=0.5, help='lambda in Eq. 2')
p.add_argument('--alpha', type=float, default=0.5, help='Dirichlet concentration')
p.add_argument('--M', type=int, default=20)
p.add_argument('--N', type=int, default=4)
p.add_argument('--head', default='cls', choices=['cls', 'prompt'],
               help='cls: CLS -> ImageNet head -> MLP (paper accuracy); prompt: mean of prompt tokens (Eq. 2 wording)')
p.add_argument('--save_rounds', default='0,1,20,60')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--tf32', action='store_true', help='allow TF32 matmul/conv (faster, slightly less precise)')
p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = args.tf32
torch.backends.cudnn.allow_tf32 = args.tf32
dev = 'cuda'
out = os.path.join(args.out, args.name)
os.makedirs(out, exist_ok=True)
json.dump(vars(args), open(os.path.join(out, 'config.json'), 'w'), indent=1)

t0 = time.time()
train_set, test_set, n_classes = load_cifar10()
train_cache, test_cache = ImageCache(train_set, dev), ImageCache(test_set, dev)
model = L2PViT(n_classes, args.M, args.N, args.head).to(dev)
train_q, test_q = compute_queries(model, train_cache), compute_queries(model, test_cache)
clients = dirichlet_split(train_set.targets, args.clients, args.alpha, args.seed)
print(f'setup {time.time()-t0:.0f}s | client sizes min {min(map(len, clients))} max {max(map(len, clients))}', flush=True)

save_rounds = {int(r) for r in args.save_rounds.split(',')}
keys, accs = {}, {}

def on_round(r, state, acc):
    keys[r] = state['pool.key'].cpu()
    accs[r] = acc
    torch.save(keys, os.path.join(out, 'keys.pt'))
    json.dump({'test_acc': accs, 'elapsed_s': time.time() - t0}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    if r in save_rounds:
        torch.save({n: v.cpu() for n, v in state.items()}, os.path.join(out, f'ckpt_round{r}.pt'))

cfg = dict(rounds=args.rounds, clients_per_round=args.clients_per_round, local_epochs=args.local_epochs,
           batch_size=args.batch_size, lr=args.lr, lam=args.lam, seed=args.seed)
run_fedavg(model, train_cache, train_q, test_cache, test_q, clients, cfg, on_round)
print(f'done in {(time.time()-t0)/60:.1f} min, final acc {accs[args.rounds]:.4f}')
