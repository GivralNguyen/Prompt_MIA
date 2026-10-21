"""Fig. 4 data for one (dataset, backbone): PromptMIA and naive injection, TPR / FPR / Advantage / ASR vs batch size.

1. q(x) of every train and test image with the frozen backbone (computed once, cached in cache/queries/).
2. The global keys after FedAvg training (App. A.6 settings), simulated from the queries alone:
   the keys' only gradient is the key-alignment term of Eq. 2, so this equals full training's keys
   (tests/check_fig4.py checks it against a full run).
3. The AMI game (Sec. 3.1) for every test image as target, at each batch size, `--reps` times.
Writes runs/fig4/<dataset>_<backbone>/{config.json, keys.pt, attack.json}.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from promptmia.data import load_dataset, dataset_queries, client_split
from promptmia.backbones import QueryModel
from promptmia.model import PromptPool
from promptmia.fl import run_fedavg_keys
from promptmia.attack import security_game

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'tinyimagenet', 'fourdataset'])
p.add_argument('--backbone', required=True, choices=['vit_b32', 'deit_b16', 'convit_base'])
p.add_argument('--clients', type=int, default=80)
p.add_argument('--clients_per_round', type=int, default=10)
p.add_argument('--rounds', type=int, default=60)
p.add_argument('--local_epochs', type=int, default=5)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--key_lr', type=float, default=1e-3)
p.add_argument('--lam', type=float, default=0.5)
p.add_argument('--alpha', type=float, default=0.5)
p.add_argument('--M', type=int, default=20)
p.add_argument('--N', type=int, default=4)
p.add_argument('--delta_min', type=float, default=0.02)
p.add_argument('--delta', type=float, default=0.05)
p.add_argument('--mia_batch_sizes', default='1,4,8,16,32,64,128,256')
p.add_argument('--reps', type=int, default=5)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--naive_slots', default='last', choices=['last', 'nearest'],
               help='slots naive injection overwrites (paper: unspecified; original code: last)')
p.add_argument('--tag', default='', help='suffix for runs/fig4<tag>/, e.g. a non-default setting')
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = 'cuda'
t0 = time.time()
name = f'{args.dataset}_{args.backbone}'
out = os.path.join(ROOT, 'runs', 'fig4' + args.tag, name)
os.makedirs(out, exist_ok=True)
json.dump(vars(args), open(os.path.join(out, 'config.json'), 'w'), indent=1)

# 1. queries (cached)
qpath = os.path.join(ROOT, 'cache', 'queries', f'{name}.pt')
if os.path.exists(qpath):
    Qd = torch.load(qpath)
else:
    data = load_dataset(args.dataset)
    qm = QueryModel(args.backbone).to(dev)
    src = data['train_source']
    Qd = dict(train=dataset_queries(qm.query, data['train'], dev).cpu(), test=dataset_queries(qm.query, data['test'], dev).cpu(),
              train_labels=torch.as_tensor(data['train_labels']), train_source=None if src is None else torch.as_tensor(src))
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    torch.save(Qd, qpath)
    del qm
train_q, test_q = Qd['train'].to(dev), Qd['test'].to(dev)
print(f'{name}: {len(train_q)} train / {len(test_q)} test queries, {time.time()-t0:.0f}s', flush=True)

# 2. global keys after federated training
src = Qd['train_source']
clients = client_split(dict(train_labels=Qd['train_labels'].numpy(), train_source=None if src is None else src.numpy()),
                       args.clients, args.alpha, args.seed)
torch.manual_seed(args.seed)
key0 = PromptPool(args.M, args.N, train_q.size(1)).key.detach().to(dev)
keys = {}
cfg = dict(rounds=args.rounds, clients_per_round=args.clients_per_round, local_epochs=args.local_epochs,
           batch_size=args.batch_size, lr=args.lr, key_lr=args.key_lr, lam=args.lam, seed=args.seed, N=args.N)
final = run_fedavg_keys(key0, train_q, clients, cfg, on_round=lambda r, k: keys.__setitem__(r, k.cpu()))
torch.save(keys, os.path.join(out, 'keys.pt'))
print(f'keys trained ({args.rounds} rounds), client sizes {min(map(len, clients))}-{max(map(len, clients))}, '
      f'{time.time()-t0:.0f}s', flush=True)

# 3. the AMI game for both attacks
res = {}
for method in ('promptmia', 'naive'):
    res[method] = {}
    for bs in [int(b) for b in args.mia_batch_sizes.split(',')]:
        runs = [security_game(final, train_q, test_q, args.N, bs, args.delta_min, args.delta,
                              seed=10_000 * rep + bs, naive=(method == 'naive'), naive_slots=args.naive_slots) for rep in range(args.reps)]
        res[method][bs] = {k: [r[k] for r in runs] for k in runs[0]}
        m = {k: float(np.mean(v)) for k, v in res[method][bs].items()}
        print(f'{method:9s} B={bs:3d}  TPR {m["tpr"]:.4f}  FPR {m["fpr"]:.4f}  Adv {m["advantage"]:.4f}  ASR {m["asr"]:.4f}', flush=True)
json.dump(res, open(os.path.join(out, 'attack.json'), 'w'), indent=1)
print(f'done in {(time.time()-t0)/60:.1f} min')
