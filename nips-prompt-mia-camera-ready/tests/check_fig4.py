"""Checks behind Fig. 4 (GPU node; needs runs/cifar10_vit_b32 from scripts/train.py):
1. the key-only FedAvg simulation reproduces the keys of full training, every round;
2. the query of each backbone equals the original code's get_cls_keys;
3. on trained keys, every injected key lies in the Eq. 7 interval, the target selects exactly the injected keys
   (training and test targets), and the measured TPR of the game is 1 (Theorem 3.1) for PromptMIA and naive injection.
Also writes cache/queries/cifar10_vit_b32.pt, which scripts/fig4_run.py reuses."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from promptmia.data import load_dataset, dataset_queries, client_split, Normalizer
from promptmia.backbones import QueryModel, BACKBONES
from promptmia.fl import run_fedavg_keys
from promptmia.attack import inject_batch, security_game

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = 'cuda'


def check(name, ok):
    print(f'[{"PASS" if ok else "FAIL"}] {name}', flush=True)
    assert ok, name


data = load_dataset('cifar10')
qpath = os.path.join(ROOT, 'cache', 'queries', 'cifar10_vit_b32.pt')
if os.path.exists(qpath):
    Qd = torch.load(qpath)
else:
    qm = QueryModel('vit_b32').to(dev)
    Qd = dict(train=dataset_queries(qm.query, data['train'], dev).cpu(), test=dataset_queries(qm.query, data['test'], dev).cpu(),
              train_labels=torch.as_tensor(data['train_labels']), train_source=None)
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    torch.save(Qd, qpath)
train_q, test_q = Qd['train'].to(dev), Qd['test'].to(dev)

# 1. key-only simulation == full training (scripts/train.py run with the same settings)
run = os.path.join(ROOT, 'runs', 'cifar10_vit_b32')
cfg = json.load(open(os.path.join(run, 'config.json')))
full = torch.load(os.path.join(run, 'keys.pt'))
clients = client_split(dict(train_labels=data['train_labels'], train_source=None), cfg['clients'], cfg['alpha'], cfg['seed'])
sim = {}
run_fedavg_keys(full[0].to(dev), train_q, clients, cfg, on_round=lambda r, k: sim.__setitem__(r, k.cpu()))
diff = max((sim[r] - full[r]).abs().max().item() for r in full)
exact = sum(torch.equal(sim[r], full[r]) for r in full)
print(f'    max |key diff| over {len(full)} rounds: {diff:.3e}; bit-identical rounds: {exact}/{len(full)}')
check('key-only FedAvg reproduces full-training keys (all rounds, |diff| < 1e-6)', diff < 1e-6)

# 2. queries == original code's get_cls_keys
sys.path.insert(0, os.path.join(ROOT, '..'))
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_DeiT_B16, L2P_ConViT_Base
x = Normalizer(dev)(torch.stack([data['test'][i][0] for i in range(16)]).to(dev))
old = {'vit_b32': L2P_ViT_B32, 'deit_b16': L2P_DeiT_B16, 'convit_base': L2P_ConViT_Base}
for b in BACKBONES:
    ref = old[b]('shallow', False, pool_size=20, top_k=4, num_classes=10).to(dev).eval()
    with torch.no_grad():
        d = (QueryModel(b).to(dev).query(x) - ref.get_cls_keys(x)).abs().max().item()
    check(f'{b}: query == original get_cls_keys (max |diff| {d:.2e})', d < 1e-4)
    del ref

# 3. the attack on trained keys, for TPR targets (training images) and FPR targets (test images)
keys = full[cfg['rounds']].to(dev)
gen = torch.Generator().manual_seed(0)
for split, tq in (('train', train_q[:2000]), ('test', test_q[:2000])):
    pools, S = inject_batch(keys, tq, cfg['N'], 0.02, 0.05, gen)
    qn = F.normalize(tq.double(), dim=1)
    s = (F.normalize(pools.double(), dim=2) @ qn[:, :, None]).squeeze(-1)                # (T, M) similarity to q(T)
    s_adv = s.gather(1, S)
    s_max = F.normalize(keys.double(), dim=1) @ qn.t()                                   # benign max from unmodified keys
    s_max = s_max.t().scatter(1, S, float('-inf')).max(1).values
    in_interval = ((s_adv >= (s_max + 0.02)[:, None] - 1e-6) & (s_adv <= (s_max + 0.07)[:, None] + 1e-6)).all().item()
    check(f'injected keys lie in [s_max + delta_min, s_max + delta_min + Delta] computed from q(T) (2000 {split} targets)', in_interval)
    check(f'each target selects exactly its injected keys (2000 {split} targets)',
          torch.equal(s.topk(cfg['N'], dim=1).indices.sort(1).values, S.sort(1).values))
for naive in (False, True):
    for bs in (1, 256):
        r = security_game(keys, train_q, test_q[:2000], cfg['N'], bs, 0.02, 0.05, seed=1, naive=naive,
                          member_idx=torch.arange(2000, device=dev))
        check(f'{"naive" if naive else "PromptMIA"} B={bs}: measured TPR == 1 on training targets (FPR {r["fpr"]:.4f})', r['tpr'] == 1.0)
print('all checks passed')
