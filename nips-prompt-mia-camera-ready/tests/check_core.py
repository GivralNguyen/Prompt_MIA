"""Checks for the properties the original code got wrong (run on a GPU node: python tests/check_core.py)."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from promptmia.data import ImageCache, NORM_STATS, dirichlet_split
from promptmia.model import L2PViT
from promptmia.fl import local_train
from promptmia.attack import gen_key_with_similarity, promptmia_inject

dev = 'cuda'
torch.manual_seed(0)


def check(name, ok):
    print(f'[{"PASS" if ok else "FAIL"}] {name}')
    assert ok, name


# 1. cached pixels == ToTensor+Normalize pipeline, bit for bit
rng = np.random.default_rng(0)
pils = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(8)]
ref = torch.stack([T.Compose([T.ToTensor(), T.Normalize(*NORM_STATS)])(im) for im in pils])
cache = ImageCache([(T.PILToTensor()(im), 0) for im in pils], dev, batch_size=4)
check('image cache is bit-identical to ToTensor+Normalize', torch.equal(cache.get(torch.arange(8, device=dev))[0].cpu(), ref))

# 2. Dirichlet split is a partition with the minimum client size
labels = rng.integers(0, 10, 5000)
parts = dirichlet_split(labels, 20, 0.5, seed=0)
allidx = np.concatenate(parts)
check('client split covers every sample exactly once', len(allidx) == 5000 and len(np.unique(allidx)) == 5000)
check('every client has >= 32 samples', min(map(len, parts)) >= 32)

# 3. precomputed q(x) gives the same forward pass as computing it inside
model = L2PViT(10).to(dev).eval()
x = cache.get(torch.arange(8, device=dev))[0]
with torch.no_grad():
    a, _, ia = model(x)
    b, _, ib = model(x, model.query(x))
check('forward with cached query == forward without', torch.equal(a, b) and torch.equal(ia, ib))

# 4. clients are independent: local training never touches the global state or another client's state
global_state = model.trainable_state()
model.load_trainable_state(global_state)
local_train(model, cache, model.query(x), np.arange(8), epochs=1, batch_size=4, lr=1e-2, lam=0.5,
            gen=torch.Generator().manual_seed(0), key_lr=1e-2)
client_a = model.trainable_state()
model.load_trainable_state(global_state)
client_b = model.trainable_state()
check('local training changed the client', any(not torch.equal(client_a[n], global_state[n]) for n in global_state))
check('global state untouched by client training', all(torch.equal(client_b[n], global_state[n]) for n in global_state))
check('client states never alias each other', all(client_a[n].data_ptr() != client_b[n].data_ptr() for n in global_state))

# 4b. the attack's observation channel: prompt/key rows never selected during local training come back unchanged,
#     rows that were selected come back changed
model.load_trainable_state(global_state)
seen, select = set(), model.pool.select
def recording_select(q):
    idx, loss = select(q)
    seen.update(idx.flatten().tolist())
    return idx, loss
model.pool.select = recording_select
local_train(model, cache, model.query(x), np.arange(8), epochs=1, batch_size=4, lr=1e-2, lam=0.5,
            gen=torch.Generator().manual_seed(0), key_lr=1e-2)
model.pool.select = select
after = model.trainable_state()
changed = {i for i in range(global_state['pool.prompt'].size(0))
           if not torch.equal(after['pool.prompt'][i], global_state['pool.prompt'][i])
           or not torch.equal(after['pool.key'][i], global_state['pool.key'][i])}
check(f'only selected rows change (selected {sorted(seen)}, changed {sorted(changed)})', changed == seen)

# 5. Algorithm 1 hits the requested similarity and norm
gen = torch.Generator().manual_seed(0)
q = torch.randn(768, dtype=torch.float64)
k = gen_key_with_similarity(q, 0.87, gen)
check('Alg. 1 cosine == s', abs(F.cosine_similarity(k, q, dim=0).item() - 0.87) < 1e-9)
check('Alg. 1 norm == ||q||', abs(k.norm().item() - q.norm().item()) < 1e-9)

# 6. Algorithm 2 / Theorem 3.1: the target always selects exactly the injected keys (TPR = 1),
#    and every injected key lies in the Eq. 7 interval computed from q(T)
fails = 0
for t in range(500):
    keys = torch.randn(20, 768, dtype=torch.float64)
    qt = keys[t % 20] + 0.5 * torch.randn(768, dtype=torch.float64)  # a query near some key, like trained pools
    new, S, (lo, hi) = promptmia_inject(keys, qt, 4, 0.02, 0.05, gen)
    sim = F.normalize(new, dim=1) @ F.normalize(qt, dim=0)
    in_interval = bool(((sim[S] >= lo - 1e-9) & (sim[S] <= hi + 1e-9)).all())
    selected = set(sim.topk(4).indices.tolist()) == set(S.tolist())
    fails += not (in_interval and selected)
check('Alg. 2: injected keys in the Eq. 7 interval and selected by q(T) in 500/500 trials', fails == 0)
print('all checks passed')
