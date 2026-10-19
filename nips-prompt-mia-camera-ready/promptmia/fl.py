"""Federated prompt tuning with FedAvg (paper Sec. 2, App. A.1).

One model on the GPU holds the frozen backbone. A client is just a copy of the
trainable tensors: it loads the global state, trains locally, and returns clones.
Clients therefore never share storage with each other or with the server.
"""
import numpy as np
import torch
import torch.nn.functional as F


def local_train(model, cache, queries, idx, epochs, batch_size, lr, lam, gen):
    """One client's local update over its sample indices `idx`; updates the model in place."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-6)
    idx = torch.as_tensor(idx, device=cache.images.device)
    for _ in range(epochs):
        order = idx[torch.randperm(len(idx), generator=gen).to(idx.device)]
        for ids in order.split(batch_size):
            x, y = cache.get(ids)
            logits, key_loss, _ = model(x, queries[ids])
            loss = F.cross_entropy(logits, y) + lam * key_loss          # Eq. 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def fedavg(states, weights):
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    return {n: sum(wi * s[n] for wi, s in zip(w.tolist(), states)) for n in states[0]}


@torch.no_grad()
def evaluate(model, cache, queries, batch_size=500):
    model.eval()
    correct = 0
    for ids in torch.arange(len(cache), device=cache.images.device).split(batch_size):
        x, y = cache.get(ids)
        logits, _, _ = model(x, queries[ids])
        correct += (logits.argmax(1) == y).sum().item()
    return correct / len(cache)


def run_fedavg(model, train_cache, train_q, test_cache, test_q, client_idx, cfg, on_round=None):
    """cfg: rounds, clients_per_round, local_epochs, batch_size, lr, lam, seed.
    on_round(r, global_state, acc) is called after initialisation (r=0) and after every round."""
    rng = np.random.default_rng(cfg['seed'])
    gen = torch.Generator().manual_seed(cfg['seed'])
    global_state = model.trainable_state()
    if on_round:
        on_round(0, global_state, None)
    sizes = np.array([len(c) for c in client_idx])
    for r in range(1, cfg['rounds'] + 1):
        chosen = np.sort(rng.choice(len(client_idx), cfg['clients_per_round'], replace=False))
        states = []
        for c in chosen:
            model.load_trainable_state(global_state)
            local_train(model, train_cache, train_q, client_idx[c], cfg['local_epochs'],
                        cfg['batch_size'], cfg['lr'], cfg['lam'], gen)
            states.append(model.trainable_state())
        global_state = fedavg(states, sizes[chosen])
        model.load_trainable_state(global_state)
        acc = evaluate(model, test_cache, test_q)
        print(f'round {r:3d}  clients {chosen.tolist()}  test acc {acc:.4f}', flush=True)
        if on_round:
            on_round(r, global_state, acc)
    return global_state
