"""Federated prompt tuning with FedAvg (paper Sec. 2, App. A.1).

One model on the GPU holds the frozen backbone. A client is just a copy of the
trainable tensors: it loads the global state, trains locally, and returns clones.
Clients therefore never share storage with each other or with the server.
"""
import numpy as np
import torch
import torch.nn.functional as F


def local_train(model, cache, queries, idx, epochs, batch_size, lr, lam, gen, key_lr):
    """One client's local update over its sample indices `idx`; updates the model in place.
    lr: prompts and classifier; key_lr: keys. The keys' only gradient is lam * key_loss, and Adam is
    invariant to gradient scale, so key_lr (not lam) sets how fast keys move."""
    model.train()
    keys = [model.pool.key]
    others = [p for n, p in model.named_parameters() if p.requires_grad and n != 'pool.key']
    opt = torch.optim.Adam([{'params': others}, {'params': keys, 'lr': key_lr}],
                           lr=lr, betas=(0.9, 0.98), eps=1e-6)
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


def local_train_keys(key, queries, idx, epochs, batch_size, lr, lam, gen, N):
    """Exactly the keys' part of local_train. The keys' only gradient in Eq. 2 is lam * key-alignment loss,
    which depends on the (frozen) queries and the keys alone, so prompts, classifier and images are not needed."""
    key = key.clone().requires_grad_(True)
    opt = torch.optim.Adam([key], lr=lr, betas=(0.9, 0.98), eps=1e-6)
    idx = torch.as_tensor(idx, device=queries.device)
    for _ in range(epochs):
        order = idx[torch.randperm(len(idx), generator=gen).to(idx.device)]
        for ids in order.split(batch_size):
            top_sim = (F.normalize(queries[ids], dim=1) @ F.normalize(key, dim=1).t()).topk(N, dim=1).values
            loss = lam * (1.0 - top_sim).sum(dim=1).mean()               # same expression as PromptPool.select
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return key.detach()


def run_fedavg_keys(key0, train_q, client_idx, cfg, on_round=None):
    """run_fedavg restricted to the keys: same client sampling, shuffling and aggregation, so it yields the
    same global keys as full training (checked in tests/check_keys_only.py). on_round(r, key)."""
    rng = np.random.default_rng(cfg['seed'])
    gen = torch.Generator().manual_seed(cfg['seed'])
    key = key0.clone()
    if on_round:
        on_round(0, key)
    sizes = np.array([len(c) for c in client_idx])
    for r in range(1, cfg['rounds'] + 1):
        chosen = np.sort(rng.choice(len(client_idx), cfg['clients_per_round'], replace=False))
        states = [{'pool.key': local_train_keys(key, train_q, client_idx[c], cfg['local_epochs'], cfg['batch_size'],
                                                cfg['key_lr'], cfg['lam'], gen, cfg['N'])}
                  for c in chosen]
        key = fedavg(states, sizes[chosen])['pool.key']
        if on_round:
            on_round(r, key)
    return key


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
    """cfg: rounds, clients_per_round, local_epochs, batch_size, lr, key_lr, lam, seed.
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
                        cfg['batch_size'], cfg['lr'], cfg['lam'], gen, cfg['key_lr'])
            states.append(model.trainable_state())
        global_state = fedavg(states, sizes[chosen])
        model.load_trainable_state(global_state)
        acc = evaluate(model, test_cache, test_q)
        print(f'round {r:3d}  clients {chosen.tolist()}  test acc {acc:.4f}', flush=True)
        if on_round:
            on_round(r, global_state, acc)
    return global_state
