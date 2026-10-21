"""Adversarial key construction (paper Sec. 3.2, Eq. 5-7, Algorithms 1-2)."""
import torch
import torch.nn.functional as F


def gen_key_with_similarity(q, s, gen):
    """Algorithm 1: a key with cosine similarity s to q and norm ||q||."""
    q_hat = F.normalize(q, dim=0)
    r = F.normalize(torch.rand(q.numel(), generator=gen, dtype=q.dtype).to(q.device), dim=0)
    o = F.normalize(r - (r @ q_hat) * q_hat, dim=0)
    return (s * q_hat + (1.0 - s * s) ** 0.5 * o) * q.norm()


def injection_slots(keys, q, N):
    """The subset S of pool indices to overwrite: the N keys closest to q(T). The rest are K_BENIGN."""
    return (F.normalize(keys, dim=1) @ F.normalize(q, dim=0)).topk(N).indices


def gen_adv_key_set(q, benign_keys, N, delta_min, delta, gen):
    """Algorithm 2: N keys whose similarity to q(T) lies in [s_max + delta_min, s_max + delta_min + delta] (Eq. 7),
    where s_max is computed from q(T) itself against the benign keys."""
    s_max = (F.normalize(benign_keys, dim=1) @ F.normalize(q, dim=0)).max().item()
    lo, hi = s_max + delta_min, s_max + delta_min + delta
    if hi > 1.0:
        raise ValueError(f'interval [{lo:.4f}, {hi:.4f}] exceeds cosine 1; reduce delta_min/delta')
    s = lo + (hi - lo) * torch.rand(N, generator=gen, dtype=torch.float64)
    return torch.stack([gen_key_with_similarity(q, si.item(), gen) for si in s]), (lo, hi)


def promptmia_inject(keys, q, N, delta_min, delta, gen):
    """Returns (modified keys, injected slot indices, Eq. 7 interval)."""
    S = injection_slots(keys, q, N)
    mask = torch.ones(keys.size(0), dtype=torch.bool, device=keys.device)
    mask[S] = False
    adv, interval = gen_adv_key_set(q.double(), keys[mask].double(), N, delta_min, delta, gen)
    out = keys.clone()
    out[S] = adv.to(keys.dtype)
    return out, S, interval


def naive_slots_for(keys, q, N, slots):
    """S for naive injection; the paper leaves it open. 'last': the last N slots, as in the original code
    (T's own nearby benign keys stay in the pool). 'nearest': the N keys nearest q(T), as PromptMIA uses."""
    if slots == 'last':
        return torch.arange(keys.size(0) - N, keys.size(0), device=keys.device)
    return injection_slots(keys, q, N)


def naive_inject(keys, q, N, slots='last'):
    """Eq. 5-6: every injected key equals q(T)."""
    S = naive_slots_for(keys, q, N, slots)
    out = keys.clone()
    out[S] = q.to(keys.dtype)
    return out, S


# ---------------------------------------------------------------- batched over many targets (Fig. 4)
def inject_batch(keys, q, N, delta_min, delta, gen, naive=False, naive_slots='last'):
    """The poisoned pool for each target. keys (M, D), q (T, D) -> pools (T, M, D), injected slots S (T, N).
    Same construction as promptmia_inject / naive_inject, computed in float64."""
    keys64, q64 = keys.double(), q.double()
    qn = F.normalize(q64, dim=1)
    sim = qn @ F.normalize(keys64, dim=1).t()                               # (T, M)
    S = sim.topk(N, dim=1).indices                                          # (T, N)
    T_, M, D = q.size(0), keys.size(0), keys.size(1)
    if naive:
        if naive_slots == 'last':
            S = torch.arange(M - N, M, device=q.device).expand(T_, N)
        adv = q64[:, None, :].expand(T_, N, D)
    else:
        s_max = sim.scatter(1, S, float('-inf')).max(dim=1).values          # max over benign keys, from q(T)
        lo, hi = s_max + delta_min, s_max + delta_min + delta
        if (hi > 1.0).any():
            raise ValueError('Eq. 7 interval exceeds cosine 1 for some target; reduce delta_min/delta')
        u = torch.rand(T_, N, generator=gen, dtype=torch.float64).to(q.device)
        s = lo[:, None] + (hi - lo)[:, None] * u                            # (T, N)
        r = F.normalize(torch.rand(T_, N, D, generator=gen, dtype=torch.float64).to(q.device), dim=2)
        o = F.normalize(r - (r * qn[:, None]).sum(-1, keepdim=True) * qn[:, None], dim=2)
        adv = (s[..., None] * qn[:, None] + (1.0 - s * s).sqrt()[..., None] * o) * q64.norm(dim=1)[:, None, None]
    pools = keys64.expand(T_, M, D).clone()
    pools.scatter_(1, S[..., None].expand(T_, N, D), adv)
    return pools.to(keys.dtype), S


def all_injected_selected(pools, S, batch_q):
    """Membership decision of Sec. 4.1: the union of the per-sample top-N selections of the client's batch
    contains every injected key. pools (T, M, D), S (T, N), batch_q (T, B, D) -> (T,) bool."""
    N = S.size(1)
    sim = F.normalize(batch_q, dim=2) @ F.normalize(pools, dim=2).transpose(1, 2)   # (T, B, M)
    sel = sim.topk(N, dim=2).indices.flatten(1)                                       # (T, B*N)
    used = torch.zeros(pools.shape[:2], dtype=torch.bool, device=pools.device).scatter_(1, sel, True)
    return used.gather(1, S).all(dim=1)


@torch.no_grad()
def security_game(keys, train_q, test_q, N, batch_size, delta_min, delta, seed, naive=False, naive_slots='last',
                  member_idx=None, chunk=250):
    """One run of the AMI game (Sec. 3.1, Eq. 3-4) with the original code's protocol.
    TPR (b = 1): each training image T is the target; the client's batch is T plus batch_size-1 other random training images.
    FPR (b = 0): each test image T is the target; the client's batch is batch_size random training images (T absent).
    member_idx: training indices used as TPR targets (default: all). Returns TPR, FPR, Advantage, ASR."""
    gen = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=train_q.device).manual_seed(seed)
    n = train_q.size(0)
    if member_idx is None:
        member_idx = torch.arange(n, device=train_q.device)

    def rate(targets, member):
        hits = 0
        for start in range(0, targets.size(0), chunk):
            t = targets[start:start + chunk]
            pools, S = inject_batch(keys, t, N, delta_min, delta, gen, naive, naive_slots)
            score = torch.rand(t.size(0), n, device=t.device, generator=g_dev)
            if member:
                score[torch.arange(t.size(0), device=t.device), member_idx[start:start + chunk]] = -1.0  # T is not its own filler
                batch = torch.cat([t[:, None], train_q[score.topk(batch_size - 1, dim=1).indices]], dim=1)
            else:
                batch = train_q[score.topk(batch_size, dim=1).indices]                                 # no repeats
            hits += all_injected_selected(pools, S, batch).sum().item()
        return hits / targets.size(0)

    tpr = rate(train_q[member_idx], member=True)
    fpr = rate(test_q, member=False)
    return dict(tpr=tpr, fpr=fpr, advantage=tpr - fpr, asr=(1 + tpr - fpr) / 2)
