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


def naive_inject(keys, q, N):
    """Eq. 5-6: every injected key equals q(T)."""
    S = injection_slots(keys, q, N)
    out = keys.clone()
    out[S] = q.to(keys.dtype)
    return out, S
