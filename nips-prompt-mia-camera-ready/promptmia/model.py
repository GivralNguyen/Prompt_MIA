"""L2P-style prompt pool on a frozen ViT-B/32 (paper Sec. 2, Eq. 1-2)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from promptmia.backbones import vit_cls_query


class PromptPool(nn.Module):
    """M (key, prompt) pairs; each input selects the N keys with the highest cosine similarity to q(x) (Eq. 1)."""

    def __init__(self, M=20, N=4, dim=768, patch=32):
        super().__init__()
        self.N = N
        val = math.sqrt(6.0 / (3 * patch * patch + dim))  # uniform init as in the original L2P code
        self.prompt = nn.Parameter(torch.empty(M, dim).uniform_(-val, val))
        self.key = nn.Parameter(torch.empty(M, dim).uniform_(-val, val))

    def select(self, q):
        sim = F.normalize(q, dim=1) @ F.normalize(self.key, dim=1).t()  # (B, M) cosine similarity
        top_sim, idx = sim.topk(self.N, dim=1)
        key_loss = (1.0 - top_sim).sum(dim=1).mean()                     # sum_i gamma(q(x), k_i), cosine distance
        return idx, key_loss


class L2PViT(nn.Module):
    """Frozen torchvision ViT-B/32 with selected prompts inserted after the class token.

    head='cls'    : CLS -> frozen ImageNet head -> Linear-Dropout-Linear (default; the head behind the paper's accuracy)
    head='prompt' : classifier on the mean of the N prompt-token hidden states (Eq. 2 as currently worded)
    """

    def __init__(self, num_classes, M=20, N=4, head='cls'):
        super().__init__()
        self.vit = torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
        self.vit.requires_grad_(False)
        self.pool = PromptPool(M, N, self.vit.hidden_dim)
        self.head_type = head
        if head == 'prompt':
            self.head = nn.Linear(self.vit.hidden_dim, num_classes)
        elif head == 'cls':
            self.head = nn.Sequential(nn.Linear(1000, 512), nn.Dropout(0.5), nn.Linear(512, num_classes))
        else:
            raise ValueError(head)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()  # frozen backbone always in eval mode
        return self

    @torch.no_grad()
    def query(self, x):
        """q(x): CLS output of the frozen, prompt-free ViT."""
        return vit_cls_query(self.vit, x)

    def forward(self, x, q=None):
        """Returns (logits, key_loss, selected indices). Pass a precomputed q(x) to skip the query pass."""
        if q is None:
            q = self.query(x)
        idx, key_loss = self.pool.select(q)
        tok = self.vit._process_input(x)
        tok = torch.cat([self.vit.class_token.expand(tok.size(0), -1, -1), tok], dim=1)
        tok = tok + self.vit.encoder.pos_embedding
        prompts = self.pool.prompt[idx]                                   # (B, N, D), no positional embedding
        h = torch.cat([tok[:, :1], prompts, tok[:, 1:]], dim=1)
        h = self.vit.encoder.ln(self.vit.encoder.layers(h))
        if self.head_type == 'prompt':
            logits = self.head(h[:, 1:1 + self.pool.N].mean(dim=1))
        else:
            logits = self.head(self.vit.heads(h[:, 0]))
        return logits, key_loss, idx

    def trainable_state(self):
        """Clones of every trainable tensor: the pool and the head (what a client sends to the server)."""
        return {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v)  # copy into this model's own storage; never alias the caller's tensors
