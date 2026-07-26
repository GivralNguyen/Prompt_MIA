import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

class Agg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                assignment: Optional[torch.Tensor] = None,
                return_assignment: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        N, L, d = x.shape
        mean_vec = x.mean(0)

        if assignment is None:
            assignment = torch.arange(L, device="cuda")
        # print(assignment)
        device = x.device
        assignment = assignment.to(device)

        used = assignment[assignment >= 0]
        K = used.max().item() + 1

        out = torch.zeros(K, d, device=device)
        cnt = torch.zeros(K, device=device)
        for j in range(L):
            k = assignment[j].item()
            if k >= 0:
                out[k] += mean_vec[j]
                cnt[k] += 1
        out = out / cnt.unsqueeze(-1)
        if return_assignment:
            return out, assignment
        else:
            return out
