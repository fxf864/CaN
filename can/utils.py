from __future__ import annotations

import random
from typing import List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def normalize_hist(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = counts.sum()
    if s <= 0:
        return np.ones_like(counts) / len(counts)
    return (counts + eps) / (s + eps * len(counts))


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
              
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def build_hist_1d(values: List[int], max_val: int) -> np.ndarray:
    h = np.zeros(max_val + 1, dtype=np.float64)
    for v in values:
        if 0 <= v <= max_val:
            h[v] += 1
    return h