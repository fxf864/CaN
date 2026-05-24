from __future__ import annotations

import random
import re
from typing import List, Tuple

import numpy as np
import torch



def _split_tokens(line: str) -> List[str]:
                                           
    return [t for t in re.split(r"[,\s;]+", line.strip()) if t]


def load_node_attributes(attr_path: str, dtype=np.float32) -> torch.Tensor:
    feats = []
    with open(attr_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = _split_tokens(line)
            try:
                row = [float(x) for x in toks]
            except ValueError as e:
                raise ValueError(f"[attrs] Line {ln} parse error: {line}") from e
            feats.append(row)

    if not feats:
        raise ValueError("Attribute file is empty or only comments.")
                             
    Fdim = len(feats[0])
    for i, r in enumerate(feats):
        if len(r) != Fdim:
            raise ValueError(f"Attribute dim mismatch at row {i}: got {len(r)} vs {Fdim}")
    x = torch.from_numpy(np.asarray(feats, dtype=dtype))
    return x         


def load_hyperedges(edge_path: str) -> List[List[int]]:
    edges = []
    with open(edge_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = _split_tokens(line)
            try:
                e = [int(x) for x in toks]
            except ValueError as e2:
                raise ValueError(f"[edges] Line {ln} parse error: {line}") from e2
            if len(e) == 0:
                continue
            edges.append(e)
    if not edges:
        raise ValueError("Edge file is empty or only comments.")
    return edges


def replicate_edges(edges: List[List[int]], mult: int, seed: int = 0, shuffle_each: bool = False) -> List[List[int]]:
    """
    Replicate edge list to enlarge |E| while keeping node set unchanged.
    Duplicates are allowed (multi-hyperedges) — fine for runtime scaling.
    """
    mult = int(mult)
    if mult <= 1:
        return edges

    rng = random.Random(seed)
    out = []
    for _ in range(mult):
        if shuffle_each:
            tmp = list(edges)
            rng.shuffle(tmp)
            out.extend(tmp)
        else:
            out.extend(edges)
    return out


def select_or_pad_attributes(X: torch.Tensor, target_dim: int, mode: str = "head", seed: int = 0) -> torch.Tensor:
    """
    Adjust attribute dimension to target_dim.
      - target_dim < F: select columns (head or random)
      - target_dim > F: pad by repeating columns (tile)
    """
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("attr_dim must be > 0")

    N, F = X.shape
    if target_dim == F:
        return X

    if target_dim < F:
        if mode == "head":
            return X[:, :target_dim].contiguous()
        elif mode == "random":
            g = torch.Generator(device=X.device)
            g.manual_seed(int(seed))
            idx = torch.randperm(F, generator=g, device=X.device)[:target_dim]
            idx, _ = torch.sort(idx)
            return X.index_select(1, idx).contiguous()
        else:
            raise ValueError("attr_select_mode must be 'head' or 'random'")
    else:
        reps = (target_dim + F - 1) // F
        return X.repeat(1, reps)[:, :target_dim].contiguous()


def fix_edge_indexing(edges: List[List[int]], num_nodes: int) -> Tuple[List[List[int]], str]:
    all_ids = [v for e in edges for v in e]
    mn, mx = min(all_ids), max(all_ids)
    if mn == 1 and mx == num_nodes:
        edges0 = [[v - 1 for v in e] for e in edges]
        return edges0, "Detected 1-indexed node ids; shifted to 0-index."
                              
    if mn < 0 or mx >= num_nodes:
        msg = (f"Warning: node ids out of range after indexing check: "
               f"min={mn}, max={mx}, num_nodes={num_nodes}. "
               f"Please verify your edge file indexing.")
        return edges, msg
    return edges, "Detected 0-indexed node ids; no shift."