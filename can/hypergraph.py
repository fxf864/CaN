from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import build_hist_1d, normalize_hist



@dataclass
class Hypergraph:
    num_nodes: int
    edges: List[List[int]]
    x: torch.Tensor         


def node_degrees(H: Hypergraph) -> List[int]:
    deg = [0] * H.num_nodes
    for e in H.edges:
        for v in e:
            deg[v] += 1
    return deg


def hyperedge_core_values(edges: List[List[int]], core_num: List[int]) -> List[int]:
    return [min(core_num[v] for v in e) if len(e) > 0 else 0 for e in edges]


def hypergraph_core_numbers(H: Hypergraph, min_edge_size_active: int = 1) -> List[int]:
    N = H.num_nodes
    edges = H.edges

    inc = [[] for _ in range(N)]
    for ei, e in enumerate(edges):
        for v in e:
            inc[v].append(ei)

    active = [True] * N
    edge_active_count = [len(e) for e in edges]
    edge_alive = [c >= min_edge_size_active for c in edge_active_count]

    deg = [0] * N
    for v in range(N):
        deg[v] = sum(1 for ei in inc[v] if edge_alive[ei])

    core = [0] * N
    in_queue = [False] * N
    queue: List[int] = []

    def push_if_needed(v: int, k: int):
        if active[v] and (deg[v] <= k) and (not in_queue[v]):
            queue.append(v)
            in_queue[v] = True

    curr_k = 0
    for v in range(N):
        push_if_needed(v, curr_k)

    while True:
        while queue:
            v = queue.pop()
            in_queue[v] = False
            if not active[v]:
                continue

            core[v] = curr_k
            active[v] = False

            for ei in inc[v]:
                if not edge_alive[ei]:
                    continue
                edge_active_count[ei] -= 1
                if edge_active_count[ei] < min_edge_size_active:
                    edge_alive[ei] = False
                    for u in edges[ei]:
                        if active[u]:
                            deg[u] -= 1
                            push_if_needed(u, curr_k)

        if not any(active):
            break

        curr_k += 1
        for v in range(N):
            push_if_needed(v, curr_k)

        if curr_k > max(deg) + len(edges) + 10:
            for v in range(N):
                if active[v]:
                    core[v] = curr_k
            break

    return core


@dataclass
class GlobalStats:
    hist_d: np.ndarray                       
    hist_cv: np.ndarray                       
    hist_k: np.ndarray                       
    hist_ce: np.ndarray                       
    joint_k_ce_counts: Dict[Tuple[int, int], int]                              
    num_edges: int
    stub_total: int
    dmax: int
    cmax: int
    kmax: int


def compute_global_stats_from_real_hypergraph(H: Hypergraph,
                                              dmax: Optional[int] = None,
                                              cmax: Optional[int] = None,
                                              kmax: Optional[int] = None,
                                              min_edge_size_active: int = 1) -> Tuple[
    GlobalStats, List[int], List[int], List[int]]:
    d_star_list = node_degrees(H)
    c_star_list = hypergraph_core_numbers(H, min_edge_size_active=min_edge_size_active)
    ce_star_list = hyperedge_core_values(H.edges, c_star_list)
    k_list = [len(e) for e in H.edges]
    c_star_np = np.array(c_star_list, dtype=np.int64)
    ce_star_np = np.array(ce_star_list, dtype=np.int64)

    print("max c_star =", int(c_star_np.max()), "count(c_star>=19) =", int((c_star_np >= 19).sum()))
    print("max ce_star =", int(ce_star_np.max()), "count(ce_star>=19) =", int((ce_star_np >= 19).sum()))

    dmax0 = max(d_star_list) if dmax is None else int(dmax)
    cmax0 = max(c_star_list) if cmax is None else int(cmax)
    kmax0 = max(k_list) if kmax is None else int(kmax)

                                 
    hist_d = normalize_hist(build_hist_1d(d_star_list, dmax0))
    hist_cv = normalize_hist(build_hist_1d(c_star_list, cmax0))
    hist_k = normalize_hist(build_hist_1d(k_list, kmax0))
    hist_ce = normalize_hist(build_hist_1d(ce_star_list, cmax0))

    joint: Dict[Tuple[int, int], int] = {}
    for k, ce in zip(k_list, ce_star_list):
        if 0 <= k <= kmax0 and 0 <= ce <= cmax0:
            joint[(k, ce)] = joint.get((k, ce), 0) + 1

    stub_total = int(sum(k_list))

    g_star = GlobalStats(
        hist_d=hist_d,
        hist_cv=hist_cv,
        hist_k=hist_k,
        hist_ce=hist_ce,
        joint_k_ce_counts=joint,
        num_edges=len(H.edges),
        stub_total=stub_total,
        dmax=dmax0,
        cmax=cmax0,
        kmax=kmax0,
    )
    return g_star, d_star_list, c_star_list, ce_star_list


def make_g_vector(g: GlobalStats, device: str) -> torch.Tensor:
    """
    Flatten g* into a vector:
      [hist_d, hist_cv, hist_k, hist_ce]
    """
    vec = np.concatenate([g.hist_d, g.hist_cv, g.hist_k, g.hist_ce], axis=0).astype(np.float32)
    return torch.from_numpy(vec).to(device)