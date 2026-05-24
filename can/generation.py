from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .hypergraph import GlobalStats, make_g_vector
from .models import HyperedgeCorePredictor, AutoregressiveMemberAssigner, NodeStructuralAllocator
from .training import TrainConfig



def _compute_need_ge_t(scheduler, t: int) -> int:
    """need(t) = sum_{(k,ce>=t)} k * R[(k,ce)]"""
    need = 0
    for (k, ce), r in scheduler.R.items():
        if r > 0 and ce >= t:
            need += int(k) * int(r)
    return int(need)


def _capacity_check_local(b: torch.Tensor, c: torch.Tensor, scheduler, thresholds=(1, 2, 4, 8, 16, 19)):
    """Return local capacity diagnostics for each threshold."""
    out = []
    for t in thresholds:
        avail = int(b[(c >= t)].sum().item())
        need = _compute_need_ge_t(scheduler, int(t))
        out.append((int(t), int(avail), int(need), int(avail - need)))
    return out


def _max_feasible_ce(b: torch.Tensor, c: torch.Tensor, k: int, ce_cap: int) -> int:
    """Return the largest feasible core threshold not greater than ce_cap."""
    ce_cap = int(ce_cap)
    k = int(k)
    for ce in range(ce_cap, -1, -1):
        cnt = int(((b > 0) & (c >= ce)).sum().item())
        if cnt >= k:
            return int(ce)
    return -1


def _pick_tau_relax_or_none(b: torch.Tensor, c: torch.Tensor, scheduler) -> Tuple[
    Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    remaining = [((int(k), int(ce)), int(r)) for (k, ce), r in scheduler.R.items() if int(r) > 0]
    if not remaining:
        return None, None
    remaining.sort(key=lambda x: x[0][1], reverse=True)              

    for (k, ce), r in remaining:
                          
        if int(((b > 0) & (c >= ce)).sum().item()) >= k:
            return (k, ce), (k, ce)
                           
        ce2 = _max_feasible_ce(b, c, k=k, ce_cap=ce)
        if ce2 >= 0:
            return (k, ce), (k, ce2)

                                    
    return remaining[0][0], None


def _force_members_fill(b: torch.Tensor, k: int) -> Optional[List[int]]:
    k = int(k)
    idx = torch.nonzero(b > 0, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return None
    vals = b[idx]
    order = torch.argsort(vals, descending=True)
    pick = idx[order][:min(k, int(idx.numel()))]
    return pick.tolist()


def _cand_count(b: torch.Tensor, c: torch.Tensor, ce: int) -> int:
    """#distinct candidates with b>0 and core>=ce"""
    return int(((b > 0) & (c >= int(ce))).sum().item())


def _downgrade_ce_until_feasible(b: torch.Tensor, c: torch.Tensor, k: int, ce: int) -> int:
    """keep decreasing ce until |Vcand|>=k or ce==1"""
    ce = int(ce)
    k = int(k)
    while ce > 1 and _cand_count(b, c, ce) < k:
        ce -= 1
    return ce


def _same_degree_swap_repair(
    b: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    k: int,
    ce: int,
    max_swaps: int = 2048,
) -> bool:
    """
    Try to increase |Vcand| by transferring 1 remaining stub from a donor v to a receiver u,
    under same-degree constraint: d_v == d_u, and both satisfy core>=ce.
    Choose donors with b_v>=2 so donor stays active (b_v>0), making candidate count +1 each swap.
    Return True if repaired to feasible (|Vcand|>=k).
    """
    k = int(k)
    ce = int(ce)

    for _ in range(int(max_swaps)):
        if _cand_count(b, c, ce) >= k:
            return True

                                                                         
        recv_mask = (b == 0) & (c >= ce) & (d > 0)
        if not bool(recv_mask.any().item()):
            return False
        u = int(torch.nonzero(recv_mask, as_tuple=False)[0].item())
        du = int(d[u].item())

                                                                                           
        donor_mask = (d == du) & (c >= ce) & (b >= 2)
        if not bool(donor_mask.any().item()):
                                                                    
                                                               
            recv_mask[u] = False                  
            alt = torch.nonzero(recv_mask, as_tuple=False)
            if alt.numel() == 0:
                return False
            u = int(alt[0].item())
            du = int(d[u].item())
            donor_mask = (d == du) & (c >= ce) & (b >= 2)
            if not bool(donor_mask.any().item()):
                return False

        v = int(torch.nonzero(donor_mask, as_tuple=False)[0].item())

                                                 
        b[v] -= 1
        b[u] += 1

    return _cand_count(b, c, ce) >= k


def _sample_k_truncated(k_vals: np.ndarray,
                        k_probs: np.ndarray,
                        min_k: int,
                        s_upper: int) -> int:
    """
    Sample k from empirical P_k truncated to [min_k, s_upper].
    If s_upper < min_k, return s_upper (end-game truncation).
    """
    s_upper = int(s_upper)
    if s_upper <= 0:
        return 0
    if s_upper < int(min_k):
        return s_upper

    mask = (k_vals >= int(min_k)) & (k_vals <= s_upper)
    if not mask.any():
        return s_upper

    sub_probs = k_probs[mask].astype("float64")
    prob_sum = sub_probs.sum()
    if prob_sum > 0:
        sub_probs = sub_probs / prob_sum
    else:
        sub_probs = np.ones_like(sub_probs) / len(sub_probs)

    return int(np.random.choice(k_vals[mask], p=sub_probs))


def _recent_edge_same_degree_swap_repair(
    out_edges: List[List[int]],
    out_taus: List[Tuple[int, int]],
    b: torch.Tensor,                                            
    d: torch.Tensor,                           
    c: torch.Tensor,                             
    k_need: int,
    ce_need: int,
    scan_recent: int = 128,                           
    max_swaps: int = 2048,
) -> bool:
    """
    Edge-member swap repair (NOT stub transfer):
    If |Vcand| < k_need under (b>0, c>=ce_need), try to increase distinct candidates by
    swapping in recent generated edges:

      pick a recent edge e, pick member u in e with b[u]==0,
      find v not in e with d[v]==d[u], c[v]>=ce_e (edge constraint), and b[v]>=2
      then replace u -> v in e and update budgets: b[u]+=1, b[v]-=1.

    This increases candidate count by +1 (u becomes active, v stays active).
    """
    k_need = int(k_need)
    ce_need = int(ce_need)

    def cand_count(_ce: int) -> int:
        return int(((b > 0) & (c >= int(_ce))).sum().item())

    if cand_count(ce_need) >= k_need:
        return True
    if not out_edges:
        return False

    n = len(out_edges)
    start = n - 1
    stop = max(-1, n - int(scan_recent) - 1)

    swaps = 0
    while cand_count(ce_need) < k_need and swaps < int(max_swaps):
        fixed = False

                                 
        for ei in range(start, stop, -1):
            e = out_edges[ei]
            k_e, ce_e = out_taus[ei]
            ce_e = int(ce_e)

                                                   
            e_set = set(e)

                                                                              
            for pos, u in enumerate(e):
                u = int(u)
                if int(b[u].item()) != 0:
                    continue                                        

                du = int(d[u].item())

                                                                                                         
                                                                          
                donor_mask = (d == du) & (c >= ce_e) & (b >= 2)
                if donor_mask.any():
                    cand_vs = torch.nonzero(donor_mask, as_tuple=False).view(-1)
                    v_pick = None
                    for vv in cand_vs.tolist():
                        if int(vv) not in e_set:
                            v_pick = int(vv)
                            break
                    if v_pick is None:
                        continue

                                              
                    e[pos] = v_pick
                                           
                    e_set.remove(u)
                    e_set.add(v_pick)

                                               
                    b[u] += 1
                    b[v_pick] -= 1

                    swaps += 1
                    fixed = True
                    break

            if fixed:
                break

        if not fixed:
                                           
            return False

    return cand_count(ce_need) >= k_need


def generate_hypergraph(X, g_star, allocator, member_model, core_predictor, cfg,
                        temperature=1.0,
                        force_fill=True,
                        max_plan_fail=100000):
    """
    Two-stage generation:
      Stage-1 (planning): sample (k, ce) sequence T with ce-downgrade + global S_cap + pre-occupy r_pre
      Stage-2 (assignment): construct edges using r_asg with member_model
        - downgrade ce if |Vcand|<k
        - if still infeasible at lower bound: recent-edge same-degree member swap repair
    """
    device = cfg.device
    X = X.to(device)
    g_vec = make_g_vector(g_star, device)

                                                     
    d, c, _ = allocator.sample_node_structures(X, g_vec, temperature=temperature)
    c = torch.minimum(c, d).clamp(0, g_star.cmax)

                                          
    r_pre = d.clone()
    r_asg = d.clone()

                    
    k_vals = np.arange(len(g_star.hist_k))
    k_probs = g_star.hist_k.astype(np.float32)
    min_k = int(k_vals[k_probs > 0].min()) if np.any(k_probs > 0) else 2

                                                                          
    T: List[Tuple[int, int]] = []
    S_total = int(r_pre.sum().item())
    S_rem = S_total

                                                                                   
    S_cap = S_total

    plan_fail = 0
    while S_rem > 0:
        if plan_fail > max_plan_fail:
            break

        s_upper = min(S_rem, S_cap)
        if s_upper <= 0:
            break

                                                       
        k = _sample_k_truncated(k_vals, k_probs, min_k=min_k, s_upper=s_upper)
        if k <= 0:
            break

                                             
        k_tensor = torch.tensor([k], device=device, dtype=torch.long)
        logits = core_predictor(k_tensor, g_vec)               
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        ce = int(torch.multinomial(probs, 1).item())

                                     
        def cand_count(_ce: int) -> int:
            return int(((r_pre > 0) & (c >= int(_ce))).sum().item())

        N = cand_count(ce)
        while (N < k) and (ce > 0):
            ce -= 1
            N = cand_count(ce)

                                                                                       
        if (ce == 0) and (N < k):
            S_cap = min(S_cap, k - 1)
            plan_fail += 1
            continue

                                                                        
        eligible = torch.nonzero((r_pre > 0) & (c >= ce), as_tuple=False).view(-1)
        if eligible.numel() < k:
            S_cap = min(S_cap, k - 1)
            plan_fail += 1
            continue

        vals = r_pre.index_select(0, eligible).to(torch.float32)
        topk_idx = torch.topk(vals, k=k, largest=True).indices
        A_pre = eligible.index_select(0, topk_idx)       

        r_pre[A_pre] -= 1

        T.append((int(k), int(ce)))
        S_rem -= int(k)

                                               
                          
    T.sort(key=lambda x: x[1], reverse=True)

    out_edges: List[List[int]] = []
    out_taus: List[Tuple[int, int]] = []                                                

    for (k, ce_j) in T:
        k = int(k)
        ce = int(ce_j)

                                                                                        
        ce = _downgrade_ce_until_feasible(r_asg, c, k=k, ce=ce)

                                                                                               
        if _cand_count(r_asg, c, ce) < k:
            _recent_edge_same_degree_swap_repair(
                out_edges=out_edges,
                out_taus=out_taus,
                b=r_asg,
                d=d,
                c=c,
                k_need=k,
                ce_need=ce,
                scan_recent=128,
                max_swaps=2048,
            )

        tau2 = (k, ce)

        members = member_model.sample_members(X, r_asg, c, tau2, temperature=temperature)

                                  
        if members is None and force_fill:
            members = _force_members_fill(r_asg, c, ce, k)

        if members:
            out_edges.append(members)
            out_taus.append(tau2)

            for v in members:
                r_asg[v] -= 1
        else:
            continue

    return out_edges


def save_edges(edges: List[List[int]], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(",".join(map(str, e)) + "\n")


def save_checkpoint(path, allocator, member_model, core_predictor, g_star, cfg, x_dim):
    """Save all models, stats and config to one file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "structural_feature_allocator_state": allocator.state_dict(),
        "dynamic_member_assignment_state": member_model.state_dict(),
        "hyperedge_structural_predictor_state": core_predictor.state_dict(),
        "structural_allocator_state": allocator.state_dict(),
        "member_assignment_state": member_model.state_dict(),
        "hyperedge_core_predictor_state": core_predictor.state_dict(),
        "g_star": g_star,
        "config_dict": {
            "hidden": cfg.hidden,
            "x_dim": x_dim,
                                                 
            "dmax": g_star.dmax,
            "cmax": g_star.cmax,
            "kmax": g_star.kmax
        }
    }
    torch.save(checkpoint, path)
    print(f"[Save] Checkpoint saved to {path}")


class HypergraphGenerator:
    """Wrapper to load models and run generation."""

    def __init__(self, model_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Load] Loading model from {model_path} to {self.device}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cp = torch.load(model_path, map_location=self.device, weights_only=False)
        self.g_star = cp["g_star"]
        conf = cp["config_dict"]

                                         
        self.g_vec = make_g_vector(self.g_star, self.device)
        g_dim = self.g_vec.numel()

                     
        x_dim = conf["x_dim"]
        hidden = conf["hidden"]

        self.allocator = NodeStructuralAllocator(x_dim, g_dim, hidden, self.g_star.dmax, self.g_star.cmax).to(self.device)
        self.member_model = AutoregressiveMemberAssigner(x_dim, hidden, self.g_star.kmax, self.g_star.cmax).to(self.device)
                                                            
        self.core_predictor = HyperedgeCorePredictor(self.g_star.kmax, self.g_star.cmax, g_dim, hidden).to(self.device)

                      
        self.allocator.load_state_dict(cp.get("structural_feature_allocator_state", cp.get("structural_allocator_state", cp.get("A_state"))))
        self.member_model.load_state_dict(cp.get("dynamic_member_assignment_state", cp.get("member_assignment_state", cp.get("B2_state"))))
        self.core_predictor.load_state_dict(cp.get("hyperedge_structural_predictor_state", cp.get("hyperedge_core_predictor_state", cp.get("HyperedgeCorePredictor_state"))))

        self.allocator.eval()
        self.member_model.eval()
        self.core_predictor.eval()

                                            
        self.cfg = TrainConfig(device=self.device)
        print("[Load] Model loaded successfully.")

    def generate(self, x_attr: torch.Tensor, temperature: float = 1.0):
        return generate_hypergraph(
            x_attr,
            self.g_star,
            self.allocator,
            self.member_model,
            self.core_predictor,
            self.cfg,                                         
            temperature=temperature,
        )