from __future__ import annotations

import random
import numpy as np
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class NodeStructuralAllocator(nn.Module):
    def __init__(self, x_dim: int, g_dim: int, hidden: int, dmax: int, cmax: int):
        super().__init__()
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.hidden = hidden
        self.dmax = dmax
        self.cmax = cmax

        self.enc_x = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.deg_head = nn.Sequential(
            nn.Linear(hidden + g_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dmax + 1),
        )

        self.d_emb = nn.Embedding(dmax + 1, hidden)
        self.core_head = nn.Sequential(
            nn.Linear(hidden + g_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, cmax + 1),
        )

        self.dec_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def mask_invalid_core_logits(self, core_logits: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        N, C = core_logits.shape
        c_idx = torch.arange(C, device=core_logits.device).view(1, C).expand(N, C)
        d_exp = d.view(N, 1).expand(N, C)
        invalid = c_idx > d_exp
        return core_logits.masked_fill(invalid, float("-inf"))

    def predict_logits(self, x: torch.Tensor, g: torch.Tensor, d_cond: Optional[torch.Tensor] = None):
        N = x.size(0)
        g_in = g.unsqueeze(0).expand(N, -1) if g.dim() == 1 else g.expand(N, -1)

        h = self.enc_x(x)
        x_recon = x + self.dec_x(h)
                                 

        deg_logits = self.deg_head(torch.cat([h, g_in], dim=-1))
        core_logits = None
        if d_cond is not None:
            d_cond = torch.clamp(d_cond, 0, self.dmax)
            d_emb = self.d_emb(d_cond)
            core_logits = self.core_head(torch.cat([h, g_in, d_emb], dim=-1))
            core_logits = self.mask_invalid_core_logits(core_logits, d_cond)
        return deg_logits, core_logits, x_recon

    @torch.no_grad()
    def sample_node_structures(self, x: torch.Tensor, g: torch.Tensor, temperature: float = 1.0):
        deg_logits, _, _ = self.predict_logits(x, g, d_cond=None)
        deg_probs = F.softmax(deg_logits / max(temperature, 1e-6), dim=-1)
        d = torch.multinomial(deg_probs, 1).squeeze(-1)

        _, core_logits, _ = self.predict_logits(x, g, d_cond=d)
        core_probs = F.softmax(core_logits / max(temperature, 1e-6), dim=-1)
        c = torch.multinomial(core_probs, 1).squeeze(-1)

        b = d.clone()
        return d, c, b

    def expected_total_degree(self, deg_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(deg_logits, dim=-1)
        d_vals = torch.arange(self.dmax + 1, device=deg_logits.device).float().view(1, -1)
        exp_d = torch.sum(probs * d_vals, dim=-1)
        return exp_d.sum()

    def predict_structural_histograms(self, deg_logits: torch.Tensor, x: torch.Tensor, g: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        N = x.size(0)
        deg_probs = F.softmax(deg_logits, dim=-1)            
        hist_d_hat = deg_probs.mean(dim=0)

                                             
        c_accum = torch.zeros((N, self.cmax + 1), device=x.device)
        for d_val in range(self.dmax + 1):
            d_cond = torch.full((N,), d_val, device=x.device, dtype=torch.long)
            _, core_logits, _ = self.predict_logits(x, g, d_cond=d_cond)
            core_probs = F.softmax(core_logits, dim=-1)
            w = deg_probs[:, d_val].unsqueeze(-1)
            c_accum += w * core_probs
        hist_c_hat = c_accum.mean(dim=0)
        return hist_d_hat, hist_c_hat


class HyperedgeCorePredictor(nn.Module):
    def __init__(self, kmax: int, cmax: int, g_dim: int, hidden: int = 128):
        super().__init__()
                                        
        self.k_emb = nn.Embedding(kmax + 1, hidden)

             
        self.mlp = nn.Sequential(
            nn.Linear(hidden + g_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, cmax + 1)                     
        )

    def forward(self, k: torch.Tensor, g_vec: torch.Tensor) -> torch.Tensor:
                    
                                          

        B = k.size(0)
        if g_vec.dim() == 1:
            g_vec = g_vec.unsqueeze(0).expand(B, -1)

        k_feat = self.k_emb(k)               
        combined = torch.cat([k_feat, g_vec], dim=-1)

        logits = self.mlp(combined)               
        return logits


class FeasibleHyperedgeSampler:
    def __init__(self, model: HyperedgeCorePredictor, g_vec: torch.Tensor, k_dist: np.ndarray, device: str, cmax: int):
        self.model = model
        self.g_vec = g_vec
        self.device = device
        self.cmax = cmax

                     
        self.k_vals = np.arange(len(k_dist))
        self.k_probs = k_dist.astype(np.float32)
                             
        self.min_k = self.k_vals[k_dist > 0].min() if any(k_dist > 0) else 2

    @torch.no_grad()
    def sample_hyperedge_signature(self, b: torch.Tensor, c: torch.Tensor) -> Optional[Tuple[int, int]]:
        S_rem = int(b.sum().item())
        if S_rem <= 0:
            return None

                                
        if S_rem < self.min_k:
            k = S_rem
        else:
            mask = (self.k_vals >= self.min_k) & (self.k_vals <= S_rem)
            if not mask.any():
                k = S_rem
            else:
                                         
                sub_probs = self.k_probs[mask].astype('float64')
                prob_sum = sub_probs.sum()

                if prob_sum > 0:
                                  
                    sub_probs = sub_probs / prob_sum
                else:
                                              
                    sub_probs = np.ones_like(sub_probs) / len(sub_probs)

                      
                k = int(np.random.choice(self.k_vals[mask], p=sub_probs))

                                          
        k_tensor = torch.tensor([k], device=self.device, dtype=torch.long)
        logits = self.model(k_tensor, self.g_vec)

                                  
        valid_mask = (b > 0)
        valid_cores = c[valid_mask]
        if valid_cores.numel() == 0:
            return (k, 0)

                                                
        counts = torch.histc(valid_cores.float(), bins=self.cmax + 1, min=0, max=self.cmax)
        count_ge = torch.flip(torch.cumsum(torch.flip(counts, [0]), 0), [0])            

                                                       
        valid_b = b[valid_mask].to(torch.float32)
        core_idx = valid_cores.to(torch.long).clamp(0, self.cmax)
        stub_counts = torch.bincount(core_idx, weights=valid_b, minlength=self.cmax + 1)            
        stub_ge = torch.flip(torch.cumsum(torch.flip(stub_counts, [0]), 0), [0])            

                                 
        mask_feasible = (count_ge >= k) & (stub_ge >= k)

        L = logits.size(1)
        valid_len = min(L, mask_feasible.size(0))

        full_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        full_mask[:valid_len] = mask_feasible[:valid_len]
        logits[0, ~full_mask] = float('-inf')

        if torch.isinf(logits).all():
            feasible_indices = torch.where(mask_feasible)[0]
            ce = int(feasible_indices.max().item()) if feasible_indices.numel() > 0 else 0
        else:
            probs = F.softmax(logits, dim=-1)
            ce = torch.multinomial(probs, 1).item()

        return (k, ce)


class AutoregressiveMemberAssigner(nn.Module):
    def __init__(self, x_dim: int, hidden: int, kmax: int, cmax: int):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

                                         
        self.step_emb = nn.Embedding(max(256, kmax + 2), hidden)
        self.k_emb = nn.Embedding(max(256, kmax + 2), hidden)
        self.ce_emb = nn.Embedding(max(256, cmax + 2), hidden)

                                                                             
        in_dim = hidden * 5 + 2
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def score_all_candidates(self, z: torch.Tensor, S_mask: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                            step_i: int, k: int, ce: int) -> torch.Tensor:
        N, H = z.shape
        if S_mask.any():
            z_set = z[S_mask].mean(dim=0, keepdim=True).expand(N, -1)
        else:
            z_set = torch.zeros((N, H), device=z.device)

        step_e = self.step_emb(torch.tensor([step_i], device=z.device)).expand(N, -1)
        k_e = self.k_emb(torch.tensor([k], device=z.device)).expand(N, -1)
        ce_e = self.ce_emb(torch.tensor([ce], device=z.device)).expand(N, -1)

        b_f = b.float().view(N, 1)
        c_f = c.float().view(N, 1)

        feats = torch.cat([z, z_set, step_e, k_e, ce_e, b_f, c_f], dim=-1)
        logits = self.score(feats).squeeze(-1)
        return logits

    def mask_infeasible_candidates(self, logits: torch.Tensor, S_mask: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                        ce: int) -> torch.Tensor:
        invalid = S_mask | (b <= 0) | (c < ce)
        return logits.masked_fill(invalid, float("-inf"))

    @torch.no_grad()
    def sample_members(self, x: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   tau: Tuple[int, int], temperature: float = 1.0) -> Optional[List[int]]:
        k, ce = tau
        z = self.encode_nodes(x)
        N = x.size(0)

        S_mask = torch.zeros((N,), dtype=torch.bool, device=x.device)
        chosen: List[int] = []

        for i in range(k):
            logits = self.score_all_candidates(z, S_mask, b, c, step_i=i, k=k, ce=ce)
            logits = self.mask_infeasible_candidates(logits, S_mask, b, c, ce)
            if torch.isinf(logits).all():
                return None
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            v = torch.multinomial(probs, 1).item()
            chosen.append(v)

                                            
            S_mask = S_mask.clone()
            S_mask[v] = True

        return chosen

    def compute_teacher_forcing_loss(self, x: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                             tau: Tuple[int, int], target_set: List[int], prefix_len: int = 0) -> torch.Tensor:
        k, ce = tau
        assert len(target_set) == k
        device = x.device
        z = self.encode_nodes(x)
        N = x.size(0)

        order = list(range(k))
        random.shuffle(order)
        seq = [target_set[i] for i in order]

        S_mask = torch.zeros((N,), dtype=torch.bool, device=device)
        b_local = b.clone()

                                       
        for i in range(prefix_len):
            v = seq[i]
            S_mask = S_mask.clone()
            S_mask[v] = True
            b_local[v] -= 1                                

        total = torch.tensor(0.0, device=device)
        steps = 0

        for i in range(prefix_len, k):
            logits = self.score_all_candidates(z, S_mask, b_local, c, step_i=i, k=k, ce=ce)
            logits = self.mask_infeasible_candidates(logits, S_mask, b_local, c, ce)

            tv = seq[i]
            if torch.isinf(logits).all() or torch.isinf(logits[tv]):
                continue

            total = total + F.cross_entropy(logits.view(1, -1), torch.tensor([tv], device=device))
            steps += 1

                                                 
            S_mask = S_mask.clone()
            S_mask[tv] = True
            b_local[tv] -= 1

        return total / max(steps, 1)

    def score_candidate_subset(
            self,
            z: torch.Tensor,         
            idx: torch.Tensor,                          
            S_mask: torch.Tensor,            
            b: torch.Tensor,                
            c: torch.Tensor,                
            step_i: int,
            k: int,
            ce: int
    ) -> torch.Tensor:
        device = z.device
        N, H = z.shape
        M = idx.numel()

                                                      
        if S_mask.any():
            z_set = z[S_mask].mean(dim=0, keepdim=True)         
        else:
            z_set = torch.zeros((1, H), device=device)

        z_set = z_set.expand(M, -1)         

                    
        step_e = self.step_emb(torch.tensor([step_i], device=device)).expand(M, -1)
        k_e = self.k_emb(torch.tensor([k], device=device)).expand(M, -1)
        ce_e = self.ce_emb(torch.tensor([ce], device=device)).expand(M, -1)

                                         
        z_sub = z[idx]         
        b_f = b[idx].float().view(M, 1)
        c_f = c[idx].float().view(M, 1)

        feats = torch.cat([z_sub, z_set, step_e, k_e, ce_e, b_f, c_f], dim=-1)             
        logits = self.score(feats).squeeze(-1)       
        return logits

    def compute_negative_sampling_loss(
            self,
            x: torch.Tensor,
            b: torch.Tensor,
            c: torch.Tensor,
            tau: Tuple[int, int],
            target_set: List[int],
            prefix_len: int = 0,
            neg_samples: int = 256,
            mode: str = "softmax",                           
    ) -> torch.Tensor:
        k, ce = tau
        assert len(target_set) == k
        device = x.device

                              
        z = self.encode_nodes(x)         
        N = x.size(0)

                                                 
        order = list(range(k))
        random.shuffle(order)
        seq = [target_set[i] for i in order]

        S_mask = torch.zeros((N,), dtype=torch.bool, device=device)
        b_local = b.clone()

                      
        for i in range(prefix_len):
            v = int(seq[i])
            S_mask = S_mask.clone()
            S_mask[v] = True
            b_local[v] -= 1

        total = torch.tensor(0.0, device=device)
        steps = 0

        for i in range(prefix_len, k):
            tv = int(seq[i])

                                                          
            eligible = (~S_mask) & (b_local > 0) & (c >= ce)

                                                                                  
            if not bool(eligible[tv].item()):
                continue

            eligible_idx = torch.nonzero(eligible, as_tuple=False).view(-1)

                                                             
            if eligible_idx.numel() <= 1:
                                                                    
                loss_step = torch.tensor(0.0, device=device)
            else:
                                                                   
                                         
                mask_not_tv = eligible_idx != tv
                neg_pool = eligible_idx[mask_not_tv]
                if neg_pool.numel() == 0:
                    loss_step = torch.tensor(0.0, device=device)
                else:
                    m = min(int(neg_samples), int(neg_pool.numel()))
                    perm = torch.randperm(neg_pool.numel(), device=device)[:m]
                    neg_idx = neg_pool[perm]       

                                                                    
                    idx = torch.cat([torch.tensor([tv], device=device), neg_idx], dim=0)

                    logits_sub = self.score_candidate_subset(
                        z=z, idx=idx, S_mask=S_mask, b=b_local, c=c,
                        step_i=i, k=k, ce=ce
                    )         

                    if mode == "softmax":
                                          
                        loss_step = F.cross_entropy(logits_sub.view(1, -1),
                                                    torch.zeros((1,), dtype=torch.long, device=device))
                    elif mode == "logistic":
                        pos = logits_sub[0]
                        neg = logits_sub[1:]
                                                             
                        loss_step = F.softplus(-pos) + F.softplus(neg).mean()
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

            total = total + loss_step
            steps += 1

                            
            S_mask = S_mask.clone()
            S_mask[tv] = True
            b_local[tv] -= 1

        return total / max(steps, 1)

    def score_candidate_subset_fast(
            self,
            z: torch.Tensor,                     
            idx: torch.Tensor,                  
            z_set: torch.Tensor,                    
            step_i: int,
            k: int,
            ce: int,
            b_local: torch.Tensor,                     
            c: torch.Tensor,       
    ) -> torch.Tensor:
        device = z.device
        H = z.size(1)
        M = idx.numel()

        z_sub = z.index_select(0, idx)         
        z_set_e = z_set.view(1, H).expand(M, -1)         

                                                               
        step_e = self.step_emb.weight[int(step_i)].view(1, -1).expand(M, -1)
        k_e = self.k_emb.weight[int(k)].view(1, -1).expand(M, -1)
        ce_e = self.ce_emb.weight[int(ce)].view(1, -1).expand(M, -1)

        b_f = b_local.index_select(0, idx).float().view(M, 1)
        c_f = c.index_select(0, idx).float().view(M, 1)

        feats = torch.cat([z_sub, z_set_e, step_e, k_e, ce_e, b_f, c_f], dim=-1)            
        logits = self.score(feats).squeeze(-1)       
        return logits

    def compute_fast_negative_sampling_loss_with_embeddings(
            self,
            z: torch.Tensor,                                             
            b: torch.Tensor,                                
            c: torch.Tensor,                           
            tau: Tuple[int, int],
            target_set: torch.Tensor,                     
            pool_tau: torch.Tensor,                                   
            prefix_len: int,
            neg_samples: int = 256,
            mode: str = "softmax",                           
            random_order: bool = True,
    ) -> torch.Tensor:
        device = z.device
        k, ce = int(tau[0]), int(tau[1])
        assert target_set.numel() == k
        assert pool_tau.numel() > 0

                                                     
        if random_order and k > 1:
            perm = torch.randperm(k, device=device)
            seq = target_set.index_select(0, perm)
        else:
            seq = target_set

                     
        b_local = b.clone()
        selected = torch.zeros((b_local.numel(),), dtype=torch.bool, device=device)

        H = z.size(1)
        z_sum = torch.zeros((H,), device=device, dtype=z.dtype)
        cnt = 0

                                      
        if prefix_len > 0:
            pref = seq[:prefix_len]
            selected[pref] = True
            b_local[pref] -= 1
            z_sum += z.index_select(0, pref).sum(dim=0)
            cnt += int(prefix_len)

        total = torch.zeros((), device=device, dtype=z.dtype)
        steps = 0

        P = int(pool_tau.numel())
        M = int(neg_samples)

                                                 
        idx_buf = torch.empty((1 + M,), device=device, dtype=torch.long)

        for step_i in range(prefix_len, k):
            tv = seq[step_i]                        

                   
            if cnt > 0:
                z_set = z_sum / float(cnt)
            else:
                z_set = torch.zeros((H,), device=device, dtype=z.dtype)

                                                                          
            neg = pool_tau[torch.randint(0, P, (M,), device=device)]
            idx = torch.cat([tv.view(1), neg], dim=0)                    

            logits = self.score_candidate_subset_fast(
                z=z,
                idx=idx,
                z_set=z_set,
                step_i=step_i,
                k=k,
                ce=ce,
                b_local=b_local,
                c=c,
            )

            invalid = selected.index_select(0, idx) | (b_local.index_select(0, idx) <= 0) | (
                    c.index_select(0, idx) < ce)
            invalid = invalid | (idx == tv)
            invalid[0] = False

            logits = logits.masked_fill(invalid, float("-inf"))

                                           
            if torch.isinf(logits).all():
                continue

            if mode == "softmax":
                                          
                loss_step = F.cross_entropy(logits.view(1, -1), torch.zeros((1,), device=device, dtype=torch.long))
            elif mode == "logistic":
                pos = logits[0]
                neg_logits = logits[1:]
                                                          
                neg_logits = neg_logits[~torch.isinf(neg_logits)]
                if neg_logits.numel() == 0:
                    loss_step = F.softplus(-pos)
                else:
                    loss_step = F.softplus(-pos) + F.softplus(neg_logits).mean()
            else:
                raise ValueError(f"Unknown mode={mode}")

            total = total + loss_step
            steps += 1

                                     
            selected[tv] = True
            b_local[tv] -= 1
            z_sum += z[tv]
            cnt += 1

        return total / max(steps, 1)
