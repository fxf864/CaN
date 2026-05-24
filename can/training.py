from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import GlobalStats, Hypergraph, make_g_vector
from .models import AutoregressiveMemberAssigner, NodeStructuralAllocator
from .utils import kl_divergence



@dataclass
class TrainConfig:
    hidden: int = 256
    lr: float = 1e-3

                      
    epoch_node: int = 1000
    epoch_edge: int = 1000
    epoch_member: int = 1000

    beta: float = 10
    alpha: float = 1
    lam_attr: float = 0.1

                  
    member_loss_weight: float = 1.0
    member_batch_size: int = 256                     
    member_steps_per_epoch: int = 1                                        
    sampled_budget_prob: float = 0.0                            

    device: str = "cuda" if torch.cuda.is_available() else "cpu"



def build_hyperedge_training_data(H: Hypergraph, g_star: GlobalStats, ce_star_list: List[int]):
    edge_data: List[Tuple[Tuple[int, int], List[int]]] = []
    for edge, ce in zip(H.edges, ce_star_list):
        k = len(edge)
        if k > 0 and k <= g_star.kmax and ce <= g_star.cmax:
            edge_data.append(((k, ce), edge))
    return edge_data


def train_structural_feature_allocator(
        H: Hypergraph,
        g_star: GlobalStats,
        d_star_list: List[int],
        c_star_list: List[int],
        cfg: TrainConfig,
):
    device = cfg.device
    X = H.x.to(device)
    _, x_dim = X.shape

    dmax, cmax = g_star.dmax, g_star.cmax
    g_vec = make_g_vector(g_star, device)
    g_dim = g_vec.numel()

    d_star = torch.tensor(d_star_list, dtype=torch.long, device=device).clamp(0, dmax)
    c_star = torch.tensor(c_star_list, dtype=torch.long, device=device).clamp(0, cmax)

    allocator = NodeStructuralAllocator(
        x_dim=x_dim,
        g_dim=g_dim,
        hidden=cfg.hidden,
        dmax=dmax,
        cmax=cmax,
    ).to(device)

    hist_d_t = torch.from_numpy(g_star.hist_d.astype(np.float32)).to(device)
    hist_c_t = torch.from_numpy(g_star.hist_cv.astype(np.float32)).to(device)
    S_star = int(g_star.stub_total)

    w_deg = torch.arange(dmax + 1, device=device).float() + 1.0
    w_deg = w_deg / w_deg.mean()
    w_core = torch.arange(cmax + 1, device=device).float() + 1.0
    w_core = w_core / w_core.mean()

    opt = torch.optim.Adam(allocator.parameters(), lr=cfg.lr)

    print(f"[Stage-1] Training structural feature allocator for {cfg.epoch_node} epochs...")
    for epoch in range(cfg.epoch_node):
        allocator.train()
        opt.zero_grad()

        deg_logits, core_logits_teacher, x_recon = allocator.predict_logits(X, g_vec, d_cond=d_star)

        loss_deg = F.cross_entropy(deg_logits, d_star, weight=w_deg)
        loss_core = F.cross_entropy(core_logits_teacher, c_star, weight=w_core)
        loss_sup = loss_deg + loss_core

        probs_d = F.softmax(deg_logits, dim=-1)
        hist_d_hat = probs_d.mean(dim=0)
        probs_c = F.softmax(core_logits_teacher, dim=-1)
        hist_c_hat = probs_c.mean(dim=0)
        loss_dist = kl_divergence(hist_d_hat, hist_d_t) + kl_divergence(hist_c_hat, hist_c_t)

        exp_sum = allocator.expected_total_degree(deg_logits)
        loss_stub = torch.abs(exp_sum - float(S_star)) / (float(S_star) + 1e-6)

        loss_attr = F.binary_cross_entropy_with_logits(x_recon, X)
        loss = loss_sup + cfg.alpha * loss_dist + cfg.beta * loss_stub + cfg.lam_attr * loss_attr
        loss.backward()
        nn.utils.clip_grad_norm_(allocator.parameters(), 2.0)
        opt.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[FeatureAllocation][Epoch {epoch + 1:03d}/{cfg.epoch_node}] "
                f"lossA={loss.item():.4f} | sup={loss_sup.item():.4f} "
                f"dist={loss_dist.item():.4f} stub={loss_stub.item():.4f} attr={loss_attr.item():.4f}"
            )

    del opt, deg_logits, core_logits_teacher, x_recon, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return allocator.eval(), d_star, c_star, g_vec


def train_feature_allocation_and_member_assignment(
        H: Hypergraph,
        g_star: GlobalStats,
        d_star_list: List[int],
        c_star_list: List[int],
        ce_star_list: List[int],
        cfg: TrainConfig,
):
    allocator, d_star, c_star, _ = train_structural_feature_allocator(
        H, g_star, d_star_list, c_star_list, cfg
    )
    edge_data = build_hyperedge_training_data(H, g_star, ce_star_list)
    member_model = AutoregressiveMemberAssigner(
        x_dim=H.x.size(1),
        hidden=cfg.hidden,
        kmax=g_star.kmax,
        cmax=g_star.cmax,
    ).to(cfg.device)
    member_model = train_dynamic_member_assignment(
        member_model=member_model,
        X=H.x.to(cfg.device),
        d_star=d_star,
        c_star=c_star,
        edge_data=edge_data,
        epoch_member=cfg.epoch_member,
        lr=cfg.lr,
        batch_size=cfg.member_batch_size,
        steps_per_epoch=cfg.member_steps_per_epoch,
        neg_samples=32,
        mode="softmax",
        use_compile=False,
    )
    return allocator.eval(), member_model.eval()

def build_hyperedge_signature_buckets(edge_data):
    buckets = {}
    for tau, nodes in edge_data:
        tau = (int(tau[0]), int(tau[1]))
        buckets.setdefault(tau, []).append(nodes)
    return buckets


def score_member_candidates_batch(member_model, z, candidates, z_set, step_i, k, ce, b_static=None, c_static=None):
    device = z.device
    B, M = candidates.shape
    H = z.shape[1]

                                        
    z_cand = z[candidates]                               

                            
    step_e = member_model.step_emb.weight[int(step_i)].view(1, 1, -1).expand(B, M, -1)
    k_e = member_model.k_emb.weight[int(k)].view(1, 1, -1).expand(B, M, -1)
    ce_e = member_model.ce_emb.weight[int(ce)].view(1, 1, -1).expand(B, M, -1)

    z_set_e = z_set.view(B, 1, H).expand(B, M, H)

                            
    if b_static is None:
        b_f = torch.zeros((B, M, 1), device=device, dtype=z.dtype)
    else:
        b_f = b_static[candidates].float().unsqueeze(-1)           

    if c_static is None:
        c_f = torch.zeros((B, M, 1), device=device, dtype=z.dtype)
    else:
        c_f = c_static[candidates].float().unsqueeze(-1)           

    feats = torch.cat([z_cand, z_set_e, step_e, k_e, ce_e, b_f, c_f], dim=-1)              
    feats = feats.reshape(B * M, -1)

    logits = member_model.score(feats).view(B, M)         
    return logits


def member_loss_batched_negative_sampling(
        member_model,
        z,                                     
        seq,                           
        tau,          
        eligible_pool,                                                     
        neg_samples=256,
        use_norepeat=True,
        b_static=None,
        c_static=None,
):
    device = z.device
    B, k = seq.shape
    k_tau, ce = int(tau[0]), int(tau[1])
    assert k == k_tau

    H = z.shape[1]
                                                     
    z_sum = torch.zeros((B, H), device=device, dtype=z.dtype)
    cnt = torch.zeros((B, 1), device=device, dtype=z.dtype)

                                   
    selected = torch.zeros((B, z.shape[0]), device=device, dtype=torch.bool) if use_norepeat else None

    total = torch.zeros((), device=device, dtype=z.dtype)

    for step_i in range(k):
        tv = seq[:, step_i]       

                      
        z_set = torch.where(cnt > 0, z_sum / torch.clamp(cnt, min=1.0), torch.zeros_like(z_sum))

                                                                                  
        P = eligible_pool.numel()
        neg = eligible_pool[torch.randint(0, P, (B, neg_samples), device=device)]           

                               
        cand = torch.cat([tv.view(B, 1), neg], dim=1)

                                                   
        logits = score_member_candidates_batch(member_model, z, cand, z_set, step_i=step_i, k=k, ce=ce,
                                        b_static=b_static, c_static=c_static)

        if use_norepeat:
                                                                 
            invalid = selected.gather(1, cand)             
                                                                      
            invalid[:, 0] = False
            logits = logits.masked_fill(invalid, float("-inf"))

                                                 
        loss_step = F.cross_entropy(logits, torch.zeros((B,), dtype=torch.long, device=device))
        total = total + loss_step

                            
        z_sum = z_sum + z[tv]         
        cnt = cnt + 1.0
        if use_norepeat:
            selected.scatter_(1, tv.view(B, 1), True)

    return total / float(k)


def train_hyperedge_structural_predictor(model, edge_data, g_vec, device, epochs=50, lr=1e-3):
    """
    edge_data: list of ((k, ce), e_nodes)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            
    ks = []
    ces = []
    for (k, ce), _ in edge_data:
        ks.append(k)
        ces.append(ce)

    ks = torch.tensor(ks, dtype=torch.long, device=device)
    ces = torch.tensor(ces, dtype=torch.long, device=device)

                    
    g_vec_batch = g_vec.unsqueeze(0).expand(len(ks), -1)

    print(f"[HyperedgeCorePredictor] Start training on {len(ks)} edges...")
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(ks, g_vec_batch)
        loss = F.cross_entropy(logits, ces)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0:
            print(f"[HyperedgeCorePredictor] Epoch {ep + 1}/{epochs} Loss={loss.item():.4f}")

    return model.eval()


def build_member_training_bank(edge_data, c_star: torch.Tensor, device: torch.device):
                            
    buckets = {}
    for (tau, e) in edge_data:
        k, ce = int(tau[0]), int(tau[1])
        if len(e) != k or k <= 0:
            continue
        buckets.setdefault((k, ce), []).append(e)

    bank = {}
    pools = {}
    taus = []

    for tau, edges in buckets.items():
        k, ce = int(tau[0]), int(tau[1])
        E = torch.tensor(edges, device=device, dtype=torch.long)             
        if E.numel() == 0:
            continue

        pool = torch.nonzero(c_star >= ce, as_tuple=False).view(-1)           
        if pool.numel() == 0:
            continue

        bank[tau] = E
        pools[tau] = pool
        taus.append(tau)

    return bank, pools, taus


def freeze_module(m: nn.Module):
    m.eval()                                
    for p in m.parameters():
        p.requires_grad_(False)


def unfreeze_module(m: nn.Module):
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)


def train_dynamic_member_assignment(
        member_model: nn.Module,
        X: torch.Tensor,              
        d_star: torch.Tensor,                                 
        c_star: torch.Tensor,            
        edge_data,                           
        epoch_member: int = 100,
        lr: float = 2e-3,
        batch_size: int = 256,
        steps_per_epoch: int = 20,
        neg_samples: int = 256,
        mode: str = "softmax",                           
        random_order: bool = True,                                    
        use_compile: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        tau_sampling: str = "uniform",                               
):
    assert X.is_cuda and d_star.is_cuda and c_star.is_cuda
    device = X.device

                 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    member_model = member_model.to(device).train()
    if use_compile and hasattr(torch, "compile"):
        member_model = torch.compile(member_model, mode="reduce-overhead")

    opt = torch.optim.Adam(member_model.parameters(), lr=lr)
    trainable = [(n, p.numel()) for n, p in member_model.named_parameters() if p.requires_grad]
    print("[MemberAssignment] trainable param groups:", len(trainable))
    print("[MemberAssignment] trainable params total:", sum(x[1] for x in trainable))
    print("[MemberAssignment] trainable names (first 30):", [n for n, _ in trainable[:30]])
    assert len(trainable) > 0, "No trainable params! (optimizer would be empty)"

    scaler = torch.amp.GradScaler("cuda")

                     
    bank, pools, taus = build_member_training_bank(edge_data, c_star=c_star, device=device)
    if not taus:
        raise ValueError("No valid signature buckets found for dynamic member assignment training.")

                 
    if tau_sampling == "proportional":
        sizes = torch.tensor([bank[t].size(0) for t in taus], device=device, dtype=torch.float32)
        tau_probs = sizes / sizes.sum()
    else:
        tau_probs = None           

                                                    
    taus_list = list(taus)                         

    for epoch in range(epoch_member):
        loss_acc = 0.0
        updates = 0

        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)

                        
            if tau_probs is None:
                ti = int(torch.randint(0, len(taus_list), (1,), device=device).item())
            else:
                ti = int(torch.multinomial(tau_probs, 1).item())
            tau = taus_list[ti]
            E = bank[tau]              
            pool_tau = pools[tau]           

                                         
            B = min(batch_size, int(E.size(0)))
            idx = torch.randint(0, E.size(0), (B,), device=device)
            batch_edges = E.index_select(0, idx)             

                                                                                         
            k = int(batch_edges.size(1))
            prefix = torch.randint(0, max(1, k), (B,), device=device)       

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                                                      
                z = member_model.encode_nodes(X)         

                                                                                       
                loss = torch.zeros((), device=device, dtype=z.dtype)

                for j in range(B):
                    loss = loss + member_model.compute_fast_negative_sampling_loss_with_embeddings(
                        z=z,
                        b=d_star,
                        c=c_star,
                        tau=tau,
                        target_set=batch_edges[j],                  
                        pool_tau=pool_tau,                  
                        prefix_len=int(prefix[j].item()),
                        neg_samples=neg_samples,
                        mode=mode,
                        random_order=random_order,
                    )

                loss = loss / float(B)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(member_model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

            loss_acc += float(loss.detach().item())
            updates += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[MemberAssignment][Epoch {epoch + 1:03d}/{epoch_member}] "
                  f"loss_member={loss_acc / max(updates, 1):.4f} | "
                  f"steps/ep={steps_per_epoch} B={batch_size} neg={neg_samples} "
                  f"mode={mode} order={'rand' if random_order else 'fixed'} compile={use_compile}")

    return member_model.eval()