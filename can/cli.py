from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .data import (
    fix_edge_indexing,
    load_hyperedges,
    load_node_attributes,
    replicate_edges,
    select_or_pad_attributes,
)
from .generation import HypergraphGenerator, generate_hypergraph, save_checkpoint, save_edges
from .hypergraph import Hypergraph, compute_global_stats_from_real_hypergraph, make_g_vector
from .models import AutoregressiveMemberAssigner, HyperedgeCorePredictor
from .training import (
    TrainConfig,
    build_hyperedge_training_data,
    train_dynamic_member_assignment,
    train_hyperedge_structural_predictor,
    train_structural_feature_allocator,
)
from .utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CaN attributed hypergraph generation")
    parser.add_argument("--attr_path", type=str, required=True, help="Node attribute file")
    parser.add_argument("--out_edge_path", type=str, required=True, help="Generated hyperedge output path")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "gen"], help="Execution mode")
    parser.add_argument("--model_path", type=str, default="saved_model.pt", help="Checkpoint path")
    parser.add_argument("--edge_path", type=str, default=None, help="Training hyperedge file")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of generated samples in generation mode")
    parser.add_argument("--dmax", type=int, default=None)
    parser.add_argument("--cmax", type=int, default=None)
    parser.add_argument("--kmax", type=int, default=None)
    parser.add_argument("--min_edge_size_active", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--edge_mult", type=int, default=1, help="Replicate hyperedges for scalability experiments")
    parser.add_argument("--attr_dim", type=int, default=None, help="Override the attribute dimension")
    parser.add_argument("--attr_select_mode", type=str, default="head", choices=["head", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_edges_each_rep", action="store_true")
    parser.add_argument("--max_edges", type=int, default=None)
    parser.add_argument("--epoch_node", type=int, default=500, help="Training epochs for the structural feature allocator")
    parser.add_argument("--epoch_member", type=int, default=500, help="Training epochs for the member assignment model")
    parser.add_argument("--epoch_edge", type=int, default=500, help="Training epochs for the hyperedge structural predictor")
    parser.add_argument("--member_batch_size", type=int, default=256)
    parser.add_argument("--member_steps_per_epoch", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for distribution alignment")
    parser.add_argument("--beta", type=float, default=10.0, help="Weight for stub consistency")
    parser.add_argument("--lam_attr", type=float, default=0.1, help="Weight for attribute reconstruction")
    parser.add_argument("--bench_json", type=str, default=None, help="Optional JSON output for timing results")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = {}
    total_start = time.perf_counter()

    print(f"Loading attributes from {args.attr_path}...")
    X = load_node_attributes(args.attr_path).to(device)
    if args.attr_dim is not None:
        X = select_or_pad_attributes(X, target_dim=args.attr_dim, mode=args.attr_select_mode, seed=args.seed)
        print(f"[AttrDim] Using attr_dim={X.size(1)} (mode={args.attr_select_mode})")

    if args.mode == "train":
        if not args.edge_path:
            raise ValueError("--edge_path is required in train mode")

        edges_raw = load_hyperedges(args.edge_path)
        edges, msg = fix_edge_indexing(edges_raw, num_nodes=X.size(0))
        print(msg)
        if args.max_edges is not None:
            edges = edges[: args.max_edges]
            print(f"[MaxEdges] Using first {len(edges)} hyperedges")
        if args.edge_mult and args.edge_mult > 1:
            edges = replicate_edges(edges, mult=args.edge_mult, seed=args.seed, shuffle_each=args.shuffle_edges_each_rep)
            print(f"[EdgeMult] Replicated edges by x{args.edge_mult}: |E|={len(edges)}")

        hypergraph = Hypergraph(num_nodes=X.size(0), edges=edges, x=X)
        fit_start = time.perf_counter()
        g_star, d_star_list, c_star_list, ce_star_list = compute_global_stats_from_real_hypergraph(
            hypergraph,
            dmax=args.dmax,
            cmax=args.cmax,
            kmax=args.kmax,
            min_edge_size_active=args.min_edge_size_active,
        )
        print(f"Computed g*: |E|={g_star.num_edges}, S*={g_star.stub_total}")

        cfg = TrainConfig(
            hidden=args.hidden,
            lr=args.lr,
            epoch_node=args.epoch_node,
            epoch_member=args.epoch_member,
            member_batch_size=args.member_batch_size,
            member_steps_per_epoch=args.member_steps_per_epoch,
            alpha=args.alpha,
            beta=args.beta,
            lam_attr=args.lam_attr,
            device=device,
        )
        print(
            f"Training Config: alpha={cfg.alpha}, beta={cfg.beta}, "
            f"lam_attr={cfg.lam_attr}, device={cfg.device}"
        )

        edge_data = build_hyperedge_training_data(hypergraph, g_star, ce_star_list)

        allocator, d_star, c_star, g_vec = train_structural_feature_allocator(
            hypergraph, g_star, d_star_list, c_star_list, cfg
        )

        print("[Main] Training hyperedge structural predictor...")
        core_predictor = HyperedgeCorePredictor(
            kmax=g_star.kmax, cmax=g_star.cmax, g_dim=g_vec.numel(), hidden=cfg.hidden
        )
        core_predictor = train_hyperedge_structural_predictor(
            core_predictor, edge_data, g_vec, cfg.device, epochs=args.epoch_edge, lr=args.lr
        )

        print("[Main] Training dynamic member assignment model...")
        member_model = AutoregressiveMemberAssigner(
            x_dim=X.size(1), hidden=cfg.hidden, kmax=g_star.kmax, cmax=g_star.cmax
        ).to(cfg.device)
        member_model = train_dynamic_member_assignment(
            member_model=member_model,
            X=hypergraph.x.to(cfg.device),
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

        save_checkpoint(args.model_path, allocator, member_model, core_predictor, g_star, cfg, x_dim=X.size(1))
        fitting_time = time.perf_counter() - fit_start
        bench["fitting_time"] = fitting_time
        print(f"[Bench] fitting_time = {fitting_time:.6f} seconds")

        print("\n[Verification] Generating 1 sample after training...")
        pred_start = time.perf_counter()
        gen_edges = generate_hypergraph(
            hypergraph.x, g_star, allocator, member_model, core_predictor, cfg, temperature=args.temperature
        )
        predict_time = time.perf_counter() - pred_start
        bench["predict_time"] = predict_time
        print(f"[Bench] predict_time = {predict_time:.6f} seconds")
        save_edges(gen_edges, args.out_edge_path)
        print(f"Saved generated edges to: {args.out_edge_path}")

    else:
        generator = HypergraphGenerator(args.model_path, device=device)
        base_name, ext = os.path.splitext(args.out_edge_path)
        for i in range(args.num_samples):
            print(f"\n--- Generating Sample {i + 1}/{args.num_samples} ---")
            gen_edges = generator.generate(X, temperature=args.temperature)
            out_name = f"{base_name}_{i}{ext}" if args.num_samples > 1 else args.out_edge_path
            save_edges(gen_edges, out_name)
            print(f"Saved sample {i + 1} to {out_name}")

    bench["total_time"] = time.perf_counter() - total_start
    if args.bench_json:
        os.makedirs(os.path.dirname(args.bench_json) or ".", exist_ok=True)
        with open(args.bench_json, "w", encoding="utf-8") as f:
            json.dump(bench, f, indent=2)


if __name__ == "__main__":
    main()
