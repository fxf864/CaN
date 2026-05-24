
import argparse
import numpy as np
from math import comb
from scipy.stats import wasserstein_distance


class HypergraphEvaluator:
    def __init__(self, structure_path, attribute_path):
        self.structure_path = structure_path
        self.attribute_path = attribute_path
        self.hyperedges = []
        self.node2edge = []
        self.attrs = None
        self.load_data()

    def load_data(self):
        """Load hyperedge structure and node attributes."""
        print(f"Loading structure: {self.structure_path}")
        node_set = set()
        with open(self.structure_path, "r") as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if not parts:
                    continue
                nodes = [int(n) for n in parts]
                self.hyperedges.append(nodes)
                node_set.update(nodes)

        max_node_id = max(node_set) if node_set else 0
        self.num_nodes = max_node_id + 1
        self.node2edge = [[] for _ in range(self.num_nodes)]
        for i, edge in enumerate(self.hyperedges):
            for node in edge:
                self.node2edge[node].append(i)

        print(f"Loading attributes: {self.attribute_path}")
        attrs_raw = []
        with open(self.attribute_path, "r") as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if not parts:
                    continue
                attrs_raw.append([float(n) for n in parts])

        self.attrs = np.array(attrs_raw)
        if self.attrs.shape[0] <= max_node_id:
            padding = np.zeros((max_node_id - self.attrs.shape[0] + 1, self.attrs.shape[1]))
            self.attrs = np.vstack([self.attrs, padding])

        self.attrs[self.attrs > 0] = 1.0
        self.attrs = self.attrs.astype(int)

    def get_semantic_distributions(self):
        """Compute only the six structure--attribute consistency distributions used in the paper."""
        dists = {
            "T2": None,
            "T3": None,
            "T4": None,
            "HE": self.calculate_entropy(self.attrs),
            "HOHE": self.calculate_hohe(),
            "NHS": self.calculate_node_homophily(),
        }
        dists.update(self.calculate_affinity_scores())
        return dists

    def calculate_node_homophily(self):
        h_scores = []
        num_nodes = len(self.node2edge)

        for node in range(num_nodes):
            if len(self.node2edge[node]) == 0:
                continue

            neighbor_list = []
            for hyperedge_idx in self.node2edge[node]:
                neighbors = self.hyperedges[hyperedge_idx].copy()
                if node in neighbors:
                    neighbors.remove(node)
                neighbor_list += neighbors

            if len(neighbor_list) == 0:
                continue

            score_vec = (
                len(neighbor_list)
                - np.sum(np.abs(self.attrs[neighbor_list] - self.attrs[node]), axis=0)
            ) / len(neighbor_list)
            h_scores.append(score_vec)

        return np.array(h_scores)

    def calculate_entropy(self, attributes):
        entropies = []
        attr_dim = attributes.shape[1]

        for hyperedge in self.hyperedges:
            size = len(hyperedge)
            if size == 0:
                continue

            hyperedge_attrs = attributes[hyperedge]
            attr_sum = np.sum(hyperedge_attrs, axis=0)

            if size == 1:
                node = hyperedge[0]
                if len(self.node2edge[node]) <= 1:
                    continue
                entropy = np.zeros(attr_dim, dtype=np.float64)
            else:
                entropy = np.zeros(attr_dim, dtype=np.float64)
                p1 = attr_sum / size
                p0 = 1.0 - p1

                mask1 = p1 > 0
                entropy[mask1] -= p1[mask1] * np.log(p1[mask1])

                mask0 = p0 > 0
                entropy[mask0] -= p0[mask0] * np.log(p0[mask0])

            entropies.append(entropy)

        return np.array(entropies)

    def calculate_hohe(self):
        n = len(self.node2edge)
        m = len(self.hyperedges)
        attr_dim = self.attrs.shape[1]

        X = np.array(self.attrs, dtype=np.float64)
        Y = np.zeros((m, attr_dim), dtype=np.float64)

        for _ in range(10):
            for he in range(m):
                cur_edge = self.hyperedges[he]
                if len(cur_edge) > 0:
                    Y[he] = np.sum(X[cur_edge], axis=0) / len(cur_edge)

            for i in range(n):
                cur_edges = self.node2edge[i]
                if len(cur_edges) > 0:
                    X[i] = np.sum(Y[cur_edges], axis=0) / len(cur_edges)

        return self.calculate_entropy(X)

    def calculate_affinity_scores(self):
        active_nodes = sorted({v for e in self.hyperedges for v in e})
        if len(active_nodes) == 0:
            return {f"T{s}": np.array([]) for s in [2, 3, 4]}

        n = len(active_nodes)
        attr_dim = self.attrs.shape[1]
        active_attrs = self.attrs[active_nodes]

        one_per_feat = np.sum(active_attrs, axis=0).astype(int)
        zero_per_feat = (n - one_per_feat).astype(int)

        size_to_check = [2, 3, 4]
        type_degree_0 = {s: np.zeros((s, attr_dim)) for s in size_to_check}
        type_degree_1 = {s: np.zeros((s, attr_dim)) for s in size_to_check}
        degree_0 = {s: np.zeros(attr_dim) for s in size_to_check}
        degree_1 = {s: np.zeros(attr_dim) for s in size_to_check}

        for hyperedge in self.hyperedges:
            size = len(hyperedge)
            if size in size_to_check:
                attr_sum = np.sum(self.attrs[hyperedge], axis=0)
                for feat in range(attr_dim):
                    ones = int(attr_sum[feat])
                    zeros = int(size - ones)

                    if zeros > 0:
                        type_degree_0[size][zeros - 1][feat] += zeros
                        degree_0[size][feat] += zeros

                    if ones > 0:
                        type_degree_1[size][ones - 1][feat] += ones
                        degree_1[size][feat] += ones

        results = {}
        for size in size_to_check:
            total_comb = comb(n - 1, size - 1)
            vals_0_list = []
            vals_1_list = []

            for feat in range(attr_dim):
                for i in range(1, size + 1):
                    z = zero_per_feat[feat]
                    if (z - 1) >= (i - 1) and (n - z) >= (size - i):
                        expected_b0 = comb(z - 1, i - 1) * comb(n - z, size - i) / total_comb
                    else:
                        expected_b0 = 0

                    if degree_0[size][feat] == 0 or expected_b0 == 0:
                        val_0 = 0.0
                    else:
                        obs_p0 = type_degree_0[size][i - 1][feat] / degree_0[size][feat]
                        val_0 = obs_p0 / expected_b0
                    vals_0_list.append(val_0)

                for i in range(1, size + 1):
                    o = one_per_feat[feat]
                    if (o - 1) >= (i - 1) and (n - o) >= (size - i):
                        expected_b1 = comb(o - 1, i - 1) * comb(n - o, size - i) / total_comb
                    else:
                        expected_b1 = 0

                    if degree_1[size][feat] == 0 or expected_b1 == 0:
                        val_1 = 0.0
                    else:
                        obs_p1 = type_degree_1[size][i - 1][feat] / degree_1[size][feat]
                        val_1 = obs_p1 / expected_b1
                    vals_1_list.append(val_1)

            results[f"T{size}"] = np.array(vals_0_list + vals_1_list)

        return results


def compute_metric_diff(gt_dist, gen_dist, metric_name):
    if gt_dist is None or gen_dist is None or len(gt_dist) == 0 or len(gen_dist) == 0:
        return np.nan

    if metric_name.startswith("T"):
        if gt_dist.shape != gen_dist.shape:
            min_len = min(len(gt_dist), len(gen_dist))
            gt_dist = gt_dist[:min_len]
            gen_dist = gen_dist[:min_len]
        return np.sum(np.abs(np.log1p(gt_dist) - np.log1p(gen_dist)))

                                                                    
    if gt_dist.ndim == 1:
        gt_dist = gt_dist.reshape(-1, 1)
    if gen_dist.ndim == 1:
        gen_dist = gen_dist.reshape(-1, 1)

    num_attrs = gt_dist.shape[1]
    if gen_dist.shape[1] != num_attrs:
        min_dim = min(num_attrs, gen_dist.shape[1])
        gt_dist = gt_dist[:, :min_dim]
        gen_dist = gen_dist[:, :min_dim]
        num_attrs = min_dim

    total_wd = 0.0
    for k in range(num_attrs):
        total_wd += wasserstein_distance(gt_dist[:, k], gen_dist[:, k])
    return total_wd


def main():
    parser = argparse.ArgumentParser(description="Structure--attribute consistency evaluation")
    parser.add_argument("--gt_structure", type=str, required=True)
    parser.add_argument("--gt_attribute", type=str, required=True)
    parser.add_argument("--gen_structure", type=str, required=True)
    parser.add_argument("--gen_attribute", type=str, required=False)
    args = parser.parse_args()

    gen_attr_path = args.gen_attribute if args.gen_attribute else args.gt_attribute

    print("\n" + "=" * 50)
    print("Step 1: Processing Ground Truth Hypergraph...")
    gt_eval = HypergraphEvaluator(args.gt_structure, args.gt_attribute)
    gt_dists = gt_eval.get_semantic_distributions()

    print("\n" + "=" * 50)
    print("Step 2: Processing Generated Hypergraph...")
    gen_eval = HypergraphEvaluator(args.gen_structure, gen_attr_path)
    gen_dists = gen_eval.get_semantic_distributions()

    metrics = ["T2", "T3", "T4", "HE", "HOHE", "NHS"]

    print(f"\n{'Metric':<10} | {'Error Value':<15}")
    print("-" * 30)
    results = {}
    for m in metrics:
        error = compute_metric_diff(gt_dists.get(m), gen_dists.get(m), m)
        results[m] = error
        print(f"{m:<10} | {error:.4f}")

    print("\n" + "###" * 10)
    print("FINAL CSV HEADER")
    print(",".join(metrics))
    print("FINAL CSV ROW")
    print(",".join([f"{results[m]:.4f}" if not np.isnan(results[m]) else "NaN" for m in metrics]))


if __name__ == "__main__":
    main()
