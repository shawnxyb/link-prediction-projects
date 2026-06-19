from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

Edge = Tuple[int, int]


@dataclass
class EdgeSplit:
    train_edges: np.ndarray
    val_edges: np.ndarray
    test_edges: np.ndarray
    val_neg_edges: np.ndarray
    test_neg_edges: np.ndarray
    num_nodes: int


def normalize_edge(u: int, v: int) -> Edge:
    if u > v:
        u, v = v, u
    return int(u), int(v)


def read_edges(path: str) -> tuple[np.ndarray, int, Dict[int, int]]:
    """
    Read SNAP facebook_combined.txt.

    Each line is an undirected edge: u v.
    Returned edges are remapped to [0, num_nodes - 1] and stored as u < v.
    """
    raw_edges: List[Edge] = []
    node_set: Set[int] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            if u == v:
                continue
            node_set.add(u)
            node_set.add(v)
            raw_edges.append(normalize_edge(u, v))

    raw_edges = sorted(set(raw_edges))
    nodes = sorted(node_set)
    node2id = {node: idx for idx, node in enumerate(nodes)}

    remapped_edges = []
    for u, v in raw_edges:
        a, b = node2id[u], node2id[v]
        remapped_edges.append(normalize_edge(a, b))

    remapped_edges = sorted(set(remapped_edges))
    return np.array(remapped_edges, dtype=np.int64), len(nodes), node2id


class DSU:
    """Disjoint Set Union for preserving a spanning forest in train graph."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def split_edges_preserve_connectivity(
    edges: np.ndarray,
    num_nodes: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split positive edges into train / validation / test.

    A spanning forest is first kept in train set to reduce isolated nodes.
    """
    rng = np.random.default_rng(seed)
    m = len(edges)
    indices = np.arange(m)
    rng.shuffle(indices)

    dsu = DSU(num_nodes)
    tree_idx: list[int] = []
    non_tree_idx: list[int] = []

    for idx in indices:
        u, v = edges[idx]
        if dsu.union(int(u), int(v)):
            tree_idx.append(int(idx))
        else:
            non_tree_idx.append(int(idx))

    non_tree_idx = np.array(non_tree_idx, dtype=np.int64)
    rng.shuffle(non_tree_idx)

    num_val = int(m * val_ratio)
    num_test = int(m * test_ratio)

    if num_val + num_test > len(non_tree_idx):
        raise ValueError("val_ratio + test_ratio too large after preserving connectivity.")

    val_idx = non_tree_idx[:num_val]
    test_idx = non_tree_idx[num_val:num_val + num_test]
    train_extra_idx = non_tree_idx[num_val + num_test:]
    train_idx = np.array(tree_idx + train_extra_idx.tolist(), dtype=np.int64)

    return edges[train_idx], edges[val_idx], edges[test_idx]


def build_edge_set(edges: np.ndarray) -> set[Edge]:
    return {normalize_edge(int(u), int(v)) for u, v in edges}


def build_neighbor_sets(num_nodes: int, edges: np.ndarray) -> list[set[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edges:
        u, v = int(u), int(v)
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def sample_negative_edges(
    num_nodes: int,
    full_edge_set: set[Edge],
    num_samples: int,
    seed: int | None = None,
    exclude_edges: Optional[set[Edge]] = None,
) -> np.ndarray:
    """
    Sample random negative edges from non-edges of the full graph.

    Important: use full_edge_set, not train_edge_set, to avoid treating val/test positives as negatives.
    """
    rng = np.random.default_rng(seed)
    neg_edges: set[Edge] = set()
    exclude_edges = exclude_edges or set()

    while len(neg_edges) < num_samples:
        need = num_samples - len(neg_edges)
        us = rng.integers(0, num_nodes, size=max(need * 3, 1000))
        vs = rng.integers(0, num_nodes, size=max(need * 3, 1000))

        for u, v in zip(us, vs):
            if int(u) == int(v):
                continue
            edge = normalize_edge(int(u), int(v))
            if edge in full_edge_set or edge in neg_edges or edge in exclude_edges:
                continue
            neg_edges.add(edge)
            if len(neg_edges) >= num_samples:
                break

    return np.array(sorted(neg_edges), dtype=np.int64)


def build_two_hop_negative_pool(
    num_nodes: int,
    train_edges: np.ndarray,
    full_edge_set: set[Edge],
) -> np.ndarray:
    """
    Build hard negative pool: two-hop node pairs in train graph but no true edge in full graph.

    These pairs have at least one common neighbor, so they are harder than random non-edges.
    """
    neighbors = build_neighbor_sets(num_nodes, train_edges)
    pool: set[Edge] = set()

    for u in range(num_nodes):
        two_hop: set[int] = set()
        for mid in neighbors[u]:
            two_hop.update(neighbors[mid])
        two_hop.discard(u)
        two_hop.difference_update(neighbors[u])

        for v in two_hop:
            edge = normalize_edge(u, v)
            if edge not in full_edge_set:
                pool.add(edge)

    if not pool:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(pool), dtype=np.int64)


def sample_edges_from_pool(pool: np.ndarray, num_samples: int, seed: int | None = None) -> np.ndarray:
    if len(pool) == 0:
        raise ValueError("Cannot sample from an empty negative pool.")
    rng = np.random.default_rng(seed)
    replace = len(pool) < num_samples
    idx = rng.choice(len(pool), size=num_samples, replace=replace)
    return pool[idx].astype(np.int64)


def sample_mixed_negative_edges(
    num_nodes: int,
    full_edge_set: set[Edge],
    hard_pool: np.ndarray,
    num_samples: int,
    hard_ratio: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    hard_ratio = min(max(float(hard_ratio), 0.0), 1.0)
    num_hard = int(num_samples * hard_ratio)
    num_random = num_samples - num_hard

    parts = []
    used: set[Edge] = set()

    if num_hard > 0 and len(hard_pool) > 0:
        hard_edges = sample_edges_from_pool(hard_pool, num_hard, seed=seed)
        parts.append(hard_edges)
        used.update(build_edge_set(hard_edges))

    if num_random > 0:
        random_edges = sample_negative_edges(
            num_nodes=num_nodes,
            full_edge_set=full_edge_set,
            num_samples=num_random,
            seed=None if seed is None else seed + 1,
            exclude_edges=used,
        )
        parts.append(random_edges)

    if not parts:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(parts).astype(np.int64)


def build_hard_candidates_by_anchor(
    num_nodes: int,
    hard_pool: np.ndarray,
) -> list[np.ndarray]:
    candidates: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in hard_pool:
        u, v = int(u), int(v)
        candidates[u].append(v)
        candidates[v].append(u)
    return [np.array(sorted(set(xs)), dtype=np.int64) for xs in candidates]


def sample_negative_nodes_for_anchors(
    anchors: np.ndarray,
    num_nodes: int,
    full_edge_set: set[Edge],
    hard_candidates: Optional[list[np.ndarray]],
    neg_type: str,
    hard_ratio: float,
    seed: int,
) -> np.ndarray:
    """
    Sample one negative node for each anchor for BPR training.

    random: any node that is not itself and not a full-graph neighbor
    hard: two-hop non-edge if available, otherwise fallback to random
    mixed: use hard negatives for a fraction of anchors, random for the rest
    """
    rng = np.random.default_rng(seed)
    neg_nodes = np.empty(len(anchors), dtype=np.int64)
    neg_type = neg_type.lower()

    for i, u in enumerate(anchors):
        u = int(u)
        use_hard = False
        if neg_type == "hard":
            use_hard = True
        elif neg_type == "mixed":
            use_hard = rng.random() < hard_ratio

        if use_hard and hard_candidates is not None and len(hard_candidates[u]) > 0:
            neg_nodes[i] = int(rng.choice(hard_candidates[u]))
            continue

        # Random fallback.
        while True:
            v = int(rng.integers(0, num_nodes))
            if v == u:
                continue
            if normalize_edge(u, v) in full_edge_set:
                continue
            neg_nodes[i] = v
            break

    return neg_nodes


def make_split(
    edges: np.ndarray,
    num_nodes: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    eval_neg_type: str = "random",
) -> EdgeSplit:
    full_edge_set = build_edge_set(edges)
    train_edges, val_edges, test_edges = split_edges_preserve_connectivity(
        edges=edges,
        num_nodes=num_nodes,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    eval_neg_type = eval_neg_type.lower()
    if eval_neg_type == "hard":
        hard_pool = build_two_hop_negative_pool(num_nodes, train_edges, full_edge_set)
        val_neg_edges = sample_edges_from_pool(hard_pool, len(val_edges), seed=seed + 100)
        # Avoid exact overlap with validation negatives when possible.
        val_neg_set = build_edge_set(val_neg_edges)
        remaining = np.array([e for e in hard_pool if normalize_edge(int(e[0]), int(e[1])) not in val_neg_set], dtype=np.int64)
        test_pool = remaining if len(remaining) >= len(test_edges) else hard_pool
        test_neg_edges = sample_edges_from_pool(test_pool, len(test_edges), seed=seed + 200)
    else:
        val_neg_edges = sample_negative_edges(
            num_nodes=num_nodes,
            full_edge_set=full_edge_set,
            num_samples=len(val_edges),
            seed=seed + 100,
        )
        test_neg_edges = sample_negative_edges(
            num_nodes=num_nodes,
            full_edge_set=full_edge_set,
            num_samples=len(test_edges),
            seed=seed + 200,
        )

    return EdgeSplit(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        val_neg_edges=val_neg_edges,
        test_neg_edges=test_neg_edges,
        num_nodes=num_nodes,
    )


def save_split(split: EdgeSplit, path: str) -> None:
    np.savez(
        path,
        train_edges=split.train_edges,
        val_edges=split.val_edges,
        test_edges=split.test_edges,
        val_neg_edges=split.val_neg_edges,
        test_neg_edges=split.test_neg_edges,
        num_nodes=np.array([split.num_nodes]),
    )


def load_split(path: str) -> EdgeSplit:
    data = np.load(path)
    return EdgeSplit(
        train_edges=data["train_edges"].astype(np.int64),
        val_edges=data["val_edges"].astype(np.int64),
        test_edges=data["test_edges"].astype(np.int64),
        val_neg_edges=data["val_neg_edges"].astype(np.int64),
        test_neg_edges=data["test_neg_edges"].astype(np.int64),
        num_nodes=int(data["num_nodes"][0]),
    )
