from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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


def sample_negative_edges(
    num_nodes: int,
    full_edge_set: set[Edge],
    num_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample negative edges from non-edges of the full graph.

    Important: use full_edge_set, not train_edge_set, to avoid treating val/test positives as negatives.
    """
    rng = np.random.default_rng(seed)
    neg_edges: set[Edge] = set()

    while len(neg_edges) < num_samples:
        need = num_samples - len(neg_edges)
        us = rng.integers(0, num_nodes, size=need * 3)
        vs = rng.integers(0, num_nodes, size=need * 3)

        for u, v in zip(us, vs):
            if int(u) == int(v):
                continue
            edge = normalize_edge(int(u), int(v))
            if edge in full_edge_set or edge in neg_edges:
                continue
            neg_edges.add(edge)
            if len(neg_edges) >= num_samples:
                break

    return np.array(sorted(neg_edges), dtype=np.int64)


def make_split(
    edges: np.ndarray,
    num_nodes: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> EdgeSplit:
    full_edge_set = build_edge_set(edges)
    train_edges, val_edges, test_edges = split_edges_preserve_connectivity(
        edges=edges,
        num_nodes=num_nodes,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
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
