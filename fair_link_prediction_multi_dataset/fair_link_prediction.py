#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair comparison for link prediction on SNAP network datasets.

Supported datasets:
  facebook       SNAP ego-Facebook, undirected social network
  ca-grqc        SNAP arXiv GR-QC collaboration network, undirected
  email-eu-core  SNAP email-Eu-core, directed communication network converted to undirected

Methods: CN, AA, RA, Jaccard, PA, HDI, HPI, MultiHop, Katz, DeepWalk, MF, MLP, GCN, GAT, LightGCN.

Output:
  results_raw.csv      each seed/method metrics
  results_summary.csv  mean/std by method; sorted by user-level NDCG@10
"""

import argparse
import copy
import gzip
import math
import os
import random
import time
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import scipy.sparse as sp
except Exception as e:
    sp = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

Edge = Tuple[int, int]

DATASETS = {
    "facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "filename": "facebook_combined.txt.gz",
        "directed": False,
        "display": "SNAP ego-Facebook",
    },
    "ca-grqc": {
        "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "filename": "ca-GrQc.txt.gz",
        "directed": False,
        "display": "SNAP ca-GrQc collaboration",
    },
    "email-eu-core": {
        "url": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "filename": "email-Eu-core.txt.gz",
        "directed": True,
        "display": "SNAP email-Eu-core converted to undirected",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def download_and_load_graph(data_dir: str, dataset: str, keep_all_components: bool = False) -> nx.Graph:
    """Download a SNAP edge-list dataset and return an undirected NetworkX graph.

    Directed datasets are symmetrized so the task remains consistent with
    undirected link prediction and all heuristic methods are comparable.
    By default, the largest connected component is used. This avoids many
    trivial negative samples from disconnected components.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {sorted(DATASETS)}")

    info = DATASETS[dataset]
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, info["filename"])
    if not os.path.exists(path):
        print(f"Downloading {info['url']} -> {path}")
        urllib.request.urlretrieve(info["url"], path)

    edges = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges.append((u, v))

    # Use an undirected graph for all methods. For directed datasets, this means
    # there is an undirected edge if either direction appears in the raw data.
    G_raw = nx.Graph()
    G_raw.add_edges_from(edges)
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))

    if G_raw.number_of_nodes() == 0 or G_raw.number_of_edges() == 0:
        raise ValueError(f"Loaded empty graph from {path}")

    if not keep_all_components and not nx.is_connected(G_raw):
        largest = max(nx.connected_components(G_raw), key=len)
        G_raw = G_raw.subgraph(largest).copy()

    # Relabel to 0..n-1 to make torch/scipy indexing safe.
    nodes = sorted(G_raw.nodes())
    mp = {old: i for i, old in enumerate(nodes)}
    G = nx.relabel_nodes(G_raw, mp, copy=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def edge_set_from_graph(G: nx.Graph) -> Set[Edge]:
    return {canonical_edge(u, v) for u, v in G.edges()}


def connected_preserving_split(
    G: nx.Graph,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Edge], List[Edge], List[Edge]]:
    """Split edges while keeping a spanning forest in the training graph."""
    rng = np.random.default_rng(seed)
    all_edges = [canonical_edge(u, v) for u, v in G.edges()]
    all_edges_set = set(all_edges)

    forest_edges: Set[Edge] = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        T = nx.minimum_spanning_tree(sub)
        forest_edges.update(canonical_edge(u, v) for u, v in T.edges())

    movable = list(all_edges_set - forest_edges)
    rng.shuffle(movable)
    n_edges = len(all_edges)
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)
    if n_test + n_val > len(movable):
        raise ValueError("val_ratio + test_ratio is too large to preserve connectivity.")

    test_edges = movable[:n_test]
    val_edges = movable[n_test : n_test + n_val]
    train_edges = list((all_edges_set - set(test_edges) - set(val_edges)))
    return train_edges, val_edges, test_edges


def make_graph_from_edges(n: int, edges: Sequence[Edge]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def sample_non_edges(
    n: int,
    forbidden_edges: Set[Edge],
    num_samples: int,
    rng: np.random.Generator,
) -> List[Edge]:
    """Uniformly sample unordered node pairs that are not in forbidden_edges."""
    ans: Set[Edge] = set()
    max_possible = n * (n - 1) // 2 - len(forbidden_edges)
    if num_samples > max_possible:
        raise ValueError("Requested too many non-edges.")
    while len(ans) < num_samples:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u == v:
            continue
        e = canonical_edge(u, v)
        if e in forbidden_edges or e in ans:
            continue
        ans.add(e)
    return list(ans)


def build_user_eval_pairs(
    pos_edges: Sequence[Edge],
    n: int,
    full_edge_set: Set[Edge],
    neg_per_pos: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build user-level recommendation candidates for Top-K evaluation.

    For an undirected held-out edge (u, v), we create two positive
    recommendation targets: user u -> v and user v -> u. For each user,
    negatives are nodes that are not connected to this user in the original
    full graph, so train/val/test true friends are never used as negatives.

    groups[i] is the source user id of pair i. Metrics are averaged over users,
    not over individual edges.
    """
    positives_by_user: Dict[int, Set[int]] = defaultdict(set)
    for u, v in pos_edges:
        u, v = int(u), int(v)
        positives_by_user[u].add(v)
        positives_by_user[v].add(u)

    pairs: List[Tuple[int, int]] = []
    labels: List[int] = []
    groups: List[int] = []

    for u in sorted(positives_by_user):
        pos_targets = sorted(positives_by_user[u])

        for v in pos_targets:
            pairs.append((u, v))
            labels.append(1)
            groups.append(u)

        # Candidate negatives: not self and not any true edge in the original graph.
        # Building this list per evaluated user is acceptable for the supported small/medium datasets.
        available_neg = [
            x for x in range(n)
            if x != u and canonical_edge(u, x) not in full_edge_set
        ]
        need = neg_per_pos * len(pos_targets)
        if need > len(available_neg):
            # Very high-degree users may not have enough non-friends. Use all available
            # negatives instead of sampling with replacement, to avoid duplicate candidates.
            need = len(available_neg)
        if need > 0:
            neg_targets = rng.choice(available_neg, size=need, replace=False)
            for x in neg_targets:
                pairs.append((u, int(x)))
                labels.append(0)
                groups.append(u)

    return np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def _dcg_at_k(binary_relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(binary_relevance[:k], dtype=np.float64)
    if len(rel) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2, dtype=np.float64))
    return float(np.sum(rel * discounts))


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    top_ks: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """
    Evaluate globally with AUC/AP, and evaluate Top-K per user.

    Per-user metrics:
      Hit@K    = 1 if the user has at least one held-out friend in top K.
      Recall@K = # held-out friends in top K / # held-out friends of this user.
      NDCG@K   = ranking quality with positions discounted by log2(rank+1).
      Precision@K is also kept for reference.
    """
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    top_ks = sorted(set(int(k) for k in top_ks))

    out: Dict[str, float] = {}
    try:
        out["AUC"] = float(roc_auc_score(labels, scores))
    except Exception:
        out["AUC"] = float("nan")
    try:
        out["AP"] = float(average_precision_score(labels, scores))
    except Exception:
        out["AP"] = float("nan")

    # User-level Top-K metrics. Each group is one source user and may contain
    # multiple held-out positives.
    per_k = {k: {"Precision": [], "Recall": [], "Hit": [], "NDCG": []} for k in top_ks}
    rr = []
    num_eval_users = 0
    num_eval_pairs = 0

    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]
        local_scores = scores[idx]
        local_labels = labels[idx].astype(np.int64)
        num_pos = int(local_labels.sum())
        if num_pos <= 0:
            continue

        num_eval_users += 1
        num_eval_pairs += len(idx)
        order = np.argsort(-local_scores)
        ranked_labels = local_labels[order]

        first_pos = np.where(ranked_labels == 1)[0]
        if len(first_pos) > 0:
            rr.append(1.0 / (int(first_pos[0]) + 1))

        for k in top_ks:
            kk = min(k, len(ranked_labels))
            top_rel = ranked_labels[:kk]
            hit_count = int(top_rel.sum())
            ideal = np.ones(min(num_pos, kk), dtype=np.float64)
            idcg = _dcg_at_k(ideal, kk)
            dcg = _dcg_at_k(ranked_labels, kk)

            per_k[k]["Precision"].append(hit_count / max(kk, 1))
            per_k[k]["Recall"].append(hit_count / num_pos)
            per_k[k]["Hit"].append(1.0 if hit_count > 0 else 0.0)
            per_k[k]["NDCG"].append(dcg / idcg if idcg > 0 else 0.0)

    out["MRR"] = float(np.mean(rr)) if rr else float("nan")
    for k in top_ks:
        for name in ["Precision", "Recall", "Hit", "NDCG"]:
            values = per_k[k][name]
            out[f"{name}@{k}"] = float(np.mean(values)) if values else float("nan")
    out["eval_users"] = float(num_eval_users)
    out["eval_pairs"] = float(num_eval_pairs)
    return out


# ----------------------------- Heuristic methods -----------------------------

def make_neighbor_cache(G: nx.Graph) -> Tuple[List[Set[int]], np.ndarray]:
    n = G.number_of_nodes()
    nbrs = [set(G.neighbors(i)) for i in range(n)]
    deg = np.asarray([len(nbrs[i]) for i in range(n)], dtype=np.float64)
    return nbrs, deg


def score_heuristic(method: str, G_train: nx.Graph, pairs: np.ndarray) -> np.ndarray:
    nbrs, deg = make_neighbor_cache(G_train)
    method = method.upper()
    ans = np.zeros(len(pairs), dtype=np.float64)

    for i, (u, v) in enumerate(pairs):
        Nu, Nv = nbrs[int(u)], nbrs[int(v)]
        inter = Nu & Nv
        cn = len(inter)

        if method in {"CN", "COMMON_FRIENDS"}:
            s = cn
        elif method == "JACCARD":
            union = len(Nu | Nv)
            s = cn / union if union > 0 else 0.0
        elif method in {"AA", "ADAMIC_ADAR"}:
            s = sum(1.0 / math.log(deg[w]) for w in inter if deg[w] > 1)
        elif method == "RA":
            s = sum(1.0 / deg[w] for w in inter if deg[w] > 0)
        elif method == "PA":
            s = deg[u] * deg[v]
        elif method == "HDI":
            s = cn / max(deg[u], deg[v]) if max(deg[u], deg[v]) > 0 else 0.0
        elif method == "HPI":
            s = cn / min(deg[u], deg[v]) if min(deg[u], deg[v]) > 0 else 0.0
        else:
            raise ValueError(f"Unknown heuristic: {method}")
        ans[i] = s
    return ans


def sparse_adj(n: int, edges: Sequence[Edge]):
    if sp is None:
        raise ImportError("scipy is required for MultiHop/Katz.")
    rows, cols = [], []
    for u, v in edges:
        rows += [u, v]
        cols += [v, u]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def sparse_pair_values(M, pairs: np.ndarray) -> np.ndarray:
    vals = M[pairs[:, 0], pairs[:, 1]]
    return np.asarray(vals).reshape(-1).astype(np.float64)


def score_multihop_and_katz(n: int, train_edges: Sequence[Edge], pairs: np.ndarray, beta: float = 0.005):
    A = sparse_adj(n, train_edges)
    A2 = A @ A
    A3 = A2 @ A
    a2 = sparse_pair_values(A2, pairs)
    a3 = sparse_pair_values(A3, pairs)
    return {
        "MultiHop_A2_A3": a2 + 0.1 * a3,
        "Katz_trunc3": (beta ** 2) * a2 + (beta ** 3) * a3,
    }


# ------------------------------- Pair features --------------------------------

def pair_feature_matrix(G_train: nx.Graph, pairs: np.ndarray) -> np.ndarray:
    nbrs, deg = make_neighbor_cache(G_train)
    feats = np.zeros((len(pairs), 9), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        Nu, Nv = nbrs[u], nbrs[v]
        inter = Nu & Nv
        cn = len(inter)
        union = len(Nu | Nv)
        aa = sum(1.0 / math.log(deg[w]) for w in inter if deg[w] > 1)
        ra = sum(1.0 / deg[w] for w in inter if deg[w] > 0)
        pa = deg[u] * deg[v]
        jaccard = cn / union if union > 0 else 0.0
        hdi = cn / max(deg[u], deg[v]) if max(deg[u], deg[v]) > 0 else 0.0
        hpi = cn / min(deg[u], deg[v]) if min(deg[u], deg[v]) > 0 else 0.0
        feats[i] = [deg[u], deg[v], abs(deg[u] - deg[v]), cn, jaccard, aa, ra, pa, hdi + hpi]
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ------------------------------- Torch models ---------------------------------

class PairMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MFModel(nn.Module):
    def __init__(self, n_nodes: int, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def encode(self):
        return self.emb.weight

    def forward(self, pairs):
        z = self.encode()
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=1)


class LightGCNModel(nn.Module):
    def __init__(self, n_nodes: int, adj_norm, dim: int = 64, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.adj_norm = adj_norm.coalesce()
        self.layers = layers

    def encode(self):
        z = self.emb.weight
        outs = [z]
        for _ in range(self.layers):
            z = torch.sparse.mm(self.adj_norm, z)
            outs.append(z)
        return torch.stack(outs, dim=0).mean(dim=0)

    def forward(self, pairs):
        z = self.encode()
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=1)


class GCNModel(nn.Module):
    def __init__(self, n_nodes: int, adj_norm, dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.adj_norm = adj_norm.coalesce()
        self.dropout = dropout
        nn.init.xavier_uniform_(self.emb.weight)

    def encode(self):
        h = self.emb.weight
        h = torch.sparse.mm(self.adj_norm, h)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(self.adj_norm, h)
        h = self.lin2(h)
        return h

    def forward(self, pairs):
        z = self.encode()
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=1)


class SparseGATModel(nn.Module):
    def __init__(self, n_nodes: int, edge_index, dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.n_nodes = n_nodes
        self.edge_index = edge_index.long()
        self.emb = nn.Embedding(n_nodes, dim)
        self.W = nn.Linear(dim, hidden_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(hidden_dim))
        self.a_dst = nn.Parameter(torch.empty(hidden_dim))
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = dropout
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_uniform_(self.a_src.view(1, -1))
        nn.init.xavier_uniform_(self.a_dst.view(1, -1))

    def encode(self):
        # edge_index[0] = src neighbor j, edge_index[1] = dst center i
        src, dst = self.edge_index[0], self.edge_index[1]
        h = self.W(self.emb.weight)
        e = self.leaky((h[src] * self.a_src).sum(-1) + (h[dst] * self.a_dst).sum(-1))

        # Stable segment softmax over incoming edges of each dst node.
        max_per_dst = torch.full((self.n_nodes,), -1e30, dtype=e.dtype, device=e.device)
        try:
            max_per_dst.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        except Exception:
            # Fallback for old PyTorch; slower but acceptable for this graph.
            for node in torch.unique(dst).tolist():
                mask = dst == node
                max_per_dst[node] = e[mask].max()

        exp_e = torch.exp(e - max_per_dst[dst])
        denom = torch.zeros((self.n_nodes,), dtype=e.dtype, device=e.device)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((self.n_nodes, h.size(1)), dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        out = F.elu(out)
        out = self.out(out)
        return out

    def forward(self, pairs):
        z = self.encode()
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=1)


def torch_adj_norm(n: int, edges: Sequence[Edge], device: str, add_self_loops: bool = True):
    rows, cols = [], []
    for u, v in edges:
        rows += [u, v]
        cols += [v, u]
    if add_self_loops:
        rows += list(range(n))
        cols += list(range(n))
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.ones(len(rows), dtype=torch.float32, device=device)
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.index_add_(0, idx[0], vals)
    norm_vals = vals / torch.sqrt(deg[idx[0]] * deg[idx[1]] + 1e-12)
    return torch.sparse_coo_tensor(idx, norm_vals, (n, n), device=device).coalesce()


def torch_edge_index_with_self_loops(n: int, edges: Sequence[Edge], device: str):
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    src += list(range(n))
    dst += list(range(n))
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def train_torch_model(
    model,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    val_pairs: np.ndarray,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> nn.Module:
    train_pairs_t = torch.tensor(train_pairs, dtype=torch.long, device=device)
    train_y_t = torch.tensor(train_labels, dtype=torch.float32, device=device)
    val_pairs_t = torch.tensor(val_pairs, dtype=torch.long, device=device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_ap = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(train_pairs_t)
        loss = loss_fn(logits, train_y_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_pairs_t).detach().cpu().numpy()
            val_scores = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -50, 50)))
            val_metrics = evaluate_scores(val_scores, val_labels, val_groups, top_ks=(10,))
            val_score = val_metrics.get("NDCG@10", float("nan"))
            if not np.isfinite(val_score):
                val_score = average_precision_score(val_labels, val_scores)

        if val_score > best_ap + 1e-6:
            best_ap = val_score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model


def predict_torch_model(model, pairs: np.ndarray, device: str, batch_size: int = 200000) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            p = torch.tensor(pairs[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(p).detach().cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
    return np.concatenate(scores, axis=0)


def fit_predict_pair_mlp(
    G_train: nx.Graph,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    val_pairs: np.ndarray,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    test_pairs: np.ndarray,
    args,
) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    device = args.device
    X_train = pair_feature_matrix(G_train, train_pairs)
    X_val = pair_feature_matrix(G_train, val_pairs)
    X_test = pair_feature_matrix(G_train, test_pairs)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = PairMLP(X_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)

    train_x_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_labels, dtype=torch.float32, device=device)
    val_x_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_ap, bad = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(train_x_t)
        loss = loss_fn(logits, train_y_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t).detach().cpu().numpy()
            val_scores = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -50, 50)))
            val_metrics = evaluate_scores(val_scores, val_labels, val_groups, top_ks=(10,))
            val_score = val_metrics.get("NDCG@10", float("nan"))
            if not np.isfinite(val_score):
                val_score = average_precision_score(val_labels, val_scores)
        if val_score > best_ap + 1e-6:
            best_ap = val_score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_test), 200000):
            x = torch.tensor(X_test[i : i + 200000], dtype=torch.float32, device=device)
            logits = model(x).detach().cpu().numpy()
            out.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
    return np.concatenate(out)


# ----------------------------- DeepWalk embedding -----------------------------

def deepwalk_scores(G_train: nx.Graph, pairs: np.ndarray, dim: int, seed: int, num_walks: int, walk_length: int, window: int, epochs: int):
    try:
        from gensim.models import Word2Vec
    except Exception as e:
        raise ImportError("gensim is required for DeepWalk/NodeEmbedding. Install it or remove DeepWalk from --methods.") from e

    rng = np.random.default_rng(seed)
    nodes = list(G_train.nodes())
    nbrs = {u: list(G_train.neighbors(u)) for u in nodes}
    walks = []
    for _ in range(num_walks):
        order = nodes[:]
        rng.shuffle(order)
        for start in order:
            walk = [start]
            cur = start
            for _step in range(walk_length - 1):
                if not nbrs[cur]:
                    break
                cur = int(rng.choice(nbrs[cur]))
                walk.append(cur)
            walks.append([str(x) for x in walk])

    w2v = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=0,
        sg=1,
        workers=1,
        epochs=epochs,
        seed=seed,
    )
    Z = np.zeros((G_train.number_of_nodes(), dim), dtype=np.float32)
    for u in nodes:
        Z[u] = w2v.wv[str(u)]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return np.sum(Z[pairs[:, 0]] * Z[pairs[:, 1]], axis=1)


# -------------------------------- Main runner ---------------------------------

def normalize_methods(methods: Sequence[str]) -> List[str]:
    if len(methods) == 1 and methods[0].lower() == "all":
        return [
            "CN", "AA", "RA", "Jaccard", "PA", "HDI", "HPI",
            "MultiHop", "Katz", "DeepWalk", "MF", "MLP", "GCN", "GAT", "LightGCN",
        ]
    aliases = {
        "COMMON_FRIENDS": "CN",
        "共同好友": "CN",
        "ADAMIC_ADAR": "AA",
        "ADAMIC-ADAR": "AA",
        "NODE_EMBEDDING": "DeepWalk",
        "NODEEMBEDDING": "DeepWalk",
        "MATRIX_FACTORIZATION": "MF",
        "MATRIXFACTORIZATION": "MF",
        "矩阵分解": "MF",
        "ATTENTION": "GAT",
        "GNN": "GCN",
    }
    out = []
    for m in methods:
        key = m.upper()
        out.append(aliases.get(key, m))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="facebook", choices=sorted(DATASETS),
                        help="Dataset to use: facebook, ca-grqc, or email-eu-core.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Default: outputs_<dataset>_lp")
    parser.add_argument("--keep-all-components", action="store_true",
                        help="Use all connected components instead of only the largest connected component.")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--neg-per-pos", type=int, default=50)
    parser.add_argument("--train-neg-ratio", type=int, default=1)
    parser.add_argument("--global-k", type=int, default=10, help="Deprecated. Use --topk instead.")
    parser.add_argument("--topk", nargs="+", type=int, default=[5, 10], help="User-level Top-K values, default: 5 10.")

    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--deepwalk-num-walks", type=int, default=10)
    parser.add_argument("--deepwalk-walk-length", type=int, default=40)
    parser.add_argument("--deepwalk-window", type=int, default=10)
    parser.add_argument("--deepwalk-epochs", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f"outputs_{args.dataset.replace('-', '_')}_lp"

    methods = normalize_methods(args.methods)
    os.makedirs(args.out_dir, exist_ok=True)

    G = download_and_load_graph(args.data_dir, args.dataset, keep_all_components=args.keep_all_components)
    n = G.number_of_nodes()
    full_edge_set = edge_set_from_graph(G)
    dataset_name = DATASETS[args.dataset]["display"]
    comp_note = "all components" if args.keep_all_components else "largest connected component"
    print(f"Loaded {dataset_name} ({comp_note}): nodes={n}, edges={G.number_of_edges()}")
    print(f"Methods: {methods}")
    print(f"Device: {args.device}")

    all_rows = []

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        rng = np.random.default_rng(seed)

        train_edges, val_edges, test_edges = connected_preserving_split(G, args.val_ratio, args.test_ratio, seed)
        G_train = make_graph_from_edges(n, train_edges)
        print(f"split: train={len(train_edges)}, val={len(val_edges)}, test={len(test_edges)}")

        val_pairs, val_labels, val_groups = build_user_eval_pairs(val_edges, n, full_edge_set, args.neg_per_pos, rng)
        test_pairs, test_labels, test_groups = build_user_eval_pairs(test_edges, n, full_edge_set, args.neg_per_pos, rng)

        train_neg_edges = sample_non_edges(n, full_edge_set, len(train_edges) * args.train_neg_ratio, rng)
        train_pairs = np.asarray(list(train_edges) + train_neg_edges, dtype=np.int64)
        train_labels = np.asarray([1] * len(train_edges) + [0] * len(train_neg_edges), dtype=np.int64)
        perm = rng.permutation(len(train_pairs))
        train_pairs, train_labels = train_pairs[perm], train_labels[perm]

        precomputed_path_scores = None

        for method in methods:
            t0 = time.time()
            print(f"Running {method} ...", flush=True)
            try:
                if method in {"CN", "AA", "RA", "Jaccard", "PA", "HDI", "HPI"}:
                    scores = score_heuristic(method, G_train, test_pairs)

                elif method in {"MultiHop", "Katz"}:
                    if precomputed_path_scores is None:
                        precomputed_path_scores = score_multihop_and_katz(n, train_edges, test_pairs)
                    scores = precomputed_path_scores["MultiHop_A2_A3"] if method == "MultiHop" else precomputed_path_scores["Katz_trunc3"]

                elif method == "DeepWalk":
                    scores = deepwalk_scores(
                        G_train, test_pairs, args.dim, seed,
                        args.deepwalk_num_walks, args.deepwalk_walk_length,
                        args.deepwalk_window, args.deepwalk_epochs,
                    )

                elif method == "MLP":
                    if torch is None:
                        raise ImportError("torch is required for MLP.")
                    scores = fit_predict_pair_mlp(G_train, train_pairs, train_labels, val_pairs, val_labels, val_groups, test_pairs, args)

                elif method in {"MF", "GCN", "GAT", "LightGCN"}:
                    if torch is None:
                        raise ImportError("torch is required for neural methods.")
                    device = args.device
                    if method == "MF":
                        model = MFModel(n, dim=args.dim)
                    elif method == "LightGCN":
                        adj_norm = torch_adj_norm(n, train_edges, device, add_self_loops=False)
                        model = LightGCNModel(n, adj_norm, dim=args.dim, layers=args.layers)
                    elif method == "GCN":
                        adj_norm = torch_adj_norm(n, train_edges, device, add_self_loops=True)
                        model = GCNModel(n, adj_norm, dim=args.dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
                    elif method == "GAT":
                        edge_index = torch_edge_index_with_self_loops(n, train_edges, device)
                        model = SparseGATModel(n, edge_index, dim=args.dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
                    else:
                        raise ValueError(method)

                    model = train_torch_model(
                        model, train_pairs, train_labels, val_pairs, val_labels, val_groups,
                        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                        patience=args.patience, device=device,
                    )
                    scores = predict_torch_model(model, test_pairs, device)

                else:
                    raise ValueError(f"Unknown method: {method}")

                metrics = evaluate_scores(scores, test_labels, test_groups, top_ks=args.topk)
                elapsed = time.time() - t0
                row = {"seed": seed, "method": method, "time_sec": elapsed}
                row.update(metrics)
                all_rows.append(row)
                focus_cols = ["method", "AUC", "AP", "MRR", "Hit@5", "Recall@5", "NDCG@5", "Hit@10", "Recall@10", "NDCG@10"]
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k in focus_cols})

            except Exception as e:
                elapsed = time.time() - t0
                row = {"seed": seed, "method": method, "time_sec": elapsed, "error": repr(e)}
                all_rows.append(row)
                print(f"ERROR in {method}: {repr(e)}")

        raw_df = pd.DataFrame(all_rows)
        raw_df.to_csv(os.path.join(args.out_dir, "results_raw.csv"), index=False)

    raw_df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.out_dir, "results_raw.csv")
    raw_df.to_csv(raw_path, index=False)

    metric_cols = [c for c in raw_df.columns if c not in {"seed", "method", "error"} and pd.api.types.is_numeric_dtype(raw_df[c])]
    summary = raw_df.groupby("method")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    # Sort by user-level NDCG@10 first; it is closer to the Top-K friend recommendation goal.
    sort_cols = [c for c in ["NDCG@10_mean", "Recall@10_mean", "Hit@10_mean", "MRR_mean", "AP_mean", "AUC_mean"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=False)

    summary_path = os.path.join(args.out_dir, "results_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print("\nSummary:")
    show_cols = ["method"] + [c for c in ["AUC_mean", "AP_mean", "MRR_mean", "Hit@5_mean", "Recall@5_mean", "NDCG@5_mean", "Hit@10_mean", "Recall@10_mean", "NDCG@10_mean", "time_sec_mean"] if c in summary.columns]
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
