from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.graph_utils import build_neighbors


@torch.no_grad()
def evaluate_auc_ap(model, pos_edges: np.ndarray, neg_edges: np.ndarray, device: torch.device) -> tuple[float, float]:
    model.eval()

    labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
    pairs = np.vstack([pos_edges, neg_edges])
    pairs_tensor = torch.tensor(pairs, dtype=torch.long, device=device)

    emb = model.get_embeddings()
    logits = model.score_edges(pairs_tensor, emb)
    scores = torch.sigmoid(logits).detach().cpu().numpy()

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return float(auc), float(ap)


@torch.no_grad()
def evaluate_ranking(
    model,
    train_edges: np.ndarray,
    test_edges: np.ndarray,
    num_nodes: int,
    device: torch.device,
    k: int = 10,
) -> dict[str, float]:
    """
    Compute Precision@K, Recall@K, NDCG@K.

    Candidate set for each user:
    all nodes - itself - training neighbors.
    """
    model.eval()

    train_neighbors = build_neighbors(num_nodes, train_edges)
    test_neighbors = build_neighbors(num_nodes, test_edges)

    emb = model.get_embeddings()
    score_matrix = torch.matmul(emb, emb.t())

    precisions = []
    recalls = []
    ndcgs = []

    for u in range(num_nodes):
        if u not in test_neighbors or len(test_neighbors[u]) == 0:
            continue

        gt_items = test_neighbors[u]
        scores = score_matrix[u].clone()
        scores[u] = -1e9

        for v in train_neighbors[u]:
            scores[v] = -1e9

        topk = torch.topk(scores, k=k).indices.detach().cpu().numpy().tolist()
        hits = [1 if v in gt_items else 0 for v in topk]
        num_hits = sum(hits)

        precision = num_hits / k
        recall = num_hits / len(gt_items)

        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
        ideal_hits = min(len(gt_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
