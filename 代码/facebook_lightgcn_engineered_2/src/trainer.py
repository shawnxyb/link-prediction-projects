from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TrainConfig
from src.data_utils import (
    build_edge_set,
    build_hard_candidates_by_anchor,
    build_two_hop_negative_pool,
    load_split,
    make_split,
    read_edges,
    sample_mixed_negative_edges,
    sample_negative_edges,
    sample_negative_nodes_for_anchors,
    save_split,
)
from src.graph_utils import build_norm_adj
from src.metrics import evaluate_auc_ap, evaluate_ranking
from src.model import LightGCN
from src.utils import count_parameters, ensure_dir, get_device, save_text, set_seed


def _run_name(cfg: TrainConfig) -> str:
    return (
        f"loss-{cfg.loss_type}_neg-{cfg.train_neg_type}"
        f"_dim{cfg.embedding_dim}_layer{cfg.num_layers}_seed{cfg.seed}"
    )


def _make_bce_epoch_data(train_edges: np.ndarray, train_neg_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.vstack([train_edges, train_neg_edges])
    labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_neg_edges))])

    perm = np.random.permutation(len(pairs))
    return pairs[perm], labels[perm]


def _make_bpr_epoch_data(
    train_edges: np.ndarray,
    num_nodes: int,
    full_edge_set: set[tuple[int, int]],
    hard_candidates: list[np.ndarray] | None,
    cfg: TrainConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create BPR triplets (anchor, positive_node, negative_node).

    Both directions of each undirected train edge are used:
    (u, v, neg_for_u) and (v, u, neg_for_v).
    """
    anchors = np.concatenate([train_edges[:, 0], train_edges[:, 1]]).astype(np.int64)
    pos_nodes = np.concatenate([train_edges[:, 1], train_edges[:, 0]]).astype(np.int64)

    neg_nodes = sample_negative_nodes_for_anchors(
        anchors=anchors,
        num_nodes=num_nodes,
        full_edge_set=full_edge_set,
        hard_candidates=hard_candidates,
        neg_type=cfg.train_neg_type,
        hard_ratio=cfg.hard_neg_ratio,
        seed=seed,
    )

    perm = np.random.permutation(len(anchors))
    return anchors[perm], pos_nodes[perm], neg_nodes[perm]


def _train_one_epoch_bce(
    model: LightGCN,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    num_samples = len(train_pairs)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_pairs = torch.tensor(train_pairs[start:end], dtype=torch.long, device=device)
        batch_labels = torch.tensor(train_labels[start:end], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        emb = model.get_embeddings()
        logits = model.score_edges(batch_pairs, emb)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        batch_n = end - start
        total_loss += loss.item() * batch_n
        total_count += batch_n

    return total_loss / max(total_count, 1)


def _train_one_epoch_bpr(
    model: LightGCN,
    optimizer: torch.optim.Optimizer,
    anchors: np.ndarray,
    pos_nodes: np.ndarray,
    neg_nodes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    num_samples = len(anchors)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        a = torch.tensor(anchors[start:end], dtype=torch.long, device=device)
        p = torch.tensor(pos_nodes[start:end], dtype=torch.long, device=device)
        n = torch.tensor(neg_nodes[start:end], dtype=torch.long, device=device)

        optimizer.zero_grad()
        emb = model.get_embeddings()

        anchor_emb = emb[a]
        pos_emb = emb[p]
        neg_emb = emb[n]

        pos_scores = torch.sum(anchor_emb * pos_emb, dim=1)
        neg_scores = torch.sum(anchor_emb * neg_emb, dim=1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        loss.backward()
        optimizer.step()

        batch_n = end - start
        total_loss += loss.item() * batch_n
        total_count += batch_n

    return total_loss / max(total_count, 1)


def _sample_train_neg_edges(
    cfg: TrainConfig,
    num_nodes: int,
    full_edge_set: set[tuple[int, int]],
    hard_pool: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if cfg.train_neg_type == "hard":
        if len(hard_pool) == 0:
            return sample_negative_edges(num_nodes, full_edge_set, num_samples, seed=seed)
        from src.data_utils import sample_edges_from_pool
        return sample_edges_from_pool(hard_pool, num_samples, seed=seed)
    if cfg.train_neg_type == "mixed":
        return sample_mixed_negative_edges(
            num_nodes=num_nodes,
            full_edge_set=full_edge_set,
            hard_pool=hard_pool,
            num_samples=num_samples,
            hard_ratio=cfg.hard_neg_ratio,
            seed=seed,
        )
    return sample_negative_edges(num_nodes, full_edge_set, num_samples, seed=seed)


def run_train(cfg: TrainConfig) -> dict[str, float]:
    cfg.loss_type = cfg.loss_type.lower()
    cfg.train_neg_type = cfg.train_neg_type.lower()
    cfg.eval_neg_type = cfg.eval_neg_type.lower()

    if cfg.loss_type not in {"bce", "bpr"}:
        raise ValueError("loss_type must be 'bce' or 'bpr'.")
    if cfg.train_neg_type not in {"random", "hard", "mixed"}:
        raise ValueError("train_neg_type must be 'random', 'hard', or 'mixed'.")
    if cfg.eval_neg_type not in {"random", "hard"}:
        raise ValueError("eval_neg_type must be 'random' or 'hard'.")

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device = get_device(cfg.cpu)
    print(f"Using device: {device}")

    edges, num_nodes, _ = read_edges(cfg.data_path)
    full_edge_set = build_edge_set(edges)

    print("=" * 60)
    print("Dataset info")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {len(edges)}")
    print("=" * 60)

    if cfg.split_path:
        split = load_split(cfg.split_path)
        if split.num_nodes != num_nodes:
            raise ValueError("Loaded split num_nodes does not match data file.")
        print(f"Loaded split from: {cfg.split_path}")
    else:
        split = make_split(
            edges=edges,
            num_nodes=num_nodes,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.seed,
            eval_neg_type=cfg.eval_neg_type,
        )

    print("Split info")
    print(f"Train edges: {len(split.train_edges)}")
    print(f"Val edges:   {len(split.val_edges)}")
    print(f"Test edges:  {len(split.test_edges)}")
    print("=" * 60)

    split_path = os.path.join(cfg.output_dir, f"facebook_split_seed{cfg.seed}_evalneg-{cfg.eval_neg_type}.npz")
    if not cfg.split_path:
        save_split(split, split_path)
        print(f"Saved split to: {split_path}")

    print("Building hard negative pool...")
    hard_pool = build_two_hop_negative_pool(num_nodes, split.train_edges, full_edge_set)
    print(f"Hard negative pool size: {len(hard_pool)}")
    hard_candidates = build_hard_candidates_by_anchor(num_nodes, hard_pool) if cfg.train_neg_type in {"hard", "mixed"} else None

    norm_adj = build_norm_adj(num_nodes, split.train_edges, device)

    model = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=cfg.embedding_dim,
        num_layers=cfg.num_layers,
        norm_adj=norm_adj,
    ).to(device)

    params = count_parameters(model)
    run_name = _run_name(cfg)

    print("=" * 60)
    print("Model info")
    print(f"Embedding dim: {cfg.embedding_dim}")
    print(f"LightGCN layers: {cfg.num_layers}")
    print(f"Loss type: {cfg.loss_type}")
    print(f"Train negative type: {cfg.train_neg_type}")
    print(f"Hard negative ratio: {cfg.hard_neg_ratio}")
    print(f"Eval negative type: {cfg.eval_neg_type}")
    print(f"Trainable parameters: {params}")
    print("=" * 60)

    if cfg.eval_init:
        init_auc, init_ap = evaluate_auc_ap(model, split.val_edges, split.val_neg_edges, device)
        print(f"Initial model | Val AUC {init_auc:.4f} | Val AP {init_ap:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_ap = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(cfg.output_dir, f"best_lightgcn_{run_name}.pt")

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if cfg.loss_type == "bpr":
            anchors, pos_nodes, neg_nodes = _make_bpr_epoch_data(
                train_edges=split.train_edges,
                num_nodes=num_nodes,
                full_edge_set=full_edge_set,
                hard_candidates=hard_candidates,
                cfg=cfg,
                seed=cfg.seed + epoch,
            )
            train_loss = _train_one_epoch_bpr(
                model=model,
                optimizer=optimizer,
                anchors=anchors,
                pos_nodes=pos_nodes,
                neg_nodes=neg_nodes,
                device=device,
                batch_size=cfg.batch_size,
            )
        else:
            train_neg_edges = _sample_train_neg_edges(
                cfg=cfg,
                num_nodes=num_nodes,
                full_edge_set=full_edge_set,
                hard_pool=hard_pool,
                num_samples=len(split.train_edges),
                seed=cfg.seed + epoch,
            )
            train_pairs, train_labels = _make_bce_epoch_data(split.train_edges, train_neg_edges)
            train_loss = _train_one_epoch_bce(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_pairs=train_pairs,
                train_labels=train_labels,
                device=device,
                batch_size=cfg.batch_size,
            )

        val_auc, val_ap = evaluate_auc_ap(model, split.val_edges, split.val_neg_edges, device)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {train_loss:.4f} | "
                f"Val AUC {val_auc:.4f} | "
                f"Val AP {val_ap:.4f}"
            )

        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val AP: {best_val_ap:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_auc, test_ap = evaluate_auc_ap(model, split.test_edges, split.test_neg_edges, device)
    ranking_10 = evaluate_ranking(model, split.train_edges, split.test_edges, num_nodes, device, k=10)
    ranking_20 = evaluate_ranking(model, split.train_edges, split.test_edges, num_nodes, device, k=20)

    results: dict[str, float] = {
        "Test AUC": test_auc,
        "Test AP": test_ap,
        **ranking_10,
        **ranking_20,
        "Best epoch": float(best_epoch),
        "Best Val AP": best_val_ap,
        "Training time": train_time,
        "Parameters": float(params),
    }

    print("Final Test Results")
    metric_order = [
        "Test AUC", "Test AP",
        "Precision@10", "Recall@10", "Hits@10", "NDCG@10",
        "Precision@20", "Recall@20", "Hits@20", "NDCG@20",
    ]
    for key in metric_order:
        print(f"{key}: {results[key]:.4f}")

    result_path = os.path.join(cfg.output_dir, f"lightgcn_result_{run_name}.txt")
    result_lines = [
        "LightGCN Results",
        f"Run name: {run_name}",
        f"Seed: {cfg.seed}",
        f"Nodes: {num_nodes}",
        f"Edges: {len(edges)}",
        f"Train edges: {len(split.train_edges)}",
        f"Val edges: {len(split.val_edges)}",
        f"Test edges: {len(split.test_edges)}",
        f"Embedding dim: {cfg.embedding_dim}",
        f"Layers: {cfg.num_layers}",
        f"Loss type: {cfg.loss_type}",
        f"Train negative type: {cfg.train_neg_type}",
        f"Hard negative ratio: {cfg.hard_neg_ratio}",
        f"Eval negative type: {cfg.eval_neg_type}",
        f"Hard negative pool size: {len(hard_pool)}",
        f"Parameters: {params}",
        f"Best epoch: {best_epoch}",
        f"Best Val AP: {best_val_ap:.4f}",
        f"Training time: {train_time:.2f} seconds",
        f"Test AUC: {test_auc:.4f}",
        f"Test AP: {test_ap:.4f}",
    ]
    for key in ["Precision@10", "Recall@10", "Hits@10", "NDCG@10", "Precision@20", "Recall@20", "Hits@20", "NDCG@20"]:
        result_lines.append(f"{key}: {results[key]:.4f}")

    save_text(result_path, "\n".join(result_lines) + "\n")
    print(f"Saved result to: {result_path}")
    return results
