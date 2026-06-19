from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.config import TrainConfig
from src.data_utils import build_edge_set, make_split, read_edges, sample_negative_edges, save_split
from src.graph_utils import build_norm_adj
from src.metrics import evaluate_auc_ap, evaluate_ranking
from src.model import LightGCN
from src.utils import count_parameters, ensure_dir, get_device, save_text, set_seed


def _make_train_epoch_data(train_edges: np.ndarray, train_neg_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.vstack([train_edges, train_neg_edges])
    labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_neg_edges))])

    perm = np.random.permutation(len(pairs))
    return pairs[perm], labels[perm]


def _train_one_epoch(
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


def run_train(cfg: TrainConfig) -> dict[str, float]:
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

    split = make_split(
        edges=edges,
        num_nodes=num_nodes,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    print("Split info")
    print(f"Train edges: {len(split.train_edges)}")
    print(f"Val edges:   {len(split.val_edges)}")
    print(f"Test edges:  {len(split.test_edges)}")
    print("=" * 60)

    split_path = os.path.join(cfg.output_dir, f"facebook_split_seed{cfg.seed}.npz")
    save_split(split, split_path)
    print(f"Saved split to: {split_path}")

    norm_adj = build_norm_adj(num_nodes, split.train_edges, device)

    model = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=cfg.embedding_dim,
        num_layers=cfg.num_layers,
        norm_adj=norm_adj,
    ).to(device)

    params = count_parameters(model)

    print("=" * 60)
    print("Model info")
    print(f"Embedding dim: {cfg.embedding_dim}")
    print(f"LightGCN layers: {cfg.num_layers}")
    print(f"Trainable parameters: {params}")
    print("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_ap = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(cfg.output_dir, f"best_lightgcn_seed{cfg.seed}.pt")

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_neg_edges = sample_negative_edges(
            num_nodes=num_nodes,
            full_edge_set=full_edge_set,
            num_samples=len(split.train_edges),
            seed=cfg.seed + epoch,
        )
        train_pairs, train_labels = _make_train_epoch_data(split.train_edges, train_neg_edges)

        train_loss = _train_one_epoch(
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
    for key in ["Test AUC", "Test AP", "Precision@10", "Recall@10", "NDCG@10", "Precision@20", "Recall@20", "NDCG@20"]:
        print(f"{key}: {results[key]:.4f}")

    result_path = os.path.join(cfg.output_dir, f"lightgcn_result_seed{cfg.seed}.txt")
    result_lines = [
        "LightGCN Results",
        f"Seed: {cfg.seed}",
        f"Nodes: {num_nodes}",
        f"Edges: {len(edges)}",
        f"Train edges: {len(split.train_edges)}",
        f"Val edges: {len(split.val_edges)}",
        f"Test edges: {len(split.test_edges)}",
        f"Embedding dim: {cfg.embedding_dim}",
        f"Layers: {cfg.num_layers}",
        f"Parameters: {params}",
        f"Best epoch: {best_epoch}",
        f"Best Val AP: {best_val_ap:.4f}",
        f"Training time: {train_time:.2f} seconds",
        f"Test AUC: {test_auc:.4f}",
        f"Test AP: {test_ap:.4f}",
    ]
    for key in ["Precision@10", "Recall@10", "NDCG@10", "Precision@20", "Recall@20", "NDCG@20"]:
        result_lines.append(f"{key}: {results[key]:.4f}")

    save_text(result_path, "\n".join(result_lines) + "\n")
    print(f"Saved result to: {result_path}")
    return results
