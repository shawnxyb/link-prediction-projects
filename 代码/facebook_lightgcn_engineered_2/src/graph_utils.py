from collections import defaultdict
from typing import DefaultDict, Set

import numpy as np
import torch


def build_norm_adj(num_nodes: int, train_edges: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Build symmetric normalized adjacency matrix for LightGCN.

    A_norm[u, v] = 1 / sqrt(deg(u) * deg(v))
    """
    row = []
    col = []

    for u, v in train_edges:
        row.extend([int(u), int(v)])
        col.extend([int(v), int(u)])

    row_tensor = torch.tensor(row, dtype=torch.long)
    col_tensor = torch.tensor(col, dtype=torch.long)

    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, row_tensor, torch.ones_like(row_tensor, dtype=torch.float32))

    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[row_tensor] * deg_inv_sqrt[col_tensor]
    indices = torch.stack([row_tensor, col_tensor], dim=0)

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
    ).coalesce().to(device)


def build_neighbors(num_nodes: int, edges: np.ndarray) -> DefaultDict[int, Set[int]]:
    neighbors: DefaultDict[int, Set[int]] = defaultdict(set)
    for u, v in edges:
        neighbors[int(u)].add(int(v))
        neighbors[int(v)].add(int(u))
    return neighbors
