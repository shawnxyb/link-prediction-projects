import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """LightGCN for homogeneous user-user link prediction."""

    def __init__(self, num_nodes: int, embedding_dim: int, num_layers: int, norm_adj: torch.Tensor):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.norm_adj = norm_adj

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def get_embeddings(self) -> torch.Tensor:
        emb = self.embedding.weight
        all_embs = [emb]

        for _ in range(self.num_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            all_embs.append(emb)

        return torch.stack(all_embs, dim=0).mean(dim=0)

    @staticmethod
    def score_edges(edge_pairs: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        u = edge_pairs[:, 0]
        v = edge_pairs[:, 1]
        return torch.sum(embeddings[u] * embeddings[v], dim=1)
