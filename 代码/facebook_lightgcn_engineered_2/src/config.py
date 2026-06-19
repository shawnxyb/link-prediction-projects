from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    data_path: str = "facebook_combined.txt"
    output_dir: str = "outputs_lightgcn"
    split_path: Optional[str] = None

    embedding_dim: int = 64
    num_layers: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-5

    epochs: int = 200
    patience: int = 20
    log_every: int = 5
    batch_size: int = 4096

    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # loss_type:
    #   bce: binary classification loss
    #   bpr: pairwise ranking loss, usually better aligned with Top-K recommendation
    loss_type: str = "bpr"

    # train_neg_type:
    #   random: random non-edges
    #   hard: two-hop non-edges, i.e. nodes with common neighbors but no true edge
    #   mixed: hard negatives + random negatives
    train_neg_type: str = "mixed"
    hard_neg_ratio: float = 0.5

    # eval_neg_type controls val/test negative samples for AUC/AP.
    # Top-K metrics always rank candidates per user and do not use these sampled negatives.
    eval_neg_type: str = "random"

    eval_init: bool = False

    seed: int = 2024
    cpu: bool = False
