from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_path: str = "facebook_combined.txt"
    output_dir: str = "outputs_lightgcn"

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

    seed: int = 2024
    cpu: bool = False
