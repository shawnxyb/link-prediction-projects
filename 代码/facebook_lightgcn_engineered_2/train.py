import argparse

from src.config import TrainConfig
from src.trainer import run_train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LightGCN on SNAP Facebook link prediction.")

    parser.add_argument("--data_path", type=str, default="facebook_combined.txt")
    parser.add_argument("--output_dir", type=str, default="outputs_lightgcn")
    parser.add_argument("--split_path", type=str, default=None, help="Optional existing .npz split file for fair comparison.")

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    parser.add_argument("--loss_type", type=str, default="bpr", choices=["bce", "bpr"])
    parser.add_argument("--train_neg_type", type=str, default="mixed", choices=["random", "hard", "mixed"])
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--eval_neg_type", type=str, default="random", choices=["random", "hard"])
    parser.add_argument("--eval_init", action="store_true", help="Evaluate the randomly initialized model before training.")

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    run_train(cfg)
