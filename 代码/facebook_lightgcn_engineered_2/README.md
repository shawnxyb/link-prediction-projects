# Facebook LightGCN Link Prediction v2

This version adds ranking-oriented training options for the SNAP Facebook link prediction task:

- `--loss_type bpr`: pairwise BPR loss, more aligned with Top-K recommendation than BCE.
- `--train_neg_type hard`: train with two-hop non-edges, i.e. users with common friends but no true link.
- `--train_neg_type mixed`: combine hard negatives and random negatives.
- `--eval_neg_type hard`: optionally evaluate AUC/AP with hard negatives.
- `Hits@K`: added to per-user Top-K metrics.
- `--split_path`: reuse an existing split file for fair comparison.

## Structure

```text
facebook_lightgcn_engineered/
├── train.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    ├── graph_utils.py
    ├── metrics.py
    ├── model.py
    ├── trainer.py
    └── utils.py
```

## Install dependencies

```bash
python -m pip install numpy scikit-learn tqdm
```

If PyTorch is already installed, do not reinstall it.

## Prepare data

Put `facebook_combined.txt` in the project root:

```text
facebook_lightgcn_engineered/
├── facebook_combined.txt
└── train.py
```

## Recommended run: BPR + mixed negatives

```bash
python train.py --data_path .\facebook_combined.txt --loss_type bpr --train_neg_type mixed --hard_neg_ratio 0.5 --embedding_dim 64 --num_layers 2 --lr 0.001 --weight_decay 1e-5 --epochs 200 --patience 20 --seed 2024
```

## Reproduce old setting: BCE + random negatives

```bash
python train.py --data_path .\facebook_combined.txt --loss_type bce --train_neg_type random --embedding_dim 64 --num_layers 2 --seed 2024
```

## Hard-negative AUC/AP evaluation

```bash
python train.py --data_path .\facebook_combined.txt --loss_type bpr --train_neg_type mixed --hard_neg_ratio 0.5 --eval_neg_type hard --seed 2024
```

## Reuse the same split

```bash
python train.py --data_path .\facebook_combined.txt --split_path .\outputs_lightgcn\facebook_split_seed2024_evalneg-random.npz --loss_type bpr --train_neg_type mixed --seed 2024
```

## Outputs

The program writes outputs to `outputs_lightgcn/`:

```text
outputs_lightgcn/
├── facebook_split_seed2024_evalneg-random.npz
├── best_lightgcn_loss-bpr_neg-mixed_dim64_layer2_seed2024.pt
└── lightgcn_result_loss-bpr_neg-mixed_dim64_layer2_seed2024.txt
```

## Suggested comparisons

```bash
python train.py --data_path .\facebook_combined.txt --loss_type bce --train_neg_type random --seed 2024
python train.py --data_path .\facebook_combined.txt --loss_type bpr --train_neg_type random --seed 2024
python train.py --data_path .\facebook_combined.txt --loss_type bpr --train_neg_type mixed --hard_neg_ratio 0.5 --seed 2024
python train.py --data_path .\facebook_combined.txt --loss_type bpr --train_neg_type hard --seed 2024
```
