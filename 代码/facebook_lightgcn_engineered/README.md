# Facebook LightGCN Link Prediction

This project trains LightGCN on SNAP Facebook Social Circles for user-user link prediction.

## 1. Project structure

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

## 2. Install dependencies

```bash
python -m pip install numpy scikit-learn tqdm
```

If PyTorch is already installed, do not reinstall it.

## 3. Prepare data

Put `facebook_combined.txt` in the project root:

```text
facebook_lightgcn_engineered/
├── facebook_combined.txt
└── train.py
```

## 4. Train

```bash
python train.py --data_path facebook_combined.txt --embedding_dim 64 --num_layers 2 --lr 0.001 --weight_decay 1e-5 --epochs 200 --patience 20 --seed 2024
```

If you want to force CPU:

```bash
python train.py --cpu
```

## 5. Outputs

The program writes outputs to `outputs_lightgcn/`:

```text
outputs_lightgcn/
├── facebook_split_seed2024.npz
├── best_lightgcn_seed2024.pt
└── lightgcn_result_seed2024.txt
```

`facebook_split_seed2024.npz` should also be used by the MLP teammate to ensure fair comparison.

## 6. Ablation examples

```bash
python train.py --embedding_dim 16 --num_layers 2 --seed 2024
python train.py --embedding_dim 32 --num_layers 2 --seed 2024
python train.py --embedding_dim 64 --num_layers 2 --seed 2024
python train.py --embedding_dim 64 --num_layers 1 --seed 2024
python train.py --embedding_dim 64 --num_layers 3 --seed 2024
```
