""" Build a validation feature cache from an existing feature cache.

The last `val_ratio` of every task's cached training features becomes the "test" split of the
new cache, the rest stays as "train". Running main.py with `--feature_cache <dst>` then reports
validation accuracy, so hyperparameters are never selected on the real test set.

    python experiments/make_val_cache.py cache/cifar100_vit_seed1993 cache/cifar100_vit_seed1993_val
"""
import os
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
val_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
os.makedirs(dst, exist_ok=True)
task = 0
while os.path.exists(f"{src}/train_task{task}.pt"):
    embeddings, labels = torch.load(f"{src}/train_task{task}.pt")  # already shuffled by the DataLoader
    n_val = int(len(labels) * val_ratio)
    torch.save((embeddings[:-n_val], labels[:-n_val]), f"{dst}/train_task{task}.pt")
    torch.save((embeddings[-n_val:], labels[-n_val:]), f"{dst}/test_task{task}.pt")
    task += 1
print(f"Wrote {task} tasks to {dst}")
