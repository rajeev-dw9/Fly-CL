""" Analysis for both tasks, computed from cached features and the states saved by `--save_state`.

Task 1: class-prototype correlation matrices before / after the sparse projection
        (fixed random vs. Hebbian-learned) and their mean absolute off-diagonal correlation.
Task 2: numerical conditioning of the ridge system (G + Lambda) and the relation between the
        per-dimension ridge, the activity of a dimension and how correlated it is with the others.

    python experiments/analyze.py --seed 1993
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1993)
parser.add_argument('--num_tasks', type=int, default=10)
parser.add_argument('--coding_level', type=float, default=0.3)
parser.add_argument('--cache', default='cache/cifar100_vit_seed{seed}')
parser.add_argument('--state_dir', default='results/state')
parser.add_argument('--out_dir', default='results/analysis')
parser.add_argument('--fixed', default='fixed', help='run name of the original Fly-CL')
parser.add_argument('--hebbian', default='hebbian_frozen,hebbian_frozen_strong,hebbian_continual_slow,hebbian_continual',
                    help='run names of the learned projections (comma separated)')
parser.add_argument('--adaptive', default='adaptive_ridge,adaptive_ridge_inverse',
                    help='run names of the generalized ridge (comma separated, the first one is plotted)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out_dir, exist_ok=True)
summary = {}


def load_state(name):
    return torch.load(f"{args.state_dir}/{name}_seed{args.seed}.pt")


def expand(projection_matrix, embeddings):
    """ Same sparse expansion + Top-k winner-take-all as main.py. """
    out = projection_matrix @ embeddings.T
    values, indices = out.topk(int(out.size(0) * args.coding_level), dim=0, largest=True)
    return torch.zeros_like(out).scatter_(0, indices, values).T


def prototype_correlation(features, labels, class_order):
    prototypes = torch.stack([features[labels == c].mean(dim=0) for c in class_order])
    return torch.corrcoef(prototypes).cpu().numpy()


def mean_abs_offdiag(corr):
    return float(np.abs(corr[~np.eye(len(corr), dtype=bool)]).mean())


# ----------------------------------------------------------------------------------------------
# Task 1: prototype correlation before / after the sparse projection
# ----------------------------------------------------------------------------------------------
fixed_state = load_state(args.fixed)
projections = {"fixed": fixed_state["initial_projection_matrix"].to(device)}
summary["projection_change"] = {}
for name in args.hebbian.split(","):
    hebbian_state = load_state(name)
    learned, initial = hebbian_state["projection_matrix"], hebbian_state["initial_projection_matrix"]
    # the binary connectivity mask must be untouched by learning
    assert ((learned != 0) == (initial != 0)).all() and (initial == fixed_state["initial_projection_matrix"]).all()
    projections[name] = learned.to(device)
    summary["projection_change"][name] = {
        "mean_row_cosine_to_initial": float(torch.cosine_similarity(learned, initial, dim=1).mean()),
        "mean_nonzeros_per_row": float((learned != 0).sum(dim=1).float().mean())}
    del hebbian_state

cache = args.cache.format(seed=args.seed)
embeddings, labels = [], []
class_order = []  # classes in the order in which the tasks introduce them
for task in range(args.num_tasks):
    e, l = torch.load(f"{cache}/train_task{task}.pt")
    embeddings.append(e)
    labels.append(l)
    class_order += sorted(l.unique().tolist())
embeddings, labels = torch.cat(embeddings).to(device), torch.cat(labels).to(device)

correlations = {"pre-trained features": prototype_correlation(embeddings, labels, class_order)}
for name, projection_matrix in projections.items():
    prototypes = []  # class by class to keep the GPU memory small
    for c in class_order:
        prototypes.append(expand(projection_matrix, embeddings[labels == c]).mean(dim=0))
    correlations[f"{name} projection + Top-k"] = torch.corrcoef(torch.stack(prototypes)).cpu().numpy()

fig, axes = plt.subplots(1, len(correlations), figsize=(4.6 * len(correlations), 4.4))
for ax, (title, corr) in zip(axes, correlations.items()):
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"{title}\nmean |off-diagonal| = {mean_abs_offdiag(corr):.3f}", fontsize=9)
    ax.set_xlabel("class (task order)")
    ax.set_ylabel("class (task order)")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Pearson correlation between class prototypes")
fig.savefig(f"{args.out_dir}/prototype_correlation_seed{args.seed}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
summary["prototype_mean_abs_offdiag_corr"] = {t: mean_abs_offdiag(c) for t, c in correlations.items()}
# classes of the first task (seen by hebbian_frozen) vs. classes of the later tasks (never seen by it)
n = len(class_order) // args.num_tasks
summary["prototype_mean_abs_offdiag_corr_first_task_classes"] = {t: mean_abs_offdiag(c[:n, :n]) for t, c in correlations.items()}
summary["prototype_mean_abs_offdiag_corr_later_task_classes"] = {t: mean_abs_offdiag(c[n:, n:]) for t, c in correlations.items()}
del embeddings, projections
torch.cuda.empty_cache()

# ----------------------------------------------------------------------------------------------
# Task 2: conditioning and which dimensions get the stronger ridge
# ----------------------------------------------------------------------------------------------
def condition_number(G, ridge_diag):
    """ Condition number of the ridge system A = G + Lambda, and of A after symmetric diagonal
    (Jacobi) scaling, which is the quantity that governs the accuracy of a Cholesky solve. """
    A = (G + torch.diag(ridge_diag)).to(device, torch.float64)
    eigenvalues = torch.linalg.eigvalsh(A)
    d = torch.diagonal(A).rsqrt()
    scaled_eigenvalues = torch.linalg.eigvalsh(A.mul_(d[:, None]).mul_(d[None, :]))
    del A
    torch.cuda.empty_cache()
    return {"raw": float(eigenvalues[-1] / eigenvalues[0]),
            "jacobi_scaled": float(scaled_eigenvalues[-1] / scaled_eigenvalues[0])}


summary["condition_number"] = {"scalar ridge": condition_number(fixed_state["G"], fixed_state["ridge_diag"])}
for name in args.adaptive.split(",")[::-1]:
    adaptive_state = load_state(name)
    summary["condition_number"][name] = condition_number(adaptive_state["G"], adaptive_state["ridge_diag"])

if True:  # per-dimension analysis of the first adaptive run
    G = adaptive_state["G"].to(device, torch.float64)
    activity = torch.diagonal(G)
    scale = activity.clamp(min=1e-12).sqrt()
    cosine = G / scale[:, None] / scale[None, :]  # correlation between expanded dimensions
    cosine.fill_diagonal_(0)
    correlation_load = (cosine.abs().sum(dim=1) / (len(G) - 1)).cpu().numpy()
    activity = activity.cpu().numpy()
    ridge_diag = adaptive_state["ridge_diag"].double().numpy()

    summary["activity"] = {"min": float(activity.min()), "median": float(np.median(activity)),
                           "max": float(activity.max())}
    summary["ridge_diag"] = {"scalar": float(fixed_state["ridge_diag"][0]), "adaptive_min": float(ridge_diag.min()),
                             "adaptive_median": float(np.median(ridge_diag)), "adaptive_max": float(ridge_diag.max())}
    summary["spearman"] = {"ridge_vs_activity": float(spearmanr(ridge_diag, activity)[0]),
                           "ridge_vs_correlation_load": float(spearmanr(ridge_diag, correlation_load)[0]),
                           "activity_vs_correlation_load": float(spearmanr(activity, correlation_load)[0])}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].scatter(activity, ridge_diag, s=3, alpha=0.4)
    axes[0].axhline(float(fixed_state["ridge_diag"][0]), color="k", ls="--", lw=1, label="scalar ridge")
    axes[0].set_xlabel("activity of dimension j: $G_{jj}$")
    axes[0].set_ylabel("ridge coefficient $\\lambda_j$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].legend()
    axes[1].scatter(correlation_load, ridge_diag, s=3, alpha=0.4)
    axes[1].axhline(float(fixed_state["ridge_diag"][0]), color="k", ls="--", lw=1)
    axes[1].set_xlabel("correlation load of dimension j: mean$_{i \\neq j}$ |corr(j, i)|")
    axes[1].set_ylabel("ridge coefficient $\\lambda_j$")
    axes[1].set_yscale("log")
    axes[2].scatter(activity, correlation_load, s=3, alpha=0.4)
    axes[2].set_xlabel("activity of dimension j: $G_{jj}$")
    axes[2].set_ylabel("correlation load of dimension j")
    axes[2].set_xscale("log")
    fig.suptitle("Variance-adaptive ridge: which expanded dimensions are regularized more?")
    fig.savefig(f"{args.out_dir}/adaptive_ridge_seed{args.seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

with open(f"{args.out_dir}/analysis_seed{args.seed}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
