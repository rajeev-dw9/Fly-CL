# Report of Changes: Two Extensions of Fly-CL

Base code: official Fly-CL repository (https://github.com/gfyddha/Fly-CL), upstream commit `193b1b8`.
Project tasks: (1) Activity-Dependent Learnable Sparse Projection, (2) Variance-Adaptive Generalized Ridge.

## 1. Principle followed

- **Minimal changes to the original code.** Every new behaviour is behind a command-line flag
  whose default reproduces the original Fly-CL. New logic is in new files.
- **Every step is a git commit**, so the history shows what changed and when.

Size of the change to the original files (`git diff 193b1b8 HEAD --numstat`):

| File | Lines added | Lines removed | Kind |
|---|---|---|---|
| `main.py` | 75 | 5 | modified |
| `utils.py` | 16 | 2 | modified |
| `extensions.py` | 60 | 0 | new (both methods) |
| `.gitignore` | 6 | 1 | modified |
| `readme.md` | 9 | 0 | modified |

In `main.py` only four original statements were touched: the `utils` import, the two
`feature_extract(...)` calls (one extra argument) and the Cholesky line (kept verbatim, moved into
an `else` branch). `datasets/`, `models/` and `scripts/` are untouched.

## 2. Git history

| Commit | Time (2026-10-13) | What |
|---|---|---|
| `6ee6341` | 17:13 | Experiment infrastructure: feature cache, BWT / last-stage metrics, json output |
| `12e121c` | 18:04 | Task 1: activity-dependent learnable sparse projection (Hebbian) |
| `d5bf460` | 18:04 | Task 2: variance-adaptive generalized ridge |
| `fb04d49` | 19:00 | Experiment, validation-sweep and analysis scripts |
| `d8a03e9` | 19:01 | Results, analysis figures and report |

Useful commands: `git log --stat`, `git show <commit>`, `git diff 193b1b8 HEAD -- main.py`.

## 3. Change 1 — Experiment infrastructure (commit `6ee6341`)

No change to the learning algorithm.

| Where | Change | Why |
|---|---|---|
| `utils.py: feature_extract` | optional `cache_path` argument: load features from disk if present, otherwise extract and save | The backbone is frozen, so features are identical for all methods. Caching makes all methods run on exactly the same inputs and cuts a run from about 7 min to about 1.5 min. |
| `utils.py: save_results` | new 4-line helper writing a json file | Results are aggregated by script instead of copied from logs. |
| `main.py` flags | `--feature_cache`, `--save_json`, `--save_state` | Switch the above on; all default to `None` (off). |
| `main.py` end of file | prints last-stage accuracy, BWT, training time excluding feature extraction | The project asks for these metrics; the original printed only per-stage and accumulated accuracy and total time. |
| `main.py` timer | `torch.cuda.synchronize()` before the training timer stops | GPU calls are asynchronous; without it the measured time can miss pending work. |
| `main.py` bookkeeping | `ridges` list, `initial_projection_matrix`, `ridge_diag` | Needed only for the analysis (GCV choice, mask check, conditioning). |

Metric definitions added:

- Last-stage accuracy = accuracy on all classes after the last task.
- BWT = mean over tasks 1..T−1 of (final accuracy on the task − accuracy right after learning it).

## 4. Change 2 — Task 1: Activity-Dependent Learnable Sparse Projection (commit `12e121c`)

**New function** `extensions.py: hebbian_update(projection_matrix, features, lr, epochs, hebb_k, delta)`.

How each project instruction is met:

| Instruction | Implementation |
|---|---|
| Fix a binary connectivity mask at initialization | `mask = (W != 0)` taken from the original random projection; all updates are multiplied by the mask. |
| Replace fixed values by weights learned with a local Hebbian / anti-Hebbian rule | BioHash-style rule on the active connections: `dW_mu = g_mu * (mask_mu * x − <W_mu, x> W_mu)`, with `g = 1` for the `hebb_k` most active units of a sample, `g = −delta` for the next `hebb_k` units, `0` otherwise. |
| Keep the projection sparsity and the same Top-k | The returned matrix has the same non-zero pattern; the Top-k code in `main.py` is unchanged. |
| Stable variant: adapt on the first task, then freeze | `--projection hebbian_frozen` |
| Optional: continual updates | `--projection hebbian_continual` |

Details of the rule: features are centred and L2-normalised for learning only; the update of each
unit is divided by the number of samples that drove it (activity-dependent step); each row is
re-normalised to the norm it had at initialization, so the expanded features keep the original
scale and the ridge grid needs no change.

**Change in `main.py`** (3 lines in the task loop, plus 5 flags):

```python
if args.projection == 'hebbian_continual' or (args.projection == 'hebbian_frozen' and task == 0):
    projection_matrix = hebbian_update(projection_matrix, train_embeddings, args.hebb_lr,
                                       args.hebb_epochs, args.hebb_k, args.hebb_delta)
```

New flags: `--projection {fixed,hebbian_frozen,hebbian_continual}` (default `fixed`),
`--hebb_lr` (0.1), `--hebb_epochs` (1), `--hebb_k` (0.01), `--hebb_delta` (0.0).
The Hebbian step is inside the training timer, so its cost is included in the reported time.

## 5. Change 3 — Task 2: Variance-Adaptive Generalized Ridge (commit `d5bf460`)

**New function** `extensions.py: adaptive_ridge_diag(G, ridge, alpha, clamp=100)`.

| Instruction | Implementation |
|---|---|
| Use the diagonal of the accumulated Gram matrix as activity / variance | `activity = torch.diagonal(G)` |
| Replace `λI` by `Λ = diag(λ_1..λ_m)` | returns the vector `λ_j`; `main.py` adds `torch.diag(ridge_diag)` to `G` |
| Dimension-specific coefficients from streaming statistics, GCV coefficient as global scale | `λ_j = λ_GCV * s_j`, `s_j = (G_jj / mean_i G_ii)^alpha`, clamped to `[0.01, 100]`, normalised to mean 1 |
| Keep streaming updates of `G`, `Q`; efficient positive-definite solver | `G`, `Q`, GCV untouched; solved with `torch.linalg.cholesky` + `cholesky_solve` |

`alpha = 0` gives the original scalar ridge; `alpha = 1` equals ridge on variance-standardised
dimensions; `alpha > 0` regularizes highly active dimensions more.

**Change in `main.py`** (the original line is kept in the `else` branch):

```python
if args.ridge_type == 'adaptive':
    ridge_diag = adaptive_ridge_diag(G, ridge.item(), args.ridge_alpha)
    L = torch.linalg.cholesky(G + torch.diag(ridge_diag))
else:
    ridge_diag = ridge * torch.ones(G.size(dim=0)).to(device)
    L = torch.linalg.cholesky(G + ridge * torch.eye(G.size(dim=0)).to(device)) # 40% faster
```

New flags: `--ridge_type {scalar,adaptive}` (default `scalar`), `--ridge_alpha` (1.0).

## 6. Change 4 — Experiment and analysis scripts (commit `fb04d49`)

All new, in `experiments/`; none of them changes the method.

| File | Purpose |
|---|---|
| `run.sh` | One CIFAR-100 run with the configuration of `scripts/test_cifar.sh`, cached features, json + log. |
| `make_val_cache.py` | Holds out the last 10% of each task's training features as a validation split, so hyperparameters are not chosen on the test set. |
| `sweep_val.sh` | The validation sweep for the Hebbian and ridge hyperparameters. |
| `run_all.sh` | The final comparison: 8 configurations × 3 seeds. |
| `analyze.py` | Prototype correlation matrices before / after projection, mask check, condition numbers, per-dimension ridge analysis, figures. |
| `summarize.py` | Aggregates all json files into `results/summary.md`. |

## 7. Change 5 — Results and documentation (commit `d8a03e9`)

- `results/` (test-set runs, summary, figures), `results_val/` (validation sweep).
- `RESULTS.md` (report), `CHANGELOG.md` (running log), a short section in `readme.md`.
- `.gitignore`: `.venv/`, `cache/`, `data/`, `*.pt` (large or machine-specific files).

## 8. How to verify that the original behaviour is preserved

- With default flags no new branch is executed: `--projection fixed`, `--ridge_type scalar`,
  no cache, no json.
- The baseline run gave 93.21 average accuracy before the extension code was added and 93.20
  after (seed 1993). A difference of 0.01 corresponds to about one test image at one stage; its
  cause was not investigated (most likely run-to-run GPU floating-point variation).
- `experiments/analyze.py` asserts that the learned projection has exactly the non-zero pattern of
  the initial one and that all methods start from the same initial matrix.

## 9. Environment notes (not code changes)

- Runs used torch 2.14.1, torchvision 0.29.1, numpy 2.5 instead of the versions in the readme;
  the original code runs on them unmodified. `timm==0.9.16` needs `huggingface_hub==0.23.5`.
- CIFAR-100 was copied into `data/` with the official md5 checksums verified.
- One row of the original random projection has 298 instead of 300 non-zeros for these seeds,
  as generated by the unmodified initialization; it is the same for all methods.

## 10. How to run

```bash
# original Fly-CL
python main.py --dataset CIFAR-100 ... 
# Task 1 (stable variant)
python main.py --dataset CIFAR-100 ... --projection hebbian_frozen
# Task 2
python main.py --dataset CIFAR-100 ... --ridge_type adaptive --ridge_alpha 1.0
# everything used in the report
experiments/run_all.sh
```
