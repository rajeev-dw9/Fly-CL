# Change log

All changes made on top of the official Fly-CL code (upstream commit `193b1b8`) for the two
project tasks. Newest entries are at the bottom. Every entry corresponds to one git commit, so
`git log --stat` / `git show <commit>` shows the exact diff.

Design rule: **minimal changes to the original code**. All new behaviour is behind command-line
flags whose defaults reproduce the original Fly-CL exactly; new logic lives in new files.

## 1. Experiment infrastructure

Files: `main.py`, `utils.py`, `.gitignore`

- `utils.py: feature_extract(..., cache_path=None)` — optional on-disk cache of the frozen
  backbone features. The backbone is never trained, so features are identical across methods;
  caching them makes every comparison run on exactly the same inputs and takes minutes, not hours.
- `utils.py: save_results()` — writes metrics to json.
- `main.py`: new flags `--feature_cache`, `--save_json`, `--save_state`.
- `main.py`: additionally prints **last-stage accuracy**, **BWT** and the **training time
  excluding feature extraction** (the original only printed per-stage/accumulated accuracy and
  total time). BWT = mean over old tasks of (final accuracy − accuracy right after learning it).
- `main.py`: `torch.cuda.synchronize()` before the training timer stops, so GPU work is fully
  counted in the reported training time.
- `main.py`: records the GCV-selected ridge of every task and keeps a handle on the initial
  projection matrix (needed for the analysis only).
- `.gitignore`: ignore `.venv/`, `cache/`, `data/`, `*.pt`.

No change to the learning algorithm: with default flags the run is the original Fly-CL.

## 2. Task 1 — Activity-Dependent Learnable Sparse Projection

Files: `extensions.py` (new), `main.py`

- `extensions.py: hebbian_update()` — new. Takes the original sparse projection matrix and the
  pre-trained features of a task and returns a projection with **the same non-zero pattern**
  (binary connectivity mask fixed at initialization) whose active weights were adapted with a
  local BioHash-style Hebbian / anti-Hebbian rule:
  `dW_mu = g_mu * (mask_mu * x - <W_mu, x> W_mu)`, with `g = 1` for the `hebb_k` most active
  units of a sample, `g = -delta` for the next `hebb_k` units, `0` otherwise. Features are
  centred and L2-normalised for learning only; every row keeps its initial norm so the expanded
  features stay on the original scale (the ridge grid therefore needs no change).
- `main.py`: new flags `--projection {fixed,hebbian_frozen,hebbian_continual}`, `--hebb_lr`,
  `--hebb_epochs`, `--hebb_k`, `--hebb_delta`.
- `main.py`: 3 lines in the task loop, right after feature extraction, call `hebbian_update`
  on the first task only (`hebbian_frozen`, the stable variant) or on every task
  (`hebbian_continual`). The Hebbian step is inside the training timer.
- Unchanged: the sparse expansion, the Top-k winner-take-all, the streaming `G`/`Q` updates, GCV
  and the Cholesky solve. `--projection fixed` (default) is the original method.

## 3. Task 2 — Variance-Adaptive Generalized Ridge

Files: `extensions.py`, `main.py`

- `extensions.py: adaptive_ridge_diag()` — new. Builds the diagonal regularizer
  `Lambda = diag(lambda_1..lambda_m)` from the streaming statistics:
  `lambda_j = ridge * s_j`, `s_j = (G_jj / mean_i G_ii) ** alpha`, clamped to `[1/100, 100]` and
  normalized to mean 1. `G_jj` (diagonal of the accumulated Gram matrix) is the activity of
  expanded dimension `j`; `ridge` is the original GCV-selected coefficient and stays the global
  scale. `alpha > 0` regularizes highly active dimensions more, `alpha = 0` is the original
  scalar ridge, `alpha = 1` equals ridge on variance-standardized dimensions.
- `main.py`: new flags `--ridge_type {scalar,adaptive}`, `--ridge_alpha`.
- `main.py`: the solve becomes `cholesky(G + diag(ridge_diag))` when `--ridge_type adaptive`;
  the original line is kept untouched in the `scalar` branch.
- Unchanged: streaming updates of `G` and `Q`, GCV selection, Cholesky positive-definite solver.

## 4. Experiment and analysis scripts

Files: `experiments/` (new)

- `experiments/run.sh` — one CIFAR-100 run with the configuration of `scripts/test_cifar.sh`,
  cached features, json + log output.
- `experiments/make_val_cache.py` — holds out the last 10% of each task's cached training
  features as a validation "test" split, so hyperparameters are not selected on the test set.
- `experiments/sweep_val.sh` — the validation sweep used to choose the Hebbian and ridge settings.
- `experiments/run_all.sh` — the final comparison (8 configurations x 3 seeds).
- `experiments/analyze.py` — prototype correlation matrices before/after the projection, change
  of the projection, condition numbers of the ridge system, per-dimension ridge analysis.
- `experiments/summarize.py` — aggregates all json files into `results/summary.md`.

No change to `main.py`, `utils.py` or `extensions.py`.

## 5. Results and report

Files: `RESULTS.md` (new), `results/`, `results_val/`, `experiments/sweep_val.sh`, `readme.md`

- `results/` — json + log of every test-set run (8 configurations x 3 seeds), `summary.md`,
  and `analysis/` (figures and json per seed). `results_val/` — the validation sweep.
  The saved projection matrices / Gram matrices (`results/state/*.pt`, 11 GB) are not in git.
- `RESULTS.md` — the comparison tables, figures, findings and limitations.
- `experiments/sweep_val.sh` — added the anti-Hebbian (negative lr) and wider `alpha` settings
  that were run after the first sweep, so the script reproduces all of `results_val/`
  (except the two `uncentered_*` runs, which used a temporary edit of `extensions.py`).
- `readme.md` — one section pointing to the extensions.

## Environment notes (not code changes)

- Runs were made with torch 2.14.1 / torchvision 0.29.1 / numpy 2.5 instead of the versions in the
  readme; the original code runs unmodified on them. `timm==0.9.16` needs `huggingface_hub==0.23.5`
  (newer hub versions removed functions it imports).
- CIFAR-100 was copied into `data/` (official md5 checksums verified) because the download
  server was too slow.
- One row of the original random projection has 298 instead of 300 non-zeros for these seeds
  (as generated by the unmodified initialization); the mask is identical for all methods.
