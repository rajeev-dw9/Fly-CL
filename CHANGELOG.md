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
