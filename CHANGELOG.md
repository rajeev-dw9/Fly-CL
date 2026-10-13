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
