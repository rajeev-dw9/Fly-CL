""" Aggregate results/<name>_seed<seed>.json (and results/analysis/*.json) into a markdown table.

    python experiments/summarize.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

METHODS = ["fixed", "hebbian_frozen", "hebbian_frozen_strong", "hebbian_continual", "hebbian_continual_slow",
           "adaptive_ridge", "adaptive_ridge_inverse", "hebbian_frozen_adaptive_ridge"]

runs = defaultdict(dict)
for path in glob.glob("results/*_seed*.json"):
    name, seed = os.path.basename(path)[:-5].rsplit("_seed", 1)
    runs[name][int(seed)] = json.load(open(path))


def fmt(values, digits=2):
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


print("| Method | Seeds | Avg. acc (%) | Last acc (%) | BWT | Train time / task (s) | First-task time (s) |")
print("|---|---|---|---|---|---|---|")
for name in METHODS:
    if name not in runs:
        continue
    r = [runs[name][seed] for seed in sorted(runs[name])]
    print(f"| {name} | {len(r)} | {fmt([x['avg_acc'] for x in r])} | {fmt([x['last_acc'] for x in r])} | "
          f"{fmt([x['bwt'] for x in r])} | {fmt([np.mean(x['algorithm_time']) for x in r])} | "
          f"{fmt([x['algorithm_time'][0] for x in r])} |")

print()
print("Paired difference to `fixed` (same seed), mean [min, max] over seeds:")
print()
print("| Method | d Avg. acc | d Last acc | d BWT |")
print("|---|---|---|---|")
for name in METHODS[1:]:
    if name not in runs:
        continue
    cells = []
    for key in ["avg_acc", "last_acc", "bwt"]:
        d = [runs[name][s][key] - runs["fixed"][s][key] for s in sorted(runs[name])]
        cells.append(f"{np.mean(d):+.2f} [{min(d):+.2f}, {max(d):+.2f}]")
    print(f"| {name} | " + " | ".join(cells) + " |")

analyses = [json.load(open(p)) for p in sorted(glob.glob("results/analysis/analysis_seed*.json"))]
if analyses:
    print()
    print(f"Analysis over {len(analyses)} seed(s):")
    print()
    for section in analyses[0]:
        print(f"**{section}**")
        print()
        for key, value in analyses[0][section].items():
            if isinstance(value, dict):
                for sub in value:
                    print(f"- {key} / {sub}: {fmt([a[section][key][sub] for a in analyses], 4)}")
            else:
                values = [a[section][key] for a in analyses]
                print(f"- {key}: {np.mean(values):.4g} ± {np.std(values):.2g}")
        print()
