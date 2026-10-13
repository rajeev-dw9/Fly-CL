#! /bin/bash
# Usage: experiments/run.sh <name> <seed> [extra main.py arguments]
# Runs the CIFAR-100 configuration of scripts/test_cifar.sh from the repository root, with cached
# backbone features, and writes results/<name>_seed<seed>.json and the matching log.
# Set CACHE_SUFFIX=_val to run on the validation cache made by experiments/make_val_cache.py.

name=$1; seed=$2; shift 2
out=results${CACHE_SUFFIX}
mkdir -p $out/logs
python -W ignore main.py --dataset CIFAR-100 --num_classes 100 --num_tasks 10 --model_name vit_base_patch16_224 \
    --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed $seed --batch_size 128 \
    --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 --root ./data \
    --feature_cache cache/cifar100_vit_seed${seed}${CACHE_SUFFIX} \
    --save_json $out/${name}_seed${seed}.json "$@" > $out/logs/${name}_seed${seed}.log 2>&1
python - <<PY
import json; r = json.load(open("$out/${name}_seed${seed}.json"))
print(f"${name} seed${seed}: avg {r['avg_acc']:.2f}  last {r['last_acc']:.2f}  bwt {r['bwt']:.2f}  time {sum(r['algorithm_time'])/len(r['algorithm_time']):.2f}s")
PY
