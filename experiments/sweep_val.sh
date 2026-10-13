#! /bin/bash
# Hyperparameter selection on the validation cache (seed 1993). Run from the repository root.
export CACHE_SUFFIX=_val
seed=1993

# Task 1: lr, epochs, hebb_k, delta
for cfg in "0.5 1 0.01 0" "0.5 5 0.01 0" "0.5 5 0.01 0.4" "0.5 5 0.001 0" "0.5 5 0.05 0" "0.5 5 0.3 0" "1.0 10 0.01 0.4"; do
    set -- $cfg
    experiments/run.sh hebb_lr$1_ep$2_k$3_d$4 $seed --projection hebbian_frozen --hebb_lr $1 --hebb_epochs $2 --hebb_k $3 --hebb_delta $4
done

# Task 2: alpha
for alpha in 0.25 0.5 1.0 -0.5; do
    experiments/run.sh ridge_alpha$alpha $seed --ridge_type adaptive --ridge_alpha $alpha
done
