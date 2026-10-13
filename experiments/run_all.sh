#! /bin/bash
# Final comparison on the CIFAR-100 test set. Run from the repository root.
# Hyperparameters were chosen with experiments/sweep_val.sh (validation split, seed 1993).
mkdir -p results/state
for seed in 1993 1994 1995; do
    s="--save_state results/state"
    # original Fly-CL (also fills the feature cache of this seed)
    experiments/run.sh fixed $seed $s/fixed_seed$seed.pt

    # Task 1
    experiments/run.sh hebbian_frozen $seed --projection hebbian_frozen $s/hebbian_frozen_seed$seed.pt
    experiments/run.sh hebbian_frozen_strong $seed --projection hebbian_frozen --hebb_lr 0.5 --hebb_epochs 5 \
        $s/hebbian_frozen_strong_seed$seed.pt
    experiments/run.sh hebbian_continual $seed --projection hebbian_continual $s/hebbian_continual_seed$seed.pt
    experiments/run.sh hebbian_continual_slow $seed --projection hebbian_continual --hebb_lr 0.01 \
        $s/hebbian_continual_slow_seed$seed.pt

    # Task 2
    experiments/run.sh adaptive_ridge $seed --ridge_type adaptive --ridge_alpha 1.0 $s/adaptive_ridge_seed$seed.pt
    experiments/run.sh adaptive_ridge_inverse $seed --ridge_type adaptive --ridge_alpha -1.0 \
        $s/adaptive_ridge_inverse_seed$seed.pt

    # Task 1 + Task 2
    experiments/run.sh hebbian_frozen_adaptive_ridge $seed --projection hebbian_frozen --ridge_type adaptive \
        --ridge_alpha 1.0 $s/hebbian_frozen_adaptive_ridge_seed$seed.pt
done
