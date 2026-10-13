| Method | Seeds | Avg. acc (%) | Last acc (%) | BWT | Train time / task (s) | First-task time (s) |
|---|---|---|---|---|---|---|
| fixed | 3 | 92.34 ± 0.63 | 88.72 ± 0.07 | -4.55 ± 0.17 | 9.75 ± 0.03 | 9.80 ± 0.11 |
| hebbian_frozen | 3 | 92.04 ± 0.69 | 88.46 ± 0.02 | -4.63 ± 0.30 | 9.72 ± 0.01 | 9.93 ± 0.03 |
| hebbian_frozen_strong | 3 | 91.86 ± 0.62 | 88.16 ± 0.09 | -4.59 ± 0.40 | 9.70 ± 0.02 | 10.19 ± 0.04 |
| hebbian_continual | 3 | 51.54 ± 0.76 | 27.86 ± 0.60 | -77.14 ± 0.71 | 9.76 ± 0.02 | 9.92 ± 0.03 |
| hebbian_continual_slow | 3 | 83.80 ± 1.11 | 74.53 ± 0.51 | -24.28 ± 0.39 | 9.78 ± 0.03 | 9.96 ± 0.03 |
| adaptive_ridge | 3 | 92.25 ± 0.59 | 88.71 ± 0.07 | -4.54 ± 0.24 | 9.69 ± 0.02 | 9.73 ± 0.02 |
| adaptive_ridge_inverse | 3 | 92.35 ± 0.63 | 88.78 ± 0.08 | -4.46 ± 0.25 | 9.69 ± 0.01 | 9.80 ± 0.08 |
| hebbian_frozen_adaptive_ridge | 3 | 92.04 ± 0.64 | 88.47 ± 0.04 | -4.62 ± 0.21 | 9.67 ± 0.01 | 9.91 ± 0.04 |

Paired difference to `fixed` (same seed), mean [min, max] over seeds:

| Method | d Avg. acc | d Last acc | d BWT |
|---|---|---|---|
| hebbian_frozen | -0.30 [-0.43, -0.21] | -0.27 [-0.34, -0.13] | -0.07 [-0.26, +0.13] |
| hebbian_frozen_strong | -0.47 [-0.55, -0.35] | -0.56 [-0.63, -0.49] | -0.04 [-0.34, +0.21] |
| hebbian_continual | -40.80 [-42.08, -39.47] | -60.86 [-61.73, -60.31] | -72.59 [-73.29, -71.78] |
| hebbian_continual_slow | -8.54 [-9.11, -7.92] | -14.20 [-14.96, -13.78] | -19.73 [-20.32, -19.28] |
| adaptive_ridge | -0.09 [-0.17, -0.01] | -0.01 [-0.07, +0.03] | +0.01 [-0.08, +0.09] |
| adaptive_ridge_inverse | +0.02 [+0.01, +0.02] | +0.06 [-0.06, +0.12] | +0.09 [+0.02, +0.21] |
| hebbian_frozen_adaptive_ridge | -0.29 [-0.38, -0.18] | -0.26 [-0.36, -0.10] | -0.07 [-0.12, +0.00] |

Analysis over 3 seed(s):

**projection_change**

- hebbian_frozen / mean_row_cosine_to_initial: 0.9048 ± 0.0010
- hebbian_frozen / mean_nonzeros_per_row: 299.9999 ± 0.0001
- hebbian_frozen_strong / mean_row_cosine_to_initial: 0.4456 ± 0.0183
- hebbian_frozen_strong / mean_nonzeros_per_row: 299.9999 ± 0.0001
- hebbian_continual_slow / mean_row_cosine_to_initial: 0.9600 ± 0.0009
- hebbian_continual_slow / mean_nonzeros_per_row: 299.9999 ± 0.0001
- hebbian_continual / mean_row_cosine_to_initial: 0.5386 ± 0.0272
- hebbian_continual / mean_nonzeros_per_row: 299.9999 ± 0.0001

**prototype_mean_abs_offdiag_corr**

- pre-trained features: 0.4346 ± 2.8e-08
- fixed projection + Top-k: 0.3953 ± 0.0024
- hebbian_frozen projection + Top-k: 0.3824 ± 0.018
- hebbian_frozen_strong projection + Top-k: 0.4328 ± 0.015
- hebbian_continual_slow projection + Top-k: 0.3766 ± 0.0059
- hebbian_continual projection + Top-k: 0.3986 ± 0.03

**prototype_mean_abs_offdiag_corr_first_task_classes**

- pre-trained features: 0.4603 ± 0.021
- fixed projection + Top-k: 0.4195 ± 0.018
- hebbian_frozen projection + Top-k: 0.1489 ± 0.012
- hebbian_frozen_strong projection + Top-k: 0.1593 ± 0.012
- hebbian_continual_slow projection + Top-k: 0.4034 ± 0.033
- hebbian_continual projection + Top-k: 0.4367 ± 0.045

**prototype_mean_abs_offdiag_corr_later_task_classes**

- pre-trained features: 0.4336 ± 0.00052
- fixed projection + Top-k: 0.3945 ± 0.0026
- hebbian_frozen projection + Top-k: 0.4135 ± 0.021
- hebbian_frozen_strong projection + Top-k: 0.4693 ± 0.016
- hebbian_continual_slow projection + Top-k: 0.3767 ± 0.0061
- hebbian_continual projection + Top-k: 0.3981 ± 0.031

**condition_number**

- scalar ridge / raw: 52949.8212 ± 658.9900
- scalar ridge / jacobi_scaled: 178781.0961 ± 10704.3238
- adaptive_ridge_inverse / raw: 98153.4500 ± 3273.8683
- adaptive_ridge_inverse / jacobi_scaled: 447661.5996 ± 27977.8989
- adaptive_ridge / raw: 1054237.5859 ± 335489.9656
- adaptive_ridge / jacobi_scaled: 35862.3822 ± 143.8269

**activity**

- min: 6.288e+04 ± 2.1e+04
- median: 1.117e+07 ± 7.3e+04
- max: 1.341e+08 ± 8.1e+06

**ridge_diag**

- scalar: 1e+06 ± 0
- adaptive_min: 1e+04 ± 0.0033
- adaptive_median: 7.511e+05 ± 4.5e+03
- adaptive_max: 9.024e+06 ± 5.4e+05

**spearman**

- ridge_vs_activity: 1 ± 8.5e-12
- ridge_vs_correlation_load: 0.989 ± 0.0004
- activity_vs_correlation_load: 0.989 ± 0.0004

