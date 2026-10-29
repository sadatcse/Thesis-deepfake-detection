# Test 6: MSTF-AV multi-stream detector with robust, consistent and shortcut-adversarial training

_Generated 2026-10-23 06:14:49 | script `161026.py` sha256 `b093dd7e223dc94e…` | data = Test 5 split `6ae9fe14ca0dbd44…` (read-only, link PASS)_

## 1. Design

- **Architecture (MSTF-AV):** RGB residual CNN + SE and an SRM noise-residual stream joined by a cross-stream gate; temporal transformer over frame and frame-difference tokens; audio tokens fused by gated cross-modal attention; 2,280,919 parameters (Test 5 model: 584,978).
- **Variants:** 6A: MSTF-AV + Test 5 baseline recipe (unpaired 50/50 original/degraded, natural mixture, CE); 6B: 6A + paired batches + Grad-CAM consistency + supervised contrastive consistency; 6C: 6B + adaptive group-DRO degradation sampling (uniform floor) + EMA weights + label smoothing; 6D: 6C + class-conditional shortcut adversaries (resolution tercile; dataset in joint training).
- **Baselines:** Test 5 variants 5A and 5C on the same split, test videos and seeds (T5-5A, T5-5C).
- **Calibration:** temperature scaling on validation; thresholds chosen on calibrated probabilities.

## 2. Runs

| run_id | best_epoch | best_val_score | temperature | threshold | val_ece_raw | val_ece | train_time_s |
|---|---|---|---|---|---|---|---|
| train-BioDeepAV_seed42_S1 | 2 | 0.933 | 2.853 | 0.320 | 0.288 | 0.250 | 370.428 |
| train-BioDeepAV_seed42_6A | 5 | 0.968 | 3.075 | 0.140 | 0.182 | 0.188 | 1835.506 |
| train-BioDeepAV_seed42_6B | 4 | 0.953 | 4.820 | 0.240 | 0.297 | 0.260 | 1678.308 |
| train-BioDeepAV_seed42_6C | 10 | 0.954 | 2.714 | 0.220 | 0.264 | 0.261 | 1731.408 |
| train-BioDeepAV_seed42_6D | 7 | 0.952 | 3.314 | 0.270 | 0.312 | 0.261 | 1767.235 |
| train-FFpp_seed42_S1 | 6 | 0.691 | 0.779 | 0.540 | 0.338 | 0.331 | 769.004 |
| train-FFpp_seed42_6A | 10 | 0.710 | 0.883 | 0.720 | 0.045 | 0.047 | 2274.007 |
| train-FFpp_seed42_6B | 10 | 0.704 | 0.840 | 0.770 | 0.051 | 0.026 | 2194.864 |
| train-FFpp_seed42_6C | 10 | 0.693 | 0.741 | 0.720 | 0.061 | 0.034 | 2263.000 |
| train-FFpp_seed42_6D | 10 | 0.686 | 0.799 | 0.740 | 0.059 | 0.027 | 2268.505 |
| train-Joint_seed42_S1 | 7 | 0.752 | 2.396 | 0.200 | 0.298 | 0.292 | 1057.043 |
| train-Joint_seed42_6A | 6 | 0.753 | 5.326 | 0.220 | 0.202 | 0.258 | 3812.762 |
| train-Joint_seed42_6B | 8 | 0.729 | 2.853 | 0.160 | 0.145 | 0.200 | 3956.158 |
| train-Joint_seed42_6C | 10 | 0.741 | 2.783 | 0.220 | 0.198 | 0.211 | 3907.719 |
| train-Joint_seed42_6D | 1 | 0.730 | 2.115 | 0.240 | 0.198 | 0.191 | 3924.911 |
| train-BioDeepAV_seed123_S1 | 6 | 0.990 | 1.418 | 0.100 | 0.106 | 0.117 | 376.834 |
| train-BioDeepAV_seed123_6A | 5 | 0.989 | 2.647 | 0.090 | 0.156 | 0.182 | 1649.040 |
| train-BioDeepAV_seed123_6B | 4 | 0.991 | 2.582 | 0.110 | 0.199 | 0.237 | 1692.164 |
| train-BioDeepAV_seed123_6C | 7 | 0.975 | 1.821 | 0.150 | 0.192 | 0.200 | 1733.848 |
| train-BioDeepAV_seed123_6D | 7 | 0.990 | 1.867 | 0.150 | 0.190 | 0.226 | 1758.242 |
| train-FFpp_seed123_S1 | 8 | 0.710 | 0.975 | 0.510 | 0.337 | 0.336 | 792.464 |
| train-FFpp_seed123_6A | 6 | 0.718 | 0.741 | 0.810 | 0.068 | 0.049 | 2128.785 |
| train-FFpp_seed123_6B | 7 | 0.700 | 0.799 | 0.710 | 0.072 | 0.045 | 2318.077 |
| train-FFpp_seed123_6C | 7 | 0.695 | 0.638 | 0.650 | 0.116 | 0.062 | 2272.176 |
| train-FFpp_seed123_6D | 1 | 0.689 | 0.671 | 0.510 | 0.188 | 0.157 | 2309.021 |
| train-Joint_seed123_S1 | 6 | 0.736 | 1.914 | 0.210 | 0.302 | 0.307 | 1081.108 |
| train-Joint_seed123_6A | 5 | 0.727 | 2.519 | 0.240 | 0.151 | 0.192 | 3658.504 |
| train-Joint_seed123_6B | 8 | 0.710 | 3.075 | 0.330 | 0.162 | 0.196 | 3860.801 |
| train-Joint_seed123_6C | 10 | 0.732 | 1.418 | 0.170 | 0.124 | 0.151 | 4032.877 |
| train-Joint_seed123_6D | 9 | 0.699 | 1.383 | 0.340 | 0.122 | 0.141 | 4063.077 |
| train-BioDeepAV_seed2026_S1 | 5 | 0.939 | 3.398 | 0.220 | 0.267 | 0.243 | 338.517 |
| train-BioDeepAV_seed2026_6A | 5 | 0.961 | 4.585 | 0.180 | 0.261 | 0.244 | 1612.136 |
| train-BioDeepAV_seed2026_6B | 3 | 0.973 | 3.314 | 0.200 | 0.259 | 0.251 | 1678.465 |
| train-BioDeepAV_seed2026_6C | 6 | 0.930 | 2.456 | 0.220 | 0.246 | 0.212 | 1704.149 |
| train-BioDeepAV_seed2026_6D | 10 | 0.929 | 4.585 | 0.300 | 0.342 | 0.170 | 1743.111 |
| train-FFpp_seed2026_S1 | 8 | 0.694 | 0.951 | 0.460 | 0.341 | 0.340 | 855.514 |
| train-FFpp_seed2026_6A | 10 | 0.712 | 0.840 | 0.760 | 0.046 | 0.025 | 2149.636 |
| train-FFpp_seed2026_6B | 10 | 0.709 | 0.799 | 0.750 | 0.059 | 0.033 | 2218.951 |
| train-FFpp_seed2026_6C | 9 | 0.691 | 0.779 | 0.750 | 0.059 | 0.026 | 2281.279 |
| train-FFpp_seed2026_6D | 4 | 0.683 | 0.461 | 0.730 | 0.133 | 0.034 | 2291.856 |
| train-Joint_seed2026_S1 | 6 | 0.739 | 3.075 | 0.200 | 0.317 | 0.318 | 1061.355 |
| train-Joint_seed2026_6A | 6 | 0.751 | 2.853 | 0.230 | 0.143 | 0.202 | 3747.271 |
| train-Joint_seed2026_6B | 4 | 0.724 | 3.398 | 0.190 | 0.166 | 0.224 | 3934.256 |
| train-Joint_seed2026_6C | 8 | 0.691 | 2.115 | 0.260 | 0.170 | 0.179 | 4024.805 |
| train-Joint_seed2026_6D | 10 | 0.705 | 2.223 | 0.200 | 0.164 | 0.180 | 4064.450 |

## 3. Results (seed mean ± SD)

**AUC, original test videos** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.897 ± 0.119 (n=3) | 0.966 ± 0.032 (n=3) | 0.952 ± 0.038 (n=3) | 0.956 ± 0.014 (n=3) | 0.941 ± 0.018 (n=3) | 0.945 ± 0.022 (n=3) | 0.928 ± 0.053 (n=3) |
| FF++ | FF++ | 0.670 ± 0.036 (n=3) | 0.770 ± 0.014 (n=3) | 0.744 ± 0.037 (n=3) | 0.726 ± 0.018 (n=3) | 0.725 ± 0.003 (n=3) | 0.716 ± 0.015 (n=3) | 0.714 ± 0.016 (n=3) |
| Joint | BioDeepAV | 0.943 ± 0.024 (n=3) | - | - | 0.953 ± 0.006 (n=3) | 0.931 ± 0.051 (n=3) | 0.949 ± 0.011 (n=3) | 0.913 ± 0.042 (n=3) |
| Joint | FF++ | 0.466 ± 0.044 (n=3) | - | - | 0.472 ± 0.030 (n=3) | 0.437 ± 0.036 (n=3) | 0.424 ± 0.033 (n=3) | 0.441 ± 0.047 (n=3) |

**AUC, degraded test videos** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.735 ± 0.098 (n=3) | 0.939 ± 0.038 (n=3) | 0.922 ± 0.047 (n=3) | 0.895 ± 0.024 (n=3) | 0.873 ± 0.028 (n=3) | 0.878 ± 0.016 (n=3) | 0.875 ± 0.070 (n=3) |
| FF++ | FF++ | 0.575 ± 0.032 (n=3) | 0.705 ± 0.008 (n=3) | 0.694 ± 0.027 (n=3) | 0.650 ± 0.027 (n=3) | 0.648 ± 0.008 (n=3) | 0.633 ± 0.016 (n=3) | 0.624 ± 0.015 (n=3) |
| Joint | BioDeepAV | 0.802 ± 0.028 (n=3) | - | - | 0.905 ± 0.010 (n=3) | 0.887 ± 0.044 (n=3) | 0.902 ± 0.023 (n=3) | 0.859 ± 0.022 (n=3) |
| Joint | FF++ | 0.462 ± 0.033 (n=3) | - | - | 0.478 ± 0.022 (n=3) | 0.457 ± 0.034 (n=3) | 0.435 ± 0.029 (n=3) | 0.438 ± 0.032 (n=3) |

**Worst degradation-type AUC** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.544 ± 0.061 (n=3) | 0.858 ± 0.053 (n=3) | 0.840 ± 0.051 (n=3) | 0.815 ± 0.053 (n=3) | 0.797 ± 0.054 (n=3) | 0.791 ± 0.050 (n=3) | 0.791 ± 0.138 (n=3) |
| FF++ | FF++ | 0.315 ± 0.033 (n=3) | 0.570 ± 0.009 (n=3) | 0.559 ± 0.041 (n=3) | 0.519 ± 0.040 (n=3) | 0.500 ± 0.004 (n=3) | 0.435 ± 0.042 (n=3) | 0.401 ± 0.051 (n=3) |
| Joint | BioDeepAV | 0.585 ± 0.095 (n=3) | - | - | 0.850 ± 0.028 (n=3) | 0.835 ± 0.053 (n=3) | 0.832 ± 0.054 (n=3) | 0.762 ± 0.081 (n=3) |
| Joint | FF++ | 0.379 ± 0.045 (n=3) | - | - | 0.402 ± 0.048 (n=3) | 0.343 ± 0.062 (n=3) | 0.315 ± 0.057 (n=3) | 0.369 ± 0.030 (n=3) |

**Noise AUC (starvation check)** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.565 ± 0.028 (n=3) | 0.961 ± 0.026 (n=3) | 0.927 ± 0.067 (n=3) | 0.922 ± 0.003 (n=3) | 0.883 ± 0.064 (n=3) | 0.872 ± 0.047 (n=3) | 0.911 ± 0.016 (n=3) |
| FF++ | FF++ | 0.664 ± 0.017 (n=3) | 0.756 ± 0.016 (n=3) | 0.740 ± 0.078 (n=3) | 0.709 ± 0.012 (n=3) | 0.703 ± 0.030 (n=3) | 0.686 ± 0.023 (n=3) | 0.681 ± 0.014 (n=3) |
| Joint | BioDeepAV | 0.623 ± 0.151 (n=3) | - | - | 0.919 ± 0.029 (n=3) | 0.916 ± 0.042 (n=3) | 0.933 ± 0.005 (n=3) | 0.873 ± 0.015 (n=3) |
| Joint | FF++ | 0.464 ± 0.007 (n=3) | - | - | 0.486 ± 0.036 (n=3) | 0.485 ± 0.019 (n=3) | 0.460 ± 0.042 (n=3) | 0.451 ± 0.020 (n=3) |

**Prediction flip rate %** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 19.954 ± 2.069 (n=3) | 6.461 ± 0.720 (n=3) | 8.701 ± 1.475 (n=3) | 6.747 ± 1.265 (n=3) | 10.329 ± 2.553 (n=3) | 11.465 ± 0.560 (n=3) | 11.956 ± 6.096 (n=3) |
| FF++ | FF++ | 14.475 ± 1.589 (n=3) | 15.097 ± 2.554 (n=3) | 15.664 ± 1.516 (n=3) | 18.194 ± 0.418 (n=3) | 15.706 ± 2.534 (n=3) | 15.830 ± 1.413 (n=3) | 15.153 ± 3.354 (n=3) |
| Joint | BioDeepAV | 18.531 ± 1.902 (n=3) | - | - | 8.317 ± 2.716 (n=3) | 9.159 ± 0.296 (n=3) | 9.781 ± 2.197 (n=3) | 8.783 ± 4.785 (n=3) |
| Joint | FF++ | 1.258 ± 2.179 (n=3) | - | - | 0.235 ± 0.213 (n=3) | 0.235 ± 0.407 (n=3) | 0.152 ± 0.086 (n=3) | 0.539 ± 0.862 (n=3) |

**Real videos recognised** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.850 ± 0.074 (n=3) | 0.621 ± 0.364 (n=3) | 0.690 ± 0.173 (n=3) | 0.386 ± 0.097 (n=3) | 0.599 ± 0.134 (n=3) | 0.573 ± 0.191 (n=3) | 0.530 ± 0.206 (n=3) |
| FF++ | FF++ | 0.503 ± 0.059 (n=3) | 0.657 ± 0.069 (n=3) | 0.790 ± 0.062 (n=3) | 0.667 ± 0.110 (n=3) | 0.627 ± 0.093 (n=3) | 0.582 ± 0.043 (n=3) | 0.562 ± 0.097 (n=3) |
| Joint | BioDeepAV | 0.766 ± 0.091 (n=3) | - | - | 0.746 ± 0.194 (n=3) | 0.692 ± 0.166 (n=3) | 0.737 ± 0.174 (n=3) | 0.709 ± 0.145 (n=3) |
| Joint | FF++ | 0.000 ± 0.000 (n=3) | - | - | 0.002 ± 0.003 (n=3) | 0.007 ± 0.012 (n=3) | 0.003 ± 0.003 (n=3) | 0.015 ± 0.026 (n=3) |

**ECE (calibrated)** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.225 ± 0.118 (n=3) | 0.108 ± 0.068 (n=3) | 0.132 ± 0.056 (n=3) | 0.177 ± 0.065 (n=3) | 0.243 ± 0.058 (n=3) | 0.227 ± 0.033 (n=3) | 0.235 ± 0.058 (n=3) |
| FF++ | FF++ | 0.353 ± 0.011 (n=3) | 0.165 ± 0.015 (n=3) | 0.185 ± 0.025 (n=3) | 0.140 ± 0.005 (n=3) | 0.139 ± 0.002 (n=3) | 0.152 ± 0.025 (n=3) | 0.187 ± 0.071 (n=3) |
| Joint | BioDeepAV | 0.182 ± 0.075 (n=3) | - | - | 0.223 ± 0.024 (n=3) | 0.185 ± 0.040 (n=3) | 0.183 ± 0.024 (n=3) | 0.158 ± 0.070 (n=3) |
| Joint | FF++ | 0.406 ± 0.001 (n=3) | - | - | 0.346 ± 0.017 (n=3) | 0.342 ± 0.005 (n=3) | 0.317 ± 0.030 (n=3) | 0.333 ± 0.051 (n=3) |

**ECE (uncalibrated)** (in-domain / joint)

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.228 ± 0.161 (n=3) | 0.108 ± 0.068 (n=3) | 0.132 ± 0.056 (n=3) | 0.116 ± 0.070 (n=3) | 0.218 ± 0.083 (n=3) | 0.204 ± 0.025 (n=3) | 0.254 ± 0.111 (n=3) |
| FF++ | FF++ | 0.358 ± 0.011 (n=3) | 0.165 ± 0.015 (n=3) | 0.185 ± 0.025 (n=3) | 0.174 ± 0.012 (n=3) | 0.175 ± 0.006 (n=3) | 0.206 ± 0.034 (n=3) | 0.255 ± 0.060 (n=3) |
| Joint | BioDeepAV | 0.174 ± 0.085 (n=3) | - | - | 0.179 ± 0.021 (n=3) | 0.142 ± 0.062 (n=3) | 0.162 ± 0.030 (n=3) | 0.144 ± 0.087 (n=3) |
| Joint | FF++ | 0.413 ± 0.004 (n=3) | - | - | 0.228 ± 0.014 (n=3) | 0.225 ± 0.012 (n=3) | 0.243 ± 0.003 (n=3) | 0.286 ± 0.077 (n=3) |

**Cross-dataset AUC (original videos)**

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | FF++ | 0.381 ± 0.016 (n=3) | 0.542 ± 0.027 (n=3) | 0.490 ± 0.048 (n=3) | 0.418 ± 0.015 (n=3) | 0.392 ± 0.030 (n=3) | 0.411 ± 0.008 (n=3) | 0.415 ± 0.014 (n=3) |
| FF++ | BioDeepAV | 0.216 ± 0.072 (n=3) | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.284 ± 0.040 (n=3) | 0.246 ± 0.026 (n=3) | 0.239 ± 0.029 (n=3) | 0.219 ± 0.047 (n=3) |

**Resolution shortcut: Spearman ρ(p_fake, native face px) within real videos**

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.519 ± 0.097 (n=3) | 0.518 ± 0.084 (n=3) | 0.501 ± 0.052 (n=3) | 0.370 ± 0.160 (n=3) | 0.366 ± 0.087 (n=3) | 0.425 ± 0.076 (n=3) | 0.378 ± 0.134 (n=3) |
| BioDeepAV | FF++ | 0.453 ± 0.103 (n=3) | 0.395 ± 0.150 (n=3) | 0.277 ± 0.164 (n=3) | 0.293 ± 0.117 (n=3) | 0.241 ± 0.112 (n=3) | 0.232 ± 0.099 (n=3) | 0.218 ± 0.049 (n=3) |
| FF++ | BioDeepAV | -0.268 ± 0.104 (n=3) | nan (n=1) | nan (n=1) | -0.290 ± 0.034 (n=3) | -0.349 ± 0.057 (n=3) | -0.280 ± 0.069 (n=3) | -0.305 ± 0.041 (n=3) |
| FF++ | FF++ | -0.291 ± 0.094 (n=3) | -0.089 ± 0.044 (n=3) | -0.012 ± 0.116 (n=3) | -0.253 ± 0.069 (n=3) | -0.255 ± 0.054 (n=3) | -0.271 ± 0.057 (n=3) | -0.231 ± 0.039 (n=3) |
| Joint | BioDeepAV | 0.438 ± 0.007 (n=3) | - | - | 0.325 ± 0.052 (n=3) | 0.308 ± 0.061 (n=3) | 0.279 ± 0.090 (n=3) | 0.302 ± 0.108 (n=3) |
| Joint | FF++ | 0.019 ± 0.145 (n=3) | - | - | -0.392 ± 0.188 (n=3) | -0.261 ± 0.151 (n=3) | -0.369 ± 0.074 (n=3) | -0.150 ± 0.152 (n=3) |

**Audio ablation (flip rate vs normal, %)**

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 9.3 ± 5.9 (n=3) | 5.5 ± 2.4 (n=3) | 6.8 ± 2.3 (n=3) | 7.5 ± 0.6 (n=3) | 9.4 ± 2.5 (n=3) | 13.6 ± 3.8 (n=3) | 8.8 ± 2.9 (n=3) |
| FF++ | BioDeepAV | - | 49.5 ± 10.8 (n=3) | 70.1 ± 20.4 (n=3) | - | - | - | - |
| Joint | BioDeepAV | 25.8 ± 3.3 (n=3) | - | - | 22.9 ± 8.1 (n=3) | 23.3 ± 4.7 (n=3) | 23.8 ± 7.6 (n=3) | 26.8 ± 7.7 (n=3) |

## 4. Grad-CAM stability

**Top-20% IoU, all types**

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.467 ± 0.041 (n=3) | 0.602 ± 0.021 (n=3) | 0.614 ± 0.013 (n=3) | 0.523 ± 0.043 (n=3) | 0.578 ± 0.036 (n=3) | 0.538 ± 0.065 (n=3) | 0.549 ± 0.124 (n=3) |
| FF++ | FF++ | 0.587 ± 0.029 (n=3) | 0.614 ± 0.026 (n=3) | 0.775 ± 0.020 (n=3) | 0.596 ± 0.021 (n=3) | 0.826 ± 0.021 (n=3) | 0.782 ± 0.023 (n=3) | 0.699 ± 0.112 (n=3) |
| Joint | BioDeepAV | 0.460 ± 0.035 (n=3) | - | - | 0.462 ± 0.017 (n=3) | 0.542 ± 0.040 (n=3) | 0.561 ± 0.056 (n=3) | 0.536 ± 0.072 (n=3) |
| Joint | FF++ | 0.446 ± 0.031 (n=3) | - | - | 0.509 ± 0.028 (n=3) | 0.657 ± 0.068 (n=3) | 0.655 ± 0.050 (n=3) | 0.583 ± 0.119 (n=3) |

**Top-20% IoU, hard types**

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.391 ± 0.061 (n=3) | 0.436 ± 0.014 (n=3) | 0.459 ± 0.017 (n=3) | 0.439 ± 0.035 (n=3) | 0.498 ± 0.018 (n=3) | 0.466 ± 0.038 (n=3) | 0.465 ± 0.120 (n=3) |
| FF++ | FF++ | 0.447 ± 0.063 (n=3) | 0.379 ± 0.025 (n=3) | 0.629 ± 0.019 (n=3) | 0.467 ± 0.026 (n=3) | 0.756 ± 0.024 (n=3) | 0.703 ± 0.029 (n=3) | 0.593 ± 0.153 (n=3) |
| Joint | BioDeepAV | 0.415 ± 0.025 (n=3) | - | - | 0.383 ± 0.006 (n=3) | 0.434 ± 0.040 (n=3) | 0.464 ± 0.058 (n=3) | 0.447 ± 0.042 (n=3) |
| Joint | FF++ | 0.351 ± 0.025 (n=3) | - | - | 0.373 ± 0.029 (n=3) | 0.515 ± 0.065 (n=3) | 0.524 ± 0.044 (n=3) | 0.428 ± 0.116 (n=3) |

**Explanation changed while prediction stable (%)**

| train_on | test_dataset | S1 | T5-5A | T5-5C | 6A | 6B | 6C | 6D |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 47.5 ± 8.0 (n=3) | 34.1 ± 1.2 (n=3) | 30.3 ± 1.2 (n=3) | 42.7 ± 9.1 (n=3) | 32.6 ± 7.1 (n=3) | 36.9 ± 8.5 (n=3) | 36.2 ± 19.1 (n=3) |
| FF++ | FF++ | 29.5 ± 3.8 (n=3) | 26.8 ± 4.3 (n=3) | 9.5 ± 2.1 (n=3) | 28.3 ± 2.2 (n=3) | 4.2 ± 0.8 (n=3) | 5.9 ± 2.3 (n=3) | 15.4 ± 15.5 (n=3) |
| Joint | BioDeepAV | 51.0 ± 7.4 (n=3) | - | - | 52.5 ± 5.6 (n=3) | 39.8 ± 8.9 (n=3) | 35.4 ± 9.6 (n=3) | 39.7 ± 9.7 (n=3) |
| Joint | FF++ | 52.2 ± 5.1 (n=3) | - | - | 44.7 ± 4.1 (n=3) | 25.3 ± 9.5 (n=3) | 25.2 ± 6.0 (n=3) | 34.3 ± 13.9 (n=3) |

Stability is not correctness: no manipulation-region ground truth exists.

## 5. Statistics (paired cluster bootstrap + seed sign agreement)

| train_on | test_dataset | setting | comparison | metric | better_direction | n_seeds | mean_diff | sd_diff_across_seeds | per_seed_diffs | sign_agreement | ci95_low | ci95_high | n_clusters | n_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | in-domain | 6A - T5-5A | hard_auc | higher | 3 | -0.0627 | 0.0590 | 42:-0.0293;123:-0.0280;2026:-0.1308 | True | -0.0880 | -0.0009 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6B - 6A | degraded_auc | higher | 3 | -0.0224 | 0.0089 | 42:-0.0326;123:-0.0162;2026:-0.0185 | True | -0.0446 | -0.0013 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6B - 6A | gradcam_iou_all | higher | 3 | 0.0550 | 0.0673 | 42:+0.1318;123:+0.0271;2026:+0.0061 | True | 0.0192 | 0.0793 | 53 | 1000 | supported improvement |
| BioDeepAV | BioDeepAV | in-domain | 6B - 6A | gradcam_iou_hard | higher | 3 | 0.0594 | 0.0330 | 42:+0.0968;123:+0.0464;2026:+0.0349 | True | 0.0143 | 0.0908 | 53 | 1000 | supported improvement |
| BioDeepAV | BioDeepAV | in-domain | 6C - 6B | gradcam_iou_all | higher | 3 | -0.0397 | 0.0363 | 42:-0.0317;123:-0.0793;2026:-0.0081 | True | -0.0605 | -0.0057 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6D - T5-5A | clean_auc | higher | 3 | -0.0379 | 0.0207 | 42:-0.0615;123:-0.0228;2026:-0.0294 | True | -0.0536 | -0.0031 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6D - T5-5A | degraded_auc | higher | 3 | -0.0641 | 0.0339 | 42:-0.1023;123:-0.0376;2026:-0.0525 | True | -0.0745 | -0.0107 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6D - T5-5A | hard_auc | higher | 3 | -0.0767 | 0.0610 | 42:-0.1424;123:-0.0219;2026:-0.0659 | True | -0.1005 | -0.0075 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6D - T5-5A | noise_auc | higher | 3 | -0.0499 | 0.0159 | 42:-0.0327;123:-0.0640;2026:-0.0529 | True | -0.0659 | -0.0016 | 53 | 999 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6D - T5-5A | flip_rate_pct | lower | 3 | 5.4956 | 6.0688 | 42:+12.4387;123:+2.8459;2026:+1.2022 | True | 2.0132 | 6.4083 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6D - T5-5A | ece_clean | lower | 3 | 0.1273 | 0.0940 | 42:+0.1322;123:+0.0309;2026:+0.2187 | True | 0.0875 | 0.2145 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6B - T5-5C | hard_auc | higher | 3 | -0.0635 | 0.0108 | 42:-0.0550;123:-0.0599;2026:-0.0756 | True | -0.1033 | -0.0001 | 53 | 1000 | supported deterioration |
| BioDeepAV | BioDeepAV | in-domain | 6B - T5-5C | ece_clean | lower | 3 | 0.1109 | 0.0158 | 42:+0.0952;123:+0.1109;2026:+0.1267 | True | 0.0167 | 0.1620 | 53 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6A - T5-5A | clean_auc | higher | 3 | -0.1235 | 0.0352 | 42:-0.1483;123:-0.0833;2026:-0.1390 | True | -0.1795 | -0.0330 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6A - T5-5A | degraded_auc | higher | 3 | -0.0727 | 0.0229 | 42:-0.0866;123:-0.0463;2026:-0.0852 | True | -0.1101 | -0.0080 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6A - T5-5A | noise_auc | higher | 3 | -0.1048 | 0.0701 | 42:-0.1744;123:-0.0342;2026:-0.1059 | True | -0.1746 | -0.0143 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6A - T5-5A | gradcam_iou_all | higher | 3 | -0.1022 | 0.0575 | 42:-0.1631;123:-0.0946;2026:-0.0489 | True | -0.1274 | -0.0695 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6B - 6A | flip_rate_pct | lower | 3 | 2.7513 | 1.2306 | 42:+3.8988;123:+1.4517;2026:+2.9034 | True | 0.5223 | 3.9595 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6C - 6B | ece_clean | lower | 3 | -0.0632 | 0.0469 | 42:-0.0577;123:-0.0193;2026:-0.1125 | True | -0.0692 | -0.0502 | 101 | 1000 | supported improvement |
| BioDeepAV | FF++ | cross-dataset | 6D - T5-5A | clean_auc | higher | 3 | -0.1270 | 0.0218 | 42:-0.1446;123:-0.1026;2026:-0.1338 | True | -0.1863 | -0.0316 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6D - T5-5A | degraded_auc | higher | 3 | -0.0849 | 0.0129 | 42:-0.0933;123:-0.0701;2026:-0.0913 | True | -0.1275 | -0.0127 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6B - T5-5C | clean_auc | higher | 3 | -0.0982 | 0.0621 | 42:-0.1638;123:-0.0402;2026:-0.0906 | True | -0.1437 | -0.0264 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6B - T5-5C | degraded_auc | higher | 3 | -0.0598 | 0.0524 | 42:-0.1115;123:-0.0067;2026:-0.0611 | True | -0.0916 | -0.0033 | 101 | 1000 | supported deterioration |
| BioDeepAV | FF++ | cross-dataset | 6B - T5-5C | gradcam_iou_all | higher | 3 | -0.0610 | 0.0430 | 42:-0.0613;123:-0.1039;2026:-0.0179 | True | -0.0805 | -0.0394 | 101 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6A - T5-5A | clean_auc | higher | 3 | -0.2156 | 0.0398 | 42:-0.2427;123:-0.1699;2026:-0.2342 | True | -0.3852 | -0.0278 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6A - T5-5A | degraded_auc | higher | 3 | -0.1002 | 0.0403 | 42:-0.1293;123:-0.0542;2026:-0.1170 | True | -0.2584 | -0.0295 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6A - T5-5A | worst_type_auc | higher | 3 | -0.2368 | 0.0504 | 42:-0.2664;123:-0.1786;2026:-0.2653 | True | -0.4513 | -0.1375 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6A - T5-5A | flip_rate_pct | lower | 3 | 25.0818 | 1.0570 | 42:+24.4112;123:+26.3003;2026:+24.5339 | True | 18.5990 | 25.8812 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - 6A | gradcam_iou_all | higher | 3 | 0.2538 | 0.0616 | 42:+0.1884;123:+0.3109;2026:+0.2621 | True | 0.1878 | 0.2879 | 53 | 1000 | supported improvement |
| FF++ | BioDeepAV | cross-dataset | 6B - 6A | gradcam_iou_hard | higher | 3 | 0.2979 | 0.0736 | 42:+0.2237;123:+0.3710;2026:+0.2989 | True | 0.2124 | 0.3231 | 53 | 1000 | supported improvement |
| FF++ | BioDeepAV | cross-dataset | 6C - 6B | gradcam_iou_all | higher | 3 | -0.0544 | 0.0224 | 42:-0.0623;123:-0.0717;2026:-0.0291 | True | -0.0819 | -0.0343 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6C - 6B | gradcam_iou_hard | higher | 3 | -0.0648 | 0.0354 | 42:-0.0811;123:-0.0890;2026:-0.0241 | True | -0.0941 | -0.0426 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - 6C | gradcam_iou_all | higher | 3 | -0.1016 | 0.1200 | 42:-0.0067;123:-0.2365;2026:-0.0615 | True | -0.1132 | -0.0723 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - 6C | gradcam_iou_hard | higher | 3 | -0.1113 | 0.1345 | 42:-0.0038;123:-0.2621;2026:-0.0681 | True | -0.1242 | -0.0778 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - T5-5A | clean_auc | higher | 3 | -0.2812 | 0.0466 | 42:-0.2814;123:-0.2345;2026:-0.3278 | True | -0.4261 | -0.0994 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - T5-5A | degraded_auc | higher | 3 | -0.1485 | 0.0302 | 42:-0.1356;123:-0.1270;2026:-0.1830 | True | -0.2872 | -0.0587 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - T5-5A | hard_auc | higher | 3 | -0.0942 | 0.0211 | 42:-0.0807;123:-0.0835;2026:-0.1185 | True | -0.2565 | -0.0321 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - T5-5A | worst_type_auc | higher | 3 | -0.2939 | 0.0359 | 42:-0.2962;123:-0.2569;2026:-0.3287 | True | -0.4720 | -0.1685 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6D - T5-5A | flip_rate_pct | lower | 3 | 24.0350 | 2.7467 | 42:+26.9136;123:+23.7488;2026:+21.4426 | True | 16.8758 | 26.5199 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | clean_auc | higher | 3 | -0.2542 | 0.0262 | 42:-0.2662;123:-0.2722;2026:-0.2242 | True | -0.3932 | -0.0456 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | degraded_auc | higher | 3 | -0.1266 | 0.0261 | 42:-0.1562;123:-0.1167;2026:-0.1069 | True | -0.2686 | -0.0421 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | hard_auc | higher | 3 | -0.0672 | 0.0339 | 42:-0.1031;123:-0.0628;2026:-0.0358 | True | -0.2286 | -0.0119 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | worst_type_auc | higher | 3 | -0.2715 | 0.0231 | 42:-0.2792;123:-0.2898;2026:-0.2456 | True | -0.4547 | -0.1478 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | flip_rate_pct | lower | 3 | 23.1109 | 3.5535 | 42:+20.2404;123:+27.0854;2026:+22.0069 | True | 18.4340 | 24.4249 | 53 | 1000 | supported deterioration |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | gradcam_iou_all | higher | 3 | 0.0829 | 0.0477 | 42:+0.0590;123:+0.0519;2026:+0.1378 | True | 0.0433 | 0.1309 | 53 | 1000 | supported improvement |
| FF++ | BioDeepAV | cross-dataset | 6B - T5-5C | gradcam_iou_hard | higher | 3 | 0.1665 | 0.0182 | 42:+0.1615;123:+0.1513;2026:+0.1868 | True | 0.1116 | 0.2028 | 53 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6A - T5-5A | flip_rate_pct | lower | 3 | 3.0969 | 2.2498 | 42:+4.7698;123:+0.5392;2026:+3.9818 | True | 0.5637 | 4.3882 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6A - T5-5A | ece_clean | lower | 3 | -0.0254 | 0.0142 | 42:-0.0299;123:-0.0094;2026:-0.0368 | True | -0.0469 | -0.0159 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6A - T5-5A | gradcam_iou_hard | higher | 3 | 0.0873 | 0.0450 | 42:+0.1368;123:+0.0487;2026:+0.0764 | True | 0.0553 | 0.1162 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6B - 6A | flip_rate_pct | lower | 3 | -2.4886 | 2.8213 | 42:-0.5392;123:-5.7238;2026:-1.2028 | True | -3.3170 | -1.0323 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6B - 6A | gradcam_iou_all | higher | 3 | 0.2302 | 0.0415 | 42:+0.1878;123:+0.2708;2026:+0.2320 | True | 0.2153 | 0.2508 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6B - 6A | gradcam_iou_hard | higher | 3 | 0.2897 | 0.0487 | 42:+0.2417;123:+0.3390;2026:+0.2884 | True | 0.2653 | 0.3298 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6C - 6B | degraded_auc | higher | 3 | -0.0152 | 0.0138 | 42:-0.0117;123:-0.0036;2026:-0.0305 | True | -0.0257 | -0.0042 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6C - 6B | hard_auc | higher | 3 | -0.0364 | 0.0101 | 42:-0.0285;123:-0.0328;2026:-0.0478 | True | -0.0576 | -0.0085 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6C - 6B | noise_auc | higher | 3 | -0.0172 | 0.0098 | 42:-0.0188;123:-0.0261;2026:-0.0067 | True | -0.0333 | -0.0021 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6C - 6B | gradcam_iou_all | higher | 3 | -0.0441 | 0.0215 | 42:-0.0429;123:-0.0663;2026:-0.0233 | True | -0.0542 | -0.0371 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6C - 6B | gradcam_iou_hard | higher | 3 | -0.0532 | 0.0269 | 42:-0.0522;123:-0.0807;2026:-0.0268 | True | -0.0737 | -0.0404 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6D - 6C | degraded_auc | higher | 3 | -0.0091 | 0.0066 | 42:-0.0160;123:-0.0029;2026:-0.0083 | True | -0.0166 | -0.0019 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6D - T5-5A | clean_auc | higher | 3 | -0.0567 | 0.0173 | 42:-0.0430;123:-0.0510;2026:-0.0762 | True | -0.0885 | -0.0064 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6D - T5-5A | degraded_auc | higher | 3 | -0.0806 | 0.0227 | 42:-0.0770;123:-0.0599;2026:-0.1049 | True | -0.1261 | -0.0075 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6D - T5-5A | noise_auc | higher | 3 | -0.0745 | 0.0279 | 42:-0.0658;123:-0.0520;2026:-0.1058 | True | -0.1729 | -0.0059 | 101 | 1000 | supported deterioration |
| FF++ | FF++ | in-domain | 6D - T5-5A | gradcam_iou_hard | higher | 3 | 0.2140 | 0.1652 | 42:+0.3438;123:+0.0280;2026:+0.2704 | True | 0.1861 | 0.2460 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6B - T5-5C | ece_clean | lower | 3 | -0.0462 | 0.0267 | 42:-0.0577;123:-0.0157;2026:-0.0653 | True | -0.0627 | -0.0377 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6B - T5-5C | gradcam_iou_all | higher | 3 | 0.0508 | 0.0278 | 42:+0.0215;123:+0.0541;2026:+0.0769 | True | 0.0383 | 0.0659 | 101 | 1000 | supported improvement |
| FF++ | FF++ | in-domain | 6B - T5-5C | gradcam_iou_hard | higher | 3 | 0.1278 | 0.0054 | 42:+0.1216;123:+0.1318;2026:+0.1299 | True | 0.0985 | 0.1747 | 101 | 1000 | supported improvement |
| Joint | BioDeepAV | joint | 6B - 6A | gradcam_iou_all | higher | 3 | 0.0799 | 0.0349 | 42:+0.0906;123:+0.0410;2026:+0.1082 | True | 0.0698 | 0.0920 | 53 | 1000 | supported improvement |
| Joint | BioDeepAV | joint | 6B - 6A | gradcam_iou_hard | higher | 3 | 0.0511 | 0.0462 | 42:+0.0555;123:+0.0029;2026:+0.0950 | True | 0.0410 | 0.0884 | 53 | 1000 | supported improvement |
| Joint | BioDeepAV | joint | 6D - 6C | degraded_auc | higher | 3 | -0.0434 | 0.0020 | 42:-0.0411;123:-0.0443;2026:-0.0447 | True | -0.0707 | -0.0079 | 53 | 1000 | supported deterioration |
| Joint | BioDeepAV | joint | 6D - 6C | hard_auc | higher | 3 | -0.0398 | 0.0035 | 42:-0.0432;123:-0.0362;2026:-0.0399 | True | -0.0645 | -0.0119 | 53 | 1000 | supported deterioration |
| Joint | BioDeepAV | joint | 6D - 6C | worst_type_auc | higher | 3 | -0.0702 | 0.0283 | 42:-0.0933;123:-0.0387;2026:-0.0787 | True | -0.1268 | -0.0203 | 53 | 1000 | supported deterioration |
| Joint | FF++ | joint | 6B - 6A | gradcam_iou_all | higher | 3 | 0.1474 | 0.0407 | 42:+0.1766;123:+0.1009;2026:+0.1648 | True | 0.1342 | 0.1616 | 101 | 1000 | supported improvement |
| Joint | FF++ | joint | 6B - 6A | gradcam_iou_hard | higher | 3 | 0.1422 | 0.0385 | 42:+0.1577;123:+0.0984;2026:+0.1706 | True | 0.1236 | 0.1710 | 101 | 1000 | supported improvement |
| Joint | FF++ | joint | 6D - 6C | ece_clean | lower | 3 | 0.0159 | 0.0250 | 42:+0.0448;123:+0.0020;2026:+0.0010 | True | 0.0135 | 0.0191 | 101 | 1000 | supported deterioration |
| Joint | FF++ | joint | 6D - 6C | gradcam_iou_hard | higher | 3 | -0.0958 | 0.1019 | 42:-0.2101;123:-0.0143;2026:-0.0631 | True | -0.1323 | -0.0733 | 101 | 1000 | supported deterioration |

_196 further comparisons inconclusive (statistics_test6.csv)._

## 6. Hypothesis verdicts

| hypothesis | statement | comparison | verdict | evidence |
|---|---|---|---|---|
| H1 | The MSTF-AV architecture improves robustness over the Test 5 baseline | 6A - T5-5A | inconclusive / not supported | BioDeepAV->BioDeepAV clean_auc: -0.0099 [-0.0337, +0.0196] (inconclusive); BioDeepAV->BioDeepAV degraded_auc: -0.0437 [-0.0673, +0.0061] (inconclusive); BioDeepAV->BioDeepAV worst_type_auc: -0.0430 [-0.0892, +0.0293] (inconclusive); FF++->FF++ clean_auc: -0.0441 [-0.0807, +0.0170] (inconclusive); FF++->FF++ degraded_auc: -0.0548 [-0.0994, +0.0209] (inconclusive); FF++->FF++ worst_type_auc: -0.0509 [-0.1281, +0.0766] (inconclusive) |
| H2 | Paired Grad-CAM + supervised contrastive consistency improves explanation stability | 6B - 6A | partially supported | BioDeepAV->BioDeepAV flip_rate_pct: +3.5819 [-0.6851, +4.8319] (inconclusive); BioDeepAV->BioDeepAV gradcam_iou_all: +0.0550 [+0.0192, +0.0793] (supported improvement); BioDeepAV->BioDeepAV gradcam_iou_hard: +0.0594 [+0.0143, +0.0908] (supported improvement); FF++->FF++ flip_rate_pct: -2.4886 [-3.3170, -1.0323] (supported improvement); FF++->FF++ gradcam_iou_all: +0.2302 [+0.2153, +0.2508] (supported improvement); FF++->FF++ gradcam_iou_hard: +0.2897 [+0.2653, +0.3298] (supported improvement); Joint->BioDeepAV flip_rate_pct: +0.8423 [+0.4219, +2.1875] (inconclusive); Joint->BioDeepAV gradcam_iou_all: +0.0799 [+0.0698, +0.0920] (supported improvement); Joint->BioDeepAV gradcam_iou_hard: +0.0511 [+0.0410, +0.0884] (supported improvement); Joint->FF++ flip_rate_pct: -0.0000 [-0.1811, +0.3981] (inconclusive); Joint->FF++ gradcam_iou_all: +0.1474 [+0.1342, +0.1616] (supported improvement); Joint->FF++ gradcam_iou_hard: +0.1422 [+0.1236, +0.1710] (supported improvement) |
| H3 | Group-DRO sampling + EMA improve worst-case robustness without starving types | 6C - 6B | not supported (supported deterioration) | BioDeepAV->BioDeepAV degraded_auc: +0.0056 [+0.0016, +0.0254] (inconclusive); BioDeepAV->BioDeepAV worst_type_auc: -0.0054 [-0.0269, +0.0573] (inconclusive); BioDeepAV->BioDeepAV noise_auc: -0.0112 [-0.0308, +0.0326] (inconclusive); FF++->FF++ degraded_auc: -0.0152 [-0.0257, -0.0042] (supported deterioration); FF++->FF++ worst_type_auc: -0.0651 [-0.1198, +0.0105] (inconclusive); FF++->FF++ noise_auc: -0.0172 [-0.0333, -0.0021] (supported deterioration); Joint->BioDeepAV degraded_auc: +0.0155 [-0.0000, +0.0220] (inconclusive); Joint->BioDeepAV worst_type_auc: -0.0030 [-0.0172, +0.0341] (inconclusive); Joint->BioDeepAV noise_auc: +0.0173 [-0.0130, +0.0297] (inconclusive); Joint->FF++ degraded_auc: -0.0215 [-0.0372, +0.0062] (inconclusive); Joint->FF++ worst_type_auc: -0.0272 [-0.0671, +0.0335] (inconclusive); Joint->FF++ noise_auc: -0.0250 [-0.0506, +0.0101] (inconclusive) |
| H4 | Shortcut adversaries reduce shortcut use (better cross-dataset AUC, calibration) | 6D - 6C | not supported (supported deterioration) | BioDeepAV->BioDeepAV ece_clean: +0.0079 [+0.0038, +0.0578] (inconclusive); BioDeepAV->FF++ clean_auc: +0.0044 [-0.0023, +0.0114] (inconclusive); BioDeepAV->FF++ degraded_auc: +0.0071 [-0.0036, +0.0152] (inconclusive); BioDeepAV->FF++ ece_clean: -0.0041 [-0.0116, +0.0125] (inconclusive); FF++->BioDeepAV clean_auc: -0.0202 [-0.0383, -0.0046] (inconclusive); FF++->BioDeepAV degraded_auc: -0.0120 [-0.0245, +0.0010] (inconclusive); FF++->BioDeepAV ece_clean: +0.0157 [-0.0201, +0.0477] (inconclusive); FF++->FF++ ece_clean: +0.0352 [+0.0331, +0.0404] (inconclusive); Joint->BioDeepAV ece_clean: -0.0252 [-0.0454, +0.0165] (inconclusive); Joint->FF++ ece_clean: +0.0159 [+0.0135, +0.0191] (supported deterioration) |
| H5 | Full Test 6 (6D) beats the Test 5 baseline | 6D - T5-5A | mixed | BioDeepAV->BioDeepAV degraded_auc: -0.0641 [-0.0745, -0.0107] (supported deterioration); BioDeepAV->BioDeepAV worst_type_auc: -0.0676 [-0.1235, -0.0312] (inconclusive); BioDeepAV->BioDeepAV flip_rate_pct: +5.4956 [+2.0132, +6.4083] (supported deterioration); BioDeepAV->BioDeepAV gradcam_iou_hard: +0.0289 [-0.0267, +0.0915] (inconclusive); FF++->FF++ degraded_auc: -0.0806 [-0.1261, -0.0075] (supported deterioration); FF++->FF++ worst_type_auc: -0.1683 [-0.2751, +0.0479] (inconclusive); FF++->FF++ flip_rate_pct: +0.0553 [-1.5114, +1.4674] (inconclusive); FF++->FF++ gradcam_iou_hard: +0.2140 [+0.1861, +0.2460] (supported improvement) |

## 7. Notes for the thesis

- Test 6 and Test 5 share split, test videos, XAI pairs and seeds: differences are paired and attributable to model/training.
- 6A − T5-5A isolates the architecture (same recipe); 6B/6C/6D add one training block each.
- Report only 'supported' effects; describe the rest as inconclusive at this sample size and seed count.
- Joint models have no held-out dataset; cross-dataset claims come only from single-dataset models.
- Grad-CAM consistency is trained in 6B-6D, so their stability is partly optimised (report alongside clean reference IoU).