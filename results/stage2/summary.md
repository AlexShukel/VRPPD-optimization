# Stage 2 summary

Gap % is relative to the CEA reference (best of the CEA runs, 60 s budget); gap-best is relative to the best value known for the instance over CEA and all LNS runs. Mean ± sd over instances × repeats.

## CEA reference (distance objective)

| cell | instances | runs per instance | mean best | mean CEA runtime (s) |
|---|---|---|---|---|
| 10_10 | 20 | 3 | 3581.1 | 1.3 |
| 10_20 | 20 | 3 | 5619.8 | 11.8 |
| 20_50 | 20 | 3 | 9767.7 | 60.2 |

## Budget: iters5000

| cell | operator | runs | objective (mean ± sd) | gap vs CEA % (mean ± sd) | gap vs best-known % | iterations (mean) | it/s | time to 1 % of CEA: reached / iters / s | accept rate | destroy share of time |
|---|---|---|---|---|---|---|---|---|---|---|
| 10_10 | random | 100 | 3639.2 ± 1131.5 | +1.70 ± 2.06 | +1.83 ± 2.00 | 5000 | 526479 | 50% / 66 / 0.00 | 0.91 | 0.30 |
| 10_10 | worst | 100 | 3645.6 ± 1143.8 | +1.79 ± 2.15 | +1.92 ± 2.15 | 5000 | 512388 | 49% / 101 / 0.00 | 0.91 | 0.37 |
| 10_10 | related | 100 | 3661.0 ± 1141.0 | +2.29 ± 2.70 | +2.42 ± 2.66 | 5000 | 509843 | 46% / 50 / 0.00 | 0.83 | 0.36 |
| 10_10 | adaptive | 100 | 3637.8 ± 1134.3 | +1.64 ± 1.99 | +1.77 ± 1.97 | 5000 | 488262 | 51% / 47 / 0.00 | 0.88 | 0.38 |
| 10_10 | neural | 100 | 3759.1 ± 1166.4 | +5.08 ± 3.80 | +5.21 ± 3.78 | 5000 | 16138 | 12% / 2 / 0.00 | 0.99 | 0.98 |
| 10_10 | neural_tau | 100 | 3642.5 ± 1133.3 | +1.78 ± 1.99 | +1.91 ± 2.00 | 5000 | 15658 | 46% / 53 / 0.00 | 0.92 | 0.98 |
| 10_20 | random | 100 | 5874.8 ± 1072.7 | +4.66 ± 3.58 | +5.51 ± 3.50 | 5000 | 253716 | 13% / 578 / 0.00 | 0.95 | 0.23 |
| 10_20 | worst | 100 | 5885.2 ± 1069.6 | +4.85 ± 3.40 | +5.70 ± 3.35 | 5000 | 231083 | 12% / 537 / 0.00 | 0.92 | 0.32 |
| 10_20 | related | 100 | 5901.9 ± 1091.3 | +5.10 ± 3.90 | +5.96 ± 4.26 | 5000 | 259181 | 14% / 335 / 0.00 | 0.88 | 0.27 |
| 10_20 | adaptive | 100 | 5854.3 ± 1065.6 | +4.30 ± 3.47 | +5.14 ± 3.40 | 5000 | 237231 | 23% / 535 / 0.00 | 0.92 | 0.28 |
| 10_20 | neural | 100 | 6393.9 ± 1196.8 | +13.82 ± 5.36 | +14.74 ± 5.48 | 5000 | 7965 | 0% / – / – | 0.98 | 0.98 |
| 10_20 | neural_tau | 100 | 5887.7 ± 1089.4 | +4.84 ± 3.40 | +5.69 ± 3.51 | 5000 | 8087 | 10% / 929 / 0.12 | 0.96 | 0.97 |
| 20_50 | random | 100 | 10154.5 ± 745.6 | +3.96 ± 2.45 | +5.59 ± 2.53 | 5000 | 46803 | 15% / 2202 / 0.05 | 0.83 | 0.09 |
| 20_50 | worst | 100 | 10223.4 ± 745.1 | +4.67 ± 2.69 | +6.32 ± 2.80 | 5000 | 46373 | 2% / 2056 / 0.05 | 0.78 | 0.18 |
| 20_50 | related | 100 | 10045.1 ± 690.7 | +2.87 ± 2.14 | +4.48 ± 2.20 | 5000 | 50870 | 24% / 1805 / 0.04 | 0.58 | 0.13 |
| 20_50 | adaptive | 100 | 10014.1 ± 726.0 | +2.52 ± 2.13 | +4.13 ± 2.31 | 5000 | 41929 | 28% / 1544 / 0.04 | 0.72 | 0.14 |
| 20_50 | neural | 100 | 11775.1 ± 881.8 | +20.65 ± 6.02 | +22.56 ± 6.54 | 5000 | 3110 | 0% / – / – | 0.90 | 0.95 |
| 20_50 | neural_tau | 100 | 10155.8 ± 714.0 | +4.00 ± 2.78 | +5.64 ± 2.90 | 5000 | 3092 | 12% / 1774 / 0.56 | 0.85 | 0.94 |

Number of instances (out of 20) on which the operator's mean over repeats is the best (ties count for all):

| cell | adaptive | neural_tau | random | related | worst |
|---|---|---|---|---|---|
| 10_10 | 14 | 11 | 12 | 10 | 12 |
| 10_20 | 9 | 2 | 2 | 4 | 4 |
| 20_50 | 12 | 0 | 2 | 6 | 0 |

## Budget: time30s

| cell | operator | runs | objective (mean ± sd) | gap vs CEA % (mean ± sd) | gap vs best-known % | iterations (mean) | it/s | time to 1 % of CEA: reached / iters / s | accept rate | destroy share of time |
|---|---|---|---|---|---|---|---|---|---|---|
| 10_10 | random | 100 | 3598.2 ± 1124.1 | +0.51 ± 1.56 | +0.64 ± 1.45 | 16782904 | 559430 | 81% / 45678 / 0.09 | 0.92 | 0.29 |
| 10_10 | worst | 100 | 3613.6 ± 1126.0 | +0.97 ± 1.84 | +1.10 ± 1.79 | 15794723 | 526491 | 72% / 114241 / 0.23 | 0.91 | 0.36 |
| 10_10 | related | 100 | 3653.6 ± 1135.8 | +2.12 ± 2.65 | +2.25 ± 2.59 | 16115145 | 537171 | 50% / 6629 / 0.01 | 0.83 | 0.35 |
| 10_10 | adaptive | 100 | 3596.0 ± 1119.1 | +0.49 ± 1.42 | +0.62 ± 1.30 | 15540591 | 518020 | 78% / 28591 / 0.06 | 0.89 | 0.35 |
| 10_10 | neural | 100 | 3759.1 ± 1166.4 | +5.08 ± 3.80 | +5.21 ± 3.78 | 492660 | 16422 | 12% / 2 / 0.00 | 0.99 | 0.98 |
| 10_10 | neural_tau | 100 | 3627.5 ± 1134.3 | +1.33 ± 1.99 | +1.46 ± 1.94 | 487912 | 16264 | 62% / 4217 / 0.25 | 0.92 | 0.98 |
| 10_20 | random | 100 | 5669.9 ± 1019.2 | +1.00 ± 2.58 | +1.81 ± 2.68 | 7794958 | 259832 | 67% / 103825 / 0.41 | 0.95 | 0.18 |
| 10_20 | worst | 100 | 5688.7 ± 1013.4 | +1.37 ± 2.81 | +2.19 ± 2.93 | 6812835 | 227094 | 61% / 208864 / 0.86 | 0.93 | 0.32 |
| 10_20 | related | 100 | 5819.5 ± 1078.8 | +3.65 ± 4.05 | +4.50 ± 4.41 | 7728686 | 257623 | 38% / 53507 / 0.21 | 0.90 | 0.26 |
| 10_20 | adaptive | 100 | 5652.8 ± 1003.8 | +0.74 ± 2.68 | +1.56 ± 2.65 | 7163387 | 238780 | 72% / 112457 / 0.49 | 0.92 | 0.27 |
| 10_20 | neural | 100 | 6393.9 ± 1196.8 | +13.82 ± 5.36 | +14.74 ± 5.48 | 246265 | 8209 | 0% / – / – | 0.98 | 0.98 |
| 10_20 | neural_tau | 100 | 5784.4 ± 1071.5 | +2.99 ± 3.26 | +3.83 ± 3.29 | 243280 | 8109 | 28% / 6785 / 0.85 | 0.96 | 0.97 |
| 20_50 | random | 100 | 9734.2 ± 682.7 | -0.34 ± 1.16 | +1.22 ± 1.08 | 1441942 | 48065 | 89% / 68492 / 1.52 | 0.83 | 0.08 |
| 20_50 | worst | 100 | 9765.6 ± 695.5 | -0.03 ± 1.35 | +1.54 ± 1.29 | 1354597 | 45153 | 87% / 88634 / 2.07 | 0.79 | 0.19 |
| 20_50 | related | 100 | 9693.6 ± 667.9 | -0.74 ± 1.42 | +0.81 ± 0.93 | 1590291 | 53010 | 94% / 129772 / 2.65 | 0.57 | 0.12 |
| 20_50 | adaptive | 100 | 9677.0 ± 664.5 | -0.92 ± 1.16 | +0.63 ± 0.77 | 1400383 | 46679 | 96% / 57637 / 1.31 | 0.73 | 0.14 |
| 20_50 | neural | 100 | 11775.1 ± 881.8 | +20.65 ± 6.02 | +22.56 ± 6.54 | 95861 | 3195 | 0% / – / – | 0.90 | 0.96 |
| 20_50 | neural_tau | 100 | 9888.4 ± 688.4 | +1.25 ± 1.51 | +2.84 ± 1.71 | 94108 | 3137 | 54% / 12376 / 3.95 | 0.84 | 0.94 |

Number of instances (out of 20) on which the operator's mean over repeats is the best (ties count for all):

| cell | adaptive | neural_tau | random | related | worst |
|---|---|---|---|---|---|
| 10_10 | 16 | 6 | 16 | 5 | 11 |
| 10_20 | 13 | 0 | 8 | 3 | 4 |
| 20_50 | 9 | 0 | 3 | 6 | 2 |

## B0 training

Validation metrics at the best (early-stopped) epoch. MSE is on per-state z-scores (predictions also z-scored per state). Spearman = mean per-state rank correlation; p@k = overlap of predicted and true top-k (random expectation in the last column).

| group | width | depth | loss | seed | epochs (best/run) | val loss | val MSE(z) | Spearman | p@k | p@k random | best-set recall | params | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best | 128 | 2 | mse | 0 | 4/14 | 0.9444 | 1.533 | 0.174 | 0.210 | 0.153 | 0.198 | 17793 | 11 |
| best | 128 | 2 | mse | 1 | 6/16 | 0.9441 | 1.547 | 0.170 | 0.207 | 0.153 | 0.205 | 17793 | 13 |
| best | 128 | 2 | mse | 2 | 5/15 | 0.9459 | 1.544 | 0.170 | 0.212 | 0.153 | 0.194 | 17793 | 12 |
| grid | 64 | 2 | mse | 0 | 9/19 | 0.9428 | 1.544 | 0.170 | 0.205 | 0.153 | 0.191 | 4801 | 10 |
| grid | 64 | 3 | mse | 0 | 9/19 | 0.9436 | 1.539 | 0.168 | 0.213 | 0.153 | 0.203 | 8961 | 12 |
| grid | 128 | 2 | mse | 0 | 4/14 | 0.9444 | 1.533 | 0.174 | 0.210 | 0.153 | 0.198 | 17793 | 11 |
| grid | 128 | 3 | mse | 0 | 6/16 | 0.9451 | 1.543 | 0.166 | 0.202 | 0.153 | 0.189 | 34305 | 17 |
| grid | 64 | 2 | rank | 0 | 3/13 | 3.2258 | 1.689 | 0.106 | 0.227 | 0.153 | 0.215 | 4801 | 7 |
| grid | 64 | 3 | rank | 0 | 5/15 | 3.2266 | 1.694 | 0.110 | 0.223 | 0.153 | 0.224 | 8961 | 11 |
| grid | 128 | 2 | rank | 0 | 5/15 | 3.2262 | 1.685 | 0.108 | 0.224 | 0.153 | 0.217 | 17793 | 12 |
| grid | 128 | 3 | rank | 0 | 3/13 | 3.2284 | 1.679 | 0.113 | 0.227 | 0.153 | 0.224 | 34305 | 15 |

## Label quality (stats_train.md)

Label statistics for `/Users/srv/Git/VRPPD-optimization/results/stage2/labels/train.jsonl`: 8550 states, 256500 order rows, M = 300.

| cell | states | try-improvement std (% of obj) | best − median try (% of obj) | best try (% of obj) | median try (% of obj) | share of tries with gain > 0 |
|---|---|---|---|---|---|---|
| 10_20 | 5700 | 0.681 | 1.433 | 1.441 | 0.008 | 0.113 |
| 20_50 | 2850 | 1.112 | 2.132 | 2.203 | 0.071 | 0.359 |

Share of states where the best try beats the median try by more than X % of the objective: 0.5%: 0.600, 1.0%: 0.499, 2.0%: 0.338, 5.0%: 0.070

| feature | Pearson r with z-score | Spearman ρ with z-score | Spearman (rank by this feature alone, per state) | precision@k (feature alone) |
|---|---|---|---|---|
| route_saving | -0.010 | -0.058 | -0.067 | 0.170 |
| start_to_pickup | -0.046 | -0.031 | -0.057 | 0.142 |
| pickup_to_delivery | -0.174 | -0.164 | -0.161 | 0.119 |
| load_factor | 0.025 | 0.015 | 0.014 | 0.157 |
| price_km | 0.000 | 0.001 | -0.002 | 0.167 |
| orders_on_route | -0.063 | -0.041 | -0.090 | 0.142 |
| nearest_other | -0.015 | -0.033 | -0.053 | 0.148 |
| relative_position | -0.021 | -0.021 | -0.027 | 0.149 |
| (random order) | – | – | -0.001 | 0.151 |


## Label quality (stats_val.md)

Label statistics for `/Users/srv/Git/VRPPD-optimization/results/stage2/labels/val.jsonl`: 855 states, 25650 order rows, M = 300.

| cell | states | try-improvement std (% of obj) | best − median try (% of obj) | best try (% of obj) | median try (% of obj) | share of tries with gain > 0 |
|---|---|---|---|---|---|---|
| 10_20 | 570 | 0.692 | 1.302 | 1.307 | 0.006 | 0.104 |
| 20_50 | 285 | 1.233 | 2.289 | 2.361 | 0.072 | 0.382 |

Share of states where the best try beats the median try by more than X % of the objective: 0.5%: 0.579, 1.0%: 0.489, 2.0%: 0.333, 5.0%: 0.077

| feature | Pearson r with z-score | Spearman ρ with z-score | Spearman (rank by this feature alone, per state) | precision@k (feature alone) |
|---|---|---|---|---|
| route_saving | -0.010 | -0.041 | -0.066 | 0.165 |
| start_to_pickup | -0.053 | -0.036 | -0.069 | 0.134 |
| pickup_to_delivery | -0.151 | -0.143 | -0.137 | 0.130 |
| load_factor | 0.016 | 0.010 | 0.010 | 0.157 |
| price_km | -0.009 | -0.007 | -0.030 | 0.147 |
| orders_on_route | -0.053 | -0.032 | -0.075 | 0.150 |
| nearest_other | -0.030 | -0.010 | -0.044 | 0.136 |
| relative_position | -0.033 | -0.039 | -0.056 | 0.148 |
| (random order) | – | – | -0.022 | 0.143 |


