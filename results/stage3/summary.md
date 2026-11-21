# Stage 3 summary

## Coverage

| batch | status | host | finished |
|---|---|---|---|
| data | done | pingwin | 2026-11-18T11:50:31 |
| b0 | done | pingwin | 2026-11-18T17:13:21 |
| v1 | done | pingwin | 2026-11-18T23:52:57 |
| ref-10_10 | done | pingwin | 2026-11-18T23:56:51 |
| ref-10_20 | done | pingwin | 2026-11-19T00:25:55 |
| ref-20_50 | done | pingwin | 2026-11-19T03:46:21 |
| v2-a | done | pingwin | 2026-11-19T08:37:50 |
| v2-b | done | pingwin | 2026-11-19T11:37:57 |
| v2-c | done | pingwin | 2026-11-19T14:37:59 |
| v2-d | done | pingwin | 2026-11-19T17:38:03 |
| v2-e | done | pingwin | 2026-11-19T20:38:10 |
| select | done | pingwin | 2026-11-19T20:54:47 |
| iters-10_10 | done | pingwin | 2026-11-19T20:56:33 |
| iters-10_20 | done | pingwin | 2026-11-19T20:59:29 |
| iters-20_50 | done | pingwin | 2026-11-19T21:07:19 |
| time-10_10 | done | pingwin | 2026-11-20T01:39:21 |
| time-10_20 | done | pingwin | 2026-11-20T06:11:22 |
| time-20_50 | done | pingwin | 2026-11-20T10:43:24 |
| ksweep | done | pingwin | 2026-11-20T11:18:37 |
| large-30_100 | done | pingwin | 2026-11-20T12:28:55 |
| large-50_200 | done | pingwin | 2026-11-20T13:47:39 |
| large-100_500 | done | pingwin | 2026-11-20T16:02:16 |
| obj-price | done | pingwin | 2026-11-20T17:26:01 |
| obj-empty | done | pingwin | 2026-11-20T18:53:49 |

Only batches marked done enter the tables, figures and CSV files below.

## Notes

- Class 100_500: a probe of the v1 operator ran at 13.9 it/s, so one 5000-iteration run takes about 359 s (limit 600.0 s); the iteration budget for this class is 5000.

Gap to CEA % is relative to the CEA reference of the instance (best of its runs, same objective); gap to best known % is relative to the best value known for the instance (CEA and all LNS runs). Means over instances and repeats; sd over instances is of per-instance means, sd over repeats the mean per-instance sd. it/s is given for wall-time runs only (iteration-budget runs share the machine with many parallel runs).

## CEA references

| objective | cell | instances | runs | mean best | mean runtime (s) | host |
|---|---|---|---|---|---|---|
| distance | 10_10 | 20 | 10 | 3577.4 | 1.2 | pingwin |
| distance | 10_20 | 20 | 10 | 5574.9 | 8.7 | pingwin |
| distance | 20_50 | 20 | 10 | 9715.6 | 60.1 | pingwin |
| distance | 30_100 | 10 | 3 | 17089.9 | 60.8 | pingwin |
| distance | 50_200 | 10 | 3 | 31033.7 | 65.5 | pingwin |
| distance | 100_500 | 10 | 3 | 69487.0 | 112.9 | pingwin |
| empty | 10_10 | 20 | 3 | 52.7 | 1.5 | pingwin |
| empty | 10_20 | 20 | 3 | 105.8 | 9.8 | pingwin |
| empty | 20_50 | 20 | 3 | 130.6 | 60.2 | pingwin |
| price | 10_10 | 20 | 3 | 4823.9 | 1.3 | pingwin |
| price | 10_20 | 20 | 3 | 6984.6 | 11.4 | pingwin |
| price | 20_50 | 20 | 3 | 11355.9 | 60.2 | pingwin |

## Main comparison, iteration budget

| cell | operator | runs | gap to CEA % (± sd over instances) | sd over repeats | gap to best known % | objective (mean) |
|---|---|---|---|---|---|---|
| 10_10 | random | 200 | +1.86 ± 1.42 | 1.62 | +2.01 ± 1.45 | 3638.4 |
| 10_10 | worst | 200 | +1.97 ± 1.72 | 1.54 | +2.12 ± 1.79 | 3645.6 |
| 10_10 | related | 200 | +2.51 ± 2.19 | 1.75 | +2.66 ± 2.21 | 3662.6 |
| 10_10 | adaptive | 200 | +1.93 ± 1.52 | 1.60 | +2.08 ± 1.56 | 3640.9 |
| 10_10 | b0 | 200 | +2.16 ± 1.80 | 1.83 | +2.31 ± 1.85 | 3650.6 |
| 10_10 | v1 | 200 | +1.90 ± 1.56 | 1.52 | +2.05 ± 1.62 | 3640.4 |
| 10_10 | v2 | 200 | +3.09 ± 1.92 | 2.37 | +3.24 ± 1.93 | 3682.9 |
| 10_10 | adaptive_n | 200 | +1.90 ± 1.48 | 1.50 | +2.05 ± 1.50 | 3639.5 |
| 10_20 | random | 200 | +5.19 ± 2.22 | 2.67 | +5.37 ± 2.14 | 5862.7 |
| 10_20 | worst | 200 | +5.63 ± 2.08 | 2.71 | +5.81 ± 2.03 | 5887.1 |
| 10_20 | related | 200 | +5.72 ± 3.12 | 2.82 | +5.90 ± 3.05 | 5892.7 |
| 10_20 | adaptive | 200 | +4.85 ± 2.28 | 2.51 | +5.03 ± 2.20 | 5844.0 |
| 10_20 | b0 | 200 | +5.31 ± 2.08 | 2.60 | +5.49 ± 1.98 | 5873.3 |
| 10_20 | v1 | 200 | +5.11 ± 2.05 | 2.58 | +5.29 ± 1.98 | 5860.8 |
| 10_20 | v2 | 200 | +9.05 ± 2.70 | 3.78 | +9.23 ± 2.62 | 6081.0 |
| 10_20 | adaptive_n | 200 | +4.46 ± 1.93 | 2.47 | +4.64 ± 1.85 | 5825.5 |
| 20_50 | random | 200 | +4.51 ± 1.54 | 2.24 | +5.70 ± 1.48 | 10153.1 |
| 20_50 | worst | 200 | +5.14 ± 1.58 | 2.21 | +6.34 ± 1.52 | 10213.5 |
| 20_50 | related | 200 | +3.22 ± 1.46 | 1.77 | +4.39 ± 1.56 | 10025.9 |
| 20_50 | adaptive | 200 | +3.10 ± 1.51 | 1.44 | +4.27 ± 1.60 | 10017.3 |
| 20_50 | b0 | 200 | +4.80 ± 1.81 | 2.60 | +6.00 ± 1.86 | 10181.3 |
| 20_50 | v1 | 200 | +4.85 ± 1.81 | 2.03 | +6.04 ± 1.88 | 10185.2 |
| 20_50 | v2 | 200 | +12.01 ± 2.99 | 3.96 | +13.29 ± 3.09 | 10875.3 |
| 20_50 | adaptive_n | 200 | +3.29 ± 1.58 | 1.87 | +4.46 ± 1.50 | 10033.8 |

## Main comparison, wall time

| cell | operator | runs | gap to CEA % (± sd over instances) | sd over repeats | gap to best known % | objective (mean) | it/s |
|---|---|---|---|---|---|---|---|
| 10_10 | random | 200 | +0.54 ± 1.01 | 0.63 | +0.69 ± 1.00 | 3595.7 | 706965 |
| 10_10 | worst | 200 | +0.98 ± 1.30 | 0.77 | +1.12 ± 1.36 | 3612.4 | 644063 |
| 10_10 | related | 200 | +2.30 ± 2.21 | 1.64 | +2.45 ± 2.21 | 3653.6 | 735701 |
| 10_10 | adaptive | 200 | +0.52 ± 1.01 | 0.47 | +0.67 ± 1.02 | 3594.3 | 651061 |
| 10_10 | b0 | 200 | +1.16 ± 1.44 | 0.90 | +1.31 ± 1.51 | 3616.6 | 152604 |
| 10_10 | v1 | 200 | +1.65 ± 1.45 | 1.34 | +1.80 ± 1.51 | 3632.5 | 1424 |
| 10_10 | v2 | 200 | +2.50 ± 1.69 | 2.05 | +2.65 ± 1.73 | 3659.2 | 9918 |
| 10_10 | adaptive_n | 200 | +1.34 ± 1.29 | 1.29 | +1.49 ± 1.27 | 3620.4 | 5442 |
| 10_20 | random | 200 | +1.42 ± 1.82 | 1.25 | +1.59 ± 1.83 | 5650.8 | 331420 |
| 10_20 | worst | 200 | +1.87 ± 1.92 | 1.60 | +2.05 ± 1.95 | 5674.5 | 292632 |
| 10_20 | related | 200 | +4.25 ± 3.54 | 2.56 | +4.43 ± 3.46 | 5809.3 | 362083 |
| 10_20 | adaptive | 200 | +1.20 ± 1.56 | 1.33 | +1.37 ± 1.57 | 5637.1 | 302893 |
| 10_20 | b0 | 200 | +2.48 ± 1.95 | 1.94 | +2.65 ± 1.92 | 5710.7 | 32906 |
| 10_20 | v1 | 200 | +3.68 ± 1.88 | 2.33 | +3.86 ± 1.78 | 5782.5 | 874 |
| 10_20 | v2 | 200 | +6.84 ± 2.64 | 3.22 | +7.02 ± 2.55 | 5959.2 | 5754 |
| 10_20 | adaptive_n | 200 | +2.56 ± 1.91 | 1.95 | +2.73 ± 1.91 | 5713.3 | 3378 |
| 20_50 | random | 200 | -0.14 ± 0.85 | 0.53 | +1.00 ± 0.76 | 9701.7 | 58037 |
| 20_50 | worst | 200 | +0.19 ± 1.18 | 0.62 | +1.33 ± 1.06 | 9733.9 | 53043 |
| 20_50 | related | 200 | -0.39 ± 1.30 | 0.33 | +0.74 ± 0.99 | 9676.6 | 62531 |
| 20_50 | adaptive | 200 | -0.63 ± 0.76 | 0.29 | +0.50 ± 0.51 | 9653.4 | 54035 |
| 20_50 | b0 | 200 | +0.91 ± 1.14 | 0.98 | +2.06 ± 1.24 | 9804.4 | 7614 |
| 20_50 | v1 | 200 | +2.98 ± 1.36 | 1.82 | +4.15 ± 1.23 | 10004.0 | 338 |
| 20_50 | v2 | 200 | +8.93 ± 2.31 | 3.39 | +10.17 ± 2.51 | 10578.7 | 2033 |
| 20_50 | adaptive_n | 200 | +0.84 ± 0.85 | 0.89 | +1.99 ± 1.08 | 9797.9 | 1221 |

## Destroy size sweep

| cell | k_frac | k | operator | runs | gap to CEA % (± sd over instances) | sd over repeats | gap to best known % | objective (mean) |
|---|---|---|---|---|---|---|---|---|
| 10_10 | 0.05 | 3 | random | 200 | +1.86 ± 1.42 | 1.62 | +2.01 ± 1.45 | 3638.4 |
| 10_10 | 0.05 | 3 | adaptive | 200 | +1.93 ± 1.52 | 1.60 | +2.08 ± 1.56 | 3640.9 |
| 10_10 | 0.05 | 3 | v1 | 200 | +1.90 ± 1.56 | 1.52 | +2.05 ± 1.62 | 3640.4 |
| 10_20 | 0.05 | 3 | random | 200 | +5.19 ± 2.22 | 2.67 | +5.37 ± 2.14 | 5862.7 |
| 10_20 | 0.05 | 3 | adaptive | 200 | +4.85 ± 2.28 | 2.51 | +5.03 ± 2.20 | 5844.0 |
| 10_20 | 0.05 | 3 | v1 | 200 | +5.11 ± 2.05 | 2.58 | +5.29 ± 1.98 | 5860.8 |
| 20_50 | 0.05 | 3 | random | 200 | +8.61 ± 1.97 | 3.06 | +9.85 ± 2.27 | 10549.5 |
| 20_50 | 0.05 | 3 | adaptive | 200 | +6.78 ± 2.03 | 2.61 | +8.00 ± 2.12 | 10373.2 |
| 20_50 | 0.05 | 3 | v1 | 200 | +8.27 ± 2.12 | 2.87 | +9.50 ± 2.29 | 10516.8 |
| 20_50 | 0.1 | 5 | random | 200 | +6.68 ± 1.60 | 2.81 | +7.89 ± 1.68 | 10361.7 |
| 20_50 | 0.1 | 5 | adaptive | 200 | +4.56 ± 1.81 | 2.21 | +5.75 ± 1.91 | 10158.3 |
| 20_50 | 0.1 | 5 | v1 | 200 | +6.24 ± 1.88 | 2.59 | +7.45 ± 1.95 | 10318.9 |
| 20_50 | 0.15 | 8 | random | 200 | +4.51 ± 1.54 | 2.24 | +5.70 ± 1.48 | 10153.1 |
| 20_50 | 0.15 | 8 | adaptive | 200 | +3.10 ± 1.51 | 1.44 | +4.27 ± 1.60 | 10017.3 |
| 20_50 | 0.15 | 8 | v1 | 200 | +4.85 ± 1.81 | 2.03 | +6.04 ± 1.88 | 10185.2 |
| 10_20 | 0.25 | 5 | random | 200 | +3.81 ± 2.00 | 2.28 | +3.99 ± 1.85 | 5790.9 |
| 10_20 | 0.25 | 5 | adaptive | 200 | +2.77 ± 1.73 | 1.91 | +2.95 ± 1.60 | 5731.2 |
| 10_20 | 0.25 | 5 | v1 | 200 | +3.56 ± 1.49 | 2.38 | +3.73 ± 1.38 | 5776.4 |
| 20_50 | 0.25 | 13 | random | 200 | +3.17 ± 1.37 | 1.83 | +4.35 ± 1.46 | 10022.8 |
| 20_50 | 0.25 | 13 | adaptive | 200 | +2.14 ± 1.26 | 1.40 | +3.30 ± 1.29 | 9922.9 |
| 20_50 | 0.25 | 13 | v1 | 200 | +2.98 ± 1.38 | 1.72 | +4.16 ± 1.47 | 10004.2 |
| 10_10 | 0.35 | 4 | random | 200 | +1.39 ± 1.25 | 1.41 | +1.54 ± 1.29 | 3622.4 |
| 10_10 | 0.35 | 4 | adaptive | 200 | +1.34 ± 1.40 | 1.25 | +1.49 ± 1.45 | 3620.5 |
| 10_10 | 0.35 | 4 | v1 | 200 | +1.62 ± 1.44 | 1.25 | +1.77 ± 1.50 | 3630.6 |
| 10_20 | 0.35 | 7 | random | 200 | +2.14 ± 1.63 | 1.68 | +2.31 ± 1.50 | 5701.0 |
| 10_20 | 0.35 | 7 | adaptive | 200 | +1.76 ± 1.78 | 1.44 | +1.93 ± 1.63 | 5674.2 |
| 10_20 | 0.35 | 7 | v1 | 200 | +2.37 ± 1.56 | 1.71 | +2.54 ± 1.47 | 5711.0 |
| 20_50 | 0.35 | 18 | random | 200 | +2.49 ± 1.25 | 1.47 | +3.66 ± 1.34 | 9955.7 |
| 20_50 | 0.35 | 18 | adaptive | 200 | +1.63 ± 1.23 | 1.29 | +2.79 ± 1.15 | 9874.6 |
| 20_50 | 0.35 | 18 | v1 | 200 | +2.23 ± 1.19 | 1.50 | +3.39 ± 1.28 | 9931.3 |

k = max(3, round(fraction x orders)) is the same for several fractions of the sweep: 10_10: k = 3 for fractions 0.05, 0.1, 0.15, 0.25; 10_20: k = 3 for fractions 0.05, 0.1, 0.15. Each such k is run and drawn once, at its smallest fraction.

## Large classes

| cell | budget | operator | runs | gap to CEA % (± sd over instances) | sd over repeats | gap to best known % | objective (mean) | it/s |
|---|---|---|---|---|---|---|---|---|
| 30_100 | iters5000 | random | 50 | +1.46 ± 1.55 | 1.96 | +7.33 ± 1.83 | 17331.7 | -- |
| 30_100 | iters5000 | adaptive | 50 | +0.33 ± 1.74 | 1.59 | +6.12 ± 1.34 | 17152.2 | -- |
| 30_100 | iters5000 | v1 | 50 | +0.95 ± 1.50 | 1.89 | +6.80 ± 2.03 | 17241.0 | -- |
| 30_100 | iters5000 | adaptive_n | 50 | +0.14 ± 2.28 | 1.93 | +5.91 ± 1.25 | 17110.1 | -- |
| 50_200 | iters5000 | random | 50 | -8.61 ± 1.61 | 1.63 | +8.21 ± 1.47 | 28362.0 | -- |
| 50_200 | iters5000 | adaptive | 50 | -9.75 ± 2.06 | 1.46 | +6.85 ± 1.78 | 27999.5 | -- |
| 50_200 | iters5000 | v1 | 50 | -9.79 ± 1.63 | 1.21 | +6.80 ± 1.15 | 27985.6 | -- |
| 50_200 | iters5000 | adaptive_n | 50 | -9.45 ± 1.69 | 1.32 | +7.20 ± 1.30 | 28089.8 | -- |
| 100_500 | iters5000 | random | 50 | -13.99 ± 0.92 | 1.43 | +6.58 ± 0.99 | 59760.9 | -- |
| 100_500 | iters5000 | adaptive | 50 | -13.06 ± 0.73 | 1.25 | +7.74 ± 0.93 | 60406.4 | -- |
| 100_500 | iters5000 | v1 | 50 | -15.01 ± 1.09 | 1.15 | +5.32 ± 1.20 | 59053.7 | -- |
| 100_500 | iters5000 | adaptive_n | 50 | -12.83 ± 1.03 | 1.23 | +8.02 ± 0.88 | 60566.9 | -- |
| 30_100 | time60s | random | 50 | -3.48 ± 1.65 | 1.01 | +2.09 ± 0.89 | 16505.5 | 15724 |
| 30_100 | time60s | adaptive | 50 | -3.94 ± 1.47 | 0.96 | +1.61 ± 0.79 | 16430.0 | 15206 |
| 30_100 | time60s | v1 | 50 | -0.01 ± 1.61 | 1.46 | +5.78 ± 1.82 | 17085.5 | 152 |
| 30_100 | time60s | adaptive_n | 50 | -1.88 ± 1.68 | 1.05 | +3.79 ± 0.87 | 16772.9 | 517 |
| 50_200 | time60s | random | 50 | -13.39 ± 1.77 | 0.69 | +2.53 ± 0.68 | 26865.4 | 4160 |
| 50_200 | time60s | adaptive | 50 | -14.55 ± 1.96 | 0.77 | +1.16 ± 0.44 | 26506.6 | 4178 |
| 50_200 | time60s | v1 | 50 | -8.85 ± 1.43 | 1.52 | +7.93 ± 1.46 | 28284.1 | 60 |
| 50_200 | time60s | adaptive_n | 50 | -10.56 ± 1.51 | 1.57 | +5.90 ± 1.33 | 27751.1 | 198 |
| 100_500 | time60s | random | 50 | -17.54 ± 0.76 | 1.24 | +2.19 ± 1.04 | 57296.2 | 467 |
| 100_500 | time60s | adaptive | 50 | -16.55 ± 1.10 | 1.64 | +3.40 ± 0.87 | 57974.8 | 508 |
| 100_500 | time60s | v1 | 50 | -12.57 ± 0.67 | 1.53 | +8.34 ± 1.18 | 60746.1 | 14 |
| 100_500 | time60s | adaptive_n | 50 | -12.00 ± 1.19 | 1.35 | +9.05 ± 1.47 | 61144.4 | 52 |

## Price and empty-distance objectives

| objective | cell | operator | runs | gap to CEA % (± sd over instances) | sd over repeats | gap to best known % | objective (mean) |
|---|---|---|---|---|---|---|---|
| empty | 10_10 | random | 200 | +130.72 ± 215.09 | 123.15 | +135.92 ± 212.25 | 90.3 |
| empty | 10_10 | adaptive | 200 | +159.59 ± 256.24 | 132.45 | +165.49 ± 253.42 | 96.6 |
| empty | 10_10 | v1 | 200 | +137.46 ± 204.81 | 125.55 | +142.59 ± 201.75 | 91.9 |
| empty | 10_10 | adaptive_n | 200 | +140.06 ± 234.01 | 121.28 | +145.37 ± 231.35 | 92.9 |
| empty | 10_20 | random | 200 | +12.90 ± 78.27 | 43.48 | +138.44 ± 83.92 | 100.1 |
| empty | 10_20 | adaptive | 200 | +1.12 ± 52.60 | 45.20 | +117.53 ± 58.36 | 92.8 |
| empty | 10_20 | v1 | 200 | +15.23 ± 77.50 | 53.98 | +144.93 ± 96.93 | 102.6 |
| empty | 10_20 | adaptive_n | 200 | +0.21 ± 54.60 | 45.62 | +115.72 ± 72.39 | 92.1 |
| empty | 20_50 | random | 200 | +5.18 ± 27.70 | 50.81 | +882.31 ± 981.02 | 132.7 |
| empty | 20_50 | adaptive | 200 | -22.35 ± 28.44 | 40.07 | +747.18 ± 1078.22 | 97.0 |
| empty | 20_50 | v1 | 200 | +14.06 ± 32.75 | 54.70 | +943.73 ± 989.02 | 142.5 |
| empty | 20_50 | adaptive_n | 200 | -15.62 ± 29.20 | 44.79 | +775.67 ± 1062.83 | 103.0 |
| price | 10_10 | random | 200 | +2.58 ± 3.71 | 2.75 | +2.58 ± 3.71 | 4931.6 |
| price | 10_10 | adaptive | 200 | +1.97 ± 3.04 | 2.45 | +1.97 ± 3.04 | 4903.6 |
| price | 10_10 | v1 | 200 | +3.31 ± 4.75 | 3.21 | +3.31 ± 4.75 | 4965.8 |
| price | 10_10 | adaptive_n | 200 | +2.39 ± 3.30 | 2.63 | +2.39 ± 3.30 | 4924.1 |
| price | 10_20 | random | 200 | +5.36 ± 4.49 | 4.21 | +5.52 ± 4.38 | 7357.4 |
| price | 10_20 | adaptive | 200 | +4.59 ± 4.35 | 3.54 | +4.76 ± 4.21 | 7306.9 |
| price | 10_20 | v1 | 200 | +5.64 ± 5.40 | 4.21 | +5.81 ± 5.28 | 7390.5 |
| price | 10_20 | adaptive_n | 200 | +4.69 ± 4.55 | 3.72 | +4.85 ± 4.45 | 7316.1 |
| price | 20_50 | random | 200 | +4.66 ± 3.80 | 4.19 | +6.99 ± 3.57 | 11894.4 |
| price | 20_50 | adaptive | 200 | +1.94 ± 3.56 | 2.37 | +4.19 ± 2.56 | 11560.5 |
| price | 20_50 | v1 | 200 | +6.05 ± 4.41 | 5.43 | +8.40 ± 4.07 | 12057.4 |
| price | 20_50 | adaptive_n | 200 | +1.92 ± 3.13 | 2.80 | +4.17 ± 2.17 | 11561.8 |

## Paired tests

Wilcoxon signed-rank test over instances on per-instance mean gaps (column metric in stats.csv); median difference = operator minus baseline in percentage points; Holm's correction within each study and budget.

| study | budget | cell | operator | baseline | n | n non-zero | median diff | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| main | iters5000 | 10_10 | b0 | random | 20 | 9 | +0.000 | 0.051 | 0.78 |
| main | iters5000 | 10_20 | b0 | random | 20 | 19 | -0.010 | 0.57 | 1 |
| main | iters5000 | 20_50 | b0 | random | 20 | 20 | +0.358 | 0.13 | 1 |
| main | iters5000 | pooled | b0 | random | 60 | 48 | +0.000 | 0.023 | 0.4 |
| main | iters5000 | 10_10 | b0 | adaptive | 20 | 15 | +0.001 | 0.21 | 1 |
| main | iters5000 | 10_20 | b0 | adaptive | 20 | 20 | +0.366 | 0.048 | 0.78 |
| main | iters5000 | 20_50 | b0 | adaptive | 20 | 20 | +1.673 | 1.9e-06 | 5.5e-05 |
| main | iters5000 | pooled | b0 | adaptive | 60 | 55 | +0.495 | 9.5e-07 | 2.9e-05 |
| main | iters5000 | 10_10 | v1 | random | 20 | 9 | +0.000 | 0.68 | 1 |
| main | iters5000 | 10_20 | v1 | random | 20 | 20 | -0.274 | 0.47 | 1 |
| main | iters5000 | 20_50 | v1 | random | 20 | 20 | +0.330 | 0.097 | 1 |
| main | iters5000 | pooled | v1 | random | 60 | 49 | +0.000 | 0.47 | 1 |
| main | iters5000 | 10_10 | v1 | adaptive | 20 | 10 | +0.000 | 0.72 | 1 |
| main | iters5000 | 10_20 | v1 | adaptive | 20 | 20 | +0.019 | 0.47 | 1 |
| main | iters5000 | 20_50 | v1 | adaptive | 20 | 20 | +1.645 | 1.9e-06 | 5.5e-05 |
| main | iters5000 | pooled | v1 | adaptive | 60 | 50 | +0.245 | 8.4e-05 | 0.0018 |
| main | iters5000 | 10_10 | v2 | random | 20 | 15 | +0.676 | 0.00065 | 0.012 |
| main | iters5000 | 10_20 | v2 | random | 20 | 20 | +3.763 | 1.9e-06 | 5.5e-05 |
| main | iters5000 | 20_50 | v2 | random | 20 | 20 | +7.575 | 1.9e-06 | 5.5e-05 |
| main | iters5000 | pooled | v2 | random | 60 | 55 | +3.772 | 1.1e-10 | 3.4e-09 |
| main | iters5000 | 10_10 | v2 | adaptive | 20 | 18 | +0.652 | 0.00054 | 0.011 |
| main | iters5000 | 10_20 | v2 | adaptive | 20 | 20 | +3.611 | 1.9e-06 | 5.5e-05 |
| main | iters5000 | 20_50 | v2 | adaptive | 20 | 20 | +9.008 | 1.9e-06 | 5.5e-05 |
| main | iters5000 | pooled | v2 | adaptive | 60 | 58 | +3.761 | 4.8e-11 | 1.5e-09 |
| main | iters5000 | 10_10 | adaptive_n | random | 20 | 11 | +0.000 | 0.42 | 1 |
| main | iters5000 | 10_20 | adaptive_n | random | 20 | 20 | -0.736 | 0.002 | 0.036 |
| main | iters5000 | 20_50 | adaptive_n | random | 20 | 20 | -1.135 | 3.8e-06 | 8.4e-05 |
| main | iters5000 | pooled | adaptive_n | random | 60 | 51 | -0.476 | 2e-06 | 5.5e-05 |
| main | iters5000 | 10_10 | adaptive_n | adaptive | 20 | 9 | +0.000 | 0.31 | 1 |
| main | iters5000 | 10_20 | adaptive_n | adaptive | 20 | 20 | -0.159 | 0.09 | 1 |
| main | iters5000 | 20_50 | adaptive_n | adaptive | 20 | 20 | +0.183 | 0.13 | 1 |
| main | iters5000 | pooled | adaptive_n | adaptive | 60 | 49 | +0.000 | 0.71 | 1 |
| large | iters5000 | 30_100 | v1 | random | 10 | 10 | -0.455 | 0.11 | 0.94 |
| large | iters5000 | 50_200 | v1 | random | 10 | 10 | -1.231 | 0.002 | 0.029 |
| large | iters5000 | 100_500 | v1 | random | 10 | 10 | -1.340 | 0.037 | 0.41 |
| large | iters5000 | pooled | v1 | random | 30 | 30 | -1.085 | 2.4e-05 | 0.00038 |
| large | iters5000 | 30_100 | v1 | adaptive | 10 | 10 | +0.391 | 0.43 | 1 |
| large | iters5000 | 50_200 | v1 | adaptive | 10 | 10 | -0.234 | 0.7 | 1 |
| large | iters5000 | 100_500 | v1 | adaptive | 10 | 10 | -2.242 | 0.002 | 0.029 |
| large | iters5000 | pooled | v1 | adaptive | 30 | 30 | -0.570 | 0.1 | 0.94 |
| large | iters5000 | 30_100 | adaptive_n | random | 10 | 10 | -0.873 | 0.0098 | 0.12 |
| large | iters5000 | 50_200 | adaptive_n | random | 10 | 10 | -0.960 | 0.064 | 0.64 |
| large | iters5000 | 100_500 | adaptive_n | random | 10 | 10 | +1.011 | 0.002 | 0.029 |
| large | iters5000 | pooled | adaptive_n | random | 30 | 30 | -0.255 | 0.35 | 1 |
| large | iters5000 | 30_100 | adaptive_n | adaptive | 10 | 10 | +0.030 | 0.92 | 1 |
| large | iters5000 | 50_200 | adaptive_n | adaptive | 10 | 10 | +0.188 | 0.19 | 1 |
| large | iters5000 | 100_500 | adaptive_n | adaptive | 10 | 10 | +0.107 | 0.38 | 1 |
| large | iters5000 | pooled | adaptive_n | adaptive | 30 | 30 | +0.109 | 0.24 | 1 |
| objective_empty | objective_empty | 10_10 | v1 | random | 20 | 18 | +9.670 | 0.12 | 1 |
| objective_empty | objective_empty | 10_20 | v1 | random | 20 | 20 | -2.646 | 0.9 | 1 |
| objective_empty | objective_empty | 20_50 | v1 | random | 20 | 20 | +3.484 | 0.11 | 1 |
| objective_empty | objective_empty | pooled | v1 | random | 60 | 58 | +2.347 | 0.06 | 0.79 |
| objective_empty | objective_empty | 10_10 | v1 | adaptive | 20 | 20 | -3.639 | 0.43 | 1 |
| objective_empty | objective_empty | 10_20 | v1 | adaptive | 20 | 20 | +6.914 | 0.07 | 0.79 |
| objective_empty | objective_empty | 20_50 | v1 | adaptive | 20 | 20 | +37.448 | 8.2e-05 | 0.0013 |
| objective_empty | objective_empty | pooled | v1 | adaptive | 60 | 60 | +13.091 | 0.0074 | 0.1 |
| objective_empty | objective_empty | 10_10 | adaptive_n | random | 20 | 17 | +2.562 | 0.14 | 1 |
| objective_empty | objective_empty | 10_20 | adaptive_n | random | 20 | 20 | -9.112 | 0.064 | 0.79 |
| objective_empty | objective_empty | 20_50 | adaptive_n | random | 20 | 20 | -23.694 | 0.0014 | 0.021 |
| objective_empty | objective_empty | pooled | adaptive_n | random | 60 | 57 | -3.857 | 0.066 | 0.79 |
| objective_empty | objective_empty | 10_10 | adaptive_n | adaptive | 20 | 19 | -2.700 | 0.17 | 1 |
| objective_empty | objective_empty | 10_20 | adaptive_n | adaptive | 20 | 20 | +1.608 | 0.93 | 1 |
| objective_empty | objective_empty | 20_50 | adaptive_n | adaptive | 20 | 20 | +6.165 | 0.11 | 1 |
| objective_empty | objective_empty | pooled | adaptive_n | adaptive | 60 | 59 | +0.019 | 0.89 | 1 |
| objective_price | objective_price | 10_10 | v1 | random | 20 | 10 | +0.000 | 0.028 | 0.23 |
| objective_price | objective_price | 10_20 | v1 | random | 20 | 20 | +0.016 | 0.99 | 1 |
| objective_price | objective_price | 20_50 | v1 | random | 20 | 20 | +1.386 | 0.012 | 0.12 |
| objective_price | objective_price | pooled | v1 | random | 60 | 50 | +0.147 | 0.01 | 0.11 |
| objective_price | objective_price | 10_10 | v1 | adaptive | 20 | 12 | +0.000 | 0.019 | 0.17 |
| objective_price | objective_price | 10_20 | v1 | adaptive | 20 | 20 | +0.289 | 0.12 | 0.62 |
| objective_price | objective_price | 20_50 | v1 | adaptive | 20 | 20 | +3.714 | 5.7e-06 | 8.6e-05 |
| objective_price | objective_price | pooled | v1 | adaptive | 60 | 52 | +1.261 | 5.5e-07 | 8.8e-06 |
| objective_price | objective_price | 10_10 | adaptive_n | random | 20 | 8 | +0.000 | 1 | 1 |
| objective_price | objective_price | 10_20 | adaptive_n | random | 20 | 20 | -0.305 | 0.0094 | 0.11 |
| objective_price | objective_price | 20_50 | adaptive_n | random | 20 | 20 | -1.627 | 6.3e-05 | 0.00082 |
| objective_price | objective_price | pooled | adaptive_n | random | 60 | 48 | -0.305 | 1.9e-05 | 0.00027 |
| objective_price | objective_price | 10_10 | adaptive_n | adaptive | 20 | 11 | +0.000 | 0.05 | 0.35 |
| objective_price | objective_price | 10_20 | adaptive_n | adaptive | 20 | 19 | +0.054 | 0.33 | 1 |
| objective_price | objective_price | 20_50 | adaptive_n | adaptive | 20 | 20 | +0.097 | 0.9 | 1 |
| objective_price | objective_price | pooled | adaptive_n | adaptive | 60 | 50 | +0.025 | 0.08 | 0.48 |
| main | time60s | 10_10 | b0 | random | 20 | 8 | +0.000 | 0.012 | 0.023 |
| main | time60s | 10_20 | b0 | random | 20 | 20 | +1.049 | 5.7e-06 | 7.4e-05 |
| main | time60s | 20_50 | b0 | random | 20 | 20 | +0.991 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | b0 | random | 60 | 48 | +0.764 | 2.2e-09 | 5.6e-08 |
| main | time60s | 10_10 | b0 | adaptive | 20 | 10 | +0.000 | 0.017 | 0.023 |
| main | time60s | 10_20 | b0 | adaptive | 20 | 20 | +0.989 | 5.7e-06 | 7.4e-05 |
| main | time60s | 20_50 | b0 | adaptive | 20 | 20 | +1.479 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | b0 | adaptive | 60 | 50 | +0.876 | 1.8e-09 | 4.6e-08 |
| main | time60s | 10_10 | v1 | random | 20 | 14 | +0.397 | 0.0023 | 0.0073 |
| main | time60s | 10_20 | v1 | random | 20 | 20 | +2.110 | 3.8e-06 | 5.7e-05 |
| main | time60s | 20_50 | v1 | random | 20 | 20 | +2.993 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | v1 | random | 60 | 54 | +2.208 | 2.7e-10 | 7.5e-09 |
| main | time60s | 10_10 | v1 | adaptive | 20 | 15 | +0.420 | 0.0018 | 0.0073 |
| main | time60s | 10_20 | v1 | adaptive | 20 | 20 | +2.177 | 5.7e-06 | 7.4e-05 |
| main | time60s | 20_50 | v1 | adaptive | 20 | 20 | +3.475 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | v1 | adaptive | 60 | 55 | +2.283 | 2.3e-10 | 6.6e-09 |
| main | time60s | 10_10 | v2 | random | 20 | 17 | +1.204 | 0.00029 | 0.0021 |
| main | time60s | 10_20 | v2 | random | 20 | 20 | +4.846 | 1.9e-06 | 4.6e-05 |
| main | time60s | 20_50 | v2 | random | 20 | 20 | +9.268 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | v2 | random | 60 | 57 | +5.036 | 5.1e-11 | 1.6e-09 |
| main | time60s | 10_10 | v2 | adaptive | 20 | 18 | +1.245 | 0.0002 | 0.0016 |
| main | time60s | 10_20 | v2 | adaptive | 20 | 20 | +5.290 | 1.9e-06 | 4.6e-05 |
| main | time60s | 20_50 | v2 | adaptive | 20 | 20 | +9.669 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | v2 | adaptive | 60 | 58 | +5.205 | 3.5e-11 | 1.1e-09 |
| main | time60s | 10_10 | adaptive_n | random | 20 | 15 | +0.425 | 0.0015 | 0.0073 |
| main | time60s | 10_20 | adaptive_n | random | 20 | 20 | +1.209 | 6.3e-05 | 0.00057 |
| main | time60s | 20_50 | adaptive_n | random | 20 | 20 | +0.830 | 9.5e-06 | 9.5e-05 |
| main | time60s | pooled | adaptive_n | random | 60 | 55 | +0.769 | 1.6e-09 | 4.4e-08 |
| main | time60s | 10_10 | adaptive_n | adaptive | 20 | 15 | +0.315 | 0.0012 | 0.0072 |
| main | time60s | 10_20 | adaptive_n | adaptive | 20 | 20 | +1.292 | 3.8e-06 | 5.7e-05 |
| main | time60s | 20_50 | adaptive_n | adaptive | 20 | 20 | +1.394 | 1.9e-06 | 4.6e-05 |
| main | time60s | pooled | adaptive_n | adaptive | 60 | 55 | +0.972 | 1.5e-10 | 4.6e-09 |
| large | time60s | 30_100 | v1 | random | 10 | 10 | +2.857 | 0.002 | 0.023 |
| large | time60s | 50_200 | v1 | random | 10 | 10 | +4.642 | 0.002 | 0.023 |
| large | time60s | 100_500 | v1 | random | 10 | 10 | +5.057 | 0.002 | 0.023 |
| large | time60s | pooled | v1 | random | 30 | 30 | +4.655 | 1.9e-09 | 3e-08 |
| large | time60s | 30_100 | v1 | adaptive | 10 | 10 | +3.605 | 0.002 | 0.023 |
| large | time60s | 50_200 | v1 | adaptive | 10 | 10 | +5.730 | 0.002 | 0.023 |
| large | time60s | 100_500 | v1 | adaptive | 10 | 10 | +4.247 | 0.002 | 0.023 |
| large | time60s | pooled | v1 | adaptive | 30 | 30 | +4.452 | 1.9e-09 | 3e-08 |
| large | time60s | 30_100 | adaptive_n | random | 10 | 10 | +1.475 | 0.002 | 0.023 |
| large | time60s | 50_200 | adaptive_n | random | 10 | 10 | +3.009 | 0.002 | 0.023 |
| large | time60s | 100_500 | adaptive_n | random | 10 | 10 | +5.600 | 0.002 | 0.023 |
| large | time60s | pooled | adaptive_n | random | 30 | 30 | +3.009 | 1.9e-09 | 3e-08 |
| large | time60s | 30_100 | adaptive_n | adaptive | 10 | 10 | +1.999 | 0.002 | 0.023 |
| large | time60s | 50_200 | adaptive_n | adaptive | 10 | 10 | +3.634 | 0.002 | 0.023 |
| large | time60s | 100_500 | adaptive_n | adaptive | 10 | 10 | +4.299 | 0.002 | 0.023 |
| large | time60s | pooled | adaptive_n | adaptive | 30 | 30 | +3.465 | 1.9e-09 | 3e-08 |

## Selection (validation instances only)

Best V2 run: v2-c; best model: v1 (tau 1); 40 validation instances, 5000 iterations, 3 repeats.

| model | tau | mean objective | mean objective by tau |
|---|---|---|---|
| b0 | 1 | 7945.98 | 0.1:8236.1 0.25:8083.2 0.5:8021.6 1.0:7946.0 |
| v1 | 1 | 7933.42 | 0.1:8164.7 0.25:8000.9 0.5:7937.9 1.0:7933.4 |
| v2-a | 1 | 8565.54 | 0.1:8904.3 0.25:8827.0 0.5:8718.1 1.0:8565.5 |
| v2-b | 1 | 8534.89 | 0.1:9011.2 0.25:8896.6 0.5:8759.5 1.0:8534.9 |
| v2-c | 1 | 8389.90 | 0.1:8919.6 0.25:8809.7 0.5:8601.9 1.0:8389.9 |
| v2-d | 1 | 8610.66 | 0.1:8920.9 0.25:8834.5 0.5:8736.1 1.0:8610.7 |
| v2-e | 1 | 8607.87 | 0.1:8901.8 0.25:8848.7 0.5:8739.1 1.0:8607.9 |

## Training

### B0

Validation metrics at the kept epoch: Spearman and p@k against the model's own training target (mean or top), p@k best try = share of the predicted top k inside the state's best try (the same for every target; models are ranked by it).

| group | model | parameters | Spearman | p@k (own target) | p@k best try | best epoch | train s |
|---|---|---|---|---|---|---|---|
| grid | w128_d2_mse_mean_s0 | 17793 | 0.275 | 0.252 | 0.196 | 32 | 804 |
| grid | w128_d2_mse_top_s0 | 17793 | 0.074 | 0.247 | 0.226 | 10 | 382 |
| grid | w128_d2_rank_mean_s0 | 17793 | 0.176 | 0.248 | 0.234 | 11 | 425 |
| grid | w128_d2_rank_top_s0 | 17793 | 0.059 | 0.233 | 0.210 | 23 | 683 |
| grid | w128_d3_mse_mean_s0 | 34305 | 0.277 | 0.248 | 0.189 | 10 | 439 |
| grid | w128_d3_mse_top_s0 | 34305 | 0.079 | 0.248 | 0.230 | 10 | 437 |
| grid | w128_d3_rank_mean_s0 | 34305 | 0.181 | 0.246 | 0.227 | 11 | 492 |
| grid | w128_d3_rank_top_s0 | 34305 | 0.056 | 0.234 | 0.212 | 11 | 495 |
| grid | w256_d2_mse_mean_s0 | 68353 | 0.276 | 0.246 | 0.189 | 10 | 381 |
| grid | w256_d2_mse_top_s0 | 68353 | 0.074 | 0.245 | 0.230 | 23 | 623 |
| grid | w256_d2_rank_mean_s0 | 68353 | 0.196 | 0.254 | 0.231 | 24 | 699 |
| grid | w256_d2_rank_top_s0 | 68353 | 0.055 | 0.233 | 0.210 | 30 | 722 |
| grid | w256_d3_mse_mean_s0 | 134145 | 0.276 | 0.248 | 0.193 | 10 | 417 |
| grid | w256_d3_mse_top_s0 | 134145 | 0.075 | 0.245 | 0.225 | 6 | 341 |
| grid | w256_d3_rank_mean_s0 | 134145 | 0.179 | 0.251 | 0.231 | 24 | 775 |
| grid | w256_d3_rank_top_s0 | 134145 | 0.064 | 0.237 | 0.214 | 40 | 907 |
| grid | w64_d2_mse_mean_s0 | 4801 | 0.275 | 0.249 | 0.192 | 24 | 430 |
| grid | w64_d2_mse_top_s0 | 4801 | 0.075 | 0.247 | 0.229 | 10 | 388 |
| grid | w64_d2_rank_mean_s0 | 4801 | 0.168 | 0.244 | 0.229 | 11 | 430 |
| grid | w64_d2_rank_top_s0 | 4801 | 0.057 | 0.234 | 0.211 | 6 | 338 |
| grid | w64_d3_mse_mean_s0 | 8961 | 0.277 | 0.250 | 0.202 | 24 | 747 |
| grid | w64_d3_mse_top_s0 | 8961 | 0.076 | 0.250 | 0.234 | 22 | 698 |
| grid | w64_d3_rank_mean_s0 | 8961 | 0.172 | 0.247 | 0.231 | 11 | 487 |
| grid | w64_d3_rank_top_s0 | 8961 | 0.056 | 0.235 | 0.213 | 11 | 510 |
| trainsize | w128_d2_rank_mean_n1000_s0 | 17793 | 0.179 | 0.247 | 0.222 | 9 | 119 |
| trainsize | w128_d2_rank_mean_n500_s0 | 17793 | 0.180 | 0.241 | 0.220 | 6 | 67 |
| seeds | w128_d2_rank_mean_s1 | 17793 | 0.171 | 0.246 | 0.227 | 14 | 366 |
| seeds | w128_d2_rank_mean_s2 | 17793 | 0.162 | 0.252 | 0.232 | 14 | 368 |
| seeds mean | w128_d2_rank_mean_s0 (3 seeds) | 17793 | 0.170 | 0.249 | 0.231 | 13.0 | 386 |
| seeds sd | w128_d2_rank_mean_s0 (3 seeds) | 0 | 0.007 | 0.003 | 0.004 | 1.7 | 34 |

### V1

Validation metrics at the kept epoch: Spearman and p@k against the model's own training target (mean or top), p@k best try = share of the predicted top k inside the state's best try (the same for every target; models are ranked by it).

| group | model | parameters | Spearman | p@k (own target) | p@k best try | best epoch | train s |
|---|---|---|---|---|---|---|---|
| grid | d128_a1_do0.1_wd1e-4_s0 | 398209 | 0.159 | 0.249 | 0.237 | 19 | 2337 |
| grid | d128_a2_do0.1_wd1e-4_s0 | 663169 | 0.182 | 0.251 | 0.235 | 72 | 7670 |
| grid | d64_a1_do0.1_wd1e-4_s0 | 100801 | 0.180 | 0.250 | 0.233 | 30 | 3418 |
| grid | d64_a2_do0.1_wd1e-4_s0 | 167745 | 0.173 | 0.254 | 0.244 | 57 | 6820 |
| reg | d64_a2_do0.1_wd0_s0 | 167745 | 0.161 | 0.240 | 0.220 | 73 | 8125 |
| reg | d64_a2_do0_wd0_s0 | 167745 | 0.186 | 0.251 | 0.233 | 19 | 3221 |
| reg | d64_a2_do0_wd1e-4_s0 | 167745 | 0.187 | 0.250 | 0.234 | 43 | 5401 |
| seeds | d64_a2_do0.1_wd1e-4_s1 | 167745 | 0.209 | 0.264 | 0.241 | 52 | 5697 |
| seeds | d64_a2_do0.1_wd1e-4_s2 | 167745 | 0.196 | 0.253 | 0.229 | 31 | 3814 |
| seeds mean | d64_a2_do0.1_wd1e-4_s0 (3 seeds) | 167745 | 0.193 | 0.257 | 0.238 | 46.7 | 5444 |
| seeds sd | d64_a2_do0.1_wd1e-4_s0 (3 seeds) | 0 | 0.018 | 0.006 | 0.008 | 13.8 | 1519 |

### V2

| run | start | S | beta | parameters | best val reward | best episode | episodes | hours |
|---|---|---|---|---|---|---|---|---|
| v2-a | V1 | 4.0 | 0.01 | 167745 | 0.0279 | 2420.0 | 2738.0 | 3.00 |
| v2-b | scratch | 4.0 | 0.01 | 167745 | 0.0163 | 1250.0 | 2757.0 | 3.00 |
| v2-c | V1 | 1.0 | 0.01 | 167745 | 0.0251 | 2490.0 | 2860.0 | 3.00 |
| v2-d | V1 | 8.0 | 0.01 | 167745 | 0.0234 | 2530.0 | 2630.0 | 3.00 |
| v2-e | V1 | 4.0 | 0.0 | 167745 | 0.0242 | 2270.0 | 2764.0 | 3.00 |

## Label quality (stats_train.md)

Label statistics for `results/stage3/labels/train.jsonl`: 57000 states, 1710000 order rows, M = 1000.

| cell | states | try-improvement std (% of obj) | best − median try (% of obj) | best try (% of obj) | median try (% of obj) | share of tries with gain > 0 |
|---|---|---|---|---|---|---|
| 10_20 | 38000 | 0.709 | 1.651 | 1.661 | 0.010 | 0.118 |
| 20_50 | 19000 | 1.164 | 2.466 | 2.538 | 0.072 | 0.356 |

Share of states where the best try beats the median try by more than X % of the objective: 0.5%: 0.641, 1.0%: 0.548, 2.0%: 0.390, 5.0%: 0.093

| feature | Pearson r with z-score | Spearman ρ with z-score | Spearman (rank by this feature alone, per state) | precision@k (feature alone) |
|---|---|---|---|---|
| route_saving | -0.019 | -0.081 | -0.094 | 0.165 |
| start_to_pickup | -0.069 | -0.051 | -0.082 | 0.133 |
| pickup_to_delivery | -0.211 | -0.207 | -0.208 | 0.111 |
| load_factor | 0.032 | 0.019 | 0.018 | 0.155 |
| price_km | 0.000 | -0.001 | -0.003 | 0.164 |
| orders_on_route | -0.075 | -0.051 | -0.104 | 0.141 |
| nearest_other | -0.031 | -0.052 | -0.073 | 0.143 |
| relative_position | -0.024 | -0.026 | -0.033 | 0.144 |
| (random order) | – | – | 0.003 | 0.153 |


## Label quality (stats_val.md)

Label statistics for `results/stage3/labels/val.jsonl`: 5700 states, 171000 order rows, M = 1000.

| cell | states | try-improvement std (% of obj) | best − median try (% of obj) | best try (% of obj) | median try (% of obj) | share of tries with gain > 0 |
|---|---|---|---|---|---|---|
| 10_20 | 3800 | 0.703 | 1.706 | 1.719 | 0.012 | 0.119 |
| 20_50 | 1900 | 1.070 | 2.404 | 2.476 | 0.072 | 0.356 |

Share of states where the best try beats the median try by more than X % of the objective: 0.5%: 0.643, 1.0%: 0.551, 2.0%: 0.396, 5.0%: 0.096

| feature | Pearson r with z-score | Spearman ρ with z-score | Spearman (rank by this feature alone, per state) | precision@k (feature alone) |
|---|---|---|---|---|
| route_saving | -0.017 | -0.081 | -0.097 | 0.166 |
| start_to_pickup | -0.057 | -0.043 | -0.078 | 0.133 |
| pickup_to_delivery | -0.209 | -0.208 | -0.215 | 0.109 |
| load_factor | 0.040 | 0.021 | 0.017 | 0.150 |
| price_km | -0.000 | -0.001 | 0.003 | 0.164 |
| orders_on_route | -0.076 | -0.048 | -0.101 | 0.138 |
| nearest_other | -0.026 | -0.047 | -0.072 | 0.144 |
| relative_position | -0.021 | -0.026 | -0.033 | 0.148 |
| (random order) | – | – | 0.003 | 0.152 |


