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
