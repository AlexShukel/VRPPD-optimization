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
