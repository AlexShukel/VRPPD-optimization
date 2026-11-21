# Stage 3 report notes

Working notes for the final report. Numbers come from `summary.md`, `training_summary.csv` and the batch manifests. Sections are filled as batches finish.

## Hosts and code

- All batches so far ran on `pingwin` (i5-14600KF, 27 GB, RTX 4070 SUPER, Ubuntu 24.04 under WSL2).
- Commits recorded in the manifests: `data` 9a4227e, `b0` 235c4f1, `v1` and the three `ref-*` batches 9996050 (later batches also run on 9996050). The commits in between change the pipeline scripts, the reference thread count and provenance checks. The whole-branch review confirmed that data generation, labels and the search code are unchanged since 9a4227e.
- CEA references use 6 workers on 6 pinned CPUs (first attempt used 18 workers on 6 CPUs and was discarded).

## Deviations from the plan

- Model ranking uses the share of the predicted top k inside the best removal try (`p_at_k_best_try`), not precision at k against the model's own target, because the two targets are not comparable.
- CEA references for 20_50 take 3 h 20 min (runs are sequential), not 35 min.
- The host went down on 2026-11-19 at about 06:01 and came back at 07:49. V2 run (a) was at about 2.25 h of its 3 h budget and was resumed from its checkpoint. No other batch was affected.

## Training (section "Training results")

- B0: 24 configurations, best-try share 0.19 to 0.234 (chance 0.153). Selected: width 128, depth 2, ranking loss, mean target, 0.234. Three seeds: mean 0.231, sd 0.004. The better configurations differ by less than the seed spread.
- B0, MSE on the mean target: highest Spearman (0.275), lowest best-try share (about 0.19).
- B0 training-set size: 500 / 1 000 / 3 000 instances give 0.220 / 0.222 / 0.234 (one seed each).
- V1: selected d = 64, 2 attention blocks, dropout 0.1, weight decay 1e-4, 0.244. Three seeds: mean 0.238, sd 0.008. Dropout 0.1 without weight decay was the worst (0.220).
- V1 against B0 on the offline measure: 0.238 against 0.231, inside the seed spread. V1 has 167 745 weights against 17 793.
- Training time: B0 6 to 13 min per model (4 at a time), V1 39 to 135 min per model (3 at a time). Times are indicative because models shared the machine.

## Selection (validation, 40 instances, 5 000 it., 3 repeats)

- Every model is best at tau = 1.0, the largest value of the grid, and the mean objective falls steadily as tau grows. The best tau may lie beyond the grid; the grid was not extended.
- Mean objective at tau = 1: V1 7933.4, B0 7946.0, V2 runs 8389.9 (c) to 8610.7 (d). Best model: V1. Best V2 run: (c), one sample per state.
- V2 best validation reward: a 0.0279 (start V1, S = 4), b 0.0163 (from scratch), c 0.0251 (S = 1), d 0.0234 (S = 8), e 0.0242 (no entropy term). The run with the best reward (a) is not the best in the search (c). Run (b), from scratch, has the lowest reward but is second in the search. The spread between V2 runs (up to 221 km) is about half the gap to V1 (456 km or more).

## Main comparison (test, 20 instances per class, 10 repeats; gap to CEA %)

5 000 iterations (10_10 / 10_20 / 20_50):
- random 1.86 / 5.19 / 4.51; worst 1.97 / 5.63 / 5.14; related 2.51 / 5.72 / 3.22; adaptive 1.93 / 4.85 / 3.10
- B0 2.16 / 5.31 / 4.80; V1 1.90 / 5.11 / 4.85; V2 3.09 / 9.05 / 12.01; adaptive with neural rule 1.90 / 4.46 / 3.29
- Holm-adjusted tests: B0 and V1 do not differ from random in any class. Both are worse than adaptive on 20_50 (p = 5.5e-5). V2 is worse than random and adaptive everywhere. Adaptive with the neural rule beats random on 10_20 and 20_50 and does not differ from adaptive.

60 seconds (10_10 / 10_20 / 20_50), with iterations per second on 20_50:
- random 0.54 / 1.42 / -0.14 (58 037 it/s); adaptive 0.52 / 1.20 / -0.63 (54 035)
- B0 1.16 / 2.48 / 0.91 (7 614); V1 1.65 / 3.68 / 2.98 (338); V2 2.50 / 6.84 / 8.93 (2 033); adaptive with neural rule 1.34 / 2.56 / 0.84 (1 221)
- Every neural variant is significantly worse than random and than adaptive in every class under the time budget. V1 does about 170 times fewer iterations than random on 20_50.
- V2 runs about 6 to 7 times more iterations per second than V1 with the same architecture (10_10: 9 918 against 1 424). V1 is the slow one: weight decay shrank many of its attention weights to subnormal float values (1 878 in the selected model; none in models trained without weight decay, none in V2), and V1 with random weights ran at about 2 000 it/s on 20_50 in `machine.txt`, the same as V2. That subnormal arithmetic is the cause is inferred, not tested: zeroing these weights and timing again on the x86 host was not done.
- V2 improves the current solution in fewer iterations than V1 (7.4 % against 13.9 % on 20_50 at 5 000 it.). Why was not measured.

## Destroy size (5 000 it.)

- A larger share of removed orders helps every operator: on 20_50 random goes from 8.61 (k = 3) to 2.49 (k = 18), adaptive from 6.78 to 1.63, V1 from 8.27 to 2.23. Adaptive is best at every k on 10_20 and 20_50; V1 and random swap places by up to 0.44 points. No tests were run for the sweep. The best k is the largest one tried for every operator (edge of the grid, like tau).
- On 10_10 and 10_20 several shares give the same k because k = max(3, round(share x orders)).

## Large classes (10 instances, 5 repeats, V1 trained only on 10_20 and 20_50)

- CEA references are weak here in 60 s, so gaps are negative; the gap to the best known value is the safer column.
- 5 000 iterations, gap to CEA: 30_100 random 1.46, adaptive 0.33, V1 0.95, adaptive_n 0.14; 50_200 -8.61, -9.75, -9.79, -9.45; 100_500 -13.99, -13.06, -15.01, -12.83. V1 is better than random on 50_200 (Holm p = 0.029) and better than adaptive on 100_500 (0.029); pooled against random p = 0.00038.
- 60 s: V1 and adaptive_n are worse than random and adaptive in every class (Holm p = 0.023 each, the smallest possible with 10 instances). V1 runs 14 it/s on 100_500 against 467 for random.

## Price and empty distance (5 000 it., V1 trained on distance)

- Price: V1 does not differ from random and is worse than adaptive on 20_50 (p = 8.6e-5). Adaptive with the neural rule beats random on 20_50 and pooled and does not differ from adaptive.
- Empty distance: the references are small numbers (mean 53 to 131 km), so percentage gaps are very large and noisy. V1 does not differ from random and is worse than adaptive on 20_50 (p = 0.0013).

## RQ answers (draft)

- RQ1: no. At equal iterations the learned operators match random removal on the training classes; at equal time they are clearly worse because scoring is slow. The only positive result is V1 at equal iterations on the two largest classes.
- RQ2: the richer input (V1) does not improve the search over B0; REINFORCE (V2) makes it much worse within a 3 h budget; the number of removed orders matters more than the operator, and the best k is the largest tried for every operator.
- RQ3: at equal iterations V1 transfers to larger instances and is not worse on other objectives than random; at equal time it does not pay off anywhere.

## Checks

- Four cells of `summary.md` were recomputed from the raw run files and match: random 10_20 at 5 000 it. (5.19 ± 2.22), V1 20_50 at 60 s (2.98 ± 1.36), V1 100_500 at 5 000 it. (-15.01 ± 1.09), adaptive 50_200 at 60 s (-14.55 ± 1.96).
- All 24 batches ran on pingwin; the last one finished on 2026-11-20 at 18:53.

## Limitations to state

- The same solution is priced about 1e-4 km differently on macOS and Linux (coincident stops), so iteration-budget results from two machines differ in the last digits.
- V1 scores differ in the last bits between CPUs (matrix kernel chosen at run time).
- The V2 ablation (runs a to e) is compared on validation instances only.
- Wall-time batches of one test class must come from one machine.
- The watcher on the Mac pulls results every 10 minutes, also during timed batches.
