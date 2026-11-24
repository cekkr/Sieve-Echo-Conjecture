# Sieve Echo v11 — claim verdicts (orders-s0)

- **Run:** `orders-s0`, config `empirical-scripts/11.0/config/orders.json` (sha256 `482cf969ff`), seed 0.
- **Code:** root `74fd86010e` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `7046ea7646`.
- **Definitions:** `None`; primary base set `None`; entropies .
- **Data:** None integers, None coprime (n, b) pairs; split None (split sha256 `unknown`).
- **Inference:** None bootstrap resamples (95% percentile CIs over test rows), None permutations; effect threshold None (ΔR², η² excess, r²).

Paper statistics are re-computed the way the v8.2 code computed them (in-sample, `H_v8`, n ∈ [2, 9999]); corrected tests fit on the train split and report on the test split.

## Verdicts

No `claims` stage in this run; see the sections below.

## Step 9b — Order-only ω classification beyond 10⁶

58808 integers, up to 3000 per class min(ω, 5) per decade; orders and Midy flags in bases 2..199; GBM accuracy averaged over 3 seeds; each base set gets log n, per-base log L, coprime count, lcm arithmetic and Midy flags.

| decade | test rows | majority | lcm19 = λ | L19 | L2..31 | L2..99 | L2..199 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10000–99999 | 2088 | 0.2031 | 1.0000 | 0.9598 | 0.9595 | 0.9772 | 0.9792 |
| 100000–999999 | 2224 | 0.2005 | 0.9999 | 0.9330 | 0.9315 | 0.9622 | 0.9732 |
| 1000000–9999999 | 2271 | 0.1964 | 0.9999 | 0.8901 | 0.8892 | 0.9275 | 0.9445 |
| 10000000–99999999 | 2238 | 0.1899 | 0.9997 | 0.8591 | 0.8554 | 0.8856 | 0.8971 |

Gains between consecutive base sets (points, 95% CI):

- 10000–99999: L2..31 vs L19: -0.03 [-0.407, +0.360]; L2..99 vs L2..31: +1.77 [+1.205, +2.364]; L2..199 vs L2..99: +0.21 [-0.192, +0.686]
- 100000–999999: L2..31 vs L19: -0.15 [-0.615, +0.330]; L2..99 vs L2..31: +3.07 [+2.368, +3.920]; L2..199 vs L2..99: +1.09 [+0.525, +1.634]
- 1000000–9999999: L2..31 vs L19: -0.09 [-0.631, +0.507]; L2..99 vs L2..31: +3.83 [+2.780, +4.770]; L2..199 vs L2..99: +1.70 [+0.858, +2.562]
- 10000000–99999999: L2..31 vs L19: -0.37 [-0.968, +0.216]; L2..99 vs L2..31: +3.02 [+2.070, +3.925]; L2..199 vs L2..99: +1.15 [+0.409, +1.914]

Sampling: 10000–99999: {'1': 3000, '2': 3000, '3': 3000, '4': 3000, '5': 1808} from 90000 draws; 100000–999999: {'1': 3000, '2': 3000, '3': 3000, '4': 3000, '5': 3000} from 179445 draws; 1000000–9999999: {'1': 3000, '2': 3000, '3': 3000, '4': 3000, '5': 3000} from 197793 draws; 10000000–99999999: {'1': 3000, '2': 3000, '3': 3000, '4': 3000, '5': 3000} from 199787 draws

Full numbers: `orders.json` in this run directory.
