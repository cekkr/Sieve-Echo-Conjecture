# Sieve Echo v11 — claim verdicts (full-s0-step4)

- **Run:** `full-s0-step4`, config `empirical-scripts/11.0/config/full.json` (sha256 `67e9705a94`), seed 0.
- **Code:** root `4c8c1b0011` (dirty), libs/evolvo `74564d817f` (dirty), v11 sources sha256 `19bb5b9539`.
- **Definitions:** `ndr-v11.1`; primary base set `v8`; entropies `H_paper`, `H_nodc`, `Hn_paper`.
- **Data:** 162007 integers, 2340678 coprime (n, b) pairs; split {'train': 113404, 'val': 24301, 'test': 24302} (split sha256 `08379e07c5`).
- **Inference:** 1000 bootstrap resamples (95% percentile CIs over test rows), 1000 permutations; effect threshold 0.01 (ΔR², η² excess, r²).

Paper statistics are re-computed the way the v8.2 code computed them (in-sample, `H_v8`, n ∈ [2, 9999]); corrected tests fit on the train split and report on the test split.

## Verdicts

| Claim | Paper says | Paper statistic re-computed | Corrected test (held out, with controls) | Verdict |
| --- | --- | --- | --- | --- |
| **C1** Sieve Echo Law H = α·log ω + β (§2.2) | H_NDR = -0.468·log(ω) + 1.917, R² = 0.234 | reproduced | H_paper: test R² law-only 0.364, baseline (log L, log n) 0.959, full 0.959; ΔR² 0.0004 CI [0.000, 0.001] (threshold 0.01); partial r(H, log ω \| log L, log n) -0.101. | **reproduced but explained by size/length** |
| **C2** Direction of the effect (§1.3, §7.1) | §1.3: primes give 'pure, low-entropy' patterns; §7.1: 'higher complexity correlates with higher entropy'; §2.2: α < 0 | reproduced | Measured on test (H_paper): primes minus composites mean H = 0.706 CI [0.686, 0.724]; Spearman(H, ω) -0.614 (negative); given log L and log n the sign is negative. Primes have the highest entropy, so §1.3 ('primes low-e… | **not supported** |
| **C3** Prime-power dichotomy (§2.3) | strongest predictor of NDR entropy is prime-power status, r = -0.843 | not reproduced | The paper's -0.843 is corr(ω, is_prime_power) (v11: -0.655 on [2, 9999]), not an entropy correlation. Actual r(H, prime power) on test = 0.389 (positive: prime powers have higher H); given log L, log n: -0.058 CI [-0.06… | **not reproduced** |
| **C4** Base invariance CV < 0.1 (§2.4, Q5.2) | base invariance in only 6% of cases | reproduced | v8.2's 6% is a 100-integer sample (v11 on all n in [2, 9999]: 0.099). Spearman(CV, log n) for v8/H_paper: -0.036 CI [-0.041, -0.032]. Highest overall rate: wide/Hn_nodc = 0.989. Normalized DC-free entropy saturates near… | **reproduced; no asymptotic emergence** |
| **C5** Correlation with n^-s (zeta, §4.1) | H correlates with n^-s, decreasing in s (zeta connection) | reproduced | n^-s is a monotone function of n; after a cubic in log n the partial correlations are s=2: -0.023, s=3: -0.019, s=4: -0.017; H ~ log n alone has test R² 0.208. | **reproduced but explained by size/length** |
| **C6** H·log n vs prime number theorem (§4.2) | corr(H·log n, H) = 0.946 links NDR entropy to the PNT | reproduced | corr(H·log n, H) = 0.946; with H randomly permuted across n the same statistic is 0.831 CI [0.826, 0.836]: most of it comes from H appearing on both sides, the rest from H growing with n (corr(H, log n) = 0.593). PNT/Li… | **reproduced but explained by size/length** |
| **C7** Residue-class structure mod 6/12/30/210 (§4.3) | variance of residue-class mean H > 0.01 marks significant modular structure | reproduced | η² of residue class on test, raw → excess over the permutation null after controls. H_paper: mod 6: 0.222 → excess 0.000 (p 0.014), mod 12: 0.246 → excess 0.033 (p 0.002), mod 30: 0.274 → excess 0.014 (p 0.002), mod 210… | **supported (residue structure beyond per-base L and ω)** |
| **C8** Fitted constants vs theory and constant matching (§5) | α ≈ -0.468 ≈ tribonacci - ln 10, β ≈ 1.917 ≈ 1/φ² + ln 10; theory α = -1.599, β = 4.933 | reproduced | Theory constants inside a 95% CI: no. A value drawn at random near α matches some library combination within 0.01 with probability 0.87 (near β: 0.88), so the matches carry no evidence; α and β also change with log(ω) v… | **not supported** |
| **C9** Information about ω beyond L and n (core question) | NDR entropy patterns encode factorization complexity (Conj. 1.3) | n/a | Held-out ω classification: baseline (n, per-base L, lcm arithmetic) accuracy 0.982; +entropy features 0.986 (Δ +0.004 CI [0.002, 0.005]); +all NDR features 0.989 (Δ +0.006 CI [0.005, 0.008], log-loss gain +0.0175 CI [0.… | **supported** |

## C1 — Sieve Echo Law H = α·log ω + β (§2.2)

- **Paper:** H_NDR = -0.468·log(ω) + 1.917, R² = 0.234
- **Re-computed paper statistic (reproduced):** {"statistic": "OLS of v8-set mean H_v8 on log(omega+1), n in [2, 9999], in-sample", "paper": {"alpha": -0.4681, "beta": 1.9174, "r2": 0.2345}, "v11": {"log(omega+1) [v8.2 code]": {"alpha": -0.4681, "beta": 1.9174, "r2": 0.2345}, "log(omega) [paper text]": {"alpha": -0.3047, "beta": 1.6049, "r2": 0.2288}}}
- **Corrected test:** H_paper: test R² law-only 0.364, baseline (log L, log n) 0.959, full 0.959; ΔR² 0.0004 CI [0.000, 0.001] (threshold 0.01); partial r(H, log ω | log L, log n) -0.101.
- **Verdict:** reproduced but explained by size/length

## C2 — Direction of the effect (§1.3, §7.1)

- **Paper:** §1.3: primes give 'pure, low-entropy' patterns; §7.1: 'higher complexity correlates with higher entropy'; §2.2: α < 0
- **Re-computed paper statistic (reproduced):** {"statistic": "sign of α in C1", "note": "α < 0 in the stored v8.2 fit"}
- **Corrected test:** Measured on test (H_paper): primes minus composites mean H = 0.706 CI [0.686, 0.724]; Spearman(H, ω) -0.614 (negative); given log L and log n the sign is negative. Primes have the highest entropy, so §1.3 ('primes low-entropy') and §7.1 item 2 are contradicted; only α < 0 holds.
- **Verdict:** not supported

## C3 — Prime-power dichotomy (§2.3)

- **Paper:** strongest predictor of NDR entropy is prime-power status, r = -0.843
- **Re-computed paper statistic (not reproduced):** {"statistic": "paper r = -0.843; source: v7.0 log line 210 `omega_vs_is_prime_power` (two factorization labels)", "paper": -0.843, "v11": {"corr(omega, is_prime_power), n in [2, 9999]": -0.6551, "corr(H_v8 mean, is_prime_power), n in [2, 9999]": 0.3311}, "note": "-0.843 is not an entropy statistic; its exact v7 sample is unknown."}
- **Corrected test:** The paper's -0.843 is corr(ω, is_prime_power) (v11: -0.655 on [2, 9999]), not an entropy correlation. Actual r(H, prime power) on test = 0.389 (positive: prime powers have higher H); given log L, log n: -0.058 CI [-0.068, -0.047].
- **Verdict:** not reproduced

## C4 — Base invariance CV < 0.1 (§2.4, Q5.2)

- **Paper:** base invariance in only 6% of cases
- **Re-computed paper statistic (reproduced):** {"statistic": "share of n with CV(H_v8 over v8 bases) < 0.1", "v11": {"paper": 0.06, "v11 first 100 of v8.2 data order": 0.06, "v11 all n in [2, 9999]": 0.0987}, "note": "v8.2 evaluated only data[:100]."}
- **Corrected test:** v8.2's 6% is a 100-integer sample (v11 on all n in [2, 9999]: 0.099). Spearman(CV, log n) for v8/H_paper: -0.036 CI [-0.041, -0.032]. Highest overall rate: wide/Hn_nodc = 0.989. Normalized DC-free entropy saturates near its maximum for long, noise-like repetends, so its CV shrinks with L; the CV of log L itself also falls with n.
- **Verdict:** reproduced; no asymptotic emergence

## C5 — Correlation with n^-s (zeta, §4.1)

- **Paper:** H correlates with n^-s, decreasing in s (zeta connection)
- **Re-computed paper statistic (reproduced):** {"statistic": "corr(n^-s, v8 mean H_v8) over v8.2 data[:100]", "v11": {"paper": {"2": -0.4407, "3": -0.3373, "4": -0.2892}, "v11 first 100 of v8.2 data order": {"2": -0.4407, "3": -0.3373, "4": -0.2892}, "v11 all n in [2, 9999]": {"2": -0.1143, "3": -0.0825, "4": -0.07}}}
- **Corrected test:** n^-s is a monotone function of n; after a cubic in log n the partial correlations are s=2: -0.023, s=3: -0.019, s=4: -0.017; H ~ log n alone has test R² 0.208.
- **Verdict:** reproduced but explained by size/length

## C6 — H·log n vs prime number theorem (§4.2)

- **Paper:** corr(H·log n, H) = 0.946 links NDR entropy to the PNT
- **Re-computed paper statistic (reproduced):** {"statistic": "corr(H·log n, H) over n in [3, 9999], v8 mean H_v8", "paper": {"pnt_entropy_product": 0.9465, "pnt_error": 0.5199, "li_error": 0.4959}, "v11": {"pnt_entropy_product": 0.9465, "pnt_error": 0.5199, "li_error": 0.4959}, "null_permuted_H": {"pnt_entropy_product_mean": 0.8307, "ci95": [0.8257, 0.8359]}}
- **Corrected test:** corr(H·log n, H) = 0.946; with H randomly permuted across n the same statistic is 0.831 CI [0.826, 0.836]: most of it comes from H appearing on both sides, the rest from H growing with n (corr(H, log n) = 0.593). PNT/Li errors are functions of n; partial r given log n: pnt -0.033, li -0.019.
- **Verdict:** reproduced but explained by size/length

## C7 — Residue-class structure mod 6/12/30/210 (§4.3)

- **Paper:** variance of residue-class mean H > 0.01 marks significant modular structure
- **Re-computed paper statistic (reproduced):** {"statistic": "variance of residue-class means of v8 mean H_v8, n in [2, 9999]", "paper": {"6": 0.0153, "12": 0.0169, "30": 0.018, "210": 0.0205}, "v11": {"6": 0.0153, "12": 0.0169, "30": 0.018, "210": 0.0205}}
- **Corrected test:** η² of residue class on test, raw → excess over the permutation null after controls. H_paper: mod 6: 0.222 → excess 0.000 (p 0.014), mod 12: 0.246 → excess 0.033 (p 0.002), mod 30: 0.274 → excess 0.014 (p 0.002), mod 210: 0.322 → excess 0.032 (p 0.002). H_nodc: mod 6: 0.129 → excess -0.000 (p 0.739), mod 12: 0.158 → excess 0.033 (p 0.002), mod 30: 0.193 → excess 0.005 (p 0.002), mod 210: 0.254 → excess 0.018 (p 0.002). Residues mod 6/12/30/210 fix divisibility by 2, 3, 5, 7 and the quadratic characters of the bases, which set the per-base orders.
- **Verdict:** supported (residue structure beyond per-base L and ω)

## C8 — Fitted constants vs theory and constant matching (§5)

- **Paper:** α ≈ -0.468 ≈ tribonacci - ln 10, β ≈ 1.917 ≈ 1/φ² + ln 10; theory α = -1.599, β = 4.933
- **Re-computed paper statistic (reproduced):** {"statistic": "v8.2 constant matching: 20 constants, singles and pairwise sums/differences, \|error\| < 0.01", "look_elsewhere": {"stored_match": "tribonacci - ln_10", "stored_match_value": -0.4633, "reproduced_alpha": -0.4681, "matches_within_0.01_of_reproduced_alpha": 1, "library_size": 20, "candidate_values": 605, "tolerance": 0.01, "chance_match_probability_near_alpha": 0.871, "chance_match_probability_near_beta": 0.8828}}
- **Corrected test:** Theory constants inside a 95% CI: no. A value drawn at random near α matches some library combination within 0.01 with probability 0.87 (near β: 0.88), so the matches carry no evidence; α and β also change with log(ω) vs log(ω+1) and with the entropy definition.
- **Verdict:** not supported

## C9 — Information about ω beyond L and n (core question)

- **Paper:** NDR entropy patterns encode factorization complexity (Conj. 1.3)
- **Re-computed paper statistic (n/a):** {"statistic": "no paper statistic controls for L"}
- **Corrected test:** Held-out ω classification: baseline (n, per-base L, lcm arithmetic) accuracy 0.982; +entropy features 0.986 (Δ +0.004 CI [0.002, 0.005]); +all NDR features 0.989 (Δ +0.006 CI [0.005, 0.008], log-loss gain +0.0175 CI [0.014, 0.021]); majority 0.372, shuffled-label null 0.372.
- **Verdict:** supported

## Step 4 — Evolvo formula search

Inputs (12, z-scored on train): `log n`, `row.v8.logL.mean`, `row.v8.H_paper.mean`, `row.v8.H_nodc.mean`, `row.v8.H_paper.std`, `row.v8.H_paper.min`, `row.v8.Hn_nodc.mean`, `row.v8.mean.mean`, `log lcm(L_b)`, `log n - log lcm`, `lcm divides n-1`, `share of L_b dividing n-1`.
Population 200, 60 generations, seeds [0, 1, 2, 3, 4], parsimony 0.002 per instruction; fitness = 1/(1+NMSE) − parsimony·size on the first N train rows (S1: N from 5000 in steps of 5000 to ≤ 100000, capped at 113404 train rows). q_target {'omega': 0.7, 'is_prime': 0.95}. Test metrics are for the genome selected on validation.

| target | guide | variant | runs | test NMSE | test accuracy | reached q_target | evaluations to target | final N | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| is_prime | none | adaptive | 5 | 0.0014 ± 0.0000 | 0.9999 ± 0.0000 | 5/5 | 696 ± 629 | 100000 ± 0 | 51.0 ± 3.3 |
| is_prime | none | fixed | 5 | 0.0014 ± 0.0001 | 0.9999 ± 0.0000 | 5/5 | 560 ± 360 | 5000 ± 0 | 45.6 ± 2.8 |
| is_prime | policy | adaptive | 5 | 0.0014 ± 0.0001 | 0.9999 ± 0.0000 | 5/5 | 620 ± 465 | 100000 ± 0 | 63.1 ± 3.0 |
| is_prime | policy | knowledge | 1 | 0.0014 ± 0.0000 | 0.9999 ± 0.0000 | 1/1 | 760 ± 0 | 100000 ± 0 | 61.3 ± 0.0 |
| is_prime | policy | warm | 5 | 0.0014 ± 0.0001 | 0.9999 ± 0.0000 | 5/5 | 504 ± 298 | 100000 ± 0 | 63.2 ± 9.5 |
| is_prime | qlearning | adaptive | 5 | 0.0014 ± 0.0001 | 0.9999 ± 0.0000 | 5/5 | 496 ± 252 | 100000 ± 0 | 69.6 ± 4.1 |
| is_prime | supervised | adaptive | 5 | 0.0014 ± 0.0000 | 0.9999 ± 0.0000 | 5/5 | 576 ± 385 | 100000 ± 0 | 81.0 ± 5.4 |
| omega | none | adaptive | 5 | 0.3087 ± 0.0088 | 0.6443 ± 0.0338 | 5/5 | 2524 ± 2358 | 100000 ± 0 | 50.5 ± 11.7 |
| omega | none | fixed | 5 | 0.3149 ± 0.0000 | 0.6505 ± 0.0000 | 5/5 | 2180 ± 2080 | 5000 ± 0 | 49.7 ± 2.0 |
| omega | policy | adaptive | 5 | 0.3102 ± 0.0068 | 0.6337 ± 0.0425 | 5/5 | 3692 ± 2420 | 98000 ± 4472 | 60.0 ± 7.9 |
| omega | policy | knowledge | 1 | 0.3149 ± 0.0000 | 0.6505 ± 0.0000 | 1/1 | 200 ± 0 | 100000 ± 0 | 79.2 ± 0.0 |
| omega | policy | warm | 5 | 0.3341 ± 0.0668 | 0.6287 ± 0.0513 | 4/5 | 2775 ± 2443 | 93000 ± 15652 | 67.9 ± 5.3 |
| omega | qlearning | adaptive | 5 | 0.3667 ± 0.0710 | 0.6059 ± 0.0611 | 3/5 | 3100 ± 4698 | 85000 ± 11180 | 60.9 ± 4.9 |
| omega | supervised | adaptive | 5 | 0.3469 ± 0.0623 | 0.6212 ± 0.0478 | 4/5 | 2745 ± 2598 | 89000 ± 15969 | 75.6 ± 4.8 |

**S1 / S2 checks** (ROADMAP done-when conditions):

- omega S1: adaptive N test NMSE 0.3087 ± 0.0088 vs fixed 0.3149 ± 0.0000; N rose in 20 runs → **met** (Mann–Whitney p = 0.18).
- omega S2: policy vs no guide — fewer evaluations to target: False, smaller spread: True; warm start better than cold: False (lower test NMSE: False, fewer evaluations to target: False) → **not met** (test-NMSE p: policy vs none 1.00, warm vs cold 0.37).
- is_prime S1: adaptive N test NMSE 0.0014 ± 0.0000 vs fixed 0.0014 ± 0.0001; N rose in 20 runs → **not met** (Mann–Whitney p = 0.50).
- is_prime S2: policy vs no guide — fewer evaluations to target: True, smaller spread: False; warm start better than cold: True (lower test NMSE: False, fewer evaluations to target: True) → **met** (test-NMSE p: policy vs none 0.24, warm vs cold 0.29).

**Best formulas (selected on validation; inputs are z-scores):**

- omega (policy/warm/seed 4, 5 instructions): `(([log n - log lcm] * sqrt(sqrt(abs([lcm divides n-1])))) + 2.71828)` — test NMSE 0.2763, accuracy 0.6550
- is_prime (none/adaptive/seed 3, 3 instructions): `([lcm divides n-1] ^ (2 % ([row.v8.mean.mean] / 10)))` — test NMSE 0.0015, accuracy 0.9999

## Held-out results (test split)

Train rows 113404, test rows 24302. ω is scored as classes min(ω, 5).

**omega**

| model | accuracy | macro-F1 | MSE (class) | log-loss |
| --- | --- | --- | --- | --- |
| majority | 0.3716 | 0.1084 | 1.0605 | 1.4009 |
| logistic: log n + mean log L | 0.5972 | 0.5916 | 0.4520 | 0.8716 |
| GBM: log n + mean log L | 0.6085 | 0.6032 | 0.4333 | 0.8503 |
| GBM: n + L arithmetic | 0.9822 | 0.9755 | 0.0178 | 0.0520 |
| GBM: formula inputs | 0.8961 | 0.8740 | 0.1124 | 0.2539 |
| GBM: all allowed | 0.9898 | 0.9853 | 0.0102 | 0.0330 |
| GFSL formula (best on validation) | 0.6550 | 0.6418 | 0.2948 (raw target) | – |
| GFSL-NN `vec` (26 architectures) | 0.9481 | 0.9501 | 0.0522 | 0.1364 |
| GFSL-NN `set` (26 architectures) | 0.9479 | 0.9501 | 0.0521 | 0.1344 |

**is_prime**

| model | accuracy | F1 | MSE (class) | log-loss |
| --- | --- | --- | --- | --- |
| majority | 0.9019 | 0.0000 | 0.0981 | 0.3209 |
| logistic: log n + mean log L | 0.9507 | 0.7271 | 0.0493 | 0.1266 |
| GBM: log n + mean log L | 0.9534 | 0.7259 | 0.0466 | 0.1228 |
| GBM: n + L arithmetic | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| GBM: formula inputs | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| GBM: all allowed | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| GFSL formula (best on validation) | 0.9999 | 0.9994 | 0.0001 (raw target) | – |

**§3.1 three-feature retest** (paper: 98.8%, v7.0 training-set accuracy with `is_prime_power`)

| features | train accuracy | test accuracy |
| --- | --- | --- |
| std H, min H | 0.5207 | 0.5085 |
| std H, min H, log n | 0.5887 | 0.5658 |
| LEAKAGE DEMO: std H, min H, is_prime_power | 0.5980 | 0.5864 |
| majority class | – | 0.3716 |

is_prime_power alone identifies omega = 1; it is a factorization label and is forbidden as an input outside this demonstration.

Full numbers: `claims.json`, `formulas.json`, `baselines.json`, `nn.json` in this run directory.
