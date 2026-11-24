# Sieve Echo v11 — claim verdicts (scale-s0-step8e)

- **Run:** `scale-s0-step8e`, config `empirical-scripts/11.0/config/scale.json` (sha256 `f0f3b481b9`), seed 0.
- **Code:** root `5570ad3b25` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `c806f2e73c`.
- **Definitions:** `ndr-v11.1`; primary base set `v8`; entropies `H_paper`, `H_nodc`, `Hn_paper`.
- **Data:** 41074 integers, 783040 coprime (n, b) pairs; split {'train': 28752, 'val': 6160, 'test': 6162} (split sha256 `9789b5b2e5`).
- **Inference:** 1000 bootstrap resamples (95% percentile CIs over test rows), 1000 permutations; effect threshold 0.01 (ΔR², η² excess, r²).

Paper statistics are re-computed the way the v8.2 code computed them (in-sample, `H_v8`, n ∈ [2, 9999]); corrected tests fit on the train split and report on the test split.

## Figures

![Test-split integers: mean H_paper against mean log L, coloured by ω. Points with more prime factors sit lower because their repetends are shorter, not off the H–log L curve (claims C1, C9).](figures/entropy_vs_length.png)

*Test-split integers: mean H_paper against mean log L, coloured by ω. Points with more prime factors sit lower because their repetends are shorter, not off the H–log L curve (claims C1, C9).*

## Verdicts

| Claim | Paper says | Paper statistic re-computed | Corrected test (held out, with controls) | Verdict |
| --- | --- | --- | --- | --- |
| **C1** Sieve Echo Law H = α·log ω + β (§2.2) | H_NDR = -0.468·log(ω) + 1.917, R² = 0.234 | reproduced | H_paper: test R² law-only 0.298, baseline (log L, log n) 0.971, full 0.972; ΔR² 0.0005 CI [0.000, 0.001] (threshold 0.01); partial r(H, log ω \| log L, log n) -0.135. | **reproduced but explained by size/length** |
| **C2** Direction of the effect (§1.3, §7.1) | §1.3: primes give 'pure, low-entropy' patterns; §7.1: 'higher complexity correlates with higher entropy'; §2.2: α < 0 | reproduced | Measured on test (H_paper): primes minus composites mean H = 0.902 CI [0.860, 0.946]; Spearman(H, ω) -0.515 (negative); given log L and log n the sign is negative. Primes have the highest entropy, so §1.3 ('primes low-e… | **not supported** |
| **C3** Prime-power dichotomy (§2.3) | strongest predictor of NDR entropy is prime-power status, r = -0.843 | not reproduced | The paper's -0.843 is corr(ω, is_prime_power) (v11: -0.655 on [2, 9999]), not an entropy correlation. Actual r(H, prime power) on test = 0.468 (positive: prime powers have higher H); given log L, log n: -0.057 CI [-0.08… | **not reproduced** |
| **C4** Base invariance CV < 0.1 (§2.4, Q5.2) | base invariance in only 6% of cases | reproduced | v8.2's 6% is a 100-integer sample (v11 on all n in [2, 9999]: 0.091). Spearman(CV, log n) for v8/H_paper: -0.102 CI [-0.111, -0.093]. Highest overall rate: v8/Hn_nodc = 0.977. Normalized DC-free entropy saturates near i… | **reproduced; invariance increases with n** |
| **C5** Correlation with n^-s (zeta, §4.1) | H correlates with n^-s, decreasing in s (zeta connection) | reproduced | n^-s is a monotone function of n; after a cubic in log n the partial correlations are s=2: -0.011, s=3: -0.011, s=4: -0.010; H ~ log n alone has test R² 0.334. | **reproduced but explained by size/length** |
| **C6** H·log n vs prime number theorem (§4.2) | corr(H·log n, H) = 0.946 links NDR entropy to the PNT | reproduced | corr(H·log n, H) = 0.946; with H randomly permuted across n the same statistic is 0.831 CI [0.826, 0.836]: most of it comes from H appearing on both sides, the rest from H growing with n (corr(H, log n) = 0.593). PNT/Li… | **reproduced but explained by size/length** |
| **C7** Residue-class structure mod 6/12/30/210 (§4.3) | variance of residue-class mean H > 0.01 marks significant modular structure | reproduced | η² of residue class on test, raw → excess over the permutation null after controls. H_paper: mod 6: 0.270 → excess 0.002 (p 0.018), mod 12: 0.283 → excess 0.035 (p 0.002), mod 30: 0.306 → excess 0.016 (p 0.002), mod 210… | **supported (residue structure beyond per-base L and ω)** |
| **C8** Fitted constants vs theory and constant matching (§5) | α ≈ -0.468 ≈ tribonacci - ln 10, β ≈ 1.917 ≈ 1/φ² + ln 10; theory α = -1.599, β = 4.933 | reproduced | Theory constants inside a 95% CI: no. A value drawn at random near α matches some library combination within 0.01 with probability 0.87 (near β: 0.88), so the matches carry no evidence; α and β also change with log(ω) v… | **not supported** |
| **C9** Information about ω beyond L and n (core question) | NDR entropy patterns encode factorization complexity (Conj. 1.3) | n/a | Held-out ω classification: baseline (n, per-base L, lcm arithmetic) accuracy 0.956; +entropy features 0.960 (Δ +0.005 CI [0.000, 0.009]); +all NDR features 0.965 (Δ +0.009 CI [0.004, 0.014], log-loss gain +0.0200 CI [0.… | **supported** |

## C1 — Sieve Echo Law H = α·log ω + β (§2.2)

- **Paper:** H_NDR = -0.468·log(ω) + 1.917, R² = 0.234
- **Re-computed paper statistic (reproduced):** {"statistic": "OLS of v8-set mean H_v8 on log(omega+1), n in [2, 9999], in-sample", "paper": {"alpha": -0.4681, "beta": 1.9174, "r2": 0.2345}, "v11": {"log(omega+1) [v8.2 code]": {"alpha": -0.4681, "beta": 1.9174, "r2": 0.2345}, "log(omega) [paper text]": {"alpha": -0.3047, "beta": 1.6049, "r2": 0.2288}}}
- **Corrected test:** H_paper: test R² law-only 0.298, baseline (log L, log n) 0.971, full 0.972; ΔR² 0.0005 CI [0.000, 0.001] (threshold 0.01); partial r(H, log ω | log L, log n) -0.135.
- **Verdict:** reproduced but explained by size/length

## C2 — Direction of the effect (§1.3, §7.1)

- **Paper:** §1.3: primes give 'pure, low-entropy' patterns; §7.1: 'higher complexity correlates with higher entropy'; §2.2: α < 0
- **Re-computed paper statistic (reproduced):** {"statistic": "sign of α in C1", "note": "α < 0 in the stored v8.2 fit"}
- **Corrected test:** Measured on test (H_paper): primes minus composites mean H = 0.902 CI [0.860, 0.946]; Spearman(H, ω) -0.515 (negative); given log L and log n the sign is negative. Primes have the highest entropy, so §1.3 ('primes low-entropy') and §7.1 item 2 are contradicted; only α < 0 holds.
- **Verdict:** not supported

## C3 — Prime-power dichotomy (§2.3)

- **Paper:** strongest predictor of NDR entropy is prime-power status, r = -0.843
- **Re-computed paper statistic (not reproduced):** {"statistic": "paper r = -0.843; source: v7.0 log line 210 `omega_vs_is_prime_power` (two factorization labels)", "paper": -0.843, "v11": {"corr(omega, is_prime_power), n in [2, 9999]": -0.6551, "corr(H_v8 mean, is_prime_power), n in [2, 9999]": 0.3311}, "note": "-0.843 is not an entropy statistic; its exact v7 sample is unknown."}
- **Corrected test:** The paper's -0.843 is corr(ω, is_prime_power) (v11: -0.655 on [2, 9999]), not an entropy correlation. Actual r(H, prime power) on test = 0.468 (positive: prime powers have higher H); given log L, log n: -0.057 CI [-0.080, -0.035].
- **Verdict:** not reproduced

## C4 — Base invariance CV < 0.1 (§2.4, Q5.2)

- **Paper:** base invariance in only 6% of cases
- **Re-computed paper statistic (reproduced):** {"statistic": "share of n with CV(H_v8 over v8 bases) < 0.1", "v11": {"paper": 0.06, "v11 first 100 of v8.2 data order": 0.06, "v11 all n in [2, 9999]": 0.0913}, "note": "v8.2 evaluated only data[:100]."}
- **Corrected test:** v8.2's 6% is a 100-integer sample (v11 on all n in [2, 9999]: 0.091). Spearman(CV, log n) for v8/H_paper: -0.102 CI [-0.111, -0.093]. Highest overall rate: v8/Hn_nodc = 0.977. Normalized DC-free entropy saturates near its maximum for long, noise-like repetends, so its CV shrinks with L; the CV of log L itself also falls with n.
- **Verdict:** reproduced; invariance increases with n

## C5 — Correlation with n^-s (zeta, §4.1)

- **Paper:** H correlates with n^-s, decreasing in s (zeta connection)
- **Re-computed paper statistic (reproduced):** {"statistic": "corr(n^-s, v8 mean H_v8) over v8.2 data[:100]", "v11": {"paper": {"2": -0.4407, "3": -0.3373, "4": -0.2892}, "v11 first 100 of v8.2 data order": {"2": -0.4407, "3": -0.3373, "4": -0.2892}, "v11 all n in [2, 9999]": {"2": -0.1143, "3": -0.0825, "4": -0.07}}}
- **Corrected test:** n^-s is a monotone function of n; after a cubic in log n the partial correlations are s=2: -0.011, s=3: -0.011, s=4: -0.010; H ~ log n alone has test R² 0.334.
- **Verdict:** reproduced but explained by size/length

## C6 — H·log n vs prime number theorem (§4.2)

- **Paper:** corr(H·log n, H) = 0.946 links NDR entropy to the PNT
- **Re-computed paper statistic (reproduced):** {"statistic": "corr(H·log n, H) over n in [3, 9999], v8 mean H_v8", "paper": {"pnt_entropy_product": 0.9465, "pnt_error": 0.5199, "li_error": 0.4959}, "v11": {"pnt_entropy_product": 0.9465, "pnt_error": 0.5199, "li_error": 0.4959}, "null_permuted_H": {"pnt_entropy_product_mean": 0.8307, "ci95": [0.8257, 0.8359]}}
- **Corrected test:** corr(H·log n, H) = 0.946; with H randomly permuted across n the same statistic is 0.831 CI [0.826, 0.836]: most of it comes from H appearing on both sides, the rest from H growing with n (corr(H, log n) = 0.593). PNT/Li errors are functions of n; partial r given log n: pnt -0.002, li -0.014.
- **Verdict:** reproduced but explained by size/length

## C7 — Residue-class structure mod 6/12/30/210 (§4.3)

- **Paper:** variance of residue-class mean H > 0.01 marks significant modular structure
- **Re-computed paper statistic (reproduced):** {"statistic": "variance of residue-class means of v8 mean H_v8, n in [2, 9999]", "paper": {"6": 0.0153, "12": 0.0169, "30": 0.018, "210": 0.0205}, "v11": {"6": 0.0153, "12": 0.0169, "30": 0.018, "210": 0.0205}}
- **Corrected test:** η² of residue class on test, raw → excess over the permutation null after controls. H_paper: mod 6: 0.270 → excess 0.002 (p 0.018), mod 12: 0.283 → excess 0.035 (p 0.002), mod 30: 0.306 → excess 0.016 (p 0.002), mod 210: 0.346 → excess 0.034 (p 0.002). H_nodc: mod 6: 0.191 → excess 0.001 (p 0.066), mod 12: 0.209 → excess 0.037 (p 0.002), mod 30: 0.237 → excess 0.006 (p 0.002), mod 210: 0.286 → excess 0.015 (p 0.002). Residues mod 6/12/30/210 fix divisibility by 2, 3, 5, 7 and the quadratic characters of the bases, which set the per-base orders.
- **Verdict:** supported (residue structure beyond per-base L and ω)

## C8 — Fitted constants vs theory and constant matching (§5)

- **Paper:** α ≈ -0.468 ≈ tribonacci - ln 10, β ≈ 1.917 ≈ 1/φ² + ln 10; theory α = -1.599, β = 4.933
- **Re-computed paper statistic (reproduced):** {"statistic": "v8.2 constant matching: 20 constants, singles and pairwise sums/differences, \|error\| < 0.01", "look_elsewhere": {"stored_match": "tribonacci - ln_10", "stored_match_value": -0.4633, "reproduced_alpha": -0.4681, "matches_within_0.01_of_reproduced_alpha": 1, "library_size": 20, "candidate_values": 605, "tolerance": 0.01, "chance_match_probability_near_alpha": 0.871, "chance_match_probability_near_beta": 0.8828}}
- **Corrected test:** Theory constants inside a 95% CI: no. A value drawn at random near α matches some library combination within 0.01 with probability 0.87 (near β: 0.88), so the matches carry no evidence; α and β also change with log(ω) vs log(ω+1) and with the entropy definition.
- **Verdict:** not supported

## C9 — Information about ω beyond L and n (core question)

- **Paper:** NDR entropy patterns encode factorization complexity (Conj. 1.3)
- **Re-computed paper statistic (n/a):** {"statistic": "no paper statistic controls for L"}
- **Corrected test:** Held-out ω classification: baseline (n, per-base L, lcm arithmetic) accuracy 0.956; +entropy features 0.960 (Δ +0.005 CI [0.000, 0.009]); +all NDR features 0.965 (Δ +0.009 CI [0.004, 0.014], log-loss gain +0.0200 CI [0.010, 0.028]); majority 0.246, shuffled-label null 0.238.
- **Verdict:** supported

## Step 8e — NDR spectra after SinWhisky resampling

4851 (n, b) pairs (19 bases, L ≤ 50000, at most 300 per n-range × ω), each NDR period resampled ×4 by SinWhisky (libs/sinwhisky `sinwhisky_resample_np`, periodic, neighbors 1) and by periodic linear interpolation (control); entropies over all non-DC bins. corr(H_fft, H_sw) = 0.9998.

**Zero-class leakage** (961 pairs with zero classes; power at the original zero bins / mean baseband power, log10): SinWhisky median -30.2, max -2.3, share above 10⁻⁶ 0.229; linear median -31.8, max -29.6, share above 10⁻⁶ 0.000.

| entropy | test pairs | Midy coefficient | test R² (log L, log b, log n, Midy, extra-zero share) | + log ω | ΔR² [95% CI] |
| --- | --- | --- | --- | --- | --- |
| H_fft | 741 | -0.694 | 0.9970 | 0.9970 | +0.00002 [-0.000, +0.000] |
| H_sw | 741 | -0.702 | 0.9967 | 0.9967 | +0.00004 [-0.000, +0.000] |
| H_lin | 741 | -0.705 | 0.9963 | 0.9964 | +0.00006 [-0.000, +0.000] |

Full numbers: `claims.json`, `whisky.json` in this run directory.
