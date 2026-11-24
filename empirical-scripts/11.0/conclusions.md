# Sieve Echo Conjecture — Conclusions of v11

What version 11 of the empirical program ([`empirical-scripts/11.0/`](.)) established, after ROADMAP steps 0–11. Every number comes from a v11 run (seed 0, NDR definitions `ndr-v11.1`), named in parentheses; run summaries, manifests and figures are under [`results/<run-id>/`](results). Earlier script generations (v3.1–v10) are not sources here: v11 reproduces their stored statistics only to explain them. The detailed argument, with proofs, is the revision-9 draft [`studies/9/sieve-echo-conjecture-9.md`](../../studies/9/sieve-echo-conjecture-9.md); the step-by-step log is the [ROADMAP Results log](../../ROADMAP.md#results-log).

## The question

The conjecture says that ω(n), the number of distinct prime factors of *n*, is encoded in the spectral entropy of the normalized repetend digits of 1/*n* across bases (NDR entropy). v11 asks this with what the earlier versions lacked: one canonical, tested definition; a train/validation/test split by *n*; baselines that already know *n* and the repetend lengths L = ord_n(b); permutation and shuffled-label nulls; bootstrap intervals; and evolutionary search through the Evolvo submodule scored only on held-out integers.

## Answer in one paragraph

**No — not beyond the repetend lengths.** The factorization of *n* does reach 1/*n*, but through the multiplicative orders ord_n(b), which are the repetend lengths. NDR entropy is almost a function of length (H ≤ log L), so it inherits the orders' information about ω and adds nothing measurable once the orders in enough bases are known: −0.07 accuracy points, 95% CI [−0.33, +0.24], at 10⁵–10⁶ (`scale-s0-step7`). The digits do carry two exact traces of the factorization — Midy's symmetry and exact spectral zeros — but both are classical number theory and both are already contained in the orders. What the orders encode is the group structure of (ℤ/nℤ)*: ω(n) is, up to a correction for the power of 2 in *n*, the 2-rank of that group.

## 1. The v8 paper's statistics: reproduced, then explained

v11 recomputes every stored statistic of the v8 paper from its own code path (v8.2, `8.0/sieve-echo-tests-2.py`) and gets the same numbers (`full-s0-claims`; unit test `ReproduceV8Tests` matches the stored per-base entropies to 10⁻⁹). Each one is explained by something other than an ω signal.

| v8 claim | reproduced | what explains it (v11 test) |
| --- | --- | --- |
| Law H = α·log ω + β, α = −0.4681, β = 1.9174, R² = 0.2345 | exactly (the code fitted log(ω+1)) | log L + log n give held-out R² 0.959; log ω adds 0.0004, CI [0.000, 0.001] |
| Constants α ≈ −1/φ², β = 5 − 1/15 | no | no 95% interval of α or β contains them under any definition |
| Prime-power dichotomy, r = −0.843 | not an entropy statistic | it is corr(ω, is_prime_power) from a v7.0 log; given log L and log n, r(H, prime power) = −0.058 |
| Base invariance, 6% | exactly, on v8's first 100 integers | a 100-integer sample: 9.1% on all n ∈ [2, 9 999] (`quick-s0-step4`); for the paper entropy, 13–21% of integers have CV < 0.1 per decade to 10⁶ (`scale-s0`) |
| Links to ζ(s) and the PNT | the correlations, yes | partial correlations given log n are −0.02 to −0.03; H·log n vs H is 0.831 with H permuted |
| Residue-class variance mod 6/30/210 | identical | which bases are coprime to n and how long their repetends are |
| 98.8% three-feature accuracy | not reproducible from v8 | training-set accuracy of a v7.0 model that used `is_prime_power` as input |

The direction is also settled: **primes have the highest entropy** (primes − composites +0.706, CI [0.686, 0.724]), because their repetends are longest. The v8 sentences that call prime patterns "low-entropy" or tie "higher complexity" to "higher entropy" contradict the data.

## 2. Entropy is a length statistic

- log L and log n explain 95.9% of the held-out variance of the paper entropy; 99.1% of the DC-free entropy (`full-s0-step4`).
- The negative slope of the "law" has a proof: for odd *n*, λ(n) ≤ φ(n)/2^{ω−1} (Theorem 1.3 of `studies/9`), so each extra prime factor shortens every repetend.
- Entropy grows with *n* as n^0.068 on ω-balanced decades and as n^0.020 once log L is controlled (`scale-s0`): its growth is the growth of L.

## 3. The information about ω is in the orders

- Gradient boosting on *n*, the per-base lengths and their lcm arithmetic classifies min(ω, 5) at **98.2%** held-out (majority 37.2%, shuffled-label null 37.2%); all NDR columns add +0.6 points (`full-s0-step4`).
- The lcm arithmetic matters: a tree cannot compute lcm(L_b) or test whether it divides n − 1. Without it the same model reaches 94.9%, with it 97.8% (`quick-s0-step6`) — a gap NDR features would otherwise appear to fill.
- Evolvo formula search rediscovers Fermat's little theorem: a formula of at most three instructions on "lcm of the lengths divides n − 1" recognizes primes at 99.99% held-out. Evolved ω formulas stay weak (65.5%), essentially log(n/λ(n)) (`full-s0-step4`).
- The orders need no digits and can be followed far beyond the NDR data. From orders in bases 2…199, ω accuracy is 97.9%, 97.3%, 94.5%, 89.7% in the decades 10⁴…10⁸ (`orders-s0`). A fixed base set reads less as *n* grows; more bases recover part of it.

## 4. The apparent residual signal was an under-powered baseline

At 10⁵–10⁶ entropy seemed to add +1.3 points over a 19-base length baseline, growing with *n* (`scale-s0`). Step 7 shows why, with five-fit averages on the same samples (`scale-s0-step7`):

| baseline at 10⁵–10⁶ | accuracy | entropy gain (points, 95% CI) |
| --- | --- | --- |
| n + 19 orders + lcm arithmetic | 92.1% | +0.90 [+0.19, +1.54] |
| … + 19 Midy flags | 92.4% | +0.47 [−0.11, +0.94] |
| n + 98 orders (bases 2…99) + lcm arithmetic | 95.3% | +0.14 [−0.22, +0.50] |
| … + 98 Midy flags | 95.7% | **−0.07 [−0.33, +0.24]** |

The 19 orders lose nothing of λ(n) — their lcm equals λ(n) for every integer of the data — but they sample the order structure of the group too sparsely as *n* grows.

## 5. What the digits really encode

- **Midy's symmetry.** When −1 ∈ ⟨b⟩ mod *n*, the halves of the repetend are digit complements and every even Fourier bin vanishes; the DC-free entropy drops by log 2 (fitted −0.694 over 783 040 pairs) and the flag explains 82% of the variance that length and size leave (`scale-s0-step7`). For odd *n* it holds iff the 2-adic valuations of the local orders are equal (Proposition 1.4), so it thins out with ω (61.5% of pairs at ω = 1, 1.0% at ω = 5). It is computed from (*n*, *b*) by one modular exponentiation.
- **Exact spectral zeros.** Whole Galois classes of DFT bins vanish exactly when the Euler factors (1 − χ*(p)) of the relevant characters vanish at primes p | n (Proposition 1.5). They appear in 0.09% of prime pairs and 13–18% of composite pairs, and the exact cyclotomic test agrees with the FFT on every sampled pair — but over 98 orders and Midy flags they add −0.05 points, CI [−0.28, +0.18] (`scale-s0-step8`).
- **SinWhisky spectra** (circle-based resampling before the DFT, directive D8) correlate 0.9998 with the plain spectrum and add no ω information; being nonlinear, they leak power into the exact zero classes for 23% of the pairs that have them, which periodic linear interpolation never does (`scale-s0-step8e`).

## 6. Why the orders know ω: the 2-rank

Proposition 1.6 of `studies/9`, verified exactly on all units for every n < 3 000 (`tests/test_tworank.py`): for a uniform unit *b*, P(v₂(ord b) ≤ k) = 2^{−Σ max(e_i − k, 0)} over the Sylow-2 exponents e_i of (ℤ/nℤ)*, so P(v₂ ≤ 1)/P(v₂ = 0) = 2^{rank₂}, and ω(n) = rank₂ − δ(v₂(n)) + [n even]. The repetend lengths are samples of element orders, and their 2-adic valuations reveal the rank.

| ω classifier, bases 2…499 | 10⁴–10⁵ | 10⁵–10⁶ | 10⁶–10⁷ | 10⁷–10⁸ |
| --- | --- | --- | --- | --- |
| ratio estimator (no training) | 81.7% | 75.5% | 71.2% | 69.4% |
| maximum-likelihood 2-rank (no training) | 83.4% | 77.5% | 73.3% | 72.0% |
| learned: raw lengths of the same bases | 97.9% | 97.6% | 95.4% | 90.6% |
| learned: raw lengths + 2- and 3-adic profiles | 98.6% | 98.2% | 97.0% | 94.2% |

All rows from `step11-s0` (up to 3 000 integers per class min(ω, 5) per decade; the ratio estimator reproduces `tworank-s0`).

At a fixed base set the valuation profiles add +0.7 to +3.5 points, growing with *n*, while 300 extra bases alone add at most +0.95 (`step11-s0`): a model fed log-lengths does not extract the 2-adic reading by itself. The training-free estimators still trail the learned models by 15–24 points, mainly because about 10% of integers have no base of odd order among 498 bases and because composite bases' orders are tied to those of their factors.

## 7. Residue-class structure

Residue classes of *n* mod 12 and mod 210 leave a small excess in entropy after per-base length, coprimality, size and ω controls (η² above the permutation null 0.033 and 0.032; `full-s0-claims`). Part of it is Euler-product structure: Midy flags plus the flags "q ∈ ⟨b⟩" for q = 2…13 reduce it to 0.011 (mod 12) and 0.023 (mod 210) (`scale-s0-step9`). The mean digit — the class-number-type DC term — does not explain the rest (0.011 → 0.013; `scale-s0-step11`).

## 8. Verdict

| statement | verdict |
| --- | --- |
| NDR entropy correlates with ω | yes, through repetend length |
| NDR entropy encodes ω beyond the repetend lengths | **not supported** (−0.07 points, CI [−0.33, +0.24], at 10⁵–10⁶) |
| the "echo" grows with *n* | not supported; the step-5 drift disappears with 98 orders |
| the digits carry factorization traces | yes: Midy's symmetry and exact spectral zeros — both classical, both redundant given the orders |
| the repetend lengths encode ω | yes: as the 2-rank of (ℤ/nℤ)*, read from the 2-adic valuations of the orders |
| law constants, universal base invariance, ζ/PNT links, prime-power dichotomy, three-feature sufficiency | not supported |

The conjecture's intuition — that the sieve leaves an echo in 1/*n* — is right, but the echo is the multiplicative order and its group structure, not the entropy of the digits.

## 9. Limits of these conclusions

- **One seed.** All runs use seed 0. Steps 7–11 average gradient-boosting fits over 3–5 model seeds, because a single fit varies by up to half a point; the step-2 to step-5 C9 numbers are single fits.
- **Samples, not all integers.** NDR data covers every n ≤ 150 000 (`full-s0`) and ω-balanced samples of 3 000 per class per decade to 10⁶ (`scale-s0`); orders-only results reach 10⁸ on the same kind of samples. Rare classes are over-represented relative to their natural frequency, and ω ≥ 5 is pooled.
- **Bases.** Entropy and zeros are measured in 19 bases (v8, v10, primes < 32); orders in up to 498. A larger digit base set was not tried, since the orders that would be its baseline would then also be larger.
- **Evolutionary search** found the known structure (Fermat's test, log(n/λ(n))) and no new formula; the differences between Evolvo guides, including the PyTorch policy guide, are not significant over 5 seeds (`full-s0-step4`).
- **Owner review.** The revision-9 draft and the step 6–11 reports await the author's review.

## 10. Open questions

1. The rest of the residue-class excess (mod 12: 0.011, mod 210: 0.023): exact subgroup membership for non-cyclic (ℤ/nℤ)*, where [ord(q) | ord(b)] is only necessary, and larger primes *q*.
2. A closed-form ω estimator from the orders that approaches the learned models: modelling the dependence between composite and prime bases, and adding the ℓ-ranks for odd ℓ.
