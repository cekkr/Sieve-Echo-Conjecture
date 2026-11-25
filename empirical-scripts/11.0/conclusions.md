# Sieve Echo Conjecture — Conclusions of v11

What version 11 of the empirical program ([`empirical-scripts/11.0/`](.)) established, after ROADMAP steps 0–16. Empirical results cite v11 runs (seed 0, NDR definitions `ndr-v11.1`) or the deterministic decoder example; run summaries, manifests and figures are under [`results/<run-id>/`](results). Earlier script generations (v3.1–v10) are not sources here: v11 reproduces their stored statistics only to explain them. The detailed argument, with proofs (Propositions 1.4–1.8), is the revision-9 draft [`studies/9/sieve-echo-conjecture-9.md`](../../studies/9/sieve-echo-conjecture-9.md); the step-by-step log is the [ROADMAP Results log](../../ROADMAP.md#results-log).

## The question

The conjecture says that ω(n), the number of distinct prime factors of *n*, is encoded in the spectral entropy of the normalized repetend digits of 1/*n* across bases (NDR entropy). v11 asks this with what the earlier versions lacked: one canonical, tested definition; a train/validation/test split by *n*; baselines that already know *n* and the repetend lengths L = ord_n(b); permutation and shuffled-label nulls; bootstrap intervals; and evolutionary search through the Evolvo submodule scored only on held-out integers.

## What v11 established

Repetend length explains most of the measured relationship between digit entropy and $\omega(n)$. The strongest baseline uses 98 orders and their Midy symmetry flags. Adding entropy changes held-out accuracy by −0.07 percentage points, with a 95% confidence interval of [−0.33, +0.24] at $10^5$–$10^6$ (`scale-s0-step7`). Exact structures remain in the periods and spectra, including Midy symmetry, order divisibility and character-induced spectral zeros. The fixed-base decoder demonstrates how much information the periods carry. Those same order observables give the next project question at prime denominators, where they define concrete prime counts.

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

## 7. Separating prime factors with order tests

Proposition 1.7 of `studies/9` (tested for every n < 30 000 in `tests/test_decode.py`): for a base b coprime to n with L = ord_n(b), a prime q | L with e = v_q(L), and i < e, a prime p | n divides gcd(b^{(L/q^e) q^i} − 1, n) iff v_q(ord_p b) ≤ i. Refining {n} by these gcds over all bases and reducing to a coprime base gives ω̂ ≤ ω, exact whenever the prime divisors of n have distinct order vectors (ord_p b)_b. Each remainder b^d mod n is the remainder after d digits of 1/n, read exactly from the ⌈log_b n⌉ digits that follow: b^d mod n = ⌊n·D/b^k⌋ + 1.

| decoder with the 19 bases, no training (`decode-s0`) | result |
| --- | --- |
| every n in [2, 10⁸], lengths only | 99 999 990 / 99 999 999 exact; the 9 exceptions are 2·3·5·11·17·23·29 = 3 741 870 and multiples, whose only coprime bases 7, 13, 19, 31 are all ≡ 1 (mod 6) |
| every n in [2, 10⁸], plus gcd(b, n) of non-coprime bases (pre-periods) | **all exact** |
| every n < 3 700 000 259 (Proposition 1.8; no two primes < 10⁸ share all 19 orders; `failures-s0`) | **all exact** (certified) |
| every n in [2, 10⁸], Shor's half-period chain only (q = 2) | 99.896% |
| 40 000 integers built from an external prime list (t5k.org, primes 961 748 941 … 982 451 653): 1–3 list primes, p²q, × primes in (31, 10⁵), × primes 2…31; n up to 2.4·10⁴⁸; lengths verified from n alone; the published `sieve_echo_decoder.py` (`validate-s0`) | **all exact**; 37 000 are coprime to every base, so no pre-period acts |
| ω-balanced decades 10⁴…10¹² (118 808 integers) | all exact; the first four decades are the `orders-s0` integers, where the best learned model reached 98.6 / 98.2 / 97.0 / 94.2% on min(ω, 5) |
| uniform samples in the same decades (157 698 integers) | all exact |
| 600 integers of 10⁴–10⁶ decoded from their digits only | all exact; 4 001 remainder readings, 0 errors, mean 6.05 digits each |
| Midy flag vs failure of Shor's split gcd(b^{L/2} − 1, n) = 1 (odd n, even L) | identical on 840 197 of 840 197 pairs |

**Where it fails (Proposition 1.8, `failures-s0`).** With the pre-periods, the decoder fails on *n* only if two prime factors of *n* above 31 have the same orders in every base coprime to *n* (for squarefree *n*, exactly then). So failures can be found from the orders of the bases modulo single primes. Modulo all 5 761 444 primes below 10⁸, no two have the same orders in all 19 bases: **the decoder recovers ω for every n < 3 700 000 259**. With fewer bases the least failing *n* is predicted exactly and matches brute force for 1–14 bases: 30, 210, 2 310, 30 030 (primorials, which leave no base coprime to *n*), 214 935 = 3·5·7·23·89 and 2 794 155 = 3·5·7·13·23·89 — pairs like 23·89 = 2¹¹ − 1 that share the order of 2 once only powers of 2 and few other bases remain.

Few bases are needed: with the first five (2, 3, 4, 5, 6) the decoder is exact for 90.0% of the ω-balanced integers at 10⁴–10⁵ and 97.2% at 10¹¹–10¹², for those samples. On average 2.45 (10⁴–10⁵) to 1.78 (10¹¹–10¹²) bases suffice.

**The final algorithm.** It is this decoder, not an Evolvo formula. Evolvo's best ω formula, essentially log(n/λ(n)), reaches 65.5% (`full-s0-step4`). The decoder is the standalone, dependency-free [`sieve_echo_decoder.py`](sieve_echo_decoder.py), tested against SymPy for every n < 20 000. Its core:

```python
def decode(n, lengths):                                  # lengths = {b: repetend length of 1/n in base b}
    parts = [n]

    def split(g):
        nonlocal parts
        parts = [y for x in parts for y in ((math.gcd(x, g), x // math.gcd(x, g)) if 1 < math.gcd(x, g) < x else (x,))]

    for b in BASES:                                      # the 19 bases
        if math.gcd(b, n) > 1:
            split(math.gcd(b, n))                        # pre-period
            continue
        L = lengths[b]
        for q in primes_of(L):
            m = L
            while m % q == 0:
                m //= q                                  # L / q^e
            while m < L:                                 # (L / q^e) q^i, i < e
                split(math.gcd(pow(b, m, n) - 1, n))
                m *= q
    factors = coprime_parts(parts)
    return {f: exponent_of(f, n) for f in factors}
```

`python sieve_echo_decoder.py` decodes hand-checkable examples and verifies each result independently:
- 2047 = 23 · 89, which base 2 cannot split;
- 1001 = 7 · 11 · 13;
- 1024 = 2¹⁰;
- 214 935 = 3 · 5 · 7 · 23 · 89, the least failure of the first 9 bases.

**What this means.** The order tests give a finite reconstruction result. Their gcd refinement is polynomial in the bit length of $n$ given the factored lengths. The standalone code uses long division to obtain the lengths and a separate routine to factor them. The classical reduction from the period of $1/n$ to factoring already appears in [Miller, Theorem 4](https://www.cs.cmu.edu/~glmiller/Publications/Papers/Mi76.pdf).

**Verified counterexample.** At $n=357{,}522{,}227{,}790$, the two primes 1187 and 3559 share the same orders in the available bases and the decoder leaves them combined. [Reproduction](../../ROADMAP.md#all-19-base-counterexample)

## 8. Residue-class structure

Residue classes of *n* mod 12 and mod 210 leave a small excess in entropy after per-base length, coprimality, size and ω controls (η² above the permutation null 0.033 and 0.032; `full-s0-claims`). It is group structure (`scale-s0-step9`, `scale-s0-step11`, `scale-s0-step14`):

| `H_paper`, excess over the permutation null | mod 12 | mod 210 |
| --- | --- | --- |
| C7 controls + Midy flags + zeros | 0.022 | 0.032 |
| + [ord q \| ord b] for q ≤ 13 (step 9a) | 0.011 | 0.023 |
| + exact q ∈ ⟨b⟩ for q ≤ 13 | 0.013 | 0.033 |
| + exact ±q ∈ ⟨b⟩ (q ≤ 13) + [ord q \| ord b] for primes q < 100 | 0.010 | 0.018 |
| … + joint coprimality pattern + v₂(n) (labelled diagnostics) | 0.002 | 0.003 |

- With the ±q and q < 100 flags, the odd prime moduli 3, 5, 7 carry no excess (≤ 0.0011 for both entropies); the remainder is 2-adic (mod 8) and follows 2 ∥ n versus 4 | n.
- The class-number-type DC term (mean digit) explains nothing (`scale-s0-step11`).
- Exact membership is a *worse* control than the divisibility test, although the two differ in 78% of the non-cyclic cells where the test holds; why membership of *q* in the L-torsion of the group matters more than membership in ⟨b⟩ is not derived here.
- The diagnostics absorb, by construction, the classes of *n* sharing a factor with small primes; the claim is that among the units no residue structure is left.

## 9. Verdict

| statement | verdict |
| --- | --- |
| NDR entropy correlates with ω | yes, through repetend length |
| NDR entropy encodes ω beyond the repetend lengths | **not supported** (−0.07 points, CI [−0.33, +0.24], at 10⁵–10⁶) |
| the "echo" grows with *n* | not supported; the step-5 drift disappears with 98 orders |
| the digits carry factorization traces | yes: Midy's symmetry and exact spectral zeros — both classical, both redundant given the orders |
| the order tests recover ω | certified below 3 700 000 259 and observed on every sampled integer up to 10¹²; the explicit counterexample bounds the first failure from above |
| residue-class structure of the entropy | group structure: ±q ∈ ⟨b⟩ and ord q \| ord b among the units, small-prime divisibility of *n* otherwise; nothing left (§8) |
| law constants, universal base invariance, ζ/PNT links, prime-power dichotomy, three-feature sufficiency | not supported |

The strongest evidence concerns the structure of multiplicative orders and specific symmetries of the digits. The entropy results delimit what the original statistic captures.

## 10. Limits of these conclusions

- **One seed.** All runs use seed 0. Steps 7–11 average gradient-boosting fits over 3–5 model seeds, because a single fit varies by up to half a point; the step-2 to step-5 C9 numbers are single fits.
- **Samples, not all integers.** NDR data covers every n ≤ 150 000 (`full-s0`) and ω-balanced samples of 3 000 per class per decade to 10⁶ (`scale-s0`); orders-only results reach 10⁸ on the same kind of samples. Rare classes are over-represented relative to their natural frequency, and ω ≥ 5 is pooled.
- **Bases.** Entropy and zeros are measured in 19 bases (v8, v10, primes < 32); orders in up to 498. A larger digit base set was not tried, since the orders that would be its baseline would then also be larger.
- **Evolutionary search** found the known structure (Fermat's test, log(n/λ(n))) and no new formula; the differences between Evolvo guides, including the PyTorch policy guide, are not significant over 5 seeds (`full-s0-step4`).
- **Finite scope of the order tests.** The certificate concerns recovery of $\omega(n)$ below 3 700 000 259 (`failures-s0`). The first failure occurs at or before 357 522 227 790. The 40 000 full factorizations checked in `validate-s0` describe the constructed input families in that run.
- **Owner review.** The revision-9 draft and the step 6–11 reports await the author's review.

## 11. Prime-order follow-up

Steps 17–19 count primes satisfying joint divisibility conditions on the indices $(p-1)/\operatorname{ord}_p(b)$ and compare two independent arithmetic implementations. They recover and replay GFSL formulas for four exact rules. The cubic joint event gives both outcomes within every represented residue class modulo 120 on held-out primes. The finite benchmarks link the repetend indices to established prime-splitting counts and delineate what small congruences can predict. [Main run](results/prime-orders-s0/report.md) · [Fresh quadratic holdout](results/prime-orders-confirm-s10/report.md) · [Research roadmap](../../ROADMAP.md#next-research-joint-distribution-of-repetend-lengths)

## 12. Full periods and L-function zeros

Step 20 identifies full-period repetends with Artin's primitive-root problem. Through $10^8$ the counts for all 19 bases agree with Hooley's GRH-conditional densities. That agreement covers the Lehmer–Heilbronn correction for bases 5, 13, 17 and 29 and the index law for $t\le12$. Evolvo recovers Artin's criterion, the nesting behind the factor $\tfrac{20}{19}$ and the Kummer degree $q(q-1)$. Step 21 shows that the halved-repetend counts for bases 2, 3, 5 and 10 are, after smoothing, the explicit-formula sums over the zeros of $L(s,\chi_\Delta)$ to $10^{-13}$. It also computes the Rubinstein–Sarnak bias from those zeros. These are verified consequences of known theorems and stated hypotheses; they say nothing new about RH. [Artin run](results/artin-s0/report.md) · [Explicit-formula run](results/explicit-s0/report.md) · [What this means for RH](../../conclusion.md#10-what-this-means-for-the-riemann-hypothesis)

## 13. Digits of n and prime densities

Step 22 (`digits-s0`, `digits-confirm-s1`, `config/digits.json` and `config/digits-confirm.json`) runs Evolvo on three kinds of input: the decimal digit statistics of $n$ itself, the repetend lengths $L_2, L_{10}$ of $1/n$, and prime-density variables.

From the repetend lengths, Evolvo selects Fermat's test in base 2. Its 93 errors in $(10^6,2\cdot10^6]$ are exactly the base-2 pseudoprimes coprime to 10.

From the digits, primality is bounded by the sieve density $W(n)$ over the primes that digit sums can see, 2, 3, 5, 7, 11, 13 and 37:
- $W(n)$ itself: Brier 0.0439, predicting 70 428 primes against 70 435.
- Evolvo alone: Brier 0.0593, from a last-digit rule.
- Evolvo with $W(n)$ as a variable: keeps $W(n)$ unchanged.

With $\mathrm{li}(x)$ and $\mathrm{li}(\sqrt x)$ as variables, Evolvo writes $\mathrm{li}(x)/\sqrt{1+\mathrm{li}(\sqrt x)\log x/x}$. Its maximum relative error on $(1.3\cdot10^8,10^{12}]$ is $1.26\cdot10^{-5}$, against $1.31\cdot10^{-5}$ for $R(x)$. From $\log x$ alone the simplest candidate is $x(\log x-1)/(\log x(\log x-2))$, with error $3.8\cdot10^{-4}$, against $1.6\cdot10^{-3}$ for Legendre's formula.

The mean gap is $\log p$. The best $\omega$ formula is the sieve mean $M$ plus $e^{-M}$. The next prime repeats the last digit of $p$ 17.3% of the time in $(10^7,10^8]$.

The GFSL programs, the controls and the selection caveats are in [`conclusion.md` §11](../../conclusion.md#11-digits-of-n-prime-densities-and-practical-conclusions).
