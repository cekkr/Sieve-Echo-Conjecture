# Sieve-Echo-Conjecture

An empirical study of the **Sieve Echo Conjecture**: the idea that the prime-factorization complexity of an integer *n* — chiefly ω(n), the number of distinct prime factors — is encoded in the spectral entropy of the repeating digits of 1/*n* written in several bases (normalized digit representation, NDR). The project grew out of questions about the Riemann zeta function and the distribution of primes.

The repository holds numbered script generations (`empirical-scripts/3.1` … `11.0`) and the papers in `studies/`. Version 11 ([`empirical-scripts/11.0/`](empirical-scripts/11.0)) is the current, reproducible, leakage-free implementation: one entry point, held-out evaluation, controls for repetend length and size, permutation nulls, and evolutionary search through the [Evolvo](libs/evolvo) submodule. The main result — an exact decoder of the factorization from the repetends, how it was found and why it works — is in [`conclusion.md`](conclusion.md); the full v11 conclusions are in [`empirical-scripts/11.0/conclusions.md`](empirical-scripts/11.0/conclusions.md); the working paper is [`studies/9/sieve-echo-conjecture-9.md`](studies/9/sieve-echo-conjecture-9.md) (draft for the author's review); the plan and the full results log are in [`ROADMAP.md`](ROADMAP.md); orientation for contributors and AI agents is in [`AGENTS.md`](AGENTS.md).

## Main result: the echo is exact

The repetends of 1/*n* in 19 bases (2–7, 9–17, 19, 23, 29, 31) determine the factorization of *n*. A training-free decoder reads ω(*n*) and the prime factors from the repetend lengths and a few digits. It is correct for **every integer below 3 700 000 259** — exhaustively to 10⁸ (`decode-s0`) and certified beyond by a proof of where it can fail (`failures-s0`) — and for all 276 506 sampled integers up to 10¹². On the same integers the best learned model of this project reached 94–99%. 3.7·10⁹ is where the certificate stops, not a failure: with all 19 bases no failing *n* is known. The full account is in [`conclusion.md`](conclusion.md).

**How it was found.** Each ROADMAP step followed the information to where it actually was:

1. The digits' spectral entropy turned out to be a function of the repetend length L = ord_n(b).
2. The lengths carry ω. As Proposition 1.6 shows, ω is the 2-rank of (ℤ/nℤ)*, which the 2-adic valuations of the lengths reveal.
3. Reading those valuations prime by prime, rather than statistically, is the reduction of factoring to order finding used by Miller and Shor. The data confirmed the link: Shor's half-period split fails exactly on Midy's pairs.

**How it works.** For each base *b* coprime to *n* and each prime q | L:

1. Compute gcd(b^{(L/q^e)·q^i} − 1, n) for i < e = v_q(L). Each remainder b^d mod n is read from about log_b *n* repetend digits.
2. Refine *n* by these gcds, and by gcd(b, n) for the bases that share a factor with *n*.
3. The coprime parts that remain are the prime-power factors.

**Why it works so well.**
- **Every test is exact.** p | b^m − 1 exactly when ord_p(b) | m, so two prime factors are separated as soon as one base gives them different orders.
- **The decoder fails only in a narrow case.** It fails only if two prime factors share their orders in every base coprime to *n* (Proposition 1.8). Modulo the 5 761 444 primes below 10⁸, no two primes do so in all 19 bases.
- **The known failures need a lot.** With fewer bases, each failure needs small prime factors that remove most bases, plus a coincidence among the rest, such as 23·89 = 2¹¹ − 1. With five bases the decoder is already exact for 90–97% of integers, more as *n* grows.

**It is not a fast factoring method.** Computing the lengths classically costs as much as factoring; Shor's algorithm computes them quantumly. So the result shows where the information is, not a shortcut to it.

## Final results (v11, seed 0)

Every number names the run that produced it; run summaries, manifests and figures are committed under `empirical-scripts/11.0/results/<run-id>/`.

1. **Every statistic of the v8 paper reproduces exactly** from its own code path (law fit α = −0.4681, β = 1.9174, R² = 0.2345; 6% base invariance; ζ and PNT correlations; residue-class variances) — and each is explained by repetend length, by the size of *n*, by construction, or by a 100-integer sample (`full-s0-claims`, `full-s0-step4`).
2. **NDR entropy is a function of repetend length** L = ord_n(b): log L and log n explain 95.9% of its held-out variance; log ω adds 0.04% (`full-s0-step4`). Primes have the *highest* entropy. The negative slope of the "law" follows from λ(n) ≤ φ(n)/2^{ω−1} (Theorem 1.3 of the paper).
3. **The factorization information lives in the lengths.** Gradient boosting on *n* and the per-base lengths with their lcm arithmetic predicts ω classes at 98.2% held-out accuracy; evolved formulas recognize primes at 99.99% from "lcm of the lengths divides n − 1" — Fermat's little theorem in repetend form (`full-s0-step4`).
4. **The apparent residual signal at 10⁵–10⁶ disappears** with a stronger baseline: entropy adds −0.07 accuracy points (95% CI [−0.33, +0.24]) once the orders in all bases 2…99 and the Midy flags are known (`scale-s0-step7`).
5. **The digits do echo the factorization** — through Midy's symmetry (when −1 ∈ ⟨b⟩ mod n the DC-free entropy drops by exactly log 2, fitted −0.694; `scale-s0-step7`) and through **exact spectral zeros** whose Euler factors vanish at the prime divisors of n (0.09% of prime pairs vs 11–30% of composite pairs; `scale-s0-step8`). Both are classical number theory, and neither adds information about ω beyond the multiplicative orders (zeros: −0.05 points, CI [−0.28, +0.18]).
6. **Residue-class structure is group structure:** whether small primes q (and −q) lie in the subgroup ⟨b⟩ mod n, or have orders dividing ord_n(b), removes the residue structure among the units — the odd moduli 3, 5, 7 keep no excess — and what is left follows which small primes divide n and the power of 2 in n (mod 12 excess 0.022 → 0.002, mod 210 0.032 → 0.003 with those labelled diagnostics; `scale-s0-step9`, `scale-s0-step14`). Exact membership is, surprisingly, a weaker control than the divisibility test.
7. **Orders alone, up to 10⁸:** classifying ω from the orders in bases 2…199 gives 97.9%, 97.3%, 94.5%, 89.7% held-out accuracy in the decades 10⁴…10⁸; a fixed base set reads less as n grows, and more bases recover part of it (`orders-s0`).
8. **ω is the 2-rank of the unit group:** ω(n) = rank₂ (ℤ/nℤ)* − δ(v₂(n)) + [n even], and the repetend lengths reveal that rank through the 2-adic valuations of the orders. Training-free estimators built on it are right for 69–82% (ratio of counts) and 72–83% (maximum likelihood) of integers from 10⁴ to 10⁸ using 498 bases; the 2-adic and 3-adic profiles of the orders add +3.2–3.5 accuracy points at 10⁷–10⁸ to a model that already has the raw lengths of the same bases (94.2% vs 90.6%; `step11-s0`) — information a model fed log-lengths cannot extract by itself.
9. **The repetends determine the factorization — exactly.** For each of 19 bases and each prime q dividing the repetend length L, gcd(n, b^{L/qⁱ} − 1) splits *n*; with no training this gives the exact ω(*n*) and the prime factors for **every integer below 3.7·10⁹** — exhaustively to 10⁸ (9 exceptions from the lengths alone, none once the pre-periods of non-coprime bases are used), and certified beyond because no two primes below 10⁸ share their orders in all 19 bases (`failures-s0`) and for all 276 506 sampled integers up to 10¹² — including the 10⁴–10⁸ integers on which the best learned model classified ω at 94–99% (`decode-s0`). The remainders b^d mod n are read from about log_b n repetend digits. It is the reduction of factoring to order finding that Shor's algorithm uses — its period is the repetend length — and Shor's half-period split fails exactly on Midy's pairs.
10. **SinWhisky spectra** (the [circle-based resampler](libs/sinwhisky) applied before the DFT) carry the same information as the plain FFT (correlation 0.9998, no extra ω information; `scale-s0-step8e`).

## Conclusions (v11)

The full statement, with tables and limits, is [`empirical-scripts/11.0/conclusions.md`](empirical-scripts/11.0/conclusions.md). In short:

- **The echo is exact.** The repetends of 1/*n* in 19 bases are a complete certificate of the factorization of *n*: lengths plus a few digits at fractions of the period give ω(*n*) and the prime factors of every *n* < 3.7·10⁹ and of every sampled *n* up to 10¹² (`decode-s0`, `failures-s0`). Computing those lengths classically costs as much as factoring — Shor's algorithm computes them quantumly — so this locates the information; it is not a shortcut.
- **The conjecture's statistic misses it.** NDR entropy correlates with ω only through repetend length; against a baseline of the orders in bases 2…99 and their Midy flags it adds −0.07 accuracy points (95% CI [−0.33, +0.24]) at 10⁵–10⁶, and the apparent growth of the signal with *n* was an under-powered 19-base baseline (`scale-s0-step7`).
- **Why the lengths know ω.** The repetend length ord_n(b) is the order of *b* in (ℤ/nℤ)*; ω(n) is, up to a correction for the power of 2 in *n*, the 2-rank of that group (Proposition 1.6), and the prime factors themselves are separated by their order vectors (Proposition 1.7).
- **The digits carry two exact traces of the factorization** — Midy's symmetry (a log 2 entropy drop when −1 ∈ ⟨b⟩) and exact spectral zeros at the Euler factors of the prime divisors — both classical, both already contained in the orders.
- **The residue-class structure is group structure** (step 14): nothing about residue classes is left once subgroup flags and the small-prime divisibility of *n* are controlled.
- **Every v8 statistic reproduces and none is an ω signal:** each is explained by repetend length, the size of *n*, a 100-integer sample, a correlation of two factorization labels, or training-set accuracy.
- **Limits:** one seed (0); ω-balanced samples rather than every integer above 150 000; entropy measured in 19 bases. **Open:** the density of decoder failures above 3.7·10⁹ (they are characterized exactly: two prime factors with the same orders in every base coprime to *n*), and why order divisibility explains the residue structure better than exact subgroup membership.

## Reproduce

```bash
git submodule update --init --recursive
/opt/homebrew/bin/python3.15 -m venv .venv
.venv/bin/pip install --only-binary :all: -r empirical-scripts/11.0/requirements.txt
.venv/bin/python -m unittest discover -s empirical-scripts/11.0/tests
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-s0
```

The full and scale configurations and their run times are described in [`AGENTS.md`](AGENTS.md) and §9 of the paper.

## Credits

The Sieve Echo Conjecture, the earlier script generations and the papers are the work of the repository author. [Evolvo](libs/evolvo) and [SinWhisky](libs/sinwhisky) are separate projects included as git submodules.

Version 11 — the reproducible pipeline, the claim-by-claim re-verification, the Evolvo and SinWhisky extensions it needed, the analyses of ROADMAP steps 0–15, the revision-9 draft, the v11 conclusions and the exact decoder of [`conclusion.md`](conclusion.md) — was developed in collaboration with **Claude (Anthropic)**, an AI assistant, working with the author.
