# Sieve Echo Conjecture

Sieve Echo studies how the arithmetic of the repeating digits of $1/n$ reflects the structure of the multiplicative group modulo $n$. The project began by testing whether digit spectral entropy reveals the number of distinct prime factors. Version 11 follows the arithmetic through the repetend length, which is the multiplicative order $\operatorname{ord}_n(b)$ in each base $b$ coprime to $n$. It now includes checked prime counts and symbolic formulas for joint conditions on those orders. At prime denominators it ends at two classical problems. Full-period repetends are Artin's primitive-root problem, and halved repetends are counted by the zeros of Dirichlet L-functions.

## What v11 established

- **Repetend length explains most of the entropy signal.** Length and integer size explain 95.9% of held-out entropy variance (`full-s0-step4`). With orders in all bases 2 through 99 and their Midy symmetry flags, adding entropy changes accuracy by −0.07 percentage points at $10^5$–$10^6$, with a 95% confidence interval of [−0.33, +0.24] (`scale-s0-step7`).
- **The digits exhibit exact arithmetic structure.** Midy symmetry and whole classes of spectral zeros have explanations in modular orders and Dirichlet characters. The character expansion involves specific Euler factors of $L(1,\chi)$. [Revision 9, Propositions 1.4 and 1.5](studies/9/sieve-echo-conjecture-9.md#1-definitions)
- **Orders distinguish prime factors in a certified range.** A fixed set of 19 small bases, with gcd checks for noncoprime bases, separates the distinct prime factors for $2\le n<3{,}700{,}000{,}259$ (`failures-s0`). For squarefree integers, failures occur when two remaining primes have the same orders in every available base. At $n=357{,}522{,}227{,}790$, the decoder leaves $1187\cdot3559$ combined. [Proof and reproduction](conclusion.md#4-why-the-order-tests-work)

- **Full-period repetends follow Hooley's GRH densities.** Through $10^8$, the share of primes $p$ for which $1/p$ repeats with the maximal length $p-1$ matches the conditional density for all 19 bases. Base 5 is 98 standard deviations from Artin's original constant and 0.65 from the Lehmer–Heilbronn corrected value $\tfrac{20}{19}A$ (`artin-s0`). [Section 8 of the conclusion](conclusion.md#8-full-period-repetends-and-artins-conjecture)
- **Halved-repetend counts are sums over L-function zeros.** A period dividing $(p-1)/2$ in base $b$ is the quadratic character of $\mathbb Q(\sqrt b)$. After smoothing, the log-weighted counts through $10^8$ equal the sum over the zeros of $L(s,\chi_\Delta)$ up to height 150 to $10^{-13}$ for bases 2, 3, 5 and 10. The same zeros give a Chebyshev bias toward unhalved periods (`explicit-s0`). [Section 9 of the conclusion](conclusion.md#9-zeros-behind-the-halved-repetend-counts)

The decoder applies the classical connection between periods and factorization. Its role here is to show how much the order data capture. [Miller, Theorem 4](https://www.cs.cmu.edu/~glmiller/Publications/Papers/Mi76.pdf)

## Prime-order benchmarks

For a prime $p$ and a base $b$ with $p\nmid b$, the index $I_b(p)=(p-1)/\operatorname{ord}_p(b)$ measures how much shorter the repetend is than its maximum possible length. Divisibility conditions on these indices turn the digit question into a prime-counting question. For example, $2\mid I_b(p)$ means that $b$ is a quadratic residue modulo an odd prime $p$. Imposing the condition for several bases counts primes in specified residue classes.

For higher prime divisors of the index, the same construction leads to primes splitting in explicit number fields. The cubic benchmark finds 4,318 joint hits through $10^6$, or 11.007% after conditioning on $p\equiv1\pmod3$, against the density $1/9$. The count differs from its integral main term by 50.14, inside the GRH-conditional error bound of 2,546.59. This comparison measures a finite discrepancy for a specified field. A consequence for ordinary RH would require a theorem connecting it to the zeros of $\zeta(s)$. [Arithmetic and bounds](ROADMAP.md#next-research-joint-distribution-of-repetend-lengths) · [Benchmark report](empirical-scripts/11.0/results/prime-orders-s0/report.md)

## Full-period repetends and the zeros of L-functions

The repetend of $1/p$ in base $b$ has the full length $p-1$ exactly when $b$ is a primitive root modulo $p$. Hooley proved Artin's density for these primes assuming GRH for the Kummer fields $\mathbb Q(\zeta_n,b^{1/n})$.

| base | full-period primes ≤ $10^8$ | rate / Artin's constant | GRH density / A |
| --- | --- | --- | --- |
| 2 | 2 154 727 | 1.00009 | 1 |
| 10 | 2 155 283 | 1.00035 | 1 |
| 5 | 2 268 685 | 1.05299 | 20/19 = 1.05263 |
| 13 | 2 168 390 | 1.00644 | 1.00645 |

The index law $I_b(p)=t$ for $t\le12$ also matches its Kummer-degree densities for all 16 non-square bases. That includes the density zero at $I_5(p)=5$, an index no prime takes. Unconditionally, Heath-Brown proved only that one of 2, 3, 5 is a primitive root for infinitely many primes. [Step-20 report](empirical-scripts/11.0/results/artin-s0/report.md)

The halved repetends are the primes with $\chi_\Delta(p)=1$, so their imbalance $\psi(x,\chi_\Delta)$ obeys the explicit formula of $L(s,\chi_\Delta)$.

| L-function | base | zeros ≤ 150 | first zero | smoothed residual | bias δ (GRH + LI) | measured share |
| --- | --- | --- | --- | --- | --- | --- |
| $L(s,\chi_8)$ | 2 | 102 | 4.899974 | $1.2\cdot10^{-13}$ | 0.9822 | 0.9748 |
| $L(s,\chi_{12})$ | 3 | 111 | 3.804628 | $9.1\cdot10^{-14}$ | 0.9602 | 0.9215 |
| $L(s,\chi_5)$ | 5 | 90 | 6.648453 | $2.9\cdot10^{-13}$ | 0.9954 | 1.0000 |
| $L(s,\chi_{40})$ | 10 | 140 | 2.488210 | $1.0\cdot10^{-13}$ | 0.8778 | 0.8107 |
| $\zeta(s)$ | full periods | 52 | 14.134725 | $6.9\cdot10^{-10}$ | – | – |

Without the zeros the residuals are 0.2–1.6. The measured share is the logarithmic proportion of $x\le10^8$ with more unhalved than halved primes. In base 2, halved periods first lead at $p=11\,100\,143$; in base 5 they never lead below $10^8$. [Step-21 report](empirical-scripts/11.0/results/explicit-s0/report.md)

## What this means for the Riemann hypothesis

The results do not prove the Riemann hypothesis, and they are not evidence for it beyond what is already known.

- **Already known far beyond this range.** The zeros of $\zeta$ are verified on the critical line to height $3\cdot10^{12}$, and GRH is verified for every Dirichlet L-function of conductor up to 400 000.
- **Reformulations, not new routes.** The repetend observables are exact rewrites of prime counts in residue classes and splitting fields. RH, for example, is equivalent to a $\sqrt x$-type bound on $\theta(x)$, the log-weighted count of the $n\le x$ that have a full-period repetend in some base. The largest ratio to Schoenfeld's RH bound on $[599,10^8]$ is 0.9836. Such a bound is exactly as hard to prove as RH itself.

What the repository does provide:
- a reproducible chain from the original digit-entropy question to Artin's conjecture and the explicit formula;
- a benchmark with independent checks;
- a clear separation of identities, conditional theorems and finite computations. [Section 10 of the conclusion](conclusion.md#10-what-this-means-for-the-riemann-hypothesis)

## Best Evolvo algorithms

Let $D_{m,b}$ be 1 when $m$ divides $I_b(p)$ and 0 otherwise. For prime $p$ outside the divisors of $mb$, compute this bit by testing $m\mid p-1$ and then testing whether $b^{(p-1)/m}\equiv1\pmod p$.

| Target | Selected Evolvo formula | Held-out errors |
| --- | --- | --- |
| Quadratic bit for base 2 | `4 % abs((p % 8) - 4)` | 0 / 70,435 |
| Joint quadratic event for bases 2, 3, 5 | $D_{2,2}D_{2,5}D_{2,3}$ | 0 / 70,435 |
| Joint cubic event for bases 2, 3 | $D_{3,2}D_{3,3}$ | 0 / 60,514 |
| Base-2 quintic and base-5 quadratic event | $\lvert D_{5,2}\rvert$ | 0 / 60,514 |
| Base-10 full period (Artin's criterion) | $N_5\,N_2\,N_{\ge11}\,N_7\,N_3$, with $N_q=1-D_{q,10}$ | 0 / 515,646 |
| Base-5 nesting behind the factor 20/19 | $\lvert D_{5,5}\rvert$ for $D_{10,5}$ | 0 / 515,646 |
| Kummer degree from counts | $q/(q-1)=q^2\Pr[q\mid I_b(p)]$, so the degree is $q(q-1)$ | held-out χ²/point 1.21 |

The quadratic rows are from `prime-orders-confirm-s10`, with test primes beyond the first run's range. The cubic and mixed rows are from `prime-orders-s0` and use the individual order predicates as inputs. The last three rows are from `artin-s0`. There, $N_{\ge11}$ means that no prime $q\ge11$ divides $I_{10}(p)$. The Kummer row is a regression over the 79 primes $q\le401$, with standardized residuals against counting noise. All four formulas also return exact values on their finite input tables. For the cubic joint event using only congruences modulo 120, the best evolved formula was the constant $1/9$, with F1 of zero. Both outcomes occur in every one of the 16 represented residue classes, forcing at least 3,342 errors among 30,243 held-out primes for any deterministic rule using those residues. The base-10 full period behaves the same way modulo 840. Both outcomes occur in 96 of 192 classes, so any congruence rule makes at least 64,411 errors, and the best evolved congruence formula reaches 82.5% accuracy. [Main run and replayable programs](empirical-scripts/11.0/results/prime-orders-s0/prime_orders.json) · [Quadratic follow-up and replayable programs](empirical-scripts/11.0/results/prime-orders-confirm-s10/prime_orders.json) · [Earlier Evolvo results for primality and ω](conclusion.md#21-what-evolvo-found-for-comparison)

**Milestone complete.** Steps 17–21 have verified counts, replayable programs, checked L-function zeros and documented limits. The original entropy experiments, the decoder and the prime-order benchmarks each have a reproducible record. [Conclusion and executable formulas](conclusion.md#7-best-evolvo-algorithms)

## Repository guide

- [Research conclusion](conclusion.md) summarizes the results and the role of the decoder.
- [Version 11 conclusions](empirical-scripts/11.0/conclusions.md) give the full experimental record with run IDs.
- [Revision 9 working paper](studies/9/sieve-echo-conjecture-9.md) gives the mathematical definitions and proofs. Earlier papers record the original conjecture.
- [Roadmap](ROADMAP.md) gives the next experiments and the completed v11 run log. [AGENTS.md](AGENTS.md) documents the code and contributor workflow.

## Reproduce

```bash
git submodule update --init --recursive
/opt/homebrew/bin/python3.15 -m venv .venv
.venv/bin/pip install --only-binary :all: -r empirical-scripts/11.0/requirements.txt
.venv/bin/python -m unittest discover -s empirical-scripts/11.0/tests
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-s0
```

Run the prime counts and both symbolic searches:

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/prime-orders.json --run-id prime-orders-s0-repro --stages prime_orders,report
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/prime-orders-confirm.json --run-id prime-orders-confirm-s10-repro --stages prime_orders,report
```

Run the full-period benchmark with its Evolvo cases, then the L-function zeros and the explicit formula. Each takes a few minutes:

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/artin.json --run-id artin-s0-repro --stages artin,report
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/explicit.json --run-id explicit-s0-repro --stages explicit,report
```

The full and scale configurations for the original conjecture are described in [`AGENTS.md`](AGENTS.md) and §9 of the paper.

## Credits

The Sieve Echo Conjecture, the earlier script generations and the papers are the work of the repository author. [Evolvo](libs/evolvo) and [SinWhisky](libs/sinwhisky) are separate projects included as git submodules.

Version 11 — the reproducible pipeline, the claim-by-claim re-verification, the Evolvo and SinWhisky extensions it needed, the analyses of ROADMAP steps 0–21, the revision-9 draft, the v11 conclusions and the exact decoder of [`conclusion.md`](conclusion.md) — was developed in collaboration with **Claude (Anthropic)**, an AI assistant, working with the author.
