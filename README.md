# Sieve Echo Conjecture

Sieve Echo studies how the arithmetic of the repeating digits of $1/n$ reflects the structure of the multiplicative group modulo $n$. The project began by testing whether digit spectral entropy reveals the number of distinct prime factors. Version 11 follows the arithmetic through the repetend length, which is the multiplicative order $\operatorname{ord}_n(b)$ in each base $b$ coprime to $n$. It now includes checked prime counts and symbolic formulas for joint conditions on those orders.

## What v11 established

- **Repetend length explains most of the entropy signal.** Length and integer size explain 95.9% of held-out entropy variance (`full-s0-step4`). With orders in all bases 2 through 99 and their Midy symmetry flags, adding entropy changes accuracy by −0.07 percentage points at $10^5$–$10^6$, with a 95% confidence interval of [−0.33, +0.24] (`scale-s0-step7`).
- **The digits exhibit exact arithmetic structure.** Midy symmetry and whole classes of spectral zeros have explanations in modular orders and Dirichlet characters. The character expansion involves specific Euler factors of $L(1,\chi)$. [Revision 9, Propositions 1.4 and 1.5](studies/9/sieve-echo-conjecture-9.md#1-definitions)
- **Orders distinguish prime factors in a certified range.** A fixed set of 19 small bases, with gcd checks for noncoprime bases, separates the distinct prime factors for $2\le n<3{,}700{,}000{,}259$ (`failures-s0`). For squarefree integers, failures occur when two remaining primes have the same orders in every available base. At $n=357{,}522{,}227{,}790$, the decoder leaves $1187\cdot3559$ combined. [Proof and reproduction](conclusion.md#4-why-the-order-tests-work)

The decoder applies the classical connection between periods and factorization. Its role here is to show how much the order data capture. [Miller, Theorem 4](https://www.cs.cmu.edu/~glmiller/Publications/Papers/Mi76.pdf)

## Prime-order benchmarks

For a prime $p$ and a base $b$ with $p\nmid b$, the index $I_b(p)=(p-1)/\operatorname{ord}_p(b)$ measures how much shorter the repetend is than its maximum possible length. Divisibility conditions on these indices turn the digit question into a prime-counting question. For example, $2\mid I_b(p)$ means that $b$ is a quadratic residue modulo an odd prime $p$. Imposing the condition for several bases counts primes in specified residue classes.

For higher prime divisors of the index, the same construction leads to primes splitting in explicit number fields. The cubic benchmark finds 4,318 joint hits through $10^6$, or 11.007% after conditioning on $p\equiv1\pmod3$, against the density $1/9$. The count differs from its integral main term by 50.14, inside the GRH-conditional error bound of 2,546.59. This comparison measures a finite discrepancy for a specified field. A consequence for ordinary RH would require a theorem connecting it to the zeros of $\zeta(s)$. [Arithmetic and bounds](ROADMAP.md#next-research-joint-distribution-of-repetend-lengths) · [Benchmark report](empirical-scripts/11.0/results/prime-orders-s0/report.md)

## Best Evolvo algorithms

Let $D_{m,b}$ be 1 when $m$ divides $I_b(p)$ and 0 otherwise. For prime $p$ outside the divisors of $mb$, compute this bit by testing $m\mid p-1$ and then testing whether $b^{(p-1)/m}\equiv1\pmod p$.

| Target | Selected Evolvo formula | Held-out errors |
| --- | --- | --- |
| Quadratic bit for base 2 | `4 % abs((p % 8) - 4)` | 0 / 70,435 |
| Joint quadratic event for bases 2, 3, 5 | $D_{2,2}D_{2,5}D_{2,3}$ | 0 / 70,435 |
| Joint cubic event for bases 2, 3 | $D_{3,2}D_{3,3}$ | 0 / 60,514 |
| Base-2 quintic and base-5 quadratic event | $\lvert D_{5,2}\rvert$ | 0 / 60,514 |

The quadratic rows are from `prime-orders-confirm-s10`, with test primes beyond the first run's range. The cubic and mixed rows are from `prime-orders-s0` and use the individual order predicates as inputs. All four formulas also return exact values on their finite input tables. For the cubic joint event using only congruences modulo 120, the best evolved formula was the constant $1/9$, with F1 of zero. Both outcomes occur in every one of the 16 represented residue classes, forcing at least 3,342 errors among 30,243 held-out primes for any deterministic rule using those residues. [Main run and replayable programs](empirical-scripts/11.0/results/prime-orders-s0/prime_orders.json) · [Quadratic follow-up and replayable programs](empirical-scripts/11.0/results/prime-orders-confirm-s10/prime_orders.json) · [Earlier Evolvo results for primality and ω](conclusion.md#21-what-evolvo-found-for-comparison)

**Milestone complete.** Steps 17–19 have verified counts, replayable programs, and documented limits. The original entropy experiments and the prime-order benchmark each have a reproducible record. [Conclusion and executable formulas](conclusion.md#7-best-evolvo-algorithms)

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

The full and scale configurations for the original conjecture are described in [`AGENTS.md`](AGENTS.md) and §9 of the paper.

## Credits

The Sieve Echo Conjecture, the earlier script generations and the papers are the work of the repository author. [Evolvo](libs/evolvo) and [SinWhisky](libs/sinwhisky) are separate projects included as git submodules.

Version 11 — the reproducible pipeline, the claim-by-claim re-verification, the Evolvo and SinWhisky extensions it needed, the analyses of ROADMAP steps 0–16, the revision-9 draft, the v11 conclusions and the exact decoder of [`conclusion.md`](conclusion.md) — was developed in collaboration with **Claude (Anthropic)**, an AI assistant, working with the author.
