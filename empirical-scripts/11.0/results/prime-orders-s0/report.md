# Sieve Echo v11 — research report (prime-orders-s0)

- **Run:** `prime-orders-s0`, config `empirical-scripts/11.0/config/prime-orders.json` (sha256 `ee96e275cc`), seed 0.
- **Code:** root `346dc45b7c` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `fe2d8ecc32`.

## Verdicts

Analyses for this run are below.

## Step 17 — Quadratic order indices

Exact orders agree prime by prime with quadratic reciprocity. Each pattern has density 1/8. Pattern bits refer to bases 2, 3 and 5 in that order.

| Through x | Eligible primes | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 1,226 | 160 | 151 | 154 | 158 | 153 | 155 | 150 | 145 |
| 100,000 | 9,589 | 1,197 | 1,213 | 1,206 | 1,190 | 1,207 | 1,201 | 1,202 | 1,173 |
| 1,000,000 | 78,495 | 9,818 | 9,815 | 9,825 | 9,816 | 9,814 | 9,830 | 9,828 | 9,749 |

The JSON records interval counts, centered count and log-weighted discrepancies, and the seven character sums with the prime-power corrections.

## Step 18 — Cubic order indices

The joint count has unconditional density 1/18 and conditional density 1/9 among primes congruent to 1 modulo 3. The last column assumes GRH for the Dedekind zeta function of the joint field. [Grenié–Molteni, Corollary 1.2](https://arxiv.org/abs/1709.07609)

| Through x | Base 2 | Base 3 | Joint | Conditional joint rate | Joint minus conditional independence | Joint minus Li₂(x)/18 | GRH error bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 200 | 191 | 58 | 9.493% | -4.52 | -11.17 | 300.84 |
| 100,000 | 1,559 | 1,582 | 516 | 10.786% | +0.46 | -18.93 | 857.92 |
| 1,000,000 | 13,032 | 13,044 | 4,318 | 11.007% | -15.04 | -50.14 | 2,546.59 |

The unconditional theorem gives an effective large-x estimate with a possible exceptional-zero term. The JSON records its range and parameters. [Thorner–Zaman, Theorem 1.1](https://doi.org/10.2140/ant.2019.13.1039)

Through 1,000,000, all 3,912 primes with `5 divides I_2` also satisfy `2 divides I_5`. The joint count is 3,912.

## Step 19 — Evolvo formulas

Training uses primes through 100,000. Selection uses validation through 200,000. Held-out scores use larger primes. The finite domain checks use the specified residue or Boolean input tables.

| Target | Selected formula | Test MSE | Constant MSE | Test F1 | Exact domain values |
| --- | --- | --- | --- | --- | --- |
| quadratic_base2 | `([1] % (2.71828 / [p mod 8]))` | 0.20531 | 0.25 | 0.666 | 1/4 |
| quadratic_joint | `(0.5 * (abs([2 divides I_3]) * [2 divides I_2]))` | 0.06249 | 0.10919 | 0.000 | 12/16 |
| cubic_joint | `([3 divides I_2] * [3 divides I_3])` | 0 | 0.052179 | 1.000 | 8/8 |
| mixed_joint | `abs([5 divides I_2])` | 0 | 0.047634 | 1.000 | 6/6 |
| cubic_congruence_only | `(1 / (10 - 1))` | 0.098294 | 0.098301 | 0.000 | – |

The cubic target takes both values in 16 of 16 held-out residue classes modulo 120. Any deterministic classifier of these residues makes at least 3,342 errors on these held-out primes. The JSON includes the train-only lookup.

The JSON includes a seeded shuffle comparison and replayable GFSL instructions.

Full numbers: `prime_orders.json` in this run directory.
