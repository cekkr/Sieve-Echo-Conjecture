# Sieve Echo v11 — research report (prime-orders-confirm-s10)

- **Run:** `prime-orders-confirm-s10`, config `empirical-scripts/11.0/config/prime-orders-confirm.json` (sha256 `1469018ee6`), seed 10.
- **Code:** root `346dc45b7c` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `c35be3ca7a`.

## Verdicts

Analyses for this run are below.

## Step 17 — Quadratic order indices

Exact orders agree prime by prime with quadratic reciprocity. Each pattern has density 1/8. Pattern bits refer to bases 2, 3 and 5 in that order.

| Through x | Eligible primes | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 1,226 | 160 | 151 | 154 | 158 | 153 | 155 | 150 | 145 |
| 100,000 | 9,589 | 1,197 | 1,213 | 1,206 | 1,190 | 1,207 | 1,201 | 1,202 | 1,173 |
| 1,000,000 | 78,495 | 9,818 | 9,815 | 9,825 | 9,816 | 9,814 | 9,830 | 9,828 | 9,749 |
| 2,000,000 | 148,930 | 18,596 | 18,623 | 18,696 | 18,644 | 18,606 | 18,642 | 18,624 | 18,499 |

The JSON records interval counts, centered count and log-weighted discrepancies, and the seven character sums with the prime-power corrections.

## Step 18 — Cubic order indices

The joint count has unconditional density 1/18 and conditional density 1/9 among primes congruent to 1 modulo 3. The last column assumes GRH for the Dedekind zeta function of the joint field. [Grenié–Molteni, Corollary 1.2](https://arxiv.org/abs/1709.07609)

| Through x | Base 2 | Base 3 | Joint | Conditional joint rate | Joint minus conditional independence | Joint minus Li₂(x)/18 | GRH error bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 200 | 191 | 58 | 9.493% | -4.52 | -11.17 | 300.84 |
| 100,000 | 1,559 | 1,582 | 516 | 10.786% | +0.46 | -18.93 | 857.92 |
| 1,000,000 | 13,032 | 13,044 | 4,318 | 11.007% | -15.04 | -50.14 | 2,546.59 |
| 2,000,000 | 24,797 | 24,716 | 8,181 | 10.994% | -55.34 | -99.77 | 3,553.27 |

The unconditional theorem gives an effective large-x estimate with a possible exceptional-zero term. The JSON records its range and parameters. [Thorner–Zaman, Theorem 1.1](https://doi.org/10.2140/ant.2019.13.1039)

Through 2,000,000, all 7,414 primes with `5 divides I_2` also satisfy `2 divides I_5`. The joint count is 7,414.

## Step 19 — Evolvo formulas

Training uses primes through 200,000. Selection uses validation through 1,000,000. Held-out scores use larger primes. The finite domain checks use the specified residue or Boolean input tables.

| Target | Selected formula | Test MSE | Constant MSE | Test F1 | Exact domain values |
| --- | --- | --- | --- | --- | --- |
| quadratic_base2 | `([4] % [abs((p mod 8) − 4)])` | 0 | 0.25 | 1.000 | 4/4 |
| quadratic_joint | `(([2 divides I_2] * [2 divides I_5]) * [2 divides I_3])` | 0 | 0.1088 | 1.000 | 16/16 |

The JSON includes a seeded shuffle comparison and replayable GFSL instructions.

Full numbers: `prime_orders.json` in this run directory.
