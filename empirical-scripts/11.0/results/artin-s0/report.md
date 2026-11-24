# Sieve Echo v11 — research report (artin-s0)

- **Run:** `artin-s0`, config `empirical-scripts/11.0/config/artin.json` (sha256 `4af31adaff`), seed 0.
- **Code:** root `e68d4397ff` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `fae15e3c1b`.

## Verdicts

Analyses for this run are below.

## Step 20 — Full-period repetends (Artin's conjecture)

Primes 37…99,999,989 (5,761,444; primes up to 31 excluded). Artin's constant A = 0.373955813619. Index parity equals quadratic reciprocity for every prime and base; SymPy agrees on the orders through 30,000 and on 4,000 sampled primes.

The density column is Hooley's, which assumes GRH for the Kummer fields Q(ζ_n, b^{1/n}).

| Base | Δ | Full-period primes | Rate / A | GRH density / A | Binomial-scale z |
| --- | --- | --- | --- | --- | --- |
| 2 | 8 | 2,154,727 | 1.00009 | 1.00000 | +0.17 |
| 3 | 12 | 2,154,027 | 0.99977 | 1.00000 | -0.43 |
| 4 | 1 | 0 | 0.00000 | 0.00000 | – |
| 5 | 5 | 2,268,685 | 1.05299 | 1.05263 | +0.65 |
| 6 | 24 | 2,154,460 | 0.99997 | 1.00000 | -0.06 |
| 7 | 28 | 2,155,993 | 1.00068 | 1.00000 | +1.26 |
| 9 | 1 | 0 | 0.00000 | 0.00000 | – |
| 10 | 40 | 2,155,283 | 1.00035 | 1.00000 | +0.65 |
| 11 | 44 | 2,155,229 | 1.00033 | 1.00000 | +0.61 |
| 12 | 12 | 2,154,290 | 0.99989 | 1.00000 | -0.20 |
| 13 | 13 | 2,168,390 | 1.00644 | 1.00645 | -0.03 |
| 14 | 56 | 2,154,482 | 0.99998 | 1.00000 | -0.04 |
| 15 | 60 | 2,154,711 | 1.00009 | 1.00000 | +0.16 |
| 16 | 1 | 0 | 0.00000 | 0.00000 | – |
| 17 | 17 | 2,162,256 | 1.00359 | 1.00369 | -0.19 |
| 19 | 76 | 2,154,280 | 0.99989 | 1.00000 | -0.21 |
| 23 | 92 | 2,155,668 | 1.00053 | 1.00000 | +0.98 |
| 29 | 29 | 2,157,549 | 1.00140 | 1.00123 | +0.32 |
| 31 | 124 | 2,155,257 | 1.00034 | 1.00000 | +0.63 |

Bases whose discriminant is 1 mod 4 (the Lehmer–Heilbronn correction):

| Base | Rate / A | Corrected density / A | z against A | z against the corrected density |
| --- | --- | --- | --- | --- |
| 5 | 1.05299 | 1.05263 | +98.3 | +0.65 |
| 13 | 1.00644 | 1.00645 | +11.9 | -0.03 |
| 17 | 1.00359 | 1.00369 | +6.7 | -0.19 |
| 29 | 1.00140 | 1.00123 | +2.6 | +0.32 |

Index law I_b(p) = t, t ≤ 12, for the 16 non-square bases: largest binomial-scale |z| = 2.14. Base 5 has density 0 at t = 5, and no prime has I_5(p) = 5. Both 2 and 3 primitive: rate 0.147315 against 0.147349. At least one of 2, 3, 5 primitive: 0.7396 of the primes (Heath-Brown: infinitely many, unconditionally).

### Evolvo formulas

| Target | Selected formula | Test MSE | Test F1 | Exact domain values |
| --- | --- | --- | --- | --- |
| artin_criterion | `([5 ∤ I_10] * ((([2 ∤ I_10] * [no prime q ≥ 11 divides I_10]) * [7 ∤ I_10]) * [3 ∤ I_10]))` | 0 | 1.000 | 64/64 |
| lehmer_nesting | `abs([5 divides I_5])` | 0 | 1.000 | 6/6 |
| artin_congruence_only | `([(10/p) = −1] / ((2 % (0.5 / 3.14159)) + (1 + [p ≡ 1 mod 3])))` | 0.08687 | 0.719 | – |
| kummer_degree | `([q] / (-1 + [q]))` | 0.0006038 | – (median relative error 0.0083) | – |

Congruence-only control: both labels occur in 96 of 192 held-out classes modulo 840; any rule of p mod 840 errs on at least 64,411 held-out primes (train lookup accuracy 0.8749).

Kummer degree: the degree q²/y from the selected formula vs q(q − 1) on 9,592 primes: max relative difference 2.220446049250313e-16; held-out mean squared standardized residual 1.212 (q/(q − 1): 1.212); implied Artin product 0.37395611362655956.

Full numbers: `artin.json` in this run directory.
