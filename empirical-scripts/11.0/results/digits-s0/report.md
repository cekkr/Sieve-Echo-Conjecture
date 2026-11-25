# Sieve Echo v11 — research report (digits-s0)

- **Run:** `digits-s0`, config `empirical-scripts/11.0/config/digits.json` (sha256 `d0260a516a`), seed 0.
- **Code:** root `7de8a72699` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `b7801aa892`.

## Verdicts

Analyses for this run are below.

## Step 22 — Evolvo on the digits of n, repetend lengths and prime densities

Sieve density W(n) = [gcd(n, 2·3·5·7·11·13·37) = 1] · 5.358362 / log n. Checks: {"integers": 1999000, "primes_in_integer_range": 148765, "digit_rules": true, "omega_sympy_sample": 3000, "repetend_lengths": {"rows": 799600, "period_checked_rows": 799600, "sympy_minimality_sample": 3000}, "prime_counts": {"points": 91, "sieve_checked_points": 51, "largest_x": 1000000000000}, "consecutive_prime_rows": 5761286}.

### prime_digits

Train 1,001…400,000 (399,000 rows), validation …1,000,000, test 1,000,001…2,000,000 (1,000,000 rows). Selected seed 2, 4 effective instructions.

Formula: `(((10 * [last digit]) * [last digit]) % (2.71828 % 0.5))`

```gfsl
# d$0 = last digit
# d$1 = digit sum
# d$2 = alternating digit sum
# d$3 = 3-digit block sum
# d$4 = alternating 3-digit block sum
# d$5 = digit count
# d$6 = log n
# d$7 = 1
d$1 = MUL(10.0, d$0)
d$7 = MOD(2.71828, 0.5)
d$1 = MUL(d$1, d$0)
d$8 = MOD(d$1, d$7)
RESULT d$8
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.06029; log loss 0.2116; AUC 0.7960; predicted 79,116 of 70,435; errors at 0.5: 70,435 |
| Cramér 1/log n | Brier 0.06547; log loss 0.2548; AUC 0.5044; predicted 70,427 of 70,435; errors at 0.5: 70,435 |
| sieve density W(n) | Brier 0.04385; log loss 0.1237; AUC 0.9383; predicted 70,428 of 70,435; errors at 0.5: 70,435 |
| train mean | Brier 0.06567; log loss 0.2561; AUC 0.5000; predicted 84,441 of 70,435; errors at 0.5: 70,435 |
| gradient boosting on the same inputs | Brier 0.06137; log loss 0.1779; AUC 0.9072; predicted 98,387 of 70,435; errors at 0.5: 99,789 |

### prime_digits_density

Train 1,001…400,000 (399,000 rows), validation …1,000,000, test 1,000,001…2,000,000 (1,000,000 rows). Selected seed 0, 1 effective instructions.

Formula: `([sieve density W(n)] / 1)`

```gfsl
# d$0 = last digit
# d$1 = digit sum
# d$2 = alternating digit sum
# d$3 = 3-digit block sum
# d$4 = alternating 3-digit block sum
# d$5 = digit count
# d$6 = log n
# d$7 = 1
# d$8 = Cramér density 1/log n
# d$9 = sieve density W(n)
d$10 = DIV(d$9, 1.0)
RESULT d$10
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.04385; log loss 0.1237; AUC 0.9383; predicted 70,428 of 70,435; errors at 0.5: 70,435 |
| Cramér 1/log n | Brier 0.06547; log loss 0.2548; AUC 0.5044; predicted 70,427 of 70,435; errors at 0.5: 70,435 |
| sieve density W(n) | Brier 0.04385; log loss 0.1237; AUC 0.9383; predicted 70,428 of 70,435; errors at 0.5: 70,435 |
| train mean | Brier 0.06567; log loss 0.2561; AUC 0.5000; predicted 84,441 of 70,435; errors at 0.5: 70,435 |
| gradient boosting on the same inputs | Brier 0.04509; log loss 0.1263; AUC 0.9373; predicted 74,163 of 70,435; errors at 0.5: 75,674 |

### prime_repetend

Train 1,001…399,999 (159,600 rows), validation …999,999, test 1,000,001…1,999,999 (400,000 rows). Selected seed 0, 4 effective instructions.

Formula: `((3.14159 / (3.14159 ^ ([n] % [L_2 (repetend length of 1/n in base 2)]))) ^ [L_10 (repetend length of 1/n in base 10)])`

```gfsl
# d$0 = n
# d$1 = L_2 (repetend length of 1/n in base 2)
# d$2 = L_10 (repetend length of 1/n in base 10)
# d$3 = 1
d$4 = MOD(d$0, d$1)
d$1 = POW(3.14159, d$4)
d$4 = DIV(3.14159, d$1)
d$4 = POW(d$4, d$2)
RESULT d$4
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.00023; log loss 0.0032; AUC 0.9999; predicted 70,528 of 70,435; errors at 0.5: 93 |
| Fermat base 2 [L_2 | n−1] | Brier 0.00023; log loss 0.0032; AUC 0.9999; predicted 70,528 of 70,435; errors at 0.5: 93 |
| Fermat base 10 [L_10 | n−1] | Brier 0.00028; log loss 0.0038; AUC 0.9998; predicted 70,545 of 70,435; errors at 0.5: 110 |
| both | Brier 0.00006; log loss 0.0008; AUC 1.0000; predicted 70,457 of 70,435; errors at 0.5: 22 |
| train mean | Brier 0.14631; log loss 0.4693; AUC 0.5000; predicted 84,441 of 70,435; errors at 0.5: 70,435 |
| gradient boosting on the same inputs | Brier 0.32140; log loss 1.9238; AUC 0.8448; predicted 202,289 of 70,435; errors at 0.5: 133,830 |

### omega_digits

Train 1,001…400,000 (399,000 rows), validation …1,000,000, test 1,000,001…2,000,000 (1,000,000 rows). Selected seed 0, 2 effective instructions.

Formula: `(3.14159 - ([last digit] % 2))`

```gfsl
# d$0 = last digit
# d$1 = digit sum
# d$2 = alternating digit sum
# d$3 = 3-digit block sum
# d$4 = alternating 3-digit block sum
# d$5 = digit count
# d$6 = log n
# d$7 = 1
d$0 = MOD(d$0, 2.0)
d$8 = SUB(3.14159, d$0)
RESULT d$8
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | MSE 0.9484; rounded accuracy 0.3855 |
| Hardy–Ramanujan log log n + B1 | MSE 1.0536; rounded accuracy 0.3712 |
| sieve mean of ω | MSE 0.3812; rounded accuracy 0.4538 |
| train mean | MSE 1.0814; rounded accuracy 0.3712 |
| gradient boosting on the same inputs | MSE 0.4864; rounded accuracy 0.4953 |

### omega_digits_density

Train 1,001…400,000 (399,000 rows), validation …1,000,000, test 1,000,001…2,000,000 (1,000,000 rows). Selected seed 2, 3 effective instructions.

Formula: `([sieve mean of ω] - (-1 / (2.71828 ^ [sieve mean of ω])))`

```gfsl
# d$0 = last digit
# d$1 = digit sum
# d$2 = alternating digit sum
# d$3 = 3-digit block sum
# d$4 = alternating 3-digit block sum
# d$5 = digit count
# d$6 = log n
# d$7 = 1
# d$8 = Hardy–Ramanujan mean log log n + B1
# d$9 = sieve mean of ω
d$1 = POW(2.71828, d$9)
d$10 = DIV(-1.0, d$1)
d$10 = SUB(d$9, d$10)
RESULT d$10
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | MSE 0.3707; rounded accuracy 0.4538 |
| Hardy–Ramanujan log log n + B1 | MSE 1.0536; rounded accuracy 0.3712 |
| sieve mean of ω | MSE 0.3812; rounded accuracy 0.4538 |
| train mean | MSE 1.0814; rounded accuracy 0.3712 |
| gradient boosting on the same inputs | MSE 0.3752; rounded accuracy 0.5209 |

### prime_count

Train 1,000…1,000,000 (31 rows), validation …100,000,000, test 125,892,541…1,000,000,000,000 (40 rows). Selected seed 4, 12 effective instructions.

Formula: `(((((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x])))) ^ (((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x])))) / sqrt(((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x]))))))) ^ (((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x])))) / sqrt(((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x]))))))) ^ (((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x])))) / sqrt(((([log x] ^ (1 / [log x])) ^ 1) ^ ((([log x] ^ (1 / [log x])) ^ 1) / log((10 + [log x])))))))`

```gfsl
# d$0 = log x
# d$1 = 1
d$1 = DIV(1.0, d$0)
d$1 = POW(d$0, d$1)
d$2 = POW(d$1, 1.0)
d$0 = ADD(10.0, d$0)
d$1 = LOG(d$0)
d$3 = DIV(d$2, d$1)
d$2 = POW(d$2, d$3)
d$1 = SQRT(d$2)
d$3 = DIV(d$2, d$1)
d$2 = POW(d$2, d$3)
d$2 = POW(d$2, d$3)
d$2 = POW(d$2, d$3)
RESULT d$2
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | max rel. error 1.09e-03; at x = 1e+12: 1.09e-03 |
| x/log x | max rel. error 5.70e-02; at x = 1e+12: 3.77e-02 |
| x/(log x − 1) | max rel. error 3.59e-03; at x = 1e+12: 1.53e-03 |
| Legendre x/(log x − 1.08366) | max rel. error 1.61e-03; at x = 1e+12: 1.61e-03 |
| li(x) | max rel. error 1.07e-04; at x = 1e+12: 1.02e-06 |
| Riemann R(x) | max rel. error 1.31e-05; at x = 1e+12: 3.92e-08 |
| train-fitted Legendre A = 1.04002 | max rel. error 1.32e-03; at x = 1e+12: 3.20e-05 |
| train mean | max rel. error 7.66e-02; at x = 1e+12: 7.66e-02 |

### prime_count_density

Train 1,000…1,000,000 (31 rows), validation …100,000,000, test 125,892,541…1,000,000,000,000 (40 rows). Selected seed 1, 3 effective instructions.

Formula: `((sqrt([li(x)·log x / x]) + 2) ^ [Cramér density 1/log x])`

```gfsl
# d$0 = log x
# d$1 = 1
# d$2 = Cramér density 1/log x
# d$3 = li(x)·log x / x
d$3 = SQRT(d$3)
d$5 = ADD(d$3, 2.0)
d$4 = POW(d$5, d$2)
RESULT d$4
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | max rel. error 1.60e-03; at x = 1e+12: 1.60e-03 |
| x/log x | max rel. error 5.70e-02; at x = 1e+12: 3.77e-02 |
| x/(log x − 1) | max rel. error 3.59e-03; at x = 1e+12: 1.53e-03 |
| Legendre x/(log x − 1.08366) | max rel. error 1.61e-03; at x = 1e+12: 1.61e-03 |
| li(x) | max rel. error 1.07e-04; at x = 1e+12: 1.02e-06 |
| Riemann R(x) | max rel. error 1.31e-05; at x = 1e+12: 3.92e-08 |
| train-fitted Legendre A = 1.04002 | max rel. error 1.32e-03; at x = 1e+12: 3.20e-05 |
| train mean | max rel. error 7.66e-02; at x = 1e+12: 7.66e-02 |

### prime_gap

Train 1,009…999,983 (78,330 rows), validation …9,999,991, test 10,000,019…99,999,971 (5,096,875 rows). Selected seed 0, 1 effective instructions.

Formula: `(0 + [log p])`

```gfsl
# d$0 = log p
# d$1 = Cramér density 1/log p
# d$2 = p mod 6
# d$3 = 1
d$4 = ADD(0.0, d$0)
RESULT d$4
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | MSE 225.954; mean 17.656 vs 17.658 |
| Cramér log p | MSE 225.954; mean 17.656 vs 17.658 |
| train fit 0.9963·log p + 0.0663 | MSE 225.954; mean 17.657 vs 17.658 |
| train mean | MSE 250.361; mean 12.754 vs 17.658 |
| gradient boosting on the same inputs | MSE 239.558; mean 13.998 vs 17.658 |

### last_digit_repeat

Train 1,009…999,983 (78,330 rows), validation …9,999,991, test 10,000,019…99,999,971 (5,096,875 rows). Selected seed 0, 1 effective instructions.

Formula: `(0.5 / 3.14159)`

```gfsl
# d$0 = log p
# d$1 = Cramér density 1/log p
# d$2 = last digit of p
# d$3 = 1
d$4 = DIV(0.5, 3.14159)
RESULT d$4
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.14324; log loss 0.4612; AUC 0.5000; predicted 811,194 of 881,553; errors at 0.5: 881,553 |
| equidistribution 1/4 | Brier 0.14898; log loss 0.4777; AUC 0.5000; predicted 1,274,219 of 881,553; errors at 0.5: 881,553 |
| train fit 0.2208 + -0.8475/log p | Brier 0.14304; log loss 0.4605; AUC 0.5037; predicted 880,208 of 881,553; errors at 0.5: 881,553 |
| train mean | Brier 0.14342; log loss 0.4619; AUC 0.5000; predicted 783,109 of 881,553; errors at 0.5: 881,553 |
| gradient boosting on the same inputs | Brier 0.14319; log loss 0.4611; AUC 0.4958; predicted 833,449 of 881,553; errors at 0.5: 881,553 |

Full numbers: `digits.json` in this run directory.
