# Sieve Echo v11 — research report (digits-confirm-s1)

- **Run:** `digits-confirm-s1`, config `empirical-scripts/11.0/config/digits-confirm.json` (sha256 `7360790714`), seed 1.
- **Code:** root `7de8a72699` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `fe0d50eb89`.

## Verdicts

Analyses for this run are below.

## Step 22 — Evolvo on the digits of n, repetend lengths and prime densities

Sieve density W(n) = [gcd(n, 2·3·5·7·11·13·37) = 1] · 5.358362 / log n. Checks: {"integers": 1999000, "primes_in_integer_range": 148765, "digit_rules": true, "omega_sympy_sample": 3000, "repetend_lengths": {"rows": 799600, "period_checked_rows": 799600, "sympy_minimality_sample": 3000}, "prime_counts": {"points": 91, "sieve_checked_points": 51, "largest_x": 1000000000000}, "consecutive_prime_rows": 5761286}.

### prime_digits

Train 1,001…400,000 (399,000 rows), validation …1,000,000, test 1,000,001…2,000,000 (1,000,000 rows). Selected seed 6, 6 effective instructions.

Formula: `((0.5 / (([last digit] % 2.71828) * (([last digit] % 2) * 2.71828))) % 0.5)`

```gfsl
# d$0 = last digit
# d$1 = digit sum
# d$2 = alternating digit sum
# d$3 = 3-digit block sum
# d$4 = alternating 3-digit block sum
# d$5 = digit count
# d$6 = log n
# d$7 = 1
d$8 = MOD(d$0, 2.0)
d$5 = MUL(d$8, 2.71828)
d$8 = MOD(d$0, 2.71828)
d$5 = MUL(d$8, d$5)
d$8 = DIV(0.5, d$5)
d$8 = MOD(d$8, 0.5)
RESULT d$8
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.05926; log loss 0.1968; AUC 0.8227; predicted 75,276 of 70,435; errors at 0.5: 70,435 |
| Cramér 1/log n | Brier 0.06547; log loss 0.2548; AUC 0.5044; predicted 70,427 of 70,435; errors at 0.5: 70,435 |
| sieve density W(n) | Brier 0.04385; log loss 0.1237; AUC 0.9383; predicted 70,428 of 70,435; errors at 0.5: 70,435 |
| train mean | Brier 0.06567; log loss 0.2561; AUC 0.5000; predicted 84,441 of 70,435; errors at 0.5: 70,435 |
| gradient boosting on the same inputs | Brier 0.05272; log loss 0.1568; AUC 0.9099; predicted 88,615 of 70,435; errors at 0.5: 75,827 |

### prime_repetend

Train 1,001…399,999 (159,600 rows), validation …999,999, test 1,000,001…1,999,999 (400,000 rows). Selected seed 3, 4 effective instructions.

Formula: `(log(2.71828) / (([n] % [L_2 (repetend length of 1/n in base 2)]) ^ 10))`

```gfsl
# d$0 = n
# d$1 = L_2 (repetend length of 1/n in base 2)
# d$2 = L_10 (repetend length of 1/n in base 10)
# d$3 = 1
d$2 = MOD(d$0, d$1)
d$0 = POW(d$2, 10.0)
d$2 = LOG(2.71828)
d$4 = DIV(d$2, d$0)
RESULT d$4
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.00023; log loss 0.0032; AUC 0.9999; predicted 70,528 of 70,435; errors at 0.5: 93 |
| Fermat base 2 [L_2 | n−1] | Brier 0.00023; log loss 0.0032; AUC 0.9999; predicted 70,528 of 70,435; errors at 0.5: 93 |
| Fermat base 10 [L_10 | n−1] | Brier 0.00028; log loss 0.0038; AUC 0.9998; predicted 70,545 of 70,435; errors at 0.5: 110 |
| both | Brier 0.00006; log loss 0.0008; AUC 1.0000; predicted 70,457 of 70,435; errors at 0.5: 22 |
| train mean | Brier 0.14631; log loss 0.4693; AUC 0.5000; predicted 84,441 of 70,435; errors at 0.5: 70,435 |
| gradient boosting on the same inputs | Brier 0.32036; log loss 1.9347; AUC 0.8433; predicted 202,857 of 70,435; errors at 0.5: 130,711 |

### omega_digits

Train 1,001…400,000 (399,000 rows), validation …1,000,000, test 1,000,001…2,000,000 (1,000,000 rows). Selected seed 5, 7 effective instructions.

Formula: `((log((0.5 % 1)) ^ (([alternating digit sum] * [last digit]) / (([alternating digit sum] * [last digit]) / [last digit]))) + 2.71828)`

```gfsl
# d$0 = last digit
# d$1 = digit sum
# d$2 = alternating digit sum
# d$3 = 3-digit block sum
# d$4 = alternating 3-digit block sum
# d$5 = digit count
# d$6 = log n
# d$7 = 1
d$2 = MUL(d$2, d$0)
d$0 = DIV(d$2, d$0)
d$0 = DIV(d$2, d$0)
d$7 = MOD(0.5, 1.0)
d$5 = LOG(d$7)
d$5 = POW(d$5, d$0)
d$8 = ADD(d$5, 2.71828)
RESULT d$8
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | MSE 0.8276; rounded accuracy 0.4181 |
| Hardy–Ramanujan log log n + B1 | MSE 1.0536; rounded accuracy 0.3712 |
| sieve mean of ω | MSE 0.3812; rounded accuracy 0.4538 |
| train mean | MSE 1.0814; rounded accuracy 0.3712 |
| gradient boosting on the same inputs | MSE 0.4923; rounded accuracy 0.4952 |

### prime_count

Train 1,000…1,000,000 (31 rows), validation …100,000,000, test 125,892,541…1,000,000,000,000 (40 rows). Selected seed 2, 3 effective instructions.

Formula: `(([1] - [log x]) / (1 + ([1] - [log x])))`

```gfsl
# d$0 = log x
# d$1 = 1
d$0 = SUB(d$1, d$0)
d$1 = ADD(1.0, d$0)
d$2 = DIV(d$0, d$1)
RESULT d$2
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | max rel. error 3.79e-04; at x = 1e+12: 1.25e-04 |
| x/log x | max rel. error 5.70e-02; at x = 1e+12: 3.77e-02 |
| x/(log x − 1) | max rel. error 3.59e-03; at x = 1e+12: 1.53e-03 |
| Legendre x/(log x − 1.08366) | max rel. error 1.61e-03; at x = 1e+12: 1.61e-03 |
| li(x) | max rel. error 1.07e-04; at x = 1e+12: 1.02e-06 |
| li(x) − li(√x)/2 | max rel. error 1.68e-05; at x = 1e+12: 2.79e-08 |
| Riemann R(x) | max rel. error 1.31e-05; at x = 1e+12: 3.92e-08 |
| train-fitted Legendre A = 1.04002 | max rel. error 1.32e-03; at x = 1e+12: 3.20e-05 |
| train mean | max rel. error 7.66e-02; at x = 1e+12: 7.66e-02 |

### prime_count_density

Train 1,000…1,000,000 (31 rows), validation …100,000,000, test 125,892,541…1,000,000,000,000 (40 rows). Selected seed 3, 3 effective instructions.

Formula: `([li(x)·log x / x] / sqrt(([li(√x)·log x / x] + [1])))`

```gfsl
# d$0 = log x
# d$1 = 1
# d$2 = Cramér density 1/log x
# d$3 = li(x)·log x / x
# d$4 = li(√x)·log x / x
d$1 = ADD(d$4, d$1)
d$1 = SQRT(d$1)
d$5 = DIV(d$3, d$1)
RESULT d$5
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | max rel. error 1.26e-05; at x = 1e+12: 6.89e-08 |
| x/log x | max rel. error 5.70e-02; at x = 1e+12: 3.77e-02 |
| x/(log x − 1) | max rel. error 3.59e-03; at x = 1e+12: 1.53e-03 |
| Legendre x/(log x − 1.08366) | max rel. error 1.61e-03; at x = 1e+12: 1.61e-03 |
| li(x) | max rel. error 1.07e-04; at x = 1e+12: 1.02e-06 |
| li(x) − li(√x)/2 | max rel. error 1.68e-05; at x = 1e+12: 2.79e-08 |
| Riemann R(x) | max rel. error 1.31e-05; at x = 1e+12: 3.92e-08 |
| train-fitted Legendre A = 1.04002 | max rel. error 1.32e-03; at x = 1e+12: 3.20e-05 |
| train mean | max rel. error 7.66e-02; at x = 1e+12: 7.66e-02 |

### last_digit_repeat

Train 1,009…999,983 (78,330 rows), validation …9,999,991, test 10,000,019…99,999,971 (5,096,875 rows). Selected seed 4, 2 effective instructions.

Formula: `(([1] / 10) + [Cramér density 1/log p])`

```gfsl
# d$0 = log p
# d$1 = Cramér density 1/log p
# d$2 = last digit of p
# d$3 = 1
d$4 = DIV(d$3, 10.0)
d$4 = ADD(d$4, d$1)
RESULT d$4
```

| Predictor | Held-out score |
| --- | --- |
| **Evolvo (GFSL above)** | Brier 0.14332; log loss 0.4616; AUC 0.4963; predicted 798,695 of 881,553; errors at 0.5: 881,553 |
| equidistribution 1/4 | Brier 0.14898; log loss 0.4777; AUC 0.5000; predicted 1,274,219 of 881,553; errors at 0.5: 881,553 |
| train fit 0.2208 + -0.8475/log p | Brier 0.14304; log loss 0.4605; AUC 0.5037; predicted 880,208 of 881,553; errors at 0.5: 881,553 |
| train mean | Brier 0.14342; log loss 0.4619; AUC 0.5000; predicted 783,109 of 881,553; errors at 0.5: 881,553 |
| gradient boosting on the same inputs | Brier 0.14324; log loss 0.4613; AUC 0.5043; predicted 810,522 of 881,553; errors at 0.5: 881,553 |

Full numbers: `digits.json` in this run directory.
