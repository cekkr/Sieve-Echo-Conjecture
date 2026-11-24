# Sieve Echo v11 — research report (explicit-s0)

- **Run:** `explicit-s0`, config `empirical-scripts/11.0/config/explicit.json` (sha256 `321e3871e7`), seed 0.
- **Code:** root `e68d4397ff` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `3087954aa1`.

## Verdicts

Analyses for this run are below.

## Step 21 — Zeros behind the halved-repetend counts

Primes up to 100,000,000. Zeros on the critical line up to height 150.0 by sign changes of Z(t). Smoothed explicit formula with a Gaussian of width σ = 0.05 in log x.

| L-function | Base | Zeros ≤ T | Smooth count | Mean S(γ) | First zero | Max residual | Signal rms | Residual without zeros | Missing zero detectable up to height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zeta | – | 52 | 52.75 | +0.001 | 14.134725 | 6.91e-10 | 0.105 | 0.224 | 109 |
| chi_8 | 2 | 102 | 101.39 | +0.002 | 4.899974 | 1.17e-13 | 0.399 | 0.914 | 136 |
| chi_12 | 3 | 111 | 111.07 | +0.001 | 3.804628 | 9.09e-14 | 0.506 | 1.291 | 137 |
| chi_5 | 5 | 90 | 90.17 | +0.003 | 6.648453 | 2.95e-13 | 0.325 | 0.669 | 134 |
| chi_40 | 10 | 140 | 139.81 | +0.001 | 2.488210 | 1.04e-13 | 0.758 | 1.602 | 137 |

Chebyshev bias between unhalved and halved repetends. δ assumes GRH and the linear independence of the zeros (Rubinstein–Sarnak); the measured share is the logarithmic measure of x in [2, X] with more unhalved primes.

| Base | Δ | δ (GRH + LI) | Measured share | Lead changes | Last halved lead | Log-mean of E(x) from 10³ | 1 + 2/log x |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 | 0.9822 | 0.9748 | 216 | 45932087 | 1.278 | 1.170 |
| 3 | 12 | 0.9602 | 0.9215 | 724 | 54987073 | 1.237 | 1.170 |
| 5 | 5 | 0.9954 | 1.0000 | 0 | never | 1.293 | 1.170 |
| 10 | 40 | 0.8778 | 0.8107 | 2307 | 38222903 | 1.239 | 1.170 |

Schoenfeld's RH-conditional bound |θ(x) − x| < √x log² x/(8π) on [599, 100,000,000]: largest ratio 0.9836 (at 797). A finite range cannot prove the bound.

Full numbers: `explicit.json` in this run directory.
