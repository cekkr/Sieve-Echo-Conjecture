# Sieve Echo v11 — claim verdicts (validate-s0)

- **Run:** `validate-s0`, config `empirical-scripts/11.0/config/orders.json` (sha256 `b7ef4a6015`), seed 0.
- **Code:** root `5151430eb2` (dirty), libs/evolvo `d2111859f0`, v11 sources sha256 `7c4249b9b5`.
- **Definitions:** `None`; primary base set `None`; entropies .
- **Data:** None integers, None coprime (n, b) pairs; split None (split sha256 `unknown`).
- **Inference:** None bootstrap resamples (95% percentile CIs over test rows), None permutations; effect threshold None (ΔR², η² excess, r²).

Paper statistics are re-computed the way the v8.2 code computed them (in-sample, `H_v8`, n ∈ [2, 9999]); corrected tests fit on the train split and report on the test split.

## Verdicts

No `claims` stage in this run; see the sections below.

## Step 16 — The final decoder on an external prime list

Prime list: 1,000,000 primes 961,748,941 … 982,451,653 from https://t5k.org/lists/small/millions/primes50.zip (SHA-256 of the text `fd6f4957747e777b…`), sorted: True, 200 random consecutive pairs confirmed by SymPy `nextprime`: True; 70,851 list primes used, all confirmed prime by SymPy: True. The list only builds the integers and checks the answers; the decoder (`sieve_echo_decoder.decode`) sees n and the repetend lengths, and every length was verified from n alone (b^L ≡ 1, b^(L/r) ≢ 1 for each prime r | L).

**40,000 of 40,000 integers decoded exactly (100.000%).**

| integers | count | n range | all 19 bases coprime | lengths verified | exact | ms per integer |
| --- | --- | --- | --- | --- | --- | --- |
| one list prime | 5,000 | 9.62e+08 – 9.82e+08 | 5,000 | 5,000 | 5,000 | 0.9 |
| two list primes | 20,000 | 9.25e+17 – 9.65e+17 | 20,000 | 20,000 | 20,000 | 8.7 |
| three list primes | 5,000 | 8.93e+26 – 9.46e+26 | 5,000 | 5,000 | 5,000 | 20.8 |
| p² · q (list primes) | 2,000 | 8.90e+26 – 9.47e+26 | 2,000 | 2,000 | 2,000 | 54.0 |
| list prime × primes in (31, 10⁵) | 5,000 | 9.36e+10 – 2.38e+48 | 5,000 | 5,000 | 5,000 | 8.3 |
| list primes × primes 2…31 (pre-periods act) | 3,000 | 1.93e+09 – 1.03e+35 | 0 | 3,000 | 3,000 | 3.7 |

Full numbers: `validate.json` in this run directory.
