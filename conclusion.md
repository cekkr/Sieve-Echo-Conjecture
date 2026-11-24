# Conclusion — the echo is exact

**The repetends of 1/*n* in 19 bases determine the factorization of *n*.** A short, training-free procedure reads ω(*n*) and the prime factors themselves off the repetend lengths and a few digits. It is

- correct for **every integer below 3 700 000 259**: exhaustively checked for every n ≤ 10⁸ and certified beyond by a proof of where it can fail;
- correct for all 276 506 sampled integers up to 10¹²;
- correct where the best learned model of this project classified ω only at 94–99%.

Every number here comes from a version-11 run (seed 0), named in parentheses; runs are under [`empirical-scripts/11.0/results/`](empirical-scripts/11.0/results). The full v11 conclusions, including what happened to every claim of the v8 paper, are in [`empirical-scripts/11.0/conclusions.md`](empirical-scripts/11.0/conclusions.md). The proofs are Propositions 1.6–1.8 of the revision-9 draft [`studies/9/sieve-echo-conjecture-9.md`](studies/9/sieve-echo-conjecture-9.md).

**3.7·10⁹ is not where the decoder fails.** It is where the certificate stops, because the search for colliding primes went up to 10⁸. With all 19 bases no failing *n* is known at all; where the first one lies is open (§6).

## 1. How the algorithm was found

The conjecture said that ω(*n*) is encoded in the spectral entropy of the repetend digits. Version 11 tested that claim with held-out data, baselines and nulls. The decoder came out of following the information wherever it actually was, one step of the [ROADMAP](ROADMAP.md) at a time.

1. **The entropy is a length statistic** (steps 2–4). log L and log n explain 95.9% of the held-out variance of the entropy, where L = ord_n(b) is the repetend length (`full-s0-step4`). Whatever the entropy knows about ω, it knows through L.
2. **The lengths know ω** (step 4). A gradient-boosting model on *n* and the per-base lengths, with their lcm arithmetic, classifies ω at 98.2% held-out. Evolvo's formula search rediscovered Fermat's little theorem in repetend form: the lcm of the lengths divides n − 1. It recognizes primes at 99.99% (`full-s0-step4`).
3. **Nothing is left for the digits** (steps 7–8). Against the orders in 98 bases and their Midy flags, the entropy adds −0.07 points (CI [−0.33, +0.24]; `scale-s0-step7`). The two real digit traces both turn out to be functions of the orders:
   - Midy's symmetry, which holds when −1 ∈ ⟨b⟩ mod *n*;
   - exact spectral zeros at the Euler factors of the prime divisors of *n* (`scale-s0-step8`).
4. **What in the lengths knows ω** (steps 10–11). ω(*n*) is the 2-rank of the unit group (ℤ/nℤ)*, up to a correction for the power of 2 in *n* (Proposition 1.6). The 2-adic valuations of the repetend lengths reveal that rank. Estimators built on this need no training, but are right for only 72–83% of integers (`step11-s0`), because they use the statistics of the valuations, not the valuations of each prime factor.
5. **From statistics to arithmetic** (step 13). The 2-adic ladder suggested raising *b* to L/2, L/4, … and asking which prime factors the result is 1 modulo. That is the reduction of factoring to order finding (Miller 1976, Shor 1994), and the 2-adic ladder of step 4 is its q = 2 case.
   - The data confirmed the link: Shor's half-period split fails exactly when Midy's symmetry holds, on 840 197 of 840 197 pairs (`decode-s0`).
   - The decoder generalizes it to every prime q dividing every repetend length, and to 19 bases at once.
6. **Where it fails** (step 15). A proof says exactly which *n* can defeat the decoder (Proposition 1.8). That turns "checked up to 10⁸" into "certified up to 3.7·10⁹" at the cost of one computation over single primes (`failures-s0`).

## 2. The final algorithm

**It is not an Evolvo formula.** It was derived from what the analyses showed, and it needs no training and no genome. Evolvo's best results are in §2.1 for comparison.

The complete algorithm is [`empirical-scripts/11.0/sieve_echo_decoder.py`](empirical-scripts/11.0/sieve_echo_decoder.py): pure Python, no dependencies. It is tested against SymPy in [`tests/test_sieve_echo_decoder.py`](empirical-scripts/11.0/tests/test_sieve_echo_decoder.py), and it is the same procedure as the `decode` stage that produced `decode-s0` and `failures-s0`.

Its core, as it appears in that file (docstring shortened):

```python
BASES = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31]

def decode(n: int, lengths: dict[int, int]) -> dict[int, int]:
    """{factor: exponent} of n from its repetend lengths {b: L_b}; a base missing from `lengths` shares a factor with n."""
    parts = [n]

    def split(g: int) -> None:
        nonlocal parts
        parts = [y for x in parts for y in ((math.gcd(x, g), x // math.gcd(x, g)) if 1 < math.gcd(x, g) < x else (x,))]

    for b in BASES:
        if math.gcd(b, n) > 1:
            split(math.gcd(b, n))                        # pre-period: the primes b shares with n
            continue
        L = lengths[b]
        for q in primes_of(L):                           # every prime q dividing the repetend length
            m = L
            while m % q == 0:
                m //= q                                  # m = L / q^e
            while m < L:                                 # m = (L / q^e) q^i, i = 0 … e−1
                split(math.gcd(pow(b, m, n) - 1, n))     # b^m mod n: a remainder of the long division of 1/n
                m *= q
    factors = coprime_parts(parts)                       # split the parts by their gcds until pairwise coprime
    return {f: exponent_of(f, n) for f in factors}
```

`primes_of(L)` factors the repetend length by trial division, `coprime_parts` splits the parts by their pairwise gcds, and `exponent_of(f, n)` is the largest e with fᵉ | n. `factor(n)` first computes every repetend length by long division (`repetend_length`) and then calls `decode`.

Run it on numbers you can check by hand (the default examples):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_decoder.py
```

```
2047 = 23 · 89   ω = 2   check: product = n, all factors prime
1001 = 7 · 11 · 13   ω = 3   check: product = n, all factors prime
1024 = 2^10   ω = 1   check: product = n, all factors prime
214935 = 3 · 5 · 7 · 23 · 89   ω = 5   check: product = n, all factors prime
```

The "check" is independent of the decoder: it multiplies the factors back and tests each by trial division. Two of these examples are hard cases:
- 2047 = 2¹¹ − 1 cannot be split by base 2;
- 214 935 is the least integer the first 9 bases fail on.

For larger tests the test file checks every n < 20 000 against SymPy's factorization (`tests/test_sieve_echo_decoder.py`). The exhaustive and certified ranges are from the `decode` and `failures` stages.

Pass other integers as arguments, but keep them small (up to about 10⁶). The long division takes L steps per base, and L can be close to *n*, so the lengths are the expensive part. The `decode` step itself is polynomial in log *n*. The `decode` step itself is polynomial in log *n*.

### 2.1 What Evolvo found (for comparison)

The Evolvo formula search of step 4 (`full-s0-step4`, inputs z-scored, selected on validation, scored once on test) found two formulas.

| target | best evolved formula | held-out test accuracy |
| --- | --- | --- |
| is_prime | `[lcm divides n−1] ^ (2 % ([mean digit] / 10))` | **99.99%** — Fermat's little theorem in repetend form |
| ω | `([log n − log lcm] · |lcm divides n−1|^(1/4)) + e` | 65.5% |

For ω the best evolved formula is essentially log(n/λ(n)). Gradient boosting on the same inputs reaches 89.6%, the length-arithmetic model 98.2%, and the decoder above 100%. Evolvo found the right ingredients — the lcm of the lengths and its relation to *n* — but a formula over aggregate features cannot express "split *n* by gcds". That step had to be derived.

**Worked example: n = 2047 = 23 · 89.**
- Base 2 has L = 11. Both primes give 2 the order 11, so every base-2 test returns 1 or *n*, and base 2 alone cannot split 2047.
- Base 3 has L = 88 = 2³ · 11, because 3 has order 11 mod 23 and 88 mod 89. The first test for q = 2 is gcd(3¹¹ − 1, 2047) = 23.

The decoder needs only a few remainders b^d mod n. Each is read from the ⌈log_b n⌉ repetend digits that start at position *d*, as ⌊n·D/b^k⌋ + 1. In a check, 4 001 such readings had no error, and 600 integers were decoded from their digits alone (`decode-s0`).

## 3. Results

| decoder, 19 bases, no training | result |
| --- | --- |
| every n in [2, 10⁸], lengths only | 99 999 990 of 99 999 999 exact (`decode-s0`) |
| every n in [2, 10⁸], with pre-periods | **all exact** (`decode-s0`) |
| every n < 3 700 000 259 | **all exact, certified** (`failures-s0`) |
| ω-balanced samples, 8 decades 10⁴…10¹² (118 808 integers) | all exact (`decode-s0`) |
| uniform samples, same decades (157 698 integers) | all exact (`decode-s0`) |
| best learned ω model, 10⁴…10⁸, on the same integers | 98.6 / 98.2 / 97.0 / 94.2% (`step11-s0`) |
| training-free 2-rank estimator, same integers | 83.4 / 77.5 / 73.3 / 72.0% (`step11-s0`) |

The 9 exceptions without pre-periods are 3 741 870 = 2·3·5·11·17·23·29 and its multiples. Their only coprime bases are 7, 13, 19 and 31, all ≡ 1 (mod 6), so nothing separates 2 from 3.

**It gets easier as *n* grows.** With only the first five bases (2, 3, 4, 5, 6) the decoder is exact for 90.0% of integers at 10⁴–10⁵ and 97.2% at 10¹¹–10¹². On average 2.45 bases suffice at 10⁴–10⁵ and 1.78 at 10¹¹–10¹² (`decode-s0`).

## 4. Why it works so well

**Each step is exact, not statistical.** For a prime p dividing n, p divides b^m − 1 exactly when ord_p(b) divides m (Lagrange). So gcd(b^m − 1, n) collects exactly the prime factors whose order divides m. Climbing the ladder m = (L/q^e)·q^i sorts the prime factors by the q-adic valuation of their order. Two primes therefore end up in different parts as soon as one base gives them different orders. This is also the only way they can stay together (Proposition 1.7).

**Two primes almost never share all their orders.** The decoder fails on squarefree *n* only if two prime factors of *n* have equal orders in every base coprime to *n* (Proposition 1.8). Suppose p < p′ give b the same order d. Then:
- both p and p′ divide b^d − 1, so for each p and each base there are only a handful of candidates p′;
- d divides gcd(p − 1, p′ − 1), so a long run of shared orders needs p − 1 and p′ − 1 to share almost all of their size.

Every collision the search found has that shape, with p′ − 1 a small multiple of p − 1:
- 89 − 1 = 4·(23 − 1);
- 109 − 1 = 3·(37 − 1);
- 3559 − 1 = 3·(1187 − 1).

A further base then has to land, modulo both primes, on the same order once more. The orders of 4, 9 and 16 follow from those of 2 and 3. Heuristically, the 11 prime bases 2…31 and the products among the bases each add a nearly independent condition. Modulo all 5 761 444 primes below 10⁸, not one pair satisfies all of them at once (`failures-s0`).

**The failures that do exist are expensive.** A base that shares a factor with *n* is lost to the order tests, though its pre-period still isolates its own primes. To fail, an integer must therefore:
- carry enough small prime factors to remove most bases;
- then contain a pair of primes that the few remaining bases cannot tell apart.

Every least failure found for a prefix of the bases has exactly this shape (`failures-s0`; brute force agrees for 1–14 bases):

| bases used | least failing n | what happens |
| --- | --- | --- |
| first 1–8 | 30, 210, 2 310, 30 030 | primorials: no base is coprime to n |
| first 9–15 | 214 935 = 3·5·7·23·89, then 2 794 155 and 47 500 635 | only 2, 4, 11 (and 16) remain; 23·89 = 2¹¹ − 1, so 23 and 89 agree on all of them |
| first 16–17 | 60 555 495 = 3·5·7·11·13·37·109, then 1 392 776 385 | 2, 4, 16, 17, 19 remain; all give 37 and 109 the same orders |
| first 18 | 11 532 975 090 = 2·3·5·7·13·1187·3559 (least found; exact below 3.1·10⁹) | 11, 17, 19, 23, 29 remain and agree on 1187 and 3559 |
| all 19 | none known; exact below 3 700 000 259 | no two primes below 10⁸ agree on all 19 bases |

The certified bound is the least integer a failure could still be: 37, the least prime outside {2…31}, times 100 000 007, the least prime above the search limit. Every pattern of small prime factors gave nothing smaller.

**Why random integers up to 10¹² never fail.** A failure needs both many small prime factors and a colliding pair among the rest. These are rare separately and rarer together. Larger primes have larger p − 1, which leaves more room for their orders to differ.

## 5. What it is, and what it is not

- **It is not a fast factoring method.** The decoder takes the repetend lengths as given. Computing a length classically needs λ(n), which needs the factorization, or about L digits of the expansion, and L can be as large as n − 1. Shor's algorithm finds that period quantumly, with a Fourier transform of the remainder sequence b^k mod n. That remainder sequence is the long division of 1 by *n*.
- **What it shows is where the information is.** The repetend of 1/*n* is a complete certificate of the factorization of *n*: its length and a few digits at fractions of the period.
- **The conjecture looked at the right object with the wrong statistic.** It looked at the digits and their Fourier spectrum, which was right. But the spectral entropy keeps the spread of the spectrum and discards the arithmetic of the period, so the echo it hoped to hear is there and is exact, but the entropy cannot hear it.

## 6. Open

- **Where the first failure lies for all 19 bases.** The certificate stops at 3.7·10⁹. Failures must exist by 3.0·10¹⁴, since 2·3·…·31·37·41 leaves no base coprime to *n*. How often the decoder fails between those two bounds is unmeasured.
- **Whether two primes can share their orders in all 19 bases at all.** If they can, they would yield failures with no small prime factors.
- **A decoder that uses only the bases a given *n* needs.** Two or three suffice on average.

The proofs, the run commands and the limits of each number are in [`studies/9`](studies/9/sieve-echo-conjecture-9.md) (§1, §6, §9) and [`empirical-scripts/11.0/conclusions.md`](empirical-scripts/11.0/conclusions.md) (§10).
