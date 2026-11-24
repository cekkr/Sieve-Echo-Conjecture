# Conclusion — the arithmetic of repetends

Version 11 finds that the relationship between digit entropy and the number of distinct prime factors is largely explained by repetend length. Length and integer size account for 95.9% of held-out entropy variance (`full-s0-step4`). With a baseline of 98 repetend lengths and their Midy symmetry flags, entropy changes prediction accuracy by −0.07 percentage points (95% CI [−0.33, +0.24], `scale-s0-step7`).

The repeating block of $1/n$ has length $\operatorname{ord}_n(b)$ in every base $b$ coprime to $n$. This identifies an exact object to study across bases. At composite denominators, comparing orders gives a finite reconstruction experiment. At prime denominators, divisibility conditions on the orders lead to counts of primes in residue classes and to splitting conditions in number fields.

The v11 runs are in [`empirical-scripts/11.0/results/`](empirical-scripts/11.0/results). The [v11 conclusions](empirical-scripts/11.0/conclusions.md) give the full experimental record, and [revision 9](studies/9/sieve-echo-conjecture-9.md) gives the proofs. The prime-order milestone is complete. The [best Evolvo algorithms and reproduction commands](#7-best-evolvo-algorithms) are below.

## What "bases" means

A **base** is the number system 1/*n* is written in, as the conjecture prescribes. It is not a list of primes to divide by.
- 1/7 is 0.142857 142857 … in base 10 and 0.001 001 … in base 2. The repeating block is the **repetend**, and its length is L = ord_n(b).
- The decoder works with 19 such number systems: binary, ternary, …, decimal, …, base 31. The v8 paper used 13 bases and v10 used 16.

When a base shares a factor with *n*, as base 10 does at *n* = 35, the expansion has a non-repeating start (the pre-period). The decoder uses gcd(b, n) to isolate primes up through 31. For larger primes it uses the order tests.

Step 16 tests exactly this point (`validate-s0`):
- The integers are built from **one million primes downloaded from The Prime Pages** (t5k.org), about 9.6–9.8·10⁸. These are beyond every prime this project sieved.
- 37 000 of the 40 000 integers share no factor with any base, so the trial-division-like step does nothing for them.
- The primes are used only to build the test integers and to check the answers, never as input to the decoder or to Evolvo.

## 1. How the algorithm was found

The conjecture said that ω(*n*) is encoded in the spectral entropy of the repetend digits. Version 11 tested that claim with held-out data, baselines and nulls. The decoder came out of following the information wherever it actually was, one step of the [ROADMAP](ROADMAP.md) at a time.

1. **Length explains most of the entropy variance** (steps 2–4). log L and log n explain 95.9% of the held-out variance of the entropy, where L = ord_n(b) is the repetend length (`full-s0-step4`). Adding log ω to those controls changes held-out R² by 0.0004 in the same run.
2. **The lengths know ω** (step 4). A gradient-boosting model on *n* and the per-base lengths, with their lcm arithmetic, classifies ω at 98.2% held-out. Evolvo's formula search rediscovered Fermat's little theorem in repetend form: the lcm of the lengths divides n − 1. It recognizes primes at 99.99% (`full-s0-step4`).
3. **The residual entropy gain is indistinguishable from zero** (steps 7–8). Against the orders in 98 bases and their Midy flags, the entropy adds −0.07 points (CI [−0.33, +0.24]; `scale-s0-step7`). Two exact digit patterns have arithmetic explanations:
   - Midy's symmetry, which holds when −1 ∈ ⟨b⟩ mod *n*;
   - exact spectral zeros at the Euler factors of the prime divisors of *n* (`scale-s0-step8`).
4. **What in the lengths knows ω** (steps 10–11). ω(*n*) is the 2-rank of the unit group (ℤ/nℤ)*, up to a correction for the power of 2 in *n* (Proposition 1.6). The 2-adic valuations of the repetend lengths reveal that rank. Estimators built on this need no training, but are right for only 72–83% of integers (`step11-s0`), because they use the statistics of the valuations, not the valuations of each prime factor.
5. **From statistics to arithmetic** (step 13). The 2-adic ladder suggested raising *b* to L/2, L/4, … and asking which prime factors the result is 1 modulo. That is the reduction of factoring to order finding (Miller 1976, Shor 1994), and the 2-adic ladder of step 4 is its q = 2 case.
   - The data confirmed the link: Shor's half-period split fails exactly when Midy's symmetry holds, on 840 197 of 840 197 pairs (`decode-s0`).
   - The decoder generalizes it to every prime q dividing every repetend length, and to 19 bases at once.
6. **Where it fails** (step 15). A proof says exactly which *n* can defeat the decoder (Proposition 1.8). That turns "checked up to 10⁸" into "certified up to 3.7·10⁹" at the cost of one computation over single primes (`failures-s0`).

## 2. The final algorithm

The decoder uses modular arithmetic directly. Evolvo's fitted models are in §2.1.

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

`primes_of(L)` finds the primes of the repetend length (Miller–Rabin and Pollard's rho), `coprime_parts` splits the parts by their pairwise gcds, and `exponent_of(f, n)` is the largest e with fᵉ | n. `factor(n)` first computes every repetend length by long division (`repetend_length`) and then calls `decode`.

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

Pass other integers as arguments, but keep them small (up to about 10⁶). The long division takes L steps per base, and L can be close to *n*, so the lengths are the expensive part. The gcd refinement is polynomial in log *n* once the prime divisors of each length are supplied. The implementation obtains those divisors with trial division and Pollard–Brent.

### 2.1 What Evolvo found (for comparison)

The Evolvo formula search of step 4 (`full-s0-step4`, inputs z-scored, selected on validation, scored once on test) found two formulas.

| target | best evolved formula | held-out test accuracy |
| --- | --- | --- |
| is_prime | `[lcm divides n−1] ^ (2 % ([mean digit] / 10))` | **99.99%** — Fermat's little theorem in repetend form |
| ω | `([log n − log lcm] · |lcm divides n−1|^(1/4)) + e` | 65.5% |

For ω the best evolved formula is essentially log(n/λ(n)). Gradient boosting on the same inputs reaches 89.6%, the length-arithmetic model 98.2%, and the decoder recovers ω on every integer in the `decode-s0` exhaustive range when the pre-period checks are included. Evolvo found the right ingredients — the lcm of the lengths and its relation to *n* — but a formula over aggregate features cannot express "split *n* by gcds". That step had to be derived.

**Worked example: n = 2047 = 23 · 89.**
- Base 2 has L = 11. Both primes give 2 the order 11, so every base-2 test returns 1 or *n*, and base 2 alone cannot split 2047.
- Base 3 has L = 88 = 2³ · 11, because 3 has order 11 mod 23 and 88 mod 89. The first test for q = 2 is gcd(3¹¹ − 1, 2047) = 23.

The decoder needs only a few remainders b^d mod n. Each is read from the ⌈log_b n⌉ repetend digits that start at position *d*, as ⌊n·D/b^k⌋ + 1. In a check, 4 001 such readings had no error, and 600 integers were decoded from their digits alone (`decode-s0`).

## 3. Results

| decoder, 19 bases, no training | result |
| --- | --- |
| every n in [2, 10⁸], lengths only, comparing ω | 99 999 990 of 99 999 999 exact (`decode-s0`) |
| every n in [2, 10⁸], with pre-periods, comparing ω | **all exact** (`decode-s0`) |
| every n < 3 700 000 259 | **all ω exact, certified** (`failures-s0`) |
| ω-balanced samples, 8 decades 10⁴…10¹² (118 808 integers), comparing ω | all exact (`decode-s0`) |
| uniform samples, same decades (157 698 integers), comparing ω | all exact (`decode-s0`) |
| **integers built from external primes (t5k.org list, 961 748 941 … 982 451 653)** | **40 000 / 40 000 exact** (`validate-s0`) |
| — one list prime (n ≈ 9.7·10⁸) | 5 000 / 5 000 |
| — two list primes (n ≈ 9.5·10¹⁷) | 20 000 / 20 000 |
| — three list primes (n ≈ 9·10²⁶) | 5 000 / 5 000 |
| — p² · q (n ≈ 9·10²⁶) | 2 000 / 2 000 |
| — a list prime × 1–4 primes in (31, 10⁵), n up to 2.4·10⁴⁸ | 5 000 / 5 000 |
| — list primes × primes 2…31 (the pre-period step acts) | 3 000 / 3 000 |
| best learned ω model, 10⁴…10⁸, on the same integers | 98.6 / 98.2 / 97.0 / 94.2% (`step11-s0`) |
| training-free 2-rank estimator, same integers | 83.4 / 77.5 / 73.3 / 72.0% (`step11-s0`) |

The 9 exceptions without pre-periods are 3 741 870 = 2·3·5·11·17·23·29 and its multiples. Their only coprime bases are 7, 13, 19 and 31, all ≡ 1 (mod 6), so nothing separates 2 from 3.

**Fewer bases suffice in the larger sampled decades.** With only the first five bases (2, 3, 4, 5, 6) the decoder is exact for 90.0% of integers at 10⁴–10⁵ and 97.2% at 10¹¹–10¹². On average 2.45 bases suffice at 10⁴–10⁵ and 1.78 at 10¹¹–10¹² (`decode-s0`).

## 4. Why the order tests work

**Each gcd test implements an order-divisibility test.** For a prime p dividing n, p divides b^m − 1 exactly when ord_p(b) divides m (Lagrange). So gcd(b^m − 1, n) collects exactly the prime factors whose order divides m. Climbing the ladder m = (L/q^e)·q^i sorts the prime factors by the q-adic valuation of their order. Two primes therefore end up in different parts as soon as one base gives them different orders. This is also the only way they can stay together (Proposition 1.7).

**Identical orders are the obstruction for squarefree integers.** The decoder fails on squarefree *n* only if two prime factors of *n* have equal orders in every base coprime to *n* (Proposition 1.8). Suppose p < p′ give b the same order d. Then:
- both p and p′ divide b^d − 1, so for each p and each base there are only a handful of candidates p′;
- d divides gcd(p − 1, p′ − 1), so a long run of shared orders needs p − 1 and p′ − 1 to share almost all of their size.

Every collision the search found has that shape, with p′ − 1 a small multiple of p − 1:
- 89 − 1 = 4·(23 − 1);
- 109 − 1 = 3·(37 − 1);
- 3559 − 1 = 3·(1187 − 1).

A further base then has to land, modulo both primes, on the same order once more. The orders of 4, 9 and 16 follow from those of 2 and 3. Among the 5 761 444 primes below 10⁸ outside the base primes, `failures-s0` found zero pairs agreeing simultaneously in the full set of bases. A composite denominator can remove enough bases to expose a collision.

**Small prime divisors restrict the available bases.** A base that shares a factor with *n* is lost to the order tests, though its pre-period still isolates its own primes. To fail, an integer must therefore:
- carry enough small prime factors to remove most bases;
- then contain a pair of primes that the few remaining bases cannot tell apart.

Every least failure found for a prefix of the bases has exactly this shape (`failures-s0`; brute force agrees for 1–14 bases):

| bases used | least failing n | what happens |
| --- | --- | --- |
| first 1–8 | 30, 210, 2 310, 30 030 | primorials: no base is coprime to n |
| first 9–15 | 214 935 = 3·5·7·23·89, then 2 794 155 and 47 500 635 | only 2, 4, 11 (and 16) remain; 23·89 = 2¹¹ − 1, so 23 and 89 agree on all of them |
| first 16–17 | 60 555 495 = 3·5·7·11·13·37·109, then 1 392 776 385 | 2, 4, 16, 17, 19 remain; all give 37 and 109 the same orders |
| first 18 | 11 532 975 090 = 2·3·5·7·13·1187·3559 (least found; exact below 3.1·10⁹) | 11, 17, 19, 23, 29 remain and agree on 1187 and 3559 |
| all 19 | failure at 357 522 227 790, exact ω certified below 3 700 000 259 | the factor 31 removes the last base that distinguishes 1187 from 3559 in the preceding example |

The certified bound is the least integer a failure could still be: 37, the least prime outside {2…31}, times 100 000 007, the least prime above the search limit. Every pattern of small prime factors gave nothing smaller.

**A failure with all 19 bases.** Multiply the 18-base example by 31 to get $n=357{,}522{,}227{,}790$. In the remaining coprime bases 11, 17, 19, 23 and 29, the primes 1187 and 3559 share the order vector (593, 1186, 1186, 593, 1186). The decoder returns their product 4 224 533 in one part. The [deterministic reproduction command and provenance](ROADMAP.md#all-19-base-counterexample) show how this example bounds the first failure together with the earlier certificate.

## 5. What the validation establishes

The decoder applies the classical relation between periods and factorization. The exhaustive range in `decode-s0` checks the number of distinct prime factors, and the certificate in `failures-s0` checks the order-vector obstruction. The external validation in `validate-s0` compares the standalone decoder's prime-and-exponent output with the known factorization on 40 000 constructed integers. It generates the lengths from the known factors and checks their minimality directly modulo each test integer. The decoder itself receives $n$ and the lengths.

## 6. Prime counts and what Evolvo recovers

For prime $p$ and $p\nmid b$, the index $I_b(p)=(p-1)/\operatorname{ord}_p(b)$ measures the gap between the observed period and its maximal length. For odd $p$, the condition $2\mid I_b(p)$ says that $b$ is a square modulo $p$. Joint conditions across bases yield explicit unions of residue classes. Counting primes in those classes gives a concrete way to compare arithmetic data with the predictions of the corresponding Dirichlet characters.

For higher divisors of the indices, the relevant counts come from splitting in number fields. The implemented benchmarks compare independently computed flags with fixed densities. The cubic joint count through $10^6$ is 4,318, about 50 below its integral main term and within the published GRH-conditional error bound. This measures an explicit prime-counting discrepancy. The finite counts give no implication for the location of the zeros of $\zeta(s)$. [Benchmark report](empirical-scripts/11.0/results/prime-orders-s0/report.md) · [Counting theorems](ROADMAP.md#step-18--cubic-residue-benchmark-and-dependence)

## 7. Best Evolvo algorithms

These four formulas are the validation-selected exact rules from the two prime-order runs. In the code, `divides_index(p, m, b)` returns the bit for $m\mid I_b(p)$ using a modular power. The functions take primes $p>5$, the common eligible domain of the formula searches.

```python
def divides_index(p, m, b):
    quotient, remainder = divmod(p - 1, m)
    return int(remainder == 0 and pow(b, quotient, p) == 1)


def order_events(p):
    quadratic_base2 = 4 % abs((p % 8) - 4)
    quadratic_joint = divides_index(p, 2, 2) * divides_index(p, 2, 5) * divides_index(p, 2, 3)
    cubic_joint = divides_index(p, 3, 2) * divides_index(p, 3, 3)
    mixed_joint = abs(divides_index(p, 5, 2))
    return quadratic_base2, quadratic_joint, cubic_joint, mixed_joint
```

- **Quadratic rules.** `prime-orders-confirm-s10` checks the base-2 formula in all four odd residue classes and the joint rule on all 16 Boolean input rows, including the distractor bit used in the search. Both give zero errors on 70,435 test primes in $(10^6,2\cdot10^6]$. The base-2 formula uses the distance from 4 of the residue modulo 8. That distance is 1 or 3, and quadratic reciprocity selects exactly the distance-3 classes. [Quadratic report](empirical-scripts/11.0/results/prime-orders-confirm-s10/report.md) · [Saved programs](empirical-scripts/11.0/results/prime-orders-confirm-s10/prime_orders.json)
- **Cubic and mixed rules.** `prime-orders-s0` checks the cubic formula on all eight Boolean input rows and the mixed formula on the six rows allowed by the mixed implication, in both cases including the distractor bit. Both give zero errors on 60,514 test primes in $(200{,}000,10^6]$. The mixed rule follows because $5\mid I_2(p)$ implies $p\equiv1\pmod5$, which implies $2\mid I_5(p)$. [Main report](empirical-scripts/11.0/results/prime-orders-s0/report.md) · [Saved programs](empirical-scripts/11.0/results/prime-orders-s0/prime_orders.json)
- **Cubic control using congruences alone.** Evolvo selected $1/9$ and achieved F1 of zero. Each of the 16 observed classes modulo 120 contains both event outcomes. Any deterministic rule using those congruences makes at least 3,342 errors on the 30,243 primes in that held-out control. [Control results](empirical-scripts/11.0/results/prime-orders-s0/report.md#step-19--evolvo-formulas)

Run both searches from the repository root with the [environment from the README](README.md#reproduce):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/prime-orders.json --run-id prime-orders-s0-repro --stages prime_orders,report
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/prime-orders-confirm.json --run-id prime-orders-confirm-s10-repro --stages prime_orders,report
```

The benchmark counts were cross-checked independently, the main run reproduced its program slots and metrics, and the selected programs were replayed. The four rules express established arithmetic. Together with the finite prime counts and the congruence-only limit, they complete steps 17–19. Ordinary RH remains open.
