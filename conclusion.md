# Conclusion — the arithmetic of repetends

Version 11 finds that the relationship between digit entropy and the number of distinct prime factors is largely explained by repetend length. Length and integer size account for 95.9% of held-out entropy variance (`full-s0-step4`). With a baseline of 98 repetend lengths and their Midy symmetry flags, entropy changes prediction accuracy by −0.07 percentage points (95% CI [−0.33, +0.24], `scale-s0-step7`).

The repeating block of $1/n$ has length $\mathrm{ord}_n(b)$ in every base $b$ coprime to $n$. This identifies an exact object to study across bases. At composite denominators, comparing orders gives a finite reconstruction experiment. At prime denominators, divisibility conditions on the orders lead to counts of primes in residue classes and to splitting conditions in number fields.

The work ends at two classical problems.
- **Full-period repetends** ($1/p$ repeats with the maximal length $p-1$) are Artin's primitive-root problem. Through $10^8$, the counts for all 19 bases match Hooley's densities, which assume GRH. That includes the Lehmer–Heilbronn correction: base 5 is 98 standard deviations from Artin's original constant and 0.65 from the corrected value (`artin-s0`, §8).
- **Halved repetends** (length dividing $(p-1)/2$) form a quadratic character. Their log-weighted count is a sum over the zeros of a Dirichlet L-function. After smoothing, the prime data through $10^8$ and the zeros up to height 150 agree to $10^{-13}$ (`explicit-s0`, §9).

§11 extends Evolvo to the digits of $n$ itself and to prime-density predictions, and lists the practical conclusions: a probable-prime test read off $1/n$, the calibrated prime probability from the digits of $n$, and elementary prime-count and gap formulas, each with GFSL code. §10 says exactly what this does and does not mean for the Riemann hypothesis. In short, these are verified consequences of known theorems and hypotheses. They are not a step toward a proof.

The v11 runs are in [`empirical-scripts/11.0/results/`](empirical-scripts/11.0/results). The [v11 conclusions](empirical-scripts/11.0/conclusions.md) give the full experimental record, and [revision 9](studies/9/sieve-echo-conjecture-9.md) gives the proofs. The [best Evolvo algorithms and reproduction commands](#7-best-evolvo-algorithms) are below.

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

The same two programs as Evolvo wrote them in GFSL. Each line is one slot-encoded instruction, and `d$k` are decimal registers. The inputs `d$0…d$11` are the z-scored features of step 4, of which only the used ones are listed. Saved in [`full-s0-step4/formulas.json`](empirical-scripts/11.0/results/full-s0-step4/formulas.json).

```gfsl
# is_prime — d$7 = mean digit (v8 bases), d$10 = [lcm of the lengths divides n − 1]
d$12 = DIV(d$7, 10.0)
d$12 = MOD(2.0, d$12)
d$12 = POW(d$10, d$12)
RESULT d$12
```

```gfsl
# ω — d$9 = log n − log lcm(L_b), d$10 = [lcm divides n − 1]
d$2 = ABS(d$10)
d$2 = SQRT(d$2)
d$2 = SQRT(d$2)
d$1 = MUL(d$9, d$2)
d$12 = ADD(d$1, 2.71828)
RESULT d$12
```

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

For prime $p$ and $p\nmid b$, the index $I_b(p)=(p-1)/\mathrm{ord}_p(b)$ measures the gap between the observed period and its maximal length. For odd $p$, the condition $2\mid I_b(p)$ says that $b$ is a square modulo $p$. Joint conditions across bases yield explicit unions of residue classes. Counting primes in those classes gives a concrete way to compare arithmetic data with the predictions of the corresponding Dirichlet characters.

For higher divisors of the indices, the relevant counts come from splitting in number fields. The implemented benchmarks compare independently computed flags with fixed densities. The cubic joint count through $10^6$ is 4,318, about 50 below its integral main term and within the published GRH-conditional error bound. This measures an explicit prime-counting discrepancy. The finite counts give no implication for the location of the zeros of $\zeta(s)$. [Benchmark report](empirical-scripts/11.0/results/prime-orders-s0/report.md) · [Counting theorems](ROADMAP.md#step-18--cubic-residue-benchmark-and-dependence)

## 7. Best Evolvo algorithms

The first four formulas are the validation-selected exact rules from the two prime-order runs. The step-20 rules follow below. In the code, `divides_index(p, m, b)` returns the bit for $m\mid I_b(p)$ using a modular power. The functions take primes $p>5$, the common eligible domain of the formula searches.

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

The benchmark counts were cross-checked independently, the main run reproduced its program slots and metrics, and the selected programs were replayed. The four rules express established arithmetic. Together with the finite prime counts and the congruence-only limit, they complete steps 17–19.

**Step-20 rules** (`artin-s0`, `config/artin.json`, policy seeds 0–3). The code below writes them with the `divides_index` helper above. It agrees with SymPy's `is_primitive_root` on every prime in $(31, 2\cdot10^5)$.

```python
def prime_factors(n):
    q, out = 2, []
    while q * q <= n:
        if n % q == 0:
            out.append(q)
            while n % q == 0:
                n //= q
        q += 1
    return out + ([n] if n > 1 else [])


def full_period_events(p):
    """For a prime p > 31: the base-10 full-period bit and the base-5 nesting bit, as Evolvo wrote them."""
    not_local = [1 - divides_index(p, q, 10) for q in (2, 3, 5, 7)]
    no_large = int(not any(divides_index(p, q, 10) for q in prime_factors(p - 1) if q >= 11))
    artin_criterion = not_local[2] * ((not_local[0] * no_large) * not_local[3]) * not_local[1]
    lehmer_nesting = abs(divides_index(p, 5, 5))  # equals divides_index(p, 10, 5)
    return artin_criterion, lehmer_nesting


def kummer_local_factor(q):
    return q / (q - 1)  # q² · P(q | I_b(p)), so [Q(ζ_q, b^(1/q)) : Q] = q(q − 1)
```

- **Artin's criterion.** All four seeds found the product of the five local bits. It has zero errors on 515 646 held-out primes in $(2\cdot10^6,10^7]$ and matches all 64 Boolean rows.
- **The nesting behind 20/19.** All four seeds found $[10\mid I_5(p)]=[5\mid I_5(p)]$, with zero held-out errors and 6/6 consistent rows. Because $5\mid I_5(p)$ forces $p\equiv1\pmod5$ and so $(5/p)=1$, the local factors at 2 and 5 are not $\tfrac12\cdot\tfrac{19}{20}$ but $\tfrac12$. That ratio is the Lehmer–Heilbronn factor $\tfrac{20}{19}$.
- **The Kummer degree.** Regressing $q^2$ times the measured share of $q\mid I_b(p)$ over the 79 primes $q\le401$ gave $q/(q-1)$ on three of four seeds. The held-out mean squared standardized residual is 1.21, consistent with counting noise. The implied degree is $q(q-1)$ to machine precision, and $\prod_q(1-1/(q(q-1)))$ is Artin's constant.
- **Congruences are not enough.** From $[(10/p)=-1]$, $[p\equiv1\bmod3]$, $[p\equiv1\bmod5]$, $[p\equiv1\bmod7]$ and $p\bmod840$, the best evolved formula reaches accuracy 0.825. The train-only lookup reaches 0.875. Any rule of $p\bmod840$ must miss at least 64 411 of the 515 646 held-out primes, because 96 of the 192 classes contain both outcomes. [Step-20 report](empirical-scripts/11.0/results/artin-s0/report.md)

### The same rules as GFSL programs

These are the selected Evolvo programs exactly as saved: the effective instructions in execution order, with the full slot vectors in the linked JSON files. `replay(program, X)` in [`prime_order_formulas.py`](empirical-scripts/11.0/prime_order_formulas.py) runs them. Inputs are numbered `d$0, d$1, …` in the order listed, and constants such as `2.71828` come from Evolvo's value table.

```gfsl
# quadratic_base2 (prime-orders-confirm-s10) — d$0 = |(p mod 8) − 4|, d$1 = 1, d$2 = 2, d$3 = 4
d$4 = MOD(d$3, d$0)
RESULT d$4

# quadratic_joint (prime-orders-confirm-s10) — d$0, d$1, d$2 = [2 | I_2], [2 | I_3], [2 | I_5], d$3 = [p ≡ 1 mod 3]
d$4 = MUL(d$0, d$2)
d$4 = MUL(d$4, d$1)
RESULT d$4

# cubic_joint (prime-orders-s0) — d$0 = [3 | I_2], d$1 = [3 | I_3], d$2 = [p ≡ 1 mod 5]
d$3 = MUL(d$0, d$1)
RESULT d$3

# mixed_joint (prime-orders-s0) — d$0 = [2 | I_5], d$1 = [5 | I_2], d$2 = [p ≡ 1 mod 3]
d$3 = ABS(d$1)
RESULT d$3

# artin_criterion (artin-s0) — d$0…d$3 = [q ∤ I_10] for q = 2, 3, 5, 7, d$4 = [no prime q ≥ 11 divides I_10], d$5 = [p ≡ 1 mod 4]
d$0 = MUL(d$0, d$4)
d$0 = MUL(d$0, d$3)
d$0 = MUL(d$0, d$1)
d$6 = MUL(d$2, d$0)
RESULT d$6

# lehmer_nesting (artin-s0) — d$0 = [2 | I_5], d$1 = [5 | I_5], d$2 = [p ≡ 1 mod 3]
d$3 = ABS(d$1)
RESULT d$3

# kummer_degree (artin-s0) — d$0 = q, d$1 = 1; output q²·P(q | I_b(p)) = q/(q − 1)
d$1 = ADD(-1.0, d$0)
d$2 = DIV(d$0, d$1)
RESULT d$2
```

Run the step-20 and step-21 analyses with:

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/artin.json --run-id artin-s0-repro --stages artin,report
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/explicit.json --run-id explicit-s0-repro --stages explicit,report
```

## 8. Full-period repetends and Artin's conjecture

The repetend of $1/p$ in base $b$ has the maximal length $p-1$ exactly when $b$ is a primitive root modulo $p$. Gauss tabulated these primes for base 10, and Artin conjectured in 1927 that they have a positive density. Artin's criterion is exact: $b$ is a primitive root iff for no prime $q\mid p-1$ is $b$ a $q$th power modulo $p$, that is, iff $p$ splits completely in none of the fields $K_q=\mathbb Q(\zeta_q,b^{1/q})$. Inclusion–exclusion over the squarefree $n$ then gives the candidate density $\sum_n\mu(n)/[K_n:\mathbb Q]$.

Hooley (1967) proved that this is the density, **assuming GRH** for the Dedekind zeta functions of the fields $K_n$. For a base that is not a perfect power, with $\Delta$ the discriminant of $\mathbb Q(\sqrt b)$, the density is Artin's constant $A=\prod_q\bigl(1-\tfrac1{q(q-1)}\bigr)=0.373955813619\ldots$ when $\Delta\not\equiv1\pmod4$, and

$$\delta(b)=A\Bigl(1-\mu(|\Delta|)\prod_{q\mid\Delta}\frac1{q^2-q-1}\Bigr)\qquad(\Delta\equiv1\bmod4).$$

The correction comes from $\sqrt\Delta\in\mathbb Q(\zeta_\Delta)$. When the Lehmers' counts in 1957 disagreed with Artin's original constant for some bases, this correction resolved the discrepancy. [Moree's survey](https://arxiv.org/abs/math/0412262)

**Results through $10^8$** (`artin-s0`, 5 761 444 primes $>31$):

| base | discriminant | full-period primes | rate / A | GRH density / A |
| --- | --- | --- | --- | --- |
| 2 | 8 | 2 154 727 | 1.00009 | 1 |
| 10 | 40 | 2 155 283 | 1.00035 | 1 |
| 5 | 5 | 2 268 685 | **1.05299** | 20/19 = 1.05263 |
| 13 | 13 | 2 168 390 | 1.00644 | 156/155 = 1.00645 |
| 17 | 17 | 2 162 256 | 1.00359 | 272/271 = 1.00369 |
| 29 | 29 | 2 157 549 | 1.00140 | 812/811 = 1.00123 |
| 4, 9, 16 | squares | 0 | 0 | 0 |

- All 15 bases with density $A$ are within 1.26 binomial standard deviations of it.
- The four corrected bases sit 98.3, 11.9, 6.7 and 2.6 standard deviations from $A$, and within 0.65 of the corrected densities.
- The full index law $\#\{p:I_b(p)=t\}$, computed from the Kummer degrees $[\mathbb Q(\zeta_m,b^{1/m}):\mathbb Q]=m\varphi(m)/\varepsilon(m)$, agrees for $t\le12$ and all 16 non-square bases (largest $|z|$ 2.14).
- It predicts a density of zero for $I_5(p)=5$, and no prime has that index.
- Both 2 and 3 are primitive roots for 0.147315 of the primes, against the Kummer-degree product 0.147349.

These numbers measure how close finite frequencies are to conditional densities. Hooley's error term has no explicit constant, so they cannot test his theorem. Unconditionally, no single integer is known to be a primitive root for infinitely many primes. Heath-Brown (1986) proved that at least one of 2, 3, 5 is; here at least one of them is primitive for 73.96% of the primes.

The congruence control explains why GRH for **Dedekind** zeta functions is needed rather than for Dirichlet L-functions. Full period is a splitting condition in non-abelian fields. Every residue class contains primes with $q\mid I_{10}(p)$ for every $q$ outside the modulus. No congruence rule is exact, and the best rule of $p\bmod840$ errs on at least 12.5% of held-out primes. [Step-20 report](empirical-scripts/11.0/results/artin-s0/report.md)

## 9. Zeros behind the halved-repetend counts

For a non-square base $b$, let $\Delta$ be the discriminant of $\mathbb Q(\sqrt b)$ and $\chi_\Delta$ the Kronecker symbol. Euler's criterion gives, for every prime $p\nmid\Delta$,

$$2\mid I_b(p)\iff b^{(p-1)/2}\equiv1\pmod p\iff\chi_\Delta(p)=1.$$

So $\psi(x,\chi_\Delta)=\sum_{n\le x}\Lambda(n)\chi_\Delta(n)$ counts halved repetends minus unhalved ones, weighted by $\log p$ and including prime powers. The explicit formula of analytic number theory expresses it through the zeros $\rho$ of $L(s,\chi_\Delta)$:

$$\psi(x,\chi_\Delta)=-\sum_\rho\frac{x^\rho}\rho-\log x-b(\chi_\Delta)-\tfrac12\log(1-x^{-2}).$$

For $\zeta(s)$ the left side is $\psi(x)$. By Lehmer's theorem, the primes are exactly the $n$ for which $1/n$ has a full-period repetend in some base.

Gaussian smoothing in $\log x$ (width 0.05) makes the zero sum converge absolutely. Step 21 computes both sides independently, from the repetend parities of all primes through $10^8$ and from zeros located on the critical line up to height 150 (`explicit-s0`):

| L-function | base | zeros found ≤ 150 | first zero | largest residual | signal rms |
| --- | --- | --- | --- | --- | --- |
| $L(s,\chi_8)$ | 2 | 102 | 4.899974 | $1.2\cdot10^{-13}$ | 0.40 |
| $L(s,\chi_{12})$ | 3 | 111 | 3.804628 | $9.1\cdot10^{-14}$ | 0.51 |
| $L(s,\chi_5)$ | 5 | 90 | 6.648453 | $2.9\cdot10^{-13}$ | 0.32 |
| $L(s,\chi_{40})$ | 10 | 140 | 2.488210 | $1.0\cdot10^{-13}$ | 0.76 |
| $\zeta(s)$ | full periods | 52 | 14.134725 | $6.9\cdot10^{-10}$ | 0.11 |

- Without the zeros the residuals are 0.22–1.60. The fluctuations of the halved-repetend counts are the zeros, to 13 digits.
- The zero counts match the Riemann–von Mangoldt main term (for example 140 against 139.81), and for $\zeta$ the list equals `mpmath.zetazero`.
- A skipped zero up to height about 110–137 would have shown in the residual.
- The constants check out: $L'(0,\chi_\Delta)$ equals $h\log\varepsilon$, which reproduces the fundamental units $1+\sqrt2$, $2+\sqrt3$ and $(1+\sqrt5)/2$, and twice $\log(3+\sqrt{10})$ for class number 2.

**Chebyshev's bias in repetends.** Rubinstein and Sarnak (1994) assume GRH and the linear independence of the zeros (LI). Under those hypotheses, the share of $x$ (in logarithmic measure) with more unhalved than halved repetends is $\delta=\tfrac12+\tfrac1\pi\int_0^\infty\frac{\sin t}t\prod_{\gamma>0}J_0\bigl(\tfrac{2t}{|\rho|}\bigr)dt$. Computed from the zeros above:

| base | δ (GRH + LI) | measured share on $[2,10^8]$ | first prime where halved repetends lead |
| --- | --- | --- | --- |
| 5 | 0.9954 | 1.0000 | none below $10^8$ |
| 2 | 0.9822 | 0.9748 | 11 100 143 |
| 3 | 0.9602 | 0.9215 | 61 463 |
| 10 | 0.8778 | 0.8107 | 3 |

So unhalved periods lead most of the time. In base 5, halved periods never lead below $10^8$. The bias is weaker for larger conductors, whose low zeros add variance. The measured shares cover about 18 units of $\log x$ and are finite-range descriptions, not limits. [Step-21 report](empirical-scripts/11.0/results/explicit-s0/report.md)

## 10. What this means for the Riemann hypothesis

The repository now connects repetend digits to the zeros of $\zeta(s)$ and of Dirichlet L-functions through checked steps. These steps are of different logical kinds, and only the first is unconditional.

1. **Exact identities, checked numerically:**
   - Euler's criterion turns repetend parity into a quadratic character.
   - Lehmer's criterion turns full periods into primality.
   - Artin's criterion turns full periods into non-splitting in Kummer fields.
   - The explicit formula turns the resulting counts into sums over zeros. Here it holds to $10^{-13}$.
2. **Conditional theorems the data agree with:**
   - Hooley's densities for full-period repetends assume GRH for Dedekind zeta functions of Kummer fields.
   - The Rubinstein–Sarnak bias assumes GRH and LI.
   - The finite data agree with both. Agreement of this kind is expected whether or not the hypotheses hold far beyond the range, because both are statements about limits.
3. **Equivalent forms of RH and GRH in repetend language.** These are reformulations of RH, not new evidence for it.
   - RH is equivalent to $\theta(x)-x=O(x^{1/2+\varepsilon})$, where $\theta$ sums $\log n$ over the $n\le x$ with a full-period repetend. Schoenfeld showed that RH gives the explicit bound $|\theta(x)-x|<\sqrt x\log^2x/(8\pi)$ for $x\ge599$. On $[599,10^8]$ the largest ratio to that bound is 0.9836, at $x\approx797$.
   - GRH for $L(s,\chi_\Delta)$ is equivalent to the halved-repetend imbalance $\psi(x,\chi_\Delta)$ being $O(x^{1/2+\varepsilon})$. On this range $|\psi(x,\chi_\Delta)|\le2.45\sqrt x$.
4. **Why this is not progress toward a proof.**
   - A finite range can only find a counterexample.
   - The ranges checked here are tiny next to existing rigorous verifications. Platt and Trudgian checked the zeros of $\zeta$ to height $3\cdot10^{12}$, and Platt checked GRH for every Dirichlet L-function of conductor up to 400 000. [Platt–Trudgian](https://arxiv.org/abs/2004.09765) · [Platt](https://doi.org/10.1090/mcom/3077)
   - The repetend observables are exact reformulations of prime counts in residue classes and splitting fields. A bound on them is therefore exactly as hard as the corresponding RH or GRH statement.
   - A proof needs an argument valid for all $x$ or all heights, and nothing computed here supplies one.

What the work does contribute:
- **A tested route** from digit spectra to the zeros of L-functions, with every link stated as a theorem, a hypothesis or a finite check.
- **A benchmark:** counts, densities, zeros and residuals with independent cross-checks and one command per step.
- **Worked examples** of how entropy-style claims fail once length and size are controlled.
- **Exact formulas** that the Evolvo searches recover when the inputs carry the arithmetic, and their stated limits when the inputs do not.

That is a finished, reproducible piece of experimental number theory. It is not, and cannot become through more computation, a proof of RH.

## 11. Digits of n, prime densities and practical conclusions

Step 22 extends the Evolvo searches to three new kinds of input. The first is the decimal digits of $n$ itself, with no $1/n$: last digit, digit sum, alternating digit sum, sum and alternating sum of 3-digit blocks, digit count, and $\log n$. The second is the repetend lengths $L_2(n), L_{10}(n)$ of $1/n$. The third is **prime-density predictions as Evolvo variables**:
- the Cramér density $1/\log n$;
- the digit-sieve density $W(n)=[\gcd(n,2\cdot3\cdot5\cdot7\cdot11\cdot13\cdot37)=1]\cdot5.358362/\log n$;
- the Hardy–Ramanujan mean $\log\log n+B_1$ and its sieve version $M(n)=\sum_{p\in S}[p\mid n]+\log\log n+B_1-\sum_{p\in S}1/p$;
- $\mathrm{li}(x)\log x/x$ and $\mathrm{li}(\sqrt x)\log x/x$.

The primes $S=\{2,3,5,7,11,13,37\}$ are exactly those that base-10 digit sums can see, because they divide $10$, $9$, $11$, $999$ or $1001$. Each target runs with and without the density inputs.

Two runs, both with the PyTorch policy guide and size-separated train, validation and test ranges:
- `digits-s0`: `config/digits.json`, seed 0, 6 policy seeds, population 200 × 120 generations.
- `digits-confirm-s1`: `config/digits-confirm.json`, seed 1, 8 policy seeds, population 400 × 300 generations, on the six cases that still had room.

The table gives, per case, the program with the lowest validation error across both runs, scored once on the test range. All counts are independently checked:
- the digit sums reproduce the residues modulo 3, 11, 37, 7 and 13 on every row;
- $\omega$ is checked against SymPy on a sample;
- the repetend lengths are checked as periods on every row and against SymPy's `n_order` on a sample;
- $\pi(x)$ is computed by SymPy at every point and by the sieve wherever $x\le10^8$.

[Report `digits-s0`](empirical-scripts/11.0/results/digits-s0/report.md) · [Report `digits-confirm-s1`](empirical-scripts/11.0/results/digits-confirm-s1/report.md) · [Saved programs](empirical-scripts/11.0/results/digits-confirm-s1/digits.json)

| Target (test range) | Selected Evolvo formula | Held-out score | Best control |
| --- | --- | --- | --- |
| $n$ prime, digits only ($10^6$–$2\cdot10^6$) | last-digit rule (GFSL below) | Brier 0.0593, AUC 0.823 | $W(n)$: Brier 0.0439, AUC 0.938; gradient boosting 0.0527 |
| $n$ prime, digits + densities | $W(n)$ | Brier 0.0439; predicts 70 428 primes, actual 70 435 | Cramér $1/\log n$: 0.0655 |
| $n$ prime, repetend lengths of $1/n$ | $1/(n \bmod L_2)^{10}$ = Fermat base 2 | 93 errors / 400 000 = the 93 base-2 pseudoprimes | Fermat in bases 2 and 10: 22 errors |
| $\omega(n)$, digits only | $e+(-\log2)^{d_0}$ ($d_0$ = last digit) | MSE 0.828 | gradient boosting 0.492 |
| $\omega(n)$, digits + densities | $M+e^{-M}$ | MSE **0.371** | sieve mean $M$: 0.381; gradient boosting 0.375 |
| $\pi(x)$, from $\log x$ ($1.3\cdot10^8$–$10^{12}$) | $\dfrac{x(\log x-1)}{\log x\,(\log x-2)}$ (simplest candidate) | max rel. error $3.8\cdot10^{-4}$ | Legendre $1.6\cdot10^{-3}$; $\mathrm{li}$ $1.1\cdot10^{-4}$ |
| $\pi(x)$, with density inputs | $\mathrm{li}(x)\big/\sqrt{1+\mathrm{li}(\sqrt x)\log x/x}$ | max rel. error **$1.26\cdot10^{-5}$** | Riemann $R(x)$: $1.31\cdot10^{-5}$; $\mathrm{li}(x)-\tfrac12\mathrm{li}(\sqrt x)$: $1.68\cdot10^{-5}$ |
| gap to the next prime ($10^7$–$10^8$) | $\log p$ | mean 17.656 vs actual 17.658, MSE 226 | train mean: MSE 250 |
| next prime ends in the same digit | $0.1+1/\log p$ | Brier 0.1433 | train fit $0.2208-0.8475/\log p$: 0.1430; equidistribution 1/4: 0.1490 |

### The programs in GFSL

Each block is the selected program exactly as Evolvo saved it: the effective instructions in execution order, then the output register. `replay(program, X)` in [`prime_order_formulas.py`](empirical-scripts/11.0/prime_order_formulas.py) runs the saved slot vectors from `digits.json`.

```gfsl
# n prime from the digits of n (digits-confirm-s1, seed 6) — d$0 = last decimal digit
d$8 = MOD(d$0, 2.0)
d$5 = MUL(d$8, 2.71828)
d$8 = MOD(d$0, 2.71828)
d$5 = MUL(d$8, d$5)
d$8 = DIV(0.5, d$5)
d$8 = MOD(d$8, 0.5)
RESULT d$8

# n prime from digits + densities (digits-s0, seed 0) — d$9 = sieve density W(n)
d$10 = DIV(d$9, 1.0)
RESULT d$10

# n prime from the repetend lengths of 1/n (digits-confirm-s1, seed 3) — d$0 = n, d$1 = L_2(n)
d$2 = MOD(d$0, d$1)
d$0 = POW(d$2, 10.0)
d$2 = LOG(2.71828)
d$4 = DIV(d$2, d$0)
RESULT d$4

# ω(n) from digits + densities (digits-s0, seed 2) — d$9 = sieve mean M(n)
d$1 = POW(2.71828, d$9)
d$10 = DIV(-1.0, d$1)
d$10 = SUB(d$9, d$10)
RESULT d$10

# ω(n) from the digits of n (digits-confirm-s1, seed 5) — d$0 = last digit, d$2 = alternating digit sum
d$2 = MUL(d$2, d$0)
d$0 = DIV(d$2, d$0)
d$0 = DIV(d$2, d$0)
d$7 = MOD(0.5, 1.0)
d$5 = LOG(d$7)
d$5 = POW(d$5, d$0)
d$8 = ADD(d$5, 2.71828)
RESULT d$8

# π(x)·log x / x from log x (digits-confirm-s1, seed 2) — d$0 = log x, d$1 = 1
d$0 = SUB(d$1, d$0)
d$1 = ADD(1.0, d$0)
d$2 = DIV(d$0, d$1)
RESULT d$2

# π(x)·log x / x with density inputs (digits-confirm-s1, seed 3) — d$1 = 1, d$3 = li(x)·log x/x, d$4 = li(√x)·log x/x
d$1 = ADD(d$4, d$1)
d$1 = SQRT(d$1)
d$5 = DIV(d$3, d$1)
RESULT d$5

# gap to the next prime (digits-s0, seed 0) — d$0 = log p
d$4 = ADD(0.0, d$0)
RESULT d$4

# P(next prime ends in the same digit as p) (digits-confirm-s1, seed 4) — d$1 = 1/log p, d$3 = 1
d$4 = DIV(d$3, 10.0)
d$4 = ADD(d$4, d$1)
RESULT d$4
```

**Two selection caveats:**
- **The $\pi(x)$ formula without densities.** By the selection rule, which uses lowest validation error, `digits-s0` wins. Its 12-instruction program built from powers of $\log x$ has validation NMSE $2.96\cdot10^{-4}$ but held-out maximum error $1.09\cdot10^{-3}$. The table shows the 3-instruction candidate from `digits-confirm-s1`, whose validation NMSE is $3.68\cdot10^{-4}$, because it is the one that explains something: $(u-1)/(u-2)=1+1/u+2/u^2+4/u^3+\dots$ with $u=\log x$ matches the asymptotic series of $\mathrm{li}$, $1+1/u+2/u^2+6/u^3+\dots$, to second order. Choosing it on its test error would be test leakage, so it is labelled as a candidate.
- **The last-digit rule.** The validation-selected $0.1+1/\log p$ predicts 798 695 same-digit pairs against 881 553 actual. The one-parameter train fit is better calibrated, with 880 208 predicted. Evolvo found the direction of the effect but not its size.

### Practical conclusions

1. **Reading primality off $1/n$.** If the repetend length of $1/n$ divides $n-1$, then $n$ is prime or a Fermat pseudoprime. That is Fermat's test, and Evolvo found it from the raw lengths: $(n \bmod L_2)=1$. In $(10^6,2\cdot10^6]$ with $\gcd(n,10)=1$, base 2 alone misses 93 composites, base 10 alone 110, and bases 2 and 10 together 22. All three counts were recounted directly with modular powers and SymPy. To remove those 22, use the step-13 decoder, which factors $n$ from the same lengths. So the repetend gives a working probable-prime test, and with 19 bases an exact factorization through $3.7\cdot10^9$.
2. **The digits of $n$ only reveal divisibility.** Digit sums in base 10 are divisibility tests for $2,3,5,7,11,13,37$, because $10^k\equiv\pm1$ modulo $9, 11, 999, 1001$. Beyond that they say nothing about primality. The best calibrated probability from the digits is therefore the sieve density $W(n)$. It predicted 70 428 primes in $(10^6,2\cdot10^6]$ against 70 435 actual, an error of 0.01%. Neither Evolvo nor gradient boosting beat it. Given $W(n)$ as a variable, Evolvo keeps it unchanged. Without it, Evolvo finds only the last-digit rule, so composing the digit-sum indicators is beyond what the searches built unaided.
3. **Counting primes with an elementary formula.** $\pi(x)\approx x(\log x-1)/(\log x(\log x-2))$ was fitted only on $x\le10^6$. It stays within $3.8\cdot10^{-4}$ up to $10^{12}$, four times closer than Legendre's $x/(\log x-1.08366)$. With the density inputs Evolvo writes Riemann's correction in its own form, $\mathrm{li}(x)/\sqrt{1+\mathrm{li}(\sqrt x)\log x/x}\approx\mathrm{li}(x)-\tfrac12\mathrm{li}(\sqrt x)$. It reaches the accuracy of $R(x)$: maximum error $1.26\cdot10^{-5}$ against $1.31\cdot10^{-5}$ over $1.3\cdot10^8$–$10^{12}$. Fitting a Legendre-type constant on $x\le10^6$ gives $A=1.040$, not Legendre's 1.08366; the best constant drifts toward 1 as $x$ grows.
4. **Expected gap and ω.** The expected gap after $p$ is $\log p$. Evolvo selects exactly that, and it matches the mean of 5.1 million gaps in $(10^7,10^8]$ to 0.01%. The best formula for $\omega(n)$ is the sieve mean plus a small correction, $M+e^{-M}$.
5. **The next prime's last digit.** In $(10^7,10^8]$ the next prime ends in the same digit as $p$ only 17.3% of the time, not 25%. The bias shrinks slowly with $\log p$, as Lemke Oliver and Soundararajan conjectured (PNAS 113, 2016). A forecaster that assumes 25% overestimates repeats by 45%.

None of these goes beyond known theory. Each says which classical statement the data and Evolvo land on, and how accurate it is on a stated range. The density functions are reusable in [`digit_primes.py`](empirical-scripts/11.0/digit_primes.py): `cramer_density`, `sieve_density`, `hardy_ramanujan_mean`, `sieve_omega_mean` and `li_ratio`. They are the starting variables for the next step.

Run both searches, about 18 and 88 minutes on 7 workers:

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/digits.json --run-id digits-s0-repro --stages digits,report
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/digits-confirm.json --run-id digits-confirm-s1-repro --stages digits,report
```
