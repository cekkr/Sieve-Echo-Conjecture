"""The Sieve Echo decoder: the factorization of n read from the repetends of 1/n in 19 bases.

This is the final algorithm of v11 (ROADMAP steps 13 and 15; Propositions 1.7 and 1.8 of
studies/9). It is derived, not evolved: it needs no training and no Evolvo genome. Pure Python, no
dependencies. It is exact for every n < 3 700 000 259 (runs `decode-s0`, `failures-s0`).

    python empirical-scripts/11.0/sieve_echo_decoder.py 2794155 1392776385

The decoder takes the repetend lengths as input. `repetend_length` below finds them by long division,
which takes L steps (L can be close to n): that is the expensive part, and the reason this is not a
fast factoring method. The decoding itself is polynomial in log n.
"""

from __future__ import annotations

import math
import sys

BASES = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31]


def repetend_length(n: int, b: int) -> int:
    """Length of the repetend of 1/n in base b (gcd(b, n) = 1): the remainders of the long division return to 1."""
    r, L = b % n, 1
    while r != 1 % n:
        r, L = r * b % n, L + 1
    return L


def primes_of(m: int) -> list[int]:
    """The primes dividing m (trial division; m is a repetend length, so m < n)."""
    out, p = [], 2
    while p * p <= m:
        if m % p == 0:
            out.append(p)
            while m % p == 0:
                m //= p
        p += 1
    return out + ([m] if m > 1 else [])


def coprime_parts(parts: list[int]) -> list[int]:
    """Split a list of divisors of n by their gcds until they are pairwise coprime."""
    parts = [x for x in parts if x > 1]
    while True:
        pair = next(((i, j) for i in range(len(parts)) for j in range(i + 1, len(parts))
                     if math.gcd(parts[i], parts[j]) > 1), None)
        if pair is None:
            return sorted(set(parts))
        i, j = pair
        g = math.gcd(parts[i], parts[j])
        parts = [x for k, x in enumerate(parts) if k not in pair] + [y for y in (g, parts[i] // g, parts[j] // g) if y > 1]


def decode(n: int, lengths: dict[int, int]) -> dict[int, int]:
    """{factor: exponent} of n from its repetend lengths {b: L_b}; a base missing from `lengths` shares a factor with n.

    The factors are the primes of n whenever the decoder is exact (every n < 3.7e9); otherwise a factor is a
    product of primes that the bases could not separate, and len(result) < ω(n).
    """
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


def exponent_of(f: int, n: int) -> int:
    """The largest e with f^e | n."""
    e = 0
    while n % f == 0:
        n //= f
        e += 1
    return e


def factor(n: int) -> dict[int, int]:
    """Repetend lengths of 1/n in the 19 bases, then decode."""
    return decode(n, {b: repetend_length(n, b) for b in BASES if math.gcd(b, n) == 1})


def is_prime(p: int) -> bool:
    """Trial division: an independent check of the output, not part of the decoder."""
    return p > 1 and all(p % d for d in range(2, math.isqrt(p) + 1))


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["2047", "1001", "1024", "214935"]:
        n = int(arg)
        f = factor(n)
        ok = math.prod(p**e for p, e in f.items()) == n and all(is_prime(p) for p in f)
        print(f"{n} = " + " · ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in f.items())
              + f"   ω = {len(f)}   check: {'product = n, all factors prime' if ok else 'FAILED'}")
