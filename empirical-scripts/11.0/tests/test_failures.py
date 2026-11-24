"""Tests for step 15: the failure set of the decoder (Proposition 1.8) and its certificate."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import decode  # noqa: E402
import failures  # noqa: E402
import ndr  # noqa: E402

SPF = ndr.spf_sieve(40000)


def decoded(n, bases):
    f = ndr.factorize(n, SPF)
    lam, lp = ndr.order_context(n, SPF)
    L = [0 if math.gcd(b, n) != 1 else ndr.multiplicative_order(b, n, lam, lp) for b in bases]
    return decode.decode(n, bases, L, lambda t: decode.prime_divisors(t, lp), preperiod=True), f


def colliding_pair(n, bases, f):
    PB = failures.base_primes(bases)
    cop = [b for b in bases if math.gcd(b, n) == 1]
    big = [p for p, _ in f if p not in PB]
    vec = {p: tuple(ndr.multiplicative_order(b % p, p, p - 1, [q for q, _ in ndr.factorize(p - 1, SPF)]) if p > 2 else 1 for b in cop) for p in big}
    return len(set(vec.values())) < len(vec)


class FailureTheoryTests(unittest.TestCase):
    def test_sieve(self):
        import sympy

        self.assertEqual(failures.sieve_primes(10, 200).tolist(), list(sympy.primerange(10, 201)))
        self.assertTrue(failures.prime_closed([2, 3, 4, 6]))
        self.assertFalse(failures.prime_closed([4, 6]))

    def test_proposition_18_failure_iff_colliding_pair(self):
        for bases in ([2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5, 6]):
            fails = 0
            for n in range(2, 25000):
                got, f = decoded(n, bases)
                pair = colliding_pair(n, bases, f)
                if all(e == 1 for _, e in f):
                    self.assertEqual(got != len(f), pair, (bases, n))
                elif got != len(f):
                    self.assertTrue(pair, (bases, n))  # every failure has a colliding pair
                fails += got != len(f)
            self.assertGreater(fails, 0)

    def test_certificate_matches_brute_force(self):
        P = failures.sieve_primes(2, 60000)
        for k in range(1, 9):
            bk = decode.BASES19[:k]
            Pk = P[~np.isin(P, failures.base_primes(bk))]
            cert = failures.certificate(bk, Pk, failures.order_matrix(Pk, bk, 1), 60000)
            self.assertTrue(cert["exact"], k)
            n = cert["first_failure"]["n"]
            self.assertEqual(failures.first_failure_brute(bk, n, 1), n, k)

    def test_decodes(self):
        self.assertFalse(failures.decodes(2794155, decode.BASES19[:14]))  # 3·5·7·13·23·89: 23·89 = 2^11 − 1
        self.assertTrue(failures.decodes(2794155, decode.BASES19))

    def test_collisions(self):
        P = np.array([5, 7, 11, 13])
        O = np.array([[4, 1], [3, 2], [4, 1], [3, 2]])
        c = failures.collisions(P, O, [0, 1])
        self.assertEqual((c["groups"], c["min_product"], c["pair"]), (2, 55, [5, 11]))
        self.assertIsNone(failures.collisions(P, O * np.arange(1, 5)[:, None], [0, 1])["min_product"])


if __name__ == "__main__":
    unittest.main()
