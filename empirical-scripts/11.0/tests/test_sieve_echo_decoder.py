"""Tests for the standalone decoder (the final v11 algorithm) against SymPy factorizations."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import sympy

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import sieve_echo_decoder as sed  # noqa: E402


def lengths(n):
    return {b: int(sympy.n_order(b, n)) for b in sed.BASES if math.gcd(b, n) == 1}


class StandaloneDecoderTests(unittest.TestCase):
    def test_exact_factorization_below_20000(self):
        for n in range(2, 20000):
            self.assertEqual(sed.factor(n), dict(sympy.factorint(n)), n)

    def test_repetend_length_is_the_order(self):
        for n in range(3, 3000, 7):
            for b in sed.BASES:
                if math.gcd(b, n) == 1:
                    self.assertEqual(sed.repetend_length(n, b), sympy.n_order(b, n))

    def test_prefix_failures_are_decoded_with_all_19_bases(self):
        # least failures of 9…18-base prefixes (failures-s0) and the 9 lengths-only exceptions of decode-s0
        for n in (214935, 2794155, 47500635, 60555495, 1392776385, 11532975090, 3741870, 2 * 3741870):
            self.assertEqual(sed.decode(n, lengths(n)), dict(sympy.factorint(n)), n)

    def test_a_failure_is_reported_as_an_unsplit_factor(self):
        # with only 2, 4, 11 left (n = 3·5·7·23·89 and the first 10 bases) 23·89 stays one factor
        saved = sed.BASES
        try:
            sed.BASES = saved[:10]
            n = 214935
            self.assertEqual(sed.decode(n, lengths(n)), {3: 1, 5: 1, 7: 1, 23 * 89: 1})
        finally:
            sed.BASES = saved


if __name__ == "__main__":
    unittest.main()
