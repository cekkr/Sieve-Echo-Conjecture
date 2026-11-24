"""Step 20: Kummer-degree densities, reciprocity, counts and the Evolvo replay contract."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artin
import artin_formulas
import failures
import runtime

A = 0.37395581361920228


class TestArtinDensities(unittest.TestCase):
    def test_closed_form_matches_kummer_inclusion_exclusion(self):
        for b in range(2, 200):
            if artin.power_exponent(b) == 1:
                self.assertAlmostEqual(artin.moree_density(b, A), artin.index_density(b, 1, A), delta=1e-13, msg=b)

    def test_hand_values(self):
        self.assertAlmostEqual(artin.moree_density(5, A), A * 20 / 19, delta=1e-15)
        self.assertAlmostEqual(artin.moree_density(13, A), A * (1 + 1 / 155), delta=1e-15)
        self.assertAlmostEqual(artin.moree_density(8, A), 3 * A / 5, delta=1e-15)
        for b in (2, 3, 6, 7, 10, 12):
            self.assertAlmostEqual(artin.moree_density(b, A), A, delta=1e-15)
        for b in (4, 9, 16, 25):
            self.assertEqual(artin.moree_density(b, A), 0.0)
        self.assertAlmostEqual(artin.index_density(2, 4, A), A / 8, delta=1e-15)
        self.assertEqual(artin.index_density(5, 5, A), 0.0)  # 5 | I_5(p) forces 2 | I_5(p)

    def test_index_law_is_a_distribution(self):
        for b in (2, 5, 10, 12):
            d = [artin.index_density(b, t, A) for t in range(1, 20001)]
            self.assertAlmostEqual(sum(d), 1.0, delta=1e-4)
            self.assertAlmostEqual(sum(d[1::2]), 0.5, delta=1e-4)  # 2 | I ⇔ b is a square mod p

    def test_kummer_degrees(self):
        self.assertEqual(artin.kummer_degree(2, 8), 16)  # √2 ∈ Q(ζ_8)
        self.assertEqual(artin.kummer_degree(5, 10), 20)  # √5 ∈ Q(ζ_5)
        self.assertEqual(artin.kummer_degree(3, 12), 24)
        self.assertEqual(artin.kummer_degree(2, 3), 6)
        self.assertEqual(artin.kummer_degree(10, 40), 320)
        self.assertEqual(artin.discriminant(12), 12)
        self.assertEqual(artin.discriminant(5), 5)
        self.assertEqual(artin.discriminant(10), 40)

    def test_pair_density_local_factor_at_2(self):
        # Both 2 and 3 non-residues: p ≡ 5, 19 (mod 24), two of the eight reduced classes.
        classes = [r for r in range(24) if math.gcd(r, 24) == 1]
        both = [r for r in classes if r % 8 in (3, 5) and r % 12 in (5, 7)]
        self.assertEqual(both, [5, 19])
        self.assertAlmostEqual(1 - (2 * 2 - 1) / (4 * 1), len(both) / len(classes))

    def test_artin_constant(self):
        self.assertAlmostEqual(artin.artin_constant(failures.sieve_primes(2, 10 ** 6)), artin.ARTIN_CONSTANT_10_DIGITS, delta=2e-10)

    def test_reciprocity_matches_sympy(self):
        from sympy import legendre_symbol

        P = failures.sieve_primes(37, 5000)
        for b in (2, 3, 5, 6, 10, 12, 14, 15, 29, 31):
            want = [legendre_symbol(b, int(p)) for p in P]
            self.assertEqual(artin.legendre_by_reciprocity(b, P).tolist(), want, msg=b)


class TestArtinStage(unittest.TestCase):
    def test_counts_against_sympy(self):
        from sympy.ntheory import is_primitive_root

        with tempfile.TemporaryDirectory() as tmp:
            out = artin.run_stage({"prime_limit": 20000, "cutoffs": [5000, 20000], "workers": 1,
                                   "sympy_order_limit": 2000, "sympy_sample": 50}, 0, Path(tmp), log=lambda _: None)
        P = failures.sieve_primes(37, 20000)
        for b in (2, 5, 10, 13):
            want = sum(1 for p in P.tolist() if is_primitive_root(b, p))
            self.assertEqual(out["rows"][-1]["bases"][str(b)]["count"], want)
        self.assertEqual(out["index_law"]["5"]["zero_density_t"], [5])
        self.assertEqual(out["index_law"]["5"]["counts"][4], 0)
        self.assertTrue(out["index_law"]["4"]["index_always_even"])
        self.assertEqual(out["rows"][-1]["bases"]["9"]["count"], 0)

    def test_kummer_check_replays_degree_formula(self):
        runtime.bootstrap_evolvo()
        from evolvo import Category, DataType, Operation
        from evolvo.slots import pack_type_index

        q_, one, out_ = [pack_type_index(DataType.DECIMAL, i) for i in range(3)]
        program = {"inputs": ["q", "1"], "output_index": 2, "slots": [  # q / (q − 1)
            [Category.VARIABLE, out_, Operation.SUB, Category.VARIABLE, q_, Category.VARIABLE, one],
            [Category.VARIABLE, out_, Operation.DIV, Category.VARIABLE, q_, Category.VARIABLE, out_],
        ]}
        qs = failures.sieve_primes(2, 400)
        local = {"q": qs.tolist(), "rates": (1 / (qs * (qs - 1.0))).tolist(), "pairs": 10 ** 9}
        check = artin_formulas.kummer_check(program, local, qs, A)
        self.assertLess(check["max_relative_difference_of_degree_from_q(q-1)"], 1e-12)
        self.assertAlmostEqual(check["implied_artin_product"], check["artin_product_same_primes"], delta=1e-12)
        self.assertLess(check["test"]["max_abs_standardized_residual"], 1e-6)
        self.assertEqual(check["train"]["primes"] + check["validation"]["primes"] + check["test"]["primes"], qs.size)


if __name__ == "__main__":
    unittest.main()
