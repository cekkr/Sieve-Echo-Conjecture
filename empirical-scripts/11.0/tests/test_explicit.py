"""Step 21: characters, L-function constants, zeros, the smoothed explicit formula and the bias integral."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import explicit
import failures


class TestCharacters(unittest.TestCase):
    def test_tables(self):
        self.assertEqual(explicit.character_table(8), [0, 1, 0, -1, 0, -1, 0, 1])
        self.assertEqual(explicit.character_table(5), [0, 1, -1, -1, 1])
        for D in (8, 12, 5, 40):
            chi = explicit.character_table(D)
            self.assertEqual(chi[D - 1], 1)  # even
            self.assertEqual(sum(chi), 0)

    def test_halved_repetends_are_the_character(self):
        P = failures.sieve_primes(2, 20000)
        chis = explicit.halved_characters(P, [2, 3, 5, 10], 1)  # raises on any disagreement
        for b, chi in chis.items():
            for p, c in zip(P.tolist()[3:200], chi.tolist()[3:200]):
                if c:
                    self.assertEqual(c == 1, pow(b, (p - 1) // 2, p) == 1)

    def test_constants(self):
        for D, unit in ((8, 1 + math.sqrt(2)), (5, (1 + math.sqrt(5)) / 2), (12, 2 + math.sqrt(3))):
            c = explicit.b_constant(explicit.character_table(D), 25)
            self.assertAlmostEqual(c["direct"], c["functional_equation"], delta=1e-12)
            self.assertAlmostEqual(c["L_prime_0"], math.log(unit), delta=1e-12)  # class number 1


class TestZerosAndFormula(unittest.TestCase):
    def test_zero_finder_matches_mpmath_for_zeta(self):
        import mpmath

        z = explicit.find_zeros((None, 40.0, 0.05, 20))
        ref = [float(mpmath.zetazero(k).imag) for k in range(1, 7)]
        np.testing.assert_allclose(z["zeros"], ref, atol=1e-9)
        self.assertLess(abs(z["S_at_zeros"]["mean"]), 0.3)

    def test_smoothed_explicit_formula(self):
        X, sigma = 200_000, 0.15
        P = failures.sieve_primes(2, X)
        chi8 = explicit.character_table(8)
        zs = explicit.find_zeros((chi8, 60.0, 0.05, 20))
        self.assertAlmostEqual(zs["zeros"][0], 4.8999739970, delta=1e-8)
        u = np.linspace(math.log(100), math.log(X) - 9 * sigma, 60)
        chi = explicit.halved_characters(P, [2], 1)[2]
        lg, w = explicit.prime_powers(P, chi, X)
        meas = explicit.smoothed_measured(lg, w, u, sigma, principal=False)
        b = explicit.b_constant(chi8, 25)["direct"]
        pred = explicit.smoothed_predicted(zs["zeros"], u, sigma, b)
        self.assertLess(np.max(np.abs(meas - pred["zero_sum"] - pred["rest"])), 1e-8)
        self.assertGreater(np.max(np.abs(meas - pred["rest"])), 0.1)  # the zeros carry the signal
        import mpmath

        zz = [float(mpmath.zetazero(k).imag) for k in range(1, 16)]  # up to height 65.1
        lg, w = explicit.prime_powers(P, np.ones(P.size, dtype=np.int8), X)
        meas = explicit.smoothed_measured(lg, w, u, sigma, principal=True)
        pred = explicit.smoothed_predicted(zz, u, sigma, None)
        self.assertLess(np.max(np.abs(meas - pred["zero_sum"] - pred["rest"])), 1e-8)


class TestBiasAndBounds(unittest.TestCase):
    def test_bias_integral_single_term(self):
        # X = 1 + 2 cos Θ: P(X > 0) = 1 − arccos(1/2)/π = 2/3.
        g = math.sqrt(3) / 2  # |ρ| = 1
        r = explicit.rubinstein_sarnak([g], 1, 1e12, 4000.0, 0.002)
        self.assertAlmostEqual(r["delta"], 2 / 3, delta=2e-3)

    def test_race_counts(self):
        P = np.array([3, 5, 7, 11, 13])
        r = explicit.race(P, np.array([-1, 1, 1, -1, -1], dtype=np.int8), 20)
        self.assertEqual(r["lead_changes"], 2)  # D = 1, 0, −1, 0, 1
        self.assertEqual(r["first_halved_lead_at"], 7)
        self.assertEqual(r["final_difference"], 1)

    def test_schoenfeld_bound_holds_through_a_million(self):
        s = explicit.schoenfeld(failures.sieve_primes(2, 10 ** 6), 10 ** 6)
        self.assertLess(s["max_ratio_to_schoenfeld_bound"], 1.0)


if __name__ == "__main__":
    unittest.main()
