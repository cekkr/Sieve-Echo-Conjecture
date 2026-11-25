"""Step 22: digit statistics, prime-density inputs, exact data and the GFSL listing contract."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import digit_primes
import failures
import search_formulas


class TestDigitStatistics(unittest.TestCase):
    def test_hand_values(self):
        d = digit_primes.digit_statistics(np.array([1234567]))
        self.assertEqual(int(d["sum"][0]), 28)
        self.assertEqual(int(d["alternating"][0]), 7 - 6 + 5 - 4 + 3 - 2 + 1)
        self.assertEqual(int(d["count"][0]), 7)
        b = digit_primes.digit_statistics(np.array([1234567]), 1000)
        self.assertEqual(int(b["sum"][0]), 567 + 234 + 1)
        self.assertEqual(int(b["alternating"][0]), 567 - 234 + 1)

    def test_divisibility_rules(self):
        n = np.arange(1, 300_000)
        d = digit_primes.digit_inputs(n)
        self.assertTrue(np.array_equal(d["digit sum"] % 9, n % 9))
        self.assertTrue(np.array_equal(d["alternating digit sum"] % 11, n % 11))
        self.assertTrue(np.array_equal(d["3-digit block sum"] % 999, n % 999))
        self.assertTrue(np.array_equal(d["alternating 3-digit block sum"] % 1001, n % 1001))
        self.assertTrue(np.array_equal(d["digit count"], np.floor(np.log10(n)).astype(int) + 1))


class TestDensities(unittest.TestCase):
    def test_sieve_density_is_calibrated(self):
        n = np.arange(100_001, 1_000_001)
        P = failures.sieve_primes(2, n[-1])
        primes = np.isin(n, P)
        w = digit_primes.sieve_density(n)
        self.assertEqual(int(np.count_nonzero(primes & (w == 0))), 0)  # no prime is sieved out
        self.assertLess(abs(w.sum() / primes.sum() - 1), 0.02)
        self.assertAlmostEqual(digit_primes.SIEVE_FACTOR, math.prod(p / (p - 1) for p in (2, 3, 5, 7, 11, 13, 37)))

    def test_omega_sieve_and_means(self):
        from sympy import primefactors

        P = failures.sieve_primes(2, 5000)
        om = digit_primes.omega_sieve(5000, P)
        for n in range(2, 5001):
            self.assertEqual(int(om[n]), len(primefactors(n)), n)
        n = np.arange(10**5, 2 * 10**5)
        self.assertLess(abs(om_mean(n) - digit_primes.hardy_ramanujan_mean(n).mean()), 0.1)


def om_mean(n):
    P = failures.sieve_primes(2, int(n[-1]))
    return digit_primes.omega_sieve(int(n[-1]), P)[n].mean()


class TestExactData(unittest.TestCase):
    def test_repetend_lengths_match_sympy(self):
        from sympy.ntheory import n_order

        n = np.array([v for v in range(3, 3000) if v % 2 and v % 5])
        L = digit_primes.repetend_lengths(n, 3000, workers=1)
        for v, (a, b) in zip(n.tolist(), L.tolist()):
            self.assertEqual((a, b), (n_order(2, v), n_order(10, v)))
        digit_primes.check_orders(n, L, 50, 0)

    def test_prime_counts(self):
        P = failures.sieve_primes(2, 10**6)
        xs = np.array([10, 100, 1000, 10**6, 10**7])
        pi, info = digit_primes.prime_counts(xs, P, 10**6)
        self.assertEqual(pi.tolist(), [4, 25, 168, 78498, 664579])
        self.assertEqual(info["sieve_checked_points"], 4)


class TestListingAndStage(unittest.TestCase):
    def test_gfsl_code(self):
        code = search_formulas.gfsl_code(["p mod 8", "4"], ["✓ d$2 = MOD(d$1, d$0)"], 2)
        self.assertEqual(code, ["# d$0 = p mod 8", "# d$1 = 4", "d$2 = MOD(d$1, d$0)", "RESULT d$2"])

    def test_stage_end_to_end(self):
        cfg = {"n_min": 1001, "n_max": 30000, "split_at": [8000, 15000], "prime_limit": 300000,
               "prime_split_at": [50000, 100000], "pi_min_exponent": 3, "pi_max_exponent": 6, "pi_per_decade": 4,
               "pi_split_at": [10000, 100000], "gbm_rows": 2000, "sympy_order_sample": 50, "order_workers": 1,
               "cases": ["prime_digits_density", "prime_repetend", "prime_count"], "seed_offsets": [0],
               "population": 16, "generations": 3, "workers": 1}
        with tempfile.TemporaryDirectory() as tmp:
            out = digit_primes.run_stage(cfg, 0, Path(tmp), log=lambda *_: None)
        self.assertEqual(set(out["targets"]), set(cfg["cases"]))
        for row in out["targets"].values():
            self.assertTrue(row["gfsl"][-1].startswith("RESULT d$"))
            self.assertIn("held_out", row)
        f = out["targets"]["prime_repetend"]["controls"]["both"]
        self.assertGreater(f["auc"], 0.99)  # Fermat in bases 2 and 10 misses only pseudoprimes


if __name__ == "__main__":
    unittest.main()
