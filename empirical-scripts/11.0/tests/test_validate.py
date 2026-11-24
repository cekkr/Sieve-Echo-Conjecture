"""Tests for step 16: external validation of the final decoder (lengths from factors, verification, stage)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sympy

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import runtime  # noqa: E402
import sieve_echo_decoder as sed  # noqa: E402
import validate  # noqa: E402


class ValidateTests(unittest.TestCase):
    def test_lengths_from_factors_are_orders(self):
        for f in ({7: 1}, {23: 1, 89: 1}, {37: 2, 101: 1}, {2: 3, 3: 1, 1009: 1}, {961748941: 1, 37: 1}):
            n = math.prod(p**e for p, e in f.items())
            for b, L in validate.lengths_from_factors(f).items():
                self.assertEqual(L, sympy.n_order(b, n), (f, b))
                self.assertTrue(validate.verify_length(n, b, L))
                if L > 1:
                    self.assertFalse(validate.verify_length(n, b, 2 * L))

    def test_primes_of_large_lengths(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            m = int(rng.integers(2, 2**62)) * int(rng.integers(1, 2**40))
            self.assertEqual(sed.primes_of(m), sorted(sympy.primefactors(m)))

    def test_stage_on_a_small_list(self):
        with tempfile.TemporaryDirectory(dir=runtime.REPO_ROOT / "empirical-scripts" / "11.0" / "results") as d:
            primes = list(sympy.primerange(10**6, 10**6 + 40000))
            path = Path(d) / "list.txt"
            path.write_text("A small list (test)\n" + " ".join(map(str, primes)) + "\n")
            cfg = {"prime_list": runtime.repo_relative(path), "expected_count": len(primes), "sha256_txt": None, "gap_checks": 20, "workers": 2, "chunk": 20,
                   "families": {"prime": 20, "two_primes": 60, "three_primes": 20, "square_times_prime": 10, "list_times_medium": 20, "list_times_small": 20}}
            out = validate.run_stage(cfg, 0, log=lambda *_: None)
        self.assertEqual(out["integers"], 150)
        self.assertEqual(out["exact"], 150)
        for k, x in out["families"].items():
            self.assertEqual(x["lengths_verified"], x["integers"])
            if k != "list_times_small":
                self.assertEqual(x["all_19_bases_coprime"], x["integers"])
        import report

        self.assertIn("## Step 16", "\n".join(report.validate_markdown(out)))


if __name__ == "__main__":
    unittest.main()
