"""Tests for step 13: exact decoding of the factorization from repetend lengths and digits (Proposition 1.7)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import decode  # noqa: E402
import lengths  # noqa: E402
import ndr  # noqa: E402

SPF = ndr.spf_sieve(60000)
B19 = decode.BASES19


def ctx(n):
    f = ndr.factorize(n, SPF)
    lam, lp = ndr.order_context(n, SPF)
    L = [0 if math.gcd(b, n) != 1 else ndr.multiplicative_order(b, n, lam, lp) for b in B19]
    return f, lam, lp, L


class DecoderTheoryTests(unittest.TestCase):
    def test_coprime_base(self):
        self.assertEqual(sorted(decode.coprime_base([12, 18])), [2, 3])
        self.assertEqual(sorted(decode.coprime_base([4, 2, 9, 1])), [2, 9])
        self.assertEqual(decode.coprime_base([35]), [35])

    def test_prime_divisors_of_L_from_lambda(self):
        import sympy

        for n in range(3, 4000, 7):
            _, lam, lp, L = ctx(n)
            for t in L:
                if t > 1:
                    self.assertEqual(decode.prime_divisors(t, lp), sorted(sympy.primefactors(t)), (n, t))

    def test_proposition_17a_parts_are_order_vector_classes(self):
        for n in range(2, 30000):
            f, lam, lp, L = ctx(n)
            qs = lambda t: decode.prime_divisors(t, lp)  # noqa: E731
            for k in (1, 2, 5, 19):
                got = decode.decode(n, B19[:k], L[:k], qs)
                classes = decode.order_vector_classes(n, B19[:k], f)
                if all(e == 1 for _, e in f):  # squarefree: exactly the order-vector classes
                    self.assertEqual(got, classes, (n, k))
                else:  # prime powers can only separate more (ord_{p^a} carries a factor p)
                    self.assertGreaterEqual(got, classes, (n, k))
                self.assertLessEqual(got, len(f))
            marks = decode.decode(n, B19, L, qs, checkpoints=True)
            self.assertEqual(marks[-1], decode.decode(n, B19, L, qs))
            self.assertEqual(marks, sorted(marks))  # more bases never merge parts

    def test_all_n_below_30000_decoded_by_19_bases(self):
        for n in range(2, 30000):
            f, lam, lp, L = ctx(n)
            self.assertEqual(decode.decode(n, B19, L, lambda t: decode.prime_divisors(t, lp)), len(f), n)

    def test_proposition_17b_shor_split_fails_exactly_on_midy_pairs(self):
        seen = 0
        for n in range(3, 6000, 2):
            _, _, _, L = ctx(n)
            for b, t in zip(B19, L):
                if t > 0 and t % 2 == 0:
                    fail = math.gcd(pow(b, t // 2, n) - 1, n) == 1
                    self.assertEqual(fail, bool(lengths.midy_flag(n, b, t)), (n, b))
                    seen += fail
        self.assertGreater(seen, 1000)

    def test_proposition_17c_remainders_from_digits(self):
        rng = np.random.default_rng(3)
        for n in rng.integers(3, 50000, size=300).tolist():
            _, lam, lp, L = ctx(n)
            for b, t in zip(B19, L):
                if t < 2:
                    continue
                dg = ndr.repetend_digits(n, b, t)
                for d in sorted(set(rng.integers(0, t, size=5).tolist()) | {t // 2 if t % 2 == 0 else 0}):
                    self.assertEqual(decode.remainder_from_digits(dg, b, n, d), pow(b, d, n), (n, b, d))
            f = ndr.factorize(n, SPF)
            dig = {b: ndr.repetend_digits(n, b, t) for b, t in zip(B19, L) if t > 1}
            via = decode.decode(n, B19, L, lambda t: decode.prime_divisors(t, lp), remainder=lambda b, d: decode.remainder_from_digits(dig[b], b, n, d))
            self.assertEqual(via, len(f), n)

    def test_long_division_agrees(self):
        for n, b in ((7, 10), (21, 2), (91, 3), (1001, 10)):
            dg = ndr.long_division_repetend(n, b)
            for d in range(len(dg)):
                self.assertEqual(decode.remainder_from_digits(dg, b, n, d), pow(b, d, n))


class DecodeStageTests(unittest.TestCase):
    def test_exhaustive_and_stage(self):
        warnings.simplefilter("ignore")
        primes = decode.orders_scale._primes_upto(400)
        ex = decode.exhaustive(5000, 1000, 20, primes, B19, [2, 3, 5, 7], 1)
        self.assertEqual(ex["integers"], 4999)
        self.assertEqual(ex["failures_all_bases"], 0)
        self.assertLess(ex["accuracy_by_prefix"][0], 0.9)
        cfg = {"decades": [[1000, 9999], [10000, 99999]], "per_class": 40, "batch": 20000, "max_batches": 5, "uniform_per_decade": 200,
               "exhaustive_max": 3000, "exhaustive_chunk": 1000, "workers": 1,
               "digit_check": {"decades": [[1000, 99999]], "pairs": 50, "max_L": 20000, "decode_integers": 20}}
        out = decode.run_stage(cfg, 2, log=lambda *_: None)
        self.assertEqual(len(out["balanced"]), 2)
        self.assertEqual(out["digit_check"]["mismatches"], 0)
        self.assertEqual(out["digit_check"]["integers_correct"], out["digit_check"]["integers_decoded_from_digits"])
        m = out["balanced"][-1]["midy_equals_shor_failure"]
        self.assertEqual(m["agree"], m["pairs"])
        self.assertTrue(all(d["underestimates_only"] for d in out["balanced"]))
        import report

        self.assertIn("## Step 13", "\n".join(report.decode_markdown(out)))
        import figures
        import runtime

        with tempfile.TemporaryDirectory() as d:
            runtime.write_json(Path(d) / "decode.json", out)
            self.assertIn("decode_bases.png", {Path(f).name for f, _ in figures.make_figures(Path(d), None)})


if __name__ == "__main__":
    unittest.main()
