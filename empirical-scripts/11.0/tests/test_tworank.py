"""Tests for step 10: the 2-rank of (Z/nZ)* and omega (exact on all units), estimators, the stage."""

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

import ndr  # noqa: E402
import tworank  # noqa: E402

SPF = ndr.spf_sieve(4000)


class TwoRankTheoremTests(unittest.TestCase):
    def test_profile_ratio_and_omega_formula_on_all_units(self):
        for n in range(3, 3000):
            f = ndr.factorize(n, SPF)
            e = tworank.sylow2_exponents(f)
            lam, lp = ndr.order_context(n, SPF)
            units = [x for x in range(1, n) if math.gcd(x, n) == 1]
            v = np.array([(t & -t).bit_length() - 1 for t in (ndr.multiplicative_order(x, n, lam, lp) for x in units)])
            phi = len(units)
            for k in range(0, 6):
                want = phi * 2.0 ** (-sum(max(ei - k, 0) for ei in e))
                self.assertEqual(int((v <= k).sum()), int(round(want)), (n, k, e))
            r = len(e)
            self.assertEqual((v <= 1).sum(), (v == 0).sum() * 2 ** r)
            self.assertEqual(int(tworank.omega_from_rank(np.array([r]), np.array([n]))[0]), len(f), n)
            # the ratio estimator is exact when the "bases" are all units
            V2 = v[None, :]
            self.assertEqual(int(tworank.ratio_estimate(V2, np.array([n]))["omega"][0]), len(f), n)
            self.assertEqual(int(tworank.mle_rank(V2)[0]), r, n)  # exact profile -> exact MLE

    def test_mle_monotone_and_ratio_consistent(self):
        rng = np.random.default_rng(1)
        V2 = rng.integers(-1, 6, size=(200, 60))
        r = tworank.mle_rank(V2)
        self.assertTrue(((r >= 0) & (r <= 12)).all())
        # one odd-order base and many v2 = 1 bases: the MLE and the ratio both see a large rank
        V = np.array([[0] + [1] * 31 + [2] * 10])
        self.assertEqual(int(tworank.ratio_estimate(V, np.array([3]))["rank"][0]), 5)
        self.assertGreaterEqual(int(tworank.mle_rank(V)[0]), 5)

    def test_run_stage_models_and_comparisons(self):
        warnings.simplefilter("ignore")
        ocfg = {"decades": [[10000, 99999]], "per_class": 60, "batch": 20000, "max_batches": 3}
        tcfg = {"bases": "2..40", "estimator_sets": {"2..40": "2..40"}, "model_seeds": 1, "bootstrap": 20, "gbm": {"max_iter": 20, "learning_rate": 0.2}, "workers": 1,
                "models": [{"lengths": "2..20"}, {"lengths": "2..20", "profiles": "2..20"}, {"lengths": "2..40"}, {"lengths": "2..40", "profiles": "2..40"}],
                "comparisons": [["lengths 2..20 + ℓ-adic 2..20", "lengths 2..20"], ["lengths 2..40 + ℓ-adic 2..40", "lengths 2..40"]]}
        out = tworank.run_stage(ocfg, tcfg, 4, log=lambda *_: None)
        self.assertEqual(set(out["decades"][0]["comparisons"]), {"lengths 2..20 + ℓ-adic 2..20 vs lengths 2..20", "lengths 2..40 + ℓ-adic 2..40 vs lengths 2..40"})

    def test_valuation_and_histogram(self):
        L = np.array([[12, 0, 9, 1], [8, 3, 0, 0]])
        np.testing.assert_array_equal(tworank.valuation(L, 2), [[2, -1, 0, 0], [3, 0, -1, -1]])
        np.testing.assert_array_equal(tworank.valuation(L, 3), [[1, -1, 2, 0], [0, 1, -1, -1]])
        H = tworank.histogram(tworank.valuation(L, 2), 4)
        np.testing.assert_allclose(H[:, :-1].sum(axis=1), 1.0)
        np.testing.assert_array_equal(H[:, -1], [3, 2])

    def test_run_stage_small(self):
        warnings.simplefilter("ignore")
        ocfg = {"decades": [[1000, 9999], [10000, 99999]], "per_class": 100, "batch": 20000, "max_batches": 5}
        tcfg = {"bases": "2..60", "estimator_sets": {"2..30": "2..30", "2..60": "2..60", "primes": "primes"}, "length_set": "2..30",
                "profile_sets": ["2..30", "2..60"], "model_seeds": 1, "bootstrap": 30, "gbm": {"max_iter": 30, "learning_rate": 0.2}, "workers": 1}
        out = tworank.run_stage(ocfg, tcfg, 3, log=lambda *_: None)
        self.assertEqual(len(out["decades"]), 2)
        self.assertEqual(set(out["estimators"]), {"2..30", "2..60", "primes"})
        acc = out["decades"][-1]["accuracy"]
        self.assertIn("ℓ-adic 2..60", acc)
        self.assertIn("accuracy_mle", out["estimators"]["2..60"]["decades"][0])
        self.assertGreater(min(acc.values()), 0.3)
        import report

        md = "\n".join(report.tworank_markdown(out))
        self.assertIn("## Step 10", md)
        import figures
        import runtime

        with tempfile.TemporaryDirectory() as d:
            runtime.write_json(Path(d) / "tworank.json", out)
            self.assertIn("tworank_accuracy.png", {Path(f).name for f, _ in figures.make_figures(Path(d), None)})


if __name__ == "__main__":
    unittest.main()
