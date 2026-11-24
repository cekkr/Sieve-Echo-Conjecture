"""Unit tests for claims.py: statistics helpers and an end-to-end run of C1-C9 on a small dataset."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import claims  # noqa: E402
import dataset  # noqa: E402
import report  # noqa: E402

SMALL = {
    "n_min": 2,
    "n_max": 1500,
    "stratified": {"enabled": True, "n_lo": 1501, "n_hi": 40000, "per_omega": 40},
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13], "wide": {"bases": "2..30", "sample": 200}},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}
CLAIMS_CFG = {"bootstrap": 30, "permutations": 30, "classifier": {"max_iter": 40, "shuffled_nulls": 1}}


class HelperTests(unittest.TestCase):
    def test_partial_corr_removes_confounder(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=4000)
        x = z + 0.1 * rng.normal(size=4000)
        y = 2 * z + 0.1 * rng.normal(size=4000)  # x and y related only through z
        self.assertGreater(claims.corr(x, y), 0.9)
        self.assertLess(abs(claims.partial_corr(x, y, z[:, None])), 0.06)
        y2 = y + 0.5 * x  # direct effect of x
        # residuals: x -> 0.1 e1, y2 -> 0.1 e2 + 0.05 e1, so the partial r is 0.05 / sqrt(0.0125) = 1/sqrt(5)
        self.assertAlmostEqual(claims.partial_corr(x, y2, z[:, None]), 1 / math.sqrt(5), delta=0.03)

    def test_r2_and_lstsq(self):
        x = np.linspace(0, 1, 50)
        y = 3 * x + 1
        coef = claims.lstsq(claims.design(x), y)
        np.testing.assert_allclose(coef, [1, 3], atol=1e-12)
        self.assertAlmostEqual(claims.r2(y, claims.design(x) @ coef), 1.0)

    def test_eta_squared(self):
        g = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(claims.eta_squared(np.array([1.0, 1.0, 3.0, 3.0]), g), 1.0)
        self.assertAlmostEqual(claims.eta_squared(np.array([1.0, 3.0, 1.0, 3.0]), g), 0.0)

    def test_bootstrap_and_permutation(self):
        rng = np.random.default_rng(1)
        v = rng.normal(5.0, 1.0, size=500)
        lo, hi = claims.bootstrap(lambda i: v[i].mean(), v.size, 300, np.random.default_rng(2))
        self.assertTrue(lo < 5.0 < hi)
        a, b = rng.normal(size=300), rng.normal(size=300)
        obs = claims.corr(a, b)
        p = claims.perm_p(lambda q: claims.corr(a[q], b), obs, a.size, 200, np.random.default_rng(3))
        self.assertGreater(p, 0.01)
        c = a + 0.5 * rng.normal(size=300)
        p2 = claims.perm_p(lambda q: claims.corr(a[q], c), claims.corr(a, c), a.size, 200, np.random.default_rng(4))
        self.assertLess(p2, 0.01)

    def test_constant_library_reproduces_v8_match(self):
        lib = claims._v8_constants()
        self.assertAlmostEqual(lib["tribonacci"] - lib["ln_10"], -0.46329833777988494, places=12)
        self.assertIn(round(lib["tribonacci"] - lib["ln_10"], 12), set(np.round(claims._candidate_values(lib), 12)))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, _ = dataset.build_dataset(SMALL, SPLIT, seed=5, log=lambda *_: None)
        cls.out = claims.run_claims(cls.table, CLAIMS_CFG, SMALL["base_sets"], 5, V11.parents[1], log=lambda *_: None)

    def test_all_claims_present_with_verdicts(self):
        ids = [c["id"] for c in self.out["claims"]]
        self.assertEqual(ids, [f"C{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9)])
        for c in self.out["claims"]:
            self.assertTrue(c["verdict"])
            self.assertTrue(c["reason"])
            self.assertIn("status", c["reproduction"])

    def test_length_arithmetic_for_primes(self):
        ctx = claims.Ctx(self.table, {**CLAIMS_CFG, "_base_sets": {k: v["bases"] for k, v in dataset.normalize_base_sets(SMALL["base_sets"]).items()}}, 5, V11.parents[1])
        feats, names = claims._length_arithmetic(ctx, [2, 3, 5, 7, 10, 11])
        divides = feats[:, names.index("lcm divides n-1")]
        primes = self.table["label.is_prime"].astype(bool)
        self.assertTrue((divides[primes & (self.table["row.n"] > 11)] == 1).all())  # ord_p(b) | p - 1

    def test_c9_never_reads_labels(self):
        cols = dataset.allowed_input_columns(self.table)
        self.assertFalse(any(c.startswith(("label.", "target_side.")) for c in cols))

    def test_report_table(self):
        md = report.verdict_table(self.out["claims"])
        self.assertEqual(md.count("\n"), len(self.out["claims"]) + 1)
        self.assertIn("**C9**", md)


if __name__ == "__main__":
    unittest.main()
