"""Tests for step 5: multi-range stratified sampling, ω-balanced bins, scale statistics, T(x)."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import dataset  # noqa: E402
import ndr  # noqa: E402
import scale  # noqa: E402
import search_formulas as sf  # noqa: E402

CFG = {
    "n_min": 2,
    "n_max": 999,
    "stratified": [
        {"n_lo": 1000, "n_hi": 9999, "per_omega": 60},
        {"n_lo": 10000, "n_hi": 60000, "per_omega": 60},
    ],
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13], "wide": {"bases": "2..20", "sample": 150}},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}


class MultiRangeSamplingTests(unittest.TestCase):
    def test_ranges_and_backward_compatibility(self):
        spf = ndr.spf_sieve(60000)
        n, src, scheme = dataset.sample_integers(CFG, 3, spf)
        self.assertEqual(len(scheme["stratified"]), 2)
        for r in scheme["stratified"]:
            for w, c in r["by_omega"].items():
                self.assertEqual(c["taken"], min(60, c["available"]))
        self.assertEqual(n.size, np.unique(n).size)
        self.assertTrue(((n >= 10000) & (n <= 60000)).sum() > 0)
        single = {**CFG, "stratified": {"enabled": True, "n_lo": 1000, "n_hi": 9999, "per_omega": 60}}
        n1, _, s1 = dataset.sample_integers(single, 3, spf)
        self.assertIsInstance(s1["stratified"], dict)
        listed = {**CFG, "stratified": [{"n_lo": 1000, "n_hi": 9999, "per_omega": 60}]}
        n2, _, _ = dataset.sample_integers(listed, 3, spf)
        np.testing.assert_array_equal(n1, n2)  # range 0 uses the same RNG stream as the dict form


class ScaleStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.table, _ = dataset.build_dataset(CFG, SPLIT, seed=3, log=lambda *_: None)

    def test_balanced(self):
        rows = np.arange(100)
        omega = np.array([1] * 70 + [2] * 25 + [3] * 5)
        out = scale._balanced(rows, omega, cap=20, min_count=10, rng=np.random.default_rng(0))
        counts = np.bincount(omega[out])
        self.assertEqual(counts[1], 20)
        self.assertEqual(counts[2], 20)
        self.assertEqual(counts[3] if counts.size > 3 else 0, 0)

    def test_run_stage(self):
        out = scale.run_stage(self.table, {"bins": [[100, 999], [1000, 9999], [10000, 60000]], "per_omega_cap": 60, "min_per_omega": 10,
                                           "bootstrap": 30, "gbm": {"max_iter": 20, "learning_rate": 0.2}, "T_x": [20, 40, 80]},
                              CFG["base_sets"], sf.DEFAULT_INPUTS, 3, log=lambda *_: None)
        self.assertEqual(len(out["bins"]), 3)
        self.assertEqual(set(out["stability"]), {"C1", "C2", "C3", "C4", "C9"})
        self.assertIn("balanced_all_bins", out["growth"])
        self.assertGreater(out["T_x"]["exponent"], 1.5)  # T(x) has ~x^2/log x terms
        import tempfile

        import figures
        import runtime

        with tempfile.TemporaryDirectory() as d:
            runtime.write_json(Path(d) / "scale.json", out)
            made = figures.make_figures(Path(d), self.table)
            names = {Path(f).name for f, _ in made}
            self.assertTrue({"entropy_vs_length.png", "c9_gain_by_decade.png", "invariance_by_decade.png", "growth_exponents.png"} <= names)
            for f, caption in made:
                self.assertTrue((Path(d) / f).stat().st_size > 1000)
                self.assertTrue(caption)


if __name__ == "__main__":
    unittest.main()
