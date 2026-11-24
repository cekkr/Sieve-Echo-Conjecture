"""Unit tests for dataset.py: sampling, aggregation, split, leakage guard, cache round trip."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import dataset  # noqa: E402
import ndr  # noqa: E402

SMALL = {
    "n_min": 2,
    "n_max": 400,
    "stratified": {"enabled": True, "n_lo": 401, "n_hi": 3000, "per_omega": 5},
    "base_sets": {"a": [2, 3, 10], "b": [5, 7], "w": {"bases": "2..20", "sample": 30}},
    "workers": 1,
}
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15, "stratify": "omega"}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.summary = dataset.build_dataset(SMALL, SPLIT, seed=3, log=lambda *_: None)

    def test_rows_and_sampling(self):
        n = self.table["row.n"]
        self.assertTrue((np.diff(n) > 0).all())
        self.assertTrue(set(range(2, 401)) <= set(n.tolist()))
        strat = self.summary["sampling"]["stratified"]["by_omega"]
        for w, c in strat.items():
            self.assertEqual(c["taken"], min(5, c["available"]))
        self.assertEqual(int((self.table["row.source"] == 1).sum()), sum(c["taken"] for c in strat.values()))

    def test_aggregate_matches_direct(self):
        spf = ndr.spf_sieve(3000)
        for n in (77, 97, 360, 391):
            i = int(np.searchsorted(self.table["row.n"], n))
            vals = []
            for b in SMALL["base_sets"]["a"]:
                if math.gcd(n, b) == 1:
                    lam, lp = ndr.order_context(n, spf)
                    L = ndr.multiplicative_order(b, n, lam, lp)
                    vals.append(dict(zip(ndr.PAIR_FEATURES, ndr.pair_features(n, b, L)))["H_paper"])
            self.assertEqual(int(self.table["row.n_bases.a"][i]), len(vals))
            if vals:
                self.assertAlmostEqual(float(self.table["row.a.H_paper.mean"][i]), float(np.mean(vals)), places=12)
                self.assertAlmostEqual(float(self.table["row.a.H_paper.std"][i]), float(np.std(vals)), places=12)
                self.assertAlmostEqual(float(self.table["row.a.H_paper.max"][i]), float(np.max(vals)), places=12)
            else:
                self.assertTrue(math.isnan(self.table["row.a.H_paper.mean"][i]))

    def test_subsampled_set(self):
        used = self.table["row.n_bases.w"]
        self.assertEqual(self.summary["base_sets"]["w"]["members"], 30)
        self.assertLessEqual(int((used > 0).sum()), 30)
        self.assertGreater(int(np.isnan(self.table["row.w.H_paper.mean"]).sum()), 0)

    def test_split_disjoint_stratified_deterministic(self):
        split = self.table["split"]
        self.assertEqual(set(np.unique(split).tolist()), {0, 1, 2})
        rec = dataset.split_record(self.table)
        self.assertEqual(sum(len(v) for v in rec.values()), self.table["row.n"].size)
        self.assertFalse(set(rec["train"]) & set(rec["test"]))
        again = dataset.split_by_n(self.table["row.n"], self.table["label.omega"], SPLIT, seed=3)
        self.assertTrue((again == split).all())
        other = dataset.split_by_n(self.table["row.n"], self.table["label.omega"], SPLIT, seed=4)
        self.assertFalse((other == split).all())
        omega = self.table["label.omega"]
        for w in np.unique(omega):
            sel = omega == w
            if sel.sum() >= 20:
                self.assertAlmostEqual(float((split[sel] == 0).mean()), 0.7, delta=0.06)

    def test_allowed_inputs_exclude_factorization(self):
        cols = dataset.allowed_input_columns(self.table)
        self.assertIn("row.n", cols)
        self.assertIn("row.a.H_paper.mean", cols)
        for c in cols:
            self.assertFalse(c.startswith(("label.", "target_side.", "pair.")), c)
            self.assertNotEqual(c, "split")

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.npz"
            dataset.save_cache(p, self.table, "k1")
            back = dataset.load_cache(p, "k1")
            self.assertIsNone(dataset.load_cache(p, "other"))
            self.assertEqual(set(back), set(self.table))
            for k, v in self.table.items():
                np.testing.assert_array_equal(back[k], v)


if __name__ == "__main__":
    unittest.main()
