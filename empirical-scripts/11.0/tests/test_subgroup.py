"""Tests for step 14: exact subgroup membership q ∈ ⟨b⟩ (also for non-cyclic (ℤ/nℤ)*) and the stage."""

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

import dataset  # noqa: E402
import lengths  # noqa: E402
import ndr  # noqa: E402
import subgroup  # noqa: E402

SPF = ndr.spf_sieve(5000)


class MembershipTests(unittest.TestCase):
    def test_exact_membership_equals_remainder_set(self):
        differ = 0
        for n in range(3, 2500):
            f = ndr.factorize(n, SPF)
            lam, lp = ndr.order_context(n, SPF)
            for b in (2, 3, 5, 10, 12):
                if math.gcd(b, n) != 1:
                    continue
                t = ndr.multiplicative_order(b, n, lam, lp)
                R = set(ndr.repetend_remainders(n, b, t).tolist()) if n < ndr._MAX_N else set()
                for q in (2, 3, 5, 7, 11):
                    got = subgroup.exact_member(n, b, q, f, SPF)
                    if math.gcd(q, n) != 1:
                        self.assertIsNone(got)
                        continue
                    self.assertEqual(got, (q % n) in R, (n, b, q))
                    oq = ndr.multiplicative_order(q, n, lam, lp)
                    differ += int((t % oq == 0) and not got)
        self.assertGreater(differ, 100)  # the divisibility test is not exact for non-cyclic groups

    def test_negative_membership(self):
        ns = np.arange(3, 400)
        neg = subgroup.negative_membership(ns, [2, 10], [3, 7], 1)
        for i, n in enumerate(ns.tolist()):
            for j, b in enumerate((2, 10)):
                if math.gcd(b, n) != 1:
                    continue
                lam, lp = ndr.order_context(n, SPF)
                R = set(ndr.repetend_remainders(n, b, ndr.multiplicative_order(b, n, lam, lp)).tolist())
                for q in (3, 7):
                    if math.gcd(q, n) == 1:
                        self.assertEqual(bool(neg[q][i, j]), ((-q) % n) in R, (n, b, q))

    def test_cyclic_group(self):
        n = np.array([2, 4, 8, 9, 18, 12, 15, 27, 54, 100])
        np.testing.assert_array_equal(subgroup.cyclic_group(n, SPF), [1, 1, 0, 1, 1, 0, 0, 1, 1, 0])

    def test_exact_membership_matrix(self):
        ns = np.arange(3, 600)
        bases = [2, 3, 5, 7, 10]
        L, _ = lengths.orders_and_midy(ns, bases, workers=1)
        exact, div, cyc, chk = subgroup.exact_membership(ns, L, bases, bases, [2, 3, 5, 7], 1)
        for i, n in enumerate(ns.tolist()):
            f = ndr.factorize(n, SPF)
            for j, b in enumerate(bases):
                for q in (2, 3, 5, 7):
                    e = subgroup.exact_member(n, b, q, f, SPF)
                    if e is None:
                        self.assertTrue(np.isnan(exact[q][i, j]))
                    else:
                        self.assertEqual(bool(exact[q][i, j]), e, (n, b, q))
        self.assertGreater(chk["divisibility_true_but_not_member"], 0)


CFG = {
    "n_min": 2, "n_max": 1999,
    "stratified": [{"n_lo": 2000, "n_hi": 9999, "per_omega": 80}, {"n_lo": 10000, "n_hi": 40000, "per_omega": 80}],
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13]},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}


class SubgroupStageTests(unittest.TestCase):
    def test_run_stage(self):
        warnings.simplefilter("ignore")
        table, _ = dataset.build_dataset(CFG, SPLIT, seed=5, log=lambda *_: None)
        ccfg = {"entropies": ["H_paper", "H_nodc"], "moduli": [6, 12], "permutations": 20, "bootstrap": 20, "primary_set": "v8"}
        with tempfile.TemporaryDirectory() as d:
            out = subgroup.run_stage(table, {"length_bases": "2..99", "workers": 1}, ccfg, CFG["base_sets"], 5, Path(d), "testkey", log=lambda *_: None)
        v = out["residues"]["H_paper"]
        self.assertEqual(set(v["columns"]), {"c7_midy_zeros", "divisibility_small", "exact_small", "divisibility_large", "exact_large", "exact_signed_small", "exact_signed_small_divisibility_large", "diag_coprime_pattern", "diag_coprime_pattern_v2", "diag_pattern_v2_without_membership"})
        self.assertGreater(out["membership"]["candidates_checked"], 0)
        import report

        self.assertIn("## Step 14", "\n".join(report.subgroup_markdown(out)))


if __name__ == "__main__":
    unittest.main()
