"""Tests for step 7: order matrix, Midy flags (exact properties), length arithmetic parity, the lengths stage."""

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
import features  # noqa: E402
import lengths  # noqa: E402
import ndr  # noqa: E402
import scale  # noqa: E402
import search_formulas as sf  # noqa: E402

SPF = ndr.spf_sieve(5000)


def _order(b, m):
    t, x = 1, b % m
    while x != 1:
        x = x * b % m
        t += 1
    return t


def _v2(x):
    return (x & -x).bit_length() - 1


class MidyTests(unittest.TestCase):
    def test_midy_iff_complementary_halves_and_even_bins_vanish(self):
        seen = [0, 0]
        for n in range(3, 400):
            lam, lp = ndr.order_context(n, SPF)
            for b in range(2, 13):
                if math.gcd(b, n) != 1:
                    continue
                L = ndr.multiplicative_order(b, n, lam, lp)
                if L < 2:
                    continue
                d = ndr.repetend_digits(n, b, L)
                comp = L % 2 == 0 and bool(np.all(d[L // 2:] == b - 1 - d[: L // 2]))
                flag = lengths.midy_flag(n, b, L)
                self.assertEqual(flag, comp, (n, b, L))
                seen[flag] += 1
                if flag and L > 2:
                    P =np.abs(np.fft.fft(ndr.ndr(d, b))) ** 2
                    self.assertLess(P[2::2].max(), 1e-9 * max(P.sum(), 1.0), (n, b))
        self.assertTrue(seen[0] > 100 and seen[1] > 100)

    def test_odd_n_two_adic_characterization(self):
        for n in range(3, 3000, 2):
            f = ndr.factorize(n, SPF)
            lam, lp = ndr.order_context(n, SPF)
            for b in (2, 3, 5, 7, 10):
                if math.gcd(b, n) != 1:
                    continue
                L = ndr.multiplicative_order(b, n, lam, lp)
                v = {_v2(_order(b, p ** e)) for p, e in f}
                self.assertEqual(lengths.midy_flag(n, b, L), len(v) == 1 and min(v) >= 1, (n, b))

    def test_orders_matrix(self):
        ns = np.array([2, 3, 12, 97, 341, 561, 1001, 4999])
        bases = list(range(2, 20))
        L1, M1 = lengths.orders_and_midy(ns, bases, workers=1)
        for i, n in enumerate(ns.tolist()):
            for j, b in enumerate(bases):
                if math.gcd(n, b) != 1:
                    self.assertEqual(L1[i, j], 0)
                    self.assertEqual(M1[i, j], 0)
                else:
                    self.assertEqual(L1[i, j], _order(b, n))
        # lcm of the orders over all coprime residues is lambda(n)
        for n in (561, 1001, 4999):
            lam = ndr.carmichael_from_factors(ndr.factorize(n, SPF))
            Lall, _ = lengths.orders_and_midy(np.array([n]), list(range(2, n)), workers=1)
            self.assertEqual(int(np.lcm.reduce(np.where(Lall > 0, Lall, 1), axis=1)[0]), lam)


CFG = {
    "n_min": 2,
    "n_max": 999,
    "stratified": [{"n_lo": 1000, "n_hi": 9999, "per_omega": 80}, {"n_lo": 10000, "n_hi": 60000, "per_omega": 80}],
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13]},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}


class LengthsStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.table, _ = dataset.build_dataset(CFG, SPLIT, seed=5, log=lambda *_: None)

    def test_arithmetic_parity(self):
        bases = features.union_bases(CFG["base_sets"], ["v8", "v10", "primes31"])
        L, _ = lengths.orders_and_midy(self.table["row.n"], bases, workers=1)
        a1, _ = features.length_arithmetic(self.table, bases)
        a2, _ = features.length_arithmetic_from_orders(self.table["row.n"], L)
        np.testing.assert_allclose(a1, a2, equal_nan=True)

    def test_run_stage(self):
        bins = [[100, 999], [1000, 9999], [10000, 60000]]
        lcfg = {"bases": "2..40", "bins": bins, "per_omega_cap": 80, "min_per_omega": 10, "bootstrap": 30, "pair_bins": 8, "workers": 1,
                "gbm": {"max_iter": 30, "learning_rate": 0.2},
                "formulas": {"guides": ["none"], "seeds": [0], "population": 16, "generations": 2, "workers": 1, "eval_size_min": 200, "eval_size_step": 200}}
        ccfg = {"entropies": ["H_paper", "H_nodc"], "moduli": [6, 12], "permutations": 20, "bootstrap": 20}
        with tempfile.TemporaryDirectory() as d:
            out = lengths.run_stage(self.table, lcfg, ccfg, CFG["base_sets"], sf.DEFAULT_INPUTS, 5, Path(d), "testkey", log=lambda *_: None)
            self.assertTrue(any(Path(d, "cache").glob("lengths-*.npz")))
            again = lengths.run_stage(self.table, {**lcfg, "formulas": None, "ablation": False}, ccfg, CFG["base_sets"], sf.DEFAULT_INPUTS, 5, Path(d), "testkey", log=lambda *_: None)
            import figures
            import runtime

            runtime.write_json(Path(d) / "lengths.json", out)
            made = {Path(f).name for f, _ in figures.make_figures(Path(d), None)}
            self.assertIn("midy_entropy_gap.png", made)
            self.assertIn("c9_gain_by_baseline.png", made)
            import report

            md = "\n".join(report.lengths_markdown(runtime.read_json(Path(d) / "lengths.json")))
            self.assertIn("## Step 7", md)
            self.assertIn("Midy coefficient", md)
            self.assertIn("lengths 2..99 + Midy share", md)
        self.assertEqual(again["c9_bins"][-1]["accuracy"], out["c9_bins"][-1]["accuracy"])  # cached orders, same result
        self.assertTrue(out["pair_mechanism"]["orders_match_pair_table"])
        gap = out["pair_mechanism"]["variants"]["H_nodc"]["midy_coefficient"]
        self.assertAlmostEqual(gap, -math.log(2), delta=0.15)
        self.assertEqual(out["lambda"]["theorem_1_3_odd_n"]["violations of 2^(ω−1) ≤ φ/λ"], 0)
        self.assertEqual(out["lambda"]["length_only_bound_odd_n"]["violations"], 0)
        self.assertTrue(out["lambda"]["coverage_all"]["lcm98 divides λ"])
        self.assertIn("c7_controls_plus_midy", out["residues"]["variants"]["H_paper"])
        self.assertEqual(set(out["formulas"]["input_sets"]), {"step-4 inputs (19 bases + entropy)", "lengths 2..99 + Midy share"})
        # L19 reproduces the scale stage's C9 baseline on the same omega-balanced samples
        sc = scale.run_stage(self.table, {"bins": bins, "per_omega_cap": 80, "min_per_omega": 10, "bootstrap": 10, "gbm": lcfg["gbm"], "T_x": None},
                             CFG["base_sets"], sf.DEFAULT_INPUTS, 5, log=lambda *_: None)
        for b_sc, b_l in zip(sc["bins"], out["c9_bins"]):
            if "skipped" in b_sc:
                continue
            self.assertAlmostEqual(b_sc["C9"]["accuracy_baseline"], b_l["accuracy"]["L19"])
            self.assertAlmostEqual(b_sc["C9"]["accuracy_entropy"], b_l["accuracy"]["L19 + entropy"])
            self.assertAlmostEqual(b_sc["C9"]["accuracy_all"], b_l["accuracy"]["L19 + all NDR"])


if __name__ == "__main__":
    unittest.main()
