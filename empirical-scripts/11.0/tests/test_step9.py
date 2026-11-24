"""Tests for step 9: small-prime membership (euler.py) and order-only asymptotics (orders_scale.py)."""

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
import euler  # noqa: E402
import lengths  # noqa: E402
import ndr  # noqa: E402
import orders_scale  # noqa: E402


class OrdersScaleUnitTests(unittest.TestCase):
    def test_omega_and_factorization(self):
        import sympy

        primes = orders_scale._primes_upto(400)  # exact for n < 400^2
        rng = np.random.default_rng(0)
        nv = rng.integers(2, 150_000, size=3000)
        w = orders_scale.omega_vectorized(nv, primes)
        for n, k in zip(nv.tolist()[:600], w.tolist()[:600]):
            f = sympy.factorint(n)
            self.assertEqual(k, len(f), n)
            self.assertEqual(dict(orders_scale.factorize_td(n, primes.tolist())), {int(p): int(e) for p, e in f.items()})

    def test_membership_is_subgroup_membership_for_cyclic_groups(self):
        spf = ndr.spf_sieve(3000)
        bases = list(range(2, 20))
        ns = np.array([n for n in range(3, 2500) if len(ndr.factorize(n, spf)) == 1 and n % 2 == 1])
        L, _ = lengths.orders_and_midy(ns, bases, workers=1)
        mem = euler.membership(L, bases, [2, 3, 5, 7])
        for i, n in enumerate(ns.tolist()[::7]):
            i = ns.tolist().index(n)
            for j, b in enumerate(bases):
                if math.gcd(b, n) != 1:
                    continue
                H = {pow(b, k, n) for k in range(int(L[i, j]))}
                for q in (2, 3, 5, 7):
                    if math.gcd(q, n) != 1:
                        self.assertTrue(np.isnan(mem[q][i, j]))
                    else:
                        self.assertEqual(bool(mem[q][i, j]), q % n in H, (n, b, q))

    def test_run_stage_small(self):
        warnings.simplefilter("ignore")
        cfg = {"decades": [[1000, 9999], [10000, 99999]], "per_class": 120, "batch": 20000, "max_batches": 5, "bases": "2..40",
               "base_sets": {"L19": [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31], "L2..40": "2..40"},
               "model_seeds": 1, "bootstrap": 30, "gbm": {"max_iter": 30, "learning_rate": 0.2}, "workers": 1}
        out = orders_scale.run_stage(cfg, 3, log=lambda *_: None)
        self.assertEqual(len(out["decades"]), 2)
        d = out["decades"][-1]
        self.assertEqual(set(d["accuracy"]), {"L19", "L2..40"})
        self.assertGreater(d["accuracy"]["L2..40"], d["majority"])
        self.assertEqual(d["lcm19_equals_lambda"], 1.0)
        import report

        self.assertIn("## Step 9b", "\n".join(report.orders_markdown(out)))


CFG = {
    "n_min": 2,
    "n_max": 999,
    "stratified": [{"n_lo": 1000, "n_hi": 9999, "per_omega": 80}, {"n_lo": 10000, "n_hi": 60000, "per_omega": 80}],
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13]},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}


class EulerStageTests(unittest.TestCase):
    def test_run_stage(self):
        warnings.simplefilter("ignore")
        table, _ = dataset.build_dataset(CFG, SPLIT, seed=5, log=lambda *_: None)
        ccfg = {"entropies": ["H_paper", "H_nodc"], "moduli": [6, 12], "permutations": 20, "bootstrap": 20, "primary_set": "v8"}
        with tempfile.TemporaryDirectory() as d:
            out = euler.run_stage(table, {"length_bases": "2..40", "bootstrap": 30, "workers": 1}, ccfg, CFG["base_sets"], 5, Path(d), "testkey", log=lambda *_: None)
        v = out["pair_level"]["variants"]["H_nodc"]
        self.assertGreaterEqual(v["r2_plus_membership"], v["r2_structure"] - 0.01)
        self.assertEqual(set(v["coefficients"]), {2, 3, 5, 7, 11, 13})
        self.assertIn("plus_membership", out["residues"]["variants"]["H_paper"])
        import report

        self.assertIn("## Step 9a", "\n".join(report.euler_markdown(out)))


if __name__ == "__main__":
    unittest.main()
