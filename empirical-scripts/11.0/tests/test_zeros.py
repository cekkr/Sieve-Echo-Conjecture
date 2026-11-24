"""Tests for step 8: spectral zero classes — DFT identities, class-wise vanishing, characters, Euler factors, stage."""

from __future__ import annotations

import cmath
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
import search_formulas as sf  # noqa: E402
import zeros  # noqa: E402

SPF = ndr.spf_sieve(5000)


def _L(n, b):
    lam, lp = ndr.order_context(n, SPF)
    return ndr.multiplicative_order(b, n, lam, lp)


class ZeroClassTests(unittest.TestCase):
    def test_digit_dft_is_remainder_dft_times_factor(self):
        for n, b in ((7, 10), (91, 10), (221, 3), (1001, 2), (4999, 7)):
            L = _L(n, b)
            r = ndr.repetend_remainders(n, b, L).astype(float)
            d = ndr.repetend_digits(n, b, L).astype(float)
            j = np.arange(L)
            np.testing.assert_allclose(np.fft.fft(d), (b - np.exp(2j * np.pi * j / L)) * np.fft.fft(r) / n, atol=1e-6 * L)

    def test_exact_classes_match_fft_and_contain_midy(self):
        seen_extra = 0
        for n in range(3, 700):
            for b in (2, 3, 10):
                if math.gcd(n, b) != 1:
                    continue
                L = _L(n, b)
                if L < 3 or L > 150:
                    continue
                e = zeros.exact_zero_classes(n, b, L)
                self.assertEqual(e, zeros.float_zero_classes(n, b, L), (n, b, L))
                f = zeros.pair_zeros(n, b, L)
                self.assertEqual(int(f[2]), len(e), (n, b))
                if lengths.midy_flag(n, b, L):
                    self.assertTrue({k for k in range(2, L + 1) if (L // 2) % k == 0} <= e, (n, b))
                seen_extra += int(f[3] > 0)
                self.assertEqual(f[7], 0)  # classes vanish all-or-nothing
        self.assertGreater(seen_extra, 50)

    def test_character_expansion_and_euler_factors(self):
        import sympy

        for p, q, b in ((7, 13, 10), (5, 11, 2), (3, 17, 2), (11, 13, 2), (13, 31, 3)):
            n = p * q
            gp, gq = int(sympy.primitive_root(p)), int(sympy.primitive_root(q))
            ind_p = {pow(gp, e, p): e for e in range(p - 1)}
            ind_q = {pow(gq, e, q): e for e in range(q - 1)}
            G = [x for x in range(1, n) if math.gcd(x, n) == 1]

            def chi(a, c, x):
                return cmath.exp(2j * math.pi * (a * ind_p[x % p] / (p - 1) + c * ind_q[x % q] / (q - 1)))

            B1 = {(a, c): sum(chi(a, c, x) * x for x in G) / n for a in range(p - 1) for c in range(q - 1)}
            # Euler factors: chi with trivial p-part is induced by chi_q^c mod q
            for c in range(1, q - 1):
                Bq = sum(cmath.exp(2j * math.pi * c * ind_q[y] / (q - 1)) * y for y in range(1, q)) / q
                self.assertAlmostEqual(abs(B1[(0, c)] - Bq * (1 - cmath.exp(2j * math.pi * c * ind_q[p % q] / (q - 1)))), 0, places=8)
            for a in range(1, p - 1):
                Bp = sum(cmath.exp(2j * math.pi * a * ind_p[y] / (p - 1)) * y for y in range(1, p)) / p
                self.assertAlmostEqual(abs(B1[(a, 0)] - Bp * (1 - cmath.exp(2j * math.pi * a * ind_p[q % p] / (p - 1)))), 0, places=8)
            # X_j(r) = (n/m) * sum of B_{1,chi} over the characters extending psi_j
            L = _L(n, b)
            m = len(G) // L
            X = np.fft.fft(ndr.repetend_remainders(n, b, L).astype(float))
            for j in range(L):
                target = cmath.exp(-2j * math.pi * j / L)
                ext = [k for k in B1 if abs(chi(*k, b) - target) < 1e-9]
                self.assertEqual(len(ext), m)
                self.assertAlmostEqual(abs(X[j] - n / m * sum(B1[k] for k in ext)), 0, delta=1e-6 * n)


CFG = {
    "n_min": 2,
    "n_max": 999,
    "stratified": [{"n_lo": 1000, "n_hi": 9999, "per_omega": 80}, {"n_lo": 10000, "n_hi": 60000, "per_omega": 80}],
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13]},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}


class ZerosStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.table, _ = dataset.build_dataset(CFG, SPLIT, seed=5, log=lambda *_: None)

    def test_run_stage(self):
        zcfg = {"bins": [[100, 999], [1000, 9999], [10000, 60000]], "per_omega_cap": 80, "min_per_omega": 10, "bootstrap": 30, "workers": 1,
                "exact_sample": 40, "exact_max_L": 120, "length_bases": "2..40", "gbm": {"max_iter": 30, "learning_rate": 0.2}}
        ccfg = {"entropies": ["H_paper", "H_nodc"], "moduli": [6, 12], "permutations": 20, "bootstrap": 20}
        with tempfile.TemporaryDirectory() as d:
            out = zeros.run_stage(self.table, zcfg, ccfg, CFG["base_sets"], sf.DEFAULT_INPUTS, 5, Path(d), "testkey", log=lambda *_: None)
            again = zeros.run_stage(self.table, zcfg, ccfg, CFG["base_sets"], sf.DEFAULT_INPUTS, 5, Path(d), "testkey", log=lambda *_: None)
            import figures
            import report
            import runtime

            runtime.write_json(Path(d) / "zeros.json", out)
            made = {Path(f).name for f, _ in figures.make_figures(Path(d), None)}
            self.assertIn("spectral_zeros_by_omega.png", made)
            md = "\n".join(report.zeros_markdown(runtime.read_json(Path(d) / "zeros.json")))
            self.assertIn("## Step 8", md)
        self.assertEqual(again["c9_bins"], out["c9_bins"])  # cached zeros, same result
        ex = out["exact_check"]
        self.assertGreater(ex["pairs"], 20)
        self.assertEqual(ex["agree"], ex["pairs"], ex["mismatches"])
        self.assertEqual(out["describe"]["numerics"]["inconsistent_classes"], 0)
        by = out["describe"]["by_omega"]
        self.assertLess(by[1]["share_with_extra"], by[2]["share_with_extra"])
        self.assertIn("L98 + Midy98 + zeros vs L98 + Midy98", out["c9_bins"][-1]["gain"])
        self.assertIn("plus_midy_zeros", out["residues"]["variants"]["H_paper"])

    def test_whisky_stage(self):
        import report
        import whisky_spectra

        x = np.random.default_rng(0).random(64)
        lin = whisky_spectra.linear_periodic(x, 3)
        np.testing.assert_array_equal(lin[::4], x)
        out = whisky_spectra.run_stage(self.table, {"bins": [[100, 999], [1000, 60000]], "per_stratum": 25, "max_L": 3000, "bootstrap": 30},
                                       CFG["base_sets"], 5, log=lambda *_: None)
        self.assertGreater(out["pairs"], 100)
        self.assertLess(out["leakage"]["lin"]["max_log10"], -12)  # linear interpolation keeps exact zeros
        self.assertAlmostEqual(out["models"]["H_fft"]["midy_coefficient"], -math.log(2), delta=0.15)
        self.assertIn("## Step 8e", "\n".join(report.whisky_markdown(out)))


if __name__ == "__main__":
    unittest.main()
