"""Unit tests for ndr.py: number theory, repetends, entropy definitions, old-version reproduction.

Run from the repository root:
    .venv/bin/python -m unittest discover -s empirical-scripts/11.0/tests -v
"""

from __future__ import annotations

import json
import math
import pickle
import random
import sys
import unittest
from pathlib import Path

import numpy as np

V11 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V11))

import ndr  # noqa: E402

REPO = V11.parents[1]
V8_PICKLE = REPO / "empirical-scripts/8.0/results/1/sieve_echo_complete_20250825_224642.pkl"
V8_JSON = REPO / "empirical-scripts/8.0/results/1/sieve_echo_results_20250825_224642.json"
V10_STATE = REPO / "empirical-scripts/results/3-Sept-2025/sieve_echo_state_v10.json"
V8_BASES = [2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16]
V10_BASES = [2, 3, 4, 5, 6, 7, 10, 12, 11, 13, 15, 17, 19, 23, 29, 31]


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """The v8 pickle only needs NumPy scalars; refuse anything else."""

    ALLOWED = {("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def features(n: int, b: int, spf) -> dict[str, float]:
    lam, lp = ndr.order_context(n, spf)
    L = ndr.multiplicative_order(b, n, lam, lp)
    return dict(zip(ndr.PAIR_FEATURES, ndr.pair_features(n, b, L).tolist()))


class NumberTheoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spf = ndr.spf_sieve(5000)

    def test_spf_sieve(self):
        for n in range(2, 5001):
            p = int(self.spf[n])
            self.assertEqual(n % p, 0)
            self.assertEqual(p, min(d for d in range(2, n + 1) if n % d == 0))

    def test_labels_match_sympy(self):
        import sympy

        ns = np.arange(2, 3001)
        lab = ndr.arithmetic_labels(ns, self.spf)
        for i, n in enumerate(ns.tolist()):
            f = sympy.factorint(n)
            self.assertEqual(lab["omega"][i], len(f))
            self.assertEqual(lab["Omega"][i], sum(f.values()))
            self.assertEqual(lab["tau"][i], sympy.divisor_count(n))
            self.assertEqual(lab["phi"][i], sympy.totient(n))
            self.assertEqual(lab["lambda"][i], sympy.reduced_totient(n))
            self.assertEqual(lab["mobius"][i], sympy.mobius(n))
            self.assertEqual(bool(lab["is_prime"][i]), sympy.isprime(n))

    def test_order_matches_long_division(self):
        for n in range(2, 2000):
            lam, lp = ndr.order_context(n, self.spf)
            for b in (2, 3, 7, 10, 16, 31, 97):
                if math.gcd(n, b) != 1:
                    with self.assertRaises(ValueError):
                        ndr.multiplicative_order(b, n, lam, lp)
                    continue
                L = ndr.multiplicative_order(b, n, lam, lp)
                self.assertEqual(L, len(ndr.long_division_repetend(n, b)), (n, b))
                self.assertEqual(pow(b, L, n), 1 % n)


class RepetendTests(unittest.TestCase):
    def test_one_seventh(self):
        self.assertEqual(ndr.repetend_digits(7, 10, 6).tolist(), [1, 4, 2, 8, 5, 7])

    def test_digits_match_long_division(self):
        spf = ndr.spf_sieve(1200)
        for n in range(2, 1200):
            lam, lp = ndr.order_context(n, spf)
            for b in (2, 10, 13):
                if math.gcd(n, b) == 1:
                    L = ndr.multiplicative_order(b, n, lam, lp)
                    self.assertEqual(ndr.repetend_digits(n, b, L).tolist(), ndr.long_division_repetend(n, b))

    def test_large_n_no_overflow(self):
        n, b = 2_000_003, 10  # prime; remainders^2 ~ 4e12 stays inside int64
        spf = ndr.spf_sieve(n)
        lam, lp = ndr.order_context(n, spf)
        L = ndr.multiplicative_order(b, n, lam, lp)
        r = ndr.repetend_remainders(n, b, L)
        self.assertEqual(int(r[-1]) * b % n, 1)
        self.assertTrue(((r >= 0) & (r < n)).all())


class EntropyTests(unittest.TestCase):
    def test_scale_invariance_d_over_b_vs_d_over_b_minus_1(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            b = int(rng.integers(2, 40))
            d = rng.integers(0, b, size=int(rng.integers(2, 300))).astype(float)
            for variant in ("paper", "nodc", "v8", "v10"):
                h1 = ndr.spectral_entropy(d / b, variant)
                h2 = ndr.spectral_entropy(d / (b - 1), variant)
                if math.isnan(h1):
                    self.assertTrue(math.isnan(h2))
                else:
                    self.assertAlmostEqual(h1, h2, places=10)

    def test_constant_sequence(self):
        x = np.full(16, 0.3)
        self.assertEqual(ndr.spectral_entropy(x, "paper"), 0.0)  # all power in DC
        self.assertTrue(math.isnan(ndr.spectral_entropy(x, "nodc")))  # no AC power

    def test_full_spectrum_matches_fft(self):
        rng = np.random.default_rng(1)
        for L in (2, 3, 4, 7, 8, 101, 256):
            x = rng.random(L)
            P = np.abs(np.fft.fft(x)) ** 2
            p = P / P.sum()
            ref = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            self.assertAlmostEqual(ndr.spectral_entropy(x, "paper"), ref, places=10)
            q = P[1:] / P[1:].sum()
            ref = float(-(q[q > 0] * np.log(q[q > 0])).sum())
            self.assertAlmostEqual(ndr.spectral_entropy(x, "nodc"), ref, places=10)

    def test_entropy_bounds(self):
        spf = ndr.spf_sieve(3000)
        for n in range(3, 3000, 7):
            for b in (2, 10):
                if math.gcd(n, b) == 1:
                    f = features(n, b, spf)
                    self.assertLessEqual(f["H_paper"], math.log(f["L"]) + 1e-9)
                    if not math.isnan(f["H_nodc"]):
                        self.assertLessEqual(f["H_nodc"], math.log(f["L"] - 1) + 1e-9)

    def test_moments_match_scipy(self):
        from scipy import stats

        spf = ndr.spf_sieve(2000)
        for n in (97, 997, 1999):
            f = features(n, 10, spf)
            lam, lp = ndr.order_context(n, spf)
            x = ndr.ndr(ndr.repetend_digits(n, 10, ndr.multiplicative_order(10, n, lam, lp)), 10)
            self.assertAlmostEqual(f["mean"], x.mean(), places=12)
            self.assertAlmostEqual(f["std"], x.std(), places=12)
            self.assertAlmostEqual(f["skew"], stats.skew(x), places=9)
            self.assertAlmostEqual(f["kurt"], stats.kurtosis(x), places=9)


@unittest.skipUnless(V8_PICKLE.is_file(), "v8 stored results not present")
class ReproduceV8Tests(unittest.TestCase):
    """v8.2 (`8.0/sieve-echo-tests-2.py`) produced the paper's numbers; re-derive its stored values."""

    @classmethod
    def setUpClass(cls):
        with open(V8_PICKLE, "rb") as fh:
            cls.rows = _NumpyOnlyUnpickler(fh).load()["data"]
        cls.spf = ndr.spf_sieve(10000)

    def test_per_base_entropy_and_length(self):
        rows = random.Random(0).sample(self.rows, 400)
        checked = 0
        for row in rows:
            n = row["n"]
            hs = []
            for b in V8_BASES:
                key = f"ndr_entropy_b{b}"
                if math.gcd(n, b) != 1:
                    self.assertNotIn(key, row)
                    continue
                f = features(n, b, self.spf)
                self.assertEqual(int(f["L"]), row[f"length_b{b}"], (n, b))
                self.assertAlmostEqual(f["H_v8"], float(row[key]), places=9, msg=(n, b))
                self.assertAlmostEqual(f["mean"], float(row[f"mean_b{b}"]), places=12)
                hs.append(f["H_v8"])
                checked += 1
            if hs:
                self.assertAlmostEqual(float(np.mean(hs)), float(row["ndr_entropy_mean"]), places=9)
        self.assertGreater(checked, 2000)

    def test_sieve_echo_law_fit(self):
        """Stored law: OLS of ndr_entropy_mean on log(omega + 1) (the code adds 1; the paper omits it)."""
        stored = json.loads(V8_JSON.read_text())["results"]["patterns"]["sieve_echo_law"]
        rows = [r for r in self.rows if "ndr_entropy_mean" in r]
        x = np.log(np.array([r["omega"] for r in rows], float) + 1)
        y = np.array([r["ndr_entropy_mean"] for r in rows], float)
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        self.assertAlmostEqual(slope, stored["alpha"], places=6)
        self.assertAlmostEqual(intercept, stored["beta"], places=6)
        self.assertAlmostEqual(r2, stored["r2"], places=6)


@unittest.skipUnless(V10_STATE.is_file(), "v10 state not present")
class ReproduceV10Tests(unittest.TestCase):
    def test_aggregates_for_shortest_repetends(self):
        from scipy import stats

        data = json.loads(V10_STATE.read_text())["data"]
        rows = sorted((r for r in data if "entropy_mean" in r), key=lambda r: r["length_mean"])[:6]
        spf = ndr.spf_sieve(max(r["n"] for r in rows))
        for row in rows:
            n = row["n"]
            H, Ls, K = [], [], []
            for b in V10_BASES:
                if math.gcd(n, b) != 1:
                    continue
                lam, lp = ndr.order_context(n, spf)
                L = ndr.multiplicative_order(b, n, lam, lp)
                if L <= 3:
                    continue
                f = dict(zip(ndr.PAIR_FEATURES, ndr.pair_features(n, b, L).tolist()))
                x = ndr.repetend_digits(n, b, L) / (b - 1)
                Ls.append(L)
                K.append(stats.kurtosis(x))
                if not math.isnan(f["H_v10"]):
                    H.append(f["H_v10"])
            self.assertAlmostEqual(float(np.mean(Ls)), row["length_mean"], places=6, msg=n)
            self.assertAlmostEqual(float(np.mean(H)), row["entropy_mean"], places=8, msg=n)
            self.assertAlmostEqual(float(np.mean(K)), row["kurtosis_mean"], places=6, msg=n)


if __name__ == "__main__":
    unittest.main()
