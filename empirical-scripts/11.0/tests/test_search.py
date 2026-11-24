"""Tests for step 4: S1 sizer, formula decoding, one formula run per guide, baselines, GFSL-NN stage."""

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

import runtime  # noqa: E402

runtime.bootstrap_evolvo()

import baselines  # noqa: E402
import dataset  # noqa: E402
import features  # noqa: E402
import search_formulas as sf  # noqa: E402

SMALL = {
    "n_min": 2,
    "n_max": 1500,
    "stratified": {"enabled": True, "n_lo": 1501, "n_hi": 30000, "per_omega": 30},
    "base_sets": {"v8": [2, 3, 5, 7, 10, 11], "v10": [3, 7, 10, 13], "primes31": [2, 3, 5, 7, 11, 13]},
    "workers": 1,
}
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": "omega"}


class SizerTests(unittest.TestCase):
    def test_monotone_and_driven_by_q(self):
        s = sf.Sizer(5000, 100000, 5000, q_target=0.9, se_trigger=False, n_train=200000)
        self.assertEqual(s.N, 5000)
        self.assertEqual(s.update(0.5, 0.1, 0.001), 5000)  # sets q0
        self.assertEqual(s.update(0.7, 0.1, 0.001), 50000)  # halfway to q_target
        self.assertEqual(s.update(0.6, 0.1, 0.001), 50000)  # never decreases
        self.assertEqual(s.update(0.95, 0.1, 0.001), 100000)

    def test_se_trigger_and_caps(self):
        s = sf.Sizer(5000, 100000, 5000, q_target=0.9, se_trigger=True, n_train=12000)
        self.assertEqual(s.n_max, 12000)
        s.update(0.5, 0.1, 0.001)
        self.assertEqual(s.update(0.5, 0.001, 0.01), 10000)  # spread < 2 SE -> one step up
        self.assertEqual(s.update(0.5, 0.001, 0.01), 12000)  # capped at the train size
        f = sf.Sizer(5000, 100000, 5000, 0.9, True, 200000, fixed=5000)
        self.assertEqual(f.update(0.99, 0.0, 1.0), 5000)


class FormulaDecodeTests(unittest.TestCase):
    def test_expression(self):
        from evolvo import Category, DataType, GFSLGenome, GFSLInstruction, Operation, pack_type_index

        g = GFSLGenome("algorithm", inputs=2)
        v = lambda i: [Category.VARIABLE, pack_type_index(DataType.DECIMAL, i)]  # noqa: E731
        g.add_instruction(GFSLInstruction([*v(2), Operation.MUL, *v(0), *v(1)]))
        g.add_instruction(GFSLInstruction([*v(2), Operation.LOG, *v(2), Category.NONE, 0]))
        g.mark_output(DataType.DECIMAL, 2)
        self.assertEqual(sf.formula_expression(g, ["a", "b"], 2), "log(([a] * [b]))")


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.table, _ = dataset.build_dataset(SMALL, SPLIT, seed=7, log=lambda *_: None)

    def test_formula_run_each_guide(self):
        cfg = {**sf.DEFAULTS, "population": 16, "generations": 3, "eval_size_min": 200, "eval_size_step": 100, "length_bases_sets": ["v8", "v10", "primes31"]}
        prep = sf.prepare(self.table, cfg, SMALL["base_sets"], 7)
        sf._init_worker({"names": prep["names"], "targets": prep["targets"]})
        with tempfile.TemporaryDirectory() as d:
            kfile = str(Path(d) / "p.pt")
            runs = []
            for guide in ("none", "qlearning", "supervised", "policy"):
                spec = {"target": "omega", "guide": guide, "seed": 1, "variant": "adaptive", "cfg": cfg}
                if guide == "policy":
                    spec["knowledge_out"] = kfile
                r = sf.run_one(spec)
                runs.append(r)
                self.assertEqual(r["generations"], 3)
                self.assertGreater(r["saved"], 0)
                self.assertIn("expression", r["selected"])
                self.assertTrue(math.isfinite(r["selected"]["test"]["accuracy"]))
                Ns = [c["N"] for c in r["curve"]]
                self.assertEqual(Ns, sorted(Ns))
            warm = sf.run_one({"target": "is_prime", "guide": "policy", "seed": 2, "variant": "warm", "knowledge_in": kfile, "cfg": cfg})
            self.assertEqual(warm["guide_summary"]["loaded_from"], kfile)
            fixed = sf.run_one({"target": "omega", "guide": "none", "seed": 1, "variant": "fixed", "cfg": {**cfg, "fixed_size": 150}})
            self.assertEqual({c["N"] for c in fixed["curve"]}, {150})
            summary = sf.summarize(runs + [warm, fixed], cfg)
            self.assertIn("omega", summary["best_by_validation"])
            self.assertIn("S1", summary["verdicts"]["omega"])

    def test_inputs_are_allowed_only(self):
        with self.assertRaises(ValueError):
            features.column(self.table, "label.omega", {}, [2, 3])
        with self.assertRaises(ValueError):
            features.column(self.table, "target_side.L_over_phi.v8.mean", {}, [2, 3])

    def test_baselines_and_three_feature(self):
        out = baselines.run_baselines(self.table, {"gbm": {"max_iter": 20, "learning_rate": 0.2}}, SMALL["base_sets"], sf.DEFAULT_INPUTS, 7, log=lambda *_: None)
        rows = out["targets"]["omega"]
        self.assertIn("majority", rows)
        self.assertIn("GBM: n + L arithmetic", rows)
        tf = out["three_feature_retest"]["variants"]
        self.assertIn("LEAKAGE DEMO: std H, min H, is_prime_power", tf)

    def test_nn_stage_runs(self):
        import search_nn

        with tempfile.TemporaryDirectory() as d:
            cfg = {"population": 2, "generations": 1, "patience": 1, "max_train_rows": 0, "max_val_rows": 0, "hard_max_epochs": 5, "max_rows": 2, "max_params": 5000}
            out = search_nn.run_stage(self.table, cfg, SMALL["base_sets"], sf.DEFAULT_INPUTS, 7, Path(d), log=lambda *_: None)
            for enc in ("vec", "set"):
                self.assertIn(enc, out["encodings"])
                self.assertTrue(0 <= out["encodings"][enc]["test"]["accuracy"] <= 1)
                self.assertTrue((Path(d) / out["encodings"][enc]["knowledge_out"]).is_file())


if __name__ == "__main__":
    unittest.main()
