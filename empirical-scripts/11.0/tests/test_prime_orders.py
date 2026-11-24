"""Independent checks of the order-index counts and formula input contracts."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prime_orders
import prime_order_formulas
import runtime


class TestPrimeOrders(unittest.TestCase):
    def test_quadratic_classes_and_prime_power_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = prime_orders.run_stage({"cutoffs": [10, 100]}, 0, Path(tmp), log=lambda _: None)
        quad = data["quadratic"]
        self.assertEqual(quad["densities"], [1 / 8] * 8)
        self.assertEqual(quad["classes"][7], [1, 49, 71, 119])
        last = quad["rows"][-1]
        self.assertEqual(last["cumulative"]["eligible"], 22)
        self.assertEqual(last["interval"]["eligible"], 21)
        # Of the eligible primes, precisely 7 contributes a higher prime power through 100.
        np.testing.assert_allclose(last["character_prime_power_corrections"], math.log(7))
        self.assertAlmostEqual(
            last["joint_character_residual"], last["cumulative"]["log_weighted_discrepancies"][7],
        )

    def test_cubic_and_mixed_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = prime_orders.run_stage({"cutoffs": [100]}, 0, Path(tmp), log=lambda _: None)
        row = data["cubic"]["rows"][0]
        self.assertEqual(row["eligible"], 23)
        self.assertEqual(row["counts"], [2, 3, 0])
        self.assertEqual(data["cubic"]["degrees"], [6, 6, 18])
        self.assertGreater(row["grh_joint_count_error_bound"], 0)
        mixed = data["mixed"]["rows"][0]
        self.assertEqual(mixed["eligible"], 23)
        self.assertEqual(mixed["quintic"], mixed["joint"])

    def test_fit_ranges_and_residue_ceiling(self):
        case = {
            "p": np.array([7, 13, 19, 31, 43, 67]),
            "X": np.array([[7], [13], [19], [31], [7], [7]]),
            "y": np.array([0, 1, 0, 1, 0, 1]),
            "names": ["synthetic residue"],
            "operations": ["ADD"],
        }
        data, baseline = prime_order_formulas.prepare(case, 13, 31, 0)
        self.assertEqual(baseline["train_rows"], 2)
        self.assertEqual(baseline["val_rows"], 2)
        self.assertEqual(baseline["test_rows"], 2)
        self.assertEqual(prime_order_formulas.residue_baseline(data)["test_errors_unavoidable_by_residue"], 1)

    def test_replay_public_evolvo_program(self):
        runtime.bootstrap_evolvo()
        from evolvo import Category, DataType, Operation
        from evolvo.slots import pack_type_index

        a, b, output = [pack_type_index(DataType.DECIMAL, index) for index in range(3)]
        slots = [Category.VARIABLE, output, Operation.MUL, Category.VARIABLE, a, Category.VARIABLE, b]
        program = {"inputs": ["a", "b"], "slots": [slots], "output_index": 2}
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(prime_order_formulas.replay(program, X), [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
