"""Evolvo symbol search for the arithmetic relationships in the prime-order benchmarks."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import runtime
import search_formulas

BIT_OPERATIONS = ["ADD", "SUB", "MUL", "ABS"]
RESIDUE_OPERATIONS = BIT_OPERATIONS + ["DIV", "MOD"]


def cases(primes: np.ndarray, quadratic: np.ndarray, cubic: np.ndarray, quintic: np.ndarray, *,
          fold_mod8: bool = False) -> dict:
    """Give the symbolic runs either marginal event flags or small congruence inputs."""
    base2, _, base5 = quadratic.T
    cube2, cube3 = cubic.T
    congruent = primes % 3 == 1
    distractor = primes % 5 == 1

    def mod8_inputs(residues):
        values = np.abs(residues - 4) if fold_mod8 else residues
        return np.column_stack((values, np.ones_like(residues), np.full_like(residues, 2), np.full_like(residues, 4)))

    bits3 = np.array(list(product((0, 1), repeat=3)))
    bits4 = np.array(list(product((0, 1), repeat=4)))
    mixed = bits3[bits3[:, 1] <= bits3[:, 0]]
    residues = np.array([1, 3, 5, 7])
    return {
        "quadratic_base2": {
            "p": primes, "X": mod8_inputs(primes % 8), "y": base2,
            "names": ["abs((p mod 8) − 4)" if fold_mod8 else "p mod 8", "1", "2", "4"],
            "operations": RESIDUE_OPERATIONS,
            "audit": (mod8_inputs(residues), np.isin(residues, [1, 7])),
        },
        "quadratic_joint": {
            "p": primes, "X": np.column_stack((quadratic, congruent)), "y": quadratic.all(axis=1),
            "names": ["2 divides I_2", "2 divides I_3", "2 divides I_5", "p ≡ 1 mod 3"],
            "operations": BIT_OPERATIONS, "audit": (bits4, bits4[:, :3].all(axis=1)),
        },
        "cubic_joint": {
            "p": primes, "X": np.column_stack((cubic, distractor)), "y": cube2 & cube3,
            "names": ["3 divides I_2", "3 divides I_3", "p ≡ 1 mod 5"], "operations": BIT_OPERATIONS,
            "audit": (bits3, bits3[:, :2].all(axis=1)),
        },
        "mixed_joint": {
            "p": primes, "X": np.column_stack((base5, quintic, congruent)), "y": base5 & quintic,
            "names": ["2 divides I_5", "5 divides I_2", "p ≡ 1 mod 3"], "operations": BIT_OPERATIONS,
            "audit": (mixed, mixed[:, 1].astype(bool)),
        },
        "cubic_congruence_only": {
            "p": primes[congruent], "X": np.column_stack([primes[congruent] % m for m in (5, 8, 12, 120)]),
            "y": (cube2 & cube3)[congruent], "names": ["p mod 5", "p mod 8", "p mod 12", "p mod 120"],
            "operations": RESIDUE_OPERATIONS,
        },
    }


def prepare(case: dict, train_end: int, val_end: int, seed: int) -> tuple[dict, dict]:
    p, X, y = case["p"], case["X"].astype(float), case["y"].astype(float)
    tr = np.flatnonzero(p <= train_end)
    va = np.flatnonzero((p > train_end) & (p <= val_end))
    te = np.flatnonzero(p > val_end)
    if not all(group.size > 0 for group in (tr, va, te)):
        raise ValueError("prime order formula search requires nonempty train, validation, and test ranges")
    if not 0 < y[tr].mean() < 1:
        raise ValueError("prime order formula search requires both classes in the train range")
    perm = np.random.default_rng([seed, 17]).permutation(tr)
    rate = float(y[tr].mean())
    target = {
        "names": case["names"], "operations": case["operations"],
        "Xp": X[perm], "yp": y[perm], "Xv": X[va], "yv": y[va], "Xt": X[te], "yt": y[te],
        "var": float(np.var(y[tr])),
    }
    if "audit" in case:
        target["audit"] = case["audit"]
    baseline = {
        "train_rows": int(tr.size), "val_rows": int(va.size), "test_rows": int(te.size),
        "test_positives": int(y[te].sum()), "train_rate": rate, "test_rate": float(y[te].mean()),
        "constant_mse": float(np.mean((y[te] - rate) ** 2)),
        "majority_accuracy": float(np.mean(y[te] == (rate > 0.5))),
    }
    return target, baseline


def residue_baseline(data: dict) -> dict:
    """Train lookup and a descriptive accuracy ceiling for any rule of the same residue."""
    train_residues = data["Xp"][:, -1].astype(int)
    test_residues = data["Xt"][:, -1].astype(int)
    counts = np.bincount(train_residues, minlength=120)
    positive = np.bincount(train_residues, weights=data["yp"], minlength=120)
    rate = np.full(120, data["yp"].mean())
    np.divide(positive, counts, out=rate, where=counts > 0)
    pred = rate[test_residues]
    n = np.bincount(test_residues, minlength=120)
    pos = np.bincount(test_residues, weights=data["yt"], minlength=120)
    return {
        "test_mse": float(np.mean((pred - data["yt"]) ** 2)),
        "test_accuracy": float(np.mean((pred > 0.5) == data["yt"])),
        "test_f1": float(f1_score(data["yt"], pred > 0.5, zero_division=0)),
        "test_residue_classes": int(np.count_nonzero(n)),
        "test_classes_with_both_labels": int(np.count_nonzero((pos > 0) & (pos < n))),
        "test_errors_unavoidable_by_residue": int(np.minimum(pos, n - pos).sum()),
    }


def replay(program: dict, X: np.ndarray) -> np.ndarray:
    """Evaluate the saved Evolvo slots on columns in the reported order."""
    runtime.bootstrap_evolvo()
    from evolvo import DataType, GFSLExecutor, GFSLGenome, GFSLInstruction

    n_inputs = len(program["inputs"])
    genome = GFSLGenome(inputs=n_inputs)
    genome.instructions = [GFSLInstruction(list(slots)) for slots in program["slots"]]
    index = program["output_index"]
    genome.mark_output(DataType.DECIMAL, index)
    executor = GFSLExecutor(compute_backend="cpu")
    inputs = {f"d${i}": X[:, i] for i in range(n_inputs)}
    return executor.execute_batch(genome, inputs, track_activity=False)[f"d${index}"]


def run_stage(cfg: dict, seed: int, run_dir: Path, primes, quadratic, cubic, fifth, log=print) -> dict:
    tasks = cases(primes, quadratic, cubic, fifth, fold_mod8=cfg.get("fold_mod8", False))
    names = cfg.get("targets", list(tasks))
    if not names or len(set(names)) != len(names) or any(name not in tasks for name in names):
        raise ValueError("prime order formula targets must be a nonempty list of distinct known targets")
    offsets = cfg["seed_offsets"]
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("prime order formula seed_offsets must be nonempty and distinct")
    train_end, val_end = cfg["split_at"]
    if train_end >= val_end:
        raise ValueError("prime order formula split cutoffs must increase")
    eval_cfg = {**search_formulas.DEFAULTS, **cfg, "q_target": 0.98}
    targets, baselines = {}, {}
    for name in names:
        targets[name], baselines[name] = prepare(tasks[name], train_end, val_end, seed)
    if "cubic_congruence_only" in names:
        baselines["cubic_congruence_only"]["mod_120_lookup"] = residue_baseline(targets["cubic_congruence_only"])

    knowledge = run_dir / "knowledge"
    knowledge.mkdir(exist_ok=True)
    specs = [{
        "target": name, "guide": "policy", "variant": "adaptive", "seed": seed + offset,
        "cfg": eval_cfg, "export_program": True,
        "knowledge_out": str(knowledge / f"prime_order_{name}_s{seed + offset}.pt"),
    } for name in targets for offset in offsets]
    log(f"  evolving {len(specs)} formulas with the PyTorch policy guide")
    results = search_formulas._run_specs(specs, {"names": [], "targets": targets}, cfg["workers"], log)
    summary = {}
    for name, data in targets.items():
        group = [r for r in results if r["target"] == name]
        best = min(group, key=lambda r: (
            r["selected"]["val"]["nmse"], r["selected"]["effective_instructions"], r["seed"],
        ))
        program = best["selected"]["program"]
        replayed = replay(program, data["Xv"])
        expected = best["selected"]["val"]["mse"]
        if not np.isclose(np.mean((replayed - data["yv"]) ** 2), expected):
            raise ArithmeticError(f"saved GFSL program does not reproduce validation for {name}")
        summary[name] = {
            "baseline": baselines[name], "selected_seed": best["seed"], **best["selected"],
            "runs": [{
                "seed": r["seed"], "expression": r["selected"]["expression"],
                "validation": r["selected"]["val"], "held_out": r["selected"]["test"],
                "audit": r["selected"].get("audit"), "evaluations": r["evaluations"],
                "final_N": r["final_N"], "N_changes": r["N_changes"],
                "knowledge": f"knowledge/{Path(r['knowledge_out']).name}",
            } for r in group],
        }
    return {
        "split": {"train_end": train_end, "val_end": val_end, "test_end": int(primes[-1])},
        "settings": cfg, "guide": "GFSLPolicyGuide", "fitness": "1/(1+NMSE) minus instruction parsimony",
        "targets": summary,
    }
