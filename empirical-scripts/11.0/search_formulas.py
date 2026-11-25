"""Evolvo GFSL formula search for omega(n) and is_prime(n) (ROADMAP step 4, S1, S2).

One *run* = one target x guide x seed x variant:

* genomes read the standardized allowed inputs ``d$0..d${k-1}`` and write ``d$k``
  (``GFSLEvolver.initialize_population(inputs=k, outputs=[(DECIMAL, k)],
  profile=InstructionProfile.formula())``), executed with ``GFSLExecutor.execute_batch``;
* fitness = ``1 / (1 + NMSE) - parsimony * effective_instructions`` on the first ``N`` rows of one
  seeded permutation of the **train** split (NMSE = MSE / Var(y_train)); bounded above by 1, the
  parsimony term is explicit (AGENTS.md contract);
* **S1**: ``N`` starts at ``eval_size_min`` and rises in ``eval_size_step`` steps to at most
  ``eval_size_max`` (capped by the train size) as ``q`` = top-``top_fraction`` mean fitness moves
  from its generation-0 value ``q0`` toward ``q_target``, or when the top-k spread falls below two
  standard errors of the best genome's fitness. ``N`` never decreases; when it changes Evolvo's
  ``fitness_epoch`` hook re-scores the whole population, so fitness values taken at different
  ``N`` are never compared. ``N`` is stored with every saved fitness;
* saving (v10): each generation the top ``save_top_per_gen`` genomes with fitness ≥
  ``save_threshold_ratio`` × q (same ``N``) are saved once by signature; the final top
  ``safety_net`` are always saved;
* selection: lowest validation NMSE among saved genomes (ties → fewer instructions); the test
  split is read once, for the selected genome.

Variants: ``adaptive`` (S1), ``fixed`` (N = ``fixed_size``, S1 ablation), ``knowledge`` (one extra
policy-guide run whose weights seed the ``warm`` runs), ``warm`` (policy guide loaded from that
file, S2 persistence check). Guides: ``none``, ``qlearning``, ``supervised``, ``policy``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import dataset
import features
import runtime

DEFAULT_INPUTS = [
    "log n",
    "row.v8.logL.mean",
    "row.v8.H_paper.mean",
    "row.v8.H_nodc.mean",
    "row.v8.H_paper.std",
    "row.v8.H_paper.min",
    "row.v8.Hn_nodc.mean",
    "row.v8.mean.mean",
    "log lcm(L_b)",
    "log n - log lcm",
    "lcm divides n-1",
    "share of L_b dividing n-1",
]

DEFAULTS: Dict[str, Any] = {
    "targets": ["omega", "is_prime"],
    "inputs": DEFAULT_INPUTS,
    "length_bases_sets": ["v8", "v10", "primes31"],
    "population": 120,
    "generations": 40,
    "initial_instructions": 8,
    "max_instructions": 32,
    "max_effective_instructions": 16,
    "parsimony": 0.002,
    "eval_size_min": 5000,
    "eval_size_max": 100000,
    "eval_size_step": 5000,
    "q_target": {"omega": 0.8, "is_prime": 0.95},
    "se_trigger": True,
    "top_fraction": 0.10,
    "save_threshold_ratio": 0.75,
    "save_top_per_gen": 5,
    "safety_net": 5,
    "guides": ["none", "qlearning", "supervised", "policy"],
    "seeds": [0, 1, 2, 3, 4],
    "ablation_guide": "none",
    "fixed_size": 5000,
    "knowledge_seed": 999,
    "warm_start": True,
    "max_omega_class": 5,
    "workers": 0,
}


# ---------------------------------------------------------------------------------------------
# S1 evaluation size
# ---------------------------------------------------------------------------------------------


class Sizer:
    """Monotone evaluation size driven by the top-k mean fitness (ROADMAP S1)."""

    def __init__(self, n_min: int, n_max: int, step: int, q_target: float, se_trigger: bool, n_train: int, fixed: Optional[int] = None):
        self.n_max = int(min(n_max, n_train))
        self.n_min = int(min(n_min, self.n_max))
        self.step = max(1, int(step))
        self.q_target = float(q_target)
        self.se_trigger = bool(se_trigger)
        self.fixed = fixed is not None
        self.N = int(min(fixed, self.n_max)) if fixed is not None else self.n_min
        self.q0: Optional[float] = None

    def update(self, q: float, spread: float, se: float) -> int:
        if self.fixed or not math.isfinite(q):
            return self.N
        if self.q0 is None:
            self.q0 = q
        if self.q_target > self.q0:
            frac = min(1.0, max(0.0, (q - self.q0) / (self.q_target - self.q0)))
        else:
            frac = 1.0
        target = self.n_min + (self.n_max - self.n_min) * frac
        target = self.n_min + math.floor((target - self.n_min) / self.step + 1e-9) * self.step
        if frac >= 1.0:
            target = self.n_max
        new = max(self.N, int(target))
        if self.se_trigger and math.isfinite(se) and math.isfinite(spread) and spread < 2.0 * se:
            new = max(new, self.N + self.step)
        self.N = int(min(new, self.n_max))
        return self.N


# ---------------------------------------------------------------------------------------------
# Readable formulas
# ---------------------------------------------------------------------------------------------

_INFIX = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "POW": "^", "MOD": "%"}


def formula_expression(genome, input_names: Sequence[str], output_index: int) -> str:
    """Symbolic expression of ``d$output_index`` over the named inputs (effective instructions)."""
    from evolvo import Category, DataType, Operation
    from evolvo.custom_ops import resolve_operation_name
    from evolvo.values import ValueEnumerations

    env: Dict[tuple, str] = {(int(Category.VARIABLE), int(DataType.DECIMAL), i): f"[{nm}]" for i, nm in enumerate(input_names)}

    def src(ins, k):
        cat = ins.source1_cat if k == 1 else ins.source2_cat
        typ = ins.source1_type if k == 1 else ins.source2_type
        val = ins.source1_value if k == 1 else ins.source2_value
        if cat == Category.NONE:
            return None
        if cat in (Category.VARIABLE, Category.CONSTANT):
            return env.get((int(cat), int(typ), int(val)), "0")
        if cat == Category.VALUE:
            enum = ValueEnumerations.get_enumeration((ins.operation, None))
            return f"{enum[val]:g}" if val < len(enum) else "0"
        return "?"

    for idx in genome.extract_effective_algorithm():
        ins = genome.instructions[idx]
        if ins.target_cat not in (Category.VARIABLE, Category.CONSTANT):
            continue
        name = resolve_operation_name(ins.operation)
        a, b = src(ins, 1), src(ins, 2)
        try:
            op = Operation(ins.operation)
        except ValueError:
            op = None
        if name in _INFIX and b is not None:
            expr = f"({a} {_INFIX[name]} {b})"
        elif op in (Operation.GTE, Operation.LTE, Operation.NEQ):
            continue  # these assign nothing (scalar semantics)
        else:
            expr = f"{name.lower()}({a}{'' if b is None else ', ' + b})"
        env[(int(ins.target_cat), int(ins.target_type), int(ins.target_index))] = expr
    return env.get((int(Category.VARIABLE), int(DataType.DECIMAL), output_index), "0")


def gfsl_code(inputs: Sequence[str], instructions: Sequence[str], output_index: int) -> List[str]:
    """GFSL listing of a selected program: input legend, the effective instructions in execution order, output.

    ``instructions`` are the ``✓`` lines of ``GFSLGenome.to_human_readable()`` (dropping the non-effective
    instructions does not change the output register)."""
    body = [line.lstrip().lstrip("✓").strip() for line in instructions]
    lines = [f"# d${i} = {name}" for i, name in enumerate(inputs)]
    lines += body
    lines.append(f"RESULT d${output_index}")
    return lines


# ---------------------------------------------------------------------------------------------
# Data preparation (parent process)
# ---------------------------------------------------------------------------------------------


def prepare(table: Dict[str, np.ndarray], fcfg: Dict[str, Any], base_sets: Dict[str, Any], seed: int) -> Dict[str, Any]:
    bases = features.union_bases(base_sets, fcfg["length_bases_sets"])
    names = list(fcfg["inputs"])
    X = features.build_matrix(table, names, bases)
    split = table["split"]
    tr, va, te = (np.nonzero(split == s)[0] for s in (0, 1, 2))
    std = features.Standardizer(X[tr])
    Xs = std(X)
    perm = tr[np.random.default_rng([seed, 20]).permutation(tr.size)]
    targets = {}
    for t in fcfg["targets"]:
        if t == "omega":
            y = table["label.omega"].astype(float)
        elif t == "is_prime":
            y = table["label.is_prime"].astype(float)
        else:
            raise ValueError(f"unknown formula target {t!r}")
        targets[t] = {
            "Xp": Xs[perm], "yp": y[perm],
            "Xv": Xs[va], "yv": y[va],
            "Xt": Xs[te], "yt": y[te],
            "var": float(np.var(y[tr])) or 1.0,
        }
    return {"names": names, "bases": bases, "standardizer": std.to_dict(), "targets": targets, "train_rows": int(tr.size)}


# ---------------------------------------------------------------------------------------------
# One run (worker process)
# ---------------------------------------------------------------------------------------------

_DATA: Dict[str, Any] = {}


def _init_worker(payload: Dict[str, Any]) -> None:
    _DATA.clear()
    _DATA.update(payload)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


def q_target(cfg: Dict[str, Any], target: str) -> float:
    qt = cfg["q_target"]
    return float(qt[target]) if isinstance(qt, dict) else float(qt)


def _make_guide(spec: Dict[str, Any], cfg: Dict[str, Any]):
    guide = spec["guide"]
    if guide == "none":
        return None
    from evolvo import GFSLPolicyGuide, GFSLQLearningGuide, GFSLSupervisedGuide

    if guide == "qlearning":
        return GFSLQLearningGuide(top_fraction=cfg["top_fraction"])
    if guide == "supervised":
        return GFSLSupervisedGuide(device="cpu")
    if guide == "policy":
        kw = dict(device="cpu", seed=spec["seed"], q_target=q_target(cfg, spec["target"]), top_fraction=cfg["top_fraction"])
        if spec.get("knowledge_in"):
            return GFSLPolicyGuide.load(spec["knowledge_in"], **kw)
        return GFSLPolicyGuide(**kw)
    raise ValueError(f"unknown guide {guide!r}")


def metrics(target: str, pred: np.ndarray, y: np.ndarray, var: float, max_class: int) -> Dict[str, float]:
    from sklearn.metrics import f1_score

    finite = np.isfinite(pred)
    p = np.where(finite, pred, 0.0)
    mse = float(np.mean((p - y) ** 2)) if finite.all() else float("inf")
    out = {"mse": mse, "nmse": mse / var, "nonfinite_rows": int((~finite).sum())}
    if target == "omega":
        cls = np.clip(np.rint(p), 1, max_class).astype(int)
        true = np.minimum(y, max_class).astype(int)
        out["accuracy"] = float((cls == true).mean())
        out["macro_f1"] = float(f1_score(true, cls, average="macro"))
    elif np.isin(y, (0, 1)).all():
        cls = (p > 0.5).astype(int)
        true = y.astype(int)
        out["accuracy"] = float((cls == true).mean())
        out["f1"] = float(f1_score(true, cls, zero_division=0))
    else:  # regression target (step 20 Kummer degrees): relative errors instead of class metrics
        out["max_relative_error"] = float(np.max(np.abs(p / y - 1))) if finite.all() else float("inf")
        out["median_relative_error"] = float(np.median(np.abs(p / y - 1))) if finite.all() else float("inf")
    return out


def run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    cfg = spec["cfg"]
    seed = int(spec["seed"])
    data = _DATA["targets"][spec["target"]]
    names = data.get("names", _DATA["names"])
    k = len(names)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
    runtime.bootstrap_evolvo()
    from evolvo import Category, DataType, GFSLEvolver, GFSLExecutor, InstructionProfile, Operation

    ex = GFSLExecutor(compute_backend="cpu")
    Xp, yp, var = data["Xp"], data["yp"], data["var"]
    cols = [np.ascontiguousarray(Xp[:, i]) for i in range(k)]
    qt = q_target(cfg, spec["target"])
    sizer = Sizer(cfg["eval_size_min"], cfg["eval_size_max"], cfg["eval_size_step"], qt, cfg["se_trigger"],
                  n_train=yp.size, fixed=cfg["fixed_size"] if spec["variant"] == "fixed" else None)
    lam = float(cfg["parsimony"])
    out_key = f"d${k}"

    def predict(genome, Xcols, rows=None):
        inputs = {f"d${i}": (c if rows is None else c[:rows]) for i, c in enumerate(Xcols)}
        return ex.execute_batch(genome, inputs, track_activity=False)[out_key]

    def errors(genome, N):
        with np.errstate(all="ignore"):
            pred = predict(genome, cols, N)
            return (pred - yp[:N]) ** 2 / var

    def fitness(genome):
        e = errors(genome, sizer.N)
        nmse = float(e.mean()) if np.isfinite(e).all() else math.inf
        size = len(genome.extract_effective_algorithm())
        genome.nmse = nmse  # type: ignore[attr-defined]
        genome.eval_size = sizer.N  # type: ignore[attr-defined]
        return (1.0 / (1.0 + nmse) if math.isfinite(nmse) else 0.0) - lam * size

    guide = _make_guide(spec, cfg)
    ev = GFSLEvolver(population_size=cfg["population"], supervised_guide=guide,
                     max_instruction_count=cfg["max_instructions"],
                     max_effective_instruction_count=cfg["max_effective_instructions"])
    profile = InstructionProfile.formula(allow_custom=False)
    if "operations" in data:
        profile = InstructionProfile.build(
            [Category.VARIABLE], [Operation[name] for name in data["operations"]], allow_custom=False,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        ev.initialize_population("algorithm", cfg["initial_instructions"], inputs=k, outputs=[(DataType.DECIMAL, k)], profile=profile)

    saved: Dict[str, Dict[str, Any]] = {}
    curve: List[Dict[str, Any]] = []
    reached: Dict[str, Any] = {"generation": None, "evaluations": None}

    def save(genome, gen, forced):
        sig = genome.get_signature()
        if sig in saved:
            return
        saved[sig] = {"genome": copy.deepcopy(genome), "generation": gen, "eval_size": getattr(genome, "eval_size", None),
                      "fitness": float(genome.fitness), "train_nmse": getattr(genome, "nmse", None), "forced": forced}

    def on_generation(gen, best, best_f):
        pop = [g for g in ev.population if g.fitness is not None and math.isfinite(float(g.fitness))]
        fits = sorted((float(g.fitness) for g in pop), reverse=True)
        kk = max(1, int(math.ceil(cfg["top_fraction"] * len(fits)))) if fits else 1
        q = float(np.mean(fits[:kk])) if fits else float("nan")
        spread = float(fits[0] - fits[kk - 1]) if fits else float("nan")
        e = errors(best, sizer.N)
        if np.isfinite(e).all() and e.size > 1:
            nm = float(e.mean())
            se = float(e.std(ddof=1) / math.sqrt(e.size)) / (1.0 + nm) ** 2
        else:
            se = float("nan")
        thr = cfg["save_threshold_ratio"] * q
        for g in ev.population[: cfg["save_top_per_gen"]]:
            if g.fitness is not None and float(g.fitness) >= thr:
                save(g, gen, False)
        if reached["generation"] is None and math.isfinite(best_f) and best_f >= qt:
            reached.update(generation=gen, evaluations=ev.evaluations)
        n_used = sizer.N
        n_next = sizer.update(q, spread, se)
        curve.append({"gen": gen, "N": n_used, "N_next": n_next, "q": q, "best": float(best_f), "spread": spread, "se": se,
                      "evaluations": ev.evaluations, "best_size": len(best.extract_effective_algorithm())})

    with contextlib.redirect_stdout(io.StringIO()):
        ev.evolve(cfg["generations"], fitness, progress_callback=on_generation, fitness_epoch=lambda: sizer.N)
    ranked = sorted((g for g in ev.population if g.fitness is not None), key=lambda g: -float(g.fitness))
    for g in ranked[: cfg["safety_net"]]:
        save(g, cfg["generations"] - 1, True)

    Xv_cols = [data["Xv"][:, i] for i in range(k)]
    cands = []
    for sig, rec in saved.items():
        with np.errstate(all="ignore"):
            pv = predict(rec["genome"], Xv_cols)
        mv = metrics(spec["target"], pv, data["yv"], var, cfg["max_omega_class"])
        cands.append((mv["nmse"] if math.isfinite(mv["nmse"]) else math.inf, len(rec["genome"].extract_effective_algorithm()), sig, mv))
    cands.sort(key=lambda c: (c[0], c[1]))
    best_sig = cands[0][2]
    chosen = saved[best_sig]
    g = chosen["genome"]
    Xt_cols = [data["Xt"][:, i] for i in range(k)]
    with np.errstate(all="ignore"):
        pt = predict(g, Xt_cols)
    test = metrics(spec["target"], pt, data["yt"], var, cfg["max_omega_class"])
    effective = g.extract_effective_algorithm()
    result = {
        "target": spec["target"], "guide": spec["guide"], "seed": seed, "variant": spec["variant"], "q_target": qt,
        "knowledge_in": spec.get("knowledge_in"),
        "evaluations": ev.evaluations, "generations": len(curve),
        "final_N": sizer.N, "N_changes": [(gen, int(e)) for gen, e in ev.fitness_epoch_changes],
        "reached_target": reached, "curve": curve,
        "saved": len(saved), "saved_forced": sum(1 for r in saved.values() if r["forced"]),
        "selected": {
            "signature": best_sig, "generation": chosen["generation"], "eval_size": chosen["eval_size"],
            "train_fitness": chosen["fitness"], "train_nmse": chosen["train_nmse"], "val": cands[0][3], "test": test,
            "effective_instructions": len(effective),
            "expression": formula_expression(g, names, k),
            "instructions": [line for line in g.to_human_readable() if line.lstrip().startswith("✓")],
        },
        "top_val": [{"signature": c[2], "val_nmse": c[0], "size": c[1]} for c in cands[:10]],
        "seconds": round(time.time() - t0, 2),
    }
    if spec.get("export_program"):
        result["selected"]["program"] = {
            "inputs": list(names),
            "slots": [list(ins.slots) for ins in g.instructions],
            "output_index": k,
        }
        rng = np.random.default_rng([seed, 133])
        result["selected"]["shuffled_test_mse"] = float(np.mean([
            np.mean((pt - rng.permutation(data["yt"])) ** 2) for _ in range(16)
        ]))
        if "audit" in data:
            rows, expected = data["audit"]
            values = predict(g, [rows[:, i] for i in range(k)])
            result["selected"]["audit"] = {
                "rows": len(expected), "exact": int(np.count_nonzero(values == expected)),
                "classification_correct": int(np.count_nonzero(np.isfinite(values) & ((values > 0.5) == expected))),
            }
    if guide is not None and hasattr(guide, "runtime_summary"):
        result["guide_summary"] = {k2: v for k2, v in guide.runtime_summary().items() if not isinstance(v, (list, dict))}
    if spec.get("knowledge_out") and spec["guide"] == "policy":
        guide.save(spec["knowledge_out"])
        result["knowledge_out"] = spec["knowledge_out"]
    return result


# ---------------------------------------------------------------------------------------------
# Experiment driver (parent process)
# ---------------------------------------------------------------------------------------------


def _specs(cfg: Dict[str, Any], knowledge_dir: Path) -> tuple[list, list]:
    phase_a, phase_b = [], []
    for t in cfg["targets"]:
        for gname in cfg["guides"]:
            for s in cfg["seeds"]:
                phase_a.append({"target": t, "guide": gname, "seed": s, "variant": "adaptive"})
        for s in cfg["seeds"]:
            phase_a.append({"target": t, "guide": cfg["ablation_guide"], "seed": s, "variant": "fixed"})
        if cfg["warm_start"] and "policy" in cfg["guides"]:
            kfile = str(knowledge_dir / f"policy_{t}.pt")
            phase_a.append({"target": t, "guide": "policy", "seed": cfg["knowledge_seed"], "variant": "knowledge", "knowledge_out": kfile})
            for s in cfg["seeds"]:
                phase_b.append({"target": t, "guide": "policy", "seed": s, "variant": "warm", "knowledge_in": kfile})
    for sp in phase_a + phase_b:
        sp["cfg"] = cfg
    return phase_a, phase_b


def _run_specs(specs: list, payload: dict, workers: int, log) -> list:
    if not specs:
        return []
    results = []
    if workers <= 1:
        _init_worker(payload)
        for i, sp in enumerate(specs):
            results.append(run_one(sp))
            log(f"    run {i + 1}/{len(specs)} {sp['target']}/{sp['guide']}/{sp['variant']}/s{sp['seed']} done")
        return results
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(payload,)) as pool:
        for i, res in enumerate(pool.map(run_one, specs)):
            results.append(res)
            if (i + 1) % max(1, len(specs) // 8) == 0 or i + 1 == len(specs):
                log(f"    {i + 1}/{len(specs)} runs done")
    return results


def _stats(values: List[float]) -> Dict[str, Any]:
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}


def _mw_p(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided Mann-Whitney U p-value (None when a group has < 2 finite values)."""
    from scipy.stats import mannwhitneyu

    a = [x for x in a if x is not None and math.isfinite(x)]
    b = [x for x in b if x is not None and math.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _test_nmse(runs, t, g, var):
    return [r["selected"]["test"]["nmse"] for r in runs if r["target"] == t and r["guide"] == g and r["variant"] == var]


def summarize(runs: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    groups: Dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["target"], r["guide"], r["variant"]), []).append(r)
    table = []
    for (t, gname, var), rs in sorted(groups.items()):
        table.append({
            "target": t, "guide": gname, "variant": var, "runs": len(rs),
            "test_nmse": _stats([r["selected"]["test"]["nmse"] for r in rs]),
            "val_nmse": _stats([r["selected"]["val"]["nmse"] for r in rs]),
            "test_accuracy": _stats([r["selected"]["test"]["accuracy"] for r in rs]),
            "reached_target": sum(1 for r in rs if r["reached_target"]["generation"] is not None),
            "evaluations_to_target": _stats([r["reached_target"]["evaluations"] for r in rs]),
            "evaluations": _stats([r["evaluations"] for r in rs]),
            "final_N": _stats([r["final_N"] for r in rs]),
            "seconds": _stats([r["seconds"] for r in rs]),
        })

    def row(t, gname, var):
        return next((x for x in table if x["target"] == t and x["guide"] == gname and x["variant"] == var), None)

    verdicts = {}
    for t in cfg["targets"]:
        v: Dict[str, Any] = {}
        a, f = row(t, cfg["ablation_guide"], "adaptive"), row(t, cfg["ablation_guide"], "fixed")
        if a and f:
            n_rose = sum(1 for r in runs if r["target"] == t and r["variant"] == "adaptive" and r["curve"] and r["final_N"] > r["curve"][0]["N"])
            v["S1"] = {
                "adaptive_test_nmse": a["test_nmse"], "fixed_test_nmse": f["test_nmse"],
                "adaptive_final_N": a["final_N"], "runs_where_N_rose": n_rose,
                "p_value_test_nmse": _mw_p(_test_nmse(runs, t, cfg["ablation_guide"], "adaptive"), _test_nmse(runs, t, cfg["ablation_guide"], "fixed")),
                "done": bool(a["test_nmse"]["mean"] is not None and f["test_nmse"]["mean"] is not None and a["test_nmse"]["mean"] <= f["test_nmse"]["mean"] and n_rose > 0),
            }
        p, n0 = row(t, "policy", "adaptive"), row(t, "none", "adaptive")
        if p and n0:
            fewer = (p["evaluations_to_target"]["mean"] is not None and n0["evaluations_to_target"]["mean"] is not None
                     and p["reached_target"] >= n0["reached_target"] and p["evaluations_to_target"]["mean"] < n0["evaluations_to_target"]["mean"]) or (p["reached_target"] > n0["reached_target"])
            smaller = p["test_nmse"]["std"] is not None and n0["test_nmse"]["std"] is not None and p["test_nmse"]["std"] < n0["test_nmse"]["std"]
            w = row(t, "policy", "warm")
            warm_better = warm_lower = warm_fewer = None
            if w:
                warm_lower = bool(w["test_nmse"]["mean"] is not None and p["test_nmse"]["mean"] is not None and w["test_nmse"]["mean"] < p["test_nmse"]["mean"])
                warm_fewer = bool((w["reached_target"] > p["reached_target"]) or (
                    w["reached_target"] >= p["reached_target"] and w["evaluations_to_target"]["mean"] is not None
                    and p["evaluations_to_target"]["mean"] is not None and w["evaluations_to_target"]["mean"] < p["evaluations_to_target"]["mean"]))
                warm_better = warm_lower or warm_fewer
            v["S2"] = {"policy_vs_none_fewer_evaluations_to_target": bool(fewer), "policy_vs_none_smaller_spread": bool(smaller),
                       "warm_lower_test_nmse": warm_lower, "warm_fewer_evaluations_to_target": warm_fewer,
                       "warm_better_than_cold": warm_better, "done": bool((fewer or smaller) and warm_better),
                       "p_value_policy_vs_none_test_nmse": _mw_p(_test_nmse(runs, t, "policy", "adaptive"), _test_nmse(runs, t, "none", "adaptive")),
                       "p_value_warm_vs_cold_test_nmse": _mw_p(_test_nmse(runs, t, "policy", "warm"), _test_nmse(runs, t, "policy", "adaptive"))}
        verdicts[t] = v
    best = {}
    for t in cfg["targets"]:
        rs = [r for r in runs if r["target"] == t]
        if rs:
            b = min(rs, key=lambda r: (r["selected"]["val"]["nmse"], r["selected"]["effective_instructions"]))
            best[t] = {"guide": b["guide"], "variant": b["variant"], "seed": b["seed"], **{k: b["selected"][k] for k in ("expression", "instructions", "val", "test", "effective_instructions", "eval_size")}}
    return {"table": table, "verdicts": verdicts, "best_by_validation": best}


def run_stage(table: Dict[str, np.ndarray], fcfg_in: Dict[str, Any], base_sets: Dict[str, Any], seed: int, run_dir: Path, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(fcfg_in or {})}
    prep = prepare(table, cfg, base_sets, seed)
    payload = {"names": prep["names"], "targets": prep["targets"]}
    knowledge_dir = run_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    workers = int(cfg["workers"]) or max(1, (os.cpu_count() or 2) - 1)
    phase_a, phase_b = _specs(cfg, knowledge_dir)
    log(f"  formulas: {len(phase_a)} runs, then {len(phase_b)} warm-start runs; {workers} workers; train rows {prep['train_rows']}")
    runs = _run_specs(phase_a, payload, workers, log)
    runs += _run_specs(phase_b, payload, workers, log)
    summary = summarize(runs, cfg)
    write_best_formulas(run_dir / "best_formulas.md", summary, prep["names"])
    return {
        "settings": cfg,
        "inputs": prep["names"],
        "length_bases": prep["bases"],
        "standardizer": prep["standardizer"],
        "train_rows": prep["train_rows"],
        "summary": summary,
        "runs": runs,
    }


def write_best_formulas(path: Path, summary: Dict[str, Any], names: Sequence[str]) -> None:
    lines = ["# Best formulas (selected on validation)", "", "Inputs are z-scored with train statistics:", ""]
    lines += [f"- `d${i}` = {nm}" for i, nm in enumerate(names)]
    for t, b in summary["best_by_validation"].items():
        lines += ["", f"## {t}", "",
                  f"- guide `{b['guide']}`, variant `{b['variant']}`, seed {b['seed']}, {b['effective_instructions']} effective instructions, selected at N = {b['eval_size']}",
                  f"- validation NMSE {b['val']['nmse']:.4f}, test NMSE {b['test']['nmse']:.4f}, test accuracy {b['test']['accuracy']:.4f}",
                  "", "```", b["expression"], "```", "", "Instructions:", "", "```", *b["instructions"], "```"]
    path.write_text("\n".join(lines) + "\n")
