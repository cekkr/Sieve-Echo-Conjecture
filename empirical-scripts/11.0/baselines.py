"""Held-out baselines for the step-4 results table and the §3.1 three-feature retest.

All models fit on the train split (gradient boosting uses its own internal early-stopping split of
train), and every number is measured once on the test split. Feature sets:

* ``log n + mean log L`` — the trivial covariates (AGENTS.md principle);
* ``n + L arithmetic`` — log n, log L per base (v8 ∪ v10 ∪ primes31), coprime counts and the length
  arithmetic of `features.length_arithmetic` (the C9 baseline);
* ``formula inputs`` — the inputs the GFSL formulas get;
* ``all allowed`` — the L-arithmetic set plus every allowed NDR aggregate (non-wide sets).

Targets: ``omega`` as classes ``min(omega, K)`` and ``is_prime``. The §3.1 retest uses the v8-set
`H_paper` std and min across bases, with and without log n, and — explicitly labelled as a
leakage demonstration — with ``is_prime_power`` added, scored on train (what v7 reported) and on
test.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

import dataset
import features

DEFAULTS: Dict[str, Any] = {
    "max_omega_class": 5,
    "max_train_rows": None,
    "gbm": {"max_iter": 300, "learning_rate": 0.1},
    "sets": ["v8", "v10", "primes31"],
}


def target_vector(table, target: str, K: int) -> np.ndarray:
    if target == "omega":
        return np.minimum(table["label.omega"], K).astype(int)
    if target == "is_prime":
        return table["label.is_prime"].astype(int)
    raise ValueError(target)


def score(target: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None, classes=None) -> Dict[str, float]:
    from sklearn.metrics import f1_score, log_loss

    out = {"accuracy": float((pred == y).mean()), "mse_class": float(np.mean((pred.astype(float) - y) ** 2))}
    if target == "omega":
        out["macro_f1"] = float(f1_score(y, pred, average="macro"))
    else:
        out["f1"] = float(f1_score(y, pred, zero_division=0))
    if proba is not None and classes is not None:
        out["log_loss"] = float(log_loss(y, np.clip(proba, 1e-12, 1), labels=list(classes)))
    return out


def feature_sets(table, base_sets, formula_inputs: Sequence[str], sets: Sequence[str]) -> Dict[str, tuple]:
    bases = features.union_bases(base_sets, sets)
    logn = np.log(table["row.n"].astype(float))
    Lmat = features.per_base(table, bases, "logL")
    nb = np.column_stack([table[f"row.n_bases.{s}"] for s in sets if f"row.n_bases.{s}" in table]).astype(float)
    arith, arith_names = features.length_arithmetic(table, bases)
    base = np.column_stack([logn, Lmat, nb, arith])
    allowed = [c for c in dataset.allowed_input_columns(table) if not c.startswith("row.wide.") and c != "row.n_bases.wide" and c != "row.n"]
    ndr = [c for c in allowed if not c.startswith("row.n_bases.")]
    out = {
        "log n + mean log L": (np.column_stack([logn, table["row.v8.logL.mean"]]), ["log n", "row.v8.logL.mean"]),
        "n + L arithmetic": (base, ["log n", *[f"logL.b{b}" for b in bases], *[f"n_bases.{s}" for s in sets], *arith_names]),
        "formula inputs": (features.build_matrix(table, formula_inputs, bases), list(formula_inputs)),
        "all allowed": (np.column_stack([base] + [table[c] for c in ndr]), ["<n + L arithmetic>", *ndr]),
    }
    return out


def _gbm(cfg, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=int(cfg["gbm"]["max_iter"]), learning_rate=float(cfg["gbm"]["learning_rate"]),
                                          early_stopping=True, validation_fraction=0.15, random_state=seed)


def run_baselines(table, cfg_in: Dict[str, Any], base_sets, formula_inputs: Sequence[str], seed: int, log=print) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = {**DEFAULTS, **(cfg_in or {})}
    K = int(cfg["max_omega_class"])
    split = table["split"]
    tr = np.nonzero(split == 0)[0]
    if cfg["max_train_rows"] and tr.size > int(cfg["max_train_rows"]):
        tr = np.sort(np.random.default_rng([seed, 30]).choice(tr, int(cfg["max_train_rows"]), replace=False))
    te = np.nonzero(split == 2)[0]
    fsets = feature_sets(table, base_sets, formula_inputs, cfg["sets"])
    results: Dict[str, Any] = {"train_rows": int(tr.size), "test_rows": int(te.size), "feature_counts": {k: v[0].shape[1] for k, v in fsets.items()}, "targets": {}}
    for target in ("omega", "is_prime"):
        y = target_vector(table, target, K)
        rows: Dict[str, Any] = {}
        classes = np.unique(y[tr])
        maj = classes[np.argmax([(y[tr] == c).sum() for c in classes])]
        prior = np.array([(y[tr] == c).mean() for c in classes])
        rows["majority"] = score(target, y[te], np.full(te.size, maj), np.tile(prior, (te.size, 1)), classes)
        X0 = fsets["log n + mean log L"][0]
        Xi = np.column_stack([np.nan_to_num(X0, nan=0.0), np.isnan(X0).any(axis=1)])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xi[tr], y[tr])
        rows["logistic: log n + mean log L"] = score(target, y[te], lr.predict(Xi[te]), lr.predict_proba(Xi[te]), lr.classes_)
        for name, (X, _) in fsets.items():
            t0 = time.time()
            m = _gbm(cfg, seed).fit(X[tr], y[tr])
            rows[f"GBM: {name}"] = {**score(target, y[te], m.predict(X[te]), m.predict_proba(X[te]), m.classes_), "seconds": round(time.time() - t0, 1)}
            log(f"    baseline {target} GBM {name}: acc {rows[f'GBM: {name}']['accuracy']:.4f} ({time.time() - t0:.0f}s)")
        results["targets"][target] = rows
    results["three_feature_retest"] = three_feature(table, cfg, tr, te, seed)
    return results


def three_feature(table, cfg, tr, te, seed) -> Dict[str, Any]:
    """Paper §3.1: 'std, is_prime_power, entropy_min' -> 98.8% (v7.0, training-set accuracy)."""
    K = int(cfg["max_omega_class"])
    y = target_vector(table, "omega", K)
    std_h, min_h = table["row.v8.H_paper.std"], table["row.v8.H_paper.min"]
    logn = np.log(table["row.n"].astype(float))
    ipp = table["label.is_prime_power"].astype(float)
    variants = {
        "std H, min H": np.column_stack([std_h, min_h]),
        "std H, min H, log n": np.column_stack([std_h, min_h, logn]),
        "LEAKAGE DEMO: std H, min H, is_prime_power": np.column_stack([std_h, min_h, ipp]),
    }
    out: Dict[str, Any] = {"entropy": "v8 set, H_paper (ndr-v11.1)", "paper_claim": 0.988, "variants": {}}
    for name, X in variants.items():
        m = _gbm(cfg, seed).fit(X[tr], y[tr])
        out["variants"][name] = {
            "train_accuracy": float((m.predict(X[tr]) == y[tr]).mean()),
            "test": score("omega", y[te], m.predict(X[te])),
        }
    maj = np.bincount(y[tr]).argmax()
    out["majority_test_accuracy"] = float((y[te] == maj).mean())
    out["note"] = "is_prime_power alone identifies omega = 1; it is a factorization label and is forbidden as an input outside this demonstration."
    return out
