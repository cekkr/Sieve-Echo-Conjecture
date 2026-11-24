"""Evolvo GFSL-NN architecture search for omega(n) classes (ROADMAP step 4).

Two input encodings, each searched with ``evolvo.nnarch.ArchitectureSearch`` and scored with
``TorchTask.classification`` (cross-entropy on the logits of the fixed head):

* ``vec``: the "all allowed" baseline features (n, per-base log L, coprime counts, length
  arithmetic, every allowed NDR aggregate), z-scored with train statistics;
* ``set``: one token per base of v8 ∪ v10 ∪ primes31 — [coprime flag, log b, per-base L / entropy /
  moments / flatness / peaks], zero for non-coprime bases, z-scored per feature over train — plus a
  vector of log n and the length arithmetic. Raw digit sequences are not used: their length L
  varies from 1 to ~10^6 and Evolvo's set inputs are fixed-size (padded sets are an Evolvo
  roadmap item).

Training uses a stratified train subsample (``max_train_rows``) and the validation split for
stopping and selection; the test split is read once for the best module. The learned search
knowledge is saved to ``<run>/knowledge/nn_<input>.json`` (and loaded from ``knowledge_in`` when
given), so later runs reuse it.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

import baselines
import features
import runtime

DEFAULTS: Dict[str, Any] = {
    "inputs": ["vec", "set"],
    "max_omega_class": 5,
    "max_train_rows": 30000,
    "max_val_rows": 10000,
    "population": 8,
    "generations": 4,
    "patience": 3,
    "max_rows": 4,
    "max_params": 60000,
    "lr": 3e-3,
    "batch_size": 512,
    "hard_max_epochs": 200,
    "sets": ["v8", "v10", "primes31"],
    "knowledge_in": {},
    "torch_threads": 4,
}


def _subsample(idx: np.ndarray, y: np.ndarray, cap: int, rng) -> np.ndarray:
    if not cap or idx.size <= cap:
        return idx
    keep = []
    for c in np.unique(y[idx]):
        part = idx[y[idx] == c]
        k = max(1, int(round(cap * part.size / idx.size)))
        keep.append(rng.choice(part, min(k, part.size), replace=False))
    return np.sort(np.concatenate(keep))


def run_stage(table, ncfg_in: Dict[str, Any], base_sets, formula_inputs, seed: int, run_dir: Path, log=print) -> Dict[str, Any]:
    runtime.bootstrap_evolvo()
    import torch
    from evolvo.nnarch import ArchitectureSearch, ArchSpace, Knowledge
    from evolvo.nnarch.train import TorchTask

    cfg = {**DEFAULTS, **(ncfg_in or {})}
    torch.set_num_threads(int(cfg["torch_threads"]))
    K = int(cfg["max_omega_class"])
    y_all = baselines.target_vector(table, "omega", K) - 1  # class indices 0..K-1
    split = table["split"]
    rng = np.random.default_rng([seed, 40])
    tr = _subsample(np.nonzero(split == 0)[0], y_all, int(cfg["max_train_rows"] or 0), rng)
    va = _subsample(np.nonzero(split == 1)[0], y_all, int(cfg["max_val_rows"] or 0), rng)
    te = np.nonzero(split == 2)[0]
    knowledge_dir = run_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    fsets = baselines.feature_sets(table, base_sets, formula_inputs, cfg["sets"])
    encodings: Dict[str, Any] = {}
    if "vec" in cfg["inputs"]:
        X = fsets["all allowed"][0]
        st = features.Standardizer(X[tr])
        encodings["vec"] = ([("vec", X.shape[1])], [st(X).astype(np.float32)])
    if "set" in cfg["inputs"]:
        bases = features.union_bases(base_sets, cfg["sets"])
        T = features.token_tensor(table, bases)  # rows x B x F
        mask = T[..., 0] > 0
        flat = T[tr][mask[tr]]
        mu, sd = flat.mean(axis=0), flat.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        mu[0], sd[0] = 0.0, 1.0  # keep the coprime flag as 0/1
        Tz = np.where(mask[..., None], (T - mu) / sd, 0.0).astype(np.float32)
        Tz[..., 0] = mask
        arith, _ = features.length_arithmetic(table, bases)
        V = np.column_stack([np.log(table["row.n"].astype(float)), arith])
        sv = features.Standardizer(V[tr])
        encodings["set"] = ([("set", T.shape[-1]), ("vec", V.shape[1])], [Tz, sv(V).astype(np.float32)])

    out: Dict[str, Any] = {"settings": {k: v for k, v in cfg.items()}, "rows": {"train": int(tr.size), "val": int(va.size), "test": int(te.size)}, "encodings": {}}
    for name, (inputs, arrays) in encodings.items():
        t0 = time.time()

        def pick(idx):
            return tuple(torch.from_numpy(a[idx]) for a in arrays)

        y_t = torch.from_numpy(y_all.astype(np.int64))
        task = TorchTask.classification((pick(tr), y_t[tr]), (pick(va), y_t[va]), lr=float(cfg["lr"]),
                                        batch_size=int(cfg["batch_size"]), hard_max_epochs=int(cfg["hard_max_epochs"]))
        space = ArchSpace(inputs, K, max_rows=int(cfg["max_rows"]), max_params=int(cfg["max_params"]), task={"loss": "ce", "input": name})
        kin = (cfg.get("knowledge_in") or {}).get(name)
        knowledge = Knowledge.load(kin) if kin and Path(kin).is_file() else None
        search = ArchitectureSearch(space, task, seed=seed, population=int(cfg["population"]), knowledge=knowledge)
        report = search.run(generations=int(cfg["generations"]), patience=int(cfg["patience"]))
        kout = knowledge_dir / f"nn_{name}.json"
        search.knowledge.save(str(kout))
        module = task.best_module
        with torch.no_grad():
            logits = module(*pick(te))
            proba = torch.softmax(logits, dim=-1).numpy()
        pred = proba.argmax(axis=1) + 1
        y_te = y_all[te] + 1
        test = baselines.score("omega", y_te, pred, proba, np.arange(1, K + 1))
        best = report.get("best") or {}
        out["encodings"][name] = {
            "inputs": inputs,
            "best": {"loss_val": best.get("loss"), "params": best.get("params"), "describe": best.get("describe"), "training": best.get("training")},
            "evaluated": len(search.history),
            "stop_reason": report.get("stop_reason"),
            "test": test,
            "knowledge_in": kin,
            "knowledge_out": str(kout.relative_to(run_dir)),
            "seconds": round(time.time() - t0, 1),
        }
        log(f"  nn[{name}]: {len(search.history)} architectures, val loss {best.get('loss', math.nan):.4f}, test acc {test['accuracy']:.4f} ({time.time() - t0:.0f}s)")
    return out
