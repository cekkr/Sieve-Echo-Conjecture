"""Step 9b: order-only classification of ω beyond 10⁶ (ROADMAP step 9).

Steps 7–8 showed that, up to 10⁶, everything NDR spectra say about ω is already in the orders
ord_n(b) over many bases (plus Midy flags). Orders need no digits, so they scale far beyond the
NDR dataset. This stage samples integers per class c = min(ω, K) in each decade up to 10⁸, computes
the orders and Midy flags in bases 2..199 exactly from λ(n), and measures held-out gradient-boosting
accuracy per decade for nested base sets (19 v8/v10/primes31 bases, 2..31, 2..99, 2..199), each
with log n, per-base log L, coprime count, lcm arithmetic and Midy flags.

Factorization (labels and λ(n) only — never predictor inputs) is trial division by the primes below
√(max n) — exact for every sampled n; ω is computed vectorized over candidate batches, full
factorizations only for accepted integers.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence

import numpy as np

import dataset
import features
import lengths
import ndr
import scale

DEFAULTS: Dict[str, Any] = {
    "decades": [[10000, 99999], [100000, 999999], [1000000, 9999999], [10000000, 99999999]],
    "per_class": 3000,
    "max_class": 5,
    "batch": 200000,
    "max_batches": 60,
    "bases": "2..199",
    "base_sets": {"L19": [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31], "L2..31": "2..31", "L2..99": "2..99", "L2..199": "2..199"},
    "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    "model_seeds": 3,
    "bootstrap": 500,
    "gbm": {"max_iter": 200, "learning_rate": 0.1},
    "workers": 0,
}


def _primes_upto(m: int) -> np.ndarray:
    spf = ndr.spf_sieve(m)
    return np.array([p for p in range(2, m + 1) if spf[p] == p], dtype=np.int64)


def omega_vectorized(nv: np.ndarray, primes: np.ndarray) -> np.ndarray:
    """ω(n) for n < (max prime)² + ..: count prime divisors ≤ max prime; a cofactor > 1 left is one prime."""
    rem = nv.astype(np.int64).copy()
    w = np.zeros(nv.size, dtype=np.int64)
    for p in primes.tolist():
        hit = rem % p == 0
        if not hit.any():
            continue
        w += hit
        while hit.any():
            rem[hit] //= p
            hit = (rem % p == 0) & hit
    return w + (rem > 1)


def factorize_td(n: int, primes: Sequence[int]) -> list[tuple[int, int]]:
    out = []
    m = n
    for p in primes:
        if p * p > m:
            break
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            out.append((p, e))
    if m > 1:
        out.append((m, 1))
    return out


def sample(decades, per_class: int, K: int, batch: int, max_batches: int, primes: np.ndarray, seed: int, log=print):
    rows_n, rows_w, scheme = [], [], []
    for d, (lo, hi) in enumerate(decades):
        rng = np.random.default_rng([seed, 70, d])
        taken = {c: [] for c in range(1, K + 1)}
        draws = 0
        if hi - lo + 1 <= 2 * batch:  # small decade: enumerate everything once
            cand = rng.permutation(np.arange(lo, hi + 1, dtype=np.int64))
            batches = [cand]
        else:
            batches = None
        seen: set = set()
        for b in range(max_batches if batches is None else 1):
            cand = batches[0] if batches is not None else np.unique(rng.integers(lo, hi + 1, size=batch, dtype=np.int64))
            if batches is None:
                cand = rng.permutation(np.array([x for x in cand.tolist() if x not in seen], dtype=np.int64))
                seen.update(cand.tolist())
            draws += cand.size
            cls = np.minimum(omega_vectorized(cand, primes), K)
            for c in range(1, K + 1):
                need = per_class - len(taken[c])
                if need > 0:
                    taken[c].extend(cand[cls == c][:need].tolist())
            if all(len(v) >= per_class for v in taken.values()):
                break
        for c, v in taken.items():
            rows_n.extend(v)
            rows_w.extend([c] * len(v))
        scheme.append({"decade": [int(lo), int(hi)], "draws": int(draws), "taken": {int(c): len(v) for c, v in taken.items()}})
        log(f"    decade [{lo}, {hi}]: {scheme[-1]['taken']} from {draws} draws")
    n = np.array(rows_n, dtype=np.int64)
    order = np.argsort(n)
    return n[order], np.array(rows_w, dtype=np.int64)[order], scheme


_P: Dict[str, Any] = {}


def _init(primes):
    _P["primes"] = [int(p) for p in primes]  # Python ints: pow() rejects NumPy exponents


def _orders_chunk(args):
    ns, bases = args
    primes = _P["primes"]
    L = np.zeros((len(ns), len(bases)), dtype=np.int64)
    M = np.zeros((len(ns), len(bases)), dtype=np.int8)
    omega = np.zeros(len(ns), dtype=np.int64)
    lam_out = np.zeros(len(ns), dtype=np.int64)
    for i, n in enumerate(ns):
        f = factorize_td(n, primes)
        omega[i] = len(f)
        lam = ndr.carmichael_from_factors(f)
        lam_out[i] = lam
        lp = [p for p, _ in factorize_td(lam, primes)]
        for j, b in enumerate(bases):
            if math.gcd(b, n) != 1:
                continue
            t = ndr.multiplicative_order(b, n, lam, lp)
            L[i, j] = t
            M[i, j] = lengths.midy_flag(n, b, t)
    return L, M, omega, lam_out


def orders(n: np.ndarray, bases: Sequence[int], primes: np.ndarray, workers: int):
    ns = n.tolist()
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    size = max(100, len(ns) // (workers * 8) + 1)
    tasks = [(ns[i:i + size], list(bases)) for i in range(0, len(ns), size)]
    if workers <= 1 or len(tasks) <= 1:
        _init(primes)
        parts = [_orders_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(primes,)) as pool:
            parts = list(pool.map(_orders_chunk, tasks))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


def run_stage(ocfg_in: Dict[str, Any], seed: int, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(ocfg_in or {})}
    t0 = time.time()
    K = int(cfg["max_class"])
    hi = max(h for _, h in cfg["decades"])
    primes = _primes_upto(int(math.isqrt(hi)) + 1)
    log(f"  9b: sampling up to {per_str(cfg)} per class min(ω, {K}) in {len(cfg['decades'])} decades")
    n, cls, scheme = sample(cfg["decades"], int(cfg["per_class"]), K, int(cfg["batch"]), int(cfg["max_batches"]), primes, seed, log)
    bases = dataset.parse_bases(cfg["bases"])
    t1 = time.time()
    L, M, omega, lam = orders(n, bases, primes, int(cfg["workers"]))
    log(f"  9b: orders of {n.size} integers in {len(bases)} bases in {time.time() - t1:.1f}s")
    assert np.array_equal(np.minimum(omega, K), cls), "vectorized ω disagrees with trial-division factorization"
    split = dataset.split_by_n(n, cls, cfg["split"], seed)
    y = cls
    sets = {name: dataset.parse_bases(spec) if not isinstance(spec, list) else list(map(int, spec)) for name, spec in cfg["base_sets"].items()}
    blocks = {}
    for name, bs in sets.items():
        j = [bases.index(b) for b in bs]
        Xl, _ = lengths.length_block(n, L[:, j])
        Xm, _ = lengths.midy_block(L[:, j], M[:, j])
        blocks[name] = np.column_stack([Xl, Xm])
    lcm19 = np.lcm.reduce(np.where(L[:, [bases.index(b) for b in sets["L19"]]] > 0, L[:, [bases.index(b) for b in sets["L19"]]], 1), axis=1) if "L19" in sets else None
    out_dec: List[Dict[str, Any]] = []
    names = list(blocks)
    for d, (lo, hi_) in enumerate(cfg["decades"]):
        rows = np.nonzero((n >= lo) & (n <= hi_))[0]
        tr, te = rows[split[rows] == 0], rows[split[rows] == 2]
        s = seed + 100 * d
        rec: Dict[str, Any] = {"decade": [int(lo), int(hi_)], "rows": int(rows.size), "train": int(tr.size), "test": int(te.size),
                               "class_counts": {int(c): int((cls[rows] == c).sum()) for c in np.unique(cls[rows])}}
        if lcm19 is not None:
            rec["lcm19_equals_lambda"] = float((lcm19[rows] == lam[rows]).mean())
        if tr.size < 50 or te.size < 30:
            rec["skipped"] = "too few rows"
            out_dec.append(rec)
            continue
        corr = {nm: lengths._correct({**cfg, "model_seeds": cfg["model_seeds"]}, s, blocks[nm], y, tr, te) for nm in names}
        rec["accuracy"] = {nm: float(v.mean()) for nm, v in corr.items()}
        rec["gain"] = {f"{names[k + 1]} vs {names[k]}": lengths._delta(corr[names[k + 1]], corr[names[k]], int(cfg["bootstrap"]), s + 7 + k) for k in range(len(names) - 1)}
        maj = np.bincount(y[tr]).argmax()
        rec["majority"] = float((y[te] == maj).mean())
        log(f"    decade [{lo}, {hi_}]: " + ", ".join(f"{nm} {rec['accuracy'][nm]:.4f}" for nm in names))
        out_dec.append(rec)
    return {"settings": cfg, "sampling": scheme, "integers": int(n.size), "length_bases": bases, "decades": out_dec, "seconds": round(time.time() - t0, 1)}


def per_str(cfg) -> str:
    return str(cfg["per_class"])
