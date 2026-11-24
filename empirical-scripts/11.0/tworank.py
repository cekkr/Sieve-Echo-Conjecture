"""Step 10: the 2-rank of (ℤ/nℤ)* — what the multiplicative orders know about ω (ROADMAP step 10).

Let G = (ℤ/nℤ)* with Sylow-2 subgroup ∏_{i ≤ r} C_{2^{e_i}} (r = rank₂ G, every e_i ≥ 1). The 2-part of a
uniform b ∈ G is uniform in the Sylow subgroup, and v₂(ord b) is the largest v₂ of its components, so

    P(v₂(ord b) ≤ k) = ∏_i 2^{min(e_i, k) − e_i} = 2^{−Σ_i max(e_i − k, 0)},   P(v₂ ≤ 1) / P(v₂ = 0) = 2^r.

With n = 2^a·m (m odd), (ℤ/p^k)* is cyclic of even order for odd p, (ℤ/4)* ≅ C₂ and (ℤ/2^a)* ≅ C₂ × C_{2^{a−2}}
for a ≥ 3, so r = ω(m) + δ(a) with δ(0) = δ(1) = 0, δ(2) = 1, δ(a ≥ 3) = 2, and

    ω(n) = rank₂ G − δ(a) + [a ≥ 1].

ω is therefore a property of the group that the orders ord_n(b) sample. For an odd prime ℓ the ℓ-rank of G
is the number of primes p | n with p ≡ 1 (mod ℓ) (plus [ℓ² | n]); its profile adds information about the
prime factors without counting them.

This stage reuses the integers of `orders_scale` (same decades, sampling and seed), computes the orders in
bases 2..B, and compares (a) the ratio estimator ω̂ = round(log₂(#{v₂ ≤ 1} / #{v₂ = 0})) − δ(a) + [a ≥ 1]
for several base sets, and (b) gradient boosting on the raw-length features of step 9b against the
ℓ-adic profiles (histograms of v₂ and v₃ of the orders, with log n and v₂(n)) and both combined.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence

import numpy as np

import dataset
import lengths
import orders_scale

DEFAULTS: Dict[str, Any] = {
    "bases": "2..499",
    "estimator_sets": {"19 bases": [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31], "2..99": "2..99", "2..199": "2..199",
                       "2..499": "2..499", "primes < 500": "primes"},
    "length_set": "2..199",
    "profile_sets": ["2..199", "2..499"],
    "v2_bins": 14,
    "v3_bins": 8,
    "model_seeds": 3,
    "bootstrap": 500,
    "gbm": {"max_iter": 200, "learning_rate": 0.1},
    "workers": 0,
}


def valuation(L: np.ndarray, ell: int) -> np.ndarray:
    """ℓ-adic valuation of each order (−1 where the base is not coprime, L = 0)."""
    V = np.zeros(L.shape, dtype=np.int64)
    X = np.where(L > 0, L, 1).astype(np.int64)
    while True:
        hit = (X % ell == 0) & (L > 0)
        if not hit.any():
            break
        V += hit
        X = np.where(hit, X // ell, X)
    return np.where(L > 0, V, -1)


def v2_int(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    return np.rint(np.log2((a & -a).astype(float))).astype(np.int64)


def delta(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return np.where(a >= 3, 2, np.where(a == 2, 1, 0))


def omega_from_rank(rank: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = v2_int(n)
    return rank - delta(a) + (a >= 1)


def sylow2_exponents(factors) -> List[int]:
    """e_i of the Sylow-2 subgroup of (ℤ/nℤ)* from the factorization of n."""
    out = []
    for p, k in factors:
        if p == 2:
            if k == 2:
                out.append(1)
            elif k >= 3:
                out += [1, k - 2]
        else:
            out.append(((p - 1) & -(p - 1)).bit_length() - 1)
    return out


def histogram(V: np.ndarray, bins: int) -> np.ndarray:
    """Fractions of coprime bases with valuation 0, 1, …, bins−2, and ≥ bins−1; plus the coprime count."""
    has = V >= 0
    cnt = has.sum(axis=1)
    cols = [((V == k) & has).sum(axis=1) for k in range(bins - 1)] + [(V >= bins - 1).sum(axis=1)]
    H = np.column_stack(cols) / np.maximum(cnt, 1)[:, None]
    return np.column_stack([H, cnt])


def ratio_estimate(V2: np.ndarray, n: np.ndarray) -> Dict[str, np.ndarray]:
    """rank₂ ≈ round(log₂(#{v₂ ≤ 1} / #{v₂ = 0})); with no odd-order base the denominator is set to 1/2."""
    has = V2 >= 0
    c0 = ((V2 == 0) & has).sum(axis=1)
    c1 = ((V2 <= 1) & has).sum(axis=1)
    rank = np.rint(np.log2(np.maximum(c1, 1) / np.maximum(c0, 0.5)))
    return {"rank": rank, "omega": omega_from_rank(rank, n), "no_odd_order": c0 == 0}


def _set(spec, bases: Sequence[int]) -> List[int]:
    if spec == "primes":
        return [b for b in bases if all(b % p for p in range(2, int(math.isqrt(b)) + 1))]
    return dataset.parse_bases(spec) if not isinstance(spec, list) else list(map(int, spec))


def run_stage(ocfg: Dict[str, Any], tcfg_in: Dict[str, Any], seed: int, log=print) -> Dict[str, Any]:
    ocfg = {**orders_scale.DEFAULTS, **(ocfg or {})}
    cfg = {**DEFAULTS, **(tcfg_in or {})}
    t0 = time.time()
    K = int(ocfg["max_class"])
    hi = max(h for _, h in ocfg["decades"])
    primes = orders_scale._primes_upto(int(math.isqrt(hi)) + 1)
    n, cls, scheme = orders_scale.sample(ocfg["decades"], int(ocfg["per_class"]), K, int(ocfg["batch"]), int(ocfg["max_batches"]), primes, seed, log)
    bases = dataset.parse_bases(cfg["bases"])
    t1 = time.time()
    L, M, omega, lam = orders_scale.orders(n, bases, primes, int(cfg["workers"]))
    log(f"  10: orders of {n.size} integers in {len(bases)} bases in {time.time() - t1:.1f}s")
    idx = {b: j for j, b in enumerate(bases)}
    V2, V3 = valuation(L, 2), valuation(L, 3)
    split = dataset.split_by_n(n, cls, ocfg["split"], seed)  # same split as orders_scale
    a = v2_int(n)

    def cols(spec):
        return [idx[b] for b in _set(spec, bases)]

    # 10b: ratio estimator per decade and base set
    est: Dict[str, Any] = {}
    for name, spec in cfg["estimator_sets"].items():
        j = cols(spec)
        r = ratio_estimate(V2[:, j], n)
        pred = np.clip(r["omega"], 1, K)
        est[name] = {"bases": len(j), "decades": []}
        for lo, hi_ in ocfg["decades"]:
            rows = (n >= lo) & (n <= hi_)
            est[name]["decades"].append({"decade": [int(lo), int(hi_)], "accuracy": float((pred[rows] == cls[rows]).mean()),
                                         "accuracy_when_odd_order_seen": float((pred[rows & ~r["no_odd_order"]] == cls[rows & ~r["no_odd_order"]]).mean()) if (rows & ~r["no_odd_order"]).any() else None,
                                         "share_no_odd_order": float(r["no_odd_order"][rows].mean())})
        log(f"    estimator [{name}]: " + ", ".join(f"{d['accuracy']:.3f}" for d in est[name]["decades"]))

    # 10c: what carries ω
    jl = cols(cfg["length_set"])
    Xl, _ = lengths.length_block(n, L[:, jl])
    Xm, _ = lengths.midy_block(L[:, jl], M[:, jl])
    base_len = np.column_stack([Xl, Xm])
    logn = np.log(n.astype(float))
    prof = {}
    for spec in cfg["profile_sets"]:
        j = cols(spec)
        r = ratio_estimate(V2[:, j], n)
        prof[spec] = np.column_stack([logn, a, histogram(V2[:, j], int(cfg["v2_bins"])), histogram(V3[:, j], int(cfg["v3_bins"])), r["rank"], r["omega"]])
    big = cfg["profile_sets"][-1]
    models = {f"lengths {cfg['length_set']} (step 9b)": base_len}
    for spec in cfg["profile_sets"]:
        models[f"ℓ-adic profiles {spec}"] = prof[spec]
    models[f"lengths {cfg['length_set']} + ℓ-adic {big}"] = np.column_stack([base_len, prof[big]])
    names = list(models)
    out_dec = []
    for d, (lo, hi_) in enumerate(ocfg["decades"]):
        rows = np.nonzero((n >= lo) & (n <= hi_))[0]
        tr, te = rows[split[rows] == 0], rows[split[rows] == 2]
        s = seed + 100 * d
        rec: Dict[str, Any] = {"decade": [int(lo), int(hi_)], "test": int(te.size)}
        corr = {nm: lengths._correct(cfg, s, X, cls, tr, te) for nm, X in models.items()}
        rec["accuracy"] = {nm: float(c.mean()) for nm, c in corr.items()}
        rec["gain_vs_lengths"] = {nm: lengths._delta(corr[nm], corr[names[0]], int(cfg["bootstrap"]), s + 11 + k) for k, nm in enumerate(names[1:])}
        log(f"    decade [{lo}, {hi_}]: " + ", ".join(f"{nm} {rec['accuracy'][nm]:.4f}" for nm in names))
        out_dec.append(rec)
    return {"settings": {**cfg, "orders": {k: ocfg[k] for k in ("decades", "per_class", "max_class")}}, "integers": int(n.size), "sampling": scheme,
            "estimators": est, "decades": out_dec, "seconds": round(time.time() - t0, 1)}
