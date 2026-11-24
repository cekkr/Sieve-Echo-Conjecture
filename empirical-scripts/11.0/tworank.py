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


def mle_rank(V2: np.ndarray, levels: int = 24, rmax: int = 12) -> np.ndarray:
    """Maximum-likelihood rank₂ from the v₂ profile of the orders (step 11b).

    Treating the bases as uniform units, Proposition 1.6 gives P(v₂ ≤ k) = 2^{−s_k} with s_k = Σ_{j ≥ k} r_j and
    r_j = #{i : e_i > j} non-increasing. With n_k bases at level k and N_{≤j} = Σ_{k ≤ j} n_k the log-likelihood is
    Σ_j [−r_j N_{≤j} ln 2 + n_{j+1} ln(1 − 2^{−r_j})], separable in j; it is maximized exactly over integers
    0 ≤ r_j ≤ rmax with r_0 ≥ r_1 ≥ … by a dynamic program over j (vectorized over rows). Levels ≥ `levels`
    are lumped into the top level; N_{≤0} = 0 (no odd-order base) is replaced by 1/2, as in the ratio estimator.
    Returns r_0 = rank₂.
    """
    has = V2 >= 0
    V = np.where(has, np.minimum(V2, levels - 1), -1)
    nk = np.stack([((V == k) & has).sum(axis=1) for k in range(levels)], axis=1).astype(float)  # rows × levels
    N = np.cumsum(nk, axis=1)
    N[:, 0] = np.maximum(N[:, 0], 0.5)
    r = np.arange(rmax + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log1m = np.log1p(-(2.0 ** -r))  # −inf at r = 0
    best = None  # B_j(r) = f_j(r) + max_{r' ≤ r} B_{j+1}(r')
    for j in range(levels - 2, -1, -1):
        nxt = nk[:, j + 1][:, None]
        with np.errstate(invalid="ignore"):  # 0 · (−inf) where n_{j+1} = 0 is discarded by the where
            f = -r[None, :] * N[:, j][:, None] * math.log(2) + np.where(nxt > 0, nxt * log1m[None, :], 0.0)
        f = np.where((nxt > 0) & (r[None, :] == 0), -np.inf, f)
        best = f if best is None else f + np.maximum.accumulate(best, axis=1)
    return np.argmax(best, axis=1).astype(float)


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
        pred_mle = np.clip(omega_from_rank(mle_rank(V2[:, j]), n), 1, K)
        est[name] = {"bases": len(j), "decades": []}
        for lo, hi_ in ocfg["decades"]:
            rows = (n >= lo) & (n <= hi_)
            est[name]["decades"].append({"decade": [int(lo), int(hi_)], "accuracy": float((pred[rows] == cls[rows]).mean()),
                                         "accuracy_mle": float((pred_mle[rows] == cls[rows]).mean()),
                                         "accuracy_when_odd_order_seen": float((pred[rows & ~r["no_odd_order"]] == cls[rows & ~r["no_odd_order"]]).mean()) if (rows & ~r["no_odd_order"]).any() else None,
                                         "share_no_odd_order": float(r["no_odd_order"][rows].mean())})
        log(f"    estimator [{name}]: ratio " + ", ".join(f"{d['accuracy']:.3f}" for d in est[name]["decades"]) + "; MLE " + ", ".join(f"{d['accuracy_mle']:.3f}" for d in est[name]["decades"]))

    # 10c / 11a: what carries ω
    logn = np.log(n.astype(float))
    cache: Dict[tuple, np.ndarray] = {}

    def lengths_X(spec):
        if ("L", spec) not in cache:
            j = cols(spec)
            Xl, _ = lengths.length_block(n, L[:, j])
            Xm, _ = lengths.midy_block(L[:, j], M[:, j])
            cache[("L", spec)] = np.column_stack([Xl, Xm])
        return cache[("L", spec)]

    def profile_X(spec):
        if ("P", spec) not in cache:
            j = cols(spec)
            r = ratio_estimate(V2[:, j], n)
            cache[("P", spec)] = np.column_stack([logn, a, histogram(V2[:, j], int(cfg["v2_bins"])), histogram(V3[:, j], int(cfg["v3_bins"])),
                                                  r["rank"], r["omega"], mle_rank(V2[:, j])])
        return cache[("P", spec)]

    specs = cfg.get("models") or ([{"lengths": cfg["length_set"]}] + [{"profiles": sp} for sp in cfg["profile_sets"]]
                                  + [{"lengths": cfg["length_set"], "profiles": cfg["profile_sets"][-1]}])
    models = {}
    for m in specs:
        parts, label = [], []
        if m.get("lengths"):
            parts.append(lengths_X(m["lengths"]))
            label.append(f"lengths {m['lengths']}")
        if m.get("profiles"):
            parts.append(profile_X(m["profiles"]))
            label.append(f"ℓ-adic {m['profiles']}")
        models[m.get("name") or " + ".join(label)] = np.column_stack(parts)
    pairs = [tuple(p_) for p_ in cfg.get("comparisons") or []]
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
        if pairs:
            rec["comparisons"] = {f"{x} vs {y}": lengths._delta(corr[x], corr[y], int(cfg["bootstrap"]), s + 31 + k) for k, (x, y) in enumerate(pairs)}
        log(f"    decade [{lo}, {hi_}]: " + ", ".join(f"{nm} {rec['accuracy'][nm]:.4f}" for nm in names))
        out_dec.append(rec)
    return {"settings": {**cfg, "orders": {k: ocfg[k] for k in ("decades", "per_class", "max_class")}}, "integers": int(n.size), "sampling": scheme,
            "estimators": est, "decades": out_dec, "seconds": round(time.time() - t0, 1)}
