"""Step 8: exact zeros of the NDR spectrum (ROADMAP step 8).

For gcd(n, b) = 1, L = ord_n(b) and remainders r_i = b^i mod n (digits d_i = (b·r_i − r_{i+1})/n),
with X_j the DFT X_j(x) = Σ_i x_i e^{−2πi·ij/L}:

* X_j(d) = (b − e^{2πi·j/L}) · X_j(r) / n, and b − e^{2πi·j/L} ≠ 0: digits and remainders vanish on
  the same bins (and NDR = d/b too).
* X_j(r) = Σ_{t<k} T_t ζ^{u·t} with k = L/gcd(j, L), ζ = e^{−2πi/k}, u = j/gcd(j, L) coprime to k and
  T_t = Σ_{i ≡ t (mod k)} r_i (integers). So X_j = 0 iff Φ_k(x) divides T(x) = Σ_t T_t x^t, which does
  not depend on u: bins vanish in whole **classes** k | L (all j with L/gcd(j, L) = k).
* By character orthogonality on G = (ℤ/nℤ)*, H = ⟨b⟩, m = [G : H]:
  X_j(r) = (n/m) Σ_{χ ∈ Ĝ, χ|_H = ψ_j} B_{1,χ}, with ψ_j(b^i) = e^{−2πi·ij/L} and
  B_{1,χ} = (1/n) Σ_{a=1}^{n} χ(a)·a. B_{1,χ} = 0 for even nontrivial χ (all extensions even: Midy's
  zeros), and for χ induced by the primitive χ*, B_{1,χ} = B_{1,χ*} ∏_{p | n} (1 − χ*(p)): an Euler
  factor vanishes when χ*(p) = 1 for a prime p | n. Composite n therefore have many more zero
  classes than primes, where zeros need a cancellation among the extensions.

A class is *Midy* when the pair is a Midy pair and k | L/2 (exactly the even bins); other vanishing
classes are *extra*. This module finds the classes from a float FFT of the remainders (relative
threshold on the mean non-DC power, which Parseval gives exactly), checks class consistency, and
verifies a sample against the exact cyclotomic test. Exact zeros vanish class-wise, but a non-zero class
can contain a single tiny conjugate value; `inconsistent` counts such classes (classified non-zero by
their maximum), it does not signal an error. It then asks whether the zero classes add
information about ω beyond the step-7 baseline (orders in bases 2..99 + Midy flags).
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import claims
import dataset
import features
import lengths
import ndr
import runtime
import scale

ZEROS_VERSION = "zeros-v11.1"

PAIR_ZERO_FEATURES = ("zero_share", "extra_share", "zero_classes", "extra_classes", "min_extra_k", "log10_max_zero", "log10_min_nonzero", "inconsistent")

DEFAULTS: Dict[str, Any] = {
    "sets": ["v8", "v10", "primes31"],
    "tolerance": 1e-9,
    "exact_sample": 400,
    "exact_max_L": 400,
    "bins": scale.DEFAULTS["bins"],
    "per_omega_cap": scale.DEFAULTS["per_omega_cap"],
    "min_per_omega": scale.DEFAULTS["min_per_omega"],
    "bootstrap": 300,
    "model_seeds": 1,
    "max_omega_class": 5,
    "gbm": {"max_iter": 200, "learning_rate": 0.1},
    "length_bases": "2..99",
    "workers": 0,
}


# ---------------------------------------------------------------------------------------------
# Zero classes of one pair
# ---------------------------------------------------------------------------------------------


def pair_zeros(n: int, b: int, L: int, tol: float = 1e-9) -> np.ndarray:
    """PAIR_ZERO_FEATURES for one coprime pair (float FFT of the remainders, class-wise)."""
    out = np.zeros(len(PAIR_ZERO_FEATURES))
    out[5:7] = np.nan
    if L < 2:
        return out
    r = ndr.repetend_remainders(n, b, L).astype(np.float64)
    X = np.fft.rfft(r)
    P = (X.real ** 2 + X.imag ** 2)[1:]  # bins j = 1 .. floor(L/2)
    ref = (L * float(np.dot(r, r)) - float(np.abs(X[0]) ** 2)) / (L - 1)  # Parseval: mean power over j = 1 .. L-1
    j = np.arange(1, L // 2 + 1)
    k = L // np.gcd(j, L)
    uk, inv = np.unique(k, return_inverse=True)
    cmax = np.zeros(uk.size)
    cmin = np.full(uk.size, np.inf)
    np.maximum.at(cmax, inv, P)
    np.minimum.at(cmin, inv, P)
    thr = tol * ref
    zero = cmax < thr
    inconsistent = int(((cmin < thr) & ~zero).sum())
    half = np.bincount(inv, minlength=uk.size)
    full = np.where(uk == 2, half, 2 * half)  # bins in the class over j = 1 .. L-1 (= φ(k))
    midy = lengths.midy_flag(n, b, L)
    midy_cls = midy & ((L // 2) % uk == 0) if L % 2 == 0 else np.zeros(uk.size, bool)
    extra = zero & ~midy_cls
    out[0] = full[zero].sum() / (L - 1)
    out[1] = full[extra].sum() / (L - 1)
    out[2] = zero.sum()
    out[3] = extra.sum()
    out[4] = uk[extra].min() if extra.any() else 0
    out[5] = math.log10(max(cmax[zero].max() / ref, 1e-300)) if zero.any() else np.nan
    out[6] = math.log10(max(cmin[~zero].min() / ref, 1e-300)) if (~zero).any() else np.nan
    out[7] = inconsistent
    return out


def _cyclotomic(k: int) -> List[int]:
    import sympy

    x = sympy.Symbol("x")
    return [int(c) for c in sympy.Poly(sympy.cyclotomic_poly(k, x), x).all_coeffs()]  # leading first


def exact_zero_classes(n: int, b: int, L: int) -> set:
    """Classes k | L (k > 1) with Φ_k | Σ_t T_t x^t — exact integer arithmetic (tests / samples only)."""
    r = [int(v) for v in ndr.repetend_remainders(n, b, L)]
    out = set()
    for k in range(2, L + 1):
        if L % k:
            continue
        T = [0] * k
        for i, v in enumerate(r):
            T[i % k] += v
        poly = T[::-1]  # leading coefficient first (degree k-1)
        div = _cyclotomic(k)
        d = len(div) - 1
        rem = poly[:]
        for i in range(len(rem) - d):  # monic long division
            c = rem[i]
            if c:
                for t in range(1, d + 1):
                    rem[i + t] -= c * div[t]
        if not any(rem[len(rem) - d:]):  # deg Φ_k = d ≥ 1 for k ≥ 2
            out.add(k)
    return out


def float_zero_classes(n: int, b: int, L: int, tol: float = 1e-9) -> set:
    r = ndr.repetend_remainders(n, b, L).astype(np.float64)
    X = np.fft.fft(r)
    P = np.abs(X) ** 2
    ref = P[1:].mean()
    j = np.arange(1, L)
    k = L // np.gcd(j, L)
    return {int(c) for c in np.unique(k) if P[1:][k == c].max() < tol * ref}


# ---------------------------------------------------------------------------------------------
# All pairs (process pool, cached)
# ---------------------------------------------------------------------------------------------


def _worker(task) -> np.ndarray:
    triples, tol = task
    return np.array([pair_zeros(n, b, L, tol) for n, b, L in triples]).reshape(-1, len(PAIR_ZERO_FEATURES))


def compute(table, bases: Sequence[int], tol: float, workers: int, log=print) -> Dict[str, np.ndarray]:
    sel = np.nonzero(np.isin(table["pair.b"], list(bases)))[0]
    pn, pb, pL = table["pair.n"][sel], table["pair.b"][sel], table["pair.L"][sel].astype(np.int64)
    triples = list(zip(pn.tolist(), pb.tolist(), pL.tolist()))
    cost = (pL + 64).astype(float)
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    target = max(float(cost.sum()) / (workers * 8), 1.0)
    tasks, cur, acc = [], [], 0.0
    for t, c in zip(triples, cost.tolist()):
        cur.append(t)
        acc += c
        if acc >= target:
            tasks.append((cur, tol))
            cur, acc = [], 0.0
    if cur:
        tasks.append((cur, tol))
    t0 = time.time()
    if workers <= 1 or len(tasks) <= 1:
        parts = [_worker(t) for t in tasks]
    else:
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, part in enumerate(pool.map(_worker, tasks)):
                parts.append(part)
                if (i + 1) % max(1, len(tasks) // 5) == 0:
                    log(f"    zeros: chunk {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    F = np.vstack(parts) if parts else np.zeros((0, len(PAIR_ZERO_FEATURES)))
    out = {"zpair.n": pn, "zpair.b": pb, "zpair.L": pL}
    for i, name in enumerate(PAIR_ZERO_FEATURES):
        out[f"zpair.{name}"] = F[:, i]
    return out


def load_or_compute(run_dir: Path, dataset_key: str, table, bases, tol, workers, log=print) -> tuple[Dict[str, np.ndarray], str]:
    key = runtime.sha256_json({"dataset": dataset_key, "bases": list(map(int, bases)), "tol": tol, "version": ZEROS_VERSION})[:16]
    path = runtime.shared_cache(run_dir, f"zeros-{key}.npz")
    if path.is_file():
        with np.load(path) as z:
            if str(z["key"]) == key:
                log(f"  zeros: cache {path.name}")
                return {k: z[k] for k in z.files if k != "key"}, key
    t0 = time.time()
    Z = compute(table, bases, tol, workers, log)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, key=np.array(key), **Z)
    log(f"  zeros: {Z['zpair.n'].size} pairs in {time.time() - t0:.1f}s")
    return Z, key


# ---------------------------------------------------------------------------------------------
# Row features
# ---------------------------------------------------------------------------------------------


def row_block(table, Z, bases: Sequence[int]) -> tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Per-base extra-zero share (NaN when not coprime) + aggregates over the coprime bases."""
    n = table["row.n"]
    idx = np.searchsorted(n, Z["zpair.n"])
    col = {int(b): j for j, b in enumerate(bases)}
    j = np.array([col[int(b)] for b in Z["zpair.b"]], dtype=np.int64)
    E = np.full((n.size, len(bases)), np.nan)
    E[idx, j] = Z["zpair.extra_share"]
    A = np.full((n.size, len(bases)), np.nan)
    A[idx, j] = (Z["zpair.extra_classes"] > 0).astype(float)
    S = np.full((n.size, len(bases)), np.nan)
    S[idx, j] = Z["zpair.zero_share"]
    with np.errstate(all="ignore"):
        agg = {
            "extra any share": np.nanmean(A, axis=1),
            "extra share mean": np.nanmean(E, axis=1),
            "extra share max": np.nanmax(E, axis=1),
            "zero share mean": np.nanmean(S, axis=1),
        }
    X = np.column_stack([E] + list(agg.values()))
    return X, [*[f"extra_share.b{b}" for b in bases], *agg], agg


# ---------------------------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------------------------


def describe(table, Z, rows_by_bin) -> Dict[str, Any]:
    n = table["row.n"]
    idx = np.searchsorted(n, Z["zpair.n"])
    omega = table["label.omega"][idx]
    prime = table["label.is_prime"][idx].astype(bool)
    has = Z["zpair.extra_classes"] > 0
    out: Dict[str, Any] = {
        "pairs": int(has.size),
        "share_with_extra_zeros": float(has.mean()),
        "by_omega": {int(w): {"pairs": int((omega == w).sum()), "share_with_extra": float(has[omega == w].mean()), "mean_extra_share": float(Z["zpair.extra_share"][omega == w].mean())}
                     for w in np.unique(omega) if (omega == w).sum() >= 50},
        "primes": {"pairs": int(prime.sum()), "share_with_extra": float(has[prime].mean())},
        "prime_powers_not_prime": {"pairs": int(((omega == 1) & ~prime).sum()), "share_with_extra": float(has[(omega == 1) & ~prime].mean()) if ((omega == 1) & ~prime).any() else None},
        "numerics": {
            "max_log10_zero_power_over_mean": float(np.nanmax(Z["zpair.log10_max_zero"])) if np.isfinite(Z["zpair.log10_max_zero"]).any() else None,
            "min_log10_nonzero_power_over_mean": float(np.nanmin(Z["zpair.log10_min_nonzero"])),
            "inconsistent_classes": int(Z["zpair.inconsistent"].sum()),
        },
    }
    per_bin = []
    for (lo, hi), rows in rows_by_bin:
        inb = np.isin(idx, rows)
        per_bin.append({"bin": [int(lo), int(hi)], "by_omega": {int(w): float(has[inb & (omega == w)].mean()) for w in np.unique(omega[inb]) if (inb & (omega == w)).sum() >= 50}})
    out["by_bin"] = per_bin
    return out


def exact_check(Z, cfg, seed) -> Dict[str, Any]:
    """Float classes vs the exact cyclotomic test on a sample of short pairs (half with extra zeros)."""
    L = Z["zpair.L"]
    ok = np.nonzero((L >= 4) & (L <= int(cfg["exact_max_L"])))[0]
    rng = np.random.default_rng([seed, 80])
    has = Z["zpair.extra_classes"][ok] > 0
    k = int(cfg["exact_sample"]) // 2
    pick = np.concatenate([rng.choice(ok[has], min(k, int(has.sum())), replace=False) if has.any() else ok[:0],
                           rng.choice(ok[~has], min(k, int((~has).sum())), replace=False) if (~has).any() else ok[:0]])
    agree = 0
    mismatches = []
    for i in pick.tolist():
        n_, b_, L_ = int(Z["zpair.n"][i]), int(Z["zpair.b"][i]), int(L[i])
        e, f = exact_zero_classes(n_, b_, L_), float_zero_classes(n_, b_, L_, float(cfg["tolerance"]))
        if e == f:
            agree += 1
        elif len(mismatches) < 10:
            mismatches.append({"n": n_, "b": b_, "L": L_, "exact": sorted(e), "float": sorted(f)})
    return {"pairs": int(pick.size), "with_extra_zeros": int(min(k, int(has.sum()))), "agree": agree, "mismatches": mismatches}


def c9(table, cfg, blocks: Dict[str, np.ndarray], ent, ndr_m, seed, log=print) -> List[Dict[str, Any]]:
    n, omega, split = table["row.n"], table["label.omega"], table["split"]
    y = np.minimum(omega, int(cfg["max_omega_class"])).astype(int)
    B = int(cfg["bootstrap"])
    rng = np.random.default_rng([seed, 50])  # the scale stage's stream: same samples as steps 5 and 7
    models = {
        "L19": blocks["L19"],
        "L19 + zeros": np.column_stack([blocks["L19"], blocks["Z"]]),
        "L98 + Midy98": blocks["B"],
        "L98 + Midy98 + entropy": np.column_stack([blocks["B"], ent]),
        "L98 + Midy98 + zeros": np.column_stack([blocks["B"], blocks["Z"]]),
        "L98 + Midy98 + zeros + entropy": np.column_stack([blocks["B"], blocks["Z"], ent]),
        "L98 + Midy98 + zeros + all NDR": np.column_stack([blocks["B"], blocks["Z"], ndr_m]),
        "log n + zero aggregates": blocks["Zagg"],
    }
    comparisons = [("L19 + zeros", "L19"), ("L98 + Midy98 + zeros", "L98 + Midy98"), ("L98 + Midy98 + entropy", "L98 + Midy98"),
                   ("L98 + Midy98 + zeros + entropy", "L98 + Midy98 + zeros"), ("L98 + Midy98 + zeros + all NDR", "L98 + Midy98 + zeros")]
    out = []
    for i, (lo, hi) in enumerate(cfg["bins"]):
        rows = scale._balanced(np.nonzero((n >= lo) & (n <= hi))[0], omega, int(cfg["per_omega_cap"]), int(cfg["min_per_omega"]), rng)
        s = seed + 100 * i
        tr, te = rows[split[rows] == 0], rows[split[rows] == 2]
        rec: Dict[str, Any] = {"bin": [int(lo), int(hi)], "rows": int(rows.size), "train": int(tr.size), "test": int(te.size)}
        if tr.size < 50 or te.size < 30:
            rec["skipped"] = "too few rows"
            out.append(rec)
            continue
        corr = {}
        for name, X in models.items():
            corr[name] = lengths._correct(cfg, s, X, y, tr, te)
            log(f"    bin [{lo}, {hi}] {name}: {corr[name].mean():.4f}")
        rec["accuracy"] = {k: float(v.mean()) for k, v in corr.items()}
        rec["gain"] = {f"{a} vs {b}": lengths._delta(corr[a], corr[b], B, s + 60 + q) for q, (a, b) in enumerate(comparisons)}
        out.append(rec)
    return out


def residues(table, claims_cfg, base_sets, seed, M98, bases98, Eb, zbases) -> Dict[str, Any]:
    cfg = {**(claims_cfg or {}), "_base_sets": {k: v["bases"] for k, v in dataset.normalize_base_sets(base_sets).items()}}
    ctx = claims.Ctx(table, cfg, seed, runtime.REPO_ROOT)
    pbases = sorted(claims._set_bases(ctx, ctx.cfg["primary_set"]))
    coprime = [(np.gcd(ctx.n, b) == 1).astype(float) for b in pbases]
    Lmat, _ = claims._per_base_logL(ctx, pbases)
    per_base_L = [np.nan_to_num(Lmat[:, j], nan=0.0) for j in range(Lmat.shape[1])]
    c98 = {b: j for j, b in enumerate(bases98)}
    cz = {b: j for j, b in enumerate(zbases)}
    midy_cols = [M98[:, c98[b]].astype(float) for b in pbases if b in c98]
    zero_cols = [np.nan_to_num(Eb[:, cz[b]], nan=0.0) for b in pbases if b in cz] + [np.nan_to_num((Eb[:, cz[b]] > 0).astype(float)) for b in pbases if b in cz]
    out: Dict[str, Any] = {"controls": "C7 controls; + Midy flag of each primary-set base; + extra-zero share and indicator of each primary-set base", "variants": {}}
    for var in ctx.cfg["entropies"][:2]:
        H, cols, tr, te = claims.c7_design(ctx, var, per_base_L, coprime)
        out["variants"][var] = {"test_rows": int(te.sum()),
                                "c7_controls": claims.residue_excess(ctx, H, cols, tr, te),
                                "plus_midy": claims.residue_excess(ctx, H, cols + midy_cols, tr, te),
                                "plus_midy_zeros": claims.residue_excess(ctx, H, cols + midy_cols + zero_cols, tr, te)}
    return out


def run_stage(table, zcfg_in: Dict[str, Any], claims_cfg, base_sets, formula_inputs, seed: int, run_dir: Path, dataset_key: str, log=print) -> Dict[str, Any]:
    import baselines

    cfg = {**DEFAULTS, **(zcfg_in or {})}
    t0 = time.time()
    n = table["row.n"].astype(np.int64)
    zbases = features.union_bases(base_sets, cfg["sets"])
    log(f"  8a: zero classes for the pairs of bases {zbases}")
    Z, zkey = load_or_compute(run_dir, dataset_key, table, zbases, float(cfg["tolerance"]), int(cfg["workers"]), log)
    bases98 = dataset.parse_bases(cfg["length_bases"])
    L98, M98, _ = lengths.load_or_compute(run_dir, dataset_key, n, bases98, int(cfg["workers"]), log)
    Xz, znames, agg = row_block(table, Z, zbases)
    Eb = Xz[:, : len(zbases)]

    rng = np.random.default_rng([seed, 50])
    rows_by_bin = [((lo, hi), scale._balanced(np.nonzero((n >= lo) & (n <= hi))[0], table["label.omega"], int(cfg["per_omega_cap"]), int(cfg["min_per_omega"]), rng))
                   for lo, hi in cfg["bins"]]
    desc = describe(table, Z, rows_by_bin)
    log("  8a: exact cyclotomic check on a sample")
    exact = exact_check(Z, cfg, seed)
    log(f"    exact check: {exact['agree']}/{exact['pairs']} agree")

    fsets = baselines.feature_sets(table, base_sets, formula_inputs, cfg["sets"])
    allowed = dataset.allowed_input_columns(table)
    ndr_cols = [c for c in allowed if not c.startswith("row.wide.") and not c.startswith("row.n_bases.") and c != "row.n"]
    entropy_cols = [c for c in ndr_cols if c.split(".")[-2].startswith(("H_", "Hn_"))]
    X98, _ = lengths.length_block(n, L98)
    m98, _ = lengths.midy_block(L98, M98)
    blocks = {
        "L19": fsets["n + L arithmetic"][0],
        "B": np.column_stack([X98, m98]),
        "Z": Xz,
        "Zagg": np.column_stack([np.log(n.astype(float))] + list(agg.values())),
    }
    log("  8b: C9 per n-range with zero features")
    bins = c9(table, cfg, blocks, np.column_stack([table[c] for c in entropy_cols]), np.column_stack([table[c] for c in ndr_cols]), seed, log)
    log("  8c: residue classes with Midy and zero controls")
    res = residues(table, claims_cfg, base_sets, seed, M98, bases98, Eb, zbases)
    return {"version": ZEROS_VERSION, "settings": cfg, "zero_bases": zbases, "zeros_cache_key": zkey, "zero_feature_names": znames,
            "describe": desc, "exact_check": exact, "c9_bins": bins, "residues": res, "seconds": round(time.time() - t0, 1)}
