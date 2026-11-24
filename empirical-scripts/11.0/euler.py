"""Step 9a: small-prime membership q ∈ ⟨b⟩ and the C7 residue structure (ROADMAP step 9).

By Proposition 1.5(c) of `studies/9`, bin j of the NDR spectrum is (n/m)·Σ B_{1,χ} over the characters χ
that extend ψ_j from H = ⟨b⟩ to G = (ℤ/nℤ)*. For primitive χ, B_{1,χ} is a multiple of L(1, χ̄), whose
Euler product has the factor (1 − χ̄(q)/q)^{-1} at each small prime q. When q ∈ H, every extension
has χ(q) = ψ_j(q), a value fixed by the bin, so the factor tilts the spectrum in a bin-dependent but
extension-independent way; when q ∉ H, χ(q) varies over the extensions and the factors average out.
By quadratic reciprocity, whether q lies in a subgroup of (ℤ/pℤ)* correlates with p modulo 4q: a
candidate for the residue-class excess of claim C7 (mod 12, mod 210) that Midy flags and spectral
zeros do not explain (step 8c).

Features (functions of the orders only): I_{q,b} = [ord_n(q) | ord_n(b)] for gcd(q, n) = 1. For
cyclic (ℤ/nℤ)* (n = p^k, 2p^k) this is exactly q ∈ ⟨b⟩; otherwise it is a necessary condition.
Residue functions of n (Jacobi symbols, n mod 4q) are deliberately not used as controls: they would
absorb the excess by construction.

Tests: (1) the C7 residue excess with C7 + Midy + zero controls, then + I_{q,b} for the primary-set
bases; (2) pair-level test R² of H ~ log L, log b, log n, Midy, extra-zero share, then + I_{q,b}
(q ∈ Q), with each q's coefficient.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

import claims
import dataset
import features
import lengths
import runtime
import zeros

DEFAULTS: Dict[str, Any] = {
    "small_primes": [2, 3, 5, 7, 11, 13],
    "length_bases": "2..99",
    "sets": ["v8", "v10", "primes31"],
    "tolerance": 1e-9,
    "bootstrap": 500,
    "workers": 0,
}


def membership(L: np.ndarray, bases: Sequence[int], qs: Sequence[int]) -> Dict[int, np.ndarray]:
    """{q: rows × bases float matrix of [ord(q) | ord(b)]}, NaN when q or b is not coprime to n."""
    col = {b: j for j, b in enumerate(bases)}
    out = {}
    for q in qs:
        Lq = L[:, col[q]]
        M = np.full(L.shape, np.nan)
        ok = (Lq[:, None] > 0) & (L > 0)
        M[ok] = (np.broadcast_to(L, L.shape)[ok] % np.broadcast_to(Lq[:, None], L.shape)[ok] == 0).astype(float)
        out[q] = M
    return out


def pair_level(table, Z, L98, M98, bases98, qs, B, seed) -> Dict[str, Any]:
    pn, pb, pL = Z["zpair.n"], Z["zpair.b"].astype(np.int64), Z["zpair.L"].astype(np.int64)
    n = table["row.n"]
    idx = np.searchsorted(n, pn)
    col = {b: j for j, b in enumerate(bases98)}
    jb = np.array([col[int(b)] for b in pb])
    midy = M98[idx, jb].astype(float)
    extra = Z["zpair.extra_share"]
    mem = membership(L98, bases98, qs)
    Iq = {q: mem[q][idx, jb] for q in qs}
    sel = {q: np.isfinite(Iq[q]) & (pb != q) for q in qs}
    split = table["split"][idx]
    lL, lb, ln = np.log(pL.astype(float)), np.log(pb.astype(float)), np.log(pn.astype(float))
    out: Dict[str, Any] = {"pairs": int(pn.size), "share_member": {int(q): float(np.nanmean(np.where(sel[q], Iq[q], np.nan))) for q in qs}, "variants": {}}
    for var in ("H_nodc", "H_paper"):
        H = np.full(pn.size, np.nan)
        # pair entropies from the dataset's pair table (same (n, b) order is not guaranteed: map by key)
        key_t = table["pair.n"].astype(np.int64) * 1000 + table["pair.b"].astype(np.int64)
        key_z = pn.astype(np.int64) * 1000 + pb
        order = np.argsort(key_t)
        pos = order[np.searchsorted(key_t[order], key_z)]
        H = table[f"pair.{var}"][pos]
        good = np.isfinite(H) & (pL > 2)
        tr, te = good & (split == 0), good & (split == 2)
        Icols = [np.nan_to_num(np.where(sel[q], Iq[q], 0.0)) for q in qs] + [sel[q].astype(float) for q in qs]
        X0 = claims.design(lL, lb, ln, midy, extra)
        X1 = claims.design(lL, lb, ln, midy, extra, *Icols)
        c0, c1 = claims.lstsq(X0[tr], H[tr]), claims.lstsq(X1[tr], H[tr])
        p0, p1 = X0[te] @ c0, X1[te] @ c1
        y = H[te]
        d_ci = claims.bootstrap(lambda i: claims.r2(y[i], p1[i]) - claims.r2(y[i], p0[i]), y.size, B, np.random.default_rng([seed, 95]))
        r0, r1 = claims.r2(y, p0), claims.r2(y, p1)
        out["variants"][var] = {"test_pairs": int(te.sum()), "r2_structure": r0, "r2_plus_membership": r1, "delta_r2": r1 - r0, "delta_r2_ci": d_ci,
                                "residual_variance_explained": (r1 - r0) / (1 - r0) if r0 < 1 else float("nan"),
                                "coefficients": {int(q): float(c1[6 + k]) for k, q in enumerate(qs)}}
    return out


def residues(table, claims_cfg, base_sets, seed, L98, M98, bases98, Eb, zbases, qs) -> Dict[str, Any]:
    cfg = {**(claims_cfg or {}), "_base_sets": {k: v["bases"] for k, v in dataset.normalize_base_sets(base_sets).items()}}
    ctx = claims.Ctx(table, cfg, seed, runtime.REPO_ROOT)
    pbases = sorted(claims._set_bases(ctx, ctx.cfg["primary_set"]))
    coprime = [(np.gcd(ctx.n, b) == 1).astype(float) for b in pbases]
    Lmat, _ = claims._per_base_logL(ctx, pbases)
    per_base_L = [np.nan_to_num(Lmat[:, j], nan=0.0) for j in range(Lmat.shape[1])]
    c98 = {b: j for j, b in enumerate(bases98)}
    cz = {b: j for j, b in enumerate(zbases)}
    midy_cols = [M98[:, c98[b]].astype(float) for b in pbases]
    zero_cols = [np.nan_to_num(Eb[:, cz[b]], nan=0.0) for b in pbases if b in cz] + [(Eb[:, cz[b]] > 0).astype(float) for b in pbases if b in cz]
    mem = membership(L98, bases98, qs)
    mem_cols = [np.nan_to_num(mem[q][:, c98[b]], nan=0.0) for q in qs for b in pbases if b != q]
    out: Dict[str, Any] = {"controls": "C7 controls + Midy flag + extra-zero share/indicator of each primary-set base; then + [ord(q) | ord(b)] for each small prime q and primary-set base b",
                           "membership_columns": len(mem_cols), "variants": {}}
    for var in ctx.cfg["entropies"][:2]:
        H, cols, tr, te = claims.c7_design(ctx, var, per_base_L, coprime)
        base = cols + midy_cols + zero_cols
        out["variants"][var] = {"test_rows": int(te.sum()),
                                "c7_midy_zeros": claims.residue_excess(ctx, H, base, tr, te),
                                "plus_membership": claims.residue_excess(ctx, H, base + mem_cols, tr, te)}
    return out


def run_stage(table, ecfg_in: Dict[str, Any], claims_cfg, base_sets, seed: int, run_dir: Path, dataset_key: str, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(ecfg_in or {})}
    t0 = time.time()
    n = table["row.n"].astype(np.int64)
    qs = [int(q) for q in cfg["small_primes"]]
    bases98 = dataset.parse_bases(cfg["length_bases"])
    if any(q not in bases98 for q in qs):
        raise ValueError("length bases must contain the small primes")
    L98, M98, _ = lengths.load_or_compute(run_dir, dataset_key, n, bases98, int(cfg["workers"]), log)
    zbases = features.union_bases(base_sets, cfg["sets"])
    Z, _ = zeros.load_or_compute(run_dir, dataset_key, table, zbases, float(cfg["tolerance"]), int(cfg["workers"]), log)
    Xz, _, _ = zeros.row_block(table, Z, zbases)
    Eb = Xz[:, : len(zbases)]
    log("  9a: pair-level membership model")
    pl = pair_level(table, Z, L98, M98, bases98, qs, int(cfg["bootstrap"]), seed)
    for var, v in pl["variants"].items():
        log(f"    {var}: test R² {v['r2_structure']:.5f} → {v['r2_plus_membership']:.5f} (ΔR² {v['delta_r2']:+.5f})")
    log("  9a: C7 residues with membership controls")
    res = residues(table, claims_cfg, base_sets, seed, L98, M98, bases98, Eb, zbases, qs)
    return {"settings": cfg, "small_primes": qs, "pair_level": pl, "residues": res, "seconds": round(time.time() - t0, 1)}
