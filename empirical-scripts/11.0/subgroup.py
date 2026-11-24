"""Step 14: exact subgroup membership q ∈ ⟨b⟩ mod n and the rest of the C7 residue structure (ROADMAP step 14).

Step 9a used I_{q,b} = [ord_n(q) | ord_n(b)], which is exactly q ∈ ⟨b⟩ only when (ℤ/nℤ)* is cyclic
(n = p^k, 2p^k); otherwise it is a necessary condition. After Midy flags, zeros and those flags, a
residue excess remains in H_paper (mod 12: 0.011, mod 210: 0.023; `scale-s0-step11`). This stage
asks whether the exact membership, and membership of more primes q, explain it.

Exact membership is a digit property: q ∈ ⟨b⟩ iff q mod n is one of the remainders b^k mod n of the
long division of 1/n in base b, i.e. iff the repetend of q/n in base b is a rotation of that of 1/n.
It is computed here by the CRT: q ∈ ⟨b⟩ iff the discrete logarithms k_i of q to base b modulo each
prime power p^a ‖ n exist (k_i mod m_i = ord_{p^a}(b)) and agree modulo gcd(m_i, m_j) for all i, j.
The factorization of n is used only to compute this function of (n, b, q) quickly; the tests
compare it with the remainder set.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import claims
import dataset
import euler
import features
import lengths
import ndr
import runtime
import zeros

DEFAULTS: Dict[str, Any] = {
    "small_primes": [2, 3, 5, 7, 11, 13],
    "large_primes": "primes<100",
    "length_bases": "2..99",
    "sets": ["v8", "v10", "primes31"],
    "tolerance": 1e-9,
    "workers": 0,
}


def _primes_below(m: int) -> List[int]:
    return [p for p in range(2, m) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def _dlog(q: int, b: int, m: int, order: int) -> Optional[int]:
    """k in [0, order) with b^k ≡ q (mod m), or None (baby-step giant-step)."""
    q %= m
    b %= m
    if order <= 64:
        x = 1
        for k in range(order):
            if x == q:
                return k
            x = x * b % m
        return None
    s = math.isqrt(order) + 1
    table = {}
    x = 1
    for j in range(s):
        table.setdefault(x, j)
        x = x * b % m
    step = pow(b, -s, m)
    y = q
    for i in range(s + 1):
        j = table.get(y)
        if j is not None:
            k = i * s + j
            if k < order:
                return k
        y = y * step % m
    return None


def exact_member(n: int, b: int, q: int, factors: Sequence[tuple], spf) -> Optional[bool]:
    """q ∈ ⟨b⟩ in (ℤ/nℤ)*, or None when gcd(b·q, n) > 1."""
    if math.gcd(b * q, n) != 1:
        return None
    if n <= 2:
        return True
    logs = []
    for p, a in factors:
        m = p ** a
        if m == 2:
            continue
        lam = ndr.carmichael_from_factors([(p, a)])
        lp = [r for r, _ in ndr.factorize(lam, spf)] if lam > 1 else []
        o = ndr.multiplicative_order(b % m, m, lam, lp)
        k = _dlog(q, b, m, o)
        if k is None:
            return False
        logs.append((k, o))
    for i in range(len(logs)):
        for j in range(i + 1, len(logs)):
            g = math.gcd(logs[i][1], logs[j][1])
            if (logs[i][0] - logs[j][0]) % g:
                return False
    return True


_P: Dict[str, Any] = {}


def _init(spf, bases, qs):
    _P.update(spf=spf, bases=list(bases), qs=list(qs))


def _chunk(args):
    ns, cand = args  # cand[i]: (j_base, q) pairs that pass the divisibility test in a non-cyclic group
    spf = _P["spf"]
    out = []
    for n, pairs in zip(ns, cand):
        f = ndr.factorize(int(n), spf)
        out.append([bool(exact_member(int(n), _P["bases"][j], int(q), f, spf)) for j, q in pairs])
    return out


def _signed_chunk(args):
    ns, = args
    spf = _P["spf"]
    out = []
    for n in ns:
        f = ndr.factorize(int(n), spf)
        row = []
        for b in _P["bases"]:
            for q in _P["qs"]:
                e = exact_member(int(n), b, (-q) % int(n), f, spf) if math.gcd(b * q, int(n)) == 1 else None
                row.append(np.nan if e is None else float(e))
        out.append(row)
    return out


def negative_membership(n: np.ndarray, bases: Sequence[int], qs: Sequence[int], workers: int) -> Dict[int, np.ndarray]:
    """{q: rows × bases flags [−q ∈ ⟨b⟩] (NaN when gcd(b·q, n) > 1)}, exact for every cell.

    Only odd characters have B_{1,χ} ≠ 0, and Σ over the odd extensions of ψ_j of χ(q) is ½ Σ_ext (χ(q) − χ(−q)):
    the Euler factor at q survives the averaging iff q ∈ ⟨b⟩ or −q ∈ ⟨b⟩."""
    spf = ndr.spf_sieve(int(n.max()) + 1)
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    ns = [int(x) for x in n.tolist()]
    size = max(50, len(ns) // (workers * 8) + 1)
    tasks = [(ns[k:k + size],) for k in range(0, len(ns), size)]
    if workers <= 1 or len(tasks) <= 1:
        _init(spf, bases, qs)
        parts = [_signed_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(spf, bases, qs)) as pool:
            parts = list(pool.map(_signed_chunk, tasks))
    A = np.array([r for p in parts for r in p], dtype=float).reshape(len(ns), len(bases), len(qs))
    return {int(q): A[:, :, k] for k, q in enumerate(qs)}


def cyclic_group(n: np.ndarray, spf) -> np.ndarray:
    """(ℤ/nℤ)* cyclic ⇔ n ∈ {1, 2, 4, p^k, 2p^k} (p odd)."""
    out = np.zeros(n.size, dtype=bool)
    for i, v in enumerate(n.tolist()):
        f = ndr.factorize(int(v), spf)
        odd = [p for p, _ in f if p != 2]
        e2 = dict(f).get(2, 0)
        out[i] = (v in (1, 2, 4)) or (len(odd) == 1 and e2 <= 1)
    return out


def exact_membership(n: np.ndarray, L_all: np.ndarray, bases_all: Sequence[int], bases: Sequence[int], qs: Sequence[int], workers: int):
    """{q: rows × bases exact flags (NaN when not coprime)} and the divisibility flags; exact only re-checked where they can differ.

    L_all holds the orders in bases_all, which must contain every base and every q."""
    spf = ndr.spf_sieve(int(n.max()) + 1)
    jb = [list(bases_all).index(b) for b in bases]
    div = {q: m[:, jb] for q, m in euler.membership(L_all, bases_all, qs).items()}
    cyc = cyclic_group(n, spf)
    exact = {q: div[q].copy() for q in qs}
    cand: List[List[tuple]] = [[] for _ in range(n.size)]
    for q in qs:
        rows, cols = np.nonzero((div[q] == 1) & ~cyc[:, None])
        for i, j in zip(rows.tolist(), cols.tolist()):
            if bases[j] != q:
                cand[i].append((j, q))
    idx = [i for i in range(n.size) if cand[i]]
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    size = max(50, len(idx) // (workers * 8) + 1)
    tasks = [([int(n[i]) for i in idx[k:k + size]], [cand[i] for i in idx[k:k + size]]) for k in range(0, len(idx), size)]
    if workers <= 1 or len(tasks) <= 1:
        _init(spf, bases, qs)
        parts = [_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(spf, bases, qs)) as pool:
            parts = list(pool.map(_chunk, tasks))
    flat = [r for p in parts for r in p]
    checked = changed = 0
    for i, res in zip(idx, flat):
        for (j, q), ok in zip(cand[i], res):
            checked += 1
            if not ok:
                exact[q][i, j] = 0.0
                changed += 1
    return exact, div, cyc, {"candidates_checked": checked, "divisibility_true_but_not_member": changed}


def _ex(r: dict, q: int) -> float:
    v = r.get(q, r.get(str(q)))
    return float("nan") if v is None else v["excess"]


def run_stage(table, scfg_in: Dict[str, Any], claims_cfg, base_sets, seed: int, run_dir: Path, dataset_key: str, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(scfg_in or {})}
    t0 = time.time()
    n = table["row.n"].astype(np.int64)
    small = [int(q) for q in cfg["small_primes"]]
    large = _primes_below(100) if cfg["large_primes"] == "primes<100" else [int(q) for q in cfg["large_primes"]]
    bases98 = dataset.parse_bases(cfg["length_bases"])
    if any(q not in bases98 for q in large):
        raise ValueError("length bases must contain every q")
    L98, M98, _ = lengths.load_or_compute(run_dir, dataset_key, n, bases98, int(cfg["workers"]), log)
    zbases = features.union_bases(base_sets, cfg["sets"])
    Z, _ = zeros.load_or_compute(run_dir, dataset_key, table, zbases, float(cfg["tolerance"]), int(cfg["workers"]), log)
    Xz, _, _ = zeros.row_block(table, Z, zbases)
    Eb = Xz[:, : len(zbases)]

    ccfg = {**(claims_cfg or {}), "_base_sets": {k: v["bases"] for k, v in dataset.normalize_base_sets(base_sets).items()}}
    ctx = claims.Ctx(table, ccfg, seed, runtime.REPO_ROOT)
    pbases = sorted(claims._set_bases(ctx, ctx.cfg["primary_set"]))
    c98 = {b: j for j, b in enumerate(bases98)}
    cz = {b: j for j, b in enumerate(zbases)}
    t1 = time.time()
    exact, div, cyc, chk = exact_membership(n, L98, bases98, pbases, large, int(cfg["workers"]))
    log(f"  14a: exact membership of {len(large)} primes in ⟨b⟩ for {len(pbases)} bases: {chk['candidates_checked']} non-cyclic candidates, "
        f"{chk['divisibility_true_but_not_member']} where [ord q | ord b] holds but q ∉ ⟨b⟩ ({time.time() - t1:.0f}s)")
    t1 = time.time()
    neg = negative_membership(n, pbases, small, int(cfg["workers"]))
    log(f"  14a: exact membership of −q (q ≤ 13) in ⟨b⟩ ({time.time() - t1:.0f}s)")
    noncyc_pairs = int(sum(((div[q] == 1) & ~cyc[:, None]).sum() for q in large))
    stats = {"rows": int(n.size), "cyclic_share": float(cyc.mean()), **chk, "noncyclic_divisibility_true": noncyc_pairs,
             "share_not_member_given_divisible_noncyclic": chk["divisibility_true_but_not_member"] / max(1, chk["candidates_checked"]),
             "member_share": {int(q): float(np.nanmean(exact[q])) for q in large},
             "minus_member_share": {int(q): float(np.nanmean(neg[q])) for q in small},
             "plus_or_minus_member_share": {int(q): float(np.nanmean(np.fmax(exact[q], neg[q]))) for q in small}}

    coprime = [(np.gcd(ctx.n, b) == 1).astype(float) for b in pbases]
    Lmat, _ = claims._per_base_logL(ctx, pbases)
    per_base_L = [np.nan_to_num(Lmat[:, j], nan=0.0) for j in range(Lmat.shape[1])]
    midy_cols = [M98[:, c98[b]].astype(float) for b in pbases]
    zero_cols = [np.nan_to_num(Eb[:, cz[b]], nan=0.0) for b in pbases if b in cz] + [(Eb[:, cz[b]] > 0).astype(float) for b in pbases if b in cz]

    def cols(flags, qs):
        return [np.nan_to_num(flags[q][:, k], nan=0.0) for q in qs for k, b in enumerate(pbases) if b != q]

    variants = {
        "c7_midy_zeros": [],
        "divisibility_small": cols(div, small),
        "exact_small": cols(exact, small),
        "divisibility_large": cols(div, large),
        "exact_large": cols(exact, large),
        "exact_signed_small": cols(exact, small) + cols(neg, small),
        "exact_signed_small_divisibility_large": cols(exact, small) + cols(neg, small) + cols(div, large),
    }
    out_var: Dict[str, Any] = {}
    for var in ctx.cfg["entropies"][:2]:
        H, c7, tr, te = claims.c7_design(ctx, var, per_base_L, coprime)
        base = c7 + midy_cols + zero_cols
        res = {name: claims.residue_excess(ctx, H, base + extra, tr, te) for name, extra in variants.items()}
        # labelled diagnostics (functions of the factorization of n, not explanations): the joint coprimality pattern of the
        # bases (H is averaged over the coprime bases, which additive per-base controls cannot express), then v2(n)
        best_cols = base + variants["exact_signed_small_divisibility_large"]
        pat = np.zeros(ctx.n.size, dtype=np.int64)
        for k, p in enumerate(sorted({p for b in pbases for p in (2, 3, 5, 7, 11, 13) if b % p == 0})):
            pat += (ctx.n % p == 0).astype(np.int64) << k
        pat_cols = claims._dummies(pat, np.unique(pat[tr]))
        nn = ctx.n.astype(np.int64)
        v2 = np.minimum(np.round(np.log2((nn & -nn).astype(float))).astype(np.int64), 3)
        v2_cols = claims._dummies(v2, np.unique(v2[tr]))
        res["diag_coprime_pattern"] = claims.residue_excess(ctx, H, best_cols + pat_cols, tr, te)
        res["diag_coprime_pattern_v2"] = claims.residue_excess(ctx, H, best_cols + pat_cols + v2_cols, tr, te)
        res["diag_pattern_v2_without_membership"] = claims.residue_excess(ctx, H, base + pat_cols + v2_cols, tr, te)
        variants_cols = {"diag_coprime_pattern": len(pat_cols), "diag_coprime_pattern_v2": len(pat_cols) + len(v2_cols), "diag_pattern_v2_without_membership": len(pat_cols) + len(v2_cols)}
        # where the remainder sits: excess for the prime-power moduli of 12 and 210, and residual means per class
        best = base + variants["exact_signed_small_divisibility_large"]
        mods = ctx.cfg["moduli"]
        ctx.cfg["moduli"] = [3, 4, 8, 5, 7]
        parts = claims.residue_excess(ctx, H, best, tr, te)
        ctx.cfg["moduli"] = mods
        _, _, pred = claims._fit_eval(ctx, H, best, tr, te)
        r = H[te] - pred
        nt = ctx.n[te]
        means = {m: {int(c): {"rows": int((nt % m == c).sum()), "mean_residual": float(r[nt % m == c].mean())} for c in range(m) if (nt % m == c).any()} for m in (12, 7)}
        out_var[var] = {"test_rows": int(te.sum()), "columns": {**{k: len(v) for k, v in variants.items()}, **variants_cols}, **res,
                        "remainder_by_prime_power_modulus": parts, "remainder_class_means": means}
        log(f"    {var}: mod-12 / mod-210 excess " + ", ".join(f"{k} {_ex(res[k], 12):.4f}/{_ex(res[k], 210):.4f}" for k in variants))
    return {"settings": cfg, "primes_small": small, "primes_large": large, "bases": pbases, "membership": stats, "residues": out_var, "seconds": round(time.time() - t0, 1)}
