"""Scale-up analysis for Sieve Echo v11 (ROADMAP step 5).

For each n-range ("bin", e.g. one decade) the key claim statistics are recomputed on an
ω-balanced sample of the bin (at most ``per_omega_cap`` integers per ω; ω strata with fewer than
``min_per_omega`` integers are dropped), fitted on the bin's train rows and measured on its test
rows:

* C1 — ΔR² of log ω over (log L, log n) and the partial r(H, log ω | log L, log n);
* C2 — Spearman(H, ω) and primes − composites mean H;
* C3 — partial r(H, is_prime_power | log L, log n);
* C4 — share of n with CV < 0.1 across bases for H_paper, Hn_nodc and log L (v8 set; wide set
  where sampled);
* C9 — gradient-boosting accuracy for min(ω, K): baseline (n + per-base L + length arithmetic),
  + entropy columns, + all NDR columns, with paired-bootstrap CIs.

Each statistic gets a label per bin; a claim is *stable* when its label is the same in the last two
bins (10⁴–10⁵ and 10⁵–10⁶ by default), otherwise the drift is reported.

Growth exponents: ``log H = a·log n + c`` (paper/v7 style: 0.142 for "theta entropy"), the same with
ω fixed effects and with ``log L`` as a control. The paper's "predicted 0.743" is the growth
exponent of the double sum ``T(x) = Σ_{p≤x prime} Σ_{2≤n≤x, gcd(n,p)=1} log ord_n(p)`` (theta-7
paper §5.1), a different quantity; it is computed directly here for small x.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence

import numpy as np

import baselines
import claims
import dataset
import features
import ndr

DEFAULTS: Dict[str, Any] = {
    "bins": [[1000, 9999], [10000, 99999], [100000, 1000000]],
    "per_omega_cap": 3000,
    "min_per_omega": 30,
    "primary_set": "v8",
    "entropy": "H_paper",
    "bootstrap": 300,
    "effect_threshold": 0.01,
    "cv_threshold": 0.1,
    "max_omega_class": 5,
    "gbm": {"max_iter": 200, "learning_rate": 0.1},
    "sets": ["v8", "v10", "primes31"],
    "T_x": [100, 200, 400, 800, 1600],
}


def _balanced(rows: np.ndarray, omega: np.ndarray, cap: int, min_count: int, rng) -> np.ndarray:
    keep = []
    for w in np.unique(omega[rows]):
        part = rows[omega[rows] == w]
        if part.size < min_count:
            continue
        keep.append(part if part.size <= cap else rng.choice(part, cap, replace=False))
    return np.sort(np.concatenate(keep)) if keep else rows[:0]


def _boot(stat, n, B, seed):
    return claims.bootstrap(stat, n, B, np.random.default_rng(seed))


def bin_statistics(table, rows: np.ndarray, cfg: Dict[str, Any], fsets: Dict[str, tuple], entropy_cols: List[str], seed: int) -> Dict[str, Any]:
    pset, var = cfg["primary_set"], cfg["entropy"]
    H = table[f"row.{pset}.{var}.mean"]
    L = table[f"row.{pset}.logL.mean"]
    logn = np.log(table["row.n"].astype(float))
    omega = table["label.omega"].astype(float)
    split = table["split"]
    ok = rows[np.isfinite(H[rows]) & np.isfinite(L[rows])]
    tr, te = ok[split[ok] == 0], ok[split[ok] == 2]
    B, thr = int(cfg["bootstrap"]), float(cfg["effect_threshold"])
    out: Dict[str, Any] = {
        "rows": int(rows.size), "train": int(tr.size), "test": int(te.size),
        "n_range": [int(table["row.n"][rows].min()), int(table["row.n"][rows].max())] if rows.size else None,
        "omega_counts": {int(w): int((omega[rows] == w).sum()) for w in np.unique(omega[rows])},
    }
    if tr.size < 50 or te.size < 30:
        out["skipped"] = "too few rows"
        return out
    lw = np.log(omega)

    # C1
    Xb_tr, Xf_tr = claims.design(L[tr], logn[tr]), claims.design(lw[tr], L[tr], logn[tr])
    cb, cf = claims.lstsq(Xb_tr, H[tr]), claims.lstsq(Xf_tr, H[tr])
    pb = claims.design(L[te], logn[te]) @ cb
    pf = claims.design(lw[te], L[te], logn[te]) @ cf
    y = H[te]
    d_ci = _boot(lambda i: claims.r2(y[i], pf[i]) - claims.r2(y[i], pb[i]), y.size, B, seed + 1)
    Z = np.column_stack([L[te], logn[te]])
    pc = claims.partial_corr(lw[te], y, Z)
    pc_ci = _boot(lambda i: claims.partial_corr(lw[te][i], y[i], Z[i]), y.size, B, seed + 2)
    out["C1"] = {"r2_baseline": claims.r2(y, pb), "r2_full": claims.r2(y, pf), "delta_r2": claims.r2(y, pf) - claims.r2(y, pb), "delta_r2_ci": d_ci,
                 "partial_r": pc, "partial_r_ci": pc_ci, "label": "ω adds ≥ threshold" if d_ci[0] > thr else "explained by size/length"}

    # C2
    rho = claims.spearman(y, omega[te])
    rho_ci = _boot(lambda i: claims.spearman(y[i], omega[te][i]), y.size, B, seed + 3)
    prime = table["label.is_prime"][te].astype(bool)
    diff = float(y[prime].mean() - y[~prime].mean()) if prime.any() and (~prime).any() else float("nan")
    sign = "negative" if rho_ci[1] < 0 else ("positive" if rho_ci[0] > 0 else "unresolved")
    out["C2"] = {"spearman_H_omega": rho, "ci": rho_ci, "primes_minus_composites": diff, "label": sign}

    # C3
    ipp = table["label.is_prime_power"][te].astype(float)
    p3 = claims.partial_corr(ipp, y, Z)
    p3_ci = _boot(lambda i: claims.partial_corr(ipp[i], y[i], Z[i]), y.size, B, seed + 4)
    out["C3"] = {"point_biserial": claims.corr(y, ipp), "partial_r": p3, "partial_r_ci": p3_ci,
                 "label": "negligible" if abs(p3) < math.sqrt(thr) or not claims.excludes_zero(p3_ci) else ("negative" if p3 < 0 else "positive")}

    # C4 (no fitted parameters: all rows of the bin)
    c4 = {}
    for bset in ("v8", "wide"):
        nb = table.get(f"row.n_bases.{bset}")
        if nb is None:
            continue
        el = rows[nb[rows] >= 2]
        if el.size < 30:
            continue
        rates = {}
        for v in ("H_paper", "Hn_nodc", "logL"):
            cv = table[f"row.{bset}.{v}.cv"][el]
            rates[v] = float(np.where(np.isfinite(cv), cv < float(cfg["cv_threshold"]), False).mean())
        c4[bset] = {"rows": int(el.size), "invariance_rate": rates}
    rate = c4.get("v8", {}).get("invariance_rate", {}).get("H_paper", float("nan"))
    out["C4"] = {**c4, "label": "rare (< 50%)" if rate < 0.5 else "common (≥ 50%)"}

    # C9
    K = int(cfg["max_omega_class"])
    yc = np.minimum(table["label.omega"], K).astype(int)
    trc = rows[split[rows] == 0]
    tec = rows[split[rows] == 2]
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_eval(X):
        m = HistGradientBoostingClassifier(max_iter=int(cfg["gbm"]["max_iter"]), learning_rate=float(cfg["gbm"]["learning_rate"]),
                                           early_stopping=True, validation_fraction=0.15, random_state=seed)
        m.fit(X[trc], yc[trc])
        return (m.predict(X[tec]) == yc[tec]).astype(float)

    base = fsets["n + L arithmetic"][0]
    ent = np.column_stack([base] + [table[c] for c in entropy_cols])
    full = fsets["all allowed"][0]
    cb_, ce_, cf_ = fit_eval(base), fit_eval(ent), fit_eval(full)
    ent_ci = _boot(lambda i: ce_[i].mean() - cb_[i].mean(), cb_.size, B, seed + 5)
    all_ci = _boot(lambda i: cf_[i].mean() - cb_[i].mean(), cb_.size, B, seed + 6)
    out["C9"] = {
        "accuracy_baseline": float(cb_.mean()), "accuracy_entropy": float(ce_.mean()), "accuracy_all": float(cf_.mean()),
        "delta_entropy": float(ce_.mean() - cb_.mean()), "delta_entropy_ci": ent_ci,
        "delta_all": float(cf_.mean() - cb_.mean()), "delta_all_ci": all_ci,
        "label": ("entropy adds" if ent_ci[0] > 0 else "entropy adds nothing") + "; " + ("NDR adds" if all_ci[0] > 0 else "NDR adds nothing"),
    }
    return out


def growth(table, rows: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    pset, var = cfg["primary_set"], cfg["entropy"]
    n = table["row.n"].astype(float)
    out: Dict[str, Any] = {"paper_exponent": 0.142, "paper_predicted": 0.743}
    # paper/v7 style on the paper range, v8-set H_v8 (all integers)
    Hv8 = table["row.v8.H_v8.mean"]
    m = (table["row.source"] == 0) & (n >= 2) & (n <= 9999) & np.isfinite(Hv8) & (Hv8 > 0)
    if m.sum() > 10:
        a = claims.lstsq(claims.design(np.log(n[m])), np.log(Hv8[m]))
        out["v8_style_H_v8_paper_range"] = {"exponent": float(a[1]), "prefactor": float(np.exp(a[0])), "rows": int(m.sum())}
    H = table[f"row.{pset}.{var}.mean"]
    L = table[f"row.{pset}.logL.mean"]
    omega = table["label.omega"]
    r = rows[np.isfinite(H[rows]) & (H[rows] > 0) & np.isfinite(L[rows])]
    ln, lh, ll = np.log(n[r]), np.log(H[r]), L[r]
    a1 = claims.lstsq(claims.design(ln), lh)
    dums = [(omega[r] == w).astype(float) for w in np.unique(omega[r])[1:]]
    a2 = claims.lstsq(claims.design(ln, *dums), lh)
    a3 = claims.lstsq(claims.design(ln, np.log(np.maximum(ll, 1e-9)), *dums), lh)
    out["balanced_all_bins"] = {
        "rows": int(r.size),
        "H ~ n^a": float(a1[1]),
        "H ~ n^a, omega fixed effects": float(a2[1]),
        "H ~ n^a · (log L)^b, omega fixed effects": {"a": float(a3[1]), "b": float(a3[2])},
    }
    per_omega = {}
    for w in np.unique(omega[r]):
        s = omega[r] == w
        if s.sum() > 30:
            per_omega[int(w)] = float(claims.lstsq(claims.design(ln[s]), lh[s])[1])
    out["per_omega_exponent"] = per_omega
    return out


def T_exponent(x_values: Sequence[int]) -> Dict[str, Any]:
    """T(x) = Σ_{p≤x prime} Σ_{2≤n≤x, gcd(n,p)=1} log ord_n(p), and its log-log slope."""
    xm = int(max(x_values))
    spf = ndr.spf_sieve(xm)
    primes = [p for p in range(2, xm + 1) if spf[p] == p]
    ctx = {n: ndr.order_context(n, spf) for n in range(2, xm + 1)}
    T = []
    for x in x_values:
        total = 0.0
        for p in primes:
            if p > x:
                break
            for nn in range(2, x + 1):
                if nn % p == 0:
                    continue
                lam, lp = ctx[nn]
                total += math.log(ndr.multiplicative_order(p, nn, lam, lp))
        T.append(total)
    lx, lt = np.log(np.asarray(x_values, float)), np.log(np.asarray(T))
    slope = float(claims.lstsq(claims.design(lx), lt)[1])
    local = [float((lt[i + 1] - lt[i]) / (lx[i + 1] - lx[i])) for i in range(len(lt) - 1)]
    return {"x": list(map(int, x_values)), "T": T, "exponent": slope, "local_exponents": local,
            "terms_exponent_reference": "number of (p, n) pairs grows like x·π(x) ≈ x²/log x"}


def run_stage(table, scfg_in: Dict[str, Any], base_sets, formula_inputs, seed: int, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(scfg_in or {})}
    t0 = time.time()
    n = table["row.n"]
    omega = table["label.omega"]
    rng = np.random.default_rng([seed, 50])
    fsets = baselines.feature_sets(table, base_sets, formula_inputs, cfg["sets"])
    allowed = dataset.allowed_input_columns(table)
    entropy_cols = [c for c in allowed if not c.startswith("row.wide.") and c.split(".")[-2].startswith(("H_", "Hn_"))]
    bins_out = []
    all_rows = []
    for i, (lo, hi) in enumerate(cfg["bins"]):
        rows = np.nonzero((n >= lo) & (n <= hi))[0]
        rows = _balanced(rows, omega, int(cfg["per_omega_cap"]), int(cfg["min_per_omega"]), rng)
        all_rows.append(rows)
        stats = bin_statistics(table, rows, cfg, fsets, entropy_cols, seed + 100 * i)
        bins_out.append({"bin": [int(lo), int(hi)], **stats})
        log(f"  scale bin [{lo}, {hi}]: {stats['rows']} rows" + ("" if "skipped" in stats else f", C9 Δentropy {stats['C9']['delta_entropy']:+.4f}"))
    usable = [b for b in bins_out if "skipped" not in b]
    stability = {}
    if len(usable) >= 2:
        a, b = usable[-2], usable[-1]
        for c in ("C1", "C2", "C3", "C4", "C9"):
            stability[c] = {"bins": [a["bin"], b["bin"]], "labels": [a[c]["label"], b[c]["label"]], "stable": a[c]["label"] == b[c]["label"]}
    rows_all = np.concatenate(all_rows) if all_rows else np.zeros(0, int)
    g = growth(table, rows_all, cfg)
    t = T_exponent(cfg["T_x"]) if cfg.get("T_x") else None
    return {"settings": cfg, "bins": bins_out, "stability": stability, "growth": g, "T_x": t, "seconds": round(time.time() - t0, 1)}
