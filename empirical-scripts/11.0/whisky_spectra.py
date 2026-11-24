"""Step 8e: NDR spectra after SinWhisky resampling (libs/sinwhisky submodule).

SinWhisky (libs/sinwhisky, `v2/whisky_np.py`) upsamples a sequence by fitting the circle through every
three consecutive samples and blending neighbouring circles with a cosine window — a *nonlinear*,
geometric interpolation. Periodic linear interpolation is the linear control: it is zero-stuffing
followed by a fixed filter, so its DFT is X_{j mod L} times a filter response and every exact zero of
the original spectrum stays exactly zero (step 8 classes). This stage asks, on a stratified sample of
(n, b) pairs of the 19 v8/v10/primes31 bases:

* **leakage** — power at the original zero classes (bins 1 ≤ j < L/2) relative to the mean baseband
  power, after SinWhisky and after linear resampling;
* **Midy** — the coefficient of the Midy flag in H ~ log L + log b + log n + Midy (log 2 for the plain
  FFT entropy, step 7);
* **ω beyond the known structure** — test ΔR² of log ω added to H ~ log L, log b, log n, Midy flag,
  extra-zero share (fit on train pairs, scored on test pairs, bootstrap CI), for the plain FFT entropy,
  the SinWhisky entropy and the linear-resampling entropy.

Entropies here are Shannon entropies (natural log) of the normalized power over all non-DC bins of the
(resampled) sequence.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict

import numpy as np

import claims
import lengths
import ndr
import runtime
import zeros

DEFAULTS: Dict[str, Any] = {
    "zoom": 3,
    "neighbors": 1,
    "bins": [[1000, 9999], [10000, 99999], [100000, 1000000]],
    "per_stratum": 300,
    "max_L": 50000,
    "min_L": 8,
    "sets": ["v8", "v10", "primes31"],
    "bootstrap": 500,
}


def _entropy(P: np.ndarray) -> float:
    s = float(P.sum())
    if s <= 0:
        return float("nan")
    p = P[P > 0] / s
    return float(-(p * np.log(p)).sum())


def linear_periodic(x: np.ndarray, zoom: int) -> np.ndarray:
    t = np.arange(zoom + 1) / (zoom + 1)
    return ((1 - t)[None, :] * x[:, None] + t[None, :] * np.roll(x, -1)[:, None]).reshape(-1)


def pair_record(sw, n: int, b: int, L: int, zoom: int, neighbors: int) -> Dict[str, float]:
    x = ndr.ndr(ndr.repetend_digits(n, b, L), b)
    P0 = np.abs(np.fft.fft(x)) ** 2
    cls = zeros.float_zero_classes(n, b, L)
    j = np.arange(1, (L + 1) // 2)
    kj = L // np.gcd(j, L)
    zb = j[np.isin(kj, list(cls))] if cls else j[:0]
    rec = {"H_fft": _entropy(P0[1:]), "zero_bins": int(zb.size)}
    for name, xs in (("sw", sw.sinwhisky_resample_np(x, zoom, neighbors=neighbors, periodic=True)), ("lin", linear_periodic(x, zoom))):
        Ps = np.abs(np.fft.fft(xs)) ** 2
        rec[f"H_{name}"] = _entropy(Ps[1:])
        base = Ps[1: max(2, (L + 1) // 2)]
        ref = float(base.mean()) if base.size else float("nan")
        rec[f"leak_{name}"] = math.log10(max(float(np.median(Ps[zb])) / ref, 1e-300)) if zb.size and ref > 0 else float("nan")
    return rec


def run_stage(table, wcfg_in: Dict[str, Any], base_sets, seed: int, log=print) -> Dict[str, Any]:
    import features

    cfg = {**DEFAULTS, **(wcfg_in or {})}
    t0 = time.time()
    sw = runtime.bootstrap_sinwhisky()
    bases = features.union_bases(base_sets, cfg["sets"])
    pn, pb, pL = table["pair.n"], table["pair.b"], table["pair.L"].astype(np.int64)
    idx = np.searchsorted(table["row.n"], pn)
    omega = table["label.omega"][idx]
    split = table["split"][idx]
    ok = np.isin(pb, bases) & (pL >= int(cfg["min_L"])) & (pL <= int(cfg["max_L"]))
    rng = np.random.default_rng([seed, 90])
    pick = []
    strata = {}
    for lo, hi in cfg["bins"]:
        inb = ok & (pn >= lo) & (pn <= hi)
        for w in np.unique(omega[inb]):
            cand = np.nonzero(inb & (omega == w))[0]
            take = rng.choice(cand, min(int(cfg["per_stratum"]), cand.size), replace=False) if cand.size else cand
            strata[f"{lo}-{hi}/ω={int(w)}"] = int(take.size)
            pick.append(take)
    pick = np.sort(np.concatenate(pick)) if pick else np.zeros(0, int)
    log(f"  8e: {pick.size} pairs (L ≤ {cfg['max_L']}), SinWhisky zoom {cfg['zoom']}, neighbors {cfg['neighbors']}")
    recs = [pair_record(sw, int(pn[i]), int(pb[i]), int(pL[i]), int(cfg["zoom"]), int(cfg["neighbors"])) for i in pick.tolist()]
    R = {k: np.array([r[k] for r in recs], dtype=float) for k in recs[0]} if recs else {}
    n_, b_, L_ = pn[pick].astype(float), pb[pick].astype(float), pL[pick]
    midy = np.array([lengths.midy_flag(int(a), int(c), int(d)) for a, c, d in zip(pn[pick], pb[pick], L_)], dtype=float)
    extra = np.array([zeros.pair_zeros(int(a), int(c), int(d))[1] for a, c, d in zip(pn[pick], pb[pick], L_)])
    w_, sp = omega[pick].astype(float), split[pick]
    out: Dict[str, Any] = {"settings": cfg, "pairs": int(pick.size), "strata": strata, "sinwhisky_git": runtime.git_info(runtime.SINWHISKY_ROOT)}
    zmask = R["zero_bins"] > 0
    out["leakage"] = {"pairs_with_zero_classes": int(zmask.sum()),
                      **{f"{nm}": {"median_log10": float(np.nanmedian(R[f"leak_{nm}"][zmask])), "max_log10": float(np.nanmax(R[f"leak_{nm}"][zmask])),
                                   "share_above_1e-6": float(np.mean(R[f"leak_{nm}"][zmask] > -6))} for nm in ("sw", "lin")}}
    lL, lb, ln = np.log(L_.astype(float)), np.log(b_), np.log(n_)
    tr, te = sp == 0, sp == 2
    B = int(cfg["bootstrap"])
    models: Dict[str, Any] = {}
    for var in ("H_fft", "H_sw", "H_lin"):
        H = R[var]
        good = np.isfinite(H)
        a, e = tr & good, te & good
        Xm = claims.design(lL, lb, ln, midy)
        cm = claims.lstsq(Xm[a], H[a])
        X0 = claims.design(lL, lb, ln, midy, extra)
        X1 = claims.design(lL, lb, ln, midy, extra, np.log(w_))
        c0, c1 = claims.lstsq(X0[a], H[a]), claims.lstsq(X1[a], H[a])
        p0, p1 = X0[e] @ c0, X1[e] @ c1
        y = H[e]
        d_ci = claims.bootstrap(lambda i: claims.r2(y[i], p1[i]) - claims.r2(y[i], p0[i]), y.size, B, np.random.default_rng([seed, 91]))
        models[var] = {"test_pairs": int(e.sum()), "midy_coefficient": float(cm[-1]), "r2_structure": claims.r2(y, p0), "r2_plus_log_omega": claims.r2(y, p1),
                       "delta_r2_log_omega": claims.r2(y, p1) - claims.r2(y, p0), "delta_r2_ci": d_ci, "log_omega_coefficient": float(c1[-1])}
        log(f"    {var}: Midy coef {cm[-1]:+.3f}; ΔR² log ω {models[var]['delta_r2_log_omega']:+.5f} CI [{d_ci[0]:+.5f}, {d_ci[1]:+.5f}]")
    out["models"] = models
    out["corr_H_fft_H_sw"] = claims.corr(R["H_fft"], R["H_sw"])
    out["seconds"] = round(time.time() - t0, 1)
    return out
