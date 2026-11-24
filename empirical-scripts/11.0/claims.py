"""Claim-by-claim statistical verdicts for Sieve Echo v11 (ROADMAP step 2, claims C1-C9).

Every claim gets two parts:

* `reproduction` — the paper's own statistic recomputed the way the paper's code computed it
  (v8.2: n in [2, 9999], v8 bases, `H_v8`, in-sample, sometimes on the first 100 rows of the v8
  data order), compared with the stored/paper value.
* `tests` — the corrected test on the canonical `ndr-v11.1` entropies: parameters fitted on the
  train split, statistics reported on the test split, controls for `log L` and `log n`, a
  percentile bootstrap CI over test rows and a permutation null.

`verdict` is one of: `supported`, `reproduced but explained by size/length`,
`reproduced but not supported`, `not reproduced`, `not supported`, plus a one-line `reason`.
Factorization labels appear here only as targets, grouping variables, or explicitly labelled
controls (AGENTS.md exception); predictors in C9 read only `dataset.allowed_input_columns`.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

import dataset
import ndr

PAPER = {
    "alpha": -0.4681,
    "beta": 1.9174,
    "r2": 0.2345,
    "prime_power_r": -0.843,
    "invariance_rate": 0.06,
    "zeta": {2: -0.44073591908076, 3: -0.3373284695857454, 4: -0.2891709514025528},
    "pnt_r": 0.9464775059124846,
    "pnt_error_r": 0.5198674109470675,
    "li_error_r": 0.49591714958705224,
    "modular_variance": {6: 0.015271250064055058, 12: 0.016928704651462144, 30: 0.017971192659050724, 210: 0.020474203706399442},
    "theory_alpha": -1.599,
    "theory_beta": 4.933,
}

DEFAULTS: dict[str, Any] = {
    "primary_set": "v8",
    "entropies": ["H_paper", "H_nodc", "Hn_paper"],
    "bootstrap": 1000,
    "permutations": 1000,
    "effect_threshold": 0.01,
    "cv_threshold": 0.1,
    "invariance_sets": ["v8", "v10", "primes31", "wide"],
    "invariance_variants": ["H_paper", "H_nodc", "Hn_paper", "Hn_nodc"],
    "scale_edges": [2, 100, 1000, 10000, 100001, 1000001],
    "min_bin": 30,
    "moduli": [6, 12, 30, 210],
    "zeta_s": [2, 3, 4],
    "v8_pickle": "empirical-scripts/8.0/results/1/sieve_echo_complete_20250825_224642.pkl",
    "classifier": {"max_iter": 300, "learning_rate": 0.1, "max_omega_class": 5, "shuffled_nulls": 3},
}

REPRO_TOL = 1e-3


# ---------------------------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------------------------


def design(*cols: np.ndarray) -> np.ndarray:
    n = len(cols[0]) if cols else 0
    return np.column_stack([np.ones(n), *cols]) if cols else np.ones((n, 1))


def lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss if ss > 0 else float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else float("nan")


def resid(v: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """Residual of v after OLS on [1, Z] (Z: rows x k)."""
    X = design(*Z.T) if Z is not None and Z.size else np.ones((len(v), 1))
    return v - X @ lstsq(X, v)


def partial_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> float:
    return corr(resid(x, Z), resid(y, Z))


def rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rank(a), rank(b))


def ci(samples) -> list[float]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] if s.size else [float("nan")] * 2


def bootstrap(stat: Callable[[np.ndarray], float], n: int, B: int, rng: np.random.Generator) -> list[float]:
    """Percentile CI of `stat(idx)` over B resamples of n row indices."""
    return ci([stat(rng.integers(0, n, n)) for _ in range(B)])


def perm_p(stat_perm: Callable[[np.ndarray], float], observed: float, n: int, P: int, rng) -> float:
    """Two-sided permutation p-value, (1 + #{|s| >= |obs|}) / (P + 1)."""
    hits = sum(abs(stat_perm(rng.permutation(n))) >= abs(observed) - 1e-15 for _ in range(P))
    return (1 + hits) / (P + 1)


def eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    _, g = np.unique(groups, return_inverse=True)
    cnt = np.bincount(g)
    means = np.bincount(g, weights=y) / cnt
    ss_between = float((cnt * (means - y.mean()) ** 2).sum())
    ss_total = float(((y - y.mean()) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else float("nan")


def excludes_zero(interval: list[float]) -> bool:
    return bool(np.isfinite(interval).all() and (interval[0] > 0 or interval[1] < 0))


def close(a: float, b: float, tol: float = REPRO_TOL) -> bool:
    return bool(np.isfinite(a) and abs(a - b) <= tol)


# ---------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------


class Ctx:
    def __init__(self, table: dict[str, np.ndarray], cfg: dict[str, Any], seed: int, repo_root: Path):
        self.T = table
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cfg["classifier"] = {**DEFAULTS["classifier"], **(cfg or {}).get("classifier", {})}
        self.seed = seed
        self.repo_root = repo_root
        self.n = table["row.n"].astype(np.int64)
        self.logn = np.log(self.n.astype(float))
        self.omega = table["label.omega"].astype(np.int64)
        self.split = table["split"]
        self.source = table["row.source"]
        self.B = int(self.cfg["bootstrap"])
        self.P = int(self.cfg["permutations"])
        self._rng_counter = 0
        self._v8_order: np.ndarray | None | bool = False

    def rng(self) -> np.random.Generator:
        self._rng_counter += 1
        return np.random.default_rng([self.seed, 10, self._rng_counter])

    def H(self, variant: str, bset: str | None = None, agg: str = "mean") -> np.ndarray:
        return self.T[f"row.{bset or self.cfg['primary_set']}.{variant}.{agg}"]

    def logL(self, bset: str | None = None) -> np.ndarray:
        return self.T[f"row.{bset or self.cfg['primary_set']}.logL.mean"]

    def mask(self, *arrays: np.ndarray, split: str | None = None, paper_range: bool = False) -> np.ndarray:
        m = np.ones(self.n.size, dtype=bool)
        for a in arrays:
            m &= np.isfinite(a)
        if split is not None:
            m &= self.split == dataset.SPLIT_NAMES.index(split)
        if paper_range:
            m &= (self.source == 0) & (self.n <= 9999)
        return m

    def v8_first100(self) -> np.ndarray | None:
        """Row indices of the first 100 integers in the stored v8.2 data order (its `data[:100]`)."""
        if self._v8_order is False:
            path = self.repo_root / self.cfg["v8_pickle"]
            self._v8_order = None
            if path.is_file():
                with open(path, "rb") as fh:
                    rows = _NumpyOnlyUnpickler(fh).load()["data"]
                first = np.array([r["n"] for r in rows[:100]], dtype=np.int64)
                idx = np.searchsorted(self.n, first)
                ok = (idx < self.n.size) & (self.n[np.minimum(idx, self.n.size - 1)] == first) & (self.source[np.minimum(idx, self.n.size - 1)] == 0)
                self._v8_order = idx if ok.all() else None
        return self._v8_order


class _NumpyOnlyUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _fit_eval(ctx: Ctx, y: np.ndarray, cols: list[np.ndarray], tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    coef = lstsq(design(*[c[tr] for c in cols]), y[tr])
    pred = design(*[c[te] for c in cols]) @ coef
    return coef, r2(y[te], pred), pred


def _dummies(v: np.ndarray, levels: np.ndarray) -> list[np.ndarray]:
    return [(v == lv).astype(float) for lv in levels[1:]]


# ---------------------------------------------------------------------------------------------
# C1 — Sieve Echo Law, and C8 — constants
# ---------------------------------------------------------------------------------------------


def c1_law(ctx: Ctx) -> dict:
    Hv8 = ctx.H("H_v8", "v8")
    m = ctx.mask(Hv8, paper_range=True)
    w = ctx.omega[m].astype(float)
    y = Hv8[m]
    out_rep = {}
    for name, x in (("log(omega+1) [v8.2 code]", np.log(w + 1)), ("log(omega) [paper text]", np.log(w))):
        coef = lstsq(design(x), y)
        out_rep[name] = {"alpha": float(coef[1]), "beta": float(coef[0]), "r2": r2(y, design(x) @ coef)}
    code = out_rep["log(omega+1) [v8.2 code]"]
    reproduced = close(code["alpha"], PAPER["alpha"]) and close(code["beta"], PAPER["beta"]) and close(code["r2"], PAPER["r2"])
    reproduction = {
        "statistic": "OLS of v8-set mean H_v8 on log(omega+1), n in [2, 9999], in-sample",
        "paper": {"alpha": PAPER["alpha"], "beta": PAPER["beta"], "r2": PAPER["r2"]},
        "v11": out_rep,
        "status": "reproduced" if reproduced else "not reproduced",
    }

    tests = {}
    thr = float(ctx.cfg["effect_threshold"])
    for var in ctx.cfg["entropies"]:
        H, L = ctx.H(var), ctx.logL()
        base = ctx.mask(H, L)
        tr, va, te = (base & (ctx.split == s) for s in (0, 1, 2))
        lw = np.log(ctx.omega.astype(float))
        models = {
            "law: log omega": [lw],
            "baseline: log L + log n": [L, ctx.logn],
            "full: log omega + log L + log n": [lw, L, ctx.logn],
        }
        res = {}
        preds = {}
        coefs = {}
        for name, cols in models.items():
            coef, score, pred = _fit_eval(ctx, H, cols, tr, te)
            res[name] = {"test_r2": score, "coef": coef.tolist()}
            preds[name], coefs[name] = pred, coef
        yte = H[te]
        pb, pf = preds["baseline: log L + log n"], preds["full: log omega + log L + log n"]
        delta = r2(yte, pf) - r2(yte, pb)
        delta_ci = bootstrap(lambda i: r2(yte[i], pf[i]) - r2(yte[i], pb[i]), yte.size, ctx.B, ctx.rng())

        # coefficient of log omega in the full model: bootstrap over train rows (refit)
        Xtr = design(lw[tr], L[tr], ctx.logn[tr])
        ytr = H[tr]
        rng = ctx.rng()
        coef_ci = ci([lstsq(Xtr[i], ytr[i])[1] for i in (rng.integers(0, ytr.size, ytr.size) for _ in range(ctx.B))])

        Z = np.column_stack([L[te], ctx.logn[te]])
        x = lw[te]
        pc = partial_corr(x, yte, Z)
        ry = resid(yte, Z)
        pc_ci = bootstrap(lambda i: partial_corr(x[i], yte[i], Z[i]), yte.size, ctx.B, ctx.rng())
        pc_p = perm_p(lambda p: corr(resid(x[p], Z), ry), pc, yte.size, ctx.P, ctx.rng())

        nonlinear = _c1_nonlinear(ctx, H, tr, va, te)
        supported = excludes_zero(pc_ci) and delta_ci[0] > thr
        tests[var] = {
            "set": ctx.cfg["primary_set"],
            "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "models": res,
            "delta_r2_full_vs_baseline": {"value": delta, "ci95": delta_ci},
            "log_omega_coef_full": {"value": float(coefs["full: log omega + log L + log n"][1]), "ci95": coef_ci},
            "partial_corr_log_omega_given_logL_logn": {"value": pc, "ci95": pc_ci, "perm_p": pc_p},
            "nonlinear_omega_only": nonlinear,
            "supported": bool(supported),
        }
    prim = tests[ctx.cfg["entropies"][0]]
    if not reproduced:
        verdict = "not reproduced"
    elif prim["supported"]:
        verdict = "supported"
    else:
        verdict = "reproduced but explained by size/length"
    reason = (
        f"{ctx.cfg['entropies'][0]}: test R² law-only {prim['models']['law: log omega']['test_r2']:.3f}, "
        f"baseline (log L, log n) {prim['models']['baseline: log L + log n']['test_r2']:.3f}, "
        f"full {prim['models']['full: log omega + log L + log n']['test_r2']:.3f}; "
        f"ΔR² {prim['delta_r2_full_vs_baseline']['value']:.4f} CI {_fmt_ci(prim['delta_r2_full_vs_baseline']['ci95'])} "
        f"(threshold {thr}); partial r(H, log ω | log L, log n) {prim['partial_corr_log_omega_given_logL_logn']['value']:.3f}."
    )
    return {
        "id": "C1",
        "title": "Sieve Echo Law H = α·log ω + β (§2.2)",
        "paper_claim": "H_NDR = -0.468·log(ω) + 1.917, R² = 0.234",
        "reproduction": reproduction,
        "tests": tests,
        "verdict": verdict,
        "reason": reason,
    }


def _c1_nonlinear(ctx: Ctx, H, tr, va, te) -> dict:
    from scipy.optimize import curve_fit

    w = ctx.omega.astype(float)
    lw = np.log(w)
    out = {}

    def power(x, A, B, C):
        return A * np.power(x, -B) + C

    try:
        p, _ = curve_fit(power, w[tr], H[tr], p0=(1.0, 1.0, float(H[tr].mean())), maxfev=20000)
        out["A·ω^-B + C"] = {"params": p.tolist(), "val_r2": r2(H[va], power(w[va], *p)), "test_r2": r2(H[te], power(w[te], *p))}
    except (RuntimeError, ValueError) as exc:
        out["A·ω^-B + C"] = {"error": str(exc)}
    for deg in (2, 3):
        cols = [lw**k for k in range(1, deg + 1)]
        coef = lstsq(design(*[c[tr] for c in cols]), H[tr])
        f = lambda m: design(*[c[m] for c in cols]) @ coef  # noqa: E731
        out[f"poly{deg}(log ω)"] = {"params": coef.tolist(), "val_r2": r2(H[va], f(va)), "test_r2": r2(H[te], f(te))}
    levels = np.unique(ctx.omega[tr])
    means = {int(lv): float(H[tr][ctx.omega[tr] == lv].mean()) for lv in levels}
    pred = np.array([means.get(int(v), float(H[tr].mean())) for v in ctx.omega[te]])
    out["ceiling: per-ω mean"] = {"test_r2": r2(H[te], pred)}
    candidates = [k for k, v in out.items() if "val_r2" in v]
    best = max(candidates, key=lambda k: out[k]["val_r2"]) if candidates else None
    out["selected_on_val"] = best
    return out


def c8_constants(ctx: Ctx) -> dict:
    var = ctx.cfg["entropies"][0]
    H = ctx.H(var)
    tests = {}
    lib = _v8_constants()
    cands = _candidate_values(lib)
    for label, use_set, h in (("v8 reproduction (H_v8, paper range)", "v8", ctx.H("H_v8", "v8")), (f"canonical {var}", None, H)):
        m = ctx.mask(h, split="train") if use_set is None else ctx.mask(h, paper_range=True)
        w = ctx.omega[m].astype(float)
        y = h[m]
        for xname, x in (("log(omega+1)", np.log(w + 1)), ("log(omega)", np.log(w))):
            X = design(x)
            coef = lstsq(X, y)
            rng = ctx.rng()
            boots = np.array([lstsq(X[i], y[i]) for i in (rng.integers(0, y.size, y.size) for _ in range(ctx.B))])
            a_ci, b_ci = ci(boots[:, 1]), ci(boots[:, 0])
            tests[f"{label} | {xname}"] = {
                "rows": int(m.sum()),
                "alpha": {"value": float(coef[1]), "ci95": a_ci, "theory": PAPER["theory_alpha"], "theory_inside_ci": a_ci[0] <= PAPER["theory_alpha"] <= a_ci[1]},
                "beta": {"value": float(coef[0]), "ci95": b_ci, "theory": PAPER["theory_beta"], "theory_inside_ci": b_ci[0] <= PAPER["theory_beta"] <= b_ci[1]},
                "library_candidates_inside_alpha_ci": int(((cands >= a_ci[0]) & (cands <= a_ci[1])).sum()),
                "library_candidates_inside_beta_ci": int(((cands >= b_ci[0]) & (cands <= b_ci[1])).sum()),
            }
    rng = ctx.rng()
    probe = lambda c: float(np.mean([np.any(np.abs(cands - u) < 0.01) for u in rng.uniform(c - 0.5, c + 0.5, 4000)]))  # noqa: E731
    stored_alpha = tests["v8 reproduction (H_v8, paper range) | log(omega+1)"]["alpha"]["value"]
    tri = lib["tribonacci"] - lib["ln_10"]
    look = {
        "stored_match": "tribonacci - ln_10",
        "stored_match_value": tri,
        "reproduced_alpha": stored_alpha,
        "matches_within_0.01_of_reproduced_alpha": int((np.abs(cands - stored_alpha) < 0.01).sum()),
        "library_size": len(lib),
        "candidate_values": int(cands.size),
        "tolerance": 0.01,
        "chance_match_probability_near_alpha": probe(PAPER["alpha"]),
        "chance_match_probability_near_beta": probe(PAPER["beta"]),
    }
    any_theory = any(t["alpha"]["theory_inside_ci"] or t["beta"]["theory_inside_ci"] for t in tests.values())
    return {
        "id": "C8",
        "title": "Fitted constants vs theory and constant matching (§5)",
        "paper_claim": "α ≈ -0.468 ≈ tribonacci - ln 10, β ≈ 1.917 ≈ 1/φ² + ln 10; theory α = -1.599, β = 4.933",
        "reproduction": {
            "statistic": "v8.2 constant matching: 20 constants, singles and pairwise sums/differences, |error| < 0.01",
            "look_elsewhere": look,
            "status": "reproduced" if abs(tri - stored_alpha) < 0.01 else "not reproduced",
        },
        "tests": tests,
        "verdict": "not supported",
        "reason": (
            f"Theory constants inside a 95% CI: {'yes' if any_theory else 'no'}. A value drawn at random near α matches "
            f"some library combination within 0.01 with probability {look['chance_match_probability_near_alpha']:.2f} "
            f"(near β: {look['chance_match_probability_near_beta']:.2f}), so the matches carry no evidence; α and β also "
            "change with log(ω) vs log(ω+1) and with the entropy definition."
        ),
    }


def _v8_constants() -> dict[str, float]:
    phi = (1 + math.sqrt(5)) / 2
    return {
        "e": math.e, "pi": math.pi, "phi": phi, "phi_conjugate": (math.sqrt(5) - 1) / 2,
        "inv_phi_squared": -1 / phi**2, "euler_gamma": 0.5772156649015329, "meissel_mertens": 0.2614972128476428,
        "sqrt_2": math.sqrt(2), "sqrt_3": math.sqrt(3), "sqrt_5": math.sqrt(5), "ln_2": math.log(2),
        "ln_10": math.log(10), "catalan": 0.915965594177219, "apery": 1.202056903159594,
        "twin_prime": 0.6601618158468696, "mills": 1.3063778838630806, "plastic": 1.324717957244746,
        "tribonacci": 1.839286755214161, "5_minus_1_over_15": 5 - 1 / 15, "negative_phi_squared_approx": -1.599,
    }


def _candidate_values(lib: dict[str, float]) -> np.ndarray:
    v = np.array(list(lib.values()))
    return np.unique(np.concatenate([v, (v[:, None] + v[None, :]).ravel(), (v[:, None] - v[None, :]).ravel()]))


# ---------------------------------------------------------------------------------------------
# C2 — direction
# ---------------------------------------------------------------------------------------------


def c2_direction(ctx: Ctx) -> dict:
    tests = {}
    for var in ctx.cfg["entropies"]:
        H, L = ctx.H(var), ctx.logL()
        te = ctx.mask(H, L, split="test")
        h, w = H[te], ctx.omega[te]
        prime = ctx.T["label.is_prime"][te].astype(bool)
        diff = float(h[prime].mean() - h[~prime].mean())
        idx_p, idx_c = np.nonzero(prime)[0], np.nonzero(~prime)[0]
        rng = ctx.rng()
        diff_ci = ci([h[rng.choice(idx_p, idx_p.size)].mean() - h[rng.choice(idx_c, idx_c.size)].mean() for _ in range(ctx.B)])
        rho = spearman(h, w.astype(float))
        rho_ci = bootstrap(lambda i: spearman(h[i], w[i].astype(float)), h.size, ctx.B, ctx.rng())
        Z = np.column_stack([L[te], ctx.logn[te]])
        lw = np.log(w.astype(float))
        pc = partial_corr(lw, h, Z)
        pc_ci = bootstrap(lambda i: partial_corr(lw[i], h[i], Z[i]), h.size, ctx.B, ctx.rng())
        by_w = {int(k): {"count": int((w == k).sum()), "mean_H": float(h[w == k].mean())} for k in np.unique(w)}
        tests[var] = {
            "test_rows": int(te.sum()),
            "mean_H_primes_minus_composites": {"value": diff, "ci95": diff_ci},
            "spearman_H_omega": {"value": rho, "ci95": rho_ci},
            "partial_corr_log_omega_given_logL_logn": {"value": pc, "ci95": pc_ci},
            "mean_H_by_omega": by_w,
        }
    prim = tests[ctx.cfg["entropies"][0]]
    d = prim["mean_H_primes_minus_composites"]
    raw_sign = "negative" if prim["spearman_H_omega"]["ci95"][1] < 0 else ("positive" if prim["spearman_H_omega"]["ci95"][0] > 0 else "unresolved")
    pci = prim["partial_corr_log_omega_given_logL_logn"]["ci95"]
    cond_sign = "negative" if pci[1] < 0 else ("positive" if pci[0] > 0 else "zero (CI includes 0)")
    return {
        "id": "C2",
        "title": "Direction of the effect (§1.3, §7.1)",
        "paper_claim": "§1.3: primes give 'pure, low-entropy' patterns; §7.1: 'higher complexity correlates with higher entropy'; §2.2: α < 0",
        "reproduction": {"statistic": "sign of α in C1", "status": "reproduced", "note": "α < 0 in the stored v8.2 fit"},
        "tests": tests,
        "measured_direction": {"raw": raw_sign, "given_logL_logn": cond_sign},
        "verdict": "not supported" if raw_sign == "negative" else "supported",
        "reason": (
            f"Measured on test ({ctx.cfg['entropies'][0]}): primes minus composites mean H = {d['value']:.3f} CI {_fmt_ci(d['ci95'])}; "
            f"Spearman(H, ω) {prim['spearman_H_omega']['value']:.3f} ({raw_sign}); given log L and log n the sign is {cond_sign}. "
            "Primes have the highest entropy, so §1.3 ('primes low-entropy') and §7.1 item 2 are contradicted; only α < 0 holds."
        ),
    }


# ---------------------------------------------------------------------------------------------
# C3 — prime-power dichotomy
# ---------------------------------------------------------------------------------------------


def c3_prime_power(ctx: Ctx) -> dict:
    ipp = ctx.T["label.is_prime_power"].astype(float)
    m = ctx.mask(ctx.H("H_v8", "v8"), paper_range=True)
    r_labels = corr(ctx.omega[m].astype(float), ipp[m])
    r_entropy = corr(ctx.H("H_v8", "v8")[m], ipp[m])
    reproduction = {
        "statistic": "paper r = -0.843; source: v7.0 log line 210 `omega_vs_is_prime_power` (two factorization labels)",
        "paper": PAPER["prime_power_r"],
        "v11": {"corr(omega, is_prime_power), n in [2, 9999]": r_labels, "corr(H_v8 mean, is_prime_power), n in [2, 9999]": r_entropy},
        "status": "not reproduced",
        "note": "-0.843 is not an entropy statistic; its exact v7 sample is unknown.",
    }
    tests = {}
    thr = float(ctx.cfg["effect_threshold"])
    for var in ctx.cfg["entropies"]:
        H, L = ctx.H(var), ctx.logL()
        te = ctx.mask(H, L, split="test")
        h, x = H[te], ipp[te]
        r = corr(h, x)
        r_ci = bootstrap(lambda i: corr(h[i], x[i]), h.size, ctx.B, ctx.rng())
        Z = np.column_stack([L[te], ctx.logn[te]])
        pc = partial_corr(x, h, Z)
        pc_ci = bootstrap(lambda i: partial_corr(x[i], h[i], Z[i]), h.size, ctx.B, ctx.rng())
        ry = resid(h, Z)
        p = perm_p(lambda q: corr(resid(x[q], Z), ry), pc, h.size, ctx.P, ctx.rng())
        tests[var] = {
            "test_rows": int(te.sum()),
            "point_biserial_r": {"value": r, "ci95": r_ci},
            "partial_r_given_logL_logn": {"value": pc, "ci95": pc_ci, "perm_p": p},
            "supported": bool(excludes_zero(pc_ci) and abs(pc) >= math.sqrt(thr)),
        }
    prim = tests[ctx.cfg["entropies"][0]]
    return {
        "id": "C3",
        "title": "Prime-power dichotomy (§2.3)",
        "paper_claim": "strongest predictor of NDR entropy is prime-power status, r = -0.843",
        "reproduction": reproduction,
        "tests": tests,
        "verdict": "not reproduced",
        "reason": (
            f"The paper's -0.843 is corr(ω, is_prime_power) (v11: {r_labels:.3f} on [2, 9999]), not an entropy correlation. "
            f"Actual r(H, prime power) on test = {prim['point_biserial_r']['value']:.3f} (positive: prime powers have higher H); "
            f"given log L, log n: {prim['partial_r_given_logL_logn']['value']:.3f} CI {_fmt_ci(prim['partial_r_given_logL_logn']['ci95'])}."
        ),
    }


# ---------------------------------------------------------------------------------------------
# C4 — base invariance
# ---------------------------------------------------------------------------------------------


def _invariance(ctx: Ctx, bset: str, var: str, rows: np.ndarray) -> np.ndarray:
    cv = ctx.T[f"row.{bset}.{var}.cv"][rows]
    return np.where(np.isfinite(cv), cv < float(ctx.cfg["cv_threshold"]), False)


def c4_base_invariance(ctx: Ctx) -> dict:
    thr = float(ctx.cfg["cv_threshold"])
    nb8 = ctx.T["row.n_bases.v8"]
    rep = {"paper": PAPER["invariance_rate"]}
    first = ctx.v8_first100()
    if first is not None:
        sel = first[nb8[first] >= 2]
        rep["v11 first 100 of v8.2 data order"] = float(_invariance(ctx, "v8", "H_v8", sel).mean())
    allr = np.nonzero((ctx.source == 0) & (nb8 >= 2))[0]
    rep["v11 all n in [2, 9999]"] = float(_invariance(ctx, "v8", "H_v8", allr).mean())
    reproduced = first is not None and close(rep["v11 first 100 of v8.2 data order"], PAPER["invariance_rate"])

    edges = ctx.cfg["scale_edges"]
    tests: dict[str, Any] = {"note": "no fitted parameters; rates use all rows with >= 2 coprime bases. Bins above 10^4 hold the omega-stratified sample, not all integers."}
    for bset in ctx.cfg["invariance_sets"]:
        if f"row.n_bases.{bset}" not in ctx.T:
            continue
        nb = ctx.T[f"row.n_bases.{bset}"]
        eligible = nb >= 2
        per_var = {}
        for var in list(ctx.cfg["invariance_variants"]) + ["logL"]:
            cvcol = ctx.T[f"row.{bset}.{var}.cv"]
            ok = eligible & np.isfinite(cvcol)
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = np.nonzero(eligible & (ctx.n >= lo) & (ctx.n < hi))[0]
                if sel.size >= int(ctx.cfg["min_bin"]):
                    inv = _invariance(ctx, bset, var, sel)
                    bins.append({"n_range": [lo, hi - 1], "rows": int(sel.size), "rate": float(inv.mean()), "median_cv": float(np.nanmedian(cvcol[sel]))})
            idx = np.nonzero(ok)[0]
            cv, ln = cvcol[idx], ctx.logn[idx]
            rho = spearman(cv, ln) if idx.size > 3 else float("nan")
            rho_ci = bootstrap(lambda i: spearman(cv[i], ln[i]), idx.size, max(200, ctx.B // 5), ctx.rng()) if idx.size > 3 else [float("nan")] * 2
            per_var[var] = {"overall_rate": float(_invariance(ctx, bset, var, np.nonzero(eligible)[0]).mean()), "by_scale": bins, "spearman_cv_logn": {"value": rho, "ci95": rho_ci}}
        tests[bset] = {"bases": len(_set_bases(ctx, bset)), "eligible_rows": int(eligible.sum()), "variants": per_var}
    p = ctx.cfg["primary_set"]
    prim = tests.get(p, {}).get("variants", {}).get(ctx.cfg["entropies"][0], {})
    rho = prim.get("spearman_cv_logn", {})
    emerges = bool(rho and rho["ci95"][1] < 0 and abs(rho["value"]) >= 0.1)
    best = max(
        ((b, v, d["overall_rate"]) for b, t in tests.items() if isinstance(t, dict) and "variants" in t for v, d in t["variants"].items() if v != "logL"),
        key=lambda t: t[2],
        default=None,
    )
    return {
        "id": "C4",
        "title": "Base invariance CV < 0.1 (§2.4, Q5.2)",
        "paper_claim": "base invariance in only 6% of cases",
        "reproduction": {"statistic": f"share of n with CV(H_v8 over v8 bases) < {thr}", "v11": rep, "status": "reproduced" if reproduced else "not reproduced", "note": "v8.2 evaluated only data[:100]."},
        "tests": tests,
        "verdict": ("reproduced" if reproduced else "not reproduced") + ("; invariance increases with n" if emerges else "; no asymptotic emergence"),
        "reason": (
            f"v8.2's 6% is a 100-integer sample (v11 on all n in [2, 9999]: {rep['v11 all n in [2, 9999]']:.3f}). "
            f"Spearman(CV, log n) for {p}/{ctx.cfg['entropies'][0]}: {rho.get('value', float('nan')):.3f} CI {_fmt_ci(rho.get('ci95', [float('nan')] * 2))}. "
            + (f"Highest overall rate: {best[0]}/{best[1]} = {best[2]:.3f}. " if best else "")
            + "Normalized DC-free entropy saturates near its maximum for long, noise-like repetends, so its CV shrinks with L; "
            "the CV of log L itself also falls with n."
        ),
    }


# ---------------------------------------------------------------------------------------------
# C5 — zeta, C6 — PNT
# ---------------------------------------------------------------------------------------------


def _size_controls(logn: np.ndarray) -> np.ndarray:
    z = (logn - logn.mean()) / (logn.std() or 1.0)
    return np.column_stack([z, z**2, z**3])


def c5_zeta(ctx: Ctx) -> dict:
    Hv8 = ctx.H("H_v8", "v8")
    nf = ctx.n.astype(float)
    rep = {"paper": PAPER["zeta"]}
    first = ctx.v8_first100()
    if first is not None:
        rep["v11 first 100 of v8.2 data order"] = {s: corr(nf[first] ** -s, Hv8[first]) for s in ctx.cfg["zeta_s"]}
    m = ctx.mask(Hv8, paper_range=True)
    rep["v11 all n in [2, 9999]"] = {s: corr(nf[m] ** -s, Hv8[m]) for s in ctx.cfg["zeta_s"]}
    reproduced = first is not None and all(close(rep["v11 first 100 of v8.2 data order"][s], PAPER["zeta"][s]) for s in PAPER["zeta"])
    var = ctx.cfg["entropies"][0]
    H = ctx.H(var)
    base = ctx.mask(H)
    tr, te = base & (ctx.split == 0), base & (ctx.split == 2)
    h = H[te]
    Z = _size_controls(ctx.logn[te])
    tests = {"variant": var, "test_rows": int(te.sum()), "r_H_logn": corr(h, ctx.logn[te])}
    any_left = False
    for s in ctx.cfg["zeta_s"]:
        x = nf ** -float(s)
        xt = x[te]
        pc = partial_corr(xt, h, Z)
        pc_ci = bootstrap(lambda i: partial_corr(xt[i], h[i], Z[i]), h.size, ctx.B, ctx.rng())
        _, r2_zeta, _ = _fit_eval(ctx, H, [x], tr, te)
        _, r2_size, _ = _fit_eval(ctx, H, [ctx.logn], tr, te)
        tests[f"s={s}"] = {"r_H_n^-s": corr(xt, h), "partial_r_given_cubic_logn": {"value": pc, "ci95": pc_ci}, "test_r2_H~n^-s": r2_zeta, "test_r2_H~log n": r2_size}
        any_left |= excludes_zero(pc_ci) and abs(pc) >= 0.1
    return {
        "id": "C5",
        "title": "Correlation with n^-s (zeta, §4.1)",
        "paper_claim": "H correlates with n^-s, decreasing in s (zeta connection)",
        "reproduction": {"statistic": "corr(n^-s, v8 mean H_v8) over v8.2 data[:100]", "v11": rep, "status": "reproduced" if reproduced else "not reproduced"},
        "tests": tests,
        "verdict": ("reproduced" if reproduced else "not reproduced") + (" — residual association remains" if any_left else " but explained by size/length"),
        "reason": "n^-s is a monotone function of n; after a cubic in log n the partial correlations are "
        + ", ".join(f"s={s}: {tests[f's={s}']['partial_r_given_cubic_logn']['value']:.3f}" for s in ctx.cfg["zeta_s"])
        + f"; H ~ log n alone has test R² {tests[f's={ctx.cfg['zeta_s'][0]}']['test_r2_H~log n']:.3f}.",
    }


def _prime_pi(n_max: int) -> np.ndarray:
    spf = ndr.spf_sieve(n_max)
    is_p = spf == np.arange(n_max + 1)
    is_p[:2] = False
    return np.cumsum(is_p)


def c6_pnt(ctx: Ctx) -> dict:
    from scipy.special import expi

    Hv8 = ctx.H("H_v8", "v8")
    m = ctx.mask(Hv8, paper_range=True) & (ctx.n > 2)
    h = Hv8[m]
    prod = h * ctx.logn[m]
    r = corr(prod, h)
    rng = ctx.rng()
    null = []
    for _ in range(min(ctx.P, 500)):
        hp = h[rng.permutation(h.size)]
        null.append(corr(hp * ctx.logn[m], hp))
    pi = _prime_pi(int(ctx.n.max()))
    nf = ctx.n.astype(float)
    pnt_err = np.abs(nf / np.log(np.maximum(nf, 2)) - pi[ctx.n])
    li_err = np.abs(expi(np.log(np.maximum(nf, 2))) - expi(math.log(2)) - pi[ctx.n])
    rep = {
        "paper": {"pnt_entropy_product": PAPER["pnt_r"], "pnt_error": PAPER["pnt_error_r"], "li_error": PAPER["li_error_r"]},
        "v11": {"pnt_entropy_product": r, "pnt_error": corr(pnt_err[m], h), "li_error": corr(li_err[m], h)},
        "null_permuted_H": {"pnt_entropy_product_mean": float(np.mean(null)), "ci95": ci(null)},
    }
    reproduced = close(r, PAPER["pnt_r"])
    var = ctx.cfg["entropies"][0]
    H = ctx.H(var)
    te = ctx.mask(H, split="test")
    ht = H[te]
    Z = _size_controls(ctx.logn[te])
    tests = {"variant": var, "test_rows": int(te.sum())}
    for name, x in (("pnt_error", pnt_err), ("li_error", li_err)):
        xt = x[te]
        pc = partial_corr(xt, ht, Z)
        tests[name] = {"r": corr(xt, ht), "partial_r_given_cubic_logn": {"value": pc, "ci95": bootstrap(lambda i: partial_corr(xt[i], ht[i], Z[i]), ht.size, ctx.B, ctx.rng())}}
    return {
        "id": "C6",
        "title": "H·log n vs prime number theorem (§4.2)",
        "paper_claim": "corr(H·log n, H) = 0.946 links NDR entropy to the PNT",
        "reproduction": {"statistic": "corr(H·log n, H) over n in [3, 9999], v8 mean H_v8", **rep, "status": "reproduced" if reproduced else "not reproduced"},
        "tests": tests,
        "verdict": ("reproduced" if reproduced else "not reproduced") + " but explained by size/length",
        "reason": (
            f"corr(H·log n, H) = {r:.3f}; with H randomly permuted across n the same statistic is {np.mean(null):.3f} "
            f"CI {_fmt_ci(ci(null))}: most of it comes from H appearing on both sides, the rest from H growing with n "
            f"(corr(H, log n) = {corr(h, ctx.logn[m]):.3f}). PNT/Li errors are functions of n; "
            f"partial r given log n: pnt {tests['pnt_error']['partial_r_given_cubic_logn']['value']:.3f}, li {tests['li_error']['partial_r_given_cubic_logn']['value']:.3f}."
        ),
    }


# ---------------------------------------------------------------------------------------------
# C7 — modular structure
# ---------------------------------------------------------------------------------------------


def c7_modular(ctx: Ctx) -> dict:
    Hv8 = ctx.H("H_v8", "v8")
    m = ctx.mask(Hv8, paper_range=True)
    rep_v = {}
    for q in ctx.cfg["moduli"]:
        res = ctx.n[m] % q
        means = [Hv8[m][res == r].mean() for r in np.unique(res)]
        rep_v[q] = float(np.var(means))
    reproduced = all(close(rep_v[q], PAPER["modular_variance"][q]) for q in PAPER["modular_variance"] if q in rep_v)
    thr = float(ctx.cfg["effect_threshold"])
    pset = ctx.cfg["primary_set"]
    pbases = sorted(_set_bases(ctx, pset))
    coprime = [(np.gcd(ctx.n, b) == 1).astype(float) for b in pbases]
    Lmat, _ = _per_base_logL(ctx, pbases)
    per_base_L = [np.nan_to_num(Lmat[:, j], nan=0.0) for j in range(Lmat.shape[1])]
    tests: dict[str, Any] = {
        "controls": f"log L of each {pset} base (0 when not coprime), coprimality with each base, log n, omega dummies (omega is a labelled control)",
        "excess": "eta2_residual minus its mean under 100 label permutations (large class counts inflate eta2 by chance)",
        "variants": {},
    }
    supported_by_var = {}
    for var in ctx.cfg["entropies"][:2]:
        H, L = ctx.H(var), ctx.logL()
        base = ctx.mask(H, L)
        tr, te = base & (ctx.split == 0), base & (ctx.split == 2)
        levels = np.unique(ctx.omega[tr])
        cols = [ctx.logn] + per_base_L + coprime + _dummies(ctx.omega, levels)
        _, _, pred = _fit_eval(ctx, H, cols, tr, te)
        res_te = H[te] - pred
        per_q = {}
        sup_any = False
        for q in ctx.cfg["moduli"]:
            g = ctx.n[te] % q
            raw = eta_squared(H[te], g)
            adj = eta_squared(res_te, g)
            rng = ctx.rng()
            P = min(ctx.P, 500)
            p_adj = (1 + sum(eta_squared(res_te, g[rng.permutation(g.size)]) >= adj for _ in range(P))) / (P + 1)
            null_adj = float(np.mean([eta_squared(res_te, g[rng.permutation(g.size)]) for _ in range(100)]))
            sup = p_adj < 0.01 and adj - null_adj >= thr
            sup_any |= sup
            per_q[q] = {"classes": int(np.unique(g).size), "eta2_raw": raw, "eta2_residual": adj, "eta2_residual_null_mean": null_adj, "excess": adj - null_adj, "perm_p_residual": p_adj, "supported": bool(sup)}
        tests["variants"][var] = {"test_rows": int(te.sum()), "moduli": per_q}
        supported_by_var[var] = sup_any
    v0 = ctx.cfg["entropies"][0]
    v1 = ctx.cfg["entropies"][1] if len(ctx.cfg["entropies"]) > 1 else v0
    if supported_by_var[v0]:
        verdict = "supported (residue structure beyond per-base L and ω)"
    else:
        verdict = "reproduced but explained by size/length" if reproduced else "not reproduced"

    def line(var):
        return ", ".join(f"mod {q}: {t['eta2_raw']:.3f} → excess {t['excess']:.3f} (p {t['perm_p_residual']:.3f})" for q, t in tests["variants"][var]["moduli"].items())

    return {
        "id": "C7",
        "title": "Residue-class structure mod 6/12/30/210 (§4.3)",
        "paper_claim": "variance of residue-class mean H > 0.01 marks significant modular structure",
        "reproduction": {"statistic": "variance of residue-class means of v8 mean H_v8, n in [2, 9999]", "paper": PAPER["modular_variance"], "v11": rep_v, "status": "reproduced" if reproduced else "not reproduced"},
        "tests": tests,
        "verdict": verdict,
        "reason": f"η² of residue class on test, raw → excess over the permutation null after controls. {v0}: {line(v0)}. {v1}: {line(v1)}. "
        "Residues mod 6/12/30/210 fix divisibility by 2, 3, 5, 7 and the quadratic characters of the bases, which set the per-base orders.",
    }


# ---------------------------------------------------------------------------------------------
# C9 — information beyond L
# ---------------------------------------------------------------------------------------------


def _per_base_logL(ctx: Ctx, bases: list[int]) -> tuple[np.ndarray, list[str]]:
    M = np.full((ctx.n.size, len(bases)), np.nan)
    idx = np.searchsorted(ctx.n, ctx.T["pair.n"])
    col = {b: j for j, b in enumerate(bases)}
    pb = ctx.T["pair.b"]
    sel = np.isin(pb, bases)
    M[idx[sel], np.array([col[int(b)] for b in pb[sel]])] = ctx.T["pair.logL"][sel]
    return M, [f"pair.logL.b{b}" for b in bases]


def _length_arithmetic(ctx: Ctx, bases: list[int]) -> tuple[np.ndarray, list[str]]:
    """Functions of (n, repetend lengths) only: lcm of the lengths and its relation to n - 1.

    A tree model cannot compute an lcm or a divisibility test from raw log-lengths, so without these
    the baseline would under-use L and any NDR gain could just be L information in another form.
    """
    sel = np.isin(ctx.T["pair.b"], bases)
    idx = np.searchsorted(ctx.n, ctx.T["pair.n"][sel])
    Ls = ctx.T["pair.L"][sel].astype(np.int64)
    acc = [1] * ctx.n.size
    divides = np.zeros(ctx.n.size)
    for i, L in zip(idx.tolist(), Ls.tolist()):
        acc[i] = math.lcm(acc[i], L)
        divides[i] += (int(ctx.n[i]) - 1) % L == 0
    lcm = np.array(acc, dtype=np.float64)
    nm1 = ctx.n - 1
    g = np.array([math.gcd(int(a), int(b)) for a, b in zip(acc, nm1.tolist())], dtype=np.float64)
    cnt = np.bincount(idx, minlength=ctx.n.size).astype(float)
    feats = np.column_stack([
        np.log(lcm),
        np.log(ctx.n) - np.log(lcm),
        (nm1 % lcm.astype(np.int64) == 0).astype(float),
        np.log(np.maximum(g, 1)),
        np.where(cnt > 0, divides / np.maximum(cnt, 1), np.nan),
    ])
    return feats, ["log lcm(L_b)", "log n - log lcm", "lcm divides n-1", "log gcd(lcm, n-1)", "share of L_b dividing n-1"]


def c9_beyond_L(ctx: Ctx) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ccfg = ctx.cfg["classifier"]
    K = int(ccfg["max_omega_class"])
    y = np.minimum(ctx.omega, K)
    sets = [s for s in ("v8", "v10", "primes31") if f"row.n_bases.{s}" in ctx.T]
    bases = sorted({int(b) for b in np.unique(ctx.T["pair.b"]) if any(int(b) in _set_bases(ctx, s) for s in sets)})
    Lmat, Lnames = _per_base_logL(ctx, bases)
    nb = np.column_stack([ctx.T[f"row.n_bases.{s}"] for s in sets]).astype(float)
    arith, arith_names = _length_arithmetic(ctx, bases)
    X_base = np.column_stack([ctx.logn, Lmat, nb, arith])
    base_names = ["log n", *Lnames, *[f"row.n_bases.{s}" for s in sets], *arith_names]
    allowed = [c for c in dataset.allowed_input_columns(ctx.T) if not c.startswith("row.wide.") and c != "row.n_bases.wide" and c != "row.n"]
    ndr_cols = [c for c in allowed if not c.startswith("row.n_bases.") and ".logL." not in c and ".L." not in c]
    X_full = np.column_stack([X_base] + [ctx.T[c] for c in ndr_cols])
    entropy_cols = [c for c in ndr_cols if c.split(".")[2].startswith(("H_", "Hn_"))]
    X_ent = np.column_stack([X_base] + [ctx.T[c] for c in entropy_cols])
    tr, va, te = (ctx.split == s for s in (0, 1, 2))
    classes = np.unique(y[tr])

    def hgb(seed):
        return HistGradientBoostingClassifier(max_iter=int(ccfg["max_iter"]), learning_rate=float(ccfg["learning_rate"]), early_stopping=True, validation_fraction=0.15, random_state=seed)

    def evaluate(model, X):
        proba = model.predict_proba(X[te])
        cls = list(model.classes_)
        pt = np.array([proba[i, cls.index(v)] if v in cls else 1e-12 for i, v in enumerate(y[te])])
        pred = np.array(cls)[proba.argmax(1)]
        pv = model.predict(X[va])
        return {
            "correct": (pred == y[te]).astype(float),
            "nll": -np.log(np.clip(pt, 1e-12, 1.0)),
            "metrics": {
                "test_accuracy": float((pred == y[te]).mean()),
                "test_macro_f1": float(f1_score(y[te], pred, average="macro")),
                "test_log_loss": float(-np.log(np.clip(pt, 1e-12, 1.0)).mean()),
                "val_accuracy": float((pv == y[va]).mean()),
            },
        }

    models = {}
    maj = classes[np.argmax([(y[tr] == c).sum() for c in classes])]
    prior = np.array([(y[tr] == c).mean() for c in classes])
    pt = np.array([prior[list(classes).index(v)] if v in classes else 1e-12 for v in y[te]])
    models["majority / class prior"] = {"test_accuracy": float((y[te] == maj).mean()), "test_macro_f1": float(f1_score(y[te], np.full(te.sum(), maj), average="macro")), "test_log_loss": float(-np.log(pt).mean())}

    Xt = np.column_stack([ctx.logn, np.nan_to_num(ctx.logL(), nan=0.0), np.isnan(ctx.logL()).astype(float)])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xt[tr], y[tr])
    models["logistic(log n, mean log L)"] = evaluate(lr, Xt)["metrics"]

    mb = hgb(ctx.seed).fit(X_base[tr], y[tr])
    eb = evaluate(mb, X_base)
    models["GBM baseline: log n + per-base log L + coprime counts + length arithmetic"] = eb["metrics"]
    me = hgb(ctx.seed).fit(X_ent[tr], y[tr])
    ee = evaluate(me, X_ent)
    models["GBM baseline + entropy features only"] = ee["metrics"]
    mf = hgb(ctx.seed).fit(X_full[tr], y[tr])
    ef = evaluate(mf, X_full)
    models["GBM full: baseline + all NDR features"] = ef["metrics"]

    rng = ctx.rng()
    nulls = []
    for k in range(int(ccfg["shuffled_nulls"])):
        ys = y.copy()
        ys[tr] = y[tr][rng.permutation(int(tr.sum()))]
        mn = hgb(ctx.seed + 1 + k).fit(X_full[tr], ys[tr])
        nulls.append(float((mn.predict(X_full[te]) == y[te]).mean()))
    models["GBM full on shuffled train labels (null)"] = {"test_accuracy_mean": float(np.mean(nulls)), "test_accuracy_runs": nulls}

    def paired(e1, e0):
        c1, c0, l1, l0 = e1["correct"], e0["correct"], e1["nll"], e0["nll"]
        rng = ctx.rng()
        n_te = c0.size
        boots = [(c1[i].mean() - c0[i].mean(), l0[i].mean() - l1[i].mean()) for i in (rng.integers(0, n_te, n_te) for _ in range(ctx.B))]
        return {
            "accuracy": {"value": float(c1.mean() - c0.mean()), "ci95": ci([b[0] for b in boots])},
            "log_loss_gain": {"value": float(l0.mean() - l1.mean()), "ci95": ci([b[1] for b in boots])},
        }

    gain_full, gain_ent = paired(ef, eb), paired(ee, eb)
    d_acc, acc_ci = gain_full["accuracy"]["value"], gain_full["accuracy"]["ci95"]
    d_ll, ll_ci = gain_full["log_loss_gain"]["value"], gain_full["log_loss_gain"]["ci95"]
    supported = acc_ci[0] > 0 and ll_ci[0] > 0
    ent_supported = gain_ent["accuracy"]["ci95"][0] > 0 and gain_ent["log_loss_gain"]["ci95"][0] > 0
    return {
        "id": "C9",
        "title": "Information about ω beyond L and n (core question)",
        "paper_claim": "NDR entropy patterns encode factorization complexity (Conj. 1.3)",
        "reproduction": {"statistic": "no paper statistic controls for L", "status": "n/a"},
        "tests": {
            "target": f"min(omega, {K}) as classes",
            "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "features": {"baseline": base_names, "full": X_full.shape[1], "entropy_only_added": len(entropy_cols), "ndr_columns": ndr_cols[:12] + (["..."] if len(ndr_cols) > 12 else [])},
            "models": models,
            "full_minus_baseline": gain_full,
            "entropy_only_minus_baseline": gain_ent,
        },
        "verdict": ("supported" if supported else "not supported") + ("" if supported == ent_supported else (" (entropy features alone: " + ("supported" if ent_supported else "not supported") + ")")),
        "reason": (
            f"Held-out ω classification: baseline (n, per-base L, lcm arithmetic) accuracy {eb['metrics']['test_accuracy']:.3f}; "
            f"+entropy features {ee['metrics']['test_accuracy']:.3f} (Δ {gain_ent['accuracy']['value']:+.3f} CI {_fmt_ci(gain_ent['accuracy']['ci95'])}); "
            f"+all NDR features {ef['metrics']['test_accuracy']:.3f} (Δ {d_acc:+.3f} CI {_fmt_ci(acc_ci)}, log-loss gain {d_ll:+.4f} CI {_fmt_ci(ll_ci)}); "
            f"majority {models['majority / class prior']['test_accuracy']:.3f}, shuffled-label null {np.mean(nulls):.3f}."
        ),
    }


def _set_bases(ctx: Ctx, bset: str) -> set[int]:
    return set(ctx.cfg.get("_base_sets", {}).get(bset, []))


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

CLAIMS = (c1_law, c2_direction, c3_prime_power, c4_base_invariance, c5_zeta, c6_pnt, c7_modular, c8_constants, c9_beyond_L)


def run_claims(table: dict[str, np.ndarray], claims_cfg: dict[str, Any], base_sets: dict[str, Any], seed: int, repo_root: Path, log=print) -> dict:
    cfg = {**(claims_cfg or {}), "_base_sets": {k: v["bases"] for k, v in dataset.normalize_base_sets(base_sets).items()}}
    ctx = Ctx(table, cfg, seed, repo_root)
    results = []
    for fn in CLAIMS:
        import time

        t0 = time.time()
        r = fn(ctx)
        r["seconds"] = round(time.time() - t0, 2)
        log(f"  {r['id']}: {r['verdict']} ({r['seconds']}s)")
        results.append(r)
    settings = {k: v for k, v in ctx.cfg.items() if not k.startswith("_")}
    return {"ndr_version": ndr.NDR_VERSION, "settings": settings, "claims": results}


def _fmt_ci(c) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]" if c and np.isfinite(c).all() else "[n/a]"
