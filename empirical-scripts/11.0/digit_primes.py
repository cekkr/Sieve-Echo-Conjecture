"""ROADMAP step 22: Evolvo on the digits of n itself, on the repetend lengths of 1/n, and on prime-density targets.

Every case runs through ``prime_order_formulas.run_cases`` (PyTorch policy guide, size-separated
train/validation/test ranges, validation selection, one test read, replayable GFSL slots). Each case
lists its inputs; the ``…_density`` variants add **prime-density predictions** as Evolvo input variables,
so a formula can start from the probabilistic model and correct it:

* ``Cramér density 1/log n`` — the probability that an integer near n is prime (prime number theorem);
* ``sieve density W(n)`` — ``[gcd(n, P_S) = 1] · ∏_{p∈S} p/(p − 1) / log n`` for the primes
  S = {2, 3, 5, 7, 11, 13, 37} that base-10 digit sums can see: 2, 5 divide 10 (last digit), 3 divides 9
  (digit sum), 11 divides 11 (alternating sum), 37 divides 999 (sum of 3-digit blocks), 7, 11, 13 divide
  1001 (alternating sum of 3-digit blocks). Given coprimality to S, the PNT density rises by ∏ p/(p − 1);
* ``Hardy–Ramanujan mean`` — ``log log n + B₁`` (Mertens' constant B₁), the mean of ω near n, and the
  sieve version ``Σ_{p∈S} [p | n] + log log n + B₁ − Σ_{p∈S} 1/p``;
* ``li(x)·log x / x`` — the prime count predicted by the integrated density, as a ratio, and
  ``li(√x)·log x / x`` — the density of prime squares, the second term of Riemann's R(x) = Σ μ(k)/k · li(x^{1/k}).

Cases (``CASES``):

* ``prime_digits`` / ``prime_digits_density`` — is_prime(n) from base-10 digit statistics of n (no 1/n).
  The selected formula is a probability: scored by Brier score, log loss, AUC and the predicted number of
  primes in the test range. Exact limit: the digit statistics are divisibility rules (b^k ≡ ±1), so they
  only reveal n modulo the primes in S; the sieve density W(n) is the matching probabilistic model.
* ``prime_repetend`` — is_prime(n) for gcd(n, 10) = 1 from the repetend lengths L₂(n), L₁₀(n) of 1/n and n.
  Exact limit: [L_b | n − 1] is Fermat's test in base b; the composites it accepts are the Fermat
  pseudoprimes, counted exactly in the test range.
* ``omega_digits`` / ``omega_digits_density`` — ω(n) from the same digit statistics.
* ``prime_count`` / ``prime_count_density`` — the ratio π(x)·log x / x at log-spaced x from log x; train
  x ≤ 10⁶, validation ≤ 10⁸, test up to 10^``pi_max_exponent``. π(x) is counted by the sieve through
  ``prime_limit`` and by SymPy's ``primepi`` everywhere (they must agree on the overlap).
* ``prime_gap`` — the gap to the next prime from log p, the density 1/log p and p mod 6 (Cramér: mean log p).
* ``last_digit_repeat`` — [the next prime ends in the same decimal digit as p] from log p, 1/log p and the
  last digit (Lemke Oliver–Soundararajan bias; equidistribution would give 1/4).

Factorization-derived quantities (ω, λ(n), spf) are targets or used only to compute the orders exactly
(the order is a function of (n, b)); they never enter the inputs.
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import artin
import failures
import ndr
import prime_order_formulas
import search_formulas

MERTENS_B1 = 0.26149721284764278
LEGENDRE_A = 1.08366
DIGIT_PRIMES = (2, 3, 5, 7, 11, 13, 37)
SIEVE_FACTOR = float(np.prod([p / (p - 1) for p in DIGIT_PRIMES]))

INTEGER_OPS = ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "ABS", "LOG"]
COUNT_OPS = ["ADD", "SUB", "MUL", "DIV", "POW", "LOG", "SQRT"]
GAP_OPS = ["ADD", "SUB", "MUL", "DIV", "MOD", "LOG", "ABS"]

CASES = ["prime_digits", "prime_digits_density", "prime_repetend", "omega_digits", "omega_digits_density",
         "prime_count", "prime_count_density", "prime_gap", "last_digit_repeat"]

DEFAULTS: Dict[str, Any] = {
    "n_min": 1001, "n_max": 2_000_000, "split_at": [400_000, 1_000_000],
    "prime_limit": 100_000_000, "prime_split_at": [1_000_000, 10_000_000],
    "pi_min_exponent": 3, "pi_max_exponent": 12, "pi_per_decade": 10, "pi_split_at": [1_000_000, 100_000_000],
    "cases": CASES, "gbm_rows": 200_000, "sympy_order_sample": 3000, "order_workers": 7,
    "seed_offsets": [0, 1, 2, 3, 4, 5], "population": 200, "generations": 120, "initial_instructions": 8,
    "max_instructions": 32, "max_effective_instructions": 16, "workers": 6,
}


# ---------------------------------------------------------------------------------------------
# Digit statistics of n and prime-density predictions
# ---------------------------------------------------------------------------------------------


def digit_statistics(n: np.ndarray, base: int = 10) -> Dict[str, np.ndarray]:
    """Sum, alternating sum (units digit +) and count of the base-``base`` digits of n."""
    m = np.asarray(n, dtype=np.int64).copy()
    s = np.zeros_like(m)
    alt = np.zeros_like(m)
    count = np.zeros_like(m)
    sign = 1
    while (m > 0).any():
        d = m % base
        s += d
        alt += sign * d
        count += m > 0
        m //= base
        sign = -sign
    return {"sum": s, "alternating": alt, "count": count}


def digit_inputs(n: np.ndarray) -> Dict[str, np.ndarray]:
    d10 = digit_statistics(n, 10)
    d1000 = digit_statistics(n, 1000)
    return {
        "last digit": n % 10,
        "digit sum": d10["sum"],
        "alternating digit sum": d10["alternating"],
        "3-digit block sum": d1000["sum"],
        "alternating 3-digit block sum": d1000["alternating"],
        "digit count": d10["count"],
        "log n": np.log(n.astype(float)),
    }


def coprime_to(n: np.ndarray, primes=DIGIT_PRIMES) -> np.ndarray:
    return np.all([n % p != 0 for p in primes], axis=0)


def cramer_density(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.log(np.asarray(x, dtype=float))


def sieve_density(n: np.ndarray) -> np.ndarray:
    return coprime_to(n) * SIEVE_FACTOR * cramer_density(n)


def hardy_ramanujan_mean(n: np.ndarray) -> np.ndarray:
    return np.log(np.log(n.astype(float))) + MERTENS_B1


def sieve_omega_mean(n: np.ndarray) -> np.ndarray:
    seen = np.sum([(n % p == 0) for p in DIGIT_PRIMES], axis=0)
    return seen + hardy_ramanujan_mean(n) - sum(1 / p for p in DIGIT_PRIMES)


def li_ratio(x: np.ndarray) -> np.ndarray:
    return np.array([artin.li(float(v)) * math.log(v) / v for v in x])


# ---------------------------------------------------------------------------------------------
# Exact data
# ---------------------------------------------------------------------------------------------


def omega_sieve(n_max: int, primes: np.ndarray) -> np.ndarray:
    om = np.zeros(n_max + 1, dtype=np.int8)
    for p in primes[primes <= n_max].tolist():
        om[p::p] += 1
    return om


_SPF: Dict[str, np.ndarray] = {}


def _init_orders(spf):
    _SPF["spf"] = spf


def _orders_chunk(ns: List[int]) -> np.ndarray:
    spf = _SPF["spf"]
    out = np.zeros((len(ns), 2), dtype=np.int64)
    for i, n in enumerate(ns):
        lam, lp = ndr.order_context(n, spf)
        out[i] = (ndr.multiplicative_order(2, n, lam, lp), ndr.multiplicative_order(10, n, lam, lp))
    return out


def repetend_lengths(ns: np.ndarray, n_max: int, workers: int) -> np.ndarray:
    """(L₂, L₁₀) = (ord_n 2, ord_n 10) for n coprime to 10, exact from λ(n)."""
    spf = ndr.spf_sieve(n_max)
    lst = [int(v) for v in ns.tolist()]
    size = max(5000, len(lst) // (max(1, workers) * 8) + 1)
    chunks = [lst[i:i + size] for i in range(0, len(lst), size)]
    if workers <= 1:
        _init_orders(spf)
        parts = [_orders_chunk(c) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_orders, initargs=(spf,)) as pool:
            parts = list(pool.map(_orders_chunk, chunks))
    return np.concatenate(parts)


def check_orders(ns: np.ndarray, L: np.ndarray, sample: int, seed: int) -> Dict[str, Any]:
    """b^L ≡ 1 (mod n) for every row; minimality against SymPy on a seeded sample."""
    from sympy.ntheory import n_order

    for j, b in enumerate((2, 10)):
        bad = [int(n) for n, ell in zip(ns.tolist(), L[:, j].tolist()) if pow(b, int(ell), int(n)) != 1]
        if bad:
            raise ArithmeticError(f"repetend length is not a period of {b} for n = {bad[:5]}")
    idx = np.random.default_rng([seed, 2201]).choice(ns.size, size=min(sample, ns.size), replace=False)
    for i in idx.tolist():
        n = int(ns[i])
        if (n_order(2, n), n_order(10, n)) != (int(L[i, 0]), int(L[i, 1])):
            raise ArithmeticError(f"repetend length differs from SymPy at n = {n}")
    return {"rows": int(ns.size), "period_checked_rows": int(ns.size), "sympy_minimality_sample": int(idx.size)}


def prime_counts(xs: np.ndarray, primes: np.ndarray, limit: int) -> tuple[np.ndarray, Dict[str, Any]]:
    """π(x) from SymPy for every x, checked against the sieve wherever x ≤ limit."""
    from sympy import primepi

    sym = np.array([int(primepi(int(x))) for x in xs.tolist()], dtype=np.int64)
    inside = xs <= limit
    sieve = np.searchsorted(primes, xs[inside], side="right")
    if not np.array_equal(sieve, sym[inside]):
        raise ArithmeticError("sieve and SymPy prime counts disagree")
    return sym, {"points": int(xs.size), "sieve_checked_points": int(inside.sum()), "largest_x": int(xs.max())}


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def _ones(n):
    return np.ones(len(n))


def build(cfg: Dict[str, Any], seed: int, log=print) -> tuple[dict, dict, dict]:
    """Cases (for ``run_cases``), baseline predictions per case and data checks."""
    limit = int(cfg["prime_limit"])
    n_min, n_max = int(cfg["n_min"]), int(cfg["n_max"])
    log(f"  sieving primes ≤ {limit:,}")
    P = failures.sieve_primes(2, max(limit, n_max))
    isp = np.zeros(n_max + 1, dtype=bool)
    isp[P[P <= n_max]] = True
    n = np.arange(n_min, n_max + 1, dtype=np.int64)
    y_prime = isp[n]
    om = omega_sieve(n_max, P)[n].astype(float)
    dig = digit_inputs(n)
    digit_names = list(dig)
    Xd = np.column_stack([dig[k] for k in digit_names] + [_ones(n)])
    dens = {"Cramér density 1/log n": cramer_density(n), "sieve density W(n)": sieve_density(n)}
    odens = {"Hardy–Ramanujan mean log log n + B1": hardy_ramanujan_mean(n), "sieve mean of ω": sieve_omega_mean(n)}
    checks: Dict[str, Any] = {"integers": int(n.size), "primes_in_integer_range": int(y_prime.sum())}
    # digit rules are divisibility rules: verify them on every row
    checks["digit_rules"] = all([
        np.array_equal(dig["digit sum"] % 3, n % 3), np.array_equal(dig["alternating digit sum"] % 11, n % 11),
        np.array_equal(dig["3-digit block sum"] % 37, n % 37),
        all(np.array_equal(dig["alternating 3-digit block sum"] % q, n % q) for q in (7, 11, 13)),
        np.array_equal(dig["last digit"] % 2, n % 2), np.array_equal(dig["last digit"] % 5, n % 5)])
    if not checks["digit_rules"]:
        raise ArithmeticError("digit statistics do not reproduce the divisibility rules")
    from sympy import primefactors

    rng = np.random.default_rng([seed, 2202])
    for i in rng.choice(n.size, 3000, replace=False).tolist():
        if len(primefactors(int(n[i]))) != int(om[i]):
            raise ArithmeticError(f"omega sieve differs from SymPy at {n[i]}")
    checks["omega_sympy_sample"] = 3000

    cases: Dict[str, dict] = {}
    base: Dict[str, Dict[str, np.ndarray]] = {}
    sp = tuple(cfg["split_at"])
    cases["prime_digits"] = {"p": n, "X": Xd, "y": y_prime, "names": digit_names + ["1"], "operations": INTEGER_OPS, "split_at": sp}
    cases["prime_digits_density"] = {**cases["prime_digits"], "X": np.column_stack([Xd] + list(dens.values())),
                                     "names": digit_names + ["1"] + list(dens)}
    base_prime = {"Cramér 1/log n": dens["Cramér density 1/log n"], "sieve density W(n)": dens["sieve density W(n)"]}
    base["prime_digits"] = base["prime_digits_density"] = base_prime
    cases["omega_digits"] = {"p": n, "X": Xd, "y": om, "names": digit_names + ["1"], "operations": INTEGER_OPS,
                             "split_at": sp, "regression": True}
    cases["omega_digits_density"] = {**cases["omega_digits"], "X": np.column_stack([Xd] + list(odens.values())),
                                     "names": digit_names + ["1"] + list(odens)}
    base["omega_digits"] = base["omega_digits_density"] = {"Hardy–Ramanujan log log n + B1": odens["Hardy–Ramanujan mean log log n + B1"],
                                                           "sieve mean of ω": odens["sieve mean of ω"]}

    if "prime_repetend" in cfg["cases"]:
        cop = (n % 2 != 0) & (n % 5 != 0)
        nr = n[cop]
        log(f"  repetend lengths in bases 2 and 10 for {nr.size:,} integers")
        L = repetend_lengths(nr, n_max, int(cfg["order_workers"]))
        checks["repetend_lengths"] = check_orders(nr, L, int(cfg["sympy_order_sample"]), seed)
        cases["prime_repetend"] = {"p": nr, "X": np.column_stack((nr, L[:, 0], L[:, 1], np.ones(nr.size))), "y": y_prime[cop],
                                   "names": ["n", "L_2 (repetend length of 1/n in base 2)", "L_10 (repetend length of 1/n in base 10)", "1"],
                                   "operations": INTEGER_OPS, "split_at": sp}
        f2 = ((nr - 1) % L[:, 0] == 0).astype(float)
        f10 = ((nr - 1) % L[:, 1] == 0).astype(float)
        base["prime_repetend"] = {"Fermat base 2 [L_2 | n−1]": f2, "Fermat base 10 [L_10 | n−1]": f10, "both": f2 * f10}

    e = np.arange(int(cfg["pi_min_exponent"]) * cfg["pi_per_decade"], int(cfg["pi_max_exponent"]) * cfg["pi_per_decade"] + 1)
    xs = np.unique(np.rint(10.0 ** (e / cfg["pi_per_decade"])).astype(np.int64))
    log(f"  π(x) at {xs.size} points up to {xs.max():.3g}")
    pi, checks["prime_counts"] = prime_counts(xs, P, limit)
    u = np.log(xs.astype(float))
    ratio = pi * u / xs
    lir = li_ratio(xs)
    Xc = np.column_stack((u, np.ones(xs.size)))
    cases["prime_count"] = {"p": xs, "X": Xc, "y": ratio, "names": ["log x", "1"], "operations": COUNT_OPS,
                            "split_at": tuple(cfg["pi_split_at"]), "regression": True, "pi": pi}
    lir2 = np.array([artin.li(math.sqrt(float(v))) for v in xs]) * u / xs
    cases["prime_count_density"] = {**cases["prime_count"], "X": np.column_stack((Xc, 1 / u, lir, lir2)),
                                    "names": ["log x", "1", "Cramér density 1/log x", "li(x)·log x / x", "li(√x)·log x / x"]}
    import mpmath

    base["prime_count"] = base["prime_count_density"] = {
        "x/log x": np.ones(xs.size), "x/(log x − 1)": u / (u - 1), f"Legendre x/(log x − {LEGENDRE_A})": u / (u - LEGENDRE_A),
        "li(x)": lir, "li(x) − li(√x)/2": lir - lir2 / 2, "Riemann R(x)": np.array([float(mpmath.riemannr(int(x))) for x in xs]) * u / xs,
    }

    sel = (P > 1000) & (P <= limit)
    k = np.flatnonzero(sel)
    k = k[k + 1 < P.size]
    p, q = P[k], P[k + 1]
    gap = (q - p).astype(float)
    lp = np.log(p.astype(float))
    psp = tuple(cfg["prime_split_at"])
    cases["prime_gap"] = {"p": p, "X": np.column_stack((lp, 1 / lp, p % 6, np.ones(p.size))), "y": gap,
                          "names": ["log p", "Cramér density 1/log p", "p mod 6", "1"], "operations": GAP_OPS,
                          "split_at": psp, "regression": True}
    base["prime_gap"] = {"Cramér log p": lp}
    same = (q % 10) == (p % 10)
    cases["last_digit_repeat"] = {"p": p, "X": np.column_stack((lp, 1 / lp, p % 10, np.ones(p.size))), "y": same,
                                  "names": ["log p", "Cramér density 1/log p", "last digit of p", "1"], "operations": GAP_OPS,
                                  "split_at": psp}
    base["last_digit_repeat"] = {"equidistribution 1/4": np.full(p.size, 0.25)}
    checks["consecutive_prime_rows"] = int(p.size)
    return {c: cases[c] for c in cfg["cases"]}, base, checks


# ---------------------------------------------------------------------------------------------
# Held-out scoring
# ---------------------------------------------------------------------------------------------


def masks(case: dict) -> List[np.ndarray]:
    lo, hi = case["split_at"]
    p = case["p"]
    return [p <= lo, (p > lo) & (p <= hi), p > hi]


def score(name: str, pred: np.ndarray, y: np.ndarray, case: dict, test: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    pred = np.where(np.isfinite(pred), pred, 0.0)
    out: Dict[str, Any] = {"mse": float(np.mean((pred - y) ** 2))}
    if name.startswith("prime_count"):
        rel = np.abs(pred / y - 1)
        out.update(max_relative_error=float(rel.max()), median_relative_error=float(np.median(rel)),
                   relative_error_at_largest_x=float(rel[-1]), largest_x=int(case["p"][test][-1]))
    elif name.startswith("omega") or name == "prime_gap":
        out["mean_prediction"], out["mean_target"] = float(pred.mean()), float(y.mean())
        if name.startswith("omega"):
            out["rounded_accuracy"] = float(np.mean(np.clip(np.rint(pred), 1, None) == y))
    else:
        pc = np.clip(pred, 1e-6, 1 - 1e-6)
        out.update(brier=out["mse"], log_loss=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                   accuracy=float(np.mean((pred > 0.5) == y)), predicted_count=float(np.clip(pred, 0, 1).sum()),
                   actual_count=int(y.sum()), errors_at_half=int(np.sum((pred > 0.5) != y)))
        try:
            out["auc"] = float(roc_auc_score(y, pred))
        except ValueError:
            out["auc"] = None
    return out


def gbm_control(case: dict, rows: int, seed: int) -> np.ndarray:
    """Gradient boosting on the same inputs (train range only): what an unconstrained model reaches."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    tr, _, te = masks(case)
    idx = np.flatnonzero(tr)
    if idx.size > rows:
        idx = np.random.default_rng([seed, 2203]).choice(idx, rows, replace=False)
    X, y = case["X"].astype(float), case["y"].astype(float)
    if case.get("regression"):
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(X[idx], y[idx])
        return m.predict(X[te])
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X[idx], y[idx].astype(int))
    return m.predict_proba(X[te])[:, 1]


def train_fits(name: str, case: dict) -> Dict[str, np.ndarray]:
    """Least-squares fits of the classical one-parameter forms on the train range, evaluated on test."""
    from scipy.optimize import minimize_scalar

    tr, _, te = masks(case)
    y = case["y"].astype(float)
    out: Dict[str, np.ndarray] = {}
    if name.startswith("prime_count"):
        u = case["X"][:, 0]
        A = minimize_scalar(lambda a: np.mean((u[tr] / (u[tr] - a) - y[tr]) ** 2), bounds=(0, 2.5), method="bounded").x
        out[f"train-fitted Legendre A = {A:.5f}"] = u[te] / (u[te] - A)
    elif name == "prime_gap":
        lp = case["X"][:, 0]
        a, b = np.polyfit(lp[tr], y[tr], 1)
        out[f"train fit {a:.4f}·log p + {b:.4f}"] = a * lp[te] + b
    elif name == "last_digit_repeat":
        inv = case["X"][:, 1]
        a, b = np.polyfit(inv[tr], y[tr], 1)
        out[f"train fit {b:.4f} + {a:.4f}/log p"] = b + a * inv[te]
    out["train mean"] = np.full(int(te.sum()), y[tr].mean())
    return out


def run_stage(cfg_in: Dict[str, Any], seed: int, run_dir: Path, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(cfg_in or {})}
    unknown = [c for c in cfg["cases"] if c not in CASES]
    if unknown:
        raise ValueError(f"unknown digit-prime cases {unknown}")
    random.seed(seed)
    cases, base, checks = build(cfg, seed, log)
    search_cfg = {k: cfg[k] for k in ("seed_offsets", "population", "generations", "initial_instructions",
                                      "max_instructions", "max_effective_instructions", "workers")}
    search_cfg["split_at"] = cfg["split_at"]
    if "eval_size_max" in cfg:
        search_cfg["eval_size_max"] = cfg["eval_size_max"]
    results: Dict[str, Any] = {}
    for name, case in cases.items():  # one case per call: workers hold only that case's arrays
        log(f"  case {name}: {case['X'].shape[0]:,} rows, inputs {case['names']}")
        out = prime_order_formulas.run_cases({name: case}, {**search_cfg, "targets": [name]}, seed, run_dir, log,
                                             int(case["p"].max()), knowledge_prefix="digits")
        row = out["targets"][name]
        tr, va, te = masks(case)
        y = case["y"].astype(float)
        pred = prime_order_formulas.replay(row["program"], case["X"][te].astype(float))
        if not np.isclose(np.mean((np.where(np.isfinite(pred), pred, 0.0) - y[te]) ** 2), row["test"]["mse"]):
            raise ArithmeticError(f"replayed GFSL program does not reproduce the test error of {name}")
        row["held_out"] = score(name, pred, y[te], case, te)
        controls = {k: v[te] for k, v in base[name].items()}
        controls.update(train_fits(name, case))
        if not name.startswith("prime_count"):
            controls["gradient boosting on the same inputs"] = gbm_control(case, int(cfg["gbm_rows"]), seed)
        row["controls"] = {k: score(name, v, y[te], case, te) for k, v in controls.items()}
        row["ranges"] = {s: [int(case["p"][m].min()), int(case["p"][m].max()), int(m.sum())] for s, m in zip(("train", "validation", "test"), (tr, va, te))}
        row["inputs"] = case["names"]
        results[name] = row
        log(f"    {name}: {row['expression']}  held-out {_headline(name, row['held_out'])}")
    return {"settings": cfg, "checks": checks, "split_rule": "train ≤ split_at[0] < validation ≤ split_at[1] < test (size coordinate)",
            "sieve_factor": SIEVE_FACTOR, "digit_primes": list(DIGIT_PRIMES), "mertens_b1": MERTENS_B1,
            "guide": "GFSLPolicyGuide", "targets": results}


def _headline(name: str, s: Dict[str, Any]) -> str:
    if name.startswith("prime_count"):
        return f"max rel. error {s['max_relative_error']:.2e}"
    if name.startswith("omega"):
        return f"MSE {s['mse']:.4f}, rounded accuracy {s['rounded_accuracy']:.4f}"
    if name == "prime_gap":
        return f"MSE {s['mse']:.2f}"
    return f"Brier {s['brier']:.5f}, AUC {s['auc']:.4f}, errors {s['errors_at_half']:,}"
