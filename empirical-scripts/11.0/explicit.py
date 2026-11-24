"""Zeros of the L-functions behind the halved-repetend counts (ROADMAP step 21).

For a base b ≥ 2 that is not a square, let Δ be the discriminant of Q(√b) and χ_Δ = (Δ/·) the Kronecker
symbol, a real primitive even character of conductor Δ. For a prime p ∤ Δ, Euler's criterion gives
    2 | I_b(p)  ⇔  b^{(p−1)/2} ≡ 1 (mod p)  ⇔  χ_Δ(p) = +1,
so the primes whose base-b repetend of 1/p is "halved" (length dividing (p − 1)/2) are those with χ_Δ(p) = 1.
ψ(x, χ_Δ) = Σ_{n ≤ x} Λ(n) χ_Δ(n) is therefore a log-weighted count of halved minus unhalved repetends,
plus prime powers. Bases 2, 3, 5, 10 give Δ = 8, 12, 5, 40. The Riemann zeta function is the
principal character: ψ(x) = Σ Λ(n), and by Lehmer's theorem n is prime iff 1/n has a full-period
repetend (length n − 1) in some base.

Exact statements used (tests: `tests/test_explicit.py`):

* Explicit formula (Davenport, ch. 17 and 19), x > 1:
      ψ(x, χ) = −Σ_ρ x^ρ/ρ − log x − b(χ) − ½ log(1 − x^{−2})        (χ even primitive),
      ψ(x)    = x − Σ_ρ x^ρ/ρ − log 2π − ½ log(1 − x^{−2}),
  with b(χ) the constant term of L'/L(s, χ) at s = 0; the functional equation gives
  b(χ) = γ + log 2 − log(Δ/π) − L'/L(1, χ), used as a cross-check.
* Gaussian smoothing in u = log x turns the conditionally convergent zero sum into an absolutely convergent
  one: with G_σ the normal density of width σ,
      M(u) := (G_σ ∗ e^{−v/2}[ψ(e^v) − main])(u) = −Σ_ρ e^{(ρ−½)u + σ²(ρ−½)²/2}/ρ + (G_σ ∗ e^{−v/2}R)(u).
  For zeros on the critical line the terms are e^{iγu − σ²γ²/2}/ρ. The left side is computed from the
  primes (exactly, with the normal CDF); the right side from the zeros found below. Agreement is a
  numerical check of the identity; a zero missing from the list (on or off the line) at height γ would
  leave a residual of size about 2e^{−σ²γ²/2}/|ρ|.
* Zeros: Z(t) = e^{iθ(t)} L(½ + it, χ) is real for real χ, θ(t) = arg Γ(¼ + it/2) + (t/2) log(Δ/π).
  Each sign change certifies a zero on the line. Completeness up to T is checked against
  N(T, χ) = θ(T)/π + S(T, χ) through the values S(γ_k) ≈ k − ½ − θ(γ_k)/π (Turing-style, not rigorous).
* Chebyshev bias (Rubinstein–Sarnak 1994), assuming GRH and the linear independence hypothesis (LI):
  E(x) = (log x/√x)(π_unhalved(x) − π_halved(x)) has a limiting distribution X = 1 + Σ_{γ>0} (2/|ρ|) cos Θ_γ,
  Θ_γ independent uniform, and the logarithmic density of {x : π_unhalved(x) > π_halved(x)} is
      δ = ½ + (1/π) ∫_0^∞ (sin t / t) ∏_{γ>0} J_0(2t/|ρ|) dt.
* Schoenfeld (1976): RH implies |θ(x) − x| < √x log² x/(8π) for x ≥ 599, and any bound O(x^{1/2+ε})
  implies RH. The finite maximum below can only find a counterexample, never prove the bound.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.special import j0, ndtr

import artin
import failures
import ndr

DEFAULTS: Dict[str, Any] = {
    "bases": [2, 3, 5, 10],
    "prime_limit": 100_000_000,
    "height": 150.0,
    "step": 0.05,
    "dps": 25,
    "sigma": 0.05,
    "u_points": 400,
    "u_min": math.log(100.0),
    "bias_t_max": 80.0,
    "bias_dt": 0.002,
    "workers": 0,
}


# ---------------------------------------------------------------------------------------------
# Characters and L-functions
# ---------------------------------------------------------------------------------------------


def kronecker(D: int, n: int) -> int:
    """Kronecker symbol (D/n) for n ≥ 1 (D a fundamental discriminant)."""
    from sympy import jacobi_symbol

    e = 0
    while n % 2 == 0:
        n //= 2
        e += 1
    two = 0 if D % 2 == 0 else (1 if D % 8 in (1, 7) else -1)
    head = two ** e if e else 1
    return head * (int(jacobi_symbol(D % n, n)) if n > 1 else 1)


def character_table(D: int) -> List[int]:
    return [0] + [kronecker(D, a) if math.gcd(a, D) == 1 else 0 for a in range(1, D)]


def b_constant(chi: Sequence[int], dps: int) -> Dict[str, float]:
    """Constant term b(χ) of L'/L(s, χ) at s = 0 for even χ, by two routes.

    Direct: L(0) = 0, so b = L''(0)/(2 L'(0)) from central differences at s = ±h (high precision).
    Functional equation: b = γ + log 2 − log(q/π) − L'/L(1, χ), with L and L' at 1 ± h
    (mpmath's Hurwitz sums are avoided at s = 1 itself)."""
    import mpmath

    with mpmath.workdps(dps + 30):
        f = lambda s: mpmath.dirichlet(s, list(chi))
        h = mpmath.mpf(10) ** (-(dps // 2 + 4))
        fp, fm, f0 = f(h), f(-h), f(mpmath.mpf(0))
        d1 = (fp - fm) / (2 * h)
        d2 = (fp + fm - 2 * f0) / h ** 2
        direct = d2 / (2 * d1)
        a, c = f(1 + h), f(1 - h)
        L1 = (a + c) / 2
        Lp1 = (a - c) / (2 * h)
        q = len(chi)
        fe = mpmath.euler + mpmath.log(2) - mpmath.log(q / mpmath.pi) - Lp1 / L1
        return {"direct": float(direct), "functional_equation": float(fe), "L_prime_0": float(d1),
                "L_1": float(L1), "L_half": float(mpmath.re(f(mpmath.mpf(0.5))))}


def theta(chi: Sequence[int] | None, t: float):
    import mpmath

    if chi is None:
        return mpmath.siegeltheta(t)
    return mpmath.im(mpmath.loggamma(mpmath.mpf(0.25) + 0.5j * t)) + t / 2 * mpmath.log(len(chi) / mpmath.pi)


def hardy_z(chi: Sequence[int] | None, t: float):
    import mpmath

    if chi is None:
        return mpmath.siegelz(t)
    val = mpmath.exp(1j * theta(chi, t)) * mpmath.dirichlet(mpmath.mpf(0.5) + 1j * t, list(chi))
    return mpmath.re(val)


def find_zeros(args) -> Dict[str, Any]:
    """Zeros ½ + iγ, 0 < γ ≤ T, located by sign changes of the real function Z on a grid and refined."""
    import mpmath

    chi, T, step, dps = args
    mpmath.mp.dps = dps
    grid = np.arange(step, T + step / 2, step)
    vals = [hardy_z(chi, float(t)) for t in grid]
    zeros = []
    imag_check = 0.0
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0:
            zeros.append(float(grid[i]))
        elif a * b < 0:
            g = mpmath.findroot(lambda t: hardy_z(chi, t), (mpmath.mpf(grid[i]), mpmath.mpf(grid[i + 1])), solver="illinois")
            zeros.append(float(g))
    if chi is not None:  # Z is real: the imaginary part of e^{iθ}L is rounding only
        for t in grid[:: max(1, len(grid) // 50)]:
            v = mpmath.exp(1j * theta(chi, float(t))) * mpmath.dirichlet(mpmath.mpf(0.5) + 1j * float(t), list(chi))
            imag_check = max(imag_check, float(abs(mpmath.im(v))) / max(1e-30, float(abs(v))))
    th = [float(theta(chi, g)) for g in zeros]
    offset = 1.0 if chi is None else 0.0  # N(T) = θ(T)/π + 1 + S(T) for ζ, θ(T)/π + S(T) for L(s, χ)
    S = [k + 0.5 - th[k] / math.pi - offset for k in range(len(zeros))]  # S(γ_k) estimate, k counted from 0
    thT = float(theta(chi, T))
    return {"conductor": 1 if chi is None else len(chi), "height": T, "step": step, "zeros": zeros,
            "count": len(zeros), "smooth_count_at_T": thT / math.pi + offset,
            "S_at_zeros": {"mean": float(np.mean(S)) if S else None, "min": float(min(S)) if S else None,
                           "max": float(max(S)) if S else None},
            "relative_imaginary_part_of_Z": imag_check,
            "min_gap": float(np.min(np.diff(zeros))) if len(zeros) > 1 else None}


# ---------------------------------------------------------------------------------------------
# Prime data
# ---------------------------------------------------------------------------------------------


def halved_characters(P: np.ndarray, bases: Sequence[int], workers: int) -> Dict[int, np.ndarray]:
    """χ_Δ(p) from the repetend index parity: +1 when 2 | I_b(p), −1 when not, 0 when p | Δ."""
    out = {}
    big = P[P > 5]
    O = failures.order_matrix(big, list(bases), workers)
    for j, b in enumerate(bases):
        D = artin.discriminant(b)
        chi = np.zeros(P.size, dtype=np.int8)
        I = np.zeros(P.size, dtype=np.int64)
        I[P > 5] = (big - 1) // O[:, j]
        for i in np.flatnonzero(P <= 5):
            p = int(P[i])
            if D % p and b % p:
                I[i] = (p - 1) // ndr.multiplicative_order(b, p, p - 1, [q for q, _ in ndr.factorize(p - 1, ndr.spf_sieve(p))])
        ok = (D % P) != 0
        chi[ok] = np.where(I[ok] % 2 == 0, 1, -1)
        table = np.array(character_table(D), dtype=np.int8)
        if not np.array_equal(chi, table[P % D]):
            raise ArithmeticError(f"index parity of base {b} disagrees with the Kronecker symbol ({D}/p)")
        out[b] = chi
    return out


def prime_powers(P: np.ndarray, chi: np.ndarray, X: int) -> tuple[np.ndarray, np.ndarray]:
    """log n and Λ(n)χ(n) for prime powers n ≤ X, sorted by n."""
    logs, ws = [np.log(P.astype(np.float64))], [np.log(P.astype(np.float64)) * chi]
    small = P[P * P <= X]
    for p, c in zip(small.tolist(), chi[: small.size].tolist()):
        pk, k = p * p, 2
        while pk <= X:
            logs.append(np.array([math.log(pk)]))
            ws.append(np.array([math.log(p) * c ** k]))
            pk *= p
            k += 1
    lg = np.concatenate(logs)
    w = np.concatenate(ws)
    order = np.argsort(lg, kind="stable")
    return lg[order], w[order]


def smoothed_measured(lg: np.ndarray, w: np.ndarray, u: np.ndarray, sigma: float, principal: bool) -> np.ndarray:
    """M(u) = (G_σ ∗ e^{−v/2}[ψ(e^v) − main])(u), exactly from the step function ψ."""
    cum = np.concatenate([[0.0], np.cumsum(w)])
    out = np.empty(u.size)
    for i, ui in enumerate(u):
        c = ui - sigma * sigma / 2
        lo = int(np.searchsorted(lg, c - 9 * sigma))
        hi = int(np.searchsorted(lg, c + 9 * sigma))
        part = cum[lo] + float(np.dot(w[lo:hi], ndtr((c - lg[lo:hi]) / sigma)))
        out[i] = math.exp(-ui / 2 + sigma * sigma / 8) * part
        if principal:
            out[i] -= math.exp(ui / 2 + sigma * sigma / 8)
    return out


def smoothed_predicted(zeros: Sequence[float], u: np.ndarray, sigma: float, bconst: float | None) -> Dict[str, np.ndarray]:
    """Zero sum and the smoothed remaining terms (Gauss–Hermite in v)."""
    g = np.array(zeros)
    rho = 0.5 + 1j * g
    damp = np.exp(-sigma * sigma * g * g / 2)
    zsum = -2 * np.real(np.exp(1j * np.outer(u, g)) @ (damp / rho))
    z, wq = np.polynomial.hermite.hermgauss(60)
    v = u[:, None] + math.sqrt(2) * sigma * z[None, :]
    x = np.exp(v)
    if bconst is None:
        R = -math.log(2 * math.pi) - 0.5 * np.log1p(-x ** -2.0)
    else:
        R = -v - bconst - 0.5 * np.log1p(-x ** -2.0)
    rest = (np.exp(-v / 2) * R) @ wq / math.sqrt(math.pi)
    return {"zero_sum": zsum, "rest": rest}


# ---------------------------------------------------------------------------------------------
# Chebyshev bias and the RH-equivalent bound
# ---------------------------------------------------------------------------------------------


def detectable_height(sigma: float, residual: float) -> float:
    """Largest γ at which one extra zero would leave a smoothed residual of 10× the observed maximum."""
    lo, hi = 0.0, 1e4
    f = lambda g: 2 * math.exp(-sigma * sigma * g * g / 2) / math.hypot(0.5, g) - 10 * residual
    if f(lo) < 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return lo


def rubinstein_sarnak(zeros: Sequence[float], conductor: int, T: float, t_max: float, dt: float) -> Dict[str, float]:
    """δ = ½ + (1/π)∫ sin t/t ∏ J_0(2t/|ρ|) dt under GRH + LI, with a Gaussian tail for γ > T."""
    g = np.array(zeros)
    mod = np.sqrt(0.25 + g * g)
    tail = (math.log(conductor * T / (2 * math.pi)) + 1) / (2 * math.pi * T)
    t = np.arange(dt, t_max + dt / 2, dt)
    phi = np.ones(t.size)
    for m in mod:
        phi *= j0(2 * t / m)
    phi *= np.exp(-t * t * tail)
    f = np.sin(t) / t * phi
    integral = dt * (np.sum(f) - 0.5 * f[-1]) + dt / 2 * 1.0  # trapezoid, f(0) = 1
    var = float(np.sum(2 / mod ** 2) + 2 * tail)
    return {"delta": 0.5 + integral / math.pi, "mean": 1.0, "variance": var, "zeros_used": int(g.size),
            "tail_sum_inverse_rho_squared": tail, "phi_at_t_max": float(abs(phi[-1]))}


def race(P: np.ndarray, chi: np.ndarray, X: int) -> Dict[str, Any]:
    """Logarithmic share of x ∈ [2, X] with more unhalved than halved repetends, and the log-mean of E(x)."""
    D = np.cumsum(-chi.astype(np.int64))  # π_unhalved − π_halved on [p_i, p_{i+1})
    edges = np.log(np.append(P.astype(np.float64), float(X)))
    seg = np.diff(edges)
    total = math.log(X / 2.0)
    lead = float(seg[D > 0].sum())
    tie = float(seg[D == 0].sum())
    a, b = edges[:-1], edges[1:]
    integ = -2 * np.exp(-b / 2) * (b + 2) + 2 * np.exp(-a / 2) * (a + 2)  # ∫ u e^{−u/2} du
    mask = a >= math.log(1000.0)
    e_mean = float(np.sum(D[mask] * integ[mask]) / (edges[-1] - a[mask][0])) if mask.any() else None
    sg = np.sign(D[D != 0])
    flips = np.flatnonzero(sg[1:] != sg[:-1])  # lead changes (ties skipped)
    neg = np.flatnonzero(D < 0)
    uu = edges[:-1][mask]
    w_seg = seg[mask]
    leading = float(1 + np.sum(2 / (uu + w_seg / 2) * w_seg) / w_seg.sum()) if mask.any() else None
    return {"log_share_unhalved_ahead": lead / total, "log_share_tied": tie / total,
            "log_mean_E_from_1000": e_mean, "leading_order_mean_1_plus_2_over_log_x": leading,
            "lead_changes": int(flips.size),
            "first_halved_lead_at": int(P[neg[0]]) if neg.size else None,
            "last_halved_lead_at": int(P[neg[-1]]) if neg.size else None,
            "primes_with_halved_lead": int(neg.size), "final_difference": int(D[-1])}


def schoenfeld(P: np.ndarray, X: int) -> Dict[str, Any]:
    """max over x ∈ [599, X] of |θ(x) − x| / (√x log² x / 8π), from both ends of every prime gap."""
    th = np.cumsum(np.log(P.astype(np.float64)))
    nxt = np.append(P[1:], X).astype(np.float64)
    keep = P >= 599
    ratios = []
    for xs, vals in ((P[keep].astype(np.float64), th[keep] - P[keep]), (nxt[keep], th[keep] - nxt[keep])):
        ratios.append(np.abs(vals) / (np.sqrt(xs) * np.log(xs) ** 2 / (8 * math.pi)))
    r = np.maximum(ratios[0], ratios[1])
    i = int(np.argmax(r))
    sq = np.maximum(np.abs(th[keep] - P[keep]), np.abs(th[keep] - nxt[keep])) / np.sqrt(P[keep].astype(np.float64))
    return {"max_ratio_to_schoenfeld_bound": float(r[i]), "at_prime": int(P[keep][i]),
            "max_abs_theta_minus_x_over_sqrt_x": float(sq.max()), "range": [599, int(X)]}


def run_stage(cfg_in: Dict[str, Any], seed: int, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(cfg_in or {})}
    t0 = time.time()
    X = int(cfg["prime_limit"])
    bases = [int(b) for b in cfg["bases"]]
    workers = int(cfg["workers"]) or max(1, (os.cpu_count() or 2) - 1)
    T, step, dps, sigma = float(cfg["height"]), float(cfg["step"]), int(cfg["dps"]), float(cfg["sigma"])
    chars = {b: character_table(artin.discriminant(b)) for b in bases}
    jobs = [(None, T, step, dps)] + [(chars[b], T, step, dps) for b in bases]
    log(f"  21: zeros up to height {T} for ζ and L(s, χ_Δ), Δ = {[artin.discriminant(b) for b in bases]}")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        zero_sets = list(pool.map(find_zeros, jobs))
    import mpmath

    mpmath.mp.dps = dps
    ref = []
    k = 1
    while True:
        g = float(mpmath.zetazero(k).imag)
        if g > T:
            break
        ref.append(g)
        k += 1
    zeta_zeros = zero_sets[0]
    zeta_zeros["mpmath_zetazero_agrees"] = bool(len(ref) == zeta_zeros["count"] and np.allclose(ref, zeta_zeros["zeros"], atol=1e-9))
    if not zeta_zeros["mpmath_zetazero_agrees"]:
        raise ArithmeticError("the sign-change zero finder disagrees with mpmath.zetazero for ζ")
    log(f"  21: ζ: {zeta_zeros['count']} zeros (= mpmath.zetazero); L: {[z['count'] for z in zero_sets[1:]]}")

    P = failures.sieve_primes(2, X)
    t1 = time.time()
    chis = halved_characters(P, bases, workers)
    log(f"  21: index parity = Kronecker symbol on {P.size:,} primes ({time.time() - t1:.0f}s)")
    u = np.linspace(float(cfg["u_min"]), math.log(X) - 9 * sigma, int(cfg["u_points"]))
    results = {}
    for name, chi, zs, bc in [("zeta", np.ones(P.size, dtype=np.int8), zeta_zeros, None)] + [
            (f"chi_{artin.discriminant(b)}", chis[b], zero_sets[i + 1], b) for i, b in enumerate(bases)]:
        rec: Dict[str, Any] = {"zeros": zs}
        if bc is not None:
            consts = b_constant(chars[bc], dps)
            if abs(consts["direct"] - consts["functional_equation"]) > 1e-10:
                raise ArithmeticError(f"b(χ) disagrees between the two computations for base {bc}")
            rec["base"] = bc
            rec["discriminant"] = artin.discriminant(bc)
            rec["constants"] = consts
            bval = consts["direct"]
        else:
            bval = None
        lg, w = prime_powers(P, chi, X)
        meas = smoothed_measured(lg, w, u, sigma, principal=bc is None)
        pred = smoothed_predicted(zs["zeros"], u, sigma, bval)
        resid = meas - (pred["zero_sum"] + pred["rest"])
        top = zs["zeros"][-1] if zs["zeros"] else T
        rec["explicit_formula"] = {
            "sigma": sigma, "u_range": [float(u[0]), float(u[-1])], "x_range": [float(math.exp(u[0])), float(math.exp(u[-1]))],
            "points": int(u.size), "rms_measured": float(np.sqrt(np.mean(meas ** 2))),
            "rms_zero_sum": float(np.sqrt(np.mean(pred["zero_sum"] ** 2))),
            "max_abs_residual": float(np.max(np.abs(resid))), "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
            "max_abs_residual_without_zeros": float(np.max(np.abs(meas - pred["rest"]))),
            "residual_of_one_missing_zero_at_height_T": 2 * math.exp(-sigma ** 2 * top ** 2 / 2) / abs(0.5 + 1j * top),
            "detectable_missing_zero_height": detectable_height(sigma, float(np.max(np.abs(resid)))),
            "samples": {"u": u[:: max(1, u.size // 40)].tolist(), "measured": meas[:: max(1, u.size // 40)].tolist(),
                        "predicted": (pred["zero_sum"] + pred["rest"])[:: max(1, u.size // 40)].tolist()},
        }
        if bc is not None:
            rec["bias"] = rubinstein_sarnak(zs["zeros"], rec["discriminant"], T, float(cfg["bias_t_max"]), float(cfg["bias_dt"]))
            rec["race"] = race(P, chi, X)
            rec["max_abs_psi_chi_over_sqrt_x"] = float(np.max(np.abs(np.cumsum(w)) / np.sqrt(np.exp(lg))))
        else:
            rec["schoenfeld"] = schoenfeld(P, X)
        results[name] = rec
        ef = rec["explicit_formula"]
        log(f"  21: {name}: {zs['count']} zeros; smoothed explicit formula residual {ef['max_abs_residual']:.2e} "
            f"(signal rms {ef['rms_measured']:.3f}, without zeros {ef['max_abs_residual_without_zeros']:.3f})"
            + (f"; bias δ = {rec['bias']['delta']:.4f}, measured log share {rec['race']['log_share_unhalved_ahead']:.4f}" if bc else ""))
    return {"settings": cfg, "primes": int(P.size), "results": results, "seconds": round(time.time() - t0, 1),
            "references": {
                "explicit_formula": "H. Davenport, Multiplicative Number Theory, 3rd ed., ch. 17 and 19",
                "bias": "M. Rubinstein and P. Sarnak, Chebyshev's bias, Experiment. Math. 3 (1994) 173–197",
                "schoenfeld": "L. Schoenfeld, Sharper bounds for the Chebyshev functions θ(x) and ψ(x). II, Math. Comp. 30 (1976) 337–360",
                "lehmer": "D. H. Lehmer (1927): n is prime iff some b has multiplicative order n − 1 modulo n",
            }}
