"""Full-period repetends at prime denominators: Artin's primitive-root conjecture (ROADMAP step 20).

For a prime p not dividing b, the repetend of 1/p in base b has length ord_p(b) = (p − 1)/I_b(p).
It has the maximal length p − 1 exactly when I_b(p) = 1, that is, when b is a primitive root modulo p.
Gauss tabulated these "full-period" primes for base 10; Artin (1927) conjectured their density.

Exact facts used here (tests: `tests/test_artin.py`):

* **Artin's criterion.** I_b(p) = 1 iff no prime q | p − 1 has b^{(p−1)/q} ≡ 1 (mod p), i.e. iff p splits
  completely in no field K_q = Q(ζ_q, b^{1/q}). Inclusion–exclusion over squarefree n gives the
  candidate density δ(b) = Σ_n μ(n)/[K_n : Q], K_n = Q(ζ_n, b^{1/n}).
* **Kummer degrees.** For b > 1 not a perfect power and Δ the discriminant of Q(√b),
  [Q(ζ_m, b^{1/m}) : Q] = m φ(m)/ε(m), with ε(m) = 2 if m is even and Δ | m, else 1.
* **Index law.** The same inclusion–exclusion at level t gives the density of I_b(p) = t:
  d_b(t) = Σ_n μ(n) ε(tn)/(tn φ(tn)), a finite combination of Euler products (`index_density`).
* **Closed form at t = 1** (Hooley; Moree's survey, h = largest exponent with b a perfect h-th power):
  δ(b) = A(h) if Δ ≢ 1 mod 4, else A(h)·(1 − μ(|Δ|) ∏_{q|Δ, q|h} 1/(q − 2) ∏_{q|Δ, q∤h} 1/(q² − q − 1)),
  A(h) = ∏_{q|h} (1 − 1/(q − 1)) ∏_{q∤h} (1 − 1/(q(q − 1))). `moree_density` implements it and
  `index_density(b, 1)` checks it independently for non-powers. A(1) = A is Artin's constant.

What is conditional: each finite piece of the inclusion–exclusion is Chebotarev (unconditional);
exchanging the limit with the infinite sum is Hooley's theorem (1967), which assumes GRH for the
Dedekind zeta functions of the fields K_n. Hooley's error term O(x log log x / log² x) has no explicit
constant, so a finite count cannot test the theorem: the counts below measure how close the finite
frequencies are to the conditional densities. Heath-Brown (1986) proved unconditionally that at least
one of 2, 3, 5 is a primitive root for infinitely many primes; the counts for that union are descriptive.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.special import exp1, expi

import decode
import failures

ARTIN_CONSTANT_10_DIGITS = 0.3739558136  # OEIS A005596, first ten digits

DEFAULTS: Dict[str, Any] = {
    "bases": decode.BASES19,
    "prime_limit": 100_000_000,
    "cutoffs": [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
    "index_max": 12,
    "local_q_max": 401,
    "sympy_order_limit": 30_000,
    "sympy_sample": 4_000,
    "workers": 0,
}


# ---------------------------------------------------------------------------------------------
# Small exact arithmetic
# ---------------------------------------------------------------------------------------------


def factor_small(n: int) -> Dict[int, int]:
    """Prime factorization of a small positive integer by trial division."""
    out: Dict[int, int] = {}
    q = 2
    while q * q <= n:
        while n % q == 0:
            out[q] = out.get(q, 0) + 1
            n //= q
        q += 1 if q == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def power_exponent(b: int) -> int:
    """Largest h with b = g^h for an integer g (b ≥ 2)."""
    return math.gcd(*factor_small(b).values())


def squarefree_kernel(b: int) -> int:
    return math.prod(q for q, e in factor_small(b).items() if e % 2)


def discriminant(b: int) -> int:
    """Discriminant of Q(√b) for b ≥ 2 (1 when b is a square)."""
    k = squarefree_kernel(b)
    if k == 1:
        return 1
    return k if k % 4 == 1 else 4 * k


def mobius(n: int) -> int:
    f = factor_small(n)
    return 0 if any(e > 1 for e in f.values()) else (-1) ** len(f)


def euler_phi(n: int) -> int:
    return math.prod((q - 1) * q ** (e - 1) for q, e in factor_small(n).items())


def kummer_degree(b: int, m: int) -> int:
    """[Q(ζ_m, b^{1/m}) : Q] for b ≥ 2 not a perfect power."""
    if power_exponent(b) != 1:
        raise ValueError("kummer_degree needs a base that is not a perfect power")
    D = discriminant(b)
    eps = 2 if m % 2 == 0 and m % D == 0 else 1
    return m * euler_phi(m) // eps


def artin_constant(primes: np.ndarray) -> float:
    """∏_q (1 − 1/(q(q − 1))) over the given primes (all primes up to P) with the tail −E1(log P)."""
    q = primes.astype(np.float64)
    log_a = math.fsum(np.log1p(-1.0 / (q * (q - 1))).tolist())
    return math.exp(log_a - float(exp1(math.log(float(primes[-1])))))


def moree_density(b: int, A: float) -> float:
    """Hooley's conditional density of primes with b as a primitive root (Moree's closed form)."""
    h = power_exponent(b)
    hp = factor_small(h)
    Ah = A
    for q in hp:
        Ah *= (1 - 1 / (q - 1)) / (1 - 1 / (q * (q - 1)))
    D = discriminant(b)
    if D % 4 != 1:
        return Ah
    corr = 1.0
    for q in factor_small(D):
        corr *= 1 / (q - 2) if q in hp else 1 / (q * q - q - 1)
    return Ah * (1 - mobius(abs(D)) * corr)


def index_density(b: int, t: int, A: float) -> float:
    """Conditional density of primes with I_b(p) = t, from the Kummer degrees (b not a perfect power).

    d = Σ_n μ(n)/(tn φ(tn)) + Σ_{n: 2 | tn, Δ | tn} μ(n)/(tn φ(tn)), each an Euler product; primes
    outside t and 2Δ contribute 1 − 1/(q(q − 1)), absorbed into A."""
    if power_exponent(b) != 1:
        raise ValueError("index_density needs a base that is not a perfect power")
    D = discriminant(b)
    ft, fD = factor_small(t), factor_small(D)
    S = sorted(set(ft) | set(fD) | {2})

    def local(q: int, v: int, w: int) -> float:
        e = v + w
        val = 1.0 if e == 0 else 1.0 / (q ** e * (q - 1) * q ** (e - 1))
        return -val if w else val

    generic = entangled = A
    for q in S:
        v = ft.get(q, 0)
        base = 1 - 1 / (q * (q - 1))
        generic *= (local(q, v, 0) + local(q, v, 1)) / base
        need = max(fD.get(q, 0), 1 if q == 2 else 0)
        entangled *= sum(local(q, v, w) for w in (0, 1) if v + w >= need) / base
    return generic + entangled


def pair_density_2_3(primes: np.ndarray) -> float:
    """Conditional density of primes with both 2 and 3 primitive roots: ∏_q (1 − (2q − 1)/(q²(q − 1))).

    The subgroup ⟨2, 3⟩ has no entanglement with squarefree cyclotomic fields (√2, √3, √6 need 8 or 4 | m),
    so [Q(ζ_m, 2^{1/n1}, 3^{1/n2}) : Q] = φ(m) n1 n2 with m = lcm(n1, n2) squarefree, and the double
    inclusion–exclusion factors over primes. Tail: −2 E1(log P)."""
    q = primes.astype(np.float64)
    log_d = math.fsum(np.log1p(-(2 * q - 1) / (q * q * (q - 1))).tolist())
    return math.exp(log_d - 2 * float(exp1(math.log(float(primes[-1])))))


def legendre_by_reciprocity(b: int, p: np.ndarray) -> np.ndarray:
    """(b/p) for odd primes p ∤ b from p mod 8 and quadratic reciprocity (independent of any order)."""
    out = np.ones(p.size, dtype=np.int8)
    for q, e in factor_small(b).items():
        if e % 2 == 0:
            continue
        if q == 2:
            s = np.where(np.isin(p % 8, (1, 7)), 1, -1)
        else:
            squares = np.zeros(q, dtype=bool)
            squares[[(x * x) % q for x in range(1, q)]] = True
            s = np.where(squares[p % q], 1, -1)
            if q % 4 == 3:
                s = np.where(p % 4 == 3, -s, s)
        out = (out * s).astype(np.int8)
    return out


# ---------------------------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------------------------


def li(x: float) -> float:
    return float(expi(math.log(x)))


def strip(I: np.ndarray, qs: Sequence[int]) -> np.ndarray:
    """I with every factor in qs removed."""
    r = I.copy()
    for q in qs:
        m = r % q == 0
        while m.any():
            r[m] //= q
            m = r % q == 0
    return r


def sympy_checks(P: np.ndarray, I: np.ndarray, bases: Sequence[int], order_limit: int, sample: int, seed: int) -> Dict[str, Any]:
    """Orders by SymPy on every prime ≤ order_limit; primitive-root status by SymPy on a seeded sample."""
    from sympy.ntheory import is_primitive_root, n_order

    small = np.flatnonzero(P <= order_limit)
    for i in small:
        p = int(P[i])
        for j, b in enumerate(bases):
            if n_order(b, p) * int(I[i, j]) != p - 1:
                raise ArithmeticError(f"order of {b} mod {p} disagrees with SymPy")
    rng = np.random.default_rng([seed, 20])
    pick = rng.choice(P.size, size=min(sample, P.size), replace=False)
    for i in pick:
        p = int(P[i])
        for j, b in enumerate(bases):
            if is_primitive_root(b, p) != (int(I[i, j]) == 1):
                raise ArithmeticError(f"primitive-root status of {b} mod {p} disagrees with SymPy")
    return {"orders_all_primes_through": int(order_limit), "order_primes": int(small.size),
            "primitive_root_sample": int(pick.size), "sample_max_prime": int(P[pick].max()) if pick.size else None}


def run_stage(cfg_in: Dict[str, Any], seed: int, run_dir, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(cfg_in or {})}
    t0 = time.time()
    bases = [int(b) for b in cfg["bases"]]
    X = int(cfg["prime_limit"])
    cutoffs = [int(x) for x in cfg["cutoffs"]]
    if cutoffs != sorted(set(cutoffs)) or cutoffs[-1] != X:
        raise ValueError("artin.cutoffs must increase and end at prime_limit")
    all_primes = failures.sieve_primes(2, X)
    excluded = failures.base_primes(bases)
    P = all_primes[all_primes > max(excluded)]
    log(f"  20: {P.size:,} primes in ({max(excluded)}, {X:,}]; orders of {len(bases)} bases")
    t1 = time.time()
    O = failures.order_matrix(P, bases, int(cfg["workers"]))
    log(f"  20: order matrix in {time.time() - t1:.0f}s")
    if np.any((P[:, None] - 1) % O):
        raise ArithmeticError("an order does not divide p − 1")
    I = (P[:, None] - 1) // O
    del O
    full = I == 1

    # Independent checks: reciprocity for the index parity of every prime and base; SymPy orders.
    for j, b in enumerate(bases):
        if not np.array_equal(I[:, j] % 2 == 0, legendre_by_reciprocity(b, P) == 1):
            raise ArithmeticError(f"index parity of base {b} disagrees with quadratic reciprocity")
    checks = {"parity_vs_reciprocity": {"primes": int(P.size), "bases": len(bases), "agree": True}}
    checks["sympy"] = sympy_checks(P, I, bases, int(cfg["sympy_order_limit"]), int(cfg["sympy_sample"]), seed)
    log(f"  20: index parity = reciprocity on all primes; SymPy agrees on {checks['sympy']['order_primes']:,} primes "
        f"and {checks['sympy']['primitive_root_sample']:,} sampled primes")

    product_primes = all_primes if X >= 10 ** 6 else failures.sieve_primes(2, 10 ** 6)
    A = artin_constant(product_primes)
    if abs(A - ARTIN_CONSTANT_10_DIGITS) > 2e-10:
        raise ArithmeticError(f"Artin constant {A} disagrees with OEIS A005596")
    dens = {}
    for b in bases:
        rec = {"h": power_exponent(b), "discriminant": discriminant(b), "density": moree_density(b, A)}
        rec["ratio_to_A"] = rec["density"] / A
        if rec["h"] == 1:
            rec["density_from_kummer_degrees"] = index_density(b, 1, A)
            if not math.isclose(rec["density"], rec["density_from_kummer_degrees"], rel_tol=1e-12):
                raise ArithmeticError(f"closed-form and Kummer-degree densities disagree for base {b}")
        dens[b] = rec

    rows = []
    for x in cutoffs:
        k = int(np.searchsorted(P, x, side="right"))
        main_li = li(x) - li(float(P[0]))
        per = {}
        for j, b in enumerate(bases):
            n = int(full[:k, j].sum())
            d = dens[b]["density"]
            per[b] = {
                "count": n, "rate": n / k, "expected": d * k, "discrepancy": n - d * k,
                "discrepancy_vs_li": n - d * main_li,
                "binomial_scale": math.sqrt(d * (1 - d) * k),
                "binomial_z": (n - d * k) / math.sqrt(d * (1 - d) * k) if 0 < d < 1 else None,
                "ratio_to_A": n / (A * k),
            }
        both23 = int((full[:k, bases.index(2)] & full[:k, bases.index(3)]).sum())
        any235 = int((full[:k, [bases.index(2), bases.index(3), bases.index(5)]]).any(axis=1).sum())
        rows.append({"x": x, "eligible": k, "li_main": main_li, "bases": per,
                     "both_2_and_3": both23, "any_of_2_3_5": any235})
        log(f"  20: x = {x:,}: base 10 full-period {per[10]['count']:,} ({per[10]['rate']:.5f}; δ = {dens[10]['density']:.5f}), "
            f"base 5 ratio to A {per[5]['ratio_to_A']:.4f} (Hooley 20/19 = {20 / 19:.4f})")

    k = P.size
    corrections = {}
    for j, b in enumerate(bases):
        if dens[b]["h"] == 1 and dens[b]["discriminant"] % 4 == 1:
            n = int(full[:, j].sum())
            d = dens[b]["density"]
            corrections[b] = {
                "count": n, "ratio_to_A": n / (A * k), "hooley_ratio": d / A,
                "z_against_A": (n - A * k) / math.sqrt(A * (1 - A) * k),
                "z_against_hooley": (n - d * k) / math.sqrt(d * (1 - d) * k),
            }

    law = {}
    tmax = int(cfg["index_max"])
    for j, b in enumerate(bases):
        if dens[b]["h"] != 1:
            law[b] = {"note": "perfect square: ord_p(b) divides (p − 1)/2, so the index is always even",
                      "full_period_count": int(full[:, j].sum()),
                      "index_always_even": bool(np.all(I[:, j] % 2 == 0))}
            continue
        counts = np.bincount(np.minimum(I[:, j], tmax + 1), minlength=tmax + 2)[1:tmax + 1]
        pred = np.array([index_density(b, t, A) for t in range(1, tmax + 1)])
        pos = pred > 0
        zero_t = [t for t in range(1, tmax + 1) if not pos[t - 1]]
        if counts[~pos].any():
            raise ArithmeticError(f"base {b}: an index of density zero occurs")
        dev = np.full(tmax, np.nan)
        dev[pos] = (counts[pos] - pred[pos] * k) / np.sqrt(pred[pos] * (1 - pred[pos]) * k)
        law[b] = {"t": list(range(1, tmax + 1)), "counts": counts.tolist(), "densities": pred.tolist(),
                  "zero_density_t": zero_t,
                  "binomial_z": [None if math.isnan(z) else float(z) for z in dev],
                  "max_abs_binomial_z": float(np.nanmax(np.abs(dev))),
                  "density_sum_t_le_20000": float(sum(index_density(b, t, A) for t in range(1, 20001)))}

    nonsquare = [j for j, b in enumerate(bases) if power_exponent(b) == 1]
    qs = [int(q) for q in all_primes[all_primes <= int(cfg["local_q_max"])]]
    local = {"bases": [bases[j] for j in nonsquare], "q": qs, "rates": [], "pairs": int(k * len(nonsquare))}
    Isub = I[:, nonsquare]
    for q in qs:
        local["rates"].append(float(np.count_nonzero(Isub % q == 0)) / (k * len(nonsquare)))

    out = {
        "settings": cfg, "domain": {"primes_from": int(P[0]), "primes_to": int(P[-1]), "primes": int(k),
                                    "excluded_primes": excluded},
        "checks": checks, "artin_constant": A, "densities": {str(b): v for b, v in dens.items()},
        "pair_2_3_density": pair_density_2_3(product_primes),
        "rows": [{**r, "bases": {str(b): v for b, v in r["bases"].items()}} for r in rows],
        "hooley_corrections": {str(b): v for b, v in corrections.items()},
        "index_law": {str(b): v for b, v in law.items()}, "local_rates": local,
        "references": {
            "hooley": "C. Hooley, On Artin's conjecture, J. reine angew. Math. 225 (1967) 209–220 (assumes GRH for Q(ζ_n, b^{1/n}))",
            "survey": "P. Moree, Artin's primitive root conjecture – a survey, Integers 12A (2012), arXiv:math/0412262",
            "heath_brown": "D. R. Heath-Brown, Artin's conjecture for primitive roots, Quart. J. Math. Oxford 37 (1986) 27–38",
        },
    }
    if cfg.get("search"):
        import artin_formulas

        out["formulas"] = artin_formulas.run_stage(cfg["search"], seed, run_dir, P, I, bases, local, A, log)
    out["seconds"] = round(time.time() - t0, 1)
    return out
