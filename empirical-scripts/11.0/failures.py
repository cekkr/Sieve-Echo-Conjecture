"""Step 15: the failure set of the step-13 decoder, exactly (ROADMAP step 15).

Proposition 1.8 of `studies/9` (tests: `tests/test_failures.py`). Let B be a set of bases in which every
prime dividing a base is itself a base, and P_B its primes (the 19 bases of this study qualify:
P_B = {2, 3, 5, …, 31}). Run the decoder of Proposition 1.7 with pre-periods (gcd(b, n) for the
bases that share a factor with n). Then
(a) every prime of P_B dividing n is isolated (gcd(p, n) = p for the base p);
(b) primes p, p' ∉ P_B dividing n are separated iff ord_p(b) ≠ ord_{p'}(b) for some b ∈ B coprime to n
    when n is squarefree; prime powers can only separate more. So a squarefree n fails iff two of its
    prime factors outside P_B share their order vector over B_n = {b ∈ B : gcd(b, n) = 1}, and every
    failing n has such a pair;
(c) B_n depends only on S = {p ∈ P_B : p | n}, hence the least failing integer is
        N*(B) = min_{S ⊆ P_B} ∏S · min{p·p' : p < p' ∉ P_B, (ord_p b)_{b ∈ B_S} = (ord_{p'} b)_{b ∈ B_S}}.
The search below computes the orders of every base modulo every prime up to X, finds equal order
vectors by hashing, and evaluates (c) over all patterns S (pruned by ∏S). When a pattern has no
collision among the primes ≤ X, its term is bounded below by ∏S · p₀ · X⁺ (p₀ the least prime outside
P_B, X⁺ the first prime above X), so the minimum over S is a certified statement either way.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import decode
import ndr
import orders_scale

DEFAULTS: Dict[str, Any] = {
    "bases": decode.BASES19,
    "prime_limit": 100_000_000,
    "prefix_checks": list(range(1, 19)),
    "brute_limit": 3_000_000,
    "workers": 0,
}


def sieve_primes(lo: int, hi: int) -> np.ndarray:
    """Primes in [lo, hi] (Eratosthenes on a byte array)."""
    if hi < 2:
        return np.zeros(0, dtype=np.int64)
    a = np.ones(hi + 1, dtype=bool)
    a[:2] = False
    for p in range(2, math.isqrt(hi) + 1):
        if a[p]:
            a[p * p::p] = False
    out = np.nonzero(a)[0]
    return out[out >= lo].astype(np.int64)


def base_primes(bases: Sequence[int]) -> List[int]:
    return sorted({p for b in bases for p, _ in ndr.factorize(int(b), ndr.spf_sieve(max(bases) + 1))})


def prime_closed(bases: Sequence[int]) -> bool:
    return all(p in bases for p in base_primes(bases))


_P: Dict[str, Any] = {}


def _init(small, bases):
    _P["small"] = [int(x) for x in small]
    _P["bases"] = [int(b) for b in bases]


def _orders_chunk(ps):
    small, bases = _P["small"], _P["bases"]
    out = np.zeros((len(ps), len(bases)), dtype=np.int32)  # orders < p < 2^31
    for i, p in enumerate(ps):
        lam = p - 1
        lp = [q for q, _ in orders_scale.factorize_td(lam, small)] if lam > 1 else []
        for j, b in enumerate(bases):
            out[i, j] = ndr.multiplicative_order(b % p, p, lam, lp) if p > 2 else 1
    return out


def order_matrix(primes: np.ndarray, bases: Sequence[int], workers: int) -> np.ndarray:
    """rows × bases matrix of ord_p(b) for primes p not dividing any base."""
    small = sieve_primes(2, math.isqrt(int(primes.max()) if primes.size else 4) + 2)
    ps = [int(p) for p in primes.tolist()]
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    size = max(2000, len(ps) // (workers * 16) + 1)
    chunks = [ps[i:i + size] for i in range(0, len(ps), size)]
    if workers <= 1 or len(chunks) <= 1:
        _init(small, bases)
        parts = [_orders_chunk(c) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(small, bases)) as pool:
            parts = list(pool.map(_orders_chunk, chunks))
    return np.concatenate(parts) if parts else np.zeros((0, len(bases)), dtype=np.int32)


_W = np.random.default_rng(12345).integers(1, 2 ** 63, size=64, dtype=np.uint64) | np.uint64(1)


def collisions(P: np.ndarray, O: np.ndarray, cols: Sequence[int]) -> Dict[str, Any]:
    """Groups of primes with equal order vectors over the columns: the least product p·p' and group counts.

    P must be ascending; within a group the two smallest primes give the least product."""
    if P.size < 2:
        return {"groups": 0, "primes_in_groups": 0, "min_product": None, "pair": None}
    if len(cols) == 0:
        return {"groups": 1, "primes_in_groups": int(P.size), "min_product": int(P[0]) * int(P[1]), "pair": [int(P[0]), int(P[1])]}
    X = O[:, list(cols)].astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (X * _W[: X.shape[1]]).sum(axis=1, dtype=np.uint64)
        h ^= (X[:, 0] * np.uint64(0x9E3779B97F4A7C15))
    order = np.lexsort((P, h))
    hs = h[order]
    start = np.ones(hs.size, dtype=bool)
    start[1:] = hs[1:] != hs[:-1]
    idx = np.nonzero(start)[0]
    sizes = np.diff(np.append(idx, hs.size))
    multi = idx[sizes >= 2]
    if multi.size == 0:
        return {"groups": 0, "primes_in_groups": 0, "min_product": None, "pair": None}
    a, b = order[multi], order[multi + 1]
    same = (X[a] == X[b]).all(axis=1)
    if not same.all():  # a 64-bit hash collision between different vectors: resolve exactly
        raise RuntimeError("hash collision between different order vectors; change _W")
    pa, pb = P[a].tolist(), P[b].tolist()
    k = min(range(len(pa)), key=lambda i: pa[i] * pb[i])
    return {"groups": int(multi.size), "primes_in_groups": int(sizes[sizes >= 2].sum()), "min_product": int(pa[k]) * int(pb[k]), "pair": [int(pa[k]), int(pb[k])]}


def certificate(bases: Sequence[int], P: np.ndarray, O: np.ndarray, limit: int) -> Dict[str, Any]:
    """Least failing n of the decoder with pre-periods over all small-prime patterns S (Proposition 1.8c)."""
    bases = [int(b) for b in bases]
    if not prime_closed(bases):
        raise ValueError("Proposition 1.8 needs every prime dividing a base to be a base")
    PB = base_primes(bases)
    keep = np.ones(P.size, dtype=bool)
    for q in PB:
        keep &= P != q
    P, O = P[keep], O[keep]
    p0 = int(P[0])
    xplus = int(sieve_primes(limit + 1, limit + 2000)[0])
    pats = []
    for r in range(len(PB) + 1):
        for S in itertools.combinations(PB, r):
            pats.append((math.prod(S), S))
    pats.sort()
    best: Optional[Dict[str, Any]] = None
    bound: Optional[Dict[str, Any]] = None
    searched = 0
    for m, S in pats:
        lowest = m * p0 * int(P[1])
        known = [x["n"] for x in (best, bound) if x]
        cur = min(known) if known else None
        if cur is not None and lowest >= cur:
            continue
        cols = [j for j, b in enumerate(bases) if math.gcd(b, m) == 1]
        cap = limit if cur is None else min(limit, cur // (m * p0) + 1)
        sel = P <= cap
        c = collisions(P[sel], O[sel], cols)
        searched += 1
        if c["min_product"] is not None:
            n = m * c["min_product"]
            if best is None or n < best["n"]:
                best = {"n": n, "pattern": list(S), "pair": c["pair"], "bases_coprime": [bases[j] for j in cols]}
        elif cap >= limit:  # truncated search: certified lower bound for this pattern
            nb = m * p0 * xplus
            if bound is None or nb < bound["n"]:
                bound = {"n": nb, "pattern": list(S), "bases_coprime": [bases[j] for j in cols]}
    exact = best is not None and (bound is None or best["n"] <= bound["n"])
    return {"bases": bases, "base_primes": PB, "prime_limit": int(limit), "least_prime_outside": p0, "patterns": len(pats), "patterns_searched": searched,
            "first_failure": best, "lower_bound": bound, "exact": bool(exact),
            "certified_exact_below": int(best["n"] if exact else bound["n"]) if (best or bound) else None}


# ---------------------------------------------------------------------------------------------
# Brute force (validation)
# ---------------------------------------------------------------------------------------------


def _brute_chunk(args):
    lo, hi = args
    bases = _P["bases"]
    primes = _P["small"]
    for n in range(lo, hi):
        f = orders_scale.factorize_td(n, primes)
        lam = ndr.carmichael_from_factors(f)
        lp = [q for q, _ in orders_scale.factorize_td(lam, primes)] if lam > 1 else []
        L = [0 if math.gcd(b, n) != 1 else ndr.multiplicative_order(b, n, lam, lp) for b in bases]
        if decode.decode(n, bases, L, lambda t: decode.prime_divisors(t, lp), preperiod=True) != len(f):
            return n
    return None


def decodes(n: int, bases: Sequence[int]) -> bool:
    """True when the decoder with pre-periods gives the exact ω(n) (trial-division factorization for the check)."""
    small = sieve_primes(2, math.isqrt(n) + 2).tolist()
    f = orders_scale.factorize_td(n, small)
    lam = ndr.carmichael_from_factors(f)
    lp = [q for q, _ in orders_scale.factorize_td(lam, small)] if lam > 1 else []
    L = [0 if math.gcd(b, n) != 1 else ndr.multiplicative_order(b, n, lam, lp) for b in bases]
    return decode.decode(n, list(bases), L, lambda t: decode.prime_divisors(t, lp), preperiod=True) == len(f)


def first_failure_brute(bases: Sequence[int], n_max: int, workers: int, chunk: int = 50_000) -> Optional[int]:
    """Least n ≤ n_max on which the decoder with pre-periods misses ω (None if none)."""
    small = sieve_primes(2, math.isqrt(n_max) + 2)
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    tasks = [(lo, min(lo + chunk, n_max + 1)) for lo in range(2, n_max + 1, chunk)]
    if workers <= 1 or len(tasks) <= 1:
        _init(small, bases)
        res = [_brute_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(small, bases)) as pool:
            res = list(pool.map(_brute_chunk, tasks))
    hits = [r for r in res if r is not None]
    return min(hits) if hits else None


def run_stage(fcfg_in: Dict[str, Any], seed: int, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(fcfg_in or {})}
    t0 = time.time()
    bases = [int(b) for b in cfg["bases"]]
    X = int(cfg["prime_limit"])
    PB = base_primes(bases)
    P = sieve_primes(2, X)
    P = P[~np.isin(P, PB)]
    log(f"  15b: {P.size} primes ≤ {X} outside {PB}; orders in {len(bases)} bases")
    t1 = time.time()
    O = order_matrix(P, bases, int(cfg["workers"]))
    log(f"  15b: order matrix in {time.time() - t1:.0f}s")
    full = certificate(bases, P, O, X)
    allcols = collisions(P, O, list(range(len(bases))))
    full["collisions_all_bases"] = allcols
    ff = full["first_failure"]
    log(f"  15b: 19 bases: least failing n = {ff['n'] if ff else None} ({'exact' if full['exact'] else 'bound ' + str(full['lower_bound'])}); "
        f"pairs of primes ≤ {X} with equal orders in all {len(bases)} bases: {allcols['groups']} groups, least product {allcols['min_product']}")
    checks = []
    for k in cfg["prefix_checks"]:
        bk = bases[:int(k)]
        if not prime_closed(bk):
            continue
        PBk = base_primes(bk)
        extra = np.array([q for q in PB if q not in PBk], dtype=np.int64)  # primes of the full set that the prefix treats as "large"
        Pk = np.concatenate([extra, P])
        Ok = np.concatenate([order_matrix(extra, bk, 1), O[:, :int(k)]]) if extra.size else O[:, :int(k)]
        srt = np.argsort(Pk, kind="stable")
        cert = certificate(bk, Pk[srt], Ok[srt], X)
        pred = cert["first_failure"]["n"] if cert["first_failure"] and cert["exact"] else None
        rec: Dict[str, Any] = {"k": int(k), "bases": bk, "certificate": cert, "predicted_first_failure": pred}
        if pred is not None and pred <= int(cfg["brute_limit"]):
            t1 = time.time()
            got = first_failure_brute(bk, pred, int(cfg["workers"]))
            rec["brute_first_failure"] = got
            rec["agrees"] = bool(got == pred)
            log(f"  15c: first {k} bases: predicted {pred}, brute force {got} ({time.time() - t1:.0f}s)")
        else:
            ff = cert["first_failure"]
            if ff:  # the least failure found (exact or not) must at least be a failure
                rec["fails_at_found"] = not decodes(ff["n"], bk)
            log(f"  15c: first {k} bases: predicted {pred} (beyond the brute-force limit {cfg['brute_limit']}); "
                f"decoder fails at the least failure found: {rec.get('fails_at_found')}")
        checks.append(rec)
    return {"settings": cfg, "full": full, "prefix_checks": checks, "seconds": round(time.time() - t0, 1)}
