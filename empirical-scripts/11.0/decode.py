"""Step 13: exact decoding of the factorization from the repetends of 1/n (ROADMAP step 13).

Steps 10–11 read ω from the orders statistically (2-rank estimators, learned models). This module
decodes it exactly, by the classical reduction of factoring to order finding (Miller 1976; the
reduction Shor's algorithm uses). For a base b coprime to n, L = ord_n(b) = lcm_p ord_{p^a}(b) over
the prime powers p^a ‖ n, and b^d ≡ 1 (mod p) iff ord_p(b) | d. For each prime q | L with
e = v_q(L) and m = L / q^e, the remainders x_i = b^{m q^i} mod n (0 ≤ i < e) give the tests

    g_i = gcd(x_i − 1, n),   p | g_i  ⇔  v_q(ord_p(b)) ≤ i.

Refining {n} by every test of every base and reducing to a coprime base gives pairwise coprime
parts; ω̂ = their number.

Proposition 1.7 of `studies/9` (tests: `tests/test_decode.py`):
(a) ω̂(n; B) ≤ ω(n). Primes p, p' | n with ord_p(b) ≠ ord_{p'}(b) for some base b ∈ B coprime to n
    always end in different parts, so ω̂(n; B) ≥ #{(ord_p(b))_{b ∈ B} : p | n}, with equality for
    squarefree n (for p² | n the orders ord_{p^a}(b) can carry a factor p and separate more). In
    particular ω̂ = ω whenever the order vectors of the prime divisors of n are distinct.
(b) For odd n and even L, gcd(b^{L/2} − 1, n) = 1 ⇔ b^{L/2} ≡ −1 (mod n): Shor's half-period split
    fails exactly on the Midy pairs of step 7 (Proposition 1.4).
(c) x_i is the remainder after m q^i digits of the long division of 1 by n in base b. It is read
    exactly from k = ⌈log_b n⌉ repetend digits starting there: with D their integer value,
    r_d = ⌊n D / b^k⌋ + 1 (`remainder_from_digits`). The decoder therefore needs n, the repetend
    lengths and O(log n) digits of each repetend per test — no spectrum and no factorization.

Inputs are n and the orders (or the digits): the factorization is used only for labels and, as a
speed shortcut, for the prime divisors of L (q | L ⇒ q | λ(n), so [q for q | λ if L % q == 0] is the
same set as factoring L; checked in the tests). Computing L itself is the hard step: classically it
needs λ(n) or Θ(L) digits; Shor's algorithm finds it with a quantum Fourier transform.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

import dataset
import lengths
import ndr
import orders_scale

BASES19 = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31]

DEFAULTS: Dict[str, Any] = {
    "bases": BASES19,
    "prime_bases": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
    "decades": [[10 ** k, 10 ** (k + 1) - 1] for k in range(4, 12)],
    "per_class": 3000,
    "max_class": 5,
    "batch": 200000,
    "max_batches": 60,
    "uniform_per_decade": 20000,
    "exhaustive_max": 10_000_000,
    "exhaustive_chunk": 50_000,
    "max_failures": 200,
    "digit_check": {"decades": [[10000, 99999], [100000, 999999]], "pairs": 4000, "max_L": 200000, "decode_integers": 600},
    "workers": 0,
}


# ---------------------------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------------------------


def coprime_base(xs: Iterable[int]) -> List[int]:
    """Pairwise coprime integers > 1 with the same prime support as xs, splitting by gcds."""
    xs = [int(x) for x in xs if x > 1]
    while True:
        hit = None
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                g = math.gcd(xs[i], xs[j])
                if g > 1:
                    hit = (i, j, g)
                    break
            if hit:
                break
        if hit is None:
            return xs
        i, j, g = hit
        a, b = xs[i] // g, xs[j] // g
        xs = [x for k, x in enumerate(xs) if k not in (i, j)] + [y for y in (g, a, b) if y > 1]


def refine(parts: List[int], g: int) -> List[int]:
    if g <= 1:
        return parts
    out = []
    for f in parts:
        a = math.gcd(f, g)
        if 1 < a < f:
            out += [a, f // a]
        else:
            out.append(f)
    return out


def prime_divisors(L: int, lam_primes: Sequence[int]) -> List[int]:
    """The primes q | L, for L | λ(n) (lam_primes = the primes of λ(n))."""
    return [q for q in lam_primes if L % q == 0]


def split_exponents(L: int, qs: Sequence[int]):
    """Yield (q, d) for the test exponents d = (L / q^e) q^i, 0 ≤ i < e = v_q(L)."""
    for q in qs:
        e, m = 0, L
        while m % q == 0:
            m //= q
            e += 1
        d = m
        for _ in range(e):
            yield q, d
            d *= q


def decode(n: int, bases: Sequence[int], L: Sequence[int], qs_of: Callable[[int], Sequence[int]], *, only_q: Optional[int] = None,
           preperiod: bool = False, remainder: Optional[Callable[[int, int], int]] = None, checkpoints: bool = False):
    """ω̂ after all bases (or after each base when checkpoints=True). L[j] = 0 marks gcd(b, n) > 1.

    remainder(b, d) must return b^d mod n; the default is pow, `remainder_from_digits` reads it from the digits.
    """
    rem = remainder or (lambda b, d: pow(b, d, n))
    parts = [int(n)]
    marks = []
    for b, t in zip(bases, L):
        t = int(t)
        if t == 0:
            if preperiod:
                parts = refine(parts, math.gcd(b, n))
        elif t > 1:
            qs = qs_of(t)
            if only_q is not None:
                qs = [q for q in qs if q == only_q]
            for _, d in split_exponents(t, qs):
                parts = refine(parts, math.gcd(rem(b, d) - 1, n))
        if checkpoints:
            parts = coprime_base(parts)
            marks.append(len(parts))
    if checkpoints:
        return marks
    return len(coprime_base(parts))


def digits_needed(n: int, b: int) -> int:
    k, p = 0, 1
    while p < n:
        p *= b
        k += 1
    return k


def remainder_from_digits(digits: Sequence[int], b: int, n: int, d: int) -> int:
    """b^d mod n read from the repetend digits of 1/n in base b (Proposition 1.7c).

    digits[i] = ⌊b r_i / n⌋ with r_i = b^i mod n, so r_d / n = Σ_{j ≥ 1} digits[d + j − 1] b^{−j} (cyclically).
    With D the integer of the k = ⌈log_b n⌉ digits from position d, n D / b^k < r_d < n D / b^k + n / b^k ≤ n D / b^k + 1.
    """
    L = len(digits)
    k = digits_needed(n, b)
    D = 0
    for i in range(k):
        D = D * b + int(digits[(d + i) % L])
    return (n * D) // b ** k + 1


def order_vector_classes(n: int, bases: Sequence[int], factors) -> int:
    """#{(ord_p(b))_{b coprime to n} : p | n} — the lower bound of Proposition 1.7(a), for tests (uses the factorization)."""
    vecs = set()
    cop = [b for b in bases if math.gcd(b, n) == 1]
    for p, _ in factors:
        lam = p - 1
        lp = [q for q, _ in ndr.factorize(lam, _SPF_SMALL())] if lam > 1 else []
        vecs.add(tuple(ndr.multiplicative_order(b, p, lam, lp) if p > 2 else 1 for b in cop))
    return len(vecs)


_SPF_CACHE: Dict[str, Any] = {}


def _SPF_SMALL():
    if "spf" not in _SPF_CACHE:
        _SPF_CACHE["spf"] = ndr.spf_sieve(2_000_000)
    return _SPF_CACHE["spf"]


# ---------------------------------------------------------------------------------------------
# Per-integer records (workers)
# ---------------------------------------------------------------------------------------------

_P: Dict[str, Any] = {}


def _init(primes, bases, prime_bases):
    _P["primes"] = [int(p) for p in primes]
    _P["bases"] = [int(b) for b in bases]
    _P["pidx"] = [list(bases).index(b) for b in prime_bases]


def _context(n: int):
    f = orders_scale.factorize_td(n, _P["primes"])
    lam = ndr.carmichael_from_factors(f)
    lp = [p for p, _ in orders_scale.factorize_td(lam, _P["primes"])] if lam > 1 else []
    return f, lam, lp


def _orders(n: int, lam: int, lp) -> List[int]:
    return [0 if math.gcd(b, n) != 1 else ndr.multiplicative_order(b, n, lam, lp) for b in _P["bases"]]


def _record(n: int) -> tuple:
    """(ω, ω̂ after each of the bases, ω̂ q=2 only, ω̂ with pre-periods, ω̂ after each prime base, Midy=Shor-fail agreements, pairs)."""
    f, lam, lp = _context(n)
    L = _orders(n, lam, lp)
    qs = lambda t: prime_divisors(t, lp)  # noqa: E731
    bases = _P["bases"]
    marks = decode(n, bases, L, qs, checkpoints=True)
    shor = decode(n, bases, L, qs, only_q=2)
    pre = marks[-1] if marks[-1] == len(f) else decode(n, bases, L, qs, preperiod=True)
    pidx = _P["pidx"]
    pmarks = decode(n, [bases[j] for j in pidx], [L[j] for j in pidx], qs, checkpoints=True)
    agree = pairs = 0
    if n % 2 == 1:
        for b, t in zip(bases, L):
            if t > 0 and t % 2 == 0:
                pairs += 1
                midy = lengths.midy_flag(n, b, t)
                fail = math.gcd(pow(b, t // 2, n) - 1, n) == 1
                agree += int(bool(midy) == fail)
    return len(f), marks, shor, pre, pmarks, agree, pairs


def _records_chunk(ns):
    return [_record(int(n)) for n in ns]


def _exhaustive_chunk(args):
    lo, hi, max_fail = args
    nb = len(_P["bases"])
    correct = np.zeros(nb, dtype=np.int64)
    shor_ok = pre_ok = total = 0
    by_omega: Dict[int, List[int]] = {}
    fails = []
    for n in range(lo, hi):
        w, marks, shor, pre, _, _, _ = _record(n)
        total += 1
        correct += np.array([m == w for m in marks], dtype=np.int64)
        shor_ok += int(shor == w)
        pre_ok += int(pre == w)
        c = by_omega.setdefault(w, [0, 0])
        c[0] += 1
        c[1] += int(marks[-1] == w)
        if marks[-1] != w and len(fails) < max_fail:
            fails.append([n, w, marks[-1], pre])
    return total, correct, shor_ok, pre_ok, by_omega, fails


def _pool(workers: int, primes, bases, prime_bases):
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    return workers, (lambda: ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(primes, bases, prime_bases)))


def records(ns: Sequence[int], primes, bases, prime_bases, workers: int) -> List[tuple]:
    workers, make = _pool(workers, primes, bases, prime_bases)
    ns = [int(x) for x in ns]
    size = max(50, len(ns) // (workers * 8) + 1)
    chunks = [ns[i:i + size] for i in range(0, len(ns), size)]
    if workers <= 1 or len(chunks) <= 1:
        _init(primes, bases, prime_bases)
        return [r for c in chunks for r in _records_chunk(c)]
    with make() as pool:
        return [r for part in pool.map(_records_chunk, chunks) for r in part]


def exhaustive(n_max: int, chunk: int, max_fail: int, primes, bases, prime_bases, workers: int) -> Dict[str, Any]:
    workers, make = _pool(workers, primes, bases, prime_bases)
    tasks = [(lo, min(lo + chunk, n_max + 1), max_fail) for lo in range(2, n_max + 1, chunk)]
    if workers <= 1 or len(tasks) <= 1:
        _init(primes, bases, prime_bases)
        parts = [_exhaustive_chunk(t) for t in tasks]
    else:
        with make() as pool:
            parts = list(pool.map(_exhaustive_chunk, tasks))
    total = sum(p[0] for p in parts)
    correct = sum(p[1] for p in parts)
    by: Dict[int, List[int]] = {}
    for p in parts:
        for w, (c, ok) in p[4].items():
            v = by.setdefault(int(w), [0, 0])
            v[0] += c
            v[1] += ok
    fails = [f for p in parts for f in p[5]][:max_fail]
    return {"n_range": [2, int(n_max)], "integers": int(total), "bases": list(map(int, bases)),
            "accuracy_by_prefix": [float(c / total) for c in correct], "failures_all_bases": int(total - correct[-1]),
            "accuracy": float(correct[-1] / total), "accuracy_shor_only": float(sum(p[2] for p in parts) / total),
            "accuracy_with_preperiod": float(sum(p[3] for p in parts) / total),
            "by_omega": {w: {"integers": c, "correct": ok} for w, (c, ok) in sorted(by.items())}, "failures": fails}


# ---------------------------------------------------------------------------------------------
# Summaries and the stage
# ---------------------------------------------------------------------------------------------


def summarize(ns: np.ndarray, recs: List[tuple], bases: Sequence[int], prime_bases: Sequence[int], K: int) -> Dict[str, Any]:
    w = np.array([r[0] for r in recs])
    marks = np.array([r[1] for r in recs])
    pm = np.array([r[4] for r in recs])
    final = marks[:, -1]
    ok = marks == w[:, None]
    need = np.where(ok.any(axis=1), ok.argmax(axis=1) + 1, len(bases) + 1)  # bases (in order) until ω̂ = ω
    cls = np.minimum(w, K)
    out = {"integers": int(w.size), "accuracy": float((final == w).mean()), "failures": int((final != w).sum()),
           "accuracy_by_prefix": [float(x) for x in ok.mean(axis=0)],
           "accuracy_prime_bases_by_prefix": [float(x) for x in (pm == w[:, None]).mean(axis=0)],
           "accuracy_shor_only": float((np.array([r[2] for r in recs]) == w).mean()),
           "accuracy_with_preperiod": float((np.array([r[3] for r in recs]) == w).mean()),
           "bases_needed": {"mean": float(need[need <= len(bases)].mean()) if (need <= len(bases)).any() else float("nan"),
                            "p99": float(np.percentile(need, 99)), "max": int(need.max())},
           "by_class": {int(c): {"integers": int((cls == c).sum()), "accuracy": float((final[cls == c] == w[cls == c]).mean()),
                                 "accuracy_shor_only": float((np.array([r[2] for r in recs])[cls == c] == w[cls == c]).mean())} for c in np.unique(cls)},
           "underestimates_only": bool((final <= w).all()),
           "midy_equals_shor_failure": {"agree": int(sum(r[5] for r in recs)), "pairs": int(sum(r[6] for r in recs))},
           "failure_examples": [[int(ns[i]), int(w[i]), int(final[i])] for i in np.nonzero(final != w)[0][:20]]}
    return out


def digit_check(n: np.ndarray, cfg: Dict[str, Any], primes, bases, seed: int, log=print) -> Dict[str, Any]:
    """13c: remainders read from repetend digits equal b^d mod n; whole integers decoded from digits only."""
    _init(primes, bases, bases)
    rng = np.random.default_rng([seed, 131])
    rows = np.concatenate([np.nonzero((n >= lo) & (n <= hi))[0] for lo, hi in cfg["decades"]])
    rows = rng.permutation(rows)
    max_L = int(cfg["max_L"])
    checked = mismatch = 0
    digits_read = []
    for i in rows:
        if checked >= int(cfg["pairs"]):
            break
        nn = int(n[i])
        _, lam, lp = _context(nn)
        b = int(rng.choice(bases))
        if math.gcd(b, nn) != 1:
            continue
        t = ndr.multiplicative_order(b, nn, lam, lp)
        if t > max_L or t < 2:
            continue
        dg = ndr.repetend_digits(nn, b, t)
        for _, d in split_exponents(t, prime_divisors(t, lp)):
            checked += 1
            mismatch += int(remainder_from_digits(dg, b, nn, d) != pow(b, d, nn))
        digits_read.append(digits_needed(nn, b))
    agree = tried = 0
    for i in rows:
        if tried >= int(cfg["decode_integers"]):
            break
        nn = int(n[i])
        f, lam, lp = _context(nn)
        L = _orders(nn, lam, lp)
        if max(L) > max_L:
            continue
        dig = {b: ndr.repetend_digits(nn, b, t) for b, t in zip(bases, L) if t > 1}
        qs = lambda t: prime_divisors(t, lp)  # noqa: E731
        via_digits = decode(nn, bases, L, qs, remainder=lambda b, d: remainder_from_digits(dig[b], b, nn, d))
        tried += 1
        agree += int(via_digits == decode(nn, bases, L, qs) == len(f))
    log(f"  13c: {checked} remainders read from digits, {mismatch} mismatches; {agree}/{tried} integers decoded from digits alone")
    return {"remainders_checked": checked, "mismatches": mismatch, "digits_per_reading": {"mean": float(np.mean(digits_read)) if digits_read else float("nan"), "max": int(max(digits_read)) if digits_read else 0},
            "integers_decoded_from_digits": tried, "integers_correct": agree, "max_L": max_L}


def run_stage(dcfg_in: Dict[str, Any], seed: int, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(dcfg_in or {})}
    cfg["digit_check"] = {**DEFAULTS["digit_check"], **(cfg.get("digit_check") or {})}
    t0 = time.time()
    bases = dataset.parse_bases(cfg["bases"]) if not isinstance(cfg["bases"], list) else [int(b) for b in cfg["bases"]]
    pbases = [int(b) for b in cfg["prime_bases"]]
    if any(b not in bases for b in pbases):
        raise ValueError("prime_bases must be a subset of bases")
    K = int(cfg["max_class"])
    hi = max(h for _, h in cfg["decades"])
    primes = orders_scale._primes_upto(int(math.isqrt(max(hi, int(cfg["exhaustive_max"])))) + 1)
    out: Dict[str, Any] = {"settings": cfg, "bases": bases, "prime_bases": pbases}
    if int(cfg["exhaustive_max"]) >= 2:
        t1 = time.time()
        out["exhaustive"] = exhaustive(int(cfg["exhaustive_max"]), int(cfg["exhaustive_chunk"]), int(cfg["max_failures"]), primes, bases, pbases, int(cfg["workers"]))
        ex = out["exhaustive"]
        log(f"  13d: exhaustive n ≤ {cfg['exhaustive_max']}: accuracy {ex['accuracy']:.7f} ({ex['failures_all_bases']} failures), "
            f"q=2 only {ex['accuracy_shor_only']:.5f}, {time.time() - t1:.0f}s")
    log(f"  13d: ω-balanced sample, {cfg['per_class']} per class min(ω, {K}) in {len(cfg['decades'])} decades")
    n, cls, scheme = orders_scale.sample(cfg["decades"], int(cfg["per_class"]), K, int(cfg["batch"]), int(cfg["max_batches"]), primes, seed, log)
    t1 = time.time()
    recs = records(n, primes, bases, pbases, int(cfg["workers"]))
    log(f"  13d: decoded {n.size} integers in {time.time() - t1:.0f}s")
    w = np.array([r[0] for r in recs])
    assert np.array_equal(np.minimum(w, K), cls), "ω from trial division disagrees with the sampler"
    dec = []
    for d, (lo, hi_) in enumerate(cfg["decades"]):
        rows = np.nonzero((n >= lo) & (n <= hi_))[0]
        s = summarize(n[rows], [recs[i] for i in rows], bases, pbases, K)
        s["decade"] = [int(lo), int(hi_)]
        dec.append(s)
        log(f"    balanced [{lo}, {hi_}]: accuracy {s['accuracy']:.5f} ({s['failures']} failures), first 5 bases {s['accuracy_by_prefix'][4]:.4f}, q=2 only {s['accuracy_shor_only']:.4f}")
    out.update({"sampling": scheme, "integers": int(n.size), "balanced": dec})
    U = int(cfg["uniform_per_decade"])
    if U > 0:
        uni = []
        for d, (lo, hi_) in enumerate(cfg["decades"]):
            rng = np.random.default_rng([seed, 130, d])
            un = np.unique(rng.integers(lo, hi_ + 1, size=U, dtype=np.int64))
            ur = records(un, primes, bases, pbases, int(cfg["workers"]))
            s = summarize(un, ur, bases, pbases, K)
            s["decade"] = [int(lo), int(hi_)]
            uni.append(s)
            log(f"    uniform [{lo}, {hi_}]: {un.size} integers, accuracy {s['accuracy']:.5f} ({s['failures']} failures)")
        out["uniform"] = uni
    out["digit_check"] = digit_check(n, cfg["digit_check"], primes, bases, seed, log)
    out["seconds"] = round(time.time() - t0, 1)
    return out
