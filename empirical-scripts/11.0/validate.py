"""ROADMAP step 16: external validation of the final decoder on integers built from an external prime list.

The decoder validated is the published one, `sieve_echo_decoder.decode` (no training, no Evolvo). The prime
list — the 49 000 001st to 50 000 000th primes from The Prime Pages (t5k.org), about 9.6e8 to 9.8e8, beyond
every prime the earlier steps sieved — is used only to build test integers and to check the answers.

Protocol per integer n (known factorization F, built from the list):
1. the repetend lengths L_b = ord_n(b) are produced from F (lcm of the orders modulo the prime powers);
2. each L_b is verified *without F*: b^L ≡ 1 (mod n) and b^(L/r) ≢ 1 for every prime r | L (SymPy);
3. the decoder sees only n and {b: L_b}; its output must equal F exactly.
Families: single primes, products of 2 and 3 list primes, p²·q, a list prime times medium primes in (31, 10⁵)
— all coprime to every base, so the pre-period step (gcd(b, n), the only step that acts like trial division by
2…31) does nothing — and, separately, list primes times small primes, where it does.
"""

from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sympy

import runtime
import sieve_echo_decoder as sed

DEFAULTS: Dict[str, Any] = {
    "prime_list": "empirical-scripts/11.0/results/primes-db/cache/txt/primes50.txt",
    "source": "https://t5k.org/lists/small/millions/primes50.zip",
    "sha256_txt": None,
    "expected_count": 1000000,
    "families": {"prime": 5000, "two_primes": 20000, "three_primes": 5000, "square_times_prime": 2000,
                 "list_times_medium": 5000, "list_times_small": 3000},
    "medium_max": 100000,
    "gap_checks": 200,
    "workers": 0,
    "chunk": 500,
}

SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def load_primes(path: Path) -> np.ndarray:
    vals = [int(t) for t in path.read_text().split() if t.isdigit()]
    return np.array(vals, dtype=np.int64)


def check_list(P: np.ndarray, expected: int, gap_checks: int, rng) -> Dict[str, Any]:
    """The list is what it claims: count, sorted, and consecutive primes at random positions (SymPy nextprime)."""
    idx = rng.choice(len(P) - 1, size=min(gap_checks, len(P) - 1), replace=False)
    gaps_ok = all(int(sympy.nextprime(int(P[i]))) == int(P[i + 1]) for i in idx)
    return {"count": int(len(P)), "count_ok": int(len(P)) == expected, "sorted": bool(np.all(np.diff(P) > 0)),
            "first": int(P[0]), "last": int(P[-1]), "consecutive_checked": int(len(idx)), "consecutive_ok": bool(gaps_ok)}


def build(cfg: Dict[str, Any], P: np.ndarray, rng) -> List[tuple]:
    """[(family, {prime: exponent})] — distinct primes within an integer."""
    fam = cfg["families"]
    medium = [p for p in sympy.primerange(37, cfg["medium_max"])]
    out = []

    def pick(k):
        return [int(x) for x in P[rng.choice(len(P), size=k, replace=False)]]

    for _ in range(fam["prime"]):
        out.append(("prime", {pick(1)[0]: 1}))
    for _ in range(fam["two_primes"]):
        p, q = pick(2)
        out.append(("two_primes", {p: 1, q: 1}))
    for _ in range(fam["three_primes"]):
        out.append(("three_primes", {p: 1 for p in pick(3)}))
    for _ in range(fam["square_times_prime"]):
        p, q = pick(2)
        out.append(("square_times_prime", {p: 2, q: 1}))
    for _ in range(fam["list_times_medium"]):
        k = int(rng.integers(1, 5))
        f = {pick(1)[0]: 1}
        for j in rng.choice(len(medium), size=k, replace=False):
            f[int(medium[j])] = int(rng.integers(1, 3))
        out.append(("list_times_medium", f))
    for _ in range(fam["list_times_small"]):
        f = {p: 1 for p in pick(int(rng.integers(1, 3)))}
        for j in rng.choice(len(SMALL), size=int(rng.integers(1, 6)), replace=False):
            f[SMALL[j]] = int(rng.integers(1, 4))
        out.append(("list_times_small", f))
    return out


def _order_mod_prime_power(b: int, p: int, a: int, pm1_primes: List[int]) -> int:
    t = p - 1
    for r in pm1_primes:
        while t % r == 0 and pow(b, t // r, p) == 1:
            t //= r
    m = p**a
    while pow(b, t, m) != 1:
        t *= p
    return t


def lengths_from_factors(f: Dict[int, int]) -> Dict[int, int]:
    n = math.prod(p**e for p, e in f.items())
    pm1 = {p: sympy.primefactors(p - 1) for p in f}
    out = {}
    for b in sed.BASES:
        if math.gcd(b, n) == 1:
            out[b] = math.lcm(*(_order_mod_prime_power(b, p, e, pm1[p]) for p, e in f.items()))
    return out


def verify_length(n: int, b: int, L: int) -> bool:
    """L = ord_n(b), checked from n, b and L alone."""
    return pow(b, L, n) == 1 and all(pow(b, L // r, n) != 1 for r in sympy.primefactors(L))


def _chunk(items: List[tuple]) -> List[tuple]:
    res = []
    for fam, f in items:
        n = math.prod(p**e for p, e in f.items())
        L = lengths_from_factors(f)
        verified = all(verify_length(n, b, t) for b, t in L.items())
        t0 = time.perf_counter()
        got = sed.decode(n, L)
        dt = time.perf_counter() - t0
        res.append((fam, n, f, verified, got, len(L), dt))
    return res


def run_stage(vcfg: Dict[str, Any], seed: int, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(vcfg or {})}
    cfg["families"] = {**DEFAULTS["families"], **cfg.get("families", {})}
    path = runtime.REPO_ROOT / cfg["prime_list"]
    if not path.is_file():
        raise SystemExit(f"validate: prime list {cfg['prime_list']} missing — download {cfg['source']} and unzip it there")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if cfg["sha256_txt"] and digest != cfg["sha256_txt"]:
        raise SystemExit(f"validate: {cfg['prime_list']} has SHA-256 {digest}, expected {cfg['sha256_txt']}")
    rng = np.random.default_rng(seed)
    P = load_primes(path)
    lst = check_list(P, cfg["expected_count"], cfg["gap_checks"], rng)
    log(f"  16a: prime list {lst['count']} primes {lst['first']}…{lst['last']}, sorted {lst['sorted']}, "
        f"{lst['consecutive_checked']} consecutive pairs checked: {lst['consecutive_ok']}")
    if not (lst["count_ok"] and lst["sorted"] and lst["consecutive_ok"]):
        raise SystemExit("validate: the prime list failed its checks")
    items = build(cfg, P, rng)
    used = sorted({p for _, f in items for p in f if p > cfg["medium_max"]})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=runtime_workers(cfg["workers"])) as ex:
        prime_ok = all(ex.map(sympy.isprime, used, chunksize=2000))
        chunks = [items[i:i + cfg["chunk"]] for i in range(0, len(items), cfg["chunk"])]
        rows = [r for part in ex.map(_chunk, chunks) for r in part]
    log(f"  16b: {len(used)} list primes used, all prime by SymPy: {prime_ok}; {len(rows)} integers decoded in {time.time() - t0:.0f}s")
    fams: Dict[str, Dict[str, Any]] = {}
    for fam, n, f, verified, got, ncop, dt in rows:
        d = fams.setdefault(fam, {"integers": 0, "exact": 0, "lengths_verified": 0, "all_19_bases_coprime": 0,
                                  "n_min": n, "n_max": n, "omega": {}, "seconds": 0.0, "failures": []})
        d["integers"] += 1
        d["lengths_verified"] += verified
        d["all_19_bases_coprime"] += ncop == len(sed.BASES)
        d["n_min"], d["n_max"] = min(d["n_min"], n), max(d["n_max"], n)
        w = str(len(f))
        d["omega"][w] = d["omega"].get(w, 0) + 1
        d["seconds"] += dt
        if got == f:
            d["exact"] += 1
        elif len(d["failures"]) < 20:
            d["failures"].append({"n": n, "factors": {str(k): v for k, v in f.items()}, "decoded": {str(k): v for k, v in got.items()}})
    for fam, d in fams.items():
        d["accuracy"] = d["exact"] / d["integers"]
        d["mean_ms"] = 1000 * d["seconds"] / d["integers"]
        log(f"    {fam}: {d['exact']}/{d['integers']} exact, lengths verified {d['lengths_verified']}, "
            f"n in [{d['n_min']:.3g}, {d['n_max']:.3g}], {d['mean_ms']:.1f} ms per integer")
    total = sum(d["integers"] for d in fams.values())
    exact = sum(d["exact"] for d in fams.values())
    return {"settings": cfg, "prime_list": {**lst, "path": cfg["prime_list"], "source": cfg["source"], "sha256": digest,
                                            "primes_used": len(used), "primes_used_all_prime": bool(prime_ok)},
            "families": fams, "integers": total, "exact": exact, "accuracy": exact / total,
            "decoder": "sieve_echo_decoder.decode", "decoder_sha256": runtime.sha256_file(Path(sed.__file__)),
            "seconds": time.time() - t0}


def runtime_workers(w: int) -> int:
    import os

    return w if w and w > 0 else max(1, (os.cpu_count() or 2) - 1)
