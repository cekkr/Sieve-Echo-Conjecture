"""Canonical number theory, repetends, NDR and spectral entropy for Sieve Echo v11.

Every definition used by v11 lives here and is tagged by `NDR_VERSION`. Change the tag whenever a
definition below changes; every result file records it (AGENTS.md "One canonical definition").

Definitions (version ndr-v11.1)
-------------------------------
* Repetend of 1/n in base b: defined only for gcd(n, b) = 1, n >= 2. It is purely periodic with
  length L = ord_n(b). Remainders r_0 = 1, r_{k+1} = b*r_k mod n; digits d_k = floor(b*r_k / n).
* L is computed exactly from the factorization of the Carmichael function lambda(n), never by
  simulating long division.
* NDR (paper Def. 1.1): x_k = d_k / b.
* Power spectrum: P_k = |DFT(x)_k|^2 over all L bins k = 0..L-1 (computed with an rFFT and the
  conjugate-symmetry weights, which is exact for real input).
* Entropy: natural log, 0*log 0 = 0, no thresholding.
    - `H_paper`  : entropy of P_k / sum(P) over all L bins, DC kept (paper Def. 1.2).
    - `H_nodc`   : same over bins 1..L-1 (equivalently, the mean-removed NDR). NaN if the NDR is
                   constant (no AC power).
    - `Hn_paper` : H_paper / log L (NaN for L = 1).  `Hn_nodc`: H_nodc / log(L - 1) (NaN for L <= 2).
  Scaling x by a constant leaves all of these unchanged, so d/b and d/(b-1) give equal entropies.
* Historical variants, kept only to reproduce old stored results:
    - `H_v8`  : v8.2 (`8.0/sieve-echo-tests-2.py`): |FFT|^2 over the first floor(L/2) bins (DC kept),
                normalized, terms with p <= 1e-10 dropped; 0 for L < 2.
    - `H_v10` : v10 (`10.0/sieve-echo-tests-3.py`): amplitude |FFT| over the first floor(L/2) bins of
                d/(b-1), normalized, p <= 1e-10 dropped; NaN for L <= 3 (v10 skipped those bases).
* Digit statistics on x: mean, population std, skewness and excess kurtosis (biased moments, as
  `scipy.stats.skew/kurtosis` defaults); NaN when x is constant.
* `flatness`: spectral flatness (geometric / arithmetic mean) of the one-sided AC power bins
  1..floor(L/2), each floored at 1e-12 * mean so exact zeros do not send it to 0 numerically.
* `peak1..3`: frequencies k/L (in (0, 0.5]) of the three largest one-sided AC power bins.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np

NDR_VERSION = "ndr-v11.1"

# Order of the per-(n, b) feature vector returned by `pair_features`.
PAIR_FEATURES = (
    "L",
    "logL",
    "H_paper",
    "H_nodc",
    "Hn_paper",
    "Hn_nodc",
    "mean",
    "std",
    "skew",
    "kurt",
    "flatness",
    "peak1",
    "peak2",
    "peak3",
    "H_v8",
    "H_v10",
)

_MAX_N = 3_000_000_000  # r * (b^m mod n) < n^2 must fit in int64


# ---------------------------------------------------------------------------------------------
# Number theory
# ---------------------------------------------------------------------------------------------


def spf_sieve(n_max: int) -> np.ndarray:
    """Smallest prime factor for 0..n_max (spf[0] = 0, spf[1] = 1)."""
    spf = np.zeros(n_max + 1, dtype=np.int64)
    if n_max >= 1:
        spf[1] = 1
    for p in range(2, int(math.isqrt(n_max)) + 1):
        if spf[p] == 0:
            block = spf[p * p :: p]
            block[block == 0] = p
            spf[p * p :: p] = block
    rest = np.nonzero(spf == 0)[0]
    spf[rest] = rest  # primes (and 0 stays 0)
    spf[0] = 0
    return spf


def factorize(n: int, spf: np.ndarray) -> list[tuple[int, int]]:
    """Prime factorization of 1 <= n <= len(spf)-1 as [(p, e), ...] with p ascending."""
    out: list[tuple[int, int]] = []
    while n > 1:
        p = int(spf[n])
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        out.append((p, e))
    return out


def _lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


def carmichael_from_factors(factors: list[tuple[int, int]]) -> int:
    parts = []
    for p, e in factors:
        if p == 2 and e >= 3:
            parts.append(1 << (e - 2))
        else:
            parts.append((p - 1) * p ** (e - 1))
    return reduce(_lcm, parts, 1)


def arithmetic_labels(n_values: np.ndarray, spf: np.ndarray) -> dict[str, np.ndarray]:
    """Factorization-derived labels. Targets / stratification only — never predictor inputs."""
    n_values = np.asarray(n_values, dtype=np.int64)
    m = len(n_values)
    omega = np.zeros(m, dtype=np.int64)
    big_omega = np.zeros(m, dtype=np.int64)
    tau = np.zeros(m, dtype=np.int64)
    phi = np.zeros(m, dtype=np.int64)
    lam = np.zeros(m, dtype=np.int64)
    mu = np.zeros(m, dtype=np.int64)
    for i, n in enumerate(n_values.tolist()):
        f = factorize(n, spf)
        omega[i] = len(f)
        big_omega[i] = sum(e for _, e in f)
        tau[i] = math.prod(e + 1 for _, e in f)
        phi[i] = math.prod((p - 1) * p ** (e - 1) for p, e in f)
        lam[i] = carmichael_from_factors(f)
        mu[i] = 0 if any(e > 1 for _, e in f) else (-1) ** len(f)
    return {
        "omega": omega,
        "Omega": big_omega,
        "tau": tau,
        "phi": phi,
        "lambda": lam,
        "mobius": mu,
        "is_prime": (big_omega == 1),
        "is_prime_power": (omega == 1),
        "squarefree": (mu != 0),
    }


def multiplicative_order(b: int, n: int, lam: int, lam_primes: list[int]) -> int:
    """ord_n(b) for gcd(b, n) = 1, from lambda(n) and its distinct prime factors (exact)."""
    if n == 1:
        return 1
    if math.gcd(b, n) != 1:
        raise ValueError(f"ord_{n}({b}) undefined: gcd = {math.gcd(b, n)}")
    b %= n
    t = lam
    for p in lam_primes:
        while t % p == 0 and pow(b, t // p, n) == 1:
            t //= p
    return t


def order_context(n: int, spf: np.ndarray) -> tuple[int, list[int]]:
    """(lambda(n), distinct primes of lambda(n)) — lambda(n) < n, so the same sieve covers it."""
    lam = carmichael_from_factors(factorize(n, spf))
    return lam, [p for p, _ in factorize(lam, spf)]


# ---------------------------------------------------------------------------------------------
# Repetend and NDR
# ---------------------------------------------------------------------------------------------


def repetend_remainders(n: int, b: int, L: int) -> np.ndarray:
    """r_k = b^k mod n for k = 0..L-1, by doubling (log L vector steps)."""
    if not 2 <= n < _MAX_N:
        raise ValueError(f"n={n} outside supported range [2, {_MAX_N})")
    r = np.empty(L, dtype=np.int64)
    r[0] = 1
    m = 1
    bm = b % n  # b^m mod n
    while m < L:
        k = min(m, L - m)
        r[m : m + k] = (r[:k] * bm) % n
        m += k
        bm = (bm * bm) % n
    return r


def repetend_digits(n: int, b: int, L: int) -> np.ndarray:
    """The L repetend digits of 1/n in base b (requires gcd(n, b) = 1 and L = ord_n(b))."""
    r = repetend_remainders(n, b, L)
    return (r * b) // n


def ndr(digits: np.ndarray, b: int) -> np.ndarray:
    """Paper Definition 1.1: d / b."""
    return np.asarray(digits, dtype=np.float64) / b


# ---------------------------------------------------------------------------------------------
# Spectra and entropies
# ---------------------------------------------------------------------------------------------


def _xlogx_sum(p: np.ndarray, w: np.ndarray | None = None) -> float:
    nz = p > 0
    terms = p[nz] * np.log(p[nz])
    if w is not None:
        terms = terms * w[nz]
    return float(-terms.sum())


def _full_weights(L: int, n_bins: int) -> np.ndarray:
    """Multiplicity of each rFFT bin in the full length-L spectrum."""
    w = np.full(n_bins, 2.0)
    w[0] = 1.0
    if L % 2 == 0:
        w[-1] = 1.0
    return w


def power_spectrum_onesided(x: np.ndarray) -> np.ndarray:
    """|rFFT(x)|^2, bins 0..floor(L/2)."""
    return np.abs(np.fft.rfft(x)) ** 2


def full_spectrum_entropy(P: np.ndarray, L: int, drop_dc: bool) -> float:
    """Entropy of the full length-L power spectrum given its one-sided half P."""
    w = _full_weights(L, len(P))
    if drop_dc:
        P = P[1:]
        w = w[1:]
    total = float((P * w).sum())
    if P.size == 0 or total <= 0.0:
        return float("nan") if drop_dc else 0.0
    return _xlogx_sum(P / total, w)


def spectral_entropy(x: np.ndarray, variant: str = "paper") -> float:
    """Entropy of one NDR sequence; `variant` in {'paper', 'nodc', 'v8', 'v10'} (see module doc)."""
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    if variant == "paper":
        if L == 1:
            return 0.0
        return full_spectrum_entropy(power_spectrum_onesided(x), L, drop_dc=False)
    if variant == "nodc":
        if L == 1:
            return float("nan")
        return full_spectrum_entropy(power_spectrum_onesided(x), L, drop_dc=True)
    if variant == "v8":
        return _entropy_v8(power_spectrum_onesided(x), L)
    if variant == "v10":
        return _entropy_v10(power_spectrum_onesided(x), L, scale=1.0)
    raise ValueError(f"unknown entropy variant {variant!r}")


def _entropy_v8(P: np.ndarray, L: int) -> float:
    if L < 2:
        return 0.0
    half = P[: L // 2]
    total = float(half.sum())
    if half.size == 0 or total == 0.0:
        return 0.0
    p = half / total
    p = p[p > 1e-10]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _entropy_v10(P: np.ndarray, L: int, scale: float) -> float:
    if L <= 3:
        return float("nan")
    amp = np.sqrt(P[: L // 2]) * scale
    total = float(amp.sum())
    if total <= 1e-10:
        return float("nan")
    p = amp / total
    p = p[p > 1e-10]
    return float(-(p * np.log(p)).sum())


def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(x.mean())
    c = x - mean
    m2 = float((c * c).mean())
    if m2 <= 1e-30:
        return mean, 0.0, float("nan"), float("nan")
    m3 = float((c**3).mean())
    m4 = float((c**4).mean())
    return mean, math.sqrt(m2), m3 / m2**1.5, m4 / (m2 * m2) - 3.0


def pair_features(n: int, b: int, L: int) -> np.ndarray:
    """All PAIR_FEATURES for one coprime (n, b) with known L = ord_n(b)."""
    digits = repetend_digits(n, b, L)
    x = ndr(digits, b)
    P = power_spectrum_onesided(x)
    H_paper = 0.0 if L == 1 else full_spectrum_entropy(P, L, drop_dc=False)
    H_nodc = float("nan") if L == 1 else full_spectrum_entropy(P, L, drop_dc=True)
    Hn_paper = H_paper / math.log(L) if L > 1 else float("nan")
    Hn_nodc = H_nodc / math.log(L - 1) if L > 2 else float("nan")
    mean, std, skew, kurt = _moments(x)

    ac = P[1:]
    if ac.size and float(ac.sum()) > 0.0:
        floor = 1e-12 * float(ac.mean())
        acf = np.maximum(ac, floor)
        flatness = float(np.exp(np.log(acf).mean()) / acf.mean())
        order = np.argsort(-ac, kind="stable")[:3]
        peaks = [(int(k) + 1) / L for k in order] + [float("nan")] * (3 - len(order))
    else:
        flatness = float("nan")
        peaks = [float("nan")] * 3

    H_v8 = _entropy_v8(P, L)
    H_v10 = _entropy_v10(P, L, scale=b / (b - 1))
    return np.array(
        [L, math.log(L), H_paper, H_nodc, Hn_paper, Hn_nodc, mean, std, skew, kurt, flatness, *peaks, H_v8, H_v10],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------------------------
# Brute-force references (tests only)
# ---------------------------------------------------------------------------------------------


def long_division_repetend(n: int, b: int) -> list[int]:
    """Reference long division with a seen-remainder dict (the v8/v10 method). O(n) — tests only."""
    if math.gcd(n, b) != 1:
        return []
    r, digits, seen = 1, [], {}
    while r not in seen:
        seen[r] = len(digits)
        r *= b
        digits.append(r // n)
        r %= n
    return digits[seen[r] :]
