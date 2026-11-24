"""Exact quadratic and cubic repetend-index benchmarks (ROADMAP steps 17–18)."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expi

import ndr

QUADRATIC_BASES = (2, 3, 5)
QUADRATIC_MODULUS = 120
CUBIC_DEGREE = 18
# Discriminant of the integral tensor-product basis of Q(zeta_3, cbrt(2), cbrt(3)).
CUBIC_DISCRIMINANT_BOUND = 2**12 * 3**57


def quadratic_flags(n: np.ndarray) -> np.ndarray:
    """Quadratic reciprocity on integers coprime to 30, in base order (2, 3, 5)."""
    return np.column_stack((np.isin(n % 8, (1, 7)), np.isin(n % 12, (1, 11)), np.isin(n % 5, (1, 4))))


def pattern_codes(flags: np.ndarray) -> np.ndarray:
    return flags @ np.array([4, 2, 1])


def order_indices(limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Return odd primes and the indices of the three base subgroups."""
    spf = ndr.spf_sieve(limit)
    primes = np.flatnonzero(spf == np.arange(limit + 1, dtype=spf.dtype))
    primes = primes[primes >= 3]
    indices = np.zeros((primes.size, len(QUADRATIC_BASES)), dtype=np.int64)
    for i, p in enumerate(map(int, primes)):
        divisors = [q for q, _ in ndr.factorize(p - 1, spf)]
        for j, b in enumerate(QUADRATIC_BASES):
            if b != p:
                indices[i, j] = (p - 1) // ndr.multiplicative_order(b, p, p - 1, divisors)
    return primes, indices


def power_flags(primes: np.ndarray, m: int, bases: tuple[int, ...]) -> np.ndarray:
    """Independent modular-power test. Callers exclude primes dividing m or any base."""
    return np.array(
        [tuple((p - 1) % m == 0 and pow(b, (p - 1) // m, p) == 1 for b in bases)
         for p in map(int, primes)],
        dtype=bool,
    ).reshape(primes.size, len(bases))


def checked_equal(a: np.ndarray, b: np.ndarray, primes: np.ndarray, description: str) -> None:
    if not np.array_equal(a, b):
        row = int(np.argwhere(a != b)[0, 0])
        raise ArithmeticError(f"{description} disagrees at p={primes[row]}")


def quadratic_summary(primes: np.ndarray, flags: np.ndarray, cutoffs: list[int]) -> dict:
    residues = np.array([r for r in range(QUADRATIC_MODULUS) if math.gcd(r, QUADRATIC_MODULUS) == 1])
    rcodes = pattern_codes(quadratic_flags(residues))
    classes = [residues[rcodes == code].tolist() for code in range(8)]
    densities = np.array([len(group) / len(residues) for group in classes])
    codes = pattern_codes(flags)
    weights = np.log(primes)
    characters = 2 * flags.astype(int) - 1
    subsets = [[j for j in range(3) if mask & (4 >> j)] for mask in range(1, 8)]
    products = np.column_stack([np.prod(characters[:, cols], axis=1) for cols in subsets])
    all_positive = (codes == 7).astype(float)
    if not np.array_equal(all_positive, np.prod(1 + characters, axis=1) / 8):
        raise ArithmeticError("quadratic character expansion disagrees with the joint event")

    rows = []
    lo = 0
    for x in cutoffs:
        stop = int(np.searchsorted(primes, x, side="right"))
        windows = {}
        for name, start in (("cumulative", 0), ("interval", lo)):
            counts = np.bincount(codes[start:stop], minlength=8)
            weighted = np.bincount(codes[start:stop], weights=weights[start:stop], minlength=8)
            total = stop - start
            if int(counts.sum()) != total:
                raise ArithmeticError("quadratic patterns do not account for every eligible prime")
            windows[name] = {
                "eligible": total,
                "counts": counts.tolist(),
                "count_discrepancies": (counts - total * densities).tolist(),
                "log_weighted_discrepancies": (weighted - weights[start:stop].sum() * densities).tolist(),
            }

        prime_sums = weights[:stop] @ products[:stop]
        power_corrections = np.zeros(7)
        for p, w, chars in zip(primes[:stop], weights[:stop], products[:stop]):
            p = int(p)
            power = p * p
            # The characters are real. Their values at even powers are all one.
            even = True
            while power <= x:
                power_corrections += w if even else w * chars
                power *= p
                even = not even
            if p * p > x:
                break
        character_residual = float(prime_sums.sum() / 8)
        direct_residual = windows["cumulative"]["log_weighted_discrepancies"][7]
        if not math.isclose(character_residual, direct_residual, abs_tol=1e-7, rel_tol=1e-9):
            raise ArithmeticError("weighted quadratic count disagrees with the character sums")
        rows.append({
            "x": x, "previous_x": rows[-1]["x"] if rows else 0, **windows,
            "joint_character_residual": character_residual,
            "character_prime_sums": prime_sums.tolist(),
            "character_prime_power_corrections": power_corrections.tolist(),
            "character_von_mangoldt_sums_mod_120": (prime_sums + power_corrections).tolist(),
        })
        lo = stop
    return {
        "bases": list(QUADRATIC_BASES), "modulus": QUADRATIC_MODULUS, "excluded_primes": [2, 3, 5],
        "pattern_order": [f"{code:03b}" for code in range(8)], "classes": classes,
        "densities": densities.tolist(),
        "character_bases": [[QUADRATIC_BASES[j] for j in cols] for cols in subsets],
        "character_note": "Characters are taken modulo 120, so all powers of 2, 3 and 5 have value zero.",
        "rows": rows,
    }


def grh_count_bound(x: int) -> float:
    """Grenié–Molteni Corollary 1.2, split class over Q, using an upper bound on |disc K|."""
    log_x = math.log(x)
    return math.sqrt(x) * (
        (1 / (2 * math.pi) + 3 / log_x) * math.log(CUBIC_DISCRIMINANT_BOUND) / CUBIC_DEGREE
        + log_x / (8 * math.pi) + 1 / (4 * math.pi) + 6 / log_x
    )


def cubic_summary(primes: np.ndarray, flags: np.ndarray, cutoffs: list[int]) -> dict:
    rows = []
    for x in cutoffs:
        k = int(np.searchsorted(primes, x, side="right"))
        congruent = int(np.count_nonzero(primes[:k] % 3 == 1))
        count2, count3 = map(int, flags[:k].sum(axis=0))
        joint = int(np.count_nonzero(flags[:k].all(axis=1)))
        # Corollary 1.2 integrates from 2. The subtracted value uses the same convention.
        main_term = float((expi(math.log(x)) - expi(math.log(2))) / CUBIC_DEGREE)
        expected = count2 * count3 / congruent
        rows.append({
            "x": x, "eligible": k, "p_eq_1_mod_3": congruent, "counts": [count2, count3, joint],
            "rates": [v / k for v in (count2, count3, joint)],
            "conditional_rates": [v / congruent for v in (count2, count3, joint)],
            "conditional_independent_joint_count": expected, "conditional_joint_excess": joint - expected,
            "li_from_2_over_18": main_term, "joint_count_minus_li_over_18": joint - main_term,
            "grh_joint_count_error_bound": grh_count_bound(x),
        })
    return {
        "bases": [2, 3], "excluded_primes": [2, 3], "field": "Q(zeta_3, cbrt(2), cbrt(3))",
        "degrees": [6, 6, CUBIC_DEGREE], "discriminant_upper_bound": CUBIC_DISCRIMINANT_BOUND,
        "densities": [1 / 6, 1 / 6, 1 / 18], "conditional_densities": [1 / 3, 1 / 3, 1 / 9],
        "grh_reference": "Grenié–Molteni (2019), Corollary 1.2, https://arxiv.org/abs/1709.07609",
        "grh_assumption": "GRH for the Dedekind zeta function of the joint splitting field.",
        "unconditional_reference": "Thorner–Zaman (2019), Theorem 1.1, https://doi.org/10.2140/ant.2019.13.1039",
        "unconditional_range": "x >= (D_L * n_L**n_L)**c2 for absolute effective c2. "
                               "The asymptotic depends on D_L, n_L, and a possible exceptional zero. "
                               "No numerical envelope is evaluated at these cutoffs.",
        "rows": rows,
    }


def run_stage(cfg: dict, seed: int, run_dir, log=print) -> dict:
    cutoffs = cfg["cutoffs"]
    if (not cutoffs or any(type(x) is not int or x < 7 for x in cutoffs)
            or cutoffs != sorted(set(cutoffs))):
        raise ValueError("prime_orders.cutoffs must be strictly increasing integers >= 7")
    primes, indices = order_indices(cutoffs[-1])
    cp = primes[primes >= 5]
    cubic = indices[primes >= 5, :2] % 3 == 0
    checked_equal(cubic, power_flags(cp, 3, (2, 3)), cp, "cubic order vs modular-power test")

    qp = primes[primes >= 7]
    qi = indices[primes >= 7]
    quadratic = qi % 2 == 0
    checked_equal(quadratic, quadratic_flags(qp), qp, "quadratic order vs reciprocity")
    mp = primes[primes != 5]
    mi = indices[primes != 5]
    fifth = mi[:, 0] % 5 == 0
    mixed_quadratic = mi[:, 2] % 2 == 0
    checked_equal(fifth, power_flags(mp, 5, (2,))[:, 0], mp, "quintic order vs modular-power test")
    checked_equal(mixed_quadratic, power_flags(mp, 2, (5,))[:, 0], mp, "quadratic order vs modular-power test")
    mixed = fifth & mixed_quadratic
    checked_equal(mixed, fifth, mp, "quintic implies quadratic at base 5")
    mixed_counts = [
        {"x": x, "eligible": int((mp <= x).sum()), "quintic": int(fifth[mp <= x].sum()),
         "quadratic": int(mixed_quadratic[mp <= x].sum()), "joint": int(mixed[mp <= x].sum())}
        for x in cutoffs
    ]

    log(f"  verified quadratic/cubic flags on {qp.size:,}/{cp.size:,} primes")
    out = {
        "ndr_version": ndr.NDR_VERSION,
        "cutoffs": cutoffs,
        "quadratic": quadratic_summary(qp, quadratic, cutoffs),
        "cubic": cubic_summary(cp, cubic, cutoffs),
        "mixed": {"excluded_primes": [2, 5], "rows": mixed_counts},
    }
    if cfg.get("search"):
        import prime_order_formulas

        out["formulas"] = prime_order_formulas.run_stage(
            cfg["search"], seed, run_dir, qp, quadratic, cubic[cp >= 7], fifth[mp >= 7], log,
        )
    return out
