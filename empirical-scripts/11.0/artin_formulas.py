"""Evolvo searches for the full-period benchmark (ROADMAP step 20).

Four cases, each with fixed train/validation/test ranges and the PyTorch policy guide of step 19:

* ``artin_criterion`` — the full-period bit of base 10 from the local bits [q ∤ I_10(p)] for q = 2, 3, 5, 7,
  the bit "no prime q ≥ 11 divides I_10(p)" and a distractor [p ≡ 1 mod 4]. Exact rule: the product of the
  five local bits (Artin's criterion). Audited on all 64 Boolean rows.
* ``lehmer_nesting`` — the bit [10 | I_5(p)] from [2 | I_5(p)], [5 | I_5(p)] and a distractor. Exact rule:
  the quintic bit alone, because 5 | I_5(p) forces p ≡ 1 mod 5 and then (5/p) = 1. This one implication is
  the source of the factor 20/19 in the density of base 5.
* ``artin_congruence_only`` — the full-period bit of base 10 from congruence data only: the quadratic bit
  [(10/p) = −1] (a function of p mod 40 by reciprocity), [p ≡ 1 mod 3], [p ≡ 1 mod 5], [p ≡ 1 mod 7] and
  p mod 840. Chebotarev puts primes with q | I_10(p) in every reduced class for every q ∤ 840, so no
  congruence rule is exact; the train-only lookup modulo 840 and the held-out classes containing both
  labels give the finite limit.
* ``kummer_degree`` — regression over the primes q ≤ ``local_q_max``: y_q = q²·r_q, where r_q is the measured
  share of (p, b) pairs with q | I_b(p) over the non-square bases. Kummer theory gives
  r_q → 1/[Q(ζ_q, b^{1/q}) : Q] = 1/(q(q − 1)), so y_q → q/(q − 1) and the degree is q²/y_q. The primes q are
  split by alternation (index mod 3 = train, validation, test), so every split contains small q, where the
  functional form is visible; y is bounded, so large-q noise does not dominate the squared error.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

import artin
import failures
import prime_order_formulas

BIT_OPERATIONS = prime_order_formulas.BIT_OPERATIONS
RESIDUE_OPERATIONS = prime_order_formulas.RESIDUE_OPERATIONS


def degree_targets(local: dict) -> dict:
    """y_q = q²·r_q and its binomial-scale standard error (pairs sharing p are treated as independent)."""
    q = np.array(local["q"], dtype=np.int64)
    r = np.array(local["rates"])
    se_r = np.sqrt(r * (1 - r) / local["pairs"])
    return {"q": q, "y": q * q * r, "se": q * q * se_r, "split": np.arange(q.size) % 3}


def cases(P: np.ndarray, I: np.ndarray, bases: list, local: dict, domain_limit: int) -> dict:
    sel = P <= domain_limit
    p = P[sel]
    I10 = I[sel, bases.index(10)]
    I5 = I[sel, bases.index(5)]
    full10 = I10 == 1
    bits6 = np.array(list(product((0, 1), repeat=6)))
    bits3 = np.array(list(product((0, 1), repeat=3)))
    nested = bits3[bits3[:, 1] <= bits3[:, 0]]
    deg = degree_targets(local)
    return {
        "artin_criterion": {
            "p": p,
            "X": np.column_stack([I10 % m != 0 for m in (2, 3, 5, 7)] + [artin.strip(I10, (2, 3, 5, 7)) == 1, p % 4 == 1]),
            "y": full10,
            "names": ["2 ∤ I_10", "3 ∤ I_10", "5 ∤ I_10", "7 ∤ I_10", "no prime q ≥ 11 divides I_10", "p ≡ 1 mod 4"],
            "operations": BIT_OPERATIONS, "audit": (bits6, bits6[:, :5].all(axis=1)),
        },
        "lehmer_nesting": {
            "p": p, "X": np.column_stack((I5 % 2 == 0, I5 % 5 == 0, p % 3 == 1)), "y": I5 % 10 == 0,
            "names": ["2 divides I_5", "5 divides I_5", "p ≡ 1 mod 3"], "operations": BIT_OPERATIONS,
            "audit": (nested, nested[:, 1].astype(bool)),
        },
        "artin_congruence_only": {
            "p": p,
            "X": np.column_stack((artin.legendre_by_reciprocity(10, p) == -1, p % 3 == 1, p % 5 == 1, p % 7 == 1, p % 840)),
            "y": full10,
            "names": ["(10/p) = −1", "p ≡ 1 mod 3", "p ≡ 1 mod 5", "p ≡ 1 mod 7", "p mod 840"],
            "operations": RESIDUE_OPERATIONS,
        },
        "kummer_degree": {
            "p": deg["q"], "X": np.column_stack((deg["q"], np.ones_like(deg["q"]))), "y": deg["y"], "names": ["q", "1"],
            "operations": ["ADD", "SUB", "MUL", "DIV"], "regression": True, "split_labels": deg["split"],
        },
    }


def kummer_check(program: dict, local: dict, all_q: np.ndarray, A: float) -> dict:
    """Standardized residuals of the selected formula per split, and the degree and Artin's constant rebuilt from it."""
    deg = degree_targets(local)
    X = np.column_stack((deg["q"].astype(float), np.ones(deg["q"].size)))
    f = prime_order_formulas.replay(program, X)
    z = (deg["y"] - f) / deg["se"]
    ze = (deg["y"] - deg["q"] / (deg["q"] - 1.0)) / deg["se"]
    out = {}
    for k, name in enumerate(("train", "validation", "test")):
        mask = deg["split"] == k
        out[name] = {"primes": int(mask.sum()), "mean_squared_standardized_residual": float(np.mean(z[mask] ** 2)),
                     "max_abs_standardized_residual": float(np.max(np.abs(z[mask]))),
                     "q_over_q_minus_1_mean_squared_standardized_residual": float(np.mean(ze[mask] ** 2))}
    Xa = np.column_stack((all_q.astype(float), np.ones(all_q.size)))
    fa = prime_order_formulas.replay(program, Xa)
    q2 = all_q.astype(float) ** 2
    ok = bool(np.all(np.isfinite(fa)) and np.all((fa > 0) & (fa < q2)))
    ref = all_q * (all_q - 1.0)
    out["primes_checked"] = int(all_q.size)
    out["max_relative_difference_of_degree_from_q(q-1)"] = float(np.max(np.abs((q2 / fa) / ref - 1))) if ok else None
    out["implied_artin_product"] = float(np.exp(np.sum(np.log1p(-fa / q2)))) if ok else None
    out["artin_product_same_primes"] = float(np.exp(np.sum(np.log1p(-1 / ref))))
    out["artin_constant"] = A
    return out


def run_stage(cfg: dict, seed: int, run_dir: Path, P, I, bases, local, A: float, log=print) -> dict:
    limit = int(cfg.get("domain_limit", 10_000_000))
    tasks = cases(P, I, bases, local, limit)
    out = prime_order_formulas.run_cases(
        tasks, cfg, seed, run_dir, log, int(P[P <= limit][-1]),
        lookups={"artin_congruence_only": ("mod_840_lookup", 840)}, knowledge_prefix="artin",
    )
    if "kummer_degree" in out["targets"]:
        row = out["targets"]["kummer_degree"]
        qs = failures.sieve_primes(2, int(cfg.get("kummer_check_max", 100_000)))
        row["kummer_check"] = kummer_check(row["program"], local, qs, A)
        row["split_q"] = "alternating: index mod 3 = train, validation, test"
    return out
