"""Predictor inputs for Sieve Echo v11 searches (formulas, GFSL-NN, baselines) and claim C9.

Every function here reads only allowed columns (`dataset.allowed_input_columns`) or the pair
table's lengths and bases: functions of n and of the repetend lengths L_b. Factorization labels
never enter (AGENTS.md "Features fed to a predictor of omega(n) MUST NOT encode factorization").
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

import dataset

# Per-base token features for the GFSL-NN set input (one token per base).
TOKEN_FEATURES = ("logL", "H_paper", "H_nodc", "Hn_paper", "Hn_nodc", "mean", "std", "skew", "kurt", "flatness", "peak1", "peak2", "peak3")

LENGTH_ARITHMETIC_NAMES = ("log lcm(L_b)", "log n - log lcm", "lcm divides n-1", "log gcd(lcm, n-1)", "share of L_b dividing n-1")


def union_bases(base_sets: Dict[str, dict], sets: Sequence[str]) -> List[int]:
    normalized = dataset.normalize_base_sets(base_sets)
    return sorted({int(b) for s in sets if s in normalized for b in normalized[s]["bases"]})


def per_base(table: Dict[str, np.ndarray], bases: Sequence[int], feature: str) -> np.ndarray:
    """rows x bases matrix of a pair feature (NaN where gcd(n, b) != 1)."""
    n = table["row.n"]
    M = np.full((n.size, len(bases)), np.nan)
    idx = np.searchsorted(n, table["pair.n"])
    col = {int(b): j for j, b in enumerate(bases)}
    pb = table["pair.b"]
    sel = np.isin(pb, list(bases))
    M[idx[sel], np.array([col[int(b)] for b in pb[sel]], dtype=np.int64)] = table[f"pair.{feature}"][sel]
    return M


def length_arithmetic(table: Dict[str, np.ndarray], bases: Sequence[int]) -> Tuple[np.ndarray, List[str]]:
    """Functions of (n, L_b) only: lcm of the lengths and its relation to n - 1.

    A tree model cannot compute an lcm or a divisibility test from raw log-lengths, so a baseline
    without these under-uses L and NDR features then look informative (claim C9).
    """
    n = table["row.n"].astype(np.int64)
    sel = np.isin(table["pair.b"], list(bases))
    idx = np.searchsorted(n, table["pair.n"][sel])
    Ls = table["pair.L"][sel].astype(np.int64)
    acc = [1] * n.size
    divides = np.zeros(n.size)
    for i, L in zip(idx.tolist(), Ls.tolist()):
        acc[i] = math.lcm(acc[i], L)
        divides[i] += (int(n[i]) - 1) % L == 0
    lcm = np.array(acc, dtype=np.float64)
    nm1 = n - 1
    g = np.array([math.gcd(int(a), int(b)) for a, b in zip(acc, nm1.tolist())], dtype=np.float64)
    cnt = np.bincount(idx, minlength=n.size).astype(float)
    feats = np.column_stack(
        [
            np.log(lcm),
            np.log(n.astype(float)) - np.log(lcm),
            (nm1 % lcm.astype(np.int64) == 0).astype(float),
            np.log(np.maximum(g, 1)),
            np.where(cnt > 0, divides / np.maximum(cnt, 1), np.nan),
        ]
    )
    return feats, list(LENGTH_ARITHMETIC_NAMES)


def length_arithmetic_from_orders(n: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """`length_arithmetic` for a rows x bases order matrix (0 = base not coprime to n): same five columns."""
    n = np.asarray(n, dtype=np.int64)
    L = np.asarray(L, dtype=np.int64)
    has = L > 0
    lcm = np.lcm.reduce(np.where(has, L, 1), axis=1)
    nm1 = n - 1
    div = has & ((nm1[:, None] % np.where(has, L, 1)) == 0)
    cnt = has.sum(axis=1)
    feats = np.column_stack(
        [
            np.log(lcm.astype(float)),
            np.log(n.astype(float)) - np.log(lcm.astype(float)),
            (nm1 % lcm == 0).astype(float),
            np.log(np.maximum(np.gcd(lcm, nm1), 1).astype(float)),
            np.where(cnt > 0, div.sum(axis=1) / np.maximum(cnt, 1), np.nan),
        ]
    )
    return feats, list(LENGTH_ARITHMETIC_NAMES)


def column(table: Dict[str, np.ndarray], name: str, cache: Dict[str, np.ndarray], bases: Sequence[int]) -> np.ndarray:
    """A named input: a `row.*` column, `log n`, or one of the length-arithmetic features."""
    if name == "log n":
        return np.log(table["row.n"].astype(float))
    if name in LENGTH_ARITHMETIC_NAMES:
        if "__arith__" not in cache:
            cache["__arith__"], _ = length_arithmetic(table, bases)
        return cache["__arith__"][:, LENGTH_ARITHMETIC_NAMES.index(name)]
    if name.startswith(dataset.FORBIDDEN_PREFIXES) or name.startswith("pair.") or name == "split":
        raise ValueError(f"{name!r} is not an allowed predictor input")
    if name not in table:
        raise KeyError(f"unknown input column {name!r}")
    return table[name].astype(float)


class Standardizer:
    """z-score with train statistics; NaN -> train median before scaling (then 0 after)."""

    def __init__(self, X_train: np.ndarray):
        med = np.nanmedian(X_train, axis=0)
        self.fill = np.where(np.isfinite(med), med, 0.0)
        Xf = self._fill(X_train)
        self.mean = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        self.std = np.where(sd > 0, sd, 1.0)

    def _fill(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        bad = ~np.isfinite(X)
        X[bad] = np.broadcast_to(self.fill, X.shape)[bad]
        return X

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (self._fill(X) - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"fill": self.fill.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist()}


def build_matrix(table: Dict[str, np.ndarray], names: Sequence[str], bases: Sequence[int]) -> np.ndarray:
    cache: Dict[str, np.ndarray] = {}
    return np.column_stack([column(table, nm, cache, bases) for nm in names])


def token_tensor(table: Dict[str, np.ndarray], bases: Sequence[int]) -> np.ndarray:
    """rows x bases x (2 + len(TOKEN_FEATURES)): [coprime flag, log b, per-base features] (0 when not coprime)."""
    mats = [per_base(table, bases, f) for f in TOKEN_FEATURES]
    coprime = np.isfinite(mats[0]).astype(float)
    logb = np.broadcast_to(np.log(np.asarray(bases, dtype=float)), coprime.shape)
    T = np.stack([coprime, logb, *[np.nan_to_num(m, nan=0.0) for m in mats]], axis=-1)
    return T
