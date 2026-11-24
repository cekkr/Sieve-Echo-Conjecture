"""Dataset stage for Sieve Echo v11: integer sampling, per-(n, b) NDR features, cross-base
aggregates, factorization labels, the seeded split by n, and the on-disk cache.

Tables (all column arrays, saved together in one `.npz`):
* `pair.*`  — one row per coprime (n, b): `pair.n`, `pair.b`, and every `ndr.PAIR_FEATURES` column.
              Non-coprime pairs are not rows; they are counted in `n_bases.<set>` and in the summary.
* `row.*`   — one row per integer n (sorted ascending):
    - `row.n`, `row.source` (0 = full range, 1 = stratified extra sample);
    - `row.n_bases.<set>` number of coprime bases of that set used for n;
    - `row.<set>.<feature>.<agg>` for agg in mean/std/min/max/cv over the set's coprime bases
      (NaN when n has none, or when the set is sub-sampled and n is not in its sample);
    - `label.<name>` factorization labels (targets / stratification only, see FORBIDDEN_INPUTS);
    - `target_side.L_over_lambda.<set>.mean`, `target_side.L_over_phi.<set>.mean` (label-derived).

Allowed predictor inputs for omega(n) are only `row.n`, `row.<set>.*` (digit/NDR/length features)
and `row.n_bases.*`; `allowed_input_columns()` returns exactly those.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import ndr
from runtime import sha256_json

AGGREGATES = ("mean", "std", "min", "max", "cv")
# Per-base features that get cross-base aggregates (peaks are per-base only; H_v10 only matters
# for reproduction of v10's base set and is aggregated too).
AGGREGATED_FEATURES = tuple(f for f in ndr.PAIR_FEATURES if not f.startswith("peak"))

FORBIDDEN_PREFIXES = ("label.", "target_side.")

DATASET_FORMAT = "dataset-v11.1"


# ---------------------------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------------------------


def parse_bases(spec: Any) -> list[int]:
    """A base list, or a string 'a..b' (inclusive range)."""
    if isinstance(spec, str):
        lo, hi = spec.split("..")
        bases = list(range(int(lo), int(hi) + 1))
    else:
        bases = [int(b) for b in spec]
    if any(b < 2 for b in bases) or len(set(bases)) != len(bases):
        raise ValueError(f"invalid base list {spec!r}")
    return bases


def normalize_base_sets(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, spec in raw.items():
        if isinstance(spec, dict):
            out[name] = {"bases": parse_bases(spec["bases"]), "sample": spec.get("sample")}
        else:
            out[name] = {"bases": parse_bases(spec), "sample": None}
    return out


def dataset_key(data_cfg: dict[str, Any], seed: int) -> str:
    """Cache key: everything that changes the dataset content."""
    return sha256_json({"format": DATASET_FORMAT, "ndr": ndr.NDR_VERSION, "data": data_cfg, "seed": seed})[:16]


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------


def sample_integers(data_cfg: dict[str, Any], seed: int, spf: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """All n in [n_min, n_max], plus a stratified-by-omega sample from (n_max, strat.n_hi]."""
    n_min, n_max = int(data_cfg["n_min"]), int(data_cfg["n_max"])
    full = np.arange(n_min, n_max + 1, dtype=np.int64)
    scheme: dict[str, Any] = {"full_range": [n_min, n_max], "full_count": int(full.size)}
    strat = data_cfg.get("stratified") or {}
    extra = np.zeros(0, dtype=np.int64)
    if strat.get("enabled"):
        lo, hi = max(int(strat["n_lo"]), n_max + 1), int(strat["n_hi"])
        per = int(strat["per_omega"])
        pool = np.arange(lo, hi + 1, dtype=np.int64)
        omega = ndr.arithmetic_labels(pool, spf)["omega"]
        rng = np.random.default_rng([seed, 1])
        picks, counts = [], {}
        for w in sorted(set(omega.tolist())):
            cand = pool[omega == w]
            take = cand if cand.size <= per else rng.choice(cand, size=per, replace=False)
            picks.append(take)
            counts[int(w)] = {"available": int(cand.size), "taken": int(take.size)}
        extra = np.sort(np.concatenate(picks)) if picks else extra
        scheme["stratified"] = {"range": [lo, hi], "per_omega": per, "by_omega": counts}
    n_values = np.concatenate([full, extra])
    source = np.concatenate([np.zeros(full.size, np.int8), np.ones(extra.size, np.int8)])
    order = np.argsort(n_values, kind="stable")
    return n_values[order], source[order], scheme


def base_set_members(n_values: np.ndarray, base_sets: dict[str, dict], seed: int) -> dict[str, np.ndarray]:
    """Boolean mask over n_values per base set (sub-sampled sets use a seeded subset)."""
    masks = {}
    for i, (name, spec) in enumerate(sorted(base_sets.items())):
        mask = np.ones(n_values.size, dtype=bool)
        if spec.get("sample") and spec["sample"] < n_values.size:
            rng = np.random.default_rng([seed, 2, i])
            mask[:] = False
            mask[rng.choice(n_values.size, size=int(spec["sample"]), replace=False)] = True
        masks[name] = mask
    return masks


# ---------------------------------------------------------------------------------------------
# Per-(n, b) features (parallel)
# ---------------------------------------------------------------------------------------------


def _worker(task: list[tuple[int, int, list[int], list[int]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ns, bs, feats = [], [], []
    for n, lam, lam_primes, bases in task:
        for b in bases:
            if math.gcd(n, b) != 1:
                continue
            L = ndr.multiplicative_order(b, n, lam, lam_primes)
            ns.append(n)
            bs.append(b)
            feats.append(ndr.pair_features(n, b, L))
    width = len(ndr.PAIR_FEATURES)
    return (
        np.asarray(ns, dtype=np.int64),
        np.asarray(bs, dtype=np.int64),
        np.asarray(feats, dtype=np.float64).reshape(-1, width),
    )


def _chunks(tasks: list, cost: list[float], target: float) -> list[list]:
    out, cur, acc = [], [], 0.0
    for t, c in zip(tasks, cost):
        cur.append(t)
        acc += c
        if acc >= target:
            out.append(cur)
            cur, acc = [], 0.0
    if cur:
        out.append(cur)
    return out


def compute_pairs(
    n_values: np.ndarray,
    bases_per_n: list[list[int]],
    spf: np.ndarray,
    workers: int,
    log=print,
) -> dict[str, np.ndarray]:
    tasks, cost = [], []
    for n, bases in zip(n_values.tolist(), bases_per_n):
        lam, lam_primes = ndr.order_context(n, spf)
        tasks.append((n, lam, lam_primes, bases))
        cost.append(len(bases) * (lam + 64))  # L <= lambda(n): a cheap upper bound on work
    total = float(sum(cost))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = _chunks(tasks, cost, max(total / (workers * 8), 1.0))
    t0 = time.time()
    parts = []
    if workers == 1 or len(chunks) == 1:
        parts = [_worker(c) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, part in enumerate(pool.map(_worker, chunks)):
                parts.append(part)
                if (i + 1) % max(1, len(chunks) // 10) == 0:
                    log(f"  pairs: chunk {i + 1}/{len(chunks)} ({time.time() - t0:.1f}s)")
    ns = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
    bs = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
    feats = np.concatenate([p[2] for p in parts]) if parts else np.zeros((0, len(ndr.PAIR_FEATURES)))
    out = {"pair.n": ns, "pair.b": bs}
    for j, name in enumerate(ndr.PAIR_FEATURES):
        out[f"pair.{name}"] = feats[:, j]
    return out


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------


def _group_stats(values: np.ndarray, group: np.ndarray, n_groups: int) -> dict[str, np.ndarray]:
    """mean/std/min/max/cv per group ignoring NaN (population std, cv = std/|mean|)."""
    ok = ~np.isnan(values)
    g, v = group[ok], values[ok]
    cnt = np.bincount(g, minlength=n_groups).astype(np.float64)
    s1 = np.bincount(g, weights=v, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        dev = v - mean[g]
        var = np.bincount(g, weights=dev * dev, minlength=n_groups) / cnt
        std = np.sqrt(var)
        cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)
    mn = np.full(n_groups, np.inf)
    mx = np.full(n_groups, -np.inf)
    np.minimum.at(mn, g, v)
    np.maximum.at(mx, g, v)
    empty = cnt == 0
    for a in (mean, std, cv, mn, mx):
        a[empty] = np.nan
    return {"mean": mean, "std": std, "min": mn, "max": mx, "cv": cv}


def aggregate(
    n_values: np.ndarray,
    pairs: dict[str, np.ndarray],
    base_sets: dict[str, dict],
    members: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    idx = np.searchsorted(n_values, pairs["pair.n"])
    for name in sorted(base_sets):
        bases = np.asarray(base_sets[name]["bases"], dtype=np.int64)
        in_set = np.isin(pairs["pair.b"], bases) & members[name][idx]
        g = idx[in_set]
        rows[f"row.n_bases.{name}"] = np.bincount(g, minlength=n_values.size).astype(np.int64)
        for feat in AGGREGATED_FEATURES:
            stats = _group_stats(pairs[f"pair.{feat}"][in_set], g, n_values.size)
            for agg in AGGREGATES:
                rows[f"row.{name}.{feat}.{agg}"] = stats[agg]
        L = pairs["pair.L"][in_set]
        for denom in ("lambda", "phi"):
            ratio = L / labels[denom][g]
            key = "L_over_lambda" if denom == "lambda" else "L_over_phi"
            rows[f"target_side.{key}.{name}.mean"] = _group_stats(ratio, g, n_values.size)["mean"]
    return rows


# ---------------------------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------------------------


def split_by_n(n_values: np.ndarray, strata: np.ndarray, fractions: dict[str, float], seed: int) -> np.ndarray:
    """0 = train, 1 = val, 2 = test; stratified by `strata`, seeded, independent of feature values."""
    f_tr, f_va = float(fractions["train"]), float(fractions["val"])
    if abs(f_tr + f_va + float(fractions["test"]) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng([seed, 3])
    split = np.empty(n_values.size, dtype=np.int8)
    for s in np.unique(strata):
        idx = np.nonzero(strata == s)[0]
        idx = idx[rng.permutation(idx.size)]
        k_tr = int(round(f_tr * idx.size))
        k_va = int(round(f_va * idx.size))
        split[idx[:k_tr]] = 0
        split[idx[k_tr : k_tr + k_va]] = 1
        split[idx[k_tr + k_va :]] = 2
    return split


SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------------------------
# Build / cache
# ---------------------------------------------------------------------------------------------


def allowed_input_columns(table: dict[str, np.ndarray]) -> list[str]:
    """Columns a predictor of omega(n) may read (AGENTS.md: no factorization-derived inputs)."""
    cols = []
    for k in table:
        if not k.startswith("row.") or k.startswith(FORBIDDEN_PREFIXES):
            continue
        if k in ("row.source",):
            continue
        cols.append(k)
    return cols


def build_dataset(data_cfg: dict[str, Any], split_cfg: dict[str, Any], seed: int, log=print) -> tuple[dict, dict]:
    t0 = time.time()
    base_sets = normalize_base_sets(data_cfg["base_sets"])
    strat = data_cfg.get("stratified") or {}
    n_cap = max(int(data_cfg["n_max"]), int(strat.get("n_hi", 0)) if strat.get("enabled") else 0)
    spf = ndr.spf_sieve(n_cap)
    log(f"  sieve to {n_cap} ({time.time() - t0:.1f}s)")

    n_values, source, scheme = sample_integers(data_cfg, seed, spf)
    labels = ndr.arithmetic_labels(n_values, spf)
    members = base_set_members(n_values, base_sets, seed)
    log(f"  {n_values.size} integers, labels done ({time.time() - t0:.1f}s)")

    bases_per_n = []
    for i in range(n_values.size):
        bs = set()
        for name, spec in base_sets.items():
            if members[name][i]:
                bs.update(spec["bases"])
        bases_per_n.append(sorted(bs))
    pairs = compute_pairs(n_values, bases_per_n, spf, int(data_cfg.get("workers", 0)), log=log)
    log(f"  {pairs['pair.n'].size} coprime pairs ({time.time() - t0:.1f}s)")

    table: dict[str, np.ndarray] = {"row.n": n_values, "row.source": source}
    table.update(aggregate(n_values, pairs, base_sets, members, labels))
    for k, v in labels.items():
        table[f"label.{k}"] = v
    table.update(pairs)

    strata_name = split_cfg.get("stratify", "omega")
    strata = labels[strata_name] if strata_name else np.zeros(n_values.size, np.int64)
    table["split"] = split_by_n(n_values, strata, split_cfg, seed)

    summary = summarize(table, base_sets, members, scheme, time.time() - t0)
    return table, summary


def summarize(table: dict, base_sets: dict, members: dict, scheme: dict, seconds: float) -> dict:
    n_values = table["row.n"]
    omega = table["label.omega"]
    split = table["split"]
    by_split = {}
    for s, name in enumerate(SPLIT_NAMES):
        sel = split == s
        w, c = np.unique(omega[sel], return_counts=True)
        by_split[name] = {"count": int(sel.sum()), "by_omega": {int(a): int(b) for a, b in zip(w, c)}}
    sets = {}
    for name, spec in sorted(base_sets.items()):
        used = table[f"row.n_bases.{name}"]
        mem = members[name]
        bases = np.asarray(spec["bases"])
        excluded = {int(b): int((np.gcd(n_values[mem], int(b)) != 1).sum()) for b in bases}
        sets[name] = {
            "bases": spec["bases"],
            "sample": spec.get("sample"),
            "members": int(mem.sum()),
            "members_with_no_coprime_base": int(((used == 0) & mem).sum()),
            "excluded_pairs_by_base": excluded,
            "excluded_pairs_total": int(sum(excluded.values())),
        }
    # Descriptive only (all integers, no split): mean paper entropy by omega, first base set.
    first = sorted(base_sets)[0]

    def _nanmean(a):
        a = a[~np.isnan(a)]
        return float(a.mean()) if a.size else float("nan")

    desc = {}
    for w in np.unique(omega):
        sel = omega == w
        desc[int(w)] = {
            "count": int(sel.sum()),
            f"{first}.H_paper.mean": _nanmean(table[f"row.{first}.H_paper.mean"][sel]),
            f"{first}.H_nodc.mean": _nanmean(table[f"row.{first}.H_nodc.mean"][sel]),
            f"{first}.logL.mean": _nanmean(table[f"row.{first}.logL.mean"][sel]),
        }
    return {
        "dataset_format": DATASET_FORMAT,
        "ndr_version": ndr.NDR_VERSION,
        "integers": int(n_values.size),
        "n_range": [int(n_values.min()), int(n_values.max())],
        "pairs": int(table["pair.n"].size),
        "sampling": scheme,
        "base_sets": sets,
        "splits": by_split,
        "columns": {
            "row": len([k for k in table if k.startswith("row.")]),
            "allowed_inputs": len(allowed_input_columns(table)),
            "label": len([k for k in table if k.startswith("label.")]),
            "target_side": len([k for k in table if k.startswith("target_side.")]),
        },
        "descriptive_by_omega": desc,
        "build_seconds": round(seconds, 2),
    }


def save_cache(path: Path, table: dict[str, np.ndarray], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, __key__=np.array(key), **table)
    tmp.replace(path)


def load_cache(path: Path, key: str) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["__key__"]) != key:
            return None
        return {k: z[k] for k in z.files if k != "__key__"}


def split_record(table: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: table["row.n"][table["split"] == s].tolist() for s, name in enumerate(SPLIT_NAMES)}
