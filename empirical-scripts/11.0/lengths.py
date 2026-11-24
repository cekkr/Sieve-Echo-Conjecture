"""Step 7: stronger length baselines, the Midy half-period symmetry, and lambda(n) (ROADMAP step 7).

The scale stage left one positive result for the conjecture: at n in [10^5, 10^6] entropy features
add about one accuracy point to an omega classifier that knows n and the repetend lengths in the 19
bases of the v8/v10/primes31 sets. This stage asks what that residual signal is.

Number theory used here (proofs in `studies/9`):

* **Order matrix.** ``ord_n(b)`` for every base b in ``bases`` (default 2..99) and every integer of
  the dataset, computed exactly from lambda(n) (`ndr.multiplicative_order`); no digits are needed,
  so lengths in 98 bases cost seconds where repetend entropies would cost hours.
* **Midy flag.** ``midy(n, b) = [L even and b^(L/2) ≡ −1 (mod n)]``, i.e. −1 ∈ ⟨b⟩ mod n. When it
  holds the remainders satisfy r_{k+L/2} = n − r_k, so the two halves of the repetend are digit
  complements (d_{k+L/2} = b − 1 − d_k, Midy's theorem) and every even DFT bin except DC vanishes:
  the DC-free spectral entropy loses about log 2. For odd n the flag holds iff the orders of b
  modulo the prime powers of n all have the same 2-adic valuation ≥ 1, so it depends on the
  factorization (share of flagged pairs falls with omega). The flag is a function of (n, b) and is
  visible in the digit sequence; it is a digit-symmetry feature, not a factorization label.
* **lambda(n) = lcm of ord_n(b) over all b coprime to n** (the exponent of (Z/nZ)*); the lcm over
  2..99 recovers it for most n. Theorem 1.3 of `studies/9` (lambda(n) ≤ phi(n)/2^(omega−1), odd n)
  gives the length-only bound omega ≤ ceil(log2(n / lcm)).

Analyses:

* **7a** — per n-range, the same omega-balanced samples and gradient-boosting settings as the scale
  stage (the L19 baseline reproduces `scale.json`), with four baselines: L19 (n + per-base L +
  length arithmetic over the 19 bases), L19 + Midy19, L98 (all bases 2..99), L98 + Midy98; each
  with and without the entropy columns, paired-bootstrap Δ. Group ablation in the last range: the
  gain of each NDR feature family over L19 and over L19 + Midy19.
* **7b** — pair-level mechanism (H ~ log L, log b, log n, ± Midy; train pairs fit, test pairs
  scored; Midy coefficient vs −log 2) and the C7 residue test with and without per-base Midy
  controls.
* **7c** — coverage of lcm(L_b) = lambda(n) (lambda is a label: diagnostic only), the Theorem 1.3
  bound, omega classifiers from (log n, log lambda) [labelled factorization-derived control] and
  from length-only estimates, and an Evolvo formula search for omega with the 2..99 length inputs.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import baselines
import claims
import dataset
import features
import ndr
import runtime
import scale

LENGTHS_VERSION = "lengths-v11.1"

DEFAULTS: Dict[str, Any] = {
    "bases": "2..99",
    "bins": scale.DEFAULTS["bins"],
    "per_omega_cap": scale.DEFAULTS["per_omega_cap"],
    "min_per_omega": scale.DEFAULTS["min_per_omega"],
    "bootstrap": 300,
    "model_seeds": 1,
    "max_omega_class": 5,
    "gbm": {"max_iter": 200, "learning_rate": 0.1},
    "sets": ["v8", "v10", "primes31"],
    "ablation": True,
    "pair_bins": 24,
    "formulas": {
        "guides": ["none", "policy"],
        "seeds": [0, 1, 2],
        "population": 120,
        "generations": 40,
        "workers": 0,
    },
    "workers": 0,
}

FORMULA_INPUTS_98 = [
    "log n",
    "row.len98.log lcm(L_b)",
    "row.len98.log n - log lcm",
    "row.len98.lcm divides n-1",
    "row.len98.log gcd(lcm, n-1)",
    "row.len98.share of L_b dividing n-1",
    "row.len98.midy share",
    "row.len98.mean log L",
]


# ---------------------------------------------------------------------------------------------
# Orders and Midy flags
# ---------------------------------------------------------------------------------------------

_SPF: Dict[str, np.ndarray] = {}


def _init(nmax: int) -> None:
    _SPF["spf"] = ndr.spf_sieve(nmax)


def midy_flag(n: int, b: int, L: int) -> bool:
    """−1 ∈ ⟨b⟩ mod n (then b^(L/2) ≡ −1 is the only way: b^k ≡ −1 forces L | 2k)."""
    return L % 2 == 0 and n > 2 and pow(b, L // 2, n) == n - 1


def _chunk(args) -> tuple:
    ns, bases = args
    spf = _SPF["spf"]
    L = np.zeros((len(ns), len(bases)), dtype=np.int64)
    M = np.zeros((len(ns), len(bases)), dtype=np.int8)
    for i, n in enumerate(ns):
        lam, lp = ndr.order_context(n, spf)
        for j, b in enumerate(bases):
            if math.gcd(b, n) != 1:
                continue
            t = ndr.multiplicative_order(b, n, lam, lp)
            L[i, j] = t
            M[i, j] = midy_flag(n, b, t)
    return L, M


def orders_and_midy(n_values: np.ndarray, bases: Sequence[int], workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """rows x bases matrices: ord_n(b) (0 when gcd(n, b) > 1) and the Midy flag (0/1)."""
    ns = [int(v) for v in np.asarray(n_values)]
    bases = [int(b) for b in bases]
    nmax = max(ns) if ns else 2
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    size = max(200, len(ns) // (workers * 8) + 1)
    tasks = [(ns[i:i + size], bases) for i in range(0, len(ns), size)]
    if workers <= 1 or len(tasks) <= 1:
        _init(nmax)
        parts = [_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(nmax,)) as pool:
            parts = list(pool.map(_chunk, tasks))
    if not parts:
        return np.zeros((0, len(bases)), np.int64), np.zeros((0, len(bases)), np.int8)
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def load_or_compute(run_dir: Path, dataset_key: str, n_values: np.ndarray, bases: Sequence[int], workers: int, log=print) -> tuple[np.ndarray, np.ndarray, str]:
    key = runtime.sha256_json({"dataset": dataset_key, "bases": list(map(int, bases)), "version": LENGTHS_VERSION})[:16]
    path = runtime.shared_cache(run_dir, f"lengths-{key}.npz")
    if path.is_file():
        with np.load(path) as z:
            if str(z["key"]) == key and np.array_equal(z["n"], n_values):
                log(f"  lengths: cache {path.name}")
                return z["L"], z["M"], key
    t0 = time.time()
    L, M = orders_and_midy(n_values, bases, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, L=L, M=M, n=n_values, key=np.array(key))
    log(f"  lengths: {L.shape[0]} integers x {L.shape[1]} bases in {time.time() - t0:.1f}s")
    return L, M, key


# ---------------------------------------------------------------------------------------------
# Feature blocks (functions of n, the lengths and the Midy flags only)
# ---------------------------------------------------------------------------------------------


def length_block(n: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, List[str]]:
    logL = np.where(L > 0, np.log(np.maximum(L, 1).astype(float)), np.nan)
    arith, an = features.length_arithmetic_from_orders(n, L)
    cnt = (L > 0).sum(axis=1).astype(float)
    X = np.column_stack([np.log(n.astype(float)), logL, cnt, arith])
    return X, ["log n", *[f"logL[{j}]" for j in range(L.shape[1])], "coprime bases", *an]


def midy_block(L: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, List[str]]:
    has = L > 0
    flags = np.where(has, M.astype(float), np.nan)
    share = np.where(has.any(axis=1), (M * has).sum(axis=1) / np.maximum(has.sum(axis=1), 1), np.nan)
    return np.column_stack([flags, share]), [*[f"midy[{j}]" for j in range(L.shape[1])], "midy share"]


def summary_columns(n: np.ndarray, L: np.ndarray, M: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    """Named row columns (for formula inputs): `<prefix>.<name>`."""
    arith, an = features.length_arithmetic_from_orders(n, L)
    out = {f"{prefix}.{nm}": arith[:, i] for i, nm in enumerate(an)}
    has = L > 0
    out[f"{prefix}.midy share"] = midy_block(L, M)[0][:, -1]
    out[f"{prefix}.mean log L"] = np.where(has.any(axis=1), (np.log(np.maximum(L, 1)) * has).sum(axis=1) / np.maximum(has.sum(axis=1), 1), np.nan)
    return out


# ---------------------------------------------------------------------------------------------
# 7a — C9 against stronger length baselines
# ---------------------------------------------------------------------------------------------


def _gbm(cfg, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=int(cfg["gbm"]["max_iter"]), learning_rate=float(cfg["gbm"]["learning_rate"]),
                                          early_stopping=True, validation_fraction=0.15, random_state=seed)


def _correct(cfg, seed, X, y, tr, te) -> np.ndarray:
    """Per-test-row correctness, averaged over `model_seeds` fits (seeds seed, seed+1, …).

    Gradient boosting's only randomness is its internal early-stopping split; with a few hundred
    columns that alone moves accuracy by about a point, which a bootstrap over test rows does not
    see. Seed `seed` alone reproduces the scale stage.
    """
    k = max(1, int(cfg.get("model_seeds", 1)))
    out = np.zeros(te.size)
    for r in range(k):
        m = _gbm(cfg, seed + r).fit(X[tr], y[tr])
        out += (m.predict(X[te]) == y[te]).astype(float)
    return out / k


def _delta(c1, c0, B, seed) -> Dict[str, Any]:
    return {"value": float(c1.mean() - c0.mean()), "ci95": scale._boot(lambda i: c1[i].mean() - c0[i].mean(), c0.size, B, seed)}


def c9_bins(table, cfg, fsets, entropy_cols, ndr_cols, blocks, seed, log=print) -> List[Dict[str, Any]]:
    n, omega, split = table["row.n"], table["label.omega"], table["split"]
    K = int(cfg["max_omega_class"])
    y = np.minimum(omega, K).astype(int)
    B = int(cfg["bootstrap"])
    rng = np.random.default_rng([seed, 50])  # the scale stage's stream: identical omega-balanced samples
    ent = np.column_stack([table[c] for c in entropy_cols])
    ndr_m = np.column_stack([table[c] for c in ndr_cols])
    out = []
    for i, (lo, hi) in enumerate(cfg["bins"]):
        rows = scale._balanced(np.nonzero((n >= lo) & (n <= hi))[0], omega, int(cfg["per_omega_cap"]), int(cfg["min_per_omega"]), rng)
        s = seed + 100 * i
        tr, te = rows[split[rows] == 0], rows[split[rows] == 2]
        rec: Dict[str, Any] = {"bin": [int(lo), int(hi)], "rows": int(rows.size), "train": int(tr.size), "test": int(te.size)}
        if tr.size < 50 or te.size < 30:
            rec["skipped"] = "too few rows"
            out.append(rec)
            continue
        acc: Dict[str, float] = {}
        gains: Dict[str, Any] = {}
        base_correct: Dict[str, np.ndarray] = {}
        for k, (name, Xb) in enumerate(blocks.items()):
            cb = base_correct[name] = _correct(cfg, s, Xb, y, tr, te)
            ce = _correct(cfg, s, np.column_stack([Xb, ent]), y, tr, te)
            acc[name] = float(cb.mean())
            acc[f"{name} + entropy"] = float(ce.mean())
            gains[name] = {"entropy": _delta(ce, cb, B, s + 10 + k)}
            if name in ("L19", "L19 + Midy19", "L98 + Midy98"):
                cf = _correct(cfg, s, np.column_stack([Xb, ndr_m]), y, tr, te)
                acc[f"{name} + all NDR"] = float(cf.mean())
                gains[name]["all NDR"] = _delta(cf, cb, B, s + 20 + k)
            log(f"    bin [{lo}, {hi}] {name}: {cb.mean():.4f} → +entropy {ce.mean():.4f}")
        pairs_ = (("L19 + Midy19", "L19"), ("L98", "L19"), ("L98 + Midy98", "L98"), ("L98 + Midy98", "L19"))
        baseline_gain = {f"{a} vs {b}": _delta(base_correct[a], base_correct[b], B, s + 30 + k) for k, (a, b) in enumerate(pairs_)
                         if a in base_correct and b in base_correct}
        rec.update({"accuracy": acc, "gain_over_baseline": gains, "baseline_gain": baseline_gain, "label": _label(gains)})
        out.append(rec)
    return out


def _label(gains) -> str:
    def adds(name):
        g = gains.get(name, {}).get("entropy")
        return g is not None and g["ci95"][0] > 0

    parts = [f"entropy {'adds' if adds(nm) else 'adds nothing'} over {nm}" for nm in gains]
    return "; ".join(parts)


def ablation(table, cfg, rows, blocks, ndr_cols, seed) -> Dict[str, Any]:
    """Gain of each NDR feature family (row.<set>.<family>.<agg>) over L19 and over L19 + Midy19."""
    split = table["split"]
    y = np.minimum(table["label.omega"], int(cfg["max_omega_class"])).astype(int)
    tr, te = rows[split[rows] == 0], rows[split[rows] == 2]
    fams: Dict[str, List[str]] = {}
    for c in ndr_cols:
        f = c.split(".")[2]
        if f not in ("logL", "L"):
            fams.setdefault(f, []).append(c)
    out: Dict[str, Any] = {"test_rows": int(te.size), "families": {f: len(v) for f, v in fams.items()}, "over": {}}
    for bname in ("L19", "L19 + Midy19"):
        Xb = blocks[bname]
        cb = _correct(cfg, seed, Xb, y, tr, te)
        res = {}
        for j, (f, cols) in enumerate(sorted(fams.items())):
            cf = _correct(cfg, seed, np.column_stack([Xb] + [table[c] for c in cols]), y, tr, te)
            res[f] = _delta(cf, cb, int(cfg["bootstrap"]), seed + 40 + j)
        out["over"][bname] = {"baseline_accuracy": float(cb.mean()), "gain": res}
    return out


# ---------------------------------------------------------------------------------------------
# 7b — mechanism: pair-level entropy deficit and residue classes
# ---------------------------------------------------------------------------------------------


def pair_mechanism(table, Lmat, Mmat, bases98: Sequence[int], nbins: int) -> Dict[str, Any]:
    pn, pb, pL = table["pair.n"], table["pair.b"].astype(np.int64), table["pair.L"].astype(np.int64)
    idx = np.searchsorted(table["row.n"], pn)
    col = {b: j for j, b in enumerate(bases98)}
    known = np.array([int(b) in col for b in pb])
    j = np.array([col.get(int(b), 0) for b in pb])
    midy = np.where(known, Mmat[idx, j], 0).astype(float)
    agree = bool(np.all(Lmat[idx[known], j[known]] == pL[known]))
    split = table["split"][idx]
    omega = table["label.omega"][idx]
    out: Dict[str, Any] = {"pairs": int(pn.size), "orders_match_pair_table": agree, "midy_share": float(midy[known].mean()),
                           "midy_share_by_omega": {int(w): float(midy[known & (omega == w)].mean()) for w in np.unique(omega) if (known & (omega == w)).sum() > 100},
                           "log2": math.log(2), "variants": {}}
    lL, lb, ln = table["pair.logL"], np.log(pb.astype(float)), np.log(pn.astype(float))
    for var in ("H_nodc", "H_paper"):
        H = table[f"pair.{var}"]
        ok = known & np.isfinite(H) & (pL > 2)
        tr, te = ok & (split == 0), ok & (split == 2)
        X0 = claims.design(lL, lb, ln)
        X1 = claims.design(lL, lb, ln, midy)
        c0, c1 = claims.lstsq(X0[tr], H[tr]), claims.lstsq(X1[tr], H[tr])
        r0, r1 = claims.r2(H[te], X0[te] @ c0), claims.r2(H[te], X1[te] @ c1)
        out["variants"][var] = {"train_pairs": int(tr.sum()), "test_pairs": int(te.sum()), "r2_baseline": r0, "r2_with_midy": r1,
                                "residual_variance_explained": (r1 - r0) / (1 - r0) if r0 < 1 else float("nan"), "midy_coefficient": float(c1[-1])}
    # binned means for the figure: test pairs, H_nodc by log L, split by flag
    H = table["pair.H_nodc"]
    ok = known & np.isfinite(H) & (pL > 2) & (split == 2)
    edges = np.quantile(lL[ok], np.linspace(0, 1, nbins + 1))
    b = np.clip(np.searchsorted(edges, lL[ok], side="right") - 1, 0, nbins - 1)
    curves = {}
    for flag in (0, 1):
        s = midy[ok] == flag
        curves[str(flag)] = [[float(lL[ok][s & (b == k)].mean()), float(H[ok][s & (b == k)].mean()), int((s & (b == k)).sum())]
                             for k in range(nbins) if (s & (b == k)).sum() >= 20]
    out["binned_H_nodc_by_logL"] = curves
    return out


def residues(table, claims_cfg, base_sets, seed, Mmat, bases98) -> Dict[str, Any]:
    cfg = {**(claims_cfg or {}), "_base_sets": {k: v["bases"] for k, v in dataset.normalize_base_sets(base_sets).items()}}
    ctx = claims.Ctx(table, cfg, seed, runtime.REPO_ROOT)
    pbases = sorted(claims._set_bases(ctx, ctx.cfg["primary_set"]))
    coprime = [(np.gcd(ctx.n, b) == 1).astype(float) for b in pbases]
    Lmat, _ = claims._per_base_logL(ctx, pbases)
    per_base_L = [np.nan_to_num(Lmat[:, j], nan=0.0) for j in range(Lmat.shape[1])]
    col = {b: j for j, b in enumerate(bases98)}
    midy_cols = [Mmat[:, col[b]].astype(float) for b in pbases if b in col]
    out: Dict[str, Any] = {"controls": "C7 controls (log n, per-base log L, coprimality, ω dummies as labelled control)",
                           "midy_controls": f"Midy flag of each {ctx.cfg['primary_set']} base", "variants": {}}
    for var in ctx.cfg["entropies"][:2]:
        H, cols, tr, te = claims.c7_design(ctx, var, per_base_L, coprime)
        out["variants"][var] = {"test_rows": int(te.sum()),
                                "c7_controls": claims.residue_excess(ctx, H, cols, tr, te),
                                "c7_controls_plus_midy": claims.residue_excess(ctx, H, cols + midy_cols, tr, te)}
    return out


# ---------------------------------------------------------------------------------------------
# 7c — lambda(n) and omega
# ---------------------------------------------------------------------------------------------


def lambda_analysis(table, cfg, rows_by_bin, L19, L98, M98, seed) -> Dict[str, Any]:
    n = table["row.n"].astype(np.int64)
    lam = table["label.lambda"].astype(np.int64)  # label: diagnostics and labelled control only
    phi = table["label.phi"].astype(np.int64)
    omega = table["label.omega"].astype(np.int64)
    lcm19 = np.lcm.reduce(np.where(L19 > 0, L19, 1), axis=1)
    lcm98 = np.lcm.reduce(np.where(L98 > 0, L98, 1), axis=1)
    odd = n % 2 == 1
    out: Dict[str, Any] = {
        "note": "λ(n) and φ(n) are factorization labels: used for coverage diagnostics and as a labelled control, never as inputs of a length-only model.",
        "coverage_all": {"lcm19 == λ": float((lcm19 == lam).mean()), "lcm98 == λ": float((lcm98 == lam).mean()), "lcm98 divides λ": bool(np.all(lam % lcm98 == 0))},
        "theorem_1_3_odd_n": {"rows": int(odd.sum()), "violations of 2^(ω−1) ≤ φ/λ": int((((2 ** (omega - 1)) * lam > phi) & odd).sum())},
    }
    bound = np.ceil(np.log2(n / lcm98) - 1e-12).astype(int)
    out["length_only_bound_odd_n"] = {
        "bound": "ω ≤ ceil(log2(n / lcm98)) (odd n; from Theorem 1.3 with φ < n and lcm98 ≤ λ)",
        "violations": int(((omega > bound) & odd & (n > 2)).sum()),
        "share_tight": float((omega == bound)[odd & (n > 2)].mean()),
        "mean_slack": float((bound - omega)[odd & (n > 2)].mean()),
    }
    K = int(cfg["max_omega_class"])
    y = np.minimum(omega, K).astype(int)
    split = table["split"]
    logn = np.log(n.astype(float))
    msh = midy_block(L98, M98)[0][:, -1]
    sets = {
        "log n, log λ (labelled factorization-derived control)": np.column_stack([logn, np.log(lam.astype(float))]),
        "log n, log lcm98": np.column_stack([logn, np.log(lcm98.astype(float))]),
        "log n, log lcm98, Midy share 98": np.column_stack([logn, np.log(lcm98.astype(float)), msh]),
    }
    per_bin = []
    for (lo, hi), rows in rows_by_bin:
        tr, te = rows[split[rows] == 0], rows[split[rows] == 2]
        rec = {"bin": [int(lo), int(hi)], "coverage": {"lcm19 == λ": float((lcm19[rows] == lam[rows]).mean()), "lcm98 == λ": float((lcm98[rows] == lam[rows]).mean())}}
        if tr.size >= 50 and te.size >= 30:
            rec["accuracy"] = {k: float(_correct(cfg, seed, X, y, tr, te).mean()) for k, X in sets.items()}
        per_bin.append(rec)
    out["bins"] = per_bin
    return out


def formula_search(table, cfg, base_sets, seed, run_dir, extra_cols, log=print) -> Dict[str, Any]:
    """Evolvo formulas for omega: the step-4 inputs vs the 2..99 length inputs, same table and budget."""
    import search_formulas as sf

    fcfg = {**sf.DEFAULTS, **cfg["formulas"], "targets": ["omega"], "warm_start": False, "q_target": {"omega": 0.7}}
    t2 = {**table, **extra_cols}
    out: Dict[str, Any] = {"settings": {k: fcfg[k] for k in ("guides", "seeds", "population", "generations", "parsimony", "eval_size_min", "eval_size_max", "eval_size_step")}, "input_sets": {}}
    workers = int(fcfg["workers"]) or max(1, (os.cpu_count() or 2) - 1)
    for label, inputs in (("step-4 inputs (19 bases + entropy)", sf.DEFAULT_INPUTS), ("lengths 2..99 + Midy share", FORMULA_INPUTS_98)):
        c = {**fcfg, "inputs": list(inputs)}
        prep = sf.prepare(t2, c, base_sets, seed)
        specs = [{"target": "omega", "guide": g, "seed": s, "variant": "adaptive", "cfg": c} for g in c["guides"] for s in c["seeds"]]
        runs = sf._run_specs(specs, {"names": prep["names"], "targets": prep["targets"]}, workers, log)
        best = min(runs, key=lambda r: (r["selected"]["val"]["nmse"], r["selected"]["effective_instructions"]))
        accs = [r["selected"]["test"]["accuracy"] for r in runs]
        out["input_sets"][label] = {
            "inputs": list(inputs),
            "runs": [{"guide": r["guide"], "seed": r["seed"], "val_nmse": r["selected"]["val"]["nmse"], "test": r["selected"]["test"], "expression": r["selected"]["expression"]} for r in runs],
            "test_accuracy_mean": float(np.mean(accs)), "test_accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "best_by_validation": {"guide": best["guide"], "seed": best["seed"], "expression": best["selected"]["expression"], "test": best["selected"]["test"], "val_nmse": best["selected"]["val"]["nmse"]},
        }
        log(f"  formulas [{label}]: mean test accuracy {np.mean(accs):.4f}; best {best['selected']['test']['accuracy']:.4f}")
    return out


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------


def run_stage(table, lcfg_in: Dict[str, Any], claims_cfg: Dict[str, Any], base_sets, formula_inputs, seed: int, run_dir: Path, dataset_key: str, log=print) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(lcfg_in or {})}
    cfg["formulas"] = {**DEFAULTS["formulas"], **((lcfg_in or {}).get("formulas") or {})} if cfg.get("formulas") is not None else None
    t0 = time.time()
    n = table["row.n"].astype(np.int64)
    bases98 = dataset.parse_bases(cfg["bases"])
    L98, M98, lkey = load_or_compute(run_dir, dataset_key, n, bases98, int(cfg["workers"]), log)
    b19 = features.union_bases(base_sets, cfg["sets"])
    missing = [b for b in b19 if b not in bases98]
    if missing:
        raise ValueError(f"length bases {cfg['bases']} must contain the {cfg['sets']} bases; missing {missing}")
    j19 = [bases98.index(b) for b in b19]
    L19, M19 = L98[:, j19], M98[:, j19]

    fsets = baselines.feature_sets(table, base_sets, formula_inputs, cfg["sets"])
    allowed = dataset.allowed_input_columns(table)
    ndr_cols = [c for c in allowed if not c.startswith("row.wide.") and not c.startswith("row.n_bases.") and c != "row.n"]
    entropy_cols = [c for c in ndr_cols if c.split(".")[-2].startswith(("H_", "Hn_"))]
    X98, _ = length_block(n, L98)
    m19, _ = midy_block(L19, M19)
    m98, _ = midy_block(L98, M98)
    blocks = {
        "L19": fsets["n + L arithmetic"][0],
        "L19 + Midy19": np.column_stack([fsets["n + L arithmetic"][0], m19]),
        "L98": X98,
        "L98 + Midy98": np.column_stack([X98, m98]),
    }
    log(f"  7a: C9 per n-range, baselines {list(blocks)}")
    bins = c9_bins(table, cfg, fsets, entropy_cols, ndr_cols, blocks, seed, log)

    rng = np.random.default_rng([seed, 50])
    rows_by_bin = [((lo, hi), scale._balanced(np.nonzero((n >= lo) & (n <= hi))[0], table["label.omega"], int(cfg["per_omega_cap"]), int(cfg["min_per_omega"]), rng))
                   for lo, hi in cfg["bins"]]
    abl = None
    usable = [(i, b, r) for i, (b, r) in enumerate(rows_by_bin) if r.size and (table["split"][r] == 2).sum() >= 30]
    if cfg["ablation"] and usable:
        i, (lo, hi), rows = usable[-1]
        log(f"  7a: NDR family ablation on [{lo}, {hi}]")
        abl = {"bin": [int(lo), int(hi)], **ablation(table, cfg, rows, blocks, ndr_cols, seed + 100 * i)}

    log("  7b: pair-level Midy mechanism and residue classes")
    mech = pair_mechanism(table, L98, M98, bases98, int(cfg["pair_bins"]))
    res = residues(table, claims_cfg, base_sets, seed, M98, bases98)

    log("  7c: λ(n) coverage, Theorem 1.3 bound, ω from n/λ")
    lam = lambda_analysis(table, cfg, rows_by_bin, L19, L98, M98, seed)
    form = None
    if cfg["formulas"]:
        log("  7c: Evolvo formula search for ω (step-4 inputs vs 2..99 length inputs)")
        form = formula_search(table, cfg, base_sets, seed, run_dir, summary_columns(n, L98, M98, "row.len98"), log)
    return {"version": LENGTHS_VERSION, "settings": cfg, "length_bases": bases98, "bases19": b19, "lengths_cache_key": lkey,
            "entropy_columns": len(entropy_cols), "ndr_columns": len(ndr_cols),
            "c9_bins": bins, "ablation": abl, "pair_mechanism": mech, "residues": res, "lambda": lam, "formulas": form,
            "seconds": round(time.time() - t0, 1)}
