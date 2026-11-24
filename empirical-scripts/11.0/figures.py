"""Static report figures for Sieve Echo v11 (ROADMAP step 6).

``make_figures(run_dir, table=None)`` writes PNGs into ``<run_dir>/figures/`` from whatever the run
contains (the dataset cache for the scatter, ``scale.json``, ``formulas.json``, ``baselines.json``,
``nn.json``) and returns ``[(file, caption), ...]`` for the report. Every number drawn is also in
the report's tables (the figures' table view).

Styling follows a validated light palette: categorical slots blue/orange/aqua (adjacent CVD ΔE ≥ 9),
a single-hue blue ramp for the ordinal ω, recessive grid, thin marks, direct labels on lines (aqua is
below 3:1 contrast, so it is always labelled), distinct markers as secondary encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import runtime

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
BLUE_RAMP = {1: "#9ec5f4", 2: "#6da7ec", 3: "#3987e5", 4: "#256abf", 5: "#184f95", 6: "#0d366b", 7: "#0d366b"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })
    return plt


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path.name


_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _decade_label(lo: int, hi: int) -> str:
    a, b = int(np.floor(np.log10(max(lo, 1)))), int(round(np.log10(max(hi, 1))))
    return f"10{str(a).translate(_SUP)}–10{str(b).translate(_SUP)}"


def fig_entropy_vs_length(table, out: Path) -> Optional[Tuple[str, str]]:
    plt = _plt()
    H, L, w = table["row.v8.H_paper.mean"], table["row.v8.logL.mean"], table["label.omega"]
    sel = np.nonzero((table["split"] == 2) & np.isfinite(H) & np.isfinite(L))[0]
    if sel.size < 10:
        return None
    rng = np.random.default_rng(0)
    if sel.size > 6000:
        sel = np.sort(rng.choice(sel, 6000, replace=False))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in sorted(np.unique(w[sel])):
        m = sel[w[sel] == k]
        ax.scatter(L[m], H[m], s=9, color=BLUE_RAMP.get(int(k), BLUE_RAMP[7]), alpha=0.75, linewidths=0, label=f"ω = {int(k)}")
    xs = np.linspace(np.nanmin(L[sel]), np.nanmax(L[sel]), 50)
    ax.plot(xs, xs, color=TEXT_2, lw=1, ls="--")
    ax.annotate("H = log L (upper bound)", (xs[-12], xs[-12]), xytext=(-150, 8), textcoords="offset points", color=TEXT_2, fontsize=9)
    ax.set_xlabel("mean log repetend length, log L (v8 bases)")
    ax.set_ylabel("mean NDR entropy H_paper (v8 bases)")
    ax.set_title("Entropy follows repetend length; ω sorts along the same line")
    ax.legend(title="distinct prime factors", ncol=2, fontsize=8, title_fontsize=8, loc="upper left")
    return _save(fig, out / "entropy_vs_length.png"), "Test-split integers: mean H_paper against mean log L, coloured by ω. Points with more prime factors sit lower because their repetends are shorter, not off the H–log L curve (claims C1, C9)."


def fig_c9_by_decade(scale: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    plt = _plt()
    bins = [b for b in scale["bins"] if "C9" in b]
    if not bins:
        return None
    labels = [f"{_decade_label(*b['bin'])}\nbaseline {b['C9']['accuracy_baseline'] * 100:.1f}%" for b in bins]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color=TEXT_2, lw=1)
    for j, (key, name) in enumerate((("delta_entropy", "+ entropy columns"), ("delta_all", "+ all NDR columns"))):
        v = np.array([b["C9"][key] for b in bins]) * 100
        ci = np.array([b["C9"][key + "_ci"] for b in bins]) * 100
        xx = x + (j - 0.5) * 0.14
        ax.errorbar(xx, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt=MARKERS[j], ms=7, color=SERIES[j], ecolor=SERIES[j], elinewidth=2, capsize=0, lw=2, label=name)
        ax.annotate(name, (xx[-1], v[-1]), xytext=(12, 9 if j == 0 else -9), textcoords="offset points", color=TEXT, fontsize=9, va="center")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.4, len(bins) - 0.4 + 0.6)
    ax.set_ylabel("accuracy gain over length baseline (points)")
    ax.set_title("What NDR adds to ω classification beyond n and repetend lengths")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out / "c9_gain_by_decade.png"), "Held-out gain (95% bootstrap CI) of gradient boosting when entropy or all NDR columns are added to the baseline of n, per-base repetend lengths and their lcm arithmetic, per ω-balanced n-range (step 5)."


def fig_invariance_by_decade(scale: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    plt = _plt()
    bins = [b for b in scale["bins"] if "C4" in b and "v8" in b["C4"]]
    if not bins:
        return None
    labels = [_decade_label(*b["bin"]) for b in bins]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = {"H_paper": "H_paper (paper entropy)", "Hn_nodc": "Hn_nodc (DC removed, / log L)", "logL": "log L itself"}
    for j, key in enumerate(("H_paper", "Hn_nodc", "logL")):
        v = np.array([b["C4"]["v8"]["invariance_rate"][key] for b in bins]) * 100
        ax.plot(x, v, marker=MARKERS[j], ms=7, lw=2, color=SERIES[j], label=names[key])
        ax.annotate(names[key], (x[-1], v[-1]), xytext=(10, 0), textcoords="offset points", color=TEXT, fontsize=9, va="center")
    ax.axhline(6, color=TEXT_2, lw=1, ls="--")
    ax.annotate("paper: 6%", (0, 6), xytext=(0, 4), textcoords="offset points", color=TEXT_2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(bins) - 0.3 + 1.4)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.74), fontsize=8)
    ax.set_ylabel("share of n with CV < 0.1 across v8 bases (%)")
    ax.set_title("Base invariance: rare for the paper entropy, near-total once it saturates")
    return _save(fig, out / "invariance_by_decade.png"), "Share of integers whose statistic varies by less than 10% (CV) across the v8 bases, per ω-balanced n-range (claim C4, step 5)."


def fig_growth(scale: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    plt = _plt()
    g = scale.get("growth") or {}
    if "balanced_all_bins" not in g:
        return None
    ga = g["balanced_all_bins"]
    rows = [("paper / v7: H ~ n^a", g.get("paper_exponent", 0.142))]
    if "v8_style_H_v8_paper_range" in g:
        rows.append(("v11, v8-style fit on [2, 9999]", g["v8_style_H_v8_paper_range"]["exponent"]))
    rows += [
        ("v11, ω-balanced decades", ga["H ~ n^a"]),
        ("  + ω fixed effects", ga["H ~ n^a, omega fixed effects"]),
        ("  + ω fixed effects + log L control", ga["H ~ n^a · (log L)^b, omega fixed effects"]["a"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(y, vals, height=0.55, color=SERIES[0])
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, max(vals) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("growth exponent a in H ~ n^a")
    t = scale.get("T_x")
    sub = f" (the paper's 'predicted 0.743' is the exponent of a different sum; recomputed: {t['exponent']:.2f})" if t else ""
    ax.set_title("Entropy growth with n disappears once repetend length is controlled")
    fig.text(0.0, -0.04, sub.strip(), color=TEXT_2, fontsize=8)
    return _save(fig, out / "growth_exponents.png"), "Growth exponents of the NDR entropy with n, with and without ω and log L controls (step 5)."


def fig_heldout(baselines: Dict[str, Any], nn: Dict[str, Any], formulas: Optional[Dict[str, Any]], out: Path) -> Optional[Tuple[str, str]]:
    plt = _plt()
    rows = baselines.get("targets", {}).get("omega")
    if not rows:
        return None
    items = [(k, v["accuracy"]) for k, v in rows.items()]
    if formulas and "omega" in formulas.get("summary", {}).get("best_by_validation", {}):
        items.append(("GFSL formula (best on validation)", formulas["summary"]["best_by_validation"]["omega"]["test"]["accuracy"]))
    for enc, r in (nn or {}).get("encodings", {}).items():
        items.append((f"GFSL-NN {enc}", r["test"]["accuracy"]))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(items) + 1.0))
    y = np.arange(len(items))
    ax.barh(y, [v * 100 for _, v in items], height=0.6, color=SERIES[0])
    for yi, (_, v) in zip(y, items):
        ax.annotate(f"{v * 100:.1f}%", (v * 100, yi), xytext=(4, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(y, [k for k, _ in items])
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test accuracy, ω classes min(ω, 5) (%)")
    ax.set_title("Held-out ω prediction: repetend-length arithmetic does almost all the work")
    return _save(fig, out / "heldout_omega.png"), "Test accuracy of every ω model of step 4 (same split, same classes)."


def fig_guides(formulas: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    plt = _plt()
    runs = [r for r in formulas.get("runs", []) if r["target"] == "omega"]
    if not runs:
        return None
    groups: Dict[str, List[float]] = {}
    order = [("none", "adaptive"), ("none", "fixed"), ("qlearning", "adaptive"), ("supervised", "adaptive"), ("policy", "adaptive"), ("policy", "warm")]
    for g, v in order:
        vals = [r["selected"]["test"]["nmse"] for r in runs if r["guide"] == g and r["variant"] == v]
        if vals:
            groups[f"{g} · {'fixed N' if v == 'fixed' else ('warm start' if v == 'warm' else 'adaptive N')}"] = vals
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names = list(groups)
    rng = np.random.default_rng(1)
    for i, k in enumerate(names):
        v = np.array(groups[k])
        ax.scatter(i + rng.uniform(-0.08, 0.08, v.size), v, s=36, color=SERIES[0], linewidths=0, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=TEXT, lw=2, zorder=4)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel("test NMSE of the selected formula")
    ax.set_title("ω formula search: guides and evaluation-size schedules (5 seeds each)")
    return _save(fig, out / "formula_guides_omega.png"), "Test NMSE of the validation-selected ω formula per seed (dots) and mean (bar) for each guide / S1 schedule / S2 warm start (step 4)."


def fig_midy_gap(lengths: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    mech = lengths.get("pair_mechanism") or {}
    curves = mech.get("binned_H_nodc_by_logL") or {}
    if not curves.get("0") or not curves.get("1"):
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, (flag, name) in enumerate((("0", "−1 ∉ ⟨b⟩ mod n"), ("1", "−1 ∈ ⟨b⟩ mod n (Midy)"))):
        c = np.array(curves[flag])
        ax.plot(c[:, 0], c[:, 1], marker=MARKERS[j], ms=5, lw=2, color=SERIES[j], label=name)
        ax.annotate(name, (c[-1, 0], c[-1, 1]), xytext=(8, 10 if j == 0 else -12), textcoords="offset points", color=TEXT, fontsize=9, va="center")
    coef = mech["variants"]["H_nodc"]["midy_coefficient"]
    ax.set_xlabel("log repetend length, log L (one (n, b) pair)")
    ax.set_ylabel("mean DC-free NDR entropy H_nodc")
    ax.set_title(f"Midy pairs sit log 2 lower: fitted gap {coef:+.3f} (log 2 = 0.693)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    return _save(fig, out / "midy_entropy_gap.png"), ("Test-split (n, b) pairs of the 19 v8/v10/primes31 bases: mean H_nodc per log L bin, split by the Midy flag b^(L/2) ≡ −1 (mod n). "
                                                      "When it holds the two halves of the repetend are digit complements and every even DFT bin except DC vanishes (step 7b).")


def fig_c9_by_baseline(lengths: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    bins = [b for b in lengths.get("c9_bins", []) if "gain_over_baseline" in b]
    names = [nm for nm in ("L19", "L19 + Midy19", "L98 + Midy98") if all(nm in b["gain_over_baseline"] for b in bins)]
    if not bins or not names:
        return None
    plt = _plt()
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhline(0, color=TEXT_2, lw=1)
    desc = {"L19": "19-base lengths", "L19 + Midy19": "19-base lengths + Midy flags", "L98 + Midy98": "lengths + Midy, bases 2…99"}
    for j, nm in enumerate(names):
        v = np.array([b["gain_over_baseline"][nm]["entropy"]["value"] for b in bins]) * 100
        ci = np.array([b["gain_over_baseline"][nm]["entropy"]["ci95"] for b in bins]) * 100
        xx = x + (j - (len(names) - 1) / 2) * 0.16
        ax.errorbar(xx, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt=MARKERS[j], ms=7, color=SERIES[j], ecolor=SERIES[j], elinewidth=2, capsize=0, lw=0, label=f"over {desc[nm]}")
        ax.annotate(desc[nm], (xx[-1], v[-1]), xytext=(x[-1] + 0.3, v[-1]), textcoords="data", color=TEXT, fontsize=8.5, va="center")
    labels = [f"{_decade_label(*b['bin'])}\n" + " / ".join(f"{b['accuracy'][nm] * 100:.1f}" for nm in names) + "%" for b in bins]
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(bins) - 0.5 + 0.9)
    ax.set_ylabel("accuracy gain from entropy columns (points)")
    ax.set_title("Entropy's gain shrinks as the length baseline gets stronger")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out / "c9_gain_by_baseline.png"), ("Held-out gain (95% bootstrap CI) from adding the entropy columns to three length baselines, per ω-balanced n-range; "
                                                         "tick labels give the baselines' own accuracies in the same order (step 7a).")


def fig_zeros_by_omega(zeros: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    bins = [b for b in (zeros.get("describe") or {}).get("by_bin", []) if b["by_omega"]]
    if not bins:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, b in enumerate(bins[:3]):
        w = np.array(sorted(int(k) for k in b["by_omega"]))
        v = np.array([b["by_omega"][str(k)] if str(k) in b["by_omega"] else b["by_omega"][k] for k in w]) * 100
        name = _decade_label(*b["bin"])
        ax.plot(w, v, marker=MARKERS[j], ms=6, lw=2, color=SERIES[j], label=f"n in {name}")
        ax.annotate(name, (w[-1], v[-1]), xytext=(8, (1 - j) * 9), textcoords="offset points", color=TEXT, fontsize=9, va="center")
    ax.set_xlabel("distinct prime factors ω(n)")
    ax.set_ylabel("(n, b) pairs with extra spectral zeros (%)")
    ax.set_xticks(sorted({int(k) for b in bins for k in b["by_omega"]}))
    ax.set_ylim(bottom=0)
    ax.set_title("Exact zeros beyond Midy's appear with more prime factors")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.5)
    return _save(fig, out / "spectral_zeros_by_omega.png"), ("Share of (n, b) pairs (19 v8/v10/primes31 bases) whose NDR spectrum has a vanishing frequency class other than Midy's even bins, "
                                                              "by ω and n-range (step 8a). Imprimitive characters contribute Euler factors (1 − χ*(p)) that vanish at the prime factors of n.")


def fig_tworank(d: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    decs = d.get("decades") or []
    if not decs:
        return None
    names = list(decs[0]["accuracy"])
    pick = [names[0], names[-2], names[-1]] if len(names) >= 3 else names
    plt = _plt()
    x = np.arange(len(decs))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    legacy = {"lengths 2..199 (step 9b)": "raw lengths (step 9b)", "ℓ-adic profiles 2..499": "ℓ-adic order profiles", "lengths 2..199 + ℓ-adic 2..499": "both"}
    short = {nm: legacy.get(nm, nm) for nm in pick}
    ends = []
    for j, nm in enumerate(pick):
        v = np.array([x_["accuracy"][nm] for x_ in decs]) * 100
        ax.plot(x, v, marker=MARKERS[j], ms=6, lw=2, color=SERIES[j], label=short.get(nm, nm))
        ends.append((v[-1], short.get(nm, nm)))
    lo_, hi_ = ax.get_ylim()
    gap = 0.06 * (hi_ - lo_)  # keep end labels apart when lines finish close together
    placed = []
    for y, label in sorted(ends):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.text(x[-1] + 0.12, y, label, color=TEXT, fontsize=9, va="center")
    ax.set_xticks(x, [_decade_label(*x_["decade"]) for x_ in decs])
    ax.set_xlim(-0.3, len(decs) - 1 + 1.1)
    ax.set_ylabel("held-out accuracy for min(ω, 5) (%)")
    ax.set_title("ω from multiplicative orders: raw lengths vs 2-adic and 3-adic structure")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, out / "tworank_accuracy.png"), ("Gradient-boosting accuracy per decade on the same integers: raw per-base log lengths with lcm arithmetic and Midy flags (bases 2…199), "
                                                      "the histograms of the 2-adic and 3-adic valuations of the orders (bases 2…499) with log n and v₂(n), and both (step 10).")


def fig_decode(d: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    rows = d.get("balanced") or []
    if not rows:
        return None
    pick = rows if len(rows) <= 4 else [rows[i] for i in sorted({0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1})]
    ramp = [BLUE_RAMP[k] for k in (2, 3, 5, 6)][-len(pick):] if len(pick) <= 4 else list(BLUE_RAMP.values())
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    k = np.arange(1, len(d["bases"]) + 1)
    ends = []
    for j, r in enumerate(pick):
        v = np.array(r["accuracy_by_prefix"]) * 100
        lab = _decade_label(*r["decade"])
        ax.plot(k, v, marker="o", ms=4, lw=2, color=ramp[j], label=lab)
        ends.append((v[2], lab))
    ax.set_xticks(k, [str(b) for b in d["bases"]], fontsize=8)
    ax.set_xlabel("bases used, in order (the last tick is the 19th base)")
    ax.set_ylabel("integers with ω̂ = ω exactly (%)")
    ax.set_ylim(min(40, ax.get_ylim()[0]), 101)
    ax.axhline(100, color=GRID, lw=1, zorder=0)
    ax.set_title("ω decoded exactly from repetend lengths, ω-balanced samples")
    ax.legend(title="decade of n", loc="lower right", fontsize=8)
    return _save(fig, out / "decode_bases.png"), ("Share of integers whose ω is recovered exactly by the training-free decoder (Proposition 1.7) using the first k of the 19 bases, "
                                                  "per decade of n (ω-balanced samples, step 13). Base 4 adds nothing to base 2 because 4 = 2².")


def fig_failures(d: Dict[str, Any], out: Path) -> Optional[Tuple[str, str]]:
    rows = [c for c in d.get("prefix_checks", []) if c["certificate"].get("certified_exact_below")]
    if not rows:
        return None
    full = d["full"]
    ks = [c["k"] for c in rows] + [len(full["bases"])]
    certs = [c["certificate"] for c in rows] + [full]
    v = np.array([c["certified_exact_below"] for c in certs], dtype=float)
    exact = np.array([bool(c["exact"]) for c in certs])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(ks, v, lw=2, color=SERIES[0], zorder=2)
    ax.scatter(np.array(ks)[exact], v[exact], s=40, color=SERIES[0], zorder=3, label="least failing n (exact)")
    if (~exact).any():
        ax.scatter(np.array(ks)[~exact], v[~exact], s=60, marker="^", facecolor=SURFACE, edgecolor=SERIES[0], lw=2, zorder=3, label="certified lower bound")
    ax.axhline(1e8, color=TEXT, lw=1, ls=(0, (4, 3)), alpha=0.55, zorder=1)
    ax.text(ks[0], 1.3e8, "step 13: all 19 bases checked exhaustively to 10⁸", fontsize=8, color=TEXT, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(ks, [str(k) for k in ks], fontsize=8)
    ax.set_xlabel("number of bases (first k of 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, …, 31)")
    ax.set_ylabel("decoder exact for every n below")
    ax.set_title("Where the exact decoder first fails (with pre-periods)")
    ax.set_ylim(top=max(v.max(), 1e8) * 12)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out / "decoder_first_failure.png"), ("Least integer on which the step-13 decoder with pre-periods misses ω, for the first k of the 19 bases, from Proposition 1.8 "
                                                          "(order vectors of all primes up to the prime limit); triangles are certified lower bounds (step 15).")


def make_figures(run_dir: Path, table: Optional[Dict[str, np.ndarray]] = None) -> List[Tuple[str, str]]:
    out = run_dir / "figures"
    made: List[Tuple[str, str]] = []

    def add(item):
        if item:
            made.append((f"figures/{item[0]}", item[1]))

    if table is not None and "row.v8.H_paper.mean" in table:
        add(fig_entropy_vs_length(table, out))
    if (run_dir / "baselines.json").is_file():
        nn = runtime.read_json(run_dir / "nn.json") if (run_dir / "nn.json").is_file() else {}
        form = runtime.read_json(run_dir / "formulas.json") if (run_dir / "formulas.json").is_file() else None
        add(fig_heldout(runtime.read_json(run_dir / "baselines.json"), nn, form, out))
    if (run_dir / "formulas.json").is_file():
        add(fig_guides(runtime.read_json(run_dir / "formulas.json"), out))
    if (run_dir / "scale.json").is_file():
        sc = runtime.read_json(run_dir / "scale.json")
        add(fig_c9_by_decade(sc, out))
        add(fig_invariance_by_decade(sc, out))
        add(fig_growth(sc, out))
    if (run_dir / "lengths.json").is_file():
        ln = runtime.read_json(run_dir / "lengths.json")
        add(fig_c9_by_baseline(ln, out))
        add(fig_midy_gap(ln, out))
    if (run_dir / "zeros.json").is_file():
        add(fig_zeros_by_omega(runtime.read_json(run_dir / "zeros.json"), out))
    if (run_dir / "tworank.json").is_file():
        add(fig_tworank(runtime.read_json(run_dir / "tworank.json"), out))
    if (run_dir / "decode.json").is_file():
        add(fig_decode(runtime.read_json(run_dir / "decode.json"), out))
    if (run_dir / "failures.json").is_file():
        add(fig_failures(runtime.read_json(run_dir / "failures.json"), out))
    return made
