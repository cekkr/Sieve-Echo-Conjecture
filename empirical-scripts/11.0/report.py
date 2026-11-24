"""Report for Sieve Echo v11: turns `claims.json` + `manifest.json` into `report.md` and `report.json`.

The Markdown has one verdict table (one row per claim) followed by a short section per claim with the
reproduced paper statistic and the corrected-test summary. Every number in it is read from
`claims.json`, so the report names the run, config, seed and code SHAs that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import runtime


def _short(sha: str | None) -> str:
    return (sha or "unknown")[:10]


def _cell(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact(obj: Any, digits: int = 4) -> str:
    def rnd(o):
        if isinstance(o, float):
            return round(o, digits)
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return o

    return json.dumps(rnd(obj), ensure_ascii=False)


def verdict_table(claims: list[dict]) -> str:
    rows = ["| Claim | Paper says | Paper statistic re-computed | Corrected test (held out, with controls) | Verdict |", "| --- | --- | --- | --- | --- |"]
    for c in claims:
        rep = c.get("reproduction", {})
        rows.append(
            f"| **{c['id']}** {_cell(c['title'], 60)} | {_cell(c['paper_claim'], 120)} | {rep.get('status', 'n/a')} | {_cell(c['reason'])} | **{c['verdict']}** |"
        )
    return "\n".join(rows)


def build_markdown(manifest: dict, claims_out: dict) -> str:
    env = manifest.get("environment", {})
    root, evo = env.get("git_root", {}), env.get("git_evolvo", {})
    data = manifest.get("stages", {}).get("data", {})
    s = claims_out.get("settings", {})
    lines = [
        f"# Sieve Echo v11 — claim verdicts ({manifest.get('run_id')})",
        "",
        f"- **Run:** `{manifest.get('run_id')}`, config `{manifest.get('config_path')}` (sha256 `{_short(manifest.get('config_sha256'))}`), seed {manifest.get('seed')}.",
        f"- **Code:** root `{_short(root.get('sha'))}`{' (dirty)' if root.get('dirty') else ''}, libs/evolvo `{_short(evo.get('sha'))}`{' (dirty)' if evo.get('dirty') else ''}, v11 sources sha256 `{_short(env.get('v11_code_sha256'))}`.",
        f"- **Definitions:** `{claims_out.get('ndr_version')}`; primary base set `{s.get('primary_set')}`; entropies {', '.join(f'`{e}`' for e in s.get('entropies', []))}.",
        f"- **Data:** {data.get('integers')} integers, {data.get('pairs')} coprime (n, b) pairs; split {data.get('split_counts')} (split sha256 `{_short(data.get('split_sha256'))}`).",
        f"- **Inference:** {s.get('bootstrap')} bootstrap resamples (95% percentile CIs over test rows), {s.get('permutations')} permutations; effect threshold {s.get('effect_threshold')} (ΔR², η² excess, r²).",
        "",
        "Paper statistics are re-computed the way the v8.2 code computed them (in-sample, `H_v8`, n ∈ [2, 9999]); "
        "corrected tests fit on the train split and report on the test split.",
        "",
        "## Verdicts",
        "",
        verdict_table(claims_out["claims"]),
        "",
    ]
    for c in claims_out["claims"]:
        rep = dict(c.get("reproduction", {}))
        status = rep.pop("status", "n/a")
        lines += [
            f"## {c['id']} — {c['title']}",
            "",
            f"- **Paper:** {c['paper_claim']}",
            f"- **Re-computed paper statistic ({status}):** {_cell(_compact(rep), 1500)}",
            f"- **Corrected test:** {c['reason']}",
            f"- **Verdict:** {c['verdict']}",
            "",
        ]
    lines += ["Full numbers: `claims.json` in this run directory.", ""]
    return "\n".join(lines)


def _ms(stat: dict, digits: int = 4) -> str:
    if not stat or stat.get("mean") is None:
        return "–"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}" if stat.get("std") is not None else f"{stat['mean']:.{digits}f}"


def _p(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def step4_markdown(run_dir: Path) -> str:
    lines: list = []
    f = run_dir / "formulas.json"
    if f.is_file():
        data = runtime.read_json(f)
        s = data["summary"]
        cfg = data["settings"]
        lines += [
            "## Step 4 — Evolvo formula search",
            "",
            f"Inputs ({len(data['inputs'])}, z-scored on train): " + ", ".join(f"`{x}`" for x in data["inputs"]) + ".",
            f"Population {cfg['population']}, {cfg['generations']} generations, seeds {cfg['seeds']}, parsimony {cfg['parsimony']} per instruction; "
            f"fitness = 1/(1+NMSE) − parsimony·size on the first N train rows (S1: N from {cfg['eval_size_min']} in steps of {cfg['eval_size_step']} to ≤ {cfg['eval_size_max']}, capped at {data['train_rows']} train rows). "
            f"q_target {cfg['q_target']}. Test metrics are for the genome selected on validation.",
            "",
            "| target | guide | variant | runs | test NMSE | test accuracy | reached q_target | evaluations to target | final N | seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in s["table"]:
            lines.append(
                f"| {r['target']} | {r['guide']} | {r['variant']} | {r['runs']} | {_ms(r['test_nmse'])} | {_ms(r['test_accuracy'])} | "
                f"{r['reached_target']}/{r['runs']} | {_ms(r['evaluations_to_target'], 0)} | {_ms(r['final_N'], 0)} | {_ms(r['seconds'], 1)} |"
            )
        lines += ["", "**S1 / S2 checks** (ROADMAP done-when conditions):", ""]
        for t, v in s["verdicts"].items():
            if "S1" in v:
                s1 = v["S1"]
                lines.append(f"- {t} S1: adaptive N test NMSE {_ms(s1['adaptive_test_nmse'])} vs fixed {_ms(s1['fixed_test_nmse'])}; N rose in {s1['runs_where_N_rose']} runs → **{'met' if s1['done'] else 'not met'}** (Mann–Whitney p = {_p(s1.get('p_value_test_nmse'))}).")
            if "S2" in v:
                s2 = v["S2"]
                lines.append(f"- {t} S2: policy vs no guide — fewer evaluations to target: {s2['policy_vs_none_fewer_evaluations_to_target']}, smaller spread: {s2['policy_vs_none_smaller_spread']}; warm start better than cold: {s2['warm_better_than_cold']} (lower test NMSE: {s2.get('warm_lower_test_nmse')}, fewer evaluations to target: {s2.get('warm_fewer_evaluations_to_target')}) → **{'met' if s2['done'] else 'not met'}** (test-NMSE p: policy vs none {_p(s2.get('p_value_policy_vs_none_test_nmse'))}, warm vs cold {_p(s2.get('p_value_warm_vs_cold_test_nmse'))}).")
        lines += ["", "**Best formulas (selected on validation; inputs are z-scores):**", ""]
        for t, b in s["best_by_validation"].items():
            lines.append(f"- {t} ({b['guide']}/{b['variant']}/seed {b['seed']}, {b['effective_instructions']} instructions): `{b['expression']}` — test NMSE {b['test']['nmse']:.4f}, accuracy {b['test']['accuracy']:.4f}")
        lines.append("")
    b = run_dir / "baselines.json"
    n = run_dir / "nn.json"
    if b.is_file():
        base = runtime.read_json(b)
        nn = runtime.read_json(n) if n.is_file() else {"encodings": {}}
        form = runtime.read_json(f)["summary"]["best_by_validation"] if f.is_file() else {}
        lines += ["## Held-out results (test split)", "", f"Train rows {base['train_rows']}, test rows {base['test_rows']}. ω is scored as classes min(ω, 5).", ""]
        for target, rows in base["targets"].items():
            second = "macro-F1" if target == "omega" else "F1"
            key = "macro_f1" if target == "omega" else "f1"
            lines += [f"**{target}**", "", f"| model | accuracy | {second} | MSE (class) | log-loss |", "| --- | --- | --- | --- | --- |"]
            for name, m in rows.items():
                lines.append(f"| {name} | {m['accuracy']:.4f} | {m.get(key, float('nan')):.4f} | {m['mse_class']:.4f} | {m.get('log_loss', float('nan')):.4f} |")
            if target in form:
                ft = form[target]["test"]
                lines.append(f"| GFSL formula (best on validation) | {ft['accuracy']:.4f} | {ft.get(key, float('nan')):.4f} | {ft['mse']:.4f} (raw target) | – |")
            if target == "omega":
                for enc, r in nn.get("encodings", {}).items():
                    m = r["test"]
                    lines.append(f"| GFSL-NN `{enc}` ({r['evaluated']} architectures) | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['mse_class']:.4f} | {m.get('log_loss', float('nan')):.4f} |")
            lines.append("")
        tf = base["three_feature_retest"]
        lines += ["**§3.1 three-feature retest** (paper: 98.8%, v7.0 training-set accuracy with `is_prime_power`)", "",
                  "| features | train accuracy | test accuracy |", "| --- | --- | --- |"]
        for name, v in tf["variants"].items():
            lines.append(f"| {name} | {v['train_accuracy']:.4f} | {v['test']['accuracy']:.4f} |")
        lines += [f"| majority class | – | {tf['majority_test_accuracy']:.4f} |", "", tf["note"], ""]
    sc = run_dir / "scale.json"
    if sc.is_file():
        lines += scale_markdown(runtime.read_json(sc))
    return "\n".join(lines)


def _ci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and all(x is not None for x in c) else "[n/a]"


def scale_markdown(d: dict) -> list:
    cfg = d["settings"]
    lines = ["## Step 5 — Scale (per n-range, ω-balanced samples)", "",
             f"Each range: at most {cfg['per_omega_cap']} integers per ω (ω strata with < {cfg['min_per_omega']} dropped); fits on the range's train rows, statistics on its test rows; {cfg['bootstrap']} bootstrap resamples. Entropy `{cfg['primary_set']}.{cfg['entropy']}`.", "",
             "| n range | rows (test) | C1 ΔR² of log ω [CI] | C1 partial r | C2 Spearman(H, ω) | C3 partial r | C4 invariance H / Hn_nodc / log L | C9 acc base → +entropy → +all NDR |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for b in d["bins"]:
        if "skipped" in b:
            lines.append(f"| {b['bin'][0]}–{b['bin'][1]} | {b['rows']} | skipped ({b['skipped']}) | | | | | |")
            continue
        c4 = b["C4"].get("v8", {}).get("invariance_rate", {})
        lines.append(
            f"| {b['bin'][0]}–{b['bin'][1]} | {b['rows']} ({b['test']}) | {b['C1']['delta_r2']:+.4f} {_ci(b['C1']['delta_r2_ci'])} | {b['C1']['partial_r']:+.3f} | "
            f"{b['C2']['spearman_H_omega']:+.3f} | {b['C3']['partial_r']:+.3f} | {c4.get('H_paper', float('nan')):.3f} / {c4.get('Hn_nodc', float('nan')):.3f} / {c4.get('logL', float('nan')):.3f} | "
            f"{b['C9']['accuracy_baseline']:.4f} → {b['C9']['accuracy_entropy']:.4f} → {b['C9']['accuracy_all']:.4f} (Δentropy {_ci(b['C9']['delta_entropy_ci'])}) |")
    lines += ["", "**Stability between the last two ranges:**", ""]
    for c, v in d["stability"].items():
        lines.append(f"- {c}: {v['labels'][0]} → {v['labels'][1]} — **{'stable' if v['stable'] else 'drift'}**")
    g = d["growth"]
    lines += ["", "**Growth exponents** (paper: H ~ n^0.142, 'predicted' 0.743):", ""]
    if "v8_style_H_v8_paper_range" in g:
        v = g["v8_style_H_v8_paper_range"]
        lines.append(f"- v8-style `H_v8` on all n in [2, 9999]: H ≈ {v['prefactor']:.3f}·n^{v['exponent']:.3f}")
    ga = g["balanced_all_bins"]
    lines.append(f"- ω-balanced ranges, `{cfg['entropy']}`: a = {ga['H ~ n^a']:.3f}; with ω fixed effects {ga['H ~ n^a, omega fixed effects']:.3f}; adding log(log L) as control: a = {ga['H ~ n^a · (log L)^b, omega fixed effects']['a']:.3f}, b = {ga['H ~ n^a · (log L)^b, omega fixed effects']['b']:.3f}")
    lines.append("- per-ω exponent: " + ", ".join(f"ω={w}: {a:.3f}" for w, a in g["per_omega_exponent"].items()))
    if d.get("T_x"):
        t = d["T_x"]
        lines.append(f"- T(x) = Σ_p Σ_n log ord_n(p) for x = {t['x']}: exponent {t['exponent']:.3f} (local {', '.join(f'{v:.2f}' for v in t['local_exponents'])}); the term count alone grows like x²/log x.")
    lines.append("")
    return lines


def write_report(run_dir: Path, table=None) -> dict:
    manifest = runtime.read_json(run_dir / "manifest.json")
    claims_out = runtime.read_json(run_dir / "claims.json")
    md = build_markdown(manifest, claims_out)
    try:
        import figures

        figs = figures.make_figures(run_dir, table)
    except ImportError:  # matplotlib missing: report without figures
        figs = []
    if figs:
        block = ["## Figures", ""]
        for path, caption in figs:
            block += [f"![{caption}]({path})", "", f"*{caption}*", ""]
        md = md.replace("## Verdicts", "\n".join(block) + "\n## Verdicts", 1)
    extra = step4_markdown(run_dir)
    if extra:
        md = md.replace("Full numbers: `claims.json` in this run directory.", extra + "\nFull numbers: `claims.json`, `formulas.json`, `baselines.json`, `nn.json`, `scale.json` in this run directory.")
    (run_dir / "report.md").write_text(md)
    runtime.write_json(
        run_dir / "report.json",
        {
            "run_id": manifest.get("run_id"),
            "seed": manifest.get("seed"),
            "config_path": manifest.get("config_path"),
            "git_root": manifest.get("environment", {}).get("git_root"),
            "git_evolvo": manifest.get("environment", {}).get("git_evolvo"),
            "v11_code_sha256": manifest.get("environment", {}).get("v11_code_sha256"),
            "ndr_version": claims_out.get("ndr_version"),
            "verdicts": {c["id"]: {"title": c["title"], "verdict": c["verdict"], "reproduction": c.get("reproduction", {}).get("status"), "reason": c["reason"]} for c in claims_out["claims"]},
        },
    )
    print(verdict_table(claims_out["claims"]))
    return {"report": "report.md", "report_json": "report.json", "figures": [p for p, _ in figs]}


def _load_table(run_dir: Path):
    import dataset

    info = runtime.read_json(run_dir / "manifest.json").get("stages", {}).get("data", {})
    if info.get("cache") and (run_dir / info["cache"]).is_file():
        return dataset.load_cache(run_dir / info["cache"], info["dataset_key"])
    return None


if __name__ == "__main__":
    # Regenerate figures + report of a finished run from its stored files (no stage reruns):
    #   .venv/bin/python empirical-scripts/11.0/report.py empirical-scripts/11.0/results/<run-id>
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    target = Path(sys.argv[1]).resolve()
    write_report(target, _load_table(target))
