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


def write_report(run_dir: Path) -> dict:
    manifest = runtime.read_json(run_dir / "manifest.json")
    claims_out = runtime.read_json(run_dir / "claims.json")
    md = build_markdown(manifest, claims_out)
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
    return {"report": "report.md", "report_json": "report.json"}
