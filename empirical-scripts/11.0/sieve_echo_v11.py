#!/usr/bin/env python3
"""Sieve Echo Conjecture — version 11 entry point.

One command runs the whole pipeline from a config file and a seed and writes everything under
`empirical-scripts/11.0/results/<run-id>/`:

    .venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json

Stages (ROADMAP.md): data (step 1) -> claims (step 2) -> formulas, nn (steps 3-4) -> report (steps 2/6).
All stages are implemented (`scale` is ROADMAP step 5, `lengths` step 7, `zeros` step 8, `whisky` step 8e, `euler` step 9a, `orders` step 9b, `tworank` step 10, `decode` step 13, `subgroup` step 14, `failures` step 15, `validate` step 16): `formulas` runs the Evolvo GFSL formula experiments (guides x
seeds, S1 adaptive evaluation size, S2 RL guide, warm starts); `nn` runs the held-out baselines,
the §3.1 three-feature retest and the GFSL-NN architecture search.

Run directory:
    manifest.json      code SHAs (root + libs/evolvo), environment, config, seed, NDR version, stages
    split.json         the train/val/test integers (sha256 recorded in the manifest)
    data_summary.json  sampling scheme, counts, excluded pairs, descriptive stats
    claims.json        claim-by-claim statistics and verdicts (C1-C9)
    formulas.json      formula-search runs, guide/S1/S2 comparison; best_formulas.md
    baselines.json     held-out baselines and the three-feature retest
    nn.json            GFSL-NN search results; knowledge/ holds reusable guide and search knowledge
    scale.json         per-n-range claim statistics, stability between the last two ranges, growth exponents
    lengths.json       step 7: C9 against lengths in bases 2..99 and Midy flags, Midy mechanism, residues, lambda(n)
    zeros.json         step 8: exact zero classes of the NDR spectra, C9 with zero features, residues
    whisky.json        step 8e: NDR spectra after SinWhisky resampling (libs/sinwhisky): zero leakage, Midy, omega
    euler.json         step 9a: small-prime membership [ord(q) | ord(b)] vs the C7 residue structure
    orders.json        step 9b: order-only omega classification per decade up to 10^8 (config/orders.json; no NDR data)
    tworank.json       step 10: 2-rank of (Z/nZ)*: ratio estimator of omega and l-adic order profiles vs raw lengths
    decode.json        step 13: exact decoding of omega (the factorization) from repetend lengths and digits
    subgroup.json      step 14: exact membership q in <b> mod n (non-cyclic groups too) and the C7 residue structure
    failures.json      step 15: the decoder's failure set: least failing n per base set, certified from prime order vectors
    validate.json      step 16: the final decoder on integers built from an external prime list (t5k.org)
    prime_orders.json  steps 17–19: exact prime counts by order index and Evolvo formulas
    report.md/.json    verdict table and per-claim summary
    log.txt            stage log
    cache/             disposable dataset cache (git-ignored)
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runtime  # noqa: E402

STAGES = ("data", "claims", "formulas", "nn", "scale", "lengths", "zeros", "whisky", "euler", "orders", "tworank", "decode", "subgroup", "failures", "validate", "prime_orders", "report")
IMPLEMENTED = set(STAGES)
ROADMAP_STEP = {"claims": "step 2", "formulas": "steps 3-4", "nn": "steps 3-4", "scale": "step 5", "lengths": "step 7", "zeros": "step 8", "whisky": "step 8e", "euler": "step 9a", "orders": "step 9b", "tworank": "step 10", "decode": "step 13", "subgroup": "step 14", "failures": "step 15", "validate": "step 16", "prime_orders": "steps 17–19", "report": "step 6"}
DEFAULT_STAGES = "data,claims,formulas,nn,scale,lengths,zeros,whisky,euler,report"


class Run:
    def __init__(self, run_dir: Path, config: dict, config_path: Path, seed: int, resume: bool):
        self.dir = run_dir
        self.config = config
        self.seed = seed
        self.manifest_path = run_dir / "manifest.json"
        self.log_path = run_dir / "log.txt"
        self.table = None
        run_dir.mkdir(parents=True, exist_ok=True)
        config_hash = runtime.sha256_json(config)
        if resume and self.manifest_path.is_file():
            self.manifest = runtime.read_json(self.manifest_path)
            if self.manifest.get("config_sha256") != config_hash or self.manifest.get("seed") != seed:
                raise SystemExit(
                    f"--resume: config or seed differ from the existing run in {run_dir}; use a new --run-id."
                )
            self.manifest.setdefault("resumes", []).append(_now())
        elif self.manifest_path.is_file():
            raise SystemExit(f"{run_dir} already holds a run; pass --resume or choose another --run-id.")
        else:
            self.manifest = {
                "run_id": run_dir.name,
                "created": _now(),
                "config_path": runtime.repo_relative(config_path),
                "config_sha256": config_hash,
                "config": config,
                "seed": seed,
                "stages": {},
            }
        import ndr

        self.manifest.update(
            {
                "argv": sys.argv,
                "environment": runtime.environment_info(),
                "ndr_version": ndr.NDR_VERSION,
            }
        )
        self.save()

    def log(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def save(self) -> None:
        runtime.write_json(self.manifest_path, self.manifest)

    def stage_done(self, name: str) -> bool:
        return self.manifest["stages"].get(name, {}).get("status") == "done"

    def mark(self, name: str, **info) -> None:
        self.manifest["stages"][name] = {"status": "done", "finished": _now(), **info}
        self.save()


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------


def stage_data(run: Run) -> dict:
    import dataset

    data_cfg = run.config["data"]
    split_cfg = run.config["split"]
    key = dataset.dataset_key({"data": data_cfg, "split": split_cfg}, run.seed)
    cache = run.dir / "cache" / f"dataset-{key}.npz"
    reused_from = None
    if not cache.is_file():
        # Same key = same data config, split config, seed, NDR version and dataset format:
        # reuse another run's build instead of recomputing it.
        for other in sorted(run.dir.parent.glob(f"*/cache/dataset-{key}.npz")):
            src_run = other.parents[1]
            if src_run != run.dir and (src_run / "data_summary.json").is_file():
                import shutil

                cache.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(other, cache)
                except OSError:
                    shutil.copy2(other, cache)
                shutil.copy2(src_run / "data_summary.json", run.dir / "data_summary.json")
                reused_from = src_run.name
                break
    table = dataset.load_cache(cache, key)
    if table is not None and (run.dir / "data_summary.json").is_file():
        run.log(f"data: cache hit {cache.name}" + (f" (reused from run {reused_from})" if reused_from else ""))
        summary = runtime.read_json(run.dir / "data_summary.json")
    else:
        run.log("data: building dataset")
        table, summary = dataset.build_dataset(data_cfg, split_cfg, run.seed, log=run.log)
        dataset.save_cache(cache, table, key)
        runtime.write_json(run.dir / "data_summary.json", summary)
    run.table = table
    split = dataset.split_record(table)
    runtime.write_json(run.dir / "split.json", split)
    return {
        "dataset_key": key,
        "cache": str(cache.relative_to(run.dir)),
        "split_file": "split.json",
        "split_sha256": runtime.sha256_json(split),
        "split_counts": {k: len(v) for k, v in split.items()},
        "integers": summary["integers"],
        "pairs": summary["pairs"],
        "build_seconds": summary["build_seconds"],
        "reused_from_run": reused_from,
    }


def load_table(run: Run) -> dict:
    """The dataset of this run: from memory if the data stage ran in this process, else from its cache."""
    import dataset

    if getattr(run, "table", None) is not None:
        return run.table
    info = run.manifest["stages"].get("data", {})
    if info.get("status") != "done":
        raise SystemExit("the data stage has not run for this run id; include `data` in --stages")
    table = dataset.load_cache(run.dir / info["cache"], info["dataset_key"])
    if table is None:
        raise SystemExit(f"dataset cache {info['cache']} is missing or stale; rerun with --stages data")
    run.table = table
    return table


def stage_claims(run: Run) -> dict:
    import claims

    table = load_table(run)
    out = claims.run_claims(table, run.config.get("claims", {}), run.config["data"]["base_sets"], run.seed, runtime.REPO_ROOT, log=run.log)
    runtime.write_json(run.dir / "claims.json", out)
    return {
        "claims_file": "claims.json",
        "claims_sha256": runtime.sha256_file(run.dir / "claims.json"),
        "verdicts": {c["id"]: c["verdict"] for c in out["claims"]},
    }


def _formula_inputs(run: Run) -> list:
    import search_formulas

    return list((run.config.get("formulas") or {}).get("inputs") or search_formulas.DEFAULT_INPUTS)


def stage_formulas(run: Run) -> dict:
    import search_formulas

    table = load_table(run)
    out = search_formulas.run_stage(table, run.config.get("formulas", {}), run.config["data"]["base_sets"], run.seed, run.dir, log=run.log)
    runtime.write_json(run.dir / "formulas.json", out)
    s = out["summary"]
    return {
        "formulas_file": "formulas.json",
        "runs": len(out["runs"]),
        "verdicts": s["verdicts"],
        "best": {t: {"expression": b["expression"], "test": b["test"]} for t, b in s["best_by_validation"].items()},
    }


def stage_nn(run: Run) -> dict:
    import baselines
    import search_nn

    table = load_table(run)
    ncfg = run.config.get("nn", {})
    run.log("  baselines")
    base = baselines.run_baselines(table, ncfg.get("baselines", {}), run.config["data"]["base_sets"], _formula_inputs(run), run.seed, log=run.log)
    runtime.write_json(run.dir / "baselines.json", base)
    run.log("  GFSL-NN search")
    nn = search_nn.run_stage(table, ncfg.get("search", {}), run.config["data"]["base_sets"], _formula_inputs(run), run.seed, run.dir, log=run.log)
    runtime.write_json(run.dir / "nn.json", nn)
    return {
        "baselines_file": "baselines.json",
        "nn_file": "nn.json",
        "nn_test_accuracy": {k: v["test"]["accuracy"] for k, v in nn["encodings"].items()},
    }


def stage_scale(run: Run) -> dict:
    import scale

    table = load_table(run)
    out = scale.run_stage(table, run.config.get("scale", {}), run.config["data"]["base_sets"], _formula_inputs(run), run.seed, log=run.log)
    runtime.write_json(run.dir / "scale.json", out)
    return {"scale_file": "scale.json", "stability": {k: v["stable"] for k, v in out["stability"].items()},
            "growth_exponent": out["growth"]["balanced_all_bins"]}


def stage_lengths(run: Run) -> dict:
    import lengths

    table = load_table(run)
    out = lengths.run_stage(table, run.config.get("lengths", {}), run.config.get("claims", {}), run.config["data"]["base_sets"], _formula_inputs(run),
                            run.seed, run.dir, run.manifest["stages"]["data"]["dataset_key"], log=run.log)
    runtime.write_json(run.dir / "lengths.json", out)
    return {"lengths_file": "lengths.json", "c9_labels": {str(b["bin"]): b.get("label", b.get("skipped")) for b in out["c9_bins"]}}


def stage_zeros(run: Run) -> dict:
    import zeros

    table = load_table(run)
    out = zeros.run_stage(table, run.config.get("zeros", {}), run.config.get("claims", {}), run.config["data"]["base_sets"], _formula_inputs(run),
                          run.seed, run.dir, run.manifest["stages"]["data"]["dataset_key"], log=run.log)
    runtime.write_json(run.dir / "zeros.json", out)
    return {"zeros_file": "zeros.json", "exact_check": f"{out['exact_check']['agree']}/{out['exact_check']['pairs']}"}


def stage_whisky(run: Run) -> dict:
    import whisky_spectra

    table = load_table(run)
    out = whisky_spectra.run_stage(table, run.config.get("whisky", {}), run.config["data"]["base_sets"], run.seed, log=run.log)
    runtime.write_json(run.dir / "whisky.json", out)
    return {"whisky_file": "whisky.json", "pairs": out["pairs"]}


def stage_euler(run: Run) -> dict:
    import euler

    table = load_table(run)
    out = euler.run_stage(table, run.config.get("euler", {}), run.config.get("claims", {}), run.config["data"]["base_sets"], run.seed, run.dir,
                          run.manifest["stages"]["data"]["dataset_key"], log=run.log)
    runtime.write_json(run.dir / "euler.json", out)
    return {"euler_file": "euler.json"}


def stage_orders(run: Run) -> dict:
    import orders_scale

    out = orders_scale.run_stage(run.config.get("orders", {}), run.seed, log=run.log)
    runtime.write_json(run.dir / "orders.json", out)
    return {"orders_file": "orders.json", "integers": out["integers"]}


def stage_tworank(run: Run) -> dict:
    import tworank

    out = tworank.run_stage(run.config.get("orders", {}), run.config.get("tworank", {}), run.seed, log=run.log)
    runtime.write_json(run.dir / "tworank.json", out)
    return {"tworank_file": "tworank.json", "integers": out["integers"]}


def stage_decode(run: Run) -> dict:
    import decode

    out = decode.run_stage(run.config.get("decode", {}), run.seed, log=run.log)
    runtime.write_json(run.dir / "decode.json", out)
    return {"decode_file": "decode.json", "integers": out["integers"]}


def stage_subgroup(run: Run) -> dict:
    import subgroup

    table = load_table(run)
    out = subgroup.run_stage(table, run.config.get("subgroup", {}), run.config.get("claims", {}), run.config["data"]["base_sets"], run.seed, run.dir,
                             run.manifest["stages"]["data"]["dataset_key"], log=run.log)
    runtime.write_json(run.dir / "subgroup.json", out)
    return {"subgroup_file": "subgroup.json"}


def stage_failures(run: Run) -> dict:
    import failures

    out = failures.run_stage(run.config.get("failures", {}), run.seed, log=run.log)
    runtime.write_json(run.dir / "failures.json", out)
    return {"failures_file": "failures.json"}


def stage_validate(run: Run) -> dict:
    import validate

    out = validate.run_stage(run.config.get("validate", {}), run.seed, log=run.log)
    runtime.write_json(run.dir / "validate.json", out)
    return {"validate_file": "validate.json"}


def stage_prime_orders(run: Run) -> dict:
    import prime_orders

    out = prime_orders.run_stage(run.config["prime_orders"], run.seed, run.dir, log=run.log)
    runtime.write_json(run.dir / "prime_orders.json", out)
    return {"prime_orders_file": "prime_orders.json", "cutoffs": out["cutoffs"]}


def stage_report(run: Run) -> dict:
    import report

    if not any((run.dir / f).is_file() for f in ("claims.json", "orders.json", "tworank.json", "decode.json", "subgroup.json", "failures.json", "validate.json", "prime_orders.json", "euler.json", "zeros.json", "lengths.json", "scale.json")):
        raise SystemExit("report needs the output of at least one analysis stage")
    try:
        table = load_table(run)
    except SystemExit:
        table = None  # figures that need the dataset are skipped
    return report.write_report(run.dir, table)


STAGE_FUNCS = {"data": stage_data, "claims": stage_claims, "formulas": stage_formulas, "nn": stage_nn, "scale": stage_scale, "lengths": stage_lengths, "zeros": stage_zeros, "whisky": stage_whisky, "euler": stage_euler, "orders": stage_orders, "tworank": stage_tworank, "decode": stage_decode, "subgroup": stage_subgroup, "failures": stage_failures, "validate": stage_validate, "prime_orders": stage_prime_orders, "report": stage_report}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="JSON config (see config/quick.json)")
    p.add_argument("--seed", type=int, default=None, help="overrides config 'seed'")
    p.add_argument("--run-id", default=None, help="run directory name (default: <config name>-s<seed>-<time>)")
    p.add_argument("--resume", action="store_true", help="continue an existing run, skipping finished stages")
    p.add_argument("--stages", default=DEFAULT_STAGES, help=f"comma list from {','.join(STAGES)} (default: {DEFAULT_STAGES})")
    p.add_argument("--results-dir", type=Path, default=runtime.RESULTS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {STAGES}")
    missing = [s for s in stages if s not in IMPLEMENTED]
    if missing:
        raise SystemExit(
            "stage(s) not implemented yet: "
            + ", ".join(f"{s} (ROADMAP {ROADMAP_STEP[s]})" for s in missing)
        )

    runtime.bootstrap_evolvo()  # fail loudly now if the submodule is missing (D2)

    config_path = args.config.resolve()
    config = runtime.read_json(config_path)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    config = copy.deepcopy(config)
    config["seed"] = seed
    run_id = args.run_id or f"{config.get('name', config_path.stem)}-s{seed}-{_dt.datetime.now():%Y%m%d-%H%M%S}"
    run = Run(args.results_dir / run_id, config, config_path, seed, args.resume)
    run.log(f"run {run_id} seed={seed} stages={stages} dir={runtime.repo_relative(run.dir)}")

    for name in STAGES:
        if name not in stages:
            continue
        if args.resume and run.stage_done(name):
            run.log(f"{name}: already done, skipping")
            continue
        t0 = time.time()
        info = STAGE_FUNCS[name](run)
        run.mark(name, seconds=round(time.time() - t0, 2), **info)
        run.log(f"{name}: done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
