# Sieve Echo Conjecture — AI Agent Reference

Fast-access operational reference for AI agents working in this repository. Read it completely before changing anything, and keep it true for the checked-out revision (see [Maintenance rules](#collaboration-and-maintenance-rules)).

**What this project is.** An empirical research repository by Riccardo Cecchini that tests the *Sieve Echo Conjecture*: the claim that the prime-factorization complexity of an integer `n` (mainly ω(n), the number of distinct prime factors) is encoded in the spectral entropy of the repeating digits of `1/n` written in several bases, after normalizing digits to `[0,1]` (NDR, see [Vocabulary](#vocabulary)). The work is a sequence of numbered script versions (`empirical-scripts/3.1` … `10.0`) plus papers in `studies/`. Maturity: exploratory research code. No package, no test suite, no CI.

**What it is not.** It is not a proof, and no script so far is a rigorous demonstration. It is not the Evolvo library: Evolvo is a separate repository vendored as the git submodule [`libs/evolvo`](libs/evolvo) with its own [`AGENTS.md`](libs/evolvo/AGENTS.md). The copies named `evolvo*.py` inside old version folders are historical forks of that library, not the library to use.

**Current goal.** Build **version 11** in [`empirical-scripts/11.0/`](empirical-scripts/11.0) (every stage exists: dataset, claim verdicts, Evolvo formula and GFSL-NN search, scale analysis, step-7 length/Midy analysis, report): one reproducible, leakage-free execution script that tests every claim of the current paper [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md) and uses the submodule [`libs/evolvo`](libs/evolvo) for all evolutionary search. The step plan lives in [`ROADMAP.md`](ROADMAP.md).

## Read This First

Source-of-truth order. When two sources disagree, the higher one wins, and the agent fixes or records the mismatch in the same change.

1. **This file's [Directives](#standing-directives-from-the-project-owner)** — standing instructions from the project owner.
2. [`ROADMAP.md`](ROADMAP.md) — the v11 implementation plan and its live status. It governs *what to do next*; it never proves that something exists.
3. Executable code: [`libs/evolvo/src/evolvo/`](libs/evolvo/src/evolvo) (the library) and [`empirical-scripts/11.0/`](empirical-scripts/11.0). Evolvo's own rules are in [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) and its specs in [`libs/evolvo/papers/`](libs/evolvo/papers).
4. [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md) — the current paper. It states the claims that v11 must test. Its numbers are **claims to re-verify**, not facts (see [Known gaps](#known-gaps)).
5. Stored results under [`empirical-scripts/results/`](empirical-scripts/results) and `empirical-scripts/<ver>/results/` — evidence of what past runs produced, with the methodological defects listed below.
6. Older papers ([`studies/7/`](studies/7), [`studies/8/future.md`](studies/8/future.md)), the notes in [`studies/claude/`](studies/claude), and commit messages — history only. Commit messages in this repo carry no technical information.

## Standing Directives from the Project Owner

These are mandatory for every task from v11 onward.

- **D1 — Version 11 lives in `empirical-scripts/11.0/`.** Create the folder; do not edit older version folders except to fix a link. Its goal is to complete the "demonstration" (empirical implementation) of [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md), reusing the best ideas and results of the previous scripts.
- **D2 — Use the Evolvo library from the submodule.** All formula search and neural-architecture search MUST import `evolvo` from [`libs/evolvo/src`](libs/evolvo/src) (upstream: `https://github.com/Geckos-Ink/evolvo`). Do NOT copy `evolvo.py`, `evolvo_model.py`, `evolvo_nn.py`, or `gold/evolvo.py` into `11.0/`, and do NOT import the historical copies in older version folders.
- **D3 — Fix Evolvo in Evolvo.** When v11 needs a feature or bug fix in Evolvo, implement it inside [`libs/evolvo`](libs/evolvo), following [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) (update its `AGENTS.md` and `README.md`, add a test under [`libs/evolvo/tests/`](libs/evolvo/tests)), commit it on the submodule's `main` branch, then commit the updated submodule pointer in this repository. Do not work around an Evolvo defect inside the v11 script when the fix belongs in the library.
- **D4 — Keep [`ROADMAP.md`](ROADMAP.md) live.** Update step status while implementing: mark steps done only after verification, add newly discovered fixes or follow-ups, and record results with the run that produced them.
- **D5 — Keep this file live.** Follow the [AGENTS.md bootstrap protocol](#collaboration-and-maintenance-rules): every change that adds, moves, or changes a file, command, contract, or result updates the matching section here in the same change.
- **D6 — Evaluation size grows with quality.** Each genome is scored on 5 000 integers at first, rising step by step to at most 100 000 as the average of the best scores improves. Never compare fitness values taken at different sizes. Spec: [`ROADMAP.md` S1](ROADMAP.md#s1--adaptive-evaluation-size-5-000--100-000).
- **D7 — Reinforcement learning through PyTorch is required.** A torch guide learns which genome edits work over time (weights saved and reused across runs) and lowers exploration as the best scores improve, which reduces the spread of results. Rewards are relative to the average best scores. PyTorch is therefore a hard dependency of v11. Spec: [`ROADMAP.md` S2](ROADMAP.md#s2--torch-reinforcement-learning-guide).

## Collaboration and Maintenance Rules

- This handbook follows the AGENTS.md bootstrap protocol kept by the owner outside this repo (`ai-agents-bootstrap/AGENTS.bootstrap.md`). Required behaviour: present tense for shipped facts, explicit `Planned` labels for the rest, a linked `###` subsection per meaningful file, no secrets or personal data, and no chronological changelog.
- **Task start:** read this file and [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) if the task touches Evolvo; run `git status` in both the root and `libs/evolvo`; read the relevant [`ROADMAP.md`](ROADMAP.md) step.
- **Task end:** update this file, [`ROADMAP.md`](ROADMAP.md) and, when Evolvo changed, Evolvo's `AGENTS.md`/`README.md`. Report what was run and what was not.
- **Submodule commits:** commit Evolvo changes inside `libs/evolvo` on `main` first, then `git add libs/evolvo` in the root and commit the pointer. A pointer to an unpushed Evolvo commit breaks fresh clones; push both repositories only when the owner asks, and say so in the handoff when they are not pushed.
- **Results are data, not source.** v11 runs write under `empirical-scripts/11.0/results/<run-id>/`; its `cache/` is git-ignored. Commit only small summaries (JSON/Markdown/PNG) that the paper or roadmap cites; never commit caches, pickles of large datasets, or per-generation dumps like the 300+ files in [`empirical-scripts/results/3-Sept-2025/algorithms_realtime/`](empirical-scripts/results/3-Sept-2025/algorithms_realtime).
- **Paper claims need a run.** Any number written into a paper or this file must name the script, config, and seed that produced it.
- Preserve unrelated user changes; never clean the tree destructively. `.DS_Store` files exist untracked-or-tracked in places; leave them alone.

## Vocabulary

- **ω(n)** — number of distinct prime factors. **Ω(n)** — with multiplicity. **τ(n)** — number of divisors. **φ(n)** — Euler totient. **λ(n)** — Carmichael function.
- **Repetend** — the repeating block of `1/n` in base `b`. Defined only when `gcd(n, b) = 1`; then it is purely periodic with length `L = ord_n(b)`, the multiplicative order of `b` modulo `n`.
- **NDR** (Normalized Digit Representation) — the repetend digits divided by the base. The paper (Definition 1.1) uses `d/b`; v10 used `d/(b-1)`. See [pitfall](#ndr-normalization-and-entropy-definition-drift).
- **H_NDR** (NDR entropy) — Shannon entropy of the normalized power spectrum of the NDR sequence (paper Definition 1.2). v11 names: `H_paper` (all L bins, DC kept), `H_nodc` (DC dropped), `Hn_*` (divided by log length), `H_v8`/`H_v10` (historical definitions, reproduction only). Definitions in [`ndr.py`](empirical-scripts/11.0/ndr.py), tag `ndr-v11.1`.
- **Base invariance** — coefficient of variation (CV = std/mean) of H_NDR across bases below 0.1.
- **GFSL** — Genetic Fixed Structure Language, the slot-encoded genome language implemented by Evolvo. **GFSL-NN** — its neural-architecture variant (`evolvo.nnarch`).

## Essential Project Principles

### A claim is tested only against a null model and on held-out data

- Every reported fit, correlation, or accuracy MUST be measured on data not used to fit it (train/validation/test split by `n`, fixed seed), and compared with a baseline that uses only trivial covariates (`log n`, `log L`) and with a permutation/shuffled-label null.
- Training-set accuracy MUST NOT be reported as accuracy. v8's "98.8%" is training-set accuracy ([pitfall](#training-set-accuracy-reported-as-model-accuracy)).

### Features fed to a predictor of ω(n) MUST NOT encode factorization

- Inputs for predicting ω(n) are derived from digit/NDR sequences, `n`, and repetend lengths only. `tau`, `phi`, `is_prime`, `is_prime_power`, `omega`, `Omega`, `mobius`, `factorint` output, or anything computed from them, are **forbidden inputs**; they are allowed only as targets or as stratification labels.
- Exception, stated explicitly: an analysis whose stated purpose is "how much does feature X add beyond factorization-derived Y" MAY use Y as a control variable, and MUST label it so.

### Confounders are controlled, not ignored

- H_NDR is bounded by `log(L/2)` and grows with repetend length `L = ord_n(b)`; `L` itself depends strongly on factorization (λ(n) divides φ(n) and is much smaller for composites). Every claim about H_NDR vs ω(n) MUST also be reported conditional on `log L` (partial correlation or residual model) and on `log n`.

### One canonical definition, versioned

- NDR normalization, the spectrum (power vs amplitude, DC bin kept or dropped, one-sided length) and the entropy base are a single function in v11 with a version tag written into every result file. Changing it changes the version tag.

### Reproducible by one command

- v11 runs from a single entry point with a config file and seed, writes a manifest (git SHAs of root and `libs/evolvo`, Python and library versions, config, seed), and can resume.

## Critical Implementation Contracts

- **Repetend only when coprime:** compute the repetend only when `gcd(n, b) == 1`; record the pair as excluded otherwise. Do not silently drop `n` from all bases.
- **Exact arithmetic for orders:** compute `ord_n(b)` from the factorization of λ(n) (exact integers), not by simulating long division up to `n` steps; the digit sequence is then generated with exactly `L` steps. Long division up to `n` steps for `n ≈ 2·10^5` across 16 bases is the main runtime cost in v9/v10.
- **Splits by `n`, fixed before search:** the train/validation/test partition of integers is created once per run from the seed and saved in the manifest; Evolvo fitness uses train (or a train minibatch) only; model selection uses validation; the test split is read once for the final report.
- **Evolvo import path:** v11 adds `<repo>/libs/evolvo/src` to `sys.path` (or uses `PYTHONPATH`) and imports `evolvo`; it MUST fail loudly if the submodule is not initialized, rather than falling back to a vendored copy.
- **Fitness is bounded and documented:** v10 fitness `(1/(1+MSE) + 0.01·#ops) · length_penalty` exceeds 1 and rewards operation variety; v11 fitness is validation-free train loss plus an explicit parsimony term, documented in the result manifest.

## Architecture and Data/Control Flow

v11 flow (every stage exists — see [`ROADMAP.md`](ROADMAP.md); `scale` and `lengths` run after the searches on ω-balanced n-ranges):

`11.0` entry script + config → **dataset builder** (integers, factorization labels, coprime bases, `ord_n(b)`, digit sequences, NDR, spectra, per-base and cross-base features, cached to disk) → **split** by `n` → **statistical claim tests** (paper §2 and §4 claims, each with baseline and null) → **Evolvo symbolic search** (`GFSLEvolver` formulas on allowed features) and **Evolvo GFSL-NN search** (`evolvo.nnarch.ArchitectureSearch` + `TorchTask`) → **held-out evaluation** → **report** (JSON + Markdown + figures, claim-by-claim verdict table).

Evolvo boundary: the v11 script owns data, features, splits, and verdicts; [`libs/evolvo`](libs/evolvo) owns genomes, validity, execution, evolution, and architecture search. Domain logic (number theory) MUST NOT be pushed into Evolvo; generic capabilities (batched execution, input declaration, custom ops) belong there.

## Standing Evolvo Facts Relevant to v11

Verified against `libs/evolvo` after ROADMAP step 3 (the commit the root repo points to):

- Public API is lazy-loaded from [`libs/evolvo/src/evolvo/__init__.py`](libs/evolvo/src/evolvo/__init__.py) (`_LAZY` table). The GFSL runtime needs NumPy; `evolvo.nnarch` is pure Python except `build.py`/`train.py`; `GFSLSupervisedGuide` and `GFSLPolicyGuide` need PyTorch.
- **Formula search setup:** `GFSLEvolver(population_size, supervised_guide=None, …)`; `initialize_population(genome_type, initial_instructions, inputs=k or {DataType.DECIMAL: k}, outputs=[(DataType.DECIMAL, k)], profile=InstructionProfile.formula())`. Inputs `d$0..d$k-1` are then readable by every genome (initial, mutated, crossed over, immigrant); without `inputs=` random genomes cannot read their inputs.
- **Batched evaluation:** `GFSLExecutor(compute_backend="cpu").execute_batch(genome, {"d$0": array, …}, track_activity=False)` → `{"d$k": float64 array}`; element-wise identical to `execute`; NumPy-vectorized unless the genome uses `CALL`/list ops (then row-by-row; `executor.last_batch_mode`). Domain ops: `register_custom_operation(name, DataType.DECIMAL, fn, arity=…, batch_function=vec_fn)`.
- **Evaluation-size epochs (S1):** `evolve(generations, evaluator, …, fitness_epoch=lambda: current_N)` re-scores the whole population when `N` changes; `genome.fitness_epoch`, `evolver.fitness_epoch_changes`, `evolver.evaluations`.
- **RL guide (S2):** `GFSLPolicyGuide(q_target=…, top_fraction=0.1, device="auto", seed=…)` passed as `supervised_guide=`; `save(path)` / `GFSLPolicyGuide.load(path, **overrides)`; `history` holds per-generation `q`, temperature, mean reward, loss.
- `evolvo.nnarch`: `ArchSpace(inputs, n_out, max_rows, max_params, task=…)`, `TorchTask((x_tr, y_tr), (x_val, y_val), lr)` for regression, `TorchTask.classification(...)` for class targets (use `task={"loss": "ce"}`), `ArchitectureSearch(space, task, seed, knowledge)`, `.run(generations, patience)`, `Knowledge.load/save` — see [`libs/evolvo/README.md`](libs/evolvo/README.md).
- Tests: from `libs/evolvo`, `../../.venv/bin/python -m unittest discover -s tests` → 33 tests (GFSL core, policy guide, GFSL-NN); with a NumPy-less `python3` the NumPy/torch modules skip.

## Linked Source Tree and File Reference

### [`studies/9/sieve-echo-conjecture-9.md`](studies/9/sieve-echo-conjecture-9.md)

Revision 9 draft (step 6, awaiting owner review): the v11 findings with every number tied to a run (`full-s0-step4`, `scale-s0`, `quick-s0-step4`, `quick-s0-step6`), the v8 reproduction table, Observations 4.1–6.2, Theorem 1.3 (λ(n) ≤ φ(n)/2^{ω−1} for odd n), the reformulated open conjecture, reproduction commands, embedded figures from the run directories. Byline intentionally empty (no personal data in new files). Change numbers here only together with the run that produced them.

### [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md)

Previous paper (revised draft, August 2025); superseded for empirical claims by `studies/9` (draft). Defines NDR (Def. 1.1), H_NDR (Def. 1.2), the Sieve Echo Law (Conj. 1.3), and reports Theorems 2.2–4.3. Change it only to replace claims with v11-verified results, citing the run.

- **Claims v11 must test:** §2.2 fit `H = -0.468·log ω + 1.917`, R² = 0.234 (the code fitted `log(ω+1)`); §2.3 prime-power correlation r = -0.843 (actually corr(ω, is_prime_power) from a v7.0 log — no entropy involved); §2.4 base invariance 6% (first 100 integers only); §3.1 three-feature 98.8% accuracy (v7.0 run, training set); §4.1 correlation with `n^-s`; §4.2 `H·log n` vs PNT r = 0.946; §4.3 residue-class variance mod 6/30/210; §5 table vs theoretical α = -1.599, β = 4.933.
- **Common mistakes:** the paper contradicts itself on direction — §1.3 says primes give "pure, low-entropy" patterns, but α < 0 and the v10 data show primes have the *highest* H_NDR (longest repetends). §7.1 item 2 says "higher complexity correlates with higher entropy", which contradicts α < 0. Do not propagate either sentence; v11 must state the measured direction. "Theorem" labels on empirical fits are not theorems.

### [`studies/8/future.md`](studies/8/future.md)

Earlier rewrite of the same paper with the same numbers. Historical; superseded by `sieve-echo-conjecture-8.md`.

### [`studies/7/`](studies/7)

Theta-era papers (`shieve-echo-theta-7.md/.txt`, `sieve-echo-deeper-links.md`, `sieve-echo-related-dev-1.md`). Source of the theoretical constants α ≈ -1/φ², β = 5 - 1/15 and the "three features (kurtosis, length, n)" claim. History only.

### [`studies/MAL-LLM.md`](studies/MAL-LLM.md)

The owner's Mathematical Annotation Language proposal. The `level: … connects_to: …` blocks in the papers use it. Not part of the computation.

### [`studies/claude/`](studies/claude)

`common_mistakes.md` and `attention_notes.md`: notes from earlier AI sessions. Durable lessons: keep Evolvo central (now D2), implement the owner's *relative* thresholds literally (v10 `save_threshold_ratio` = fraction of the top-performer mean, not an absolute value), and carry config values into every function that should use them.

### [`empirical-scripts/11.0/sieve_echo_v11.py`](empirical-scripts/11.0/sieve_echo_v11.py)

v11 entry point. Parses `--config --seed --run-id --resume --stages --results-dir`, imports Evolvo through `runtime.bootstrap_evolvo()` before any stage, creates `results/<run-id>/`, and runs the requested stages in order (`data, claims, formulas, nn, scale, lengths, report`).

- **Key parts:** `Run` (manifest, log, stage status; refuses to reuse a run dir without `--resume`, refuses `--resume` with a different config or seed), `stage_data` (build, load, or reuse the cached dataset; writes `data_summary.json` and `split.json`), `load_table`, `stage_claims` (`claims.json`), `stage_formulas` (`formulas.json`, `best_formulas.md`, `knowledge/policy_<target>.pt`), `stage_nn` (`baselines.json`, `nn.json`, `knowledge/nn_<input>.json`), `stage_scale` (`scale.json`), `stage_lengths` (`lengths.json`, `cache/lengths-<key>.npz`), `stage_report` (`report.md`, `report.json`, prints the verdict table), `IMPLEMENTED` = all seven stages.
- **Contract:** default `--stages` is `data,claims,formulas,nn,scale,lengths,report`. Add each new stage to `STAGE_FUNCS` and `IMPLEMENTED`. When no local cache exists, `stage_data` hard-links (or copies) `results/*/cache/dataset-<key>.npz` and its `data_summary.json` from another run with the same key and records `reused_from_run`; bump `dataset.DATASET_FORMAT` or `ndr.NDR_VERSION` whenever code changes the dataset content, or stale caches will be reused.

### [`empirical-scripts/11.0/runtime.py`](empirical-scripts/11.0/runtime.py)

Non-scientific helpers: repo paths, `bootstrap_evolvo()` (submodule only; raises `EvolvoMissingError`), `git_info()` (SHA, dirty flag and changed paths, ignoring `11.0/results/` and `.DS_Store`), `code_sha256()` over the v11 `.py` files, `environment_info()`, `repo_relative()` (keeps home-directory paths out of results), NaN-safe atomic `write_json`, SHA-256 helpers.

### [`empirical-scripts/11.0/ndr.py`](empirical-scripts/11.0/ndr.py)

The canonical, versioned definitions (`NDR_VERSION = "ndr-v11.1"`; the module docstring is the spec). Smallest-prime-factor sieve, `factorize`, `arithmetic_labels` (ω, Ω, τ, φ, λ, μ, is_prime, is_prime_power, squarefree), `multiplicative_order` from λ(n), `repetend_digits` by vector doubling, `ndr = d/b`, `spectral_entropy(x, variant)` with `paper | nodc | v8 | v10`, `pair_features(n, b, L)` returning `PAIR_FEATURES`, and `long_division_repetend` (O(n) reference for tests only).

- **Common mistakes:** changing a definition without bumping `NDR_VERSION` (old caches would then be silently reused — the cache key includes the tag); computing digits with long division in production code; feeding `arithmetic_labels` output to a predictor.

### [`empirical-scripts/11.0/dataset.py`](empirical-scripts/11.0/dataset.py)

Data stage: `sample_integers` (full range + seeded per-ω stratified samples from one range or a list of ranges; range 0 keeps the single-range RNG stream so old configs reproduce), `base_set_members` (seeded sub-sample for sets with `sample`), `compute_pairs` (process pool, chunks balanced by λ(n)·#bases), `aggregate` (mean/std/min/max/CV per base set, NaN-aware), `split_by_n` (stratified by ω, seeded), `allowed_input_columns`, `.npz` cache keyed by `dataset_key`. Column families: `row.*` (inputs), `label.*` and `target_side.*` (never inputs), `pair.*` (long per-(n, b) table), `split` (0/1/2 = train/val/test).

### [`empirical-scripts/11.0/claims.py`](empirical-scripts/11.0/claims.py)

ROADMAP step 2: one function per claim (`c1_law` … `c9_beyond_L`), driven by `run_claims(table, claims_cfg, base_sets, seed, repo_root)`. Each returns `reproduction` (the paper's statistic re-computed the v8.2 way: `H_v8`, n ∈ [2, 9999], in-sample, and for C4/C5 the first 100 integers of the stored v8.2 data order read from the v8 pickle), `tests` (train-fitted, test-reported, `log L`/`log n` controls, percentile bootstrap, permutation null), `verdict` and `reason`.

- **Key parts:** helpers `lstsq`, `r2`, `partial_corr`, `bootstrap`, `perm_p`, `eta_squared`; `Ctx` (table access, masks, seeded RNG stream, v8 order loader with a NumPy-only unpickler); `_per_base_logL` (per-base log L matrix from the pair table); `_length_arithmetic` (lcm of the per-base lengths, its relation to n - 1: functions of n and L only); C9 compares gradient-boosting classifiers of ω (baseline n + per-base L + length arithmetic; + entropy columns; + all NDR columns) with paired bootstrap on test rows, a majority baseline and shuffled-label nulls.
- **Contract:** settings come from the config's `claims` block over `DEFAULTS`; `PAPER` holds the stored v8.2/v7.0 values. Factorization labels are used only as targets, groupings, or labelled controls (C7 ω dummies).
- **C7 helpers:** `c7_design(ctx, var, per_base_L, coprime)` → (H, controls, train, test) and `residue_excess(ctx, H, cols, train, test)` → per-modulus η² excess; `lengths.residues` reuses them with extra Midy controls.
- **Common mistakes:** controlling for the *mean* log L only (residue classes change which per-base orders are short — C7 needs per-base log L); a C9 baseline without length arithmetic (trees cannot compute an lcm, so NDR features then look informative); reading `row.wide.*` as if every n had it (sub-sampled).

### [`empirical-scripts/11.0/report.py`](empirical-scripts/11.0/report.py)

`write_report(run_dir, table=None)` reads `manifest.json` + `claims.json` (+ `formulas.json`, `baselines.json`, `nn.json`, `scale.json` when present), builds the figures (`figures.make_figures`), writes `report.md` (figures first, then (run provenance, verdict table, one section per claim, step-4 guide/variant table with S1/S2 checks and Mann–Whitney p-values, best formulas, held-out results table, three-feature retest, step-5 scale table) and `report.json`, and prints the verdict table. `python empirical-scripts/11.0/report.py <run_dir>` regenerates figures + report of a finished run (loads the dataset cache when present), without rerunning stages or touching the manifest.

### [`empirical-scripts/11.0/figures.py`](empirical-scripts/11.0/figures.py)

Static PNG figures into `<run>/figures/`: `entropy_vs_length` (needs the dataset), `heldout_omega` (baselines/nn/formulas), `formula_guides_omega` (formulas), `c9_gain_by_decade`, `invariance_by_decade`, `growth_exponents` (scale), `c9_gain_by_baseline`, `midy_entropy_gap` (lengths). Palette: categorical blue/orange/aqua (validated with the dataviz validator: adjacent CVD ΔE ≥ 9.2; aqua < 3:1 contrast, so lines are always direct-labelled), single-hue blue ramp for ω, legend plus direct labels for multi-series charts.

### [`empirical-scripts/11.0/features.py`](empirical-scripts/11.0/features.py)

Predictor inputs shared by claims C9, baselines, formulas and GFSL-NN: `per_base` (rows × bases matrix of a pair feature), `length_arithmetic` (log lcm of the L_b, log n − log lcm, lcm | n−1, log gcd(lcm, n−1), share of L_b dividing n−1 — functions of n and L only) and `length_arithmetic_from_orders` (the same from a rows × bases order matrix), `column`/`build_matrix` (named allowed inputs; raise on `label.*`, `target_side.*`, `pair.*`), `Standardizer` (train z-score, NaN → train median), `token_tensor` (per-base tokens for the GFSL-NN set input).

### [`empirical-scripts/11.0/search_formulas.py`](empirical-scripts/11.0/search_formulas.py)

ROADMAP step 4 / S1 / S2. `prepare` (12 standardized inputs, train permutation, val/test arrays per target), `Sizer` (S1: monotone N driven by the top-k mean toward `q_target` and by the 2-SE spread trigger, capped by train size), `run_one` (one target × guide × seed × variant in a worker: `GFSLEvolver` with `inputs=`, `outputs=`, `InstructionProfile.formula(allow_custom=False)`, fitness `1/(1+NMSE) − parsimony·size` via `execute_batch` on the first N permuted train rows, `fitness_epoch=lambda: N`, v10 relative saving + safety net, validation selection, one test read), `formula_expression` (readable formula over named inputs), `summarize` (guide/variant table, S1/S2 checks with Mann–Whitney p), `run_stage` (phase A: guides × seeds adaptive, fixed-N ablation, one knowledge run per target; phase B: warm policy runs loading it).

- **Common mistakes:** reading S1/S2 "met" without the p-values — at 5 seeds the differences are within seed noise; a `q_target` no run reaches makes "evaluations to target" undefined.

### [`empirical-scripts/11.0/baselines.py`](empirical-scripts/11.0/baselines.py)

Held-out baselines for ω classes (min(ω, 5)) and is_prime: majority, logistic(log n, mean log L), gradient boosting on four feature sets (`log n + mean log L`, `n + L arithmetic`, `formula inputs`, `all allowed`); `three_feature` = §3.1 retest (std/min H, + log n, and a labelled `is_prime_power` leakage demo scored on train and test).

### [`empirical-scripts/11.0/search_nn.py`](empirical-scripts/11.0/search_nn.py)

GFSL-NN search for ω classes with `TorchTask.classification`: `vec` input (the `all allowed` features) and `set` input (one token per base + a vector of log n and length arithmetic). Stratified train subsample, validation for stopping/selection, one test read; search knowledge saved to `knowledge/nn_<input>.json` and loadable via `nn.search.knowledge_in`. Raw digit sequences are not used (variable length up to ~10⁶; Evolvo sets are fixed-size).

### [`empirical-scripts/11.0/scale.py`](empirical-scripts/11.0/scale.py)

ROADMAP step 5. `run_stage` splits integers into n-ranges (`scale.bins`), takes an ω-balanced sample per range (`_balanced`), and `bin_statistics` recomputes C1 (ΔR², partial r), C2 (Spearman, primes − composites), C3 (partial r), C4 (CV < 0.1 rates for H_paper, Hn_nodc, log L; v8 and wide), C9 (gradient boosting: baseline → + entropy → + all NDR, paired-bootstrap Δ) on the range's train/test rows; `stability` compares labels of the last two ranges. `growth` fits H ~ n^a (v8-style on [2, 9999]; ω-balanced with ω fixed effects; with log L control; per ω). `T_exponent` computes the theta-7 double sum T(x) = Σ_p Σ_n log ord_n(p) and its log-log slope.

### [`empirical-scripts/11.0/lengths.py`](empirical-scripts/11.0/lengths.py)

ROADMAP step 7: what the residual entropy signal is. `orders_and_midy` (rows × bases matrices of ord_n(b), exact from λ(n), and the Midy flag `b^(L/2) ≡ −1 (mod n)`, process pool; cached as `cache/lengths-<key>.npz` keyed by dataset key, bases and `LENGTHS_VERSION`), `length_block`/`midy_block`/`summary_columns` (functions of n, orders and flags only), `c9_bins` (7a: per n-range C9 with baselines L19, L19 + Midy19, L98, L98 + Midy98 on the scale stage's ω-balanced samples — L19 reproduces `scale.json` exactly), `ablation` (NDR feature families over L19 and L19 + Midy19), `pair_mechanism` (7b: pair-level H ~ log L, log b, log n ± Midy), `residues` (C7 test ± per-base Midy controls via `claims.c7_design`/`claims.residue_excess`), `lambda_analysis` (7c: lcm-of-orders = λ coverage, Theorem 1.3 check, length-only bound ω ≤ ⌈log₂(n/lcm)⌉, ω classifiers from log λ [labelled control] and from length-only estimates), `formula_search` (Evolvo formulas for ω: step-4 inputs vs `FORMULA_INPUTS_98`, same table and budget), `run_stage`.

- **Contract:** `lengths.bases` must contain the v8/v10/primes31 bases (raises otherwise). λ(n) and φ(n) are read only as diagnostics and as an explicitly labelled control.
- **Common mistakes:** treating the Midy flag as a factorization label (it is a function of (n, b), visible in the digit sequence); comparing `lengths.json` gains with `claims.json` C9 (different samples: claims uses the whole split, lengths the ω-balanced ranges).

### [`empirical-scripts/11.0/config/`](empirical-scripts/11.0/config)

`quick.json` (n ∈ [2, 9999] + 100 per ω from [10⁴, 10⁵], 4 base sets; builds in ~8 s), `full.json` (n ∈ [2, 150 000] + 2 000 per ω up to 10⁶; data ~20 min) and `scale.json` (step 5: n ∈ [2, 9999] + 3 000 per ω in [10⁴, 10⁵) and in [10⁵, 10⁶], wide set on 6 000; run with `--stages data,claims,scale,report`). Keys: `seed`, `data.{n_min, n_max, stratified, base_sets, workers}`, `split.{train, val, test, stratify}`, `claims.{primary_set, entropies, bootstrap, permutations, effect_threshold, cv_threshold, invariance_sets, invariance_variants, moduli, zeta_s, classifier}`. `lengths.{bases, bins, per_omega_cap, min_per_omega, bootstrap, max_omega_class, gbm, ablation, pair_bins, formulas.{guides, seeds, population, generations, …}, workers}` (step 7; `formulas: null` skips the Evolvo part). `formulas.{targets, guides, seeds, population, generations, parsimony, eval_size_min/max/step, q_target (per target), top_fraction, save_threshold_ratio, save_top_per_gen, safety_net, ablation_guide, fixed_size, warm_start, knowledge_seed, inputs, workers}`; `nn.baselines.{max_train_rows, gbm}`, `nn.search.{population, generations, patience, max_train_rows, max_val_rows, hard_max_epochs, max_rows, max_params, knowledge_in}`. Quick: population 100 × 30 generations, N step 500 (train has 7 366 rows); full: 200 × 60, N step 5 000 up to 100 000. A base set is a list, `"a..b"`, or `{"bases": …, "sample": k}`.

### [`empirical-scripts/11.0/tests/`](empirical-scripts/11.0/tests)

`test_ndr.py` (number theory vs SymPy and long division, entropy definitions, exact reproduction of the stored v8 per-base entropies and law fit and of v10 aggregates), `test_dataset.py` (sampling, aggregation, sub-sampled sets, split, leakage guard, cache) and `test_claims.py` (statistics helpers against known answers, length arithmetic for primes, end-to-end C1–C9 on a small dataset, report table). The v8 pickle is loaded with an unpickler that only allows NumPy scalars.

### [`empirical-scripts/11.0/requirements.txt`](empirical-scripts/11.0/requirements.txt)

`pip freeze` of the `.venv` used for the verified runs (Python 3.15.2, torch 2.15.0, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, SymPy 1.14.0).

### [`empirical-scripts/10.0/sieve-echo-tests-3.py`](empirical-scripts/10.0/sieve-echo-tests-3.py)

Latest v10 discovery engine (perpetual cycles over growing `n`). Best starting point for v11's loop structure, real-time saving, and state resume; not for its math.

- **Key parts:** `Config` (16 bases `[2,3,4,5,6,7,10,12,11,13,15,17,19,23,29,31]`, population 10 000, adaptive generations 50–250, `save_threshold_ratio` 0.75 of top-10% mean, top-5 safety-net save); `MultiBaseNDRComputer.compute_ndr` (long division with a remainder dict, normalizes by `base-1`), `compute_multi_base_features` (amplitude spectrum `|FFT|` half-length *with DC*, entropy of amplitude not power; features length/kurtosis/entropy mean/std/CV); `UnifiedFormulaDiscoverer` (inputs include `phi` and `tau` — leakage; fitness on a random 200-row sample of all data); `SieveEchoDiscoverySystem` (cycles, state file `sieve_echo_state_v10.json`).
- **Depends on:** [`empirical-scripts/10.0/evolvo.py`](empirical-scripts/10.0/evolvo.py) (historical fork; forbidden for v11 by D2).
- **Common mistakes:** reusing `compute_ndr`/entropy as-is (definition differs from the paper); keeping `phi`/`tau` as inputs; treating fitness > 1 as accuracy.

### [`empirical-scripts/10.0/sieve-echo-tests-1.py`](empirical-scripts/10.0/sieve-echo-tests-1.py), [`empirical-scripts/10.0/sieve-echo-tests-2.py`](empirical-scripts/10.0/sieve-echo-tests-2.py)

Earlier v10 iterations of the same engine (`-1` still labelled v9 and includes `UnifiedNeuralSearcher`; `-2` adds co-evolution notes in its docstring). Superseded by `-3`.

### [`empirical-scripts/10.0/gold_apart.py`](empirical-scripts/10.0/gold_apart.py)

Fragment with `RealTimeSaver` (save each algorithm immediately, dedupe by signature), `MemoryManager`, `EnhancedDiscoverySystem`. Reusable pattern for v11 incremental saving.

### [`empirical-scripts/10.0/evolvo.py`](empirical-scripts/10.0/evolvo.py), [`empirical-scripts/9.0/evolvo.py`](empirical-scripts/9.0/evolvo.py)

Historical "Unified Evolvo" forks (`AlgorithmGenome`, `NeuralGenome`, `UnifiedEvolver`, `ResourceAwareEvolver`, `QLearningGuide`, `FormulaResultsManager`). Their API differs from the submodule (`GFSLGenome`, `GFSLEvolver`). Do not import or edit.

### [`empirical-scripts/9.0/sieve-echo-tests-1.py`](empirical-scripts/9.0/sieve-echo-tests-1.py)

v9 engine: multi-base NDR, formula discoverer, neural searcher, co-evolution, `RobustSerializer`, `EnhancedResultsHandler`. Superseded by v10.

### [`empirical-scripts/8.0/sieve-echo-tests-2.py`](empirical-scripts/8.0/sieve-echo-tests-2.py)

v8.2 "Unified Sieve Echo Conjecture Analysis System" — the script whose run produced the paper's §2.2, §2.4 and §4 numbers (`8.0/results/1/`, 2025-08-25; verified by key names and by exact re-fit in `tests/test_ndr.py`). n ∈ [2, 9999], 13 bases `[2..7, 9, 10, 11, 13, 14, 15, 16]`, NDR `d/b`, entropy `|FFT|²` over the first ⌊L/2⌋ bins with DC, p ≤ 1e-10 dropped (v11 `H_v8`).

- **Key parts:** `compute_ndr_entropy`, `extract_comprehensive_features`, `analyze_sieve_echo_law` (OLS on `log(ω + 1)`), `test_riemann_zeta_connections` and `test_base_invariance` (both on `self.data[:100]` only), `test_prime_number_theorem_connections` (`H·log n`), `test_modular_patterns` (variance of residue-class means > 0.01 called "significant", no test).
- **Common mistakes:** see [training-set accuracy](#training-set-accuracy-reported-as-model-accuracy) and [trivial correlations](#correlations-driven-by-n-alone).

### [`empirical-scripts/8.0/sieve-echo-tests-1.py`](empirical-scripts/8.0/sieve-echo-tests-1.py)

v8.0 "Enhanced Empirical Analysis" (20 test bases, two concatenated system classes). Its output files are named `sieve_echo_enhanced_*`; it did **not** produce `8.0/results/1/`. Contains `BaseInvarianceTester`, `RiemannZetaAnalyzer`, `PrimeNumberTheoremAnalyzer`, `CRTValidator`, `ModularFormsExplorer`, and a neural classifier around line 2225 (accuracy measured on the training tensors).

### [`empirical-scripts/8.0/`](empirical-scripts/8.0) other files

`sieve-echo-tests-3.py` … `-6.py` (v8.3–v8.6 variants: pattern discovery, "grand unified engine"), `evolvo_model.py`, `evolvo_nn.py`, `evolvo_nn-0.1.py` (historical library forks). Superseded.

### [`empirical-scripts/7.0/`](empirical-scripts/7.0), [`empirical-scripts/6.0/`](empirical-scripts/6.0), [`empirical-scripts/5.0/`](empirical-scripts/5.0), [`empirical-scripts/4.0/`](empirical-scripts/4.0), [`empirical-scripts/3.1/`](empirical-scripts/3.1), [`empirical-scripts/sieve-genetic-test-5.py`](empirical-scripts/sieve-genetic-test-5.py)

Older generations, each a test script plus a forked engine (`evolvo_model.py` / `evolvo_engine.py`). [`empirical-scripts/7.0/results/3/conclusions.md`](empirical-scripts/7.0/results/3/conclusions.md) records the theta-era fit `H = -0.9988·log ω + 2.42`, R² = 0.292, and the origin of the "std, is_prime_power, entropy_min" feature triple. Its log [`sieve_echo_log_20250825_135643.txt`](empirical-scripts/7.0/results/3/sieve_echo_log_20250825_135643.txt) is the source of the paper's "98.8%" (line 187) and "r = -0.843" (line 210, `omega_vs_is_prime_power`: a correlation between two factorization labels). History only.

### [`empirical-scripts/results/`](empirical-scripts/results)

Stored outputs: `20-Aug-2025/` and `v5_22-Aug-2025/` (text logs/results of v4–v5), `3-Sept-2025/` (v10: [`sieve_echo_state_v10.json`](empirical-scripts/results/3-Sept-2025/sieve_echo_state_v10.json) with per-`n` features up to `n ≈ 2·10^5`, 19 cycle summaries, ~300 per-generation algorithm dumps; best saved fitness 1.1235, cycle 12). Read-only evidence.

### [`empirical-scripts/8.0/results/1/`](empirical-scripts/8.0/results/1)

The v8.2 run cited by the paper: [`sieve_echo_results_20250825_224642.json`](empirical-scripts/8.0/results/1/sieve_echo_results_20250825_224642.json) (config, `sieve_echo_law` α = -0.4681, β = 1.9174, R² = 0.2345, zeta/PNT/modular results, base invariance 0.06, evolvo best fitness 0.24 with formula `DIV`), plus a `.png` and a `.pkl` whose `data` list holds all 9 998 per-n feature dicts (only NumPy scalar globals; load it with a restricted unpickler as in `tests/test_ndr.py`).

### [`libs/evolvo`](libs/evolvo) (git submodule)

Evolvo library, upstream `https://github.com/Geckos-Ink/evolvo`, branch `main`, registered in [`.gitmodules`](.gitmodules). Step 3 added declared inputs, `InstructionProfile`, `execute_batch`, `fitness_epoch`, `GFSLPolicyGuide`, `TorchTask.classification` and fixed a validator op-list aliasing bug; those commits are on upstream `main` (pushed by the owner after step 6). Own handbook: [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) — it applies to every file under `libs/evolvo/`. Source in [`libs/evolvo/src/evolvo/`](libs/evolvo/src/evolvo), tests in [`libs/evolvo/tests/`](libs/evolvo/tests), specs in [`libs/evolvo/papers/`](libs/evolvo/papers). `libs/evolvo/gold/evolvo.py` is a single-file legacy snapshot; do not use it.

### [`libs/genetic_ml/`](libs/genetic_ml), [`libs/evolvo_engines/`](libs/evolvo_engines)

Pre-submodule copies (`evolvo_model.py`, `model_evolver.py`, `evolvo_engine-1.0.py`). Superseded by [`libs/evolvo`](libs/evolvo); do not import.

### [`.gitignore`](.gitignore)

Standard Python template plus `empirical-scripts/11.0/results/*/cache/` (v11 dataset caches, tens of MB per quick run), `results/*/knowledge/` (reusable guide/search knowledge, MB-sized binaries) and `results/full-*/split.json` (1.8 MB; its SHA-256 is in the manifest). Other run outputs are not ignored; commit only the small summaries a paper or the roadmap cites.

## Features and Recurring Development Pitfalls

### Multi-base NDR dataset — Shipped (v11 data stage)

- **Behavior:** for each `n` and coprime base `b`, compute the repetend (exact order, vector doubling), NDR, spectrum, entropies; aggregate across four base sets; label, split, cache.
- **Owners:** [`ndr.py`](empirical-scripts/11.0/ndr.py), [`dataset.py`](empirical-scripts/11.0/dataset.py). Historical: v10 `MultiBaseNDRComputer`, v8.2 `extract_comprehensive_features`.
- **Gaps:** `config/full.json` not yet built; no Parquet export (`.npz` only).

### Evolutionary formula search for ω(n) — Experimental

- **Behavior:** evolve GFSL programs mapping NDR features to ω(n).
- **Owners:** v10 `UnifiedFormulaDiscoverer` on the historical fork; v11 will use `evolvo.GFSLEvolver`.
- **Gaps:** leakage through `phi`/`tau`; no held-out evaluation; slow scalar execution.

### Neural prediction of ω(n) — Experimental

- **Behavior:** v8 trained a fixed MLP classifier; v9/v10 a neural searcher on the fork. v11 will use `evolvo.nnarch` (GFSL-NN).
- **Gaps:** v8 reported training accuracy with `is_prime_power` as input.

### NDR normalization and entropy definition drift

- **Symptom:** two versions give different H_NDR for the same `n`.
- **Cause:** v8 uses `d/b` and power `|FFT|²`; v10 uses `d/(b-1)` and amplitude `|FFT|`; both keep the DC bin (k = 0), which carries the digit mean and dominates the spectrum, depressing entropy. The paper defines power spectrum over all `L` bins and `d/b`.
- **Safe pattern:** one versioned function; report the paper definition plus a DC-removed variant; normalizing by `b` vs `b-1` only rescales the sequence, so it does not change normalized-spectrum entropy (unit-tested in `test_ndr.py`).
- **Status:** fixed in v11 (`ndr.py`, `ndr-v11.1`); old scripts unchanged. The DC bin is large: on the quick run, ω = 1 integers have mean `H_paper` 2.65 but `H_nodc` 6.40 (`primes31` set).

### Repetend-length arithmetic hidden in "NDR" predictors

- **Symptom:** NDR features seem to add a lot to an ω classifier that already has every per-base L.
- **Cause:** a tree model cannot compute lcm(L_b) or test whether it divides n - 1 (λ(n) and Fermat-type structure); NDR features re-encode part of that.
- **Safe pattern:** give the baseline the length arithmetic (`features.length_arithmetic`): quick data 94.9% without it vs 97.8% with it (`quick-s0-step6` baselines); with it the NDR gain is +0.6–0.8 points.

### Midy symmetry read as "NDR information beyond L"

- **Symptom:** entropy features (especially DC-free normalized entropy and spectral flatness) add a point of ω accuracy over a length baseline at large n.
- **Cause:** when b^(L/2) ≡ −1 (mod n) the repetend's halves are digit complements and every even DFT bin except DC vanishes, so `H_nodc` is log 2 lower at fixed L (pair-level coefficient −0.694, `scale-s0-step7`). For odd n the flag needs equal 2-adic valuations of the local orders, so it is rarer for larger ω. A baseline of 19 orders also under-describes the group: orders in 98 bases add +3.3 points at 10⁵–10⁶.
- **Safe pattern:** compare against `lengths` baselines (L98 + Midy98); average gradient-boosting fits over several seeds (`lengths.model_seeds`) before reading sub-point gains.
- **Status:** with that baseline the entropy gain at 10⁵–10⁶ is −0.07 points (CI [−0.33, +0.24]).

### Training-set accuracy reported as model accuracy

- **Symptom:** "98.8% accuracy" in the paper (§3.1–3.2).
- **Cause:** v8 classifier evaluates on the same tensors it trains on, with `is_prime_power` (which alone identifies ω = 1) among three inputs; the paper also calls it a regression with MSE.
- **Safe pattern:** held-out test split, majority-class and `log n`/`log L` baselines, no factorization-derived inputs.
- **Status:** the paper claim is unverified.

### Correlations driven by `n` alone

- **Symptom:** r = 0.946 for `H·log n` (§4.2) and decreasing correlation with `n^-s` (§4.1).
- **Cause:** both covariates are monotone functions of `n`, and H grows with `n` through `L`. They measure size, not primes or zeta.
- **Safe pattern:** report partial correlations controlling for `log n` and `log L`; compare against the same statistic on a shuffled-ω null.
- **Status:** active in the paper.

### Repetend length as hidden variable

- **Symptom:** H vs log ω fit with low R², and "prime power dominance".
- **Cause:** H ≲ log L, and `L = ord_n(b)` is large for primes (divides `p-1`) and small for highly composite `n` (divides λ(n)). v10 state shows prime `n ≈ 2·10^5` with mean L ≈ 1.5·10^5 and H ≈ 10.6, ω = 4 with L ≈ 1.3·10^3 and H ≈ 5.6.
- **Safe pattern:** every H-claim also reported for the residual `H − f(log L)`; test whether NDR adds information about ω beyond `L` (the real content of the conjecture).

### Fitness above 1 and variety bonus

- **Symptom:** v10 saved "fitness 1.1235".
- **Cause:** `+0.01` per distinct operation, multiplied by a length penalty; fitness on a random 200-row resample each call, so values are noisy and not comparable across generations.
- **Safe pattern:** fixed train minibatches per generation, bounded loss-based fitness, separate parsimony term.

## Interface Ownership Map

- [`empirical-scripts/11.0/sieve_echo_v11.py`](empirical-scripts/11.0/sieve_echo_v11.py) → owns config, seed, run directory, manifest, stages.
- [`ndr.py`](empirical-scripts/11.0/ndr.py) → every number-theory and NDR/entropy definition (`NDR_VERSION`). [`dataset.py`](empirical-scripts/11.0/dataset.py) → sampling, tables, split, cache, allowed-input list. [`features.py`](empirical-scripts/11.0/features.py) → predictor inputs. [`search_formulas.py`](empirical-scripts/11.0/search_formulas.py), [`search_nn.py`](empirical-scripts/11.0/search_nn.py), [`baselines.py`](empirical-scripts/11.0/baselines.py) → step-4 experiments.
- `evolvo.GFSLEvolver`, `evolvo.GFSLExecutor`, `evolvo.register_custom_operation` → [`libs/evolvo/src/evolvo/`](libs/evolvo/src/evolvo).
- `evolvo.nnarch.ArchitectureSearch`, `TorchTask`, `Knowledge` → [`libs/evolvo/src/evolvo/nnarch/`](libs/evolvo/src/evolvo/nnarch).

## Build, Run, Debug, and Release

Prerequisites (verified 2026-11-24 on the owner's macOS arm64 machine, 8 cores): the system interpreters (`/opt/homebrew/bin/python3` 3.15.2, `/usr/bin/python3` 3.9.6) have no scientific packages. v11 uses a repo-root `.venv` on Python 3.15.2 (PyTorch publishes 3.15 wheels; torch 2.15.0 sees `mps`). Old scripts additionally need `dill` (v9/v10), not installed.

Create the environment (pins in [`empirical-scripts/11.0/requirements.txt`](empirical-scripts/11.0/requirements.txt)):

```bash
/opt/homebrew/bin/python3.15 -m venv .venv
```

```bash
.venv/bin/pip install --only-binary :all: -r empirical-scripts/11.0/requirements.txt
```

Initialize the submodule after cloning:

```bash
git submodule update --init --recursive
```

Run Evolvo's tests:

```bash
cd libs/evolvo && PYTHONPATH=src ../../.venv/bin/python -m unittest discover -s tests
```

Run the v11 tests from the repo root (44 tests incl. figures, ~1–1.5 min — the end-to-end claims test dominates; the reproduction tests need the stored v8/v10 results):

```bash
.venv/bin/python -m unittest discover -s empirical-scripts/11.0/tests -v
```

Run the v11 quick pipeline (data ~8 s, claims ~30 s, formulas ~2 min, nn ~2.5 min, report; ~5 min in all; writes `empirical-scripts/11.0/results/<run-id>/` and prints the verdict table):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-s0
```

Run only some stages, e.g. the data stage:

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-data --stages data
```

Run the step-5 scale analysis (data build is the long part; run it in the background):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/scale.json --run-id scale-s0 --stages data,claims,scale,report
```

Run the step-7 analysis (reuses the scale dataset cache; orders in bases 2..99 ~1 min, C9 baselines and ablation, Evolvo formulas):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/scale.json --run-id scale-s0-step7 --stages data,claims,scale,lengths,report
```

Resume an existing run (finished stages are skipped; config and seed must match):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-s0 --resume
```

The full config's data build takes ~20 min on 7 workers and its other stages longer; run it in the background. A new run id with an unchanged data/split config reuses an existing dataset cache.

Long runs: v10 ran perpetually with multi-GB memory. Run v11 full-scale jobs in the background with a resumable state file; use a small `--quick` config for verification.

## Test Ownership Map

- Evolvo library → [`libs/evolvo/tests/test_gfsl_core.py`](libs/evolvo/tests/test_gfsl_core.py) (GFSL core), [`libs/evolvo/tests/test_policy_guide.py`](libs/evolvo/tests/test_policy_guide.py), [`libs/evolvo/tests/test_nnarch.py`](libs/evolvo/tests/test_nnarch.py) (GFSL-NN). Functions, lists and Kompute have only smoke demos.
- v11 `ndr.py` → [`empirical-scripts/11.0/tests/test_ndr.py`](empirical-scripts/11.0/tests/test_ndr.py); `dataset.py` → [`empirical-scripts/11.0/tests/test_dataset.py`](empirical-scripts/11.0/tests/test_dataset.py); `claims.py`, `report.py` → [`empirical-scripts/11.0/tests/test_claims.py`](empirical-scripts/11.0/tests/test_claims.py); `features.py`, `search_formulas.py`, `baselines.py`, `search_nn.py` → [`empirical-scripts/11.0/tests/test_search.py`](empirical-scripts/11.0/tests/test_search.py); `scale.py` and multi-range sampling → [`empirical-scripts/11.0/tests/test_scale.py`](empirical-scripts/11.0/tests/test_scale.py); `lengths.py` (Midy ⇔ complementary halves ⇔ vanishing even DFT bins, the 2-adic characterization for odd n, order matrix, lcm of all orders = λ, length-arithmetic parity, stage end-to-end incl. reproduction of the scale stage's C9) → [`empirical-scripts/11.0/tests/test_lengths.py`](empirical-scripts/11.0/tests/test_lengths.py). The entry script has no unit test; it is verified by the quick run. Older version folders have no tests.

## Data, Security, Privacy, and Compatibility Boundaries

- Canonical data is computed from integers; everything in `results/` is derived and can be regenerated. Caches are disposable.
- No secrets, credentials, or network services are involved. Do not write the owner's email or personal data into new files.
- Result files written by v11 MUST record the NDR/entropy definition version, root and submodule git SHAs, and seed, so old results stay interpretable after definitions change.

## Current Status and Known Gaps

### Shipped

- Historical scripts v3.1–v10 and their stored results (exploratory, see pitfalls).
- Evolvo submodule at [`libs/evolvo`](libs/evolvo) with the step-3 features (see [Standing Evolvo Facts](#standing-evolvo-facts-relevant-to-v11)).

- v11 steps 0–5 and 7 ([`ROADMAP.md`](ROADMAP.md)): entry script, manifest, canonical `ndr-v11.1` definitions, dataset with split and cache, C1–C9 tests with reproduction of every stored v8.2 statistic, Evolvo formula search (4 guides × 5 seeds, S1 adaptive evaluation size, S2 RL guide with warm starts), held-out baselines, §3.1 retest, GFSL-NN search, scale analysis to 10⁶ with growth exponents, step-7 length/Midy/λ analysis (`lengths` stage), Markdown/JSON report with figures, 44 unit tests, `.venv` with torch. Verdicts and their runs are in [`ROADMAP.md` Results log](ROADMAP.md#results-log).

### Experimental / Scaffold

- Every n up to 10⁶ was not built (≈ 15 h); steps 5 and 7 use ω-stratified decades instead. The `studies/9` draft (now including step 7) awaits owner review.

### Known Gaps

- Paper numbers (§2–§4) come from one v8.2 run (plus a v7.0 log for "98.8%" and "r = -0.843") with leakage, training-set accuracy, 100-row samples and size-driven correlations. v11 reproduces every stored v8.2 statistic and has corrected verdicts for C1–C9 (ROADMAP Results log); the §3.1 three-feature claim is retested in step 4.
- Paper direction-of-effect statements contradict each other and the data.

### Near-Term Priorities

1. Owner review of the step-6/7 reports and the `studies/9` draft ([`ROADMAP.md`](ROADMAP.md) steps 6–7).
2. Remaining open questions of `studies/9` §8: the rest of the C7 residue structure (n mod 7, part of n mod 4; candidate: k-block Midy conditions) and order-only ω classification beyond 10⁶.

## Task Start and Handoff Checklist

1. Read this file, the relevant [`ROADMAP.md`](ROADMAP.md) step, and [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) when touching Evolvo.
2. `git status` in the root and in `libs/evolvo`; preserve unrelated changes.
3. Check the [principles](#essential-project-principles): no factorization-derived inputs, held-out evaluation, `log L` control, versioned definitions.
4. Put library fixes in `libs/evolvo` (D3), with a test; commit there on `main`, then commit the pointer here.
5. Run the focused tests and a `--quick` v11 run; report what ran and what did not.
6. Update [`ROADMAP.md`](ROADMAP.md) status and this file's source map, commands, and gaps in the same change.
