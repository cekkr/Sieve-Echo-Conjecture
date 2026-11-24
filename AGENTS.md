# Sieve Echo Conjecture — AI Agent Reference

Fast-access operational reference for AI agents working in this repository. Read it completely before changing anything, and keep it true for the checked-out revision (see [Maintenance rules](#collaboration-and-maintenance-rules)).

**What this project is.** An empirical number theory repository that studies the arithmetic of the repetends of `1/n` across bases. The original Sieve Echo Conjecture proposed spectral entropy as a signal of ω(n). Version 11 tests that claim and identifies multiplicative order, Midy symmetry, and character-induced spectral zeros as the central objects. The v11 test suite lives in [`empirical-scripts/11.0/tests/`](empirical-scripts/11.0/tests). Evolvo and SinWhisky have separate submodule handbooks.

**Current research goal.** Study the joint distribution of repetend lengths at prime denominators. Use fixed divisibility conditions on the index `(p−1)/ord_p(b)`. Count primes over explicit ranges, verify the conditions independently, and compare the counts with densities from residue classes or splitting fields. Steps 20–21 close this phase: full-period repetends are Artin's primitive-root problem (Hooley's densities under GRH), and the halved-repetend counts are explicit sums over the zeros of named Dirichlet L-functions. Keep exact identities, finite computations and asymptotic estimates distinct. A Riemann-related statement needs a named L-function, a precise observable or bound, and an explicit implication or hypothesis. No finite computation here proves or supports RH beyond the verified range; do not write that it does. See the [research roadmap](ROADMAP.md#next-research-joint-distribution-of-repetend-lengths).

## Read This First

Source-of-truth order. When two sources disagree, the higher one wins, and the agent fixes or records the mismatch in the same change.

1. **This file's [Directives](#standing-directives-from-the-project-owner)** — standing instructions from the project owner.
2. [`ROADMAP.md`](ROADMAP.md) — the research plan and the completed v11 experimental record. It governs *what to do next*; it never proves that something exists.
3. Executable code: [`libs/evolvo/src/evolvo/`](libs/evolvo/src/evolvo) (the library) and [`empirical-scripts/11.0/`](empirical-scripts/11.0). Evolvo's own rules are in [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) and its specs in [`libs/evolvo/papers/`](libs/evolvo/papers).
4. [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md) — the historical claims tested by v11. The current research summary is the [revision-9 draft](studies/9/sieve-echo-conjecture-9.md).
5. Stored results under [`empirical-scripts/results/`](empirical-scripts/results) and `empirical-scripts/<ver>/results/` — evidence of what past runs produced, with the methodological defects listed below.
6. Older papers ([`studies/7/`](studies/7), [`studies/8/future.md`](studies/8/future.md)), the notes in [`studies/claude/`](studies/claude), and commit messages — history only. Commit messages in this repo carry no technical information.

## Standing Directives from the Project Owner

These are mandatory for every task from v11 onward.

- **D1 — Version 11 lives in `empirical-scripts/11.0/`.** Keep the reproducible evaluation of the [revision-8 claims](studies/8/sieve-echo-conjecture-8.md) available as the experimental record. Edit older version folders only to fix a link. The [research roadmap](ROADMAP.md#next-research-joint-distribution-of-repetend-lengths) defines the order-distribution follow-up. The fixed-base decoder serves as a benchmark showing what the orders capture.
- **D2 — Use the Evolvo library from the submodule.** All formula search and neural-architecture search MUST import `evolvo` from [`libs/evolvo/src`](libs/evolvo/src) (upstream: `https://github.com/Geckos-Ink/evolvo`). Do NOT copy `evolvo.py`, `evolvo_model.py`, `evolvo_nn.py`, or `gold/evolvo.py` into `11.0/`, and do NOT import the historical copies in older version folders.
- **D3 — Fix Evolvo in Evolvo.** When v11 needs a feature or bug fix in Evolvo, implement it inside [`libs/evolvo`](libs/evolvo), following [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) (update its `AGENTS.md` and `README.md`, add a test under [`libs/evolvo/tests/`](libs/evolvo/tests)), commit it on the submodule's `main` branch, then commit the updated submodule pointer in this repository. Do not work around an Evolvo defect inside the v11 script when the fix belongs in the library.
- **D4 — Keep [`ROADMAP.md`](ROADMAP.md) live.** Update step status while implementing: mark steps done only after verification, add newly discovered fixes or follow-ups, and record results with the run that produced them.
- **D5 — Keep this file live.** Follow the [AGENTS.md bootstrap protocol](#collaboration-and-maintenance-rules): every change that adds, moves, or changes a file, command, contract, or result updates the matching section here in the same change.
- **D6 — Evaluation size grows with quality.** Each genome is scored on 5 000 integers at first, rising step by step to at most 100 000 as the average of the best scores improves. Never compare fitness values taken at different sizes. Spec: [`ROADMAP.md` S1](ROADMAP.md#s1--adaptive-evaluation-size-5-000--100-000).
- **D7 — Reinforcement learning through PyTorch is required.** A torch guide learns which genome edits work over time (weights saved and reused across runs) and lowers exploration as the best scores improve, which reduces the spread of results. Rewards are relative to the average best scores. PyTorch is therefore a hard dependency of v11. Spec: [`ROADMAP.md` S2](ROADMAP.md#s2--torch-reinforcement-learning-guide).
- **D8 — Where v11 takes an FFT of a digit sequence, also use SinWhisky.** The owner's circle-based resampler lives in the git submodule [`libs/sinwhisky`](libs/sinwhisky) (upstream `https://github.com/cekkr/sinwhisky.py`); v11 imports its vectorized `v2/whisky_np.py` through `runtime.bootstrap_sinwhisky()` and compares spectra after SinWhisky resampling with the plain FFT (step 8e). Edit SinWhisky inside the submodule (keep its `AGENTS.md`/`README.md` and tests current, commit on its `main`), then commit the pointer here, as for Evolvo (D3).

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
- **Order index** — for a prime `p ∤ b`, `I_b(p) = (p−1)/ord_p(b)`, the index of the subgroup generated by the base. The planned prime-counting benchmarks use fixed divisibility conditions on these indices.
- **NDR** (Normalized Digit Representation) — the repetend digits divided by the base. The paper (Definition 1.1) uses `d/b`; v10 used `d/(b-1)`. See [pitfall](#ndr-normalization-and-entropy-definition-drift).
- **H_NDR** (NDR entropy) — Shannon entropy of the normalized power spectrum of the NDR sequence (paper Definition 1.2). v11 names: `H_paper` (all L bins, DC kept), `H_nodc` (DC dropped), `Hn_*` (divided by log length), `H_v8`/`H_v10` (historical definitions, reproduction only). Definitions in [`ndr.py`](empirical-scripts/11.0/ndr.py), tag `ndr-v11.1`.
- **Base invariance** — coefficient of variation (CV = std/mean) of H_NDR across bases below 0.1.
- **GFSL** — Genetic Fixed Structure Language, the slot-encoded genome language implemented by Evolvo. **GFSL-NN** — its neural-architecture variant (`evolvo.nnarch`).

## Essential Project Principles

### Fitted claims use a null model and held-out data

- Every reported fit, correlation, or accuracy MUST be measured on data not used to fit it (train/validation/test split by `n`, fixed seed), and compared with a baseline that uses only trivial covariates (`log n`, `log L`) and with a permutation/shuffled-label null.
- Exact enumerations use a declared finite range, an independent arithmetic check, and a theoretical comparison when one is available.
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

### [`README.md`](README.md)

Public entry point. Summarizes the entropy controls, exact order results, prime-counting benchmarks, the full-period (Artin) benchmark and the explicit-formula check against L-function zeros, with a section stating what the results do and do not say about the Riemann hypothesis. Its best-Evolvo-algorithms table lists the exact formulas of steps 19–20 with their held-out counts, the congruence-only controls and links to the replayable program files. Reproduction includes the prime-order, Artin and explicit-formula run commands.

### [`ROADMAP.md`](ROADMAP.md)

Live research plan. The prime-counting milestones cover verified quadratic and cubic counts, a mixed-index implication and a symbolic search with Evolvo. The completed v11 steps and their results preserve the experiment history. The [counterexample entry](ROADMAP.md#all-19-base-counterexample) records the deterministic standalone decoder command, its input, and the matching prime-order vectors.

### [`conclusion.md`](conclusion.md)

Research synthesis. Explains the entropy result and the order tests, gives the fixed-base decoder as a reconstruction example with explicit failure and supporting run IDs, and connects the order index to the prime-counting benchmarks. Section 7 includes the selected Evolvo algorithms as runnable Python, the scope of their checks and reproduction commands. Sections 8–9 give the Artin benchmark (`artin-s0`) and the explicit-formula and Chebyshev-bias results (`explicit-s0`); section 10 separates exact identities, conditional theorems, finite checks and what a proof of RH would need.

### [`empirical-scripts/11.0/conclusions.md`](empirical-scripts/11.0/conclusions.md)

Version 11 experimental record. Preserves the original conjecture, reproduction results, statistical tables, exact calculations and assumptions. Change empirical numbers with their backing run or Results-log entry.

### [`studies/9/sieve-echo-conjecture-9.md`](studies/9/sieve-echo-conjecture-9.md)

Revision-9 draft for owner review. Makes the v11 claims with their run IDs, gives the arithmetic definitions and proofs, and states the fixed prime-counting observables at prime denominators in §8.

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

v11 entry point. Parses `--config --seed --run-id --resume --stages --results-dir`, imports Evolvo through `runtime.bootstrap_evolvo()` before any stage, creates `results/<run-id>/`, and runs the requested stages in order (`data, claims, formulas, nn, scale, lengths, zeros, whisky, euler, orders, tworank, decode, subgroup, failures, validate, prime_orders, artin, explicit, report`).

- **Key parts:** `Run` (manifest, log, stage status; refuses to reuse a run dir without `--resume`, refuses `--resume` with a different config or seed), `stage_data` (build, load, or reuse the cached dataset; writes `data_summary.json` and `split.json`), `load_table`, `stage_claims` (`claims.json`), `stage_formulas` (`formulas.json`, `best_formulas.md`, `knowledge/policy_<target>.pt`), `stage_nn` (`baselines.json`, `nn.json`, `knowledge/nn_<input>.json`), `stage_scale` (`scale.json`), `stage_lengths` (`lengths.json`, `cache/lengths-<key>.npz`), `stage_zeros` (`zeros.json`, `cache/zeros-<key>.npz`), `stage_whisky` (`whisky.json`), `stage_euler` (`euler.json`), `stage_orders` (`orders.json`; needs no dataset), `stage_tworank` (`tworank.json`; needs no dataset), `stage_decode` (`decode.json`; needs no dataset), `stage_subgroup` (`subgroup.json`), `stage_failures` (`failures.json`; needs no dataset), `stage_validate` (`validate.json`; needs the external prime list), `stage_prime_orders` (`prime_orders.json`; no digit dataset or external prime list), `stage_artin` (`artin.json`, `config/artin.json`), `stage_explicit` (`explicit.json`, `config/explicit.json`), `stage_report` (`report.md`, `report.json`, prints the verdict table), `IMPLEMENTED` contains every registered stage.
- **Contract:** default `--stages` is `data,claims,formulas,nn,scale,lengths,zeros,whisky,euler,report` (`orders`, `tworank`, `decode` and `failures` run with `config/orders.json`; `prime_orders`, `artin` and `explicit` have their own configurations). `report` needs at least one analysis output; without `claims.json` it writes the other sections only. Add each new stage to `STAGE_FUNCS` and `IMPLEMENTED`. When no local cache exists, `stage_data` hard-links (or copies) `results/*/cache/dataset-<key>.npz` and its `data_summary.json` from another run with the same key and records `reused_from_run`; bump `dataset.DATASET_FORMAT` or `ndr.NDR_VERSION` whenever code changes the dataset content, or stale caches will be reused.

### [`empirical-scripts/11.0/runtime.py`](empirical-scripts/11.0/runtime.py)

Non-scientific helpers: repo paths, `bootstrap_evolvo()` (submodule only; raises `EvolvoMissingError`), `git_info()` (SHA, dirty flag and changed paths, ignoring `11.0/results/` and `.DS_Store`), `code_sha256()` over the v11 `.py` files, `environment_info()`, `repo_relative()` (keeps home-directory paths out of results), `bootstrap_sinwhisky()` (D8), `shared_cache(run_dir, name)` (reuse a content-keyed cache file from another run by hard link — used by the lengths and zeros caches), NaN-safe atomic `write_json`, SHA-256 helpers.

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

`write_report(run_dir, table=None)` reads `manifest.json` and available stage outputs. It writes `report.md` and `report.json`, including verdicts for claim runs, figures when available, and the exact counts and GFSL formula summary for `prime_orders.json`. `python empirical-scripts/11.0/report.py <run_dir>` regenerates the report and figures of a finished run without rerunning the analyses or changing the manifest.

### [`empirical-scripts/11.0/figures.py`](empirical-scripts/11.0/figures.py)

Static PNG figures into `<run>/figures/`: `entropy_vs_length` (needs the dataset), `heldout_omega` (baselines/nn/formulas), `formula_guides_omega` (formulas), `c9_gain_by_decade`, `invariance_by_decade`, `growth_exponents` (scale), `c9_gain_by_baseline`, `midy_entropy_gap` (lengths), `spectral_zeros_by_omega` (zeros). Palette: categorical blue/orange/aqua (validated with the dataviz validator: adjacent CVD ΔE ≥ 9.2; aqua < 3:1 contrast, so lines are always direct-labelled), single-hue blue ramp for ω, legend plus direct labels for multi-series charts.

### [`empirical-scripts/11.0/features.py`](empirical-scripts/11.0/features.py)

Predictor inputs shared by claims C9, baselines, formulas and GFSL-NN: `per_base` (rows × bases matrix of a pair feature), `length_arithmetic` (log lcm of the L_b, log n − log lcm, lcm | n−1, log gcd(lcm, n−1), share of L_b dividing n−1 — functions of n and L only) and `length_arithmetic_from_orders` (the same from a rows × bases order matrix), `column`/`build_matrix` (named allowed inputs; raise on `label.*`, `target_side.*`, `pair.*`), `Standardizer` (train z-score, NaN → train median), `token_tensor` (per-base tokens for the GFSL-NN set input).

### [`empirical-scripts/11.0/search_formulas.py`](empirical-scripts/11.0/search_formulas.py)

ROADMAP step 4 / S1 / S2. `prepare` (12 standardized inputs, train permutation, val/test arrays per target), `Sizer` (S1: monotone N driven by the top-k mean toward `q_target` and by the 2-SE spread trigger, capped by train size), `run_one` (one target × guide × seed × variant in a worker: `GFSLEvolver` with `inputs=`, `outputs=`, `InstructionProfile.formula(allow_custom=False)`, fitness `1/(1+NMSE) − parsimony·size` via `execute_batch` on the first N permuted train rows, `fitness_epoch=lambda: N`, v10 relative saving + safety net, validation selection, one test read), `formula_expression` (readable formula over named inputs), optional target-specific inputs, operation profiles, program slots and finite-domain audits for steps 17–19, `summarize` (guide/variant table, S1/S2 checks with Mann–Whitney p), `run_stage` (phase A: guides × seeds adaptive, fixed-N ablation, one knowledge run per target; phase B: warm policy runs loading it).

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

### [`empirical-scripts/11.0/zeros.py`](empirical-scripts/11.0/zeros.py)

ROADMAP step 8: exact zeros of the NDR spectrum. `pair_zeros` (per pair: vanishing classes k | L from the float FFT of the remainders — all bins with L/gcd(j, L) = k vanish together — split into Midy classes and *extra* classes; shares, counts, smallest extra k, power-gap diagnostics, partly-vanishing classes), `exact_zero_classes` (exact test Φ_k | Σ_t T_t x^t with T_t = Σ_{i≡t mod k} r_i; tests and run samples only), `float_zero_classes`, `compute`/`load_or_compute` (process pool over the pairs of the v8/v10/primes31 bases; cache `cache/zeros-<key>.npz` keyed by dataset key, bases, tolerance and `ZEROS_VERSION`), `row_block` (per-base extra-zero share + aggregates), `describe`, `exact_check`, `c9` (L19 ± zeros, L98 + Midy98 ± zeros ± entropy/all NDR, zero aggregates alone; same samples and seeds as step 7), `residues` (C7 ± Midy ± zero controls), `run_stage`.

- **Contract:** the theory is in the module docstring (DFT identity, class-wise vanishing, character expansion with Euler factors) and in `tests/test_zeros.py`; zero features are digit (NDR) features, allowed as inputs.
- **Common mistakes:** thresholding bins individually instead of by class; using `flatness` as a zero detector (it floors zero bins at 1e-12 of the mean and mixes all classes).

### [`empirical-scripts/11.0/whisky_spectra.py`](empirical-scripts/11.0/whisky_spectra.py)

ROADMAP step 8e (directive D8): on an ω × n-range stratified sample of (n, b) pairs (19 bases, `max_L`), each NDR period is resampled ×(zoom + 1) by SinWhisky (`sinwhisky_resample_np(..., periodic=True)` from [`libs/sinwhisky`](libs/sinwhisky)) and by periodic linear interpolation (`linear_periodic`, the linear control that keeps exact spectral zeros). `pair_record` (entropies of the plain, SinWhisky and linear spectra; leakage of power into the original zero classes), `run_stage` (leakage summary, Midy coefficient per entropy, test ΔR² of log ω beyond log L, log b, log n, Midy flag and extra-zero share, bootstrap CI). Output `whisky.json` records the SinWhisky submodule's git state.

### [`empirical-scripts/11.0/euler.py`](empirical-scripts/11.0/euler.py)

ROADMAP step 9a: small-prime membership I_{q,b} = [ord_n(q) | ord_n(b)] (q ∈ ⟨b⟩ exactly for cyclic (ℤ/nℤ)*), motivated by the Euler factors of L(1, χ) in Proposition 1.5(c). `membership`, `pair_level` (test ΔR² of H beyond log L, log b, log n, Midy, extra-zero share; per-q coefficients), `residues` (C7 excess with C7 + Midy + zero controls, then + membership), `run_stage` (reuses the lengths and zeros caches).

- **Common mistakes:** controlling with Jacobi symbols or other functions of n mod 4q — they absorb the residue excess by construction and explain nothing.

### [`empirical-scripts/11.0/orders_scale.py`](empirical-scripts/11.0/orders_scale.py)

ROADMAP step 9b: order-only ω classification per decade up to 10⁸, no NDR data. `omega_vectorized` (trial division by primes ≤ √max n over candidate batches), `factorize_td`, `sample` (per class min(ω, K) per decade; small decades enumerated), `orders` (process pool: orders, Midy flags, ω and λ from the factorization), `run_stage` (nested base sets, GBM averaged over `model_seeds`, paired-bootstrap gains, lcm19 = λ coverage). Labels and λ never enter the features.

### [`empirical-scripts/11.0/tworank.py`](empirical-scripts/11.0/tworank.py)

ROADMAP step 10: the 2-rank of (ℤ/nℤ)* (docstring = Proposition 1.6 of `studies/9`: P(v₂(ord b) ≤ k) = 2^{−Σ max(e_i − k, 0)}, ratio 2^{rank₂}, ω = rank₂ − δ(v₂(n)) + [n even]). `valuation` (ℓ-adic valuations of an order matrix), `histogram`, `ratio_estimate`, `mle_rank` (step 11b: exact integer DP maximizing the v₂-profile likelihood under r_0 ≥ r_1 ≥ …), `omega_from_rank`, `sylow2_exponents` (tests), `run_stage` (same integers as `orders_scale` via its `sample`; orders in bases 2..499; ratio and MLE accuracy per base set and decade; GBM models from `tworank.models` — lengths and/or profiles per base set — with fixed-base-set `comparisons`).

### [`empirical-scripts/11.0/decode.py`](empirical-scripts/11.0/decode.py)

ROADMAP step 13: exact, training-free decoding of ω — and of the prime factors — from the repetend lengths (docstring = Proposition 1.7 of `studies/9`). For each base b coprime to n and each prime q | L = ord_n(b), the tests gcd(b^{(L/q^e) q^i} − 1, n) refine {n}; `coprime_base` reduces the parts; ω̂ = their number (ω̂ ≤ ω; exact when the prime divisors have distinct order vectors). `decode` (variants: q = 2 only — Shor's half-period split —, pre-periods of non-coprime bases, per-base checkpoints, a `remainder` callback), `remainder_from_digits` (b^d mod n from ⌈log_b n⌉ repetend digits: ⌊nD/b^k⌋ + 1), `prime_divisors` (primes of L from those of λ(n): a speed shortcut, same set), `order_vector_classes` (lower bound, tests only), `records`/`exhaustive` (process pools), `summarize`, `digit_check` (13c: remainders read from digits and whole integers decoded from digits only), `run_stage` (exhaustive range, ω-balanced decades via `orders_scale.sample` — the first four reproduce the `orders-s0` integers —, uniform samples, Midy ⇔ Shor-failure check).

- **Common mistakes:** treating ω̂ = ω as a check of the full factor-and-exponent output. The experiment counts pairwise coprime parts. The standalone decoder must also compute or receive the lengths and find their prime divisors.

### [`empirical-scripts/11.0/sieve_echo_decoder.py`](empirical-scripts/11.0/sieve_echo_decoder.py)

The final v11 algorithm as a standalone, dependency-free script (the same procedure as `decode.decode` with pre-periods over the 19 bases, returning the factors): `repetend_length` (long division, L steps — the expensive part), `primes_of`, `coprime_parts`, `decode(n, lengths)` → {factor: exponent}, `exponent_of`, `factor(n)`, `is_prime` (trial division, used only to check the output). `python sieve_echo_decoder.py [n …]` decodes and self-checks; the defaults are hand-checkable (2047, 1001, 1024, 214935). Its code is quoted in `conclusion.md` and `11.0/conclusions.md` §7 — keep them identical when it changes.

- **Common mistakes:** demo inputs above ~10⁶ (the long division takes up to n steps per base); presenting it as an Evolvo result (it is derived; Evolvo's best ω formula reaches 65.5%).

### [`empirical-scripts/11.0/subgroup.py`](empirical-scripts/11.0/subgroup.py)

ROADMAP step 14: exact subgroup membership q ∈ ⟨b⟩ mod n and the rest of the C7 residue structure. `exact_member` (CRT over the prime powers of n: baby-step giant-step discrete logs `_dlog`, consistency modulo gcd of the local orders; the factorization is only a speed-up for this function of (n, b, q), which equals "q mod n is a remainder of 1/n in base b"), `cyclic_group`, `exact_membership` (divisibility flags from `euler.membership`, exact recomputation only where they can differ: divisibility true in a non-cyclic group; process pool), `negative_membership` (exact −q ∈ ⟨b⟩ for every cell), `run_stage` (reuses the lengths and zeros caches; C7 excess with C7 + Midy + zeros, then + divisibility/exact flags for q ≤ 13 and for the primes below 100, ±q flags, prime-power-modulus breakdown, residual means by n mod 12/7, and labelled diagnostics: joint coprimality pattern, min(v₂(n), 3)).

- **Common mistakes:** reading the diagnostic rows as explanations — the pattern and v₂(n) are functions of the small-prime divisibility of n and absorb those residue classes by construction; the claim is about the units. Treating [ord q | ord b] as membership for non-cyclic groups (it differs in 78% of such cells).

### [`empirical-scripts/11.0/failures.py`](empirical-scripts/11.0/failures.py)

ROADMAP step 15: the failure set of the step-13 decoder (docstring = Proposition 1.8 of `studies/9`). `sieve_primes`, `base_primes`/`prime_closed` (every prime dividing a base must be a base), `order_matrix` (int32 orders of the bases modulo primes, process pool), `collisions` (equal order vectors by 64-bit hashing, verified exactly; least product p·p'), `certificate` (least failing n over all small-prime patterns S, pruned by ∏S; exact or a certified lower bound ∏S · p₀ · X⁺), `decodes`, `first_failure_brute` (validation), `run_stage` (19 bases with primes up to `prime_limit`, prefixes of the bases with brute-force agreement or a direct check at the least failure found).

- **Common mistakes:** applying Proposition 1.8 to a base set that is not prime-closed (e.g. 6 without 2 and 3: the pre-periods then do not isolate the small primes); reading a lower bound as the first failure (`exact` false).

### [`empirical-scripts/11.0/validate.py`](empirical-scripts/11.0/validate.py)

ROADMAP step 16: the published decoder (`sieve_echo_decoder.decode`) on integers built from an external prime list (`primes50.zip` of The Prime Pages, t5k.org: the 49 000 001st–50 000 000th primes, ≈ 9.6–9.8·10⁸), kept in the git-ignored `results/primes-db/cache/txt/primes50.txt` (download and unzip it there; the stage checks its SHA-256, count, order, random consecutive pairs and the primality of every prime used). `load_primes`, `check_list`, `build` (families: one/two/three list primes, p²·q, list prime × primes in (31, 10⁵), list primes × primes 2…31), `lengths_from_factors` (the lengths are produced from the known factors — validation only), `verify_length` (L = ord_n(b) checked from n alone), `run_stage`.

- **Common mistakes:** feeding the list or the factors to the decoder (it sees n and the lengths only); reading the first five families as a test of the pre-period step (they are coprime to every base by design; the last family tests it).

### [`empirical-scripts/11.0/prime_orders.py`](empirical-scripts/11.0/prime_orders.py)

ROADMAP steps 17–18. Enumerates primes via the smallest-prime-factor sieve and computes exact order indices for bases 2, 3 and 5 using `ndr.multiplicative_order`. Checks every eligible prime against reciprocity or modular powers. Reports quadratic pattern counts, interval and cumulative discrepancies, and character von Mangoldt sums with prime-power corrections. Reports cubic marginal/joint rates, the conditional independence comparison, the mixed implication, and the GRH-conditional count envelope alongside the unconditional theorem's range. Exclusions are part of each output. The stage calls the optional formula search on primes coprime to all three bases.

### [`empirical-scripts/11.0/prime_order_formulas.py`](empirical-scripts/11.0/prime_order_formulas.py)

ROADMAP step 19. `run_cases(tasks, cfg, seed, run_dir, log, test_end, lookups=…, knowledge_prefix=…)` is the shared runner for steps 19 and 20 (`run_stage` builds the step-19 cases and calls it). A case may carry its own `split_at` and `regression: True` (no two-class check); `residue_baseline(data, modulus)` takes the residue from the last column. Uses `search_formulas` and the Evolvo PyTorch policy guide. Uses size-separated train/validation/test primes, exact finite-domain audits, a constant, seeded shuffle and a train-only residue lookup where relevant. Saves the selected programs' full GFSL slots and verifies replay by validation error. `replay(program, X)` takes the numeric input columns in `program["inputs"]` order. The marginal-event search inputs already include the individual order predicates. The cubic congruence-only problem uses functions of the residue modulo 120 and records a held-out error lower bound for that representation.

### [`empirical-scripts/11.0/artin.py`](empirical-scripts/11.0/artin.py)

ROADMAP step 20: full-period repetends (I_b(p) = 1, Artin's primitive-root problem) for the 19 decoder bases over the primes in (31, 10⁸]. The module docstring states the exact facts and what is conditional. Orders come from `failures.order_matrix`; the index parity is checked against quadratic reciprocity (`legendre_by_reciprocity`) for every prime and base, and SymPy checks the orders on small primes and the primitive-root status on a seeded sample. Densities: `moree_density` (Hooley's closed form with the Lehmer–Heilbronn correction, perfect powers included) and `index_density(b, t, A)` (the index law from the Kummer degrees `kummer_degree`, an independent route that must agree at t = 1); `artin_constant` (product over the sieved primes with an E1 tail, checked against ten digits of OEIS A005596); `pair_density_2_3`. `run_stage` reports counts per cutoff and base, the corrected bases (Δ ≡ 1 mod 4), the index law for t ≤ `index_max` (densities that vanish must have zero counts), the local rates r_q of q | I_b(p) for q ≤ `local_q_max`, the joint (2, 3) count and the Heath-Brown union count, then calls `artin_formulas.run_stage` when `search` is set.

- **Common mistakes:** reading the binomial-scale z values as a test of Hooley's theorem (its error term has no explicit constant; the z values only scale the finite discrepancy); applying `index_density` to perfect powers (it raises).

### [`empirical-scripts/11.0/artin_formulas.py`](empirical-scripts/11.0/artin_formulas.py)

ROADMAP step 20 Evolvo cases through `prime_order_formulas.run_cases`: `artin_criterion` (full period of base 10 from the local bits, audited on 64 Boolean rows), `lehmer_nesting` ([10 | I_5] from [2 | I_5], [5 | I_5]; the implication behind the factor 20/19), `artin_congruence_only` (full period from congruence data only, with the mod-840 lookup and the unavoidable-error count), `kummer_degree` (regression of y_q = q²·r_q → q/(q − 1) over the primes q ≤ 401, split by alternation so every split contains small q; `kummer_check` gives binomial-scale standardized residuals, the degree q²/y against q(q − 1) and the Artin product rebuilt from the formula). A size split over q fails here: validation primes above 23 cannot see an additive offset in the degree, which only shows at q = 2, 3. `search_formulas.metrics` reports relative errors instead of F1 for non-binary targets.

### [`empirical-scripts/11.0/explicit.py`](empirical-scripts/11.0/explicit.py)

ROADMAP step 21. The docstring gives the identities and hypotheses. `kronecker`/`character_table` (real primitive even characters χ_Δ, Δ = disc Q(√b)), `halved_characters` (χ_Δ(p) from the repetend index parity, checked against the Kronecker symbol on every prime), `b_constant` (constant term of L'/L at 0 by central differences and by the functional equation; mpmath is avoided at s = 1 itself, where its Hurwitz sum stalls for conductor 40), `find_zeros` (sign changes of the real function Z(t) on a grid, refined by `findroot`; S(γ_k) estimates as a completeness check; for ζ the list must equal `mpmath.zetazero`), `smoothed_measured`/`smoothed_predicted` (the Gaussian-smoothed explicit formula, exact on the prime side), `detectable_height`, `rubinstein_sarnak` (bias δ under GRH + LI from the zeros with a Gaussian tail), `race` (logarithmic share, lead changes, log-mean of E(x) and its 1 + 2/log x leading order), `schoenfeld` (largest ratio to the RH-conditional bound on [599, X]).

- **Common mistakes:** calling the finite Schoenfeld ratio or the zero check evidence for RH (both are far inside long-verified ranges); comparing the measured race share with δ as if the limit were reached (the oscillation period in log x is about 2π/γ₁).

### [`empirical-scripts/11.0/config/artin.json`](empirical-scripts/11.0/config/artin.json) and [`explicit.json`](empirical-scripts/11.0/config/explicit.json)

`artin.{bases, prime_limit, cutoffs, index_max, local_q_max, sympy_order_limit, sympy_sample, workers, search.{split_at, domain_limit, kummer_check_max, seed_offsets, population, generations, initial_instructions, max_instructions, max_effective_instructions, workers}}`; `explicit.{bases, prime_limit, height, step, dps, sigma, u_points, u_min, bias_t_max, bias_dt, workers}`. Run with `--stages artin` and `--stages explicit` (add `report`, or regenerate with `report.py`).

### [`empirical-scripts/11.0/tests/test_artin.py`](empirical-scripts/11.0/tests/test_artin.py) and [`test_explicit.py`](empirical-scripts/11.0/tests/test_explicit.py)

Closed-form vs Kummer-degree densities for every non-power b < 200, hand values (20/19, 3/5, A/8 at t = 4 for base 2, zero at t = 5 for base 5), the index law summing to 1, Kummer degrees, reciprocity vs SymPy, counts vs SymPy primitive roots, the Evolvo replay of q(q − 1); characters, L'(0, χ) = log ε for class number 1, the zero finder vs `mpmath.zetazero`, the smoothed explicit formula to 10⁻⁸ for χ₈ and ζ, the bias integral on a one-term model, race bookkeeping and the Schoenfeld bound through 10⁶.

### [`empirical-scripts/11.0/config/prime-orders.json`](empirical-scripts/11.0/config/prime-orders.json) and [`prime-orders-confirm.json`](empirical-scripts/11.0/config/prime-orders-confirm.json)

Configurations for `--stages prime_orders,report`. The first runs the fixed benchmarks and all five formula cases. The follow-up folds the modulo-8 coordinate by its distance from 4, expands the search budget for the quadratic formulas, and uses only primes beyond the first run's range as test inputs. Omit the `search` block for counts alone.

### [`empirical-scripts/11.0/tests/test_prime_orders.py`](empirical-scripts/11.0/tests/test_prime_orders.py)

Checks reduced residue classes, character prime-power corrections, cubic and mixed counts, size splits, modulo-class ambiguity bounds and Evolvo slot replay.

### [`empirical-scripts/11.0/config/`](empirical-scripts/11.0/config)

`quick.json` (n ∈ [2, 9999] + 100 per ω from [10⁴, 10⁵], 4 base sets; builds in ~8 s), `full.json` (n ∈ [2, 150 000] + 2 000 per ω up to 10⁶; data ~20 min) and `scale.json` (step 5: n ∈ [2, 9999] + 3 000 per ω in [10⁴, 10⁵) and in [10⁵, 10⁶], wide set on 6 000; run with `--stages data,claims,scale,report`). Keys: `seed`, `data.{n_min, n_max, stratified, base_sets, workers}`, `split.{train, val, test, stratify}`, `claims.{primary_set, entropies, bootstrap, permutations, effect_threshold, cv_threshold, invariance_sets, invariance_variants, moduli, zeta_s, classifier}`. `lengths.{bases, bins, per_omega_cap, min_per_omega, bootstrap, max_omega_class, gbm, ablation, pair_bins, formulas.{guides, seeds, population, generations, …}, workers}` (step 7; `formulas: null` skips the Evolvo part). `zeros.{tolerance, exact_sample, exact_max_L, bins, per_omega_cap, min_per_omega, bootstrap, model_seeds, max_omega_class, gbm, length_bases, sets, workers}` (step 8). `whisky.{zoom, neighbors, bins, per_stratum, max_L, min_L, sets, bootstrap}` (step 8e). `euler.{small_primes, length_bases, sets, tolerance, bootstrap, workers}` (step 9a). `subgroup.{small_primes, large_primes, length_bases, sets, tolerance, workers}` (step 14). [`config/orders.json`](empirical-scripts/11.0/config/orders.json): `orders.{decades, per_class, max_class, batch, max_batches, bases, base_sets, split, model_seeds, bootstrap, gbm, workers}` (step 9b; no `data` block — run `--stages orders,report`); `tworank.{bases, estimator_sets, length_set, profile_sets, models, comparisons, v2_bins, v3_bins, model_seeds, bootstrap, gbm, workers}` (step 10, `--stages tworank,report`); `failures.{bases, prime_limit, prefix_checks, brute_limit, workers}` (step 15, `--stages failures,report`); `validate.{prime_list, source, sha256_txt, expected_count, families, medium_max, gap_checks, workers, chunk}` (step 16, `--stages validate,report`); `decode.{bases, prime_bases, decades, per_class, max_class, batch, max_batches, uniform_per_decade, exhaustive_max, exhaustive_chunk, max_failures, digit_check.{decades, pairs, max_L, decode_integers}, workers}` (step 13, `--stages decode,report`). `formulas.{targets, guides, seeds, population, generations, parsimony, eval_size_min/max/step, q_target (per target), top_fraction, save_threshold_ratio, save_top_per_gen, safety_net, ablation_guide, fixed_size, warm_start, knowledge_seed, inputs, workers}`; `nn.baselines.{max_train_rows, gbm}`, `nn.search.{population, generations, patience, max_train_rows, max_val_rows, hard_max_epochs, max_rows, max_params, knowledge_in}`. Quick: population 100 × 30 generations, N step 500 (train has 7 366 rows); full: 200 × 60, N step 5 000 up to 100 000. A base set is a list, `"a..b"`, or `{"bases": …, "sample": k}`.

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

### [`libs/sinwhisky`](libs/sinwhisky) (git submodule)

SinWhisky by the owner, upstream `https://github.com/cekkr/sinwhisky.py`, branch `main`: a circle-based 1-D resampler (circle through every three consecutive samples, cosine-window blend; C99 core in `sinwhisky_c/`, scalar Python reference `v2/whisky.py`). v11 uses the vectorized `v2/whisky_np.py::sinwhisky_resample_np` (added for step 8e; identical results to the reference, `periodic=True` mode). Own handbook: [`libs/sinwhisky/AGENTS.md`](libs/sinwhisky/AGENTS.md). Commits made from this repository go on its `main` and are pushed only when the owner asks (`4fa78f3`, the vectorized resampler, is on upstream `main`).

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

### Spectral zeros read as new factorization information

- **Symptom:** features counting exactly vanishing DFT bins of the digit sequence separate primes from composites and add ω accuracy over a 19-base length baseline (+0.75 points at 10⁵–10⁶, `scale-s0-step8`).
- **Cause:** zeros come in Galois classes k | L and are explained by characters: even extensions (Midy) and Euler factors (1 − χ*(p)) for primes p | n (`zeros.py` docstring, `studies/9` Proposition 1.5). They are real digit information, but the orders in 98 bases already carry it (−0.05 points over L98 + Midy98).
- **Safe pattern:** detect zeros class-wise (not bin-wise), check against the exact Φ_k test, and compare against the step-7 baseline.

### Training-set accuracy reported as model accuracy

- **Symptom:** "98.8% accuracy" in the paper (§3.1–3.2).
- **Cause:** v8 classifier evaluates on the same tensors it trains on, with `is_prime_power` (which alone identifies ω = 1) among three inputs; the paper also calls it a regression with MSE.
- **Safe pattern:** held-out test split, majority-class and `log n`/`log L` baselines, no factorization-derived inputs.
- **Status:** the paper claim is unverified.

### Correlations driven by `n` alone

- **Symptom:** r = 0.946 for `H·log n` (§4.2) and decreasing correlation with `n^-s` (§4.1).
- **Cause:** both covariates are monotone functions of `n`, and H grows with `n` through `L`. They measure size, not primes or zeta.
- **Safe pattern:** report partial correlations controlling for `log n` and `log L`; compare against the same statistic on a shuffled-ω null.
- **Status:** v11 records the weak size-controlled correlations in revision 9. The planned prime-counting benchmarks use explicit index divisibility and residue classes.

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

Run the v11 tests from the repo root (104 tests incl. figures, ~3 min idle, up to 8 min next to a background run — the end-to-end claims test dominates; the reproduction tests need the stored v8/v10 results):

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

Run the step-8 analysis (reuses the scale dataset cache; FFT zero classes of every pair, then 5-seed C9 models; run in the background):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/scale.json --run-id scale-s0-step8 --stages data,claims,zeros,report
```

Run the step-8e SinWhisky spectra analysis (single process, stratified pair sample):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/scale.json --run-id scale-s0-step8e --stages data,claims,whisky,report
```

Run step 9 (9a reuses the scale dataset, lengths and zeros caches; 9b samples its own integers up to 10⁸ — run in the background):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/scale.json --run-id scale-s0-step9 --stages data,euler,report
```

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/orders.json --run-id orders-s0 --stages orders,report
```

Run step 10/11 (same integers as `orders-s0`, orders in bases 2..499; the current `tworank` block is step 11's model list — `tworank-s0` used the step-10 defaults) and step 11c (`--config …/scale.json --run-id scale-s0-step11 --stages data,euler,report`):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/orders.json --run-id step11-s0 --stages tworank,report
```

Run step 20 (full-period repetends to 10⁸ and the Evolvo cases; ~5 min) and step 21 (L-function zeros to height 150 and the explicit formula; ~10 min):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/artin.json --run-id artin-s0 --stages artin,report
```

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/explicit.json --run-id explicit-s0 --stages explicit,report
```

Run steps 17–19 (prime counts and Evolvo formula search, no digit dataset):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/prime-orders.json --run-id prime-orders-s0 --stages prime_orders,report
```

Resume an existing run (finished stages are skipped; config and seed must match):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-s0 --resume
```

Run step 16 (the final decoder on integers built from the external prime list; download it first into the git-ignored cache):

```bash
mkdir -p empirical-scripts/11.0/results/primes-db/cache/txt && curl -L -o /tmp/primes50.zip https://t5k.org/lists/small/millions/primes50.zip && unzip -o /tmp/primes50.zip -d empirical-scripts/11.0/results/primes-db/cache/txt
```

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/orders.json --run-id validate-s0 --stages validate,report
```

Run step 15 (failure set of the decoder: orders of the 19 bases modulo every prime ≤ 10⁸, prefix checks with brute force; ~6 min):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/orders.json --run-id failures-s0 --stages failures,report
```

Run step 14 (exact membership and residues; reuses the scale dataset, lengths and zeros caches):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/scale.json --run-id scale-s0-step14 --stages data,subgroup,report
```

Run step 13 (exact decoding: every n ≤ 10⁸ and ω-balanced decades to 10¹²; 78 min on 7 workers, 72 of them for the exhaustive range; run in the background):

```bash
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/orders.json --run-id decode-s0 --stages decode,report
```

The full config's data build takes ~20 min on 7 workers and its other stages longer; run it in the background. A new run id with an unchanged data/split config reuses an existing dataset cache.

Long runs: v10 ran perpetually with multi-GB memory. Run v11 full-scale jobs in the background with a resumable state file; use a small `--quick` config for verification.

## Test Ownership Map

- Evolvo library → [`libs/evolvo/tests/test_gfsl_core.py`](libs/evolvo/tests/test_gfsl_core.py) (GFSL core), [`libs/evolvo/tests/test_policy_guide.py`](libs/evolvo/tests/test_policy_guide.py), [`libs/evolvo/tests/test_nnarch.py`](libs/evolvo/tests/test_nnarch.py) (GFSL-NN). Functions, lists and Kompute have only smoke demos.
- v11 `ndr.py` → [`empirical-scripts/11.0/tests/test_ndr.py`](empirical-scripts/11.0/tests/test_ndr.py); `dataset.py` → [`empirical-scripts/11.0/tests/test_dataset.py`](empirical-scripts/11.0/tests/test_dataset.py); `claims.py`, `report.py` → [`empirical-scripts/11.0/tests/test_claims.py`](empirical-scripts/11.0/tests/test_claims.py); `features.py`, `search_formulas.py`, `baselines.py`, `search_nn.py` → [`empirical-scripts/11.0/tests/test_search.py`](empirical-scripts/11.0/tests/test_search.py); `scale.py` and multi-range sampling → [`empirical-scripts/11.0/tests/test_scale.py`](empirical-scripts/11.0/tests/test_scale.py); `lengths.py` (Midy ⇔ complementary halves ⇔ vanishing even DFT bins, the 2-adic characterization for odd n, order matrix, lcm of all orders = λ, length-arithmetic parity, stage end-to-end incl. reproduction of the scale stage's C9) → [`empirical-scripts/11.0/tests/test_lengths.py`](empirical-scripts/11.0/tests/test_lengths.py); `zeros.py` (digit/remainder DFT identity, exact Φ_k classes = FFT classes for n < 700, character expansion and Euler factors for n = p·q, stage end-to-end) and `whisky_spectra.py` (linear control keeps zeros, Midy coefficient, report) → [`empirical-scripts/11.0/tests/test_zeros.py`](empirical-scripts/11.0/tests/test_zeros.py); `euler.py` (membership = subgroup membership for cyclic groups, stage) and `orders_scale.py` (vectorized ω and trial-division factorization vs SymPy, stage) → [`empirical-scripts/11.0/tests/test_step9.py`](empirical-scripts/11.0/tests/test_step9.py); `tworank.py` (Proposition 1.6 exactly on all units of (ℤ/nℤ)* for n < 3 000, valuations, stage) → [`empirical-scripts/11.0/tests/test_tworank.py`](empirical-scripts/11.0/tests/test_tworank.py); `decode.py` (Proposition 1.7: parts vs order-vector classes, exact ω for every n < 30 000, Shor failure ⇔ Midy, remainders from digits and long division, exhaustive and stage end-to-end) → [`empirical-scripts/11.0/tests/test_decode.py`](empirical-scripts/11.0/tests/test_decode.py); `subgroup.py` (exact membership = remainder set for n < 2 500, divisibility differs in non-cyclic groups, cyclic test, stage) → [`empirical-scripts/11.0/tests/test_subgroup.py`](empirical-scripts/11.0/tests/test_subgroup.py); `failures.py` (Proposition 1.8: failure iff colliding pair for every n < 25 000 and three base sets, certificate = brute-force first failure for 1–8 bases, collisions, `decodes`) → [`empirical-scripts/11.0/tests/test_failures.py`](empirical-scripts/11.0/tests/test_failures.py); `sieve_echo_decoder.py` (exact factorization for every n < 20 000 vs SymPy, long-division lengths = orders, the prefix failures decoded with all 19 bases, an unsplit pair reported as one factor) → [`empirical-scripts/11.0/tests/test_sieve_echo_decoder.py`](empirical-scripts/11.0/tests/test_sieve_echo_decoder.py); `validate.py` (lengths from factors = orders, length verification, `primes_of` on large lengths vs SymPy, stage end-to-end on a small generated list) → [`empirical-scripts/11.0/tests/test_validate.py`](empirical-scripts/11.0/tests/test_validate.py). `prime_orders.py`, `prime_order_formulas.py` → [`empirical-scripts/11.0/tests/test_prime_orders.py`](empirical-scripts/11.0/tests/test_prime_orders.py); `artin.py`, `artin_formulas.py` → [`empirical-scripts/11.0/tests/test_artin.py`](empirical-scripts/11.0/tests/test_artin.py); `explicit.py` → [`empirical-scripts/11.0/tests/test_explicit.py`](empirical-scripts/11.0/tests/test_explicit.py). SinWhisky's own suite (`cd libs/sinwhisky && ../../.venv/bin/python -m unittest discover -s tests`, 21 tests incl. the vectorized/periodic cross-validation) owns `libs/sinwhisky`. The entry script has no unit test; it is verified by the quick run. Older version folders have no tests.

## Data, Security, Privacy, and Compatibility Boundaries

- Canonical data is computed from integers; everything in `results/` is derived and can be regenerated. Caches are disposable.
- No secrets, credentials, or network services are involved. Do not write the owner's email or personal data into new files.
- Result files written by v11 MUST record the NDR/entropy definition version, root and submodule git SHAs, and seed, so old results stay interpretable after definitions change.

## Current Status and Known Gaps

### Shipped

- The `prime_orders` stage implements the verified steps 17–19, including independent prime counts, GRH-conditional comparison, finite input audits, and executable GFSL programs. The [run records](ROADMAP.md#prime-order-benchmarks) identify the scripts, settings and seeds.
- The `artin` and `explicit` stages implement steps 20–21 (runs `artin-s0`, `explicit-s0`): full-period counts against Hooley's densities with the Lehmer–Heilbronn correction, the index law, four Evolvo cases, L-function zeros, the smoothed explicit formula, the Rubinstein–Sarnak bias and the Schoenfeld ratio.
- Historical scripts v3.1–v10 and their stored results (exploratory, see pitfalls).
- Evolvo submodule at [`libs/evolvo`](libs/evolvo) with the step-3 features (see [Standing Evolvo Facts](#standing-evolvo-facts-relevant-to-v11)).

- v11 steps 0–5 and 7–21 ([`ROADMAP.md`](ROADMAP.md); conclusions in [`empirical-scripts/11.0/conclusions.md`](empirical-scripts/11.0/conclusions.md)): entry script, manifest, canonical `ndr-v11.1` definitions, dataset with split and cache, C1–C9 tests with reproduction of every stored v8.2 statistic, Evolvo formula search (4 guides × 5 seeds, S1 adaptive evaluation size, S2 RL guide with warm starts), held-out baselines, §3.1 retest, GFSL-NN search, scale analysis to 10⁶ with growth exponents, step-7 length/Midy/λ analysis (`lengths` stage), step-8 exact spectral zeros (`zeros`) and SinWhisky spectra (`whisky`, submodule `libs/sinwhisky`), step-9 small-prime membership (`euler`) and order-only ω to 10⁸ (`orders`), step-10/11 2-rank estimators (ratio and maximum likelihood) and ℓ-adic profiles (`tworank`), step-13 order tests for recovering ω (`decode`; exact for every n ≤ 10⁸ with pre-periods, run `decode-s0`), step-14 exact subgroup membership and the residue structure accounted for (`subgroup`, run `scale-s0-step14`), step-15 failure set of the order tests (`failures`; exact ω for every n < 3.7·10⁹, run `failures-s0`), step-16 external validation of the standalone decoder on integers built from the t5k.org prime list (`validate`; 40 000/40 000 exact up to 2.4·10⁴⁸, run `validate-s0`), Markdown/JSON report with figures, the standalone decoder `sieve_echo_decoder.py`, steps 17–21 (prime-order counts, Artin densities and L-function zeros: `prime_orders`, `artin`, `explicit`), 104 unit tests, `.venv` with torch. Verdicts and their runs are in [`ROADMAP.md` Results log](ROADMAP.md#results-log).

### Experimental / Scaffold

- Every n up to 10⁶ was not built (≈ 15 h); steps 5 and 7 use ω-stratified decades instead. The `studies/9` draft (now including steps 7–11) awaits owner review.

### Known Gaps

- Paper numbers (§2–§4) come from one v8.2 run (plus a v7.0 log for "98.8%" and "r = -0.843") with leakage, training-set accuracy, 100-row samples and size-driven correlations. v11 reproduces every stored v8.2 statistic and has corrected verdicts for C1–C9 (ROADMAP Results log); the §3.1 three-feature claim is retested in step 4.
- Paper direction-of-effect statements contradict each other and the data.

### Near-Term Priorities

1. Review the [prime-order benchmark results](ROADMAP.md#next-research-joint-distribution-of-repetend-lengths), steps 20–21 and the [revision-9 draft](studies/9/sieve-echo-conjecture-9.md) against the stored runs.
2. Any further counting experiment needs a stated observable, prime range, arithmetic comparison and explicit decision criterion. Claims about RH must follow [`conclusion.md` §10](conclusion.md#10-what-this-means-for-the-riemann-hypothesis).

## Task Start and Handoff Checklist

1. Read this file, the relevant [`ROADMAP.md`](ROADMAP.md) step, and [`libs/evolvo/AGENTS.md`](libs/evolvo/AGENTS.md) when touching Evolvo.
2. `git status` in the root and in `libs/evolvo`; preserve unrelated changes.
3. Check the [principles](#essential-project-principles): no factorization-derived inputs, held-out evaluation, `log L` control, versioned definitions.
4. Put library fixes in `libs/evolvo` (D3), with a test; commit there on `main`, then commit the pointer here.
5. Run the focused tests and a `--quick` v11 run; report what ran and what did not.
6. Update [`ROADMAP.md`](ROADMAP.md) status and this file's source map, commands, and gaps in the same change.
