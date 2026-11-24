# Sieve Echo Conjecture — v11 Roadmap

Live plan for **version 11** (`empirical-scripts/11.0/`): a single reproducible script that tests every claim of [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md) and uses the Evolvo submodule [`libs/evolvo`](libs/evolvo) for all evolutionary search. Rules and context are in [`AGENTS.md`](AGENTS.md).

Update this file while implementing (directive D4): change a step's status only after its "Done when" check passes, add discovered work under the right step or under [Discovered work](#discovered-work), and record results with the run id that produced them.

Status labels: `TODO`, `IN PROGRESS`, `DONE (verified: …)`, `BLOCKED (reason)`.

---

## What v11 keeps from previous versions

| From | Keep | Drop / fix |
| --- | --- | --- |
| v8.0 `sieve-echo-tests-1.py` | Claim list (base invariance, zeta, PNT, modular, sieve-echo law fit, constant matching), `d/b` normalization, power spectrum | Training-set accuracy; `is_prime_power` as input; correlations without `log n` control |
| v10 `sieve-echo-tests-3.py` | Cycle structure over growing `n`, state resume, relative save threshold (0.75 × top-10% mean) and top-5 safety net, multi-base aggregation (mean/std/CV) | Amplitude spectrum; `phi`/`tau` as inputs; fitness > 1 with variety bonus; random 200-row resample per evaluation; O(n) long division |
| v10 `gold_apart.py` | Immediate save with signature dedupe | Per-generation dump of every algorithm |
| v7 `conclusions.md` | Feature ideas: cross-base std, min entropy, growth exponents | "three-feature sufficiency" until re-tested without leakage |

## Planned layout (provisional names)

```
empirical-scripts/11.0/
  sieve_echo_v11.py      # entry point: CLI, config, stages, run manifest
  config/quick.json      # minutes-scale smoke config
  config/full.json       # paper-scale config
  ndr.py                 # number theory + NDR + spectra (canonical, versioned)
  dataset.py             # feature table build, cache, split by n
  claims.py              # statistical tests of paper claims with baselines + nulls
  search_formulas.py     # evolvo.GFSLEvolver symbolic search
  search_nn.py           # evolvo.nnarch GFSL-NN search
  report.py              # JSON + Markdown verdict table + figures
  tests/                 # unit tests for ndr.py / dataset.py / claims.py
  results/<run-id>/      # run outputs (cache/ subfolder git-ignored)
```

Split into modules only where it keeps files readable; a single script is acceptable if it stays under ~1 500 lines. Update the layout here and in [`AGENTS.md`](AGENTS.md) when files are created.

---

## Step 0 — Environment and skeleton — TODO

- [ ] Choose interpreter: check PyTorch wheel availability for Python 3.15 (`/opt/homebrew/bin/python3`); otherwise install/use 3.12 or 3.13.
- [ ] Create `.venv` at the repo root (already ignored by [`.gitignore`](.gitignore)); install `numpy scipy sympy scikit-learn matplotlib torch psutil`; write `empirical-scripts/11.0/requirements.txt` with the versions actually installed.
- [ ] Entry point with `--config`, `--seed`, `--run-id`, `--resume`, `--stages` (`data,claims,formulas,nn,report`).
- [ ] Bootstrap `libs/evolvo/src` onto `sys.path`; fail with a clear message if the submodule is missing (`git submodule update --init`).
- [ ] Run manifest: root SHA, `libs/evolvo` SHA, dirty flags, Python + package versions, config, seed, NDR definition version.
- [ ] Add `empirical-scripts/11.0/results/*/cache/` to `.gitignore`.
- **Done when:** `--config config/quick.json --stages data` runs end to end and writes a manifest; verified commands copied into `AGENTS.md`.

## Step 1 — Canonical NDR dataset — TODO

- [ ] `multiplicative_order(b, n)`: from the factorization of λ(n), exact integers; only for `gcd(b, n) = 1`.
- [ ] `repetend_digits(n, b)`: exactly `L` digits via `r_{k+1} = b·r_k mod n`, `d_k = floor(b·r_k / n)`, vectorized in chunks (NumPy int64 is safe while `b·n < 2^63`).
- [ ] `ndr(digits, b) = digits / b` (paper Def. 1.1).
- [ ] `spectral_entropy(x, variant)`: variant `paper` = power over all `L` bins (Def. 1.2); variant `nodc` = mean removed; optional `onesided`. Natural log. Also report the normalized form `H / log(L)` so lengths are comparable.
- [ ] Per-(n, b) features: `L`, `log L`, `L/λ(n)`, `L/φ(n)` (target-side only, see contracts), H variants, normalized H, digit mean/std/skew/kurtosis, spectral flatness, top-k peak frequencies.
- [ ] Cross-base aggregates: mean, std, min, max, CV of each feature; base sets: v8's 13 bases, v10's 16 bases, primes ≤ 31, and a wide `2..99` set for invariance (paper §6.1 priority 1).
- [ ] Labels (targets only): ω, Ω, τ, φ, λ, μ, is_prime, is_prime_power, squarefree.
- [ ] Sampling: all `n` in a range plus stratified samples by ω for large `n` (ω ≥ 5 is rare below 10⁵); record the sampling scheme.
- [ ] Split by `n` (train/val/test), seeded, saved in the manifest; disk cache (Parquet or `.npz`).
- [ ] Unit tests: `1/7` base 10 → `142857`; `L = ord_n(b)` against brute-force long division for `n < 2000`; entropy invariance under `d/b` vs `d/(b-1)`; constant sequence → entropy of DC-only spectrum = 0; v8 entropy reproduced for a few stored `n` from [`8.0/results/1`](empirical-scripts/8.0/results/1) and v10 values from [`sieve_echo_state_v10.json`](empirical-scripts/results/3-Sept-2025/sieve_echo_state_v10.json) using their definitions.
- **Done when:** tests pass; dataset for `n ≤ 10⁴` builds in minutes; old-version values reproduced or the difference explained.

## Step 2 — Claim-by-claim statistical verdicts — TODO

Each claim gets: the paper's statistic recomputed on the same range (`n ∈ [4, 9999]`, v8 bases) to check reproduction, then the corrected test, a baseline, a null, a bootstrap CI, and a verdict (`reproduced`, `reproduced but explained by size/length`, `not reproduced`, `supported`).

- [ ] **C1 Sieve Echo Law (§2.2):** OLS `H ~ log ω`; then `H ~ log ω + log L + log n`; report partial effect of `log ω`, R² gain over the `log L + log n` baseline; non-linear forms from §6.1 priority 2 (`A·ω^-B + C`, polynomial in `log ω`).
- [ ] **C2 Direction:** state the measured sign for primes vs composites (paper text contradicts itself; see AGENTS.md).
- [ ] **C3 Prime-power dichotomy (§2.3):** point-biserial r with and without `log L` control.
- [ ] **C4 Base invariance (§2.4, Q5.2):** rate of CV < 0.1 vs `n` scale (10³…10⁶) and number of bases; test the "emerges asymptotically" hypothesis; compare normalized `H/log L`.
- [ ] **C5 Zeta (§4.1):** correlation with `n^-s` vs the same with `log n` partialled out; shuffled-ω null.
- [ ] **C6 PNT (§4.2):** `H·log n` correlation vs `log n` alone; show what remains.
- [ ] **C7 Modular structure (§4.3):** residue-class means mod 6/30/210 with ANOVA/permutation p-values, controlling for ω and `log L` (residues mod 6 partly encode divisibility by 2 and 3).
- [ ] **C8 Constants (§5, Table 1):** report fitted α, β with CIs; drop constant-matching ("tribonacci − ln 10") unless a pre-registered small constant set is used with a look-elsewhere correction.
- [ ] **C9 Information beyond L (core question):** does any NDR feature predict ω better than `L` and `n` alone? Mutual information / held-out model comparison. This is the real test of the conjecture.
- **Done when:** `report.py` prints a verdict table for C1–C9 on the quick and full configs.

## Step 3 — Evolvo fixes needed by v11 (in `libs/evolvo`, commit on its `main`) — TODO

Do these in the submodule per directive D3, each with a test in `libs/evolvo/tests/` and an update to `libs/evolvo/AGENTS.md` and `README.md`.

- [ ] **E0 Verify inputs:** confirm how random genomes reference input variables `d$0..d$k` (no input count in `GFSLEvolver.initialize_population`). If unsupported, add an explicit input declaration (e.g. `input_count`/`declare_inputs`) honoured by the validator, mutation, and `_make_random_genome`.
- [ ] **E1 Batched execution:** a vectorized execution path (`execute_batch(genome, {"d$0": np.ndarray, …})`) for decimal/boolean ops, so one genome is evaluated over all training rows at once; keep scalar semantics identical (test equality on random genomes).
- [ ] **E2 Protected numerics:** confirm NaN/inf handling of `DIV`, `LOG`, `SQRT`, `EXP`, `POW` in batch mode matches scalar mode.
- [ ] **E3 Domain ops via custom registry:** verify `register_custom_operation` works with batch mode (for `LOG`, `GCD`-free ops — no factorization ops, by contract).
- [ ] **E4 GFSL-NN tabular inputs:** confirm `ArchSpace([("vec", k)], n_out, …)` + `TorchTask` support classification (ω as class) or add a cross-entropy loss option.
- [ ] **E5 Core tests:** add GFSL core unit tests (executor, evolver smoke) — Evolvo currently tests only `nnarch`.
- [ ] **E6 Evaluation-size hook:** let `GFSLEvolver.evolve` invalidate and re-evaluate fitness when the caller signals a new evaluation size (e.g. a `fitness_epoch` counter or an `on_resize` callback), so the elite is not compared on scores from different sample sizes. Needed by [S1](#s1--adaptive-evaluation-size-5-000--100-000).
- [ ] **E7 Torch RL guide:** a PyTorch policy-gradient guide (e.g. `GFSLPolicyGuide` in `src/evolvo/`) next to `GFSLQLearningGuide` and `GFSLSupervisedGuide`, with save/load of its weights. Needed by [S2](#s2--torch-reinforcement-learning-guide). Check first whether `GFSLSupervisedGuide` can be extended instead of adding a class.
- **Done when:** Evolvo tests pass; root repo commits the new submodule pointer.

## Step 4 — Evolutionary search with Evolvo — TODO

- [ ] **Formulas:** `GFSLEvolver` predicting ω (and separately `is_prime`) from allowed features only; fitness = train-minibatch loss (fixed per generation) + explicit parsimony; validation for selection; test once at the end. Keep v10's relative save threshold and safety net.
- [ ] **Guidance:** compare plain evolution vs `GFSLQLearningGuide` vs `GFSLSupervisedGuide` on the same budget and seed set.
- [ ] **GFSL-NN:** `evolvo.nnarch.ArchitectureSearch` on the feature table (vector input) and on raw NDR sequences (set input / `deepsets`, `transformer` macros), with `Knowledge` saved per run for reuse.
- [ ] **Baselines:** majority class, logistic/GBM on `(log n, log L)` only, same on full NDR features. Report held-out accuracy, macro-F1, and MSE for ω.
- [ ] **Three-feature claim (§3.1):** retest (std H, min H) without `is_prime_power`; report against baselines.
- [ ] **Adaptive evaluation size** — see [S1](#s1--adaptive-evaluation-size-5-000--100-000) (required).
- [ ] **Torch reinforcement learning guide** — see [S2](#s2--torch-reinforcement-learning-guide) (required).
- **Done when:** held-out results table with baselines; best formulas saved in human-readable form.

### S1 — Adaptive evaluation size (5 000 → 100 000)

Status: TODO

How many integers each genome is scored on grows step by step, driven by the average of the best scores, so precision rises as the population gets better. The rule is a requirement set by the owner.

- **Range:** start at `N_min = 5 000` integers per evaluation; never exceed `N_max = 100 000`. Both are config keys (`eval_size_min`, `eval_size_max`).
- **Driver:** `q = mean fitness of the top-k genomes` (k = top 10% by default, the same "average best scores" v10 used for saving). Map it to size by `N = N_min + (N_max − N_min) · clip((q − q_0) / (q_target − q_0), 0, 1)`, where `q_0` is the top-k mean in generation 0 and `q_target` a config value. Round `N` to steps (e.g. multiples of 5 000) so size does not change every generation.
- **Statistical trigger (extra):** also step up when the spread among the top-k is smaller than about 2 standard errors of the fitness estimate at the current `N` (SE ∝ 1/√N): at that point the current size can no longer rank the best genomes.
- **Monotone:** `N` never decreases within a run.
- **Nested samples:** take the evaluation rows as a prefix of one seeded permutation of the **train** split, so the rows used at size N are always part of the rows used at any larger size. Validation and test splits are not used.
- **Comparable scores:** when `N` changes, re-score the elite and the hall-of-fame at the new `N` before selection (needs Evolvo E6). Store `N` next to every saved fitness; the relative save threshold (0.75 × top-10% mean) compares only scores taken at the same `N`.
- **Data requirement:** the train split must hold at least 100 000 rows, so the integer range for formula search is at least about 1.5·10⁵ with a 70/15/15 split. Step 1 must build it (stratified by ω).
- **Cost:** needs batched execution (E1); without it 100 000 rows per genome is not feasible.
- **Log:** write `N`, `q`, the top-k spread and the SE per generation to the run directory.
- **Done when:** a quick run shows `N` rising from 5 000 as `q` improves, elite scores re-taken at each step, and held-out error at the end lower or equal compared with a fixed `N = 5 000` run on the same budget.

### S2 — Torch reinforcement learning guide

Status: TODO

Reinforcement learning with PyTorch is **required** (owner decision): it learns over time which genome edits work and reduces the spread of results, measured relative to the average best scores. Torch therefore becomes a hard dependency of v11 (step 0 must install it).

- **Owner:** the guide lives in Evolvo (E7); v11 only configures it and stores its weights under `results/<run-id>/` and in a reusable knowledge file across runs (like `evolvo.nnarch.Knowledge`).
- **Actions:** the policy chooses the mutation operator and slot values (operation, source categories, input indices), conditioned on genome features from `GFSLFeatureExtractor`.
- **Reward:** relative to the average best scores, `r = (f_child − q) / (|q| + ε)`, where `q` is the current top-k mean at the same evaluation size `N`. Rewards are recomputed when `N` changes (S1), so learning never mixes scales.
- **Learning:** policy gradient (REINFORCE with a baseline, or actor-critic) on batches of offspring per generation; AdamW; device from `resolve_torch_accelerator` (`mps` on this Mac when available).
- **Less dispersion over time:** an entropy bonus or softmax temperature that falls as `q / q_target` rises; keep a uniform floor (as `evolvo.nnarch.AdaptivePolicy` does) so the search does not freeze.
- **Persistence:** save and load policy weights so later runs start from what earlier runs learned ("best practice over time"); record which knowledge file a run started from in the manifest.
- **Comparison:** same budget and seeds for no guide, `GFSLQLearningGuide`, `GFSLSupervisedGuide` and the RL guide. Report the mean and the spread across seeds of held-out error and of generations to target.
- **Done when:** across ≥ 5 seeds the RL guide reaches the target in fewer evaluations, or with smaller spread, than plain evolution, and a run that loads saved weights does better than a cold start. If it does not, record the result here and keep the guide configurable.

## Step 5 — Scale-up and asymptotics — TODO

- [ ] Run full config to `n ≈ 10⁶` with stratified ω sampling and the wide base set; background job, resumable state.
- [ ] Growth exponents (H ~ n^a) recomputed with `log L` control (paper reports 0.142 vs predicted 0.743).
- **Done when:** verdicts stable between `10⁵` and `10⁶`, or the drift is reported.

## Step 6 — Report and paper update — TODO

- [ ] `results/<run-id>/report.md` + `report.json` + figures; commit the small summary files only.
- [ ] Draft `studies/9/` (or revise `studies/8/`) with verified numbers, each citing the run id; fix the direction contradictions; replace "Theorem" with "Observation" for empirical fits.
- [ ] Update [`AGENTS.md`](AGENTS.md) (source map, commands, status) and this roadmap.
- **Done when:** owner reviews the report.

---

## Discovered work

Add items found during implementation here, with the step they affect.

- (none yet)

## Results log

Record verified results as `run-id — config — headline numbers — verdict`. Empty until step 2 produces a run.
