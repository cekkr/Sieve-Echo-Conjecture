# Sieve Echo Conjecture — v11 Roadmap

Live plan for **version 11** (`empirical-scripts/11.0/`): a single reproducible script that tests every claim of [`studies/8/sieve-echo-conjecture-8.md`](studies/8/sieve-echo-conjecture-8.md) and uses the Evolvo submodule [`libs/evolvo`](libs/evolvo) for all evolutionary search. Rules and context are in [`AGENTS.md`](AGENTS.md).

Update this file while implementing (directive D4): change a step's status only after its "Done when" check passes, add discovered work under the right step or under [Discovered work](#discovered-work), and record results with the run id that produced them.

Status labels: `TODO`, `IN PROGRESS`, `DONE (verified: …)`, `BLOCKED (reason)`.

---

## What v11 keeps from previous versions

| From | Keep | Drop / fix |
| --- | --- | --- |
| v8.2 `sieve-echo-tests-2.py` (produced the paper's v8 numbers) | Claim list (base invariance, zeta, PNT, modular, sieve-echo law fit, constant matching), `d/b` normalization, power spectrum | Training-set accuracy; `is_prime_power` as input; correlations without `log n` control |
| v10 `sieve-echo-tests-3.py` | Cycle structure over growing `n`, state resume, relative save threshold (0.75 × top-10% mean) and top-5 safety net, multi-base aggregation (mean/std/CV) | Amplitude spectrum; `phi`/`tau` as inputs; fitness > 1 with variety bonus; random 200-row resample per evaluation; O(n) long division |
| v10 `gold_apart.py` | Immediate save with signature dedupe | Per-generation dump of every algorithm |
| v7 `conclusions.md` | Feature ideas: cross-base std, min entropy, growth exponents | "three-feature sufficiency" until re-tested without leakage |

## Layout

```
empirical-scripts/11.0/
  sieve_echo_v11.py      # entry point: CLI, config, stages, run manifest            (exists)
  runtime.py             # Evolvo bootstrap, git/env info, JSON + hashing helpers    (exists)
  config/quick.json      # minutes-scale smoke config                                (exists)
  config/full.json       # paper-scale config                                        (exists)
  requirements.txt       # pinned versions of the .venv actually used                (exists)
  ndr.py                 # number theory + NDR + spectra (canonical, versioned)      (exists)
  dataset.py             # feature table build, cache, split by n                    (exists)
  claims.py              # statistical tests of paper claims with baselines + nulls  (exists)
  search_formulas.py     # evolvo.GFSLEvolver symbolic search                        (planned, step 4)
  search_nn.py           # evolvo.nnarch GFSL-NN search                              (planned, step 4)
  report.py              # JSON + Markdown verdict table (figures planned, step 6)   (exists)
  tests/                 # test_ndr.py, test_dataset.py, test_claims.py              (exists)
  results/<run-id>/      # run outputs (cache/ subfolder git-ignored)
```

Split into modules only where it keeps files readable; a single script is acceptable if it stays under ~1 500 lines. Update the layout here and in [`AGENTS.md`](AGENTS.md) when files are created.

---

## Step 0 — Environment and skeleton — DONE (verified: run `quick-s0-verify`, 2026-11-24)

- [x] Interpreter: PyTorch 2.15.0 publishes macOS arm64 wheels for Python 3.15, so v11 uses `/opt/homebrew/bin/python3.15` (3.15.2). torch reports `mps` available.
- [x] `.venv` at the repo root; `pip install --only-binary :all: torch numpy scipy scikit-learn sympy matplotlib psutil`; pinned in [`empirical-scripts/11.0/requirements.txt`](empirical-scripts/11.0/requirements.txt).
- [x] Entry point [`sieve_echo_v11.py`](empirical-scripts/11.0/sieve_echo_v11.py) with `--config`, `--seed`, `--run-id`, `--resume`, `--stages`, `--results-dir`. Unimplemented stages exit with the roadmap step that will add them. `--resume` refuses a changed config or seed; an existing run dir without `--resume` is refused.
- [x] `runtime.bootstrap_evolvo()` puts `libs/evolvo/src` first on `sys.path`, imports `evolvo`, and fails if the submodule is missing or `evolvo` resolves elsewhere.
- [x] `manifest.json`: root and Evolvo SHAs, dirty flags with the changed/untracked paths, a SHA-256 of the v11 sources (`v11_code_sha256`), Python, platform, package versions, torch devices, config + its SHA-256, seed, NDR version, per-stage status and timings. Paths are repo-relative.
- [x] `.gitignore` ignores `empirical-scripts/11.0/results/*/cache/`.
- **Verified:** `--config empirical-scripts/11.0/config/quick.json` (stage `data`) ran end to end and wrote the manifest; `--resume` skipped the finished stage; Evolvo's own tests pass in the venv (14 tests, torch case no longer skipped).

## Step 1 — Canonical NDR dataset — DONE (verified: 20 unit tests + run `quick-s0-verify`, 2026-11-24)

- [x] `ndr.multiplicative_order(b, n, λ, primes(λ))`: exact, from λ(n) computed from a smallest-prime-factor sieve; raises for `gcd(b, n) ≠ 1`.
- [x] `ndr.repetend_digits(n, b, L)`: remainders `b^k mod n` by vector doubling (log L NumPy steps), digits `floor(b·r_k / n)`; int64-safe for `n < 3·10^9`.
- [x] `ndr.ndr(digits, b) = digits / b`.
- [x] `ndr.spectral_entropy(x, variant)`: `paper` (power over all L bins, DC kept — Def. 1.2), `nodc` (bins 1..L-1), plus `v8` and `v10` for reproduction. `onesided` dropped: for real input the full spectrum is the conjugate-symmetric extension of the one-sided one, so it adds nothing. Normalized `Hn_paper = H/log L`, `Hn_nodc = H/log(L-1)`. Version tag `ndr-v11.1`.
- [x] Per-(n, b) features (`ndr.PAIR_FEATURES`): L, log L, H_paper, H_nodc, Hn_paper, Hn_nodc, NDR mean/std/skew/kurtosis, spectral flatness, top-3 peak frequencies, H_v8, H_v10. `L/λ(n)` and `L/φ(n)` are stored under `target_side.*` (label-derived, never inputs).
- [x] Cross-base aggregates mean/std/min/max/CV per base set: `v8` (13), `v10` (16), `primes31` (11), `wide` (2..99, sub-sampled: 500 integers in quick, 5 000 in full).
- [x] Labels `label.*`: ω, Ω, τ, φ, λ, μ, is_prime, is_prime_power, squarefree. `dataset.allowed_input_columns()` excludes `label.*`, `target_side.*`, `pair.*`, `split`.
- [x] Sampling: all `n` in `[n_min, n_max]` plus a seeded sample of `per_omega` integers per ω from `[n_lo, n_hi]`; the scheme with available/taken counts per ω goes to `data_summary.json`.
- [x] Split by `n`, stratified by ω, seeded (70/15/15); `split.json` in the run dir, its SHA-256 in the manifest. Cache: `cache/dataset-<key>.npz`, key = hash of data + split config, seed, NDR version, dataset format.
- [x] Unit tests ([`tests/test_ndr.py`](empirical-scripts/11.0/tests/test_ndr.py), [`tests/test_dataset.py`](empirical-scripts/11.0/tests/test_dataset.py)): `1/7 → 142857`; L vs long division for `n < 2000` × 7 bases; digits vs long division for `n < 1200`; no int64 overflow at `n = 2 000 003`; labels vs SymPy for `n ≤ 3000`; d/b vs d/(b-1) invariance for all variants; constant sequence (paper 0, nodc NaN); entropy vs direct `np.fft.fft`; `H ≤ log L`; moments vs SciPy; aggregates vs direct computation; split disjoint/stratified/deterministic; leakage guard; cache round trip.
- [x] **v8 reproduced exactly:** 400 seeded stored integers from the v8 pickle × 13 bases — `H_v8`, `L` and NDR mean equal the stored values (1e-9); the stored Sieve Echo Law (α = -0.46810, β = 1.91743, R² = 0.23447) is re-fitted from the stored per-n entropies to 1e-6.
- [x] **v10 reproduced:** `entropy_mean`, `length_mean`, `kurtosis_mean` of the 6 shortest-repetend rows of `sieve_echo_state_v10.json` (16 bases, amplitude spectrum of `d/(b-1)`, bases with L ≤ 3 skipped).
- **Verified:** quick config (9 998 integers in [2, 9999] + 525 stratified from [10⁴, 10⁵]; 160 064 coprime pairs) builds in 7.6 s with 7 worker processes (8-core machine).
- **Verified (run `full-s0`):** `config/full.json` — 162 007 integers (all n in [2, 150 000] + 2 000 per ω ≤ 6 and all 8 integers with ω = 7 from [150 001, 10⁶]), 2 340 678 coprime pairs; 1 225 s wall time on 7 workers, 1.23 GB peak RSS, 470 MB compressed cache. Train split 113 404 rows (≥ 100 000 as S1 requires).

## Step 2 — Claim-by-claim statistical verdicts — DONE (verified: runs `quick-s0`, `full-s0-claims`, 2026-11-24)

Implemented in [`claims.py`](empirical-scripts/11.0/claims.py) and [`report.py`](empirical-scripts/11.0/report.py); stages `claims` and `report` of the entry point. Each claim re-computes the paper statistic the v8.2 way (`H_v8`, v8 bases, n ∈ [2, 9999], in-sample; C4/C5 on the first 100 integers of the stored v8.2 data order), then a corrected test: parameters fitted on train, statistics on test, `log L`/`log n` controls, 95% percentile bootstrap over test rows, permutation null, effect threshold 0.01. Primary set `v8`, entropies `H_paper`, `H_nodc`, `Hn_paper`.

- [x] **C1 Sieve Echo Law:** OLS `H ~ log ω`, `H ~ log L + log n` (baseline), `H ~ log ω + log L + log n`; ΔR² over baseline with CI; bootstrap CI of the `log ω` coefficient; partial r with permutation p; `A·ω^-B + C`, poly2/poly3 in `log ω` selected on val, plus the per-ω-mean ceiling.
- [x] **C2 Direction:** primes − composites mean H, Spearman(H, ω), partial sign given `log L`, `log n`.
- [x] **C3 Prime-power dichotomy:** point-biserial r and partial r given `log L`, `log n`; reproduction shows -0.843 is corr(ω, is_prime_power).
- [x] **C4 Base invariance:** CV < 0.1 rate per base set (v8, v10, primes31, wide) × variant (`H_paper`, `H_nodc`, `Hn_paper`, `Hn_nodc`, and `logL` as baseline) by n scale (10²…10⁶); Spearman(CV, log n) with CI; "emerges asymptotically" requires CI < 0 and |ρ| ≥ 0.1.
- [x] **C5 Zeta:** r(H, n^-s), partial r given a cubic in `log n`, test R² of `H ~ n^-s` vs `H ~ log n`. The roadmap's "shuffled-ω null" does not apply (ω is not in the statistic); the size control is the null model.
- [x] **C6 PNT:** r(H·log n, H) vs the same with H permuted across n; partial r of PNT/Li errors given a cubic in `log n`.
- [x] **C7 Modular structure:** η² of residue class mod 6/12/30/210 on test, raw and after controls (per-base log L, coprimality with each base, log n, ω dummies as a labelled control), permutation p and excess over the permutation mean; for `H_paper` and `H_nodc`.
- [x] **C8 Constants:** α, β with bootstrap CIs (log ω and log(ω+1), H_v8 and canonical H) vs theory α = -1.599, β = 4.933; look-elsewhere: probability that a random value near α/β matches the v8.2 20-constant library within 0.01.
- [x] **C9 Information beyond L:** gradient-boosting ω classifier (classes 1…5+) — baseline: log n, per-base log L (19 bases), coprime counts, length arithmetic (lcm of the L_b, its relation to n - 1); + entropy columns; + all NDR columns; paired bootstrap of Δaccuracy and Δlog-loss on test; majority and logistic(log n, mean log L) baselines; 3 shuffled-label nulls.
- [x] Unit tests: [`tests/test_claims.py`](empirical-scripts/11.0/tests/test_claims.py) (helpers against known answers, length arithmetic for primes, end-to-end C1–C9 on a small dataset, report table). 29/29 v11 tests pass.
- **Verified:** both configs print the verdict table; quick claims 31 s, full claims 391 s (1.8 GB peak RSS). Every stored v8.2 statistic is reproduced (law α/β/R², 6% invariance, zeta r for s = 2, 3, 4, PNT r = 0.946 and both error correlations, all four modular variances).
- **Not done:** figures (step 6); C4 bins above 10⁴ use the ω-stratified sample, not all integers.

## Step 3 — Evolvo fixes needed by v11 (in `libs/evolvo`, commit on its `main`) — DONE (verified: Evolvo suite 33/33, 2026-11-24)

Implemented in the submodule per directive D3, each with tests in `libs/evolvo/tests/` and updates to `libs/evolvo/AGENTS.md` and `README.md`.

- [x] **E0 Verify inputs:** confirmed broken — a fresh genome can read `d$k` only after assigning it (`variable_counts` grows from targets only); the README's hand-set `validator.variable_counts` was erased by every `rebuild_validator_state()` (mutation, crossover); immigrants had no outputs. Added `GFSLGenome(inputs=...)` / `declare_inputs`, `SlotValidator(input_counts=...)`, re-applied on rebuild, kept by crossover; `GFSLEvolver.initialize_population(inputs=, outputs=, profile=)` applies them to every created genome including immigrants. Demos/README now declare their input.
- [x] **E1 Batched execution:** `GFSLExecutor.execute_batch(genome, {"d$0": array, ...}, mode="auto"|"vector"|"scalar")` with NumPy semantics in `evolvo/batch.py`; automatic scalar fallback for `CALL`/list ops. Equality with the scalar path tested on ~800 random genomes (with and without a profile) plus special values. Speed: 100 000 rows, 6 effective instructions: 2.4 ms vector vs ≈ 4.1 s scalar.
- [x] **E2 Protected numerics:** every decimal op checked on the 16×16 grid of special values (±0, ±1e308, ±inf, NaN, 1e-12, ...) against the scalar path, including Python's `OverflowError`/`ZeroDivisionError` cases of `POW` (→ 0.0).
- [x] **E3 Custom ops in batch mode:** `register_custom_operation(..., batch_function=...)`; without it (or if it raises) the batch path loops over the scalar function; tested both.
- [x] **E4 GFSL-NN classification:** the head is a plain `Linear`, so cross-entropy works; added `TorchTask.classification(...)` (long targets, validation `accuracy`) and a test (3-class toy, accuracy > 0.7). Use `ArchSpace(..., task={"loss": "ce"})`.
- [x] **E5 Core tests:** `tests/test_gfsl_core.py` (12 tests) and `tests/test_policy_guide.py` (6 tests); NumPy-less interpreters skip them.
- [x] **E6 Evaluation-size hook:** `GFSLEvolver.evolve(..., fitness_epoch=callable)` + `invalidate_fitness()`: an epoch change re-scores the whole population (elite included) and resets stagnation trackers; genomes carry `fitness_epoch`; `fitness_epoch_changes` and `evaluations` are logged.
- [x] **E7 Torch RL guide:** new `GFSLPolicyGuide` (`evolvo/policy_guide.py`) instead of extending `GFSLSupervisedGuide` (that one predicts fitness to rank candidates; a policy needs action log-probabilities). REINFORCE over the edit operator and every slot value it picks (valid options only), reward `(f − q)/|q|` against the same-generation top-10% mean, running baseline, entropy bonus, AdamW, temperature/entropy annealed by `q_target`, uniform floor, `save`/`load`. Tests: relative reward and clipping, schedule, a bandit check that the rewarded operator's probability rises, save/load, evolver plug-in.
- [x] **E8 Instruction profile (discovered):** `InstructionProfile` restricts target categories and operations; `InstructionProfile.formula()` = variable targets + decimal ops (+ custom ops). Without it, ~60% of random instructions targeted lists/functions/control flow and 125/200 random genomes needed the scalar fallback; with it, 300/300 run vectorized.
- [x] **E9 Validator op-list aliasing (discovered, pre-existing bug):** `valid_ops = DECIMAL_OPS` followed by `.extend(...)` mutated the module-level op groups on every call, leaking list ops and `CALL` into every later decimal choice and slowing option generation over a process lifetime (test file: 558 s → 2.3 s after the fix). Fixed with copies; regression test added.
- **Verified:** `libs/evolvo`: `../../.venv/bin/python -m unittest discover -s tests` → 33 tests OK; system `python3` (no NumPy/torch) → OK, 4 skipped; legacy demos (`example.py` default/function/nested/void/expression/neural/extraction/personalization) run; the README formula-search snippet runs (guide on `mps`).
- **Not done here:** guide comparisons across seeds and the S1/S2 "done when" checks — they are v11 experiments (step 4).

## Step 4 — Evolutionary search with Evolvo — DONE (verified: runs `quick-s0-step4`, `full-s0-step4`, 2026-11-24)

Implemented in [`search_formulas.py`](empirical-scripts/11.0/search_formulas.py), [`baselines.py`](empirical-scripts/11.0/baselines.py), [`search_nn.py`](empirical-scripts/11.0/search_nn.py), [`features.py`](empirical-scripts/11.0/features.py); stages `formulas` and `nn`; tests in [`tests/test_search.py`](empirical-scripts/11.0/tests/test_search.py) (36/36 v11 tests pass).

- [x] **Formulas:** `GFSLEvolver` predicting ω and, separately, is_prime from 12 allowed inputs (log n, v8 log L and entropy aggregates, NDR mean, length arithmetic), z-scored on train; fitness `1/(1+NMSE) − 0.002·effective instructions` on a train prefix; v10 relative save rule (≥ 0.75 × top-10% mean, top 5 per generation) + top-5 safety net; selection by validation NMSE; test read once. Best formulas in `results/<run>/best_formulas.md`.
- [x] **Guidance:** no guide, `GFSLQLearningGuide`, `GFSLSupervisedGuide`, `GFSLPolicyGuide`, 5 seeds each, same population/generations. (Evolvo needed an evolver interface for the Q-learning guide — added in the submodule, see Discovered work.)
- [x] **GFSL-NN:** `ArchitectureSearch` + `TorchTask.classification` on the `vec` feature table and on a `set` of per-base tokens (+ log n and length arithmetic); knowledge saved per run (`knowledge/nn_<input>.json`). Raw digit sequences are not used: their length varies up to ~10⁶ and Evolvo set inputs are fixed-size (padded sets are an Evolvo roadmap item).
- [x] **Baselines:** majority, logistic(log n, mean log L), gradient boosting on `log n + mean log L`, `n + L arithmetic`, `formula inputs`, `all allowed`; held-out accuracy, macro-F1, MSE (classes min(ω, 5)), log-loss.
- [x] **Three-feature claim (§3.1):** retested (std H, min H; + log n; + `is_prime_power` as a labelled leakage demo).
- [x] **Adaptive evaluation size** — S1 implemented and run (below).
- [x] **Torch reinforcement learning guide** — S2 implemented and run (below).
- **Verified:** held-out table and best formulas produced for quick and full (Results log). Full run: 3 198 s wall (claims 402 s, formulas 624 s for 62 runs on 7 workers, baselines ~2 min, GFSL-NN 2 168 s), peak RSS 9.2 GB in the main process.

### S1 — Adaptive evaluation size (5 000 → 100 000)

Status: DONE — implemented and verified; benefit not significant (runs `quick-s0-step4`, `full-s0-step4`). N rises from 5 000 as q improves (full, ω, no guide, seed 0: 5 000 → 30 000 at generation 2 → 60 000 at 6 → 100 000), every N change re-scores the whole population (`fitness_epoch`), N is stored with each saved fitness. Held-out (full, no guide, 5 seeds): ω test NMSE 0.3087 ± 0.0088 adaptive vs 0.3149 ± 0.0000 fixed N = 5 000 (lower, Mann–Whitney p = 0.18); is_prime 0.00141 vs 0.00139 (equal within noise, p = 0.50). The done-when condition holds for ω and is a tie for is_prime.

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

Status: DONE — implemented and run; the advantage is not established (runs `quick-s0-step4`, `full-s0-step4`). Full, 5 seeds: ω — policy guide reached q_target in 5/5 runs but with more evaluations than no guide (3 692 ± 2 420 vs 2 524 ± 2 358); smaller spread of test NMSE (0.0068 vs 0.0088); warm start not better (test NMSE 0.3341 vs 0.3102, 4/5 reached). is_prime — policy reached the target in fewer evaluations (620 vs 696), warm start fewer still (504), test NMSE identical (0.0014). No test-NMSE difference is significant (p ≥ 0.24). Per the rule below, the result is recorded and the guide stays configurable (`formulas.guides`).

Reinforcement learning with PyTorch is **required** (owner decision): it learns over time which genome edits work and reduces the spread of results, measured relative to the average best scores. Torch therefore becomes a hard dependency of v11 (step 0 must install it).

- **Owner:** the guide lives in Evolvo (E7); v11 only configures it and stores its weights under `results/<run-id>/` and in a reusable knowledge file across runs (like `evolvo.nnarch.Knowledge`).
- **Actions:** the policy chooses the mutation operator and slot values (operation, source categories, input indices), conditioned on genome features from `GFSLFeatureExtractor`.
- **Reward:** relative to the average best scores, `r = (f_child − q) / (|q| + ε)`, where `q` is the current top-k mean at the same evaluation size `N`. Rewards are recomputed when `N` changes (S1), so learning never mixes scales.
- **Learning:** policy gradient (REINFORCE with a baseline, or actor-critic) on batches of offspring per generation; AdamW; device from `resolve_torch_accelerator` (`mps` on this Mac when available).
- **Less dispersion over time:** an entropy bonus or softmax temperature that falls as `q / q_target` rises; keep a uniform floor (as `evolvo.nnarch.AdaptivePolicy` does) so the search does not freeze.
- **Persistence:** save and load policy weights so later runs start from what earlier runs learned ("best practice over time"); record which knowledge file a run started from in the manifest.
- **Comparison:** same budget and seeds for no guide, `GFSLQLearningGuide`, `GFSLSupervisedGuide` and the RL guide. Report the mean and the spread across seeds of held-out error and of generations to target.
- **Done when:** across ≥ 5 seeds the RL guide reaches the target in fewer evaluations, or with smaller spread, than plain evolution, and a run that loads saved weights does better than a cold start. If it does not, record the result here and keep the guide configurable.

## Step 5 — Scale-up and asymptotics — DONE (verified: run `scale-s0`, 2026-11-24; drift reported)

Implemented in [`scale.py`](empirical-scripts/11.0/scale.py) (stage `scale`), multi-range stratified sampling in `dataset.sample_integers`, [`config/scale.json`](empirical-scripts/11.0/config/scale.json); tests [`tests/test_scale.py`](empirical-scripts/11.0/tests/test_scale.py) (39/39 v11 tests pass).

- [x] Scale config to n ≈ 10⁶: all n in [2, 9999] + 3 000 integers per ω in [10⁴, 10⁵) and in [10⁵, 10⁶] (all available for rare ω), wide base set (2..99) on 6 000 integers; 41 074 integers, 783 040 coprime pairs, data build 1 132 s on 7 workers, 1.07 GB peak RSS; whole run 1 294 s. Every n up to 10⁶ was not feasible (cost ∝ Σ L ≈ n²: ~15 h); ω-stratified decades answer the asymptotic question with comparable samples.
- [x] Per-decade claim statistics on ω-balanced samples (C1, C2, C3, C4, C9) and stability between [10⁴, 10⁵) and [10⁵, 10⁶].
- [x] Growth exponents recomputed with and without `log L` control; the paper's "predicted 0.743" traced and recomputed.
- **Result:** C1, C2, C3, C4 stable; **C9 drifts** — entropy features add +1.3 accuracy points at 10⁵–10⁶ (CI [+0.4, +2.0]) but not at 10⁴–10⁵ (+0.4, CI [−0.2, +1.1]), while the length baseline weakens with n (97.4% → 95.6% → 91.9%). See Results log.

## Step 6 — Report and paper update — IN PROGRESS (awaiting owner review)

- [x] `results/<run-id>/report.md` + `report.json` + figures: [`figures.py`](empirical-scripts/11.0/figures.py) (six PNGs: entropy vs length, held-out ω models, formula guides, C9 gain by decade, invariance by decade, growth exponents; validated light palette, direct labels), embedded by `report.write_report`; `python empirical-scripts/11.0/report.py <run_dir>` regenerates figures + report of a finished run from its stored files. Reports of `quick-s0-step4`, `full-s0-step4`, `scale-s0` regenerated this way; `quick-s0-step6` produced them in its own report stage. Small summary files committed; caches, knowledge and full-size splits git-ignored.
- [x] Draft [`studies/9/sieve-echo-conjecture-9.md`](studies/9/sieve-echo-conjecture-9.md): every number cites its run; direction contradictions of the v8 paper corrected; "Theorem" → "Observation" for empirical fits; one actual theorem (λ(n) ≤ φ(n)/2^{ω−1} for odd n) with proof as the mechanism of the negative slope; reproduction table of the v8 statistics; reformulated open conjecture. Author byline left to the owner.
- [x] [`AGENTS.md`](AGENTS.md) and this roadmap updated.
- **Done when:** owner reviews the report and the draft (not yet).

---

## Discovered work

Add items found during implementation here, with the step they affect.

- **Paper numbers come from v8.2, not v8.0** (step 2). `8.0/results/1` holds keys (`ndr_entropy_b*`) that only [`8.0/sieve-echo-tests-2.py`](empirical-scripts/8.0/sieve-echo-tests-2.py) writes. Its law fit regresses on **`log(ω + 1)`**, not the `log ω` printed in the paper (C1, C8: α and β depend on this); its entropy is `|FFT|²` over the first ⌊L/2⌋ bins with DC kept (`H_v8`).
- **v8 base invariance (6%) and zeta correlations use only the first 100 rows** of a shuffled list (`self.data[:100]`), not 10 000 integers (C4, C5).
- **r = -0.843 is corr(ω, is_prime_power), not an entropy correlation** (C3). It comes from the v7.0 log [`7.0/results/3/sieve_echo_log_20250825_135643.txt`](empirical-scripts/7.0/results/3/sieve_echo_log_20250825_135643.txt) line 210 (`omega_vs_is_prime_power`), which compares two factorization labels. On v11 labels this correlation is -0.75 for n < 10³ and -0.66 for n < 10⁴. Paper Theorem 2.3 attributes it to NDR entropy.
- **98.8% comes from the same v7.0 run** (log line 187), not from v8 (C-§3.1, step 4).
- **C9 baseline needs length arithmetic** (step 4). Without lcm(L_b) and its relation to n - 1, gradient boosting on per-base log L reaches 94.9%; with it 97.8% (`quick-s0-step6`, baselines `n + per-base L (no lcm arithmetic)` vs `n + L arithmetic`). The 2.9-point gap is what NDR features appeared to add in an interim step-2 version without the arithmetic. Step 4 search must be compared against the length-arithmetic baseline, and GFSL formulas should get `LCM`/`MOD` style ops or these derived inputs.
- **Evolvo: Q-learning guide had no evolver interface** (step 4, done in `libs/evolvo`): added `propose_mutation`/`observe_population` (relative rewards) and a fast `use_viability=False` construction (probability-tree viability made it ~80× slower).
- **Evolvo: in-place updates lost their history** (step 4, done in `libs/evolvo`): `_build_dependency_graph` registered an instruction as its target's producer before resolving its sources, so `d$2 = LOG(d$2)` dropped the earlier writers of `d$2` from the effective algorithm and from execution. Fixed + regression test.
- **ω q_target lowered from 0.8 to 0.7** (step 4): no run reached 0.8 on the quick config (best q ≈ 0.6), which leaves "evaluations to target" undefined; chosen from train fitness only.
- **GFSL-NN `set` encoding ignores the tokens** (step 4/5): the best full-run architecture marks the per-base token branch as an intron and predicts from log n + length arithmetic. Per-base NDR tokens add nothing the search could use.
- **Full-run memory** (step 5): peak RSS 9.2 GB in the main process (dataset + feature matrices + GFSL-NN tensors). Step 5's larger data must stream or subsample feature construction.
- **C7 controls must be per-base** (step 2, done). Controlling for the mean log L left a large residue effect (mod 12 excess 0.13) that was mostly which bases have short orders.
- **C4 "all n in [2, 9999]" label bug** (step 6, fixed): the rate used every full-range integer, i.e. n ≤ 150 000 on the full config (0.099 in `full-s0-claims`/`full-s0-step4`); the [2, 9999] value is 0.091. Code now restricts to n ≤ 9 999; stored full-run files keep the old label.
- **The DC bin dominates the paper entropy.** Quick run, `primes31` mean over integers: ω = 1 → H_paper 2.65, H_nodc 6.40, log L 7.31; ω = 4 → 1.85, 3.91, 4.39. Mean H falls with ω, and so does log L; whether anything is left after controlling for log L is C9. (Descriptive, all integers, no split; not a claim.)

## Results log

Record verified results as `run-id — config — headline numbers — verdict`. All runs: seed 0, `ndr-v11.1`, root `641d9b1` + the then-uncommitted v11 sources (v11 code sha256 `37b9b17ceb6a`, committed as `b4c9080`), libs/evolvo `4ab85e5`. Numbers are from `full-s0-claims` (test split, 24 302 integers) unless marked quick; quick (`quick-s0`, 1 579 test integers) gives the same verdicts except where noted.

- `full-s0` — `config/full.json` — data stage only: 162 007 integers, 2 340 678 pairs, 1 225 s.
- **C1** — `full-s0-claims` — v8.2 fit reproduced exactly (α -0.4681, β 1.9174, R² 0.2345 with log(ω+1); with the paper's log ω: α -0.305, R² 0.229). Test R² `H_paper`: law 0.364, log L + log n 0.959, full 0.959; ΔR² 0.0004 CI [0.000, 0.001]; partial r(H, log ω | log L, log n) -0.101. `H_nodc`: baseline 0.991, ΔR² 0.002. — **reproduced but explained by size/length.**
- **C2** — primes − composites mean `H_paper` +0.706 CI [0.686, 0.724]; Spearman(H, ω) -0.614; still negative given log L, log n. — primes have the **highest** entropy; §1.3 and §7.1 item 2 are **not supported**, α < 0 holds.
- **C3** — the paper's -0.843 is corr(ω, is_prime_power) (v11 on [2, 9999]: -0.655). r(H_paper, prime power) = +0.389; partial given log L, log n -0.058 CI [-0.068, -0.047]. — **not reproduced** (misattributed statistic; residual effect below threshold).
- **C4** — 6% reproduced exactly on v8.2's first 100 integers; all n in [2, 9999]: 9.1% (`quick-s0`, `scale-s0`; the 9.9% printed by the full runs covered all n ≤ 150 000 — label bug fixed in step 6). `v8`/`H_paper` invariance by scale 9% → 14% (10² → 10⁶), Spearman(CV, log n) -0.036. `Hn_nodc` reaches 99.6% above 10⁵, but CV(log L) < 0.1 also rises 7% → 62%: saturation of normalized entropy, not a property of the factorization. — **reproduced; no asymptotic emergence** of invariance for the paper entropy.
- **C5** — zeta r reproduced exactly on the 100-row sample (-0.441, -0.337, -0.289); on all n in [2, 9999] -0.114, -0.083, -0.070; partial r given cubic log n ≈ -0.02. — **reproduced but explained by size.**
- **C6** — r(H·log n, H) = 0.946 reproduced; with H permuted across n it is 0.831 CI [0.826, 0.836]; PNT/Li errors partial r ≈ -0.03. — **reproduced but explained by construction/size.**
- **C7** — v8.2 variances reproduced exactly. After per-base log L, coprimality, log n and ω controls, residue-class η² excess over the permutation null: mod 6 0.000, mod 12 0.033, mod 30 0.014, mod 210 0.032 (`H_paper`), similar for `H_nodc`; p ≤ 0.002. — **small residue structure beyond L and ω**: the excess appears with mod 12 (adds n mod 4) and mod 210 (adds n mod 7), not with mod 6. Not an ω effect.
- **C8** — theory α = -1.599 and β = 4.933 lie outside every 95% CI; a random value near α matches a v8.2 constant combination within 0.01 with probability 0.87. — **not supported** (constant matching carries no evidence).
- Step-4 runs (`quick-s0-step4`, `full-s0-step4`): root `4c8c1b0` + then-uncommitted step-4 sources (v11 code sha256 `19bb5b95393e`), libs/evolvo `74564d8` + then-uncommitted Q-learning interface and dependency fix; all committed in the step-4 commits.
- **Step 4 formulas** — `full-s0-step4` — is_prime: every guide finds a ≤ 3-instruction formula on "lcm of repetend lengths divides n−1" with test accuracy 0.9999 (NMSE 0.0014), a Fermat-type primality test from lengths. ω: best formula `(([log n − log lcm] · |lcm divides n−1|^(1/4)) + e)` (z-scored inputs), test NMSE 0.276, accuracy 0.655 — far below gradient boosting on the same 12 inputs (0.896) and the length-arithmetic baseline (0.982). Guide means (ω test NMSE): none 0.309, policy 0.310, supervised 0.347, Q-learning 0.367.
- **Step 4 held-out table** — `full-s0-step4` (24 302 test integers) — ω classes: majority 0.372, logistic(log n, log L) 0.597, GBM(log n, log L) 0.609, GBM(n + L arithmetic) 0.982, GBM(all allowed) 0.990, GFSL formula 0.655, GFSL-NN vec 0.948, GFSL-NN set 0.948. is_prime: majority 0.902, GBM(log n, log L) 0.953, L arithmetic / formula / all 1.000. — **Repetend-length arithmetic carries almost all predictable information about ω and all of it about primality; NDR entropy adds ≤ 1 point.**
- **§3.1 three-feature** — `full-s0-step4` — GBM test accuracy: (std H, min H) 0.509; + log n 0.566; + is_prime_power (leakage demo) 0.586, train 0.598; majority 0.372. — **98.8% not reproduced** even with the leaking feature; it was training-set accuracy of a different (v7) model.
- **Step 5 scale** — `scale-s0` (`config/scale.json`, seed 0, root `9fe7b75` + then-uncommitted step-5 sources, v11 code sha256 `c8979cf2cdd5`, libs/evolvo `d211185`) — per ω-balanced decade [10³,10⁴) / [10⁴,10⁵) / [10⁵,10⁶]: C1 ΔR² of log ω beyond (log L, log n) +0.0007 / +0.0008 / +0.0014 (all < 0.01; partial r −0.08 / −0.14 / −0.22, growing in size); C2 Spearman(H, ω) −0.64 / −0.83 / −0.87; C3 partial r −0.02 / −0.04 / −0.01; C4 invariance (v8) H_paper 0.13 / 0.18 / 0.21, Hn_nodc 0.95 / 0.99 / 1.00, CV(log L) < 0.1 0.39 / 0.57 / 0.69; wide set H_paper 0.05 / 0.08 / 0.13; C9 GBM accuracy baseline → + entropy: 0.974 → 0.972, 0.956 → 0.960, 0.919 → 0.932. — **C1–C4 stable; C9 drifts toward a small entropy gain at 10⁵–10⁶ as the 19-base length baseline loses accuracy.** Normalized DC-free entropy becomes base-invariant because it saturates, not because of factorization.
- **Growth exponents** — `scale-s0` — v8-style fit on [2, 9999]: H_v8 ≈ 0.460·n^0.129 (paper/v7: 0.431·n^0.142, reproduced up to the unknown v7 base set). ω-balanced decades, H_paper: a = 0.068; with ω fixed effects 0.096 (per ω 0.088–0.099); **with log(log L) as control a = 0.020** (b = 0.61). The paper's "predicted 0.743" is the exponent of the theta-7 double sum T(x) = Σ_{p≤x} Σ_{n≤x} log ord_n(p), a different quantity; recomputed for x = 100…3200 it is **2.07** (term count ∝ x²/log x). — **Entropy growth with n is repetend-length growth; the 0.142 vs 0.743 comparison is not meaningful.**
- **C9** — held-out ω classification: baseline (n, per-base L, length arithmetic) 98.2%; + entropy columns 98.6% (Δ +0.4 pts CI [0.2, 0.5]); + all NDR columns 98.9% (Δ +0.6 pts CI [0.5, 0.8], log-loss gain 0.018 CI [0.014, 0.021]); majority 37.2%, shuffled-label null 37.2%. Quick: entropy-only Δ +0.3 CI [-0.3, 0.9] (not significant at 1.6 k test rows). — **supported but small:** almost all information about ω in NDR is repetend-length information; the remainder is ~0.5 accuracy points.
