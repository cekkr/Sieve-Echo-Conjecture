# Sieve-Echo-Conjecture

An empirical study of the **Sieve Echo Conjecture**: the idea that the prime-factorization complexity of an integer *n* — chiefly ω(n), the number of distinct prime factors — is encoded in the spectral entropy of the repeating digits of 1/*n* written in several bases (normalized digit representation, NDR). The project grew out of questions about the Riemann zeta function and the distribution of primes.

The repository holds numbered script generations (`empirical-scripts/3.1` … `11.0`) and the papers in `studies/`. Version 11 ([`empirical-scripts/11.0/`](empirical-scripts/11.0)) is the current, reproducible, leakage-free implementation: one entry point, held-out evaluation, controls for repetend length and size, permutation nulls, and evolutionary search through the [Evolvo](libs/evolvo) submodule. The working paper is [`studies/9/sieve-echo-conjecture-9.md`](studies/9/sieve-echo-conjecture-9.md) (draft for the author's review); the plan and the full results log are in [`ROADMAP.md`](ROADMAP.md); orientation for contributors and AI agents is in [`AGENTS.md`](AGENTS.md).

## Final results (v11, seed 0)

Every number names the run that produced it; run summaries, manifests and figures are committed under `empirical-scripts/11.0/results/<run-id>/`.

1. **Every statistic of the v8 paper reproduces exactly** from its own code path (law fit α = −0.4681, β = 1.9174, R² = 0.2345; 6% base invariance; ζ and PNT correlations; residue-class variances) — and each is explained by repetend length, by the size of *n*, by construction, or by a 100-integer sample (`full-s0-claims`, `full-s0-step4`).
2. **NDR entropy is a function of repetend length** L = ord_n(b): log L and log n explain 95.9% of its held-out variance; log ω adds 0.04% (`full-s0-step4`). Primes have the *highest* entropy. The negative slope of the "law" follows from λ(n) ≤ φ(n)/2^{ω−1} (Theorem 1.3 of the paper).
3. **The factorization information lives in the lengths.** Gradient boosting on *n* and the per-base lengths with their lcm arithmetic predicts ω classes at 98.2% held-out accuracy; evolved formulas recognize primes at 99.99% from "lcm of the lengths divides n − 1" — Fermat's little theorem in repetend form (`full-s0-step4`).
4. **The apparent residual signal at 10⁵–10⁶ disappears** with a stronger baseline: entropy adds −0.07 accuracy points (95% CI [−0.33, +0.24]) once the orders in all bases 2…99 and the Midy flags are known (`scale-s0-step7`).
5. **The digits do echo the factorization** — through Midy's symmetry (when −1 ∈ ⟨b⟩ mod n the DC-free entropy drops by exactly log 2, fitted −0.694; `scale-s0-step7`) and through **exact spectral zeros** whose Euler factors vanish at the prime divisors of n (0.09% of prime pairs vs 11–30% of composite pairs; `scale-s0-step8`). Both are classical number theory, and neither adds information about ω beyond the multiplicative orders (zeros: −0.05 points, CI [−0.28, +0.18]).
6. **Residue-class structure is partly Euler-product structure:** whether the small primes 2…13 lie in the subgroup ⟨b⟩ mod n halves the remaining mod-12 excess and cuts the mod-210 excess by 30% (`scale-s0-step9`); the rest is open.
7. **Orders alone, up to 10⁸:** classifying ω from the orders in bases 2…199 gives 97.9%, 97.3%, 94.5%, 89.7% held-out accuracy in the decades 10⁴…10⁸; a fixed base set reads less as n grows, and more bases recover part of it (`orders-s0`).
8. **SinWhisky spectra** (the [circle-based resampler](libs/sinwhisky) applied before the DFT) carry the same information as the plain FFT (correlation 0.9998, no extra ω information; `scale-s0-step8e`).

**Conclusion so far:** the "echo of the sieve" in 1/n is real, but it is the multiplicative order ord_n(b) — and its group structure — that carries it; the digit entropy adds nothing measurable about ω beyond the orders. Open questions are listed in §8 of the paper and in the roadmap.

## Reproduce

```bash
git submodule update --init --recursive
/opt/homebrew/bin/python3.15 -m venv .venv
.venv/bin/pip install --only-binary :all: -r empirical-scripts/11.0/requirements.txt
.venv/bin/python -m unittest discover -s empirical-scripts/11.0/tests
.venv/bin/python empirical-scripts/11.0/sieve_echo_v11.py --config empirical-scripts/11.0/config/quick.json --run-id quick-s0
```

The full and scale configurations and their run times are described in [`AGENTS.md`](AGENTS.md) and §9 of the paper.

## Credits

The Sieve Echo Conjecture, the earlier script generations and the papers are the work of the repository author. [Evolvo](libs/evolvo) and [SinWhisky](libs/sinwhisky) are separate projects included as git submodules.

Version 11 — the reproducible pipeline, the claim-by-claim re-verification, the Evolvo and SinWhisky extensions it needed, the analyses of ROADMAP steps 0–9 and the revision-9 draft — was developed in collaboration with **Claude (Anthropic)**, an AI assistant, working with the author.
