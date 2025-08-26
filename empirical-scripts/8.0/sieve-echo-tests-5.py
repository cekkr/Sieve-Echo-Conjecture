#!/usr/bin/env python3
"""
Sieve Echo Conjecture - The Grand Unified Engine
Version 12.1 - The Resilient Engine

This version provides critical bug fixes and adds data integrity checks to
prevent crashes during long-running, autonomous operation.

Fixes:
- Resolves TypeError in correlation analysis by robustly handling missing feature values.
- Resolves ValueError in reporting by gracefully formatting potentially missing fitness scores.
- Enhances data generation and feature extraction to minimize creation of incomplete data points.
"""
import signal

import numpy as np
import math
import random
import pickle
import json
import time
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
import traceback
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# Core mathematical libraries
from sympy import factorint, isprime, totient, divisors, mobius
from scipy import stats
from scipy.fft import fft
from sklearn.linear_model import LinearRegression

# Device configuration
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import Evolvo and PyTorch libraries
try:
    import evolvo_model as em
    import evolvo_nn as en
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"FATAL: A required library is missing ({e}). The script cannot run.")
    LIBRARIES_AVAILABLE = False
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# CONFIGURATION AND UTILITIES
# ==============================================================================
@dataclass
class Config:
    state_file: str = "sieve_echo_state_v12.pkl"
    perpetual_mode: bool = True
    max_cycles: int = 500
    data_chunk_size: int = 1000
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 19])
    n_workers: int = max(1, cpu_count() - 1)
    analysis_interval_cycles: int = 2
    correlation_threshold: float = 0.15
    constant_match_tolerance: float = 0.02
    formula_generations: int = 200
    formula_population_size: int = 1000
    max_algorithm_length: int = 60
    nn_generations: int = 50
    nn_population_size: int = 100

CONFIG = Config()

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    def _handle_signal(self, *args):
        if self.kill_now: sys.exit(1)
        self.kill_now = True
        print("\n🛑 Graceful shutdown initiated... will report after current cycle.")

killer = GracefulKiller()

def format_value(value, format_spec=".4f"):
    """Safely formats a value, returning a default string if not a number."""
    if isinstance(value, (int, float, np.number)):
        return f"{value:{format_spec}}"
    return str(value)

# ==============================================================================
# MATHEMATICAL CONCEPTS AND LIBRARIES
# ==============================================================================
class NDRComputer:
    def __init__(self): self.cache = {}
    def compute_repetend(self, n: int, base: int) -> List[int]:
        if (n, base) in self.cache: return self.cache[(n, base)]
        if math.gcd(n, base) != 1: return []
        r, d, s, max_len = 1, [], {}, n + 1
        while r != 0 and r not in s and len(d) < max_len:
            s[r], r = len(d), r * base; d.append(r // n); r %= n
        res = d[s[r]:] if r in s else d
        self.cache[(n, base)] = res; return res
    def compute_ndr_features(self, n: int, bases: List[int]) -> Dict:
        try:
            factors = factorint(n); omega = len(factors)
            is_prime = omega == 1 and factors.get(n) == 1
            features = {'n': n, 'omega': omega, 'phi': totient(n), 'is_prime': is_prime}
        except Exception: return {}
        ents, lens, kurts = [], [], []
        for base in bases:
            pattern = self.compute_repetend(n, base)
            if not pattern: continue
            ndr = np.array(pattern) / base
            lens.append(len(pattern))
            if len(ndr) > 3: kurts.append(stats.kurtosis(ndr))
            if len(ndr) > 1:
                power = np.abs(fft(ndr))[:len(ndr)//2]**2
                total_power = np.sum(power)
                if total_power > 1e-9:
                    p = power / total_power; p = p[p > 1e-10]
                    if len(p) > 0: ents.append(-np.sum(p * np.log(p)))
        if ents: features['h_mean'], features['h_std'] = np.mean(ents), np.std(ents)
        if lens: features['len_mean'] = np.mean(lens)
        if kurts: features['kurt_mean'] = np.mean(kurts)
        return features

class EnhancedMathematicalConstantsLibrary:
    def __init__(self):
        self.constants = {'e': math.e, 'pi': math.pi, 'phi': (1 + math.sqrt(5)) / 2, 'gamma': 0.5772156649}
        self.expressions = {'5_minus_1_over_15': 5 - 1/15, 'neg_inv_phi_sq': -1 / self.constants['phi']**2}
        self.discovered_strong_constants = {}
        self.update_all_consts()
    def update_all_consts(self): self.all_consts = {**self.constants, **self.expressions, **self.discovered_strong_constants}
    def add_strong_constant(self, name, value):
        self.discovered_strong_constants[name] = value; self.update_all_consts()
    def find_closest_match(self, value):
        if not np.isfinite(value): return None
        matches = [{'expr': n, 'val': c, 'err': abs(value - c)}
                   for n, c in self.all_consts.items() if abs(value - c) < CONFIG.constant_match_tolerance]
        return min(matches, key=lambda x: x['err']) if matches else None

# ==============================================================================
# DISCOVERY ANALYZER - THE "AUTONOMOUS THEORIST"
# ==============================================================================
class DiscoveryAnalyzer:
    def __init__(self, data: List[Dict], const_lib: EnhancedMathematicalConstantsLibrary):
        self.data, self.const_lib = data, const_lib
        self.results = {}
    def run_full_analysis(self):
        print("\n  ANALYZER: Conducting full analysis on current dataset...")
        if len(self.data) < 200:
            print("  ANALYZER: Not enough data for robust analysis. Skipping."); return {}
        self._analyze_correlations()
        self._analyze_sieve_echo_law()
        print("  ANALYZER: Analysis complete."); return self.results

    def _analyze_correlations(self):
        corrs = {}
        # Ensure a consistent set of features based on recent data
        sample_keys = self.data[-1].keys()
        feature_keys = [k for k in sample_keys if isinstance(self.data[-1][k], (float, int)) and k not in ['n', 'omega']]
        
        # This comprehension robustly filters out data points missing 'omega'
        omegas = np.array([d['omega'] for d in self.data if 'omega' in d])
        
        for key in feature_keys:
            ### START BUG FIX ###
            # Original code created `vals` with `None` if a key was missing in some data points.
            # This version explicitly filters for existence and numeric type.
            valid_pairs = [(d[key], d['omega']) for d in self.data if key in d and 'omega' in d and isinstance(d[key], (int, float))]
            if len(valid_pairs) < 100: continue
            
            vals, omegas_for_key = zip(*valid_pairs)
            vals_arr, omegas_arr = np.array(vals), np.array(omegas_for_key)
            
            if np.std(vals_arr) > 0 and np.std(omegas_arr) > 0:
                corrs[key] = np.corrcoef(vals_arr, omegas_arr)[0, 1]
            ### END BUG FIX ###

        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        self.results['all_correlations'] = dict(sorted_corrs)
        self.results['promising_features'] = [f for f, c in sorted_corrs if abs(c) > CONFIG.correlation_threshold]
        self.results['dead_end_features'] = [f for f, c in sorted_corrs if abs(c) < 0.05]

    def _analyze_sieve_echo_law(self):
        points = [(math.log(d['omega']), d['h_mean']) for d in self.data if d.get('h_mean') and d.get('omega', 0) > 0]
        if len(points) < 100: return
        X, y = np.array([p[0] for p in points]).reshape(-1, 1), np.array([p[1] for p in points])
        model = LinearRegression().fit(X, y)
        alpha, beta, r2 = model.coef_[0], model.intercept_, model.score(X, y)
        self.results['sieve_echo_law'] = {
            'alpha': alpha, 'beta': beta, 'r_squared': r2,
            'alpha_match': self.const_lib.find_closest_match(alpha),
            'beta_match': self.const_lib.find_closest_match(beta)
        }
    def get_promising_features(self): return self.results.get('promising_features', [])

# ==============================================================================
# EVOLVO FORMULA DISCOVERY SYSTEM (GENETIC PROGRAMMING)
# ==============================================================================
class EvolvoFormulaDiscoverer:
    def __init__(self, data, const_lib):
        self.data, self.const_lib = data, const_lib
        self.instruction_set = self._create_instruction_set()
        self.best_unbiased = {'fitness': float('inf'), 'algorithm': [], 'formula': ""}
        self.best_theory_guided = {'fitness': float('inf'), 'algorithm': [], 'formula': ""}
    def _create_instruction_set(self):
        iset = em.get_default_instruction_set(); iset.register('SQRT', lambda a: math.sqrt(abs(a)), ['d'],'decimal'); iset.register('LOG', lambda a: math.log(abs(a)+1e-9), ['d'],'decimal'); return iset
    def evolve_cycle(self, promising_features):
        print("\n  EVOLVO: Evolving formulas (Unbiased Track)...")
        self._evolve(False, promising_features)
        print("  EVOLVO: Evolving formulas (Theory-Guided Track)...")
        self._evolve(True, [])
    def _evolve(self, theory_guided, promising_features):
        is_unbiased = not theory_guided
        if is_unbiased and not promising_features: print("    Skipping unbiased evolution: no promising features yet."); return
        store_config = self._get_store_config(theory_guided, promising_features)
        evaluator = self._create_evaluator(store_config, theory_guided)
        pop = [{'alg': self._generate_random_alg(store_config), 'fitness': float('inf')} for _ in range(CONFIG.formula_population_size)]
        for ind in pop: ind['fitness'] = evaluator.evaluate(ind['alg'])
        for gen in range(CONFIG.formula_generations):
            pop.sort(key=lambda x: x['fitness'])
            best_dict = self.best_unbiased if is_unbiased else self.best_theory_guided
            if pop[0]['fitness'] < best_dict['fitness']:
                best_dict.update(pop[0]); best_dict['formula'] = self._decode_algorithm(pop[0]['alg'], store_config)
                print(f"    Gen {gen}: New best! Fitness={best_dict['fitness']:.4f}, Formula: {best_dict['formula']}")
            elite_size = CONFIG.formula_population_size // 5
            new_pop = pop[:elite_size]
            while len(new_pop) < CONFIG.formula_population_size:
                p1, p2 = self._tournament_select(pop), self._tournament_select(pop)
                child_alg = self._crossover(p1['alg'], p2['alg'])
                if random.random() < 0.7: child_alg = self._mutate(child_alg, store_config)
                new_pop.append({'alg': child_alg, 'fitness': evaluator.evaluate(child_alg)})
            pop = new_pop
    def _get_store_config(self, theory_guided, promising_features):
        d_vars = ['result', 'temp1']; b_vars = ['flag']
        if theory_guided:
            d_consts = ['omega_log', 'n'] + list(self.const_lib.all_consts.keys())
            return {'d#': d_consts, 'b#': ['is_prime'], 'd$': d_vars, 'b$': b_vars}
        else:
            return {'d#': promising_features + ['n', 'one'], 'b#': ['is_prime'], 'd$': d_vars, 'b$': b_vars}
    def _create_evaluator(self, store_config, theory_guided):
        class FormulaEvaluator(em.BaseEvaluator):
            def __init__(self, data, store_config, instruction_set, theory_guided, const_lib):
                super().__init__(store_config, instruction_set)
                self.data, self.theory_guided, self.const_lib = data, theory_guided, const_lib
            def evaluate(self, algorithm, **kwargs):
                if not algorithm or len(algorithm) > CONFIG.max_algorithm_length: return float('inf')
                err, count = 0.0, 0
                for d in random.sample(self.data, min(200, len(self.data))):
                    if 'omega' not in d or d['omega'] == 0: continue
                    ds = em.DataStore(store_config)
                    for key in store_config['d#']:
                        if key == 'omega_log': ds.set_initial_value(key, math.log(d['omega']))
                        elif key == 'one': ds.set_initial_value(key, 1.0)
                        elif key in self.const_lib.all_consts: ds.set_initial_value(key, self.const_lib.all_consts[key])
                        else: ds.set_initial_value(key, d.get(key, 0.0))
                    try:
                        self.interpreter.execute(algorithm, ds); pred = ds.get('result')
                        actual = d.get('h_mean') if self.theory_guided else d['omega']
                        if actual is not None and np.isfinite(pred): err += (pred - actual)**2; count += 1
                        else: err += 1e6; count += 1
                    except Exception: return float('inf')
                return (err / count if count > 0 else float('inf')) + len(algorithm) * 0.001
        return FormulaEvaluator(self.data, store_config, self.instruction_set, theory_guided, self.const_lib)
    def _generate_random_alg(self, store_config, max_len=5):
        alg = []
        ops = [op for op in self.instruction_set.operations.keys() if op not in ['IF', 'END', 'ASSIGN']]
        for _ in range(random.randint(1, max_len)):
            op = random.choice(ops); prop = self.instruction_set.op_properties[op]
            target_var = random.choice(store_config['d$']); target = ['d$', store_config['d$'].index(target_var)]
            instr = target + [op]
            for arg_type in prop['arg_types']:
                store = 'd#' if random.random() < 0.8 else 'd$'
                idx = random.randint(0, len(store_config[store]) - 1); instr.extend([store, idx])
            alg.append(instr)
        final_instr = ['d$', 0, 'ASSIGN', 'd$', random.randint(0, len(store_config['d$'])-1)]
        alg.append(final_instr); return alg
    def _tournament_select(self, pop): return min(random.sample(pop, min(5, len(pop))), key=lambda x: x['fitness'])
    def _crossover(self, p1, p2):
        if not p1 or not p2: return p1 or p2; pt = random.randint(1, min(len(p1), len(p2))); return p1[:pt] + p2[pt:]
    def _mutate(self, alg, store_config):
        if not alg or random.random() < 0.2: return self._generate_random_alg(store_config)
        idx = random.randint(0, len(alg)-1); alg[idx] = self._generate_random_alg(store_config, 1)[0]
        return alg
    def _decode_algorithm(self, alg, store_config):
        if not alg: return "N/A"
        try:
            final_line = alg[-1]; source_type, source_idx = final_line[3], final_line[4]
            return f"result = {store_config[source_type][source_idx]}"
        except (IndexError, KeyError): return "Decoding Error"

# ==============================================================================
# NEURAL ARCHITECTURE SEARCH (NAS)
# ==============================================================================
class NeuralArchitectureSearcher:
    def __init__(self, data): self.data, self.best_genome, self.best_fitness = data, None, 0.0
    def evolve_cycle(self, promising_features):
        print("  NAS: Evolving neural architectures for ω(n)...")
        feature_names = [f for f in promising_features if any(f in d for d in self.data)]
        if len(feature_names) < 2: print("     NAS: Not enough promising features. Skipping."); return
        input_shape = en.TensorShape(features=len(feature_names)); output_shape = en.TensorShape(features=1)
        evolver = en.AdvancedModelEvolver(input_shape, output_shape, task_type='regression')
        pop = evolver.evolve(CONFIG.nn_generations, CONFIG.nn_population_size, 
                             lambda g: self._evaluate_genome(g, feature_names), multi_objective=False)
        if pop and (pop[0].fitness or 0) > self.best_fitness:
            self.best_genome, self.best_fitness = pop[0], pop[0].fitness
            print(f"    NAS: New best NN! Fitness={self.best_fitness:.4f}, Layers={len(self.best_genome.layers)}")
    def _evaluate_genome(self, genome, feature_names):
        try:
            X = np.array([[d.get(f, 0.0) for f in feature_names] for d in self.data])
            y = np.array([d.get('omega', 0.0) for d in self.data])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_t, y_train_t = torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).view(-1, 1).to(device)
            X_test_t, y_test_t = torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).view(-1, 1).to(device)
            model = genome.to_pytorch_model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005); criterion = nn.MSELoss()
            for _ in range(20):
                optimizer.zero_grad(); loss = criterion(model(X_train_t), y_train_t); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad(): test_loss = criterion(model(X_test_t), y_test_t).item()
            return 1 / (1 + test_loss)
        except Exception: return 0.0

# ==============================================================================
# MAIN DISCOVERY ORCHESTRATOR
# ==============================================================================
class SieveEchoDiscoverySystem:
    def __init__(self):
        self.start_time = time.time(); self.ndr_computer = NDRComputer()
        self.const_lib = EnhancedMathematicalConstantsLibrary(); self.load_state()
        self.analyzer = DiscoveryAnalyzer(self.data, self.const_lib)
        self.formula_evolver = EvolvoFormulaDiscoverer(self.data, self.const_lib)
        self.nas_searcher = NeuralArchitectureSearcher(self.data)
    def load_state(self):
        if os.path.exists(CONFIG.state_file):
            print(f"🔄 Resuming from {CONFIG.state_file}");
            with open(CONFIG.state_file, 'rb') as f: state = pickle.load(f)
            self.data = state.get('data', []); self.discoveries = state.get('discoveries', self._get_default_discoveries())
            self.current_n = state.get('current_n', 2); self.start_time -= self.discoveries.get('total_runtime', 0)
        else:
            print("🌱 Starting new session."); self.data, self.discoveries, self.current_n = [], self._get_default_discoveries(), 2
    def _get_default_discoveries(self):
        return {'unbiased_formula': {}, 'theory_formula': {}, 'best_nn': {}, 'sieve_echo_law': {},
                'correlations': {}, 'total_runtime': 0, 'last_cycle': 0}
    def save_state(self):
        print(f"\n💾 Saving state..."); self.discoveries['total_runtime'] = time.time() - self.start_time
        state = {'data': self.data, 'discoveries': self.discoveries, 'current_n': self.current_n}
        with open(CONFIG.state_file, 'wb') as f: pickle.dump(state, f); print("  ✅ State saved.")
    def generate_data_chunk(self):
        start_n, end_n = self.current_n, self.current_n + CONFIG.data_chunk_size
        print(f"\nDATA GEN: n = {start_n} to {end_n-1}..."); tasks = range(start_n, end_n); new_data = []
        with ProcessPoolExecutor(max_workers=CONFIG.n_workers) as executor:
            futures = {executor.submit(self.ndr_computer.compute_ndr_features, n, CONFIG.test_bases): n for n in tasks}
            for i, future in enumerate(as_completed(futures)):
                if (i+1) % 250 == 0: print(f"  Processed {(i+1)} numbers...")
                res = future.result();
                if res and 'omega' in res: new_data.append(res)
        self.data.extend(new_data); self.current_n = end_n
        print(f"  DATA GEN: Added {len(new_data)} data points. Total: {len(self.data)}")
    def run_discovery_engine(self):
        cycle = self.discoveries.get('last_cycle', 0)
        while not killer.kill_now:
            if not CONFIG.perpetual_mode and cycle >= CONFIG.max_cycles: print("🏁 Reached max cycles."); break
            cycle += 1; print("\n"+"="*80+f"\n🚀 STARTING DISCOVERY CYCLE {cycle} | Current n = {self.current_n}\n"+"="*80)
            self.generate_data_chunk()
            if len(self.data) < 200: continue
            if cycle % CONFIG.analysis_interval_cycles == 0:
                analysis_results = self.analyzer.run_full_analysis()
                self.discoveries.update(analysis_results)
                promising_features = self.analyzer.get_promising_features()
                self.formula_evolver.evolve_cycle(promising_features)
                self.nas_searcher.evolve_cycle(promising_features)
                self.discoveries['unbiased_formula'] = self.formula_evolver.best_unbiased
                self.discoveries['theory_formula'] = self.formula_evolver.best_theory_guided
                self.discoveries['best_nn'] = {'fitness': self.nas_searcher.best_fitness, 'genome': self.nas_searcher.best_genome}
            self.discoveries['last_cycle'] = cycle; self.periodic_report(); self.save_state()
        print("\nλούπ-α-ν-τ-ό-ρ has been terminated by user.")
    def periodic_report(self):
        print("\n" + "-"*80 + "\n📈 CURRENT DISCOVERY STATUS\n" + "-"*80)
        runtime = time.time() - self.start_time
        print(f"  - Runtime: {runtime/3600:.2f} hrs | Data points: {len(self.data)} | Explored up to n={self.current_n-1}")
        law = self.discoveries.get('sieve_echo_law', {})
        if law and 'alpha' in law:
            print("\n  --- Sieve Echo Law ---"); print(f"  - Empirical: H = {law['alpha']:.4f} * log(ω) + {law['beta']:.4f} (R² = {law['r_squared']:.3f})")
            if law.get('alpha_match'): print(f"  - α ({law['alpha']:.4f}) is close to: '{law['alpha_match']['expr']}'")
            if law.get('beta_match'): print(f"  - β ({law['beta']:.4f}) is close to: '{law['beta_match']['expr']}'")
        corrs = self.discoveries.get('all_correlations', {})
        if corrs:
            print("\n  --- Top 5 Promising Features (Correlation with ω) ---")
            for i, (f, c) in enumerate(list(corrs.items())[:5]): print(f"  {i+1}. {f}: {c:.3f}")
        
        ### START BUG FIX ###
        # Original code had a formatting error for missing values.
        unbiased_f = self.discoveries.get('unbiased_formula', {})
        theory_f = self.discoveries.get('theory_formula', {})
        nn = self.discoveries.get('best_nn', {})
        
        print(f"\n  - Best Unbiased Formula (MSE): {format_value(unbiased_f.get('fitness', 'N/A'))}")
        print(f"  - Best Theory-Guided Formula (MSE): {format_value(theory_f.get('fitness', 'N/A'))}")
        print(f"  - Best NN Fitness (1/(1+MSE)): {format_value(nn.get('fitness', 'N/A'))}")
        ### END BUG FIX ###
        print("-" * 80)
    def final_report(self):
        print("\n" + "="*80 + "\n📋 FINAL DISCOVERY REPORT\n" + "="*80)
        self.save_state(); self.periodic_report()
        dead_ends = self.discoveries.get('dead_end_features', [])
        if dead_ends:
            print("\n  --- Dead-End Features (Low Correlation with ω) ---")
            print(f"  - Features with low predictive power: {', '.join(dead_ends[:10])}...")
        print("\n  --- Best Discovered Formulas ---")
        unbiased_f = self.discoveries.get('unbiased_formula', {})
        theory_f = self.discoveries.get('theory_formula', {})
        print(f"  - Unbiased: {unbiased_f.get('formula', 'N/A')} (Fitness: {format_value(unbiased_f.get('fitness'))})")
        print(f"  - Theory-Guided: {theory_f.get('formula', 'N/A')} (Fitness: {format_value(theory_f.get('fitness'))})")
        print("\n✨ Discovery engine shut down. Findings preserved in state file.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("="*80); print("🔬 SIEVE ECHO GRAND UNIFIED ENGINE - V12.1 (Resilient)"); print("="*80)
    print("This will run continuously to find patterns and validate theories.")
    print("Press Ctrl+C to stop gracefully and generate a final report.")
    system = SieveEchoDiscoverySystem()
    try: system.run_discovery_engine()
    except Exception as e:
        print(f"\n💥 A fatal error occurred: {e}"); traceback.print_exc()
    finally:
        print("\n🏁 Discovery process terminated. Generating final analysis...")
        system.final_report()
    print("\n✨ Session complete!")

if __name__ == "__main__":
    main()