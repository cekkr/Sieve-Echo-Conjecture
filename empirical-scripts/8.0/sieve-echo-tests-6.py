#!/usr/bin/env python3
"""
Sieve Echo Conjecture - Enhanced Discovery Engine v8.6
Improvements:
- Real-time results saving in JSON/TXT format
- Dynamic exploration/exploitation balancing
- Adaptive parameter tuning
- Better utilization of historical best results
- Enhanced logging and visualization
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
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import warnings
import traceback
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

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
# ENHANCED CONFIGURATION WITH ADAPTIVE PARAMETERS
# ==============================================================================
@dataclass
class AdaptiveConfig:
    # File management
    state_file: str = "sieve_echo_state_v6.pkl"
    results_dir: str = "results_v6"
    log_interval_seconds: int = 60  # Save results every minute
    
    # Core parameters
    perpetual_mode: bool = True
    max_cycles: int = 1000
    data_chunk_size: int = 1000
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 19])
    n_workers: int = max(1, cpu_count() - 1)
    
    # Adaptive parameters (will be tuned during runtime)
    analysis_interval_cycles: int = 2
    correlation_threshold: float = 0.15
    constant_match_tolerance: float = 0.02
    
    # Evolution parameters (adaptive)
    formula_generations: int = 200
    formula_population_size: int = 1000
    max_algorithm_length: int = 60
    nn_generations: int = 50
    nn_population_size: int = 100
    
    # Exploration vs Exploitation balance
    exploration_rate: float = 0.3  # Will adapt based on progress
    elite_preservation_rate: float = 0.2
    mutation_rate: float = 0.7
    
    # Historical memory
    hall_of_fame_size: int = 50  # Best results to keep
    history_window: int = 10  # Cycles to consider for trend analysis
    
    def adapt_parameters(self, performance_metrics: Dict):
        """Dynamically adjust parameters based on performance"""
        # Check if we're stagnating
        if performance_metrics.get('stagnation_cycles', 0) > 5:
            # Increase exploration
            self.exploration_rate = min(0.8, self.exploration_rate * 1.2)
            self.mutation_rate = min(0.9, self.mutation_rate * 1.1)
            self.formula_population_size = min(2000, int(self.formula_population_size * 1.2))
        elif performance_metrics.get('improvement_rate', 0) > 0.1:
            # We're improving fast, can afford to exploit more
            self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
            self.elite_preservation_rate = min(0.4, self.elite_preservation_rate * 1.1)
        
        # Adjust analysis frequency based on data growth
        data_size = performance_metrics.get('data_size', 1000)
        if data_size > 10000:
            self.analysis_interval_cycles = 3
        elif data_size > 50000:
            self.analysis_interval_cycles = 5

CONFIG = AdaptiveConfig()

# ==============================================================================
# RESULTS LOGGER
# ==============================================================================
class ResultsLogger:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(results_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Different log files
        self.main_log = open(os.path.join(self.session_dir, "main.log"), 'w')
        self.results_json = os.path.join(self.session_dir, "results.json")
        self.formulas_log = open(os.path.join(self.session_dir, "formulas.txt"), 'w')
        self.metrics_csv = open(os.path.join(self.session_dir, "metrics.csv"), 'w')
        
        # Write CSV header
        self.metrics_csv.write("cycle,timestamp,data_size,best_fitness,alpha,beta,r_squared,exploration_rate\n")
        
    def log(self, message: str, console: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.main_log.write(log_msg + "\n")
        self.main_log.flush()
        if console:
            print(message)
    
    def save_results(self, results: Dict):
        """Save results to JSON with pretty formatting"""
        with open(self.results_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def log_formula(self, formula: str, fitness: float, formula_type: str):
        """Log discovered formulas"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.formulas_log.write(f"\n[{timestamp}] {formula_type}\n")
        self.formulas_log.write(f"Fitness: {fitness:.6f}\n")
        self.formulas_log.write(f"Formula: {formula}\n")
        self.formulas_log.write("-" * 80 + "\n")
        self.formulas_log.flush()
    
    def log_metrics(self, cycle: int, metrics: Dict):
        """Log metrics to CSV for analysis"""
        row = [
            cycle,
            datetime.now().isoformat(),
            metrics.get('data_size', 0),
            metrics.get('best_fitness', 'N/A'),
            metrics.get('alpha', 'N/A'),
            metrics.get('beta', 'N/A'),
            metrics.get('r_squared', 'N/A'),
            metrics.get('exploration_rate', CONFIG.exploration_rate)
        ]
        self.metrics_csv.write(','.join(str(x) for x in row) + "\n")
        self.metrics_csv.flush()
    
    def close(self):
        self.main_log.close()
        self.formulas_log.close()
        self.metrics_csv.close()

# ==============================================================================
# HALL OF FAME - Preserves best results across generations
# ==============================================================================
class HallOfFame:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.entries = []
        self.signatures = set()  # Avoid duplicates
    
    def add(self, entry: Dict):
        """Add entry if it's good enough and unique"""
        signature = self._get_signature(entry)
        if signature not in self.signatures:
            self.entries.append(entry)
            self.signatures.add(signature)
            # Sort by fitness (assuming lower is better)
            self.entries.sort(key=lambda x: x.get('fitness', float('inf')))
            # Keep only best entries
            if len(self.entries) > self.max_size:
                removed = self.entries.pop()
                self.signatures.discard(self._get_signature(removed))
    
    def _get_signature(self, entry: Dict) -> str:
        """Create unique signature for entry"""
        if 'algorithm' in entry:
            return hashlib.md5(str(entry['algorithm']).encode()).hexdigest()
        elif 'genome' in entry:
            return entry['genome'].get_signature() if hasattr(entry['genome'], 'get_signature') else str(entry)
        return str(entry)
    
    def get_best(self, n: int = 5) -> List[Dict]:
        return self.entries[:n]
    
    def get_random_elite(self, n: int = 10) -> List[Dict]:
        """Get random selection from top performers"""
        elite_pool = self.entries[:min(20, len(self.entries))]
        if len(elite_pool) <= n:
            return elite_pool
        return random.sample(elite_pool, n)

# ==============================================================================
# PERFORMANCE TRACKER
# ==============================================================================
class PerformanceTracker:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.fitness_history = deque(maxlen=window_size)
        self.cycle_times = deque(maxlen=window_size)
        self.discovery_times = {}
        self.stagnation_cycles = 0
        self.last_best_fitness = float('inf')
    
    def update(self, cycle: int, fitness: float, cycle_time: float):
        self.fitness_history.append(fitness)
        self.cycle_times.append(cycle_time)
        
        # Check for stagnation
        if fitness >= self.last_best_fitness:
            self.stagnation_cycles += 1
        else:
            self.stagnation_cycles = 0
            self.last_best_fitness = fitness
            self.discovery_times[cycle] = fitness
    
    def get_metrics(self) -> Dict:
        if len(self.fitness_history) < 2:
            return {'stagnation_cycles': 0, 'improvement_rate': 0}
        
        # Calculate improvement rate
        recent_avg = np.mean(list(self.fitness_history)[-5:])
        older_avg = np.mean(list(self.fitness_history)[:-5]) if len(self.fitness_history) > 5 else recent_avg
        improvement_rate = (older_avg - recent_avg) / older_avg if older_avg != 0 else 0
        
        return {
            'stagnation_cycles': self.stagnation_cycles,
            'improvement_rate': improvement_rate,
            'avg_cycle_time': np.mean(list(self.cycle_times)),
            'best_fitness_trend': list(self.fitness_history)
        }

# ==============================================================================
# Keep existing core classes with minor enhancements
# ==============================================================================
class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    def _handle_signal(self, *args):
        if self.kill_now: sys.exit(1)
        self.kill_now = True
        print("\n🛑 Graceful shutdown initiated... will save and report after current cycle.")

killer = GracefulKiller()

# [Keep existing NDRComputer and EnhancedMathematicalConstantsLibrary classes as-is]
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
# ENHANCED EVOLVO FORMULA DISCOVERER WITH HISTORY
# ==============================================================================
class EnhancedEvolvoFormulaDiscoverer:
    def __init__(self, data, const_lib, logger, hall_of_fame):
        self.data = data
        self.const_lib = const_lib
        self.logger = logger
        self.hall_of_fame = hall_of_fame
        self.instruction_set = self._create_instruction_set()
        self.best_unbiased = {'fitness': float('inf'), 'algorithm': [], 'formula': ""}
        self.best_theory_guided = {'fitness': float('inf'), 'algorithm': [], 'formula': ""}
        self.evolution_history = []
        
    def _create_instruction_set(self):
        iset = em.get_default_instruction_set()
        iset.register('SQRT', lambda a: math.sqrt(abs(a)), ['d'],'decimal')
        iset.register('LOG', lambda a: math.log(abs(a)+1e-9), ['d'],'decimal')
        iset.register('SIN', lambda a: math.sin(a), ['d'],'decimal')
        iset.register('COS', lambda a: math.cos(a), ['d'],'decimal')
        return iset
    
    def evolve_cycle(self, promising_features):
        self.logger.log("  EVOLVO: Starting evolution cycle...")
        
        # Unbiased track
        self._evolve(False, promising_features)
        
        # Theory-guided track
        self._evolve(True, [])
        
        # Save best formulas
        if self.best_unbiased['fitness'] < float('inf'):
            self.hall_of_fame.add({
                'type': 'formula_unbiased',
                'fitness': self.best_unbiased['fitness'],
                'algorithm': self.best_unbiased['algorithm'],
                'formula': self.best_unbiased['formula']
            })
            self.logger.log_formula(
                self.best_unbiased['formula'],
                self.best_unbiased['fitness'],
                "UNBIASED"
            )
        
        if self.best_theory_guided['fitness'] < float('inf'):
            self.hall_of_fame.add({
                'type': 'formula_theory',
                'fitness': self.best_theory_guided['fitness'],
                'algorithm': self.best_theory_guided['algorithm'],
                'formula': self.best_theory_guided['formula']
            })
            self.logger.log_formula(
                self.best_theory_guided['formula'],
                self.best_theory_guided['fitness'],
                "THEORY-GUIDED"
            )
    
    def _evolve(self, theory_guided, promising_features):
        is_unbiased = not theory_guided
        track_name = "Theory-Guided" if theory_guided else "Unbiased"
        
        if is_unbiased and not promising_features:
            self.logger.log(f"    Skipping {track_name} evolution: no promising features yet.")
            return
        
        self.logger.log(f"    Evolving {track_name} formulas...")
        
        store_config = self._get_store_config(theory_guided, promising_features)
        evaluator = self._create_evaluator(store_config, theory_guided)
        
        # Initialize population with mix of random and elite
        pop = []
        
        # Add elite from hall of fame
        elite_from_hof = [
            e for e in self.hall_of_fame.get_best(10)
            if e.get('type', '').startswith('formula')
        ]
        
        for elite in elite_from_hof[:int(CONFIG.formula_population_size * CONFIG.elite_preservation_rate)]:
            if 'algorithm' in elite:
                pop.append({'alg': elite['algorithm'], 'fitness': float('inf')})
        
        # Fill rest with random
        while len(pop) < CONFIG.formula_population_size:
            pop.append({'alg': self._generate_random_alg(store_config), 'fitness': float('inf')})
        
        # Evaluate initial population
        for ind in pop:
            ind['fitness'] = evaluator.evaluate(ind['alg'])
        
        # Evolution loop with adaptive parameters
        best_fitness_history = []
        for gen in range(CONFIG.formula_generations):
            pop.sort(key=lambda x: x['fitness'])
            best_fitness_history.append(pop[0]['fitness'])
            
            # Update best
            best_dict = self.best_unbiased if is_unbiased else self.best_theory_guided
            if pop[0]['fitness'] < best_dict['fitness']:
                best_dict.update(pop[0])
                best_dict['formula'] = self._decode_algorithm(pop[0]['alg'], store_config)
                self.logger.log(f"    Gen {gen}: New best! Fitness={best_dict['fitness']:.6f}")
            
            # Check for early stopping
            if gen > 50 and len(set(best_fitness_history[-20:])) == 1:
                self.logger.log(f"    Early stopping at generation {gen} - converged")
                break
            
            # Create new population
            elite_size = int(CONFIG.formula_population_size * CONFIG.elite_preservation_rate)
            new_pop = pop[:elite_size]
            
            # Adaptive exploration
            if random.random() < CONFIG.exploration_rate:
                # Add some completely random individuals
                for _ in range(int(CONFIG.formula_population_size * 0.1)):
                    new_pop.append({
                        'alg': self._generate_random_alg(store_config),
                        'fitness': evaluator.evaluate(self._generate_random_alg(store_config))
                    })
            
            # Standard breeding
            while len(new_pop) < CONFIG.formula_population_size:
                p1, p2 = self._tournament_select(pop), self._tournament_select(pop)
                child_alg = self._crossover(p1['alg'], p2['alg'])
                if random.random() < CONFIG.mutation_rate:
                    child_alg = self._mutate(child_alg, store_config)
                new_pop.append({'alg': child_alg, 'fitness': evaluator.evaluate(child_alg)})
            
            pop = new_pop[:CONFIG.formula_population_size]
    
    def _get_store_config(self, theory_guided, promising_features):
        d_vars = ['result', 'temp1', 'temp2']
        b_vars = ['flag']
        if theory_guided:
            d_consts = ['omega_log', 'n'] + list(self.const_lib.all_consts.keys())
            return {'d#': d_consts, 'b#': ['is_prime'], 'd$': d_vars, 'b$': b_vars}
        else:
            return {'d#': promising_features + ['n', 'one'], 'b#': ['is_prime'], 'd$': d_vars, 'b$': b_vars}
    
    def _create_evaluator(self, store_config, theory_guided):
        class FormulaEvaluator(em.BaseEvaluator):
            def __init__(self, data, store_config, instruction_set, theory_guided, const_lib):
                super().__init__(store_config, instruction_set)
                self.data = data
                self.theory_guided = theory_guided
                self.const_lib = const_lib
                
            def evaluate(self, algorithm, **kwargs):
                if not algorithm or len(algorithm) > CONFIG.max_algorithm_length:
                    return float('inf')
                err, count = 0.0, 0
                sample_size = min(200, len(self.data))
                for d in random.sample(self.data, sample_size):
                    if 'omega' not in d or d['omega'] == 0:
                        continue
                    ds = em.DataStore(store_config)
                    for key in store_config['d#']:
                        if key == 'omega_log':
                            ds.set_initial_value(key, math.log(d['omega']))
                        elif key == 'one':
                            ds.set_initial_value(key, 1.0)
                        elif key in self.const_lib.all_consts:
                            ds.set_initial_value(key, self.const_lib.all_consts[key])
                        else:
                            ds.set_initial_value(key, d.get(key, 0.0))
                    try:
                        self.interpreter.execute(algorithm, ds)
                        pred = ds.get('result')
                        actual = d.get('h_mean') if self.theory_guided else d['omega']
                        if actual is not None and np.isfinite(pred):
                            err += (pred - actual)**2
                            count += 1
                        else:
                            err += 1e6
                            count += 1
                    except Exception:
                        return float('inf')
                return (err / count if count > 0 else float('inf')) + len(algorithm) * 0.001
        
        return FormulaEvaluator(self.data, store_config, self.instruction_set, theory_guided, self.const_lib)
    
    def _generate_random_alg(self, store_config, max_len=None):
        if max_len is None:
            max_len = random.randint(2, 8)
        alg = []
        ops = [op for op in self.instruction_set.operations.keys() if op not in ['IF', 'END', 'ASSIGN']]
        for _ in range(random.randint(1, max_len)):
            op = random.choice(ops)
            prop = self.instruction_set.op_properties[op]
            target_var = random.choice(store_config['d$'])
            target = ['d$', store_config['d$'].index(target_var)]
            instr = target + [op]
            for arg_type in prop['arg_types']:
                store = 'd#' if random.random() < 0.8 else 'd$'
                idx = random.randint(0, len(store_config[store]) - 1)
                instr.extend([store, idx])
            alg.append(instr)
        final_instr = ['d$', 0, 'ASSIGN', 'd$', random.randint(0, len(store_config['d$'])-1)]
        alg.append(final_instr)
        return alg
    
    def _tournament_select(self, pop, tournament_size=5):
        return min(random.sample(pop, min(tournament_size, len(pop))), key=lambda x: x['fitness'])
    
    def _crossover(self, p1, p2):
        if not p1 or not p2:
            return p1 or p2
        pt = random.randint(1, min(len(p1), len(p2)))
        return p1[:pt] + p2[pt:]
    
    def _mutate(self, alg, store_config):
        if not alg or random.random() < 0.2:
            return self._generate_random_alg(store_config)
        mutation_type = random.choice(['modify', 'insert', 'delete'])
        if mutation_type == 'modify':
            idx = random.randint(0, len(alg)-1)
            alg[idx] = self._generate_random_alg(store_config, 1)[0]
        elif mutation_type == 'insert' and len(alg) < CONFIG.max_algorithm_length:
            idx = random.randint(0, len(alg))
            alg.insert(idx, self._generate_random_alg(store_config, 1)[0])
        elif mutation_type == 'delete' and len(alg) > 2:
            idx = random.randint(0, len(alg)-2)
            alg.pop(idx)
        return alg
    
    def _decode_algorithm(self, alg, store_config):
        if not alg:
            return "N/A"
        try:
            lines = []
            for instr in alg:
                if len(instr) >= 5:
                    target = f"{store_config[instr[0]][instr[1]]}"
                    op = instr[2]
                    if op == 'ASSIGN':
                        source = f"{store_config[instr[3]][instr[4]]}"
                        lines.append(f"{target} = {source}")
                    else:
                        args = []
                        for i in range(3, len(instr), 2):
                            if i+1 < len(instr):
                                args.append(f"{store_config[instr[i]][instr[i+1]]}")
                        lines.append(f"{target} = {op}({', '.join(args)})")
            return "; ".join(lines)
        except (IndexError, KeyError):
            return "Decoding Error"

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
# ENHANCED MAIN DISCOVERY ORCHESTRATOR
# ==============================================================================
class EnhancedSieveEchoDiscoverySystem:
    def __init__(self):
        self.start_time = time.time()
        self.logger = ResultsLogger(CONFIG.results_dir)
        self.logger.log("🌱 Initializing Enhanced Sieve Echo Discovery System v6.0")
        
        self.ndr_computer = NDRComputer()
        self.const_lib = EnhancedMathematicalConstantsLibrary()
        self.hall_of_fame = HallOfFame(CONFIG.hall_of_fame_size)
        self.performance_tracker = PerformanceTracker(CONFIG.history_window)
        
        self.load_state()
                
        self.analyzer = DiscoveryAnalyzer(self.data, self.const_lib)
        self.formula_evolver = EnhancedEvolvoFormulaDiscoverer(self.data, self.const_lib, self.logger, self.hall_of_fame)
        
        self.nas_searcher = NeuralArchitectureSearcher(self.data)
        
        self.last_save_time = time.time()
    
    def load_state(self):
        if os.path.exists(CONFIG.state_file):
            self.logger.log(f"📄 Resuming from {CONFIG.state_file}")
            with open(CONFIG.state_file, 'rb') as f:
                state = pickle.load(f)
            self.data = state.get('data', [])
            self.discoveries = state.get('discoveries', self._get_default_discoveries())
            self.current_n = state.get('current_n', 2)
            self.hall_of_fame.entries = state.get('hall_of_fame', [])
            self.start_time -= self.discoveries.get('total_runtime', 0)
            
            # Load performance history
            if 'performance_metrics' in state:
                self.performance_tracker.fitness_history = deque(
                    state['performance_metrics'].get('fitness_history', []),
                    maxlen=CONFIG.history_window
                )
        else:
            self.logger.log("🌱 Starting new session.")
            self.data = []
            self.discoveries = self._get_default_discoveries()
            self.current_n = 2
    
    def _get_default_discoveries(self):
        return {
            'unbiased_formula': {},
            'theory_formula': {},
            'best_nn': {},
            'sieve_echo_law': {},
            'correlations': {},
            'total_runtime': 0,
            'last_cycle': 0,
            'performance_metrics': {}
        }
    
    def save_state(self):
        """Enhanced state saving with performance metrics"""
        self.logger.log("💾 Saving state...")
        self.discoveries['total_runtime'] = time.time() - self.start_time
        
        # Get current performance metrics
        perf_metrics = self.performance_tracker.get_metrics()
        self.discoveries['performance_metrics'] = perf_metrics
        
        state = {
            'data': self.data[-10000:],  # Keep only recent data to manage file size
            'discoveries': self.discoveries,
            'current_n': self.current_n,
            'hall_of_fame': self.hall_of_fame.entries,
            'performance_metrics': {
                'fitness_history': list(self.performance_tracker.fitness_history)
            },
            'config': asdict(CONFIG)
        }
        
        with open(CONFIG.state_file, 'wb') as f:
            pickle.dump(state, f)
        
        # Also save to JSON for easy inspection
        self.logger.save_results({
            'discoveries': self.discoveries,
            'current_n': self.current_n,
            'data_size': len(self.data),
            'hall_of_fame_size': len(self.hall_of_fame.entries),
            'config': asdict(CONFIG)
        })
        
        self.logger.log("  ✅ State saved.")
    
    def generate_data_chunk(self):
        """Generate data with parallel processing"""
        start_n = self.current_n
        end_n = self.current_n + CONFIG.data_chunk_size
        self.logger.log(f"\nDATA GEN: n = {start_n} to {end_n-1}...")
        
        tasks = range(start_n, end_n)
        new_data = []
        
        with ProcessPoolExecutor(max_workers=CONFIG.n_workers) as executor:
            futures = {executor.submit(self.ndr_computer.compute_ndr_features, n, CONFIG.test_bases): n for n in tasks}
            for i, future in enumerate(as_completed(futures)):
                if (i+1) % 250 == 0:
                    self.logger.log(f"  Processed {(i+1)} numbers...", console=False)
                res = future.result()
                if res and 'omega' in res:
                    new_data.append(res)
        
        self.data.extend(new_data)
        self.current_n = end_n
        self.logger.log(f"  DATA GEN: Added {len(new_data)} data points. Total: {len(self.data)}")
    
    def run_discovery_engine(self):
        """Main discovery loop with adaptive parameters"""
        cycle = self.discoveries.get('last_cycle', 0)
        
        while not killer.kill_now:
            if not CONFIG.perpetual_mode and cycle >= CONFIG.max_cycles:
                self.logger.log("🏁 Reached max cycles.")
                break
            
            cycle += 1
            cycle_start_time = time.time()
            
            self.logger.log("\n" + "="*80)
            self.logger.log(f"🚀 STARTING DISCOVERY CYCLE {cycle} | n = {self.current_n}")
            self.logger.log("="*80)
            
            # Generate data
            self.generate_data_chunk()
            
            if len(self.data) < 200:
                continue
            
            # Adaptive analysis
            if cycle % CONFIG.analysis_interval_cycles == 0:
                self.logger.log("\n🔬 Running analysis cycle...")
                
                # Run analysis
                analysis_results = self.analyzer.run_full_analysis()
                self.discoveries.update(analysis_results)
                promising_features = self.analyzer.get_promising_features()
                
                # Evolve formulas
                self.formula_evolver.evolve_cycle(promising_features)
                
                # Evolve neural networks
                self.nas_searcher.evolve_cycle(promising_features)
                
                # Update discoveries
                self.discoveries['unbiased_formula'] = self.formula_evolver.best_unbiased
                self.discoveries['theory_formula'] = self.formula_evolver.best_theory_guided
                self.discoveries['best_nn'] = {
                    'fitness': self.nas_searcher.best_fitness,
                    'genome': self.nas_searcher.best_genome
                }
            
            # Track performance
            best_fitness = min(
                self.formula_evolver.best_unbiased.get('fitness', float('inf')),
                self.formula_evolver.best_theory_guided.get('fitness', float('inf'))
            )
            cycle_time = time.time() - cycle_start_time
            self.performance_tracker.update(cycle, best_fitness, cycle_time)
            
            # Get performance metrics
            perf_metrics = self.performance_tracker.get_metrics()
            perf_metrics['data_size'] = len(self.data)
            
            # Adapt parameters based on performance
            CONFIG.adapt_parameters(perf_metrics)
            
            # Log metrics
            metrics_to_log = {
                'data_size': len(self.data),
                'best_fitness': best_fitness if best_fitness < float('inf') else 'N/A',
                'alpha': self.discoveries.get('sieve_echo_law', {}).get('alpha', 'N/A'),
                'beta': self.discoveries.get('sieve_echo_law', {}).get('beta', 'N/A'),
                'r_squared': self.discoveries.get('sieve_echo_law', {}).get('r_squared', 'N/A'),
                'exploration_rate': CONFIG.exploration_rate
            }
            self.logger.log_metrics(cycle, metrics_to_log)
            
            # Update cycle counter
            self.discoveries['last_cycle'] = cycle
            
            # Periodic report
            self.periodic_report()
            
            # Save state periodically
            if time.time() - self.last_save_time > CONFIG.log_interval_seconds:
                self.save_state()
                self.last_save_time = time.time()
            
            self.logger.log(f"\n⏱️ Cycle {cycle} completed in {cycle_time:.2f} seconds")
            
            # Show adaptive parameter updates
            if perf_metrics['stagnation_cycles'] > 0:
                self.logger.log(f"📊 Stagnation detected ({perf_metrics['stagnation_cycles']} cycles)")
                self.logger.log(f"   Exploration rate: {CONFIG.exploration_rate:.3f}")
                self.logger.log(f"   Mutation rate: {CONFIG.mutation_rate:.3f}")
        
        self.logger.log("\n🛑 Discovery engine stopped by user.")
    
    def periodic_report(self):
        """Generate periodic status report"""
        self.logger.log("\n" + "-"*80)
        self.logger.log("📈 CURRENT DISCOVERY STATUS")
        self.logger.log("-"*80)
        
        runtime = time.time() - self.start_time
        self.logger.log(f"  Runtime: {runtime/3600:.2f} hrs | Data: {len(self.data)} | n={self.current_n-1}")
        
        # Sieve Echo Law
        law = self.discoveries.get('sieve_echo_law', {})
        if law and 'alpha' in law:
            self.logger.log("\n  --- Sieve Echo Law ---")
            self.logger.log(f"  H = {law['alpha']:.4f} * log(ω) + {law['beta']:.4f} (R² = {law['r_squared']:.3f})")
            if law.get('alpha_match'):
                self.logger.log(f"  α matches: '{law['alpha_match']['expr']}'")
            if law.get('beta_match'):
                self.logger.log(f"  β matches: '{law['beta_match']['expr']}'")
        
        # Top correlations
        corrs = self.discoveries.get('all_correlations', {})
        if corrs:
            self.logger.log("\n  --- Top Features ---")
            for i, (f, c) in enumerate(list(corrs.items())[:5]):
                self.logger.log(f"  {i+1}. {f}: {c:.3f}")
        
        # Best formulas
        unbiased_f = self.discoveries.get('unbiased_formula', {})
        theory_f = self.discoveries.get('theory_formula', {})
        nn = self.discoveries.get('best_nn', {})
        
        self.logger.log(f"\n  Best Unbiased MSE: {unbiased_f.get('fitness', 'N/A')}")
        self.logger.log(f"  Best Theory MSE: {theory_f.get('fitness', 'N/A')}")
        self.logger.log(f"  Best NN Fitness: {nn.get('fitness', 'N/A')}")
        
        # Hall of Fame summary
        self.logger.log(f"\n  Hall of Fame: {len(self.hall_of_fame.entries)} entries")
        
        # Performance metrics
        perf = self.performance_tracker.get_metrics()
        self.logger.log(f"  Stagnation: {perf['stagnation_cycles']} cycles")
        self.logger.log(f"  Improvement rate: {perf['improvement_rate']:.3%}")
        
        self.logger.log("-" * 80)
    
    def final_report(self):
        """Generate comprehensive final report"""
        self.logger.log("\n" + "="*80)
        self.logger.log("📋 FINAL DISCOVERY REPORT")
        self.logger.log("="*80)
        
        self.save_state()
        self.periodic_report()
        
        # Additional final analysis
        self.logger.log("\n  --- Session Summary ---")
        self.logger.log(f"  Total cycles: {self.discoveries['last_cycle']}")
        self.logger.log(f"  Data points analyzed: {len(self.data)}")
        self.logger.log(f"  Hall of Fame size: {len(self.hall_of_fame.entries)}")
        
        # Best discoveries
        best_entries = self.hall_of_fame.get_best(10)
        if best_entries:
            self.logger.log("\n  --- Top 10 Discoveries ---")
            for i, entry in enumerate(best_entries):
                self.logger.log(f"  {i+1}. Type: {entry.get('type', 'unknown')}, Fitness: {entry.get('fitness', 'N/A')}")
        
        self.logger.log("\n✨ Discovery session complete. All results saved.")
        self.logger.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("="*80)
    print("🔬 SIEVE ECHO ENHANCED DISCOVERY ENGINE - V6.0")
    print("="*80)
    print("Features:")
    print("  • Real-time results logging")
    print("  • Adaptive parameter tuning")
    print("  • Hall of Fame preservation")
    print("  • Performance tracking")
    print("  • Dynamic exploration/exploitation balance")
    print("\nPress Ctrl+C to stop gracefully.")
    print("="*80)
    
    system = EnhancedSieveEchoDiscoverySystem()
    
    try:
        system.run_discovery_engine()
    except Exception as e:
        system.logger.log(f"\n💥 Fatal error: {e}")
        traceback.print_exc()
    finally:
        system.logger.log("\n🔬 Generating final analysis...")
        system.final_report()
    
    print("\n✨ Session complete! Check results in:", CONFIG.results_dir)

if __name__ == "__main__":
    main()