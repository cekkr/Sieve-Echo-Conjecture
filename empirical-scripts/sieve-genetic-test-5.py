#!/usr/bin/env python3
"""
Sieve Echo Conjecture - Complete Framework with Enhanced Verbosity and GPU Support
Version 3.1 - Combines v3 performance optimizations (pre-computation, interruptibility)
              with v2's full feature set and detailed verbosity.

Riccardo Cecchini rcecchini.ds[at]gmail.com - 21 Aug 2025
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
import random
import pickle
import json
import time
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
from sympy import factorint, primerange, isprime, totient
from scipy import stats
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import traceback

import warnings
warnings.filterwarnings('ignore')

# Force CUDA if available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f"CUDA ENABLED: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available, using CPU")

# Configuration
@dataclass
class Config:
    max_n: int = 50000
    population_size: int = 500
    generations: int = 5000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 10
    checkpoint_interval: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    runtime_hours: float = 16.0
    pattern_cache_size: int = 50000
    log_file: str = f"sieve_echo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results_file: str = f"sieve_echo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    stop_file: str = "STOP_SIEVE_ECHO"

CONFIG = Config()

class Logger:
    def __init__(self, log_file: str, results_file: str):
        self.log_file = log_file
        self.results_file = results_file
        self.start_time = time.time()
        with open(self.log_file, 'w') as f: f.write(f"Sieve Echo Conjecture - Detailed Log | Started: {datetime.now()} | Device: {CONFIG.device}\n\n")
        with open(self.results_file, 'w') as f: f.write(f"Sieve Echo Conjecture - Results Summary | Started: {datetime.now()}\n\n")

    def log(self, message: str, level: str = "INFO", to_console: bool = True):
        elapsed = time.time() - self.start_time
        formatted_msg = f"[{datetime.now().strftime('%H:%M:%S')}][{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}][{level}] {message}"
        with open(self.log_file, 'a') as f: f.write(formatted_msg + "\n"); f.flush()
        if to_console: print(formatted_msg); sys.stdout.flush()

    def result(self, message: str):
        with open(self.results_file, 'a') as f: f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"); f.flush()
        self.log(f"RESULT: {message}", "RESULT")

    def progress(self, current: int, total: int, message: str = ""):
        if total == 0: return
        percent = current * 100 / total
        bar = '█' * int(50 * current / total) + '░' * (50 - int(50 * current / total))
        print(f"\rProgress: [{bar}] {percent:.1f}% ({current}/{total}) {message}", end='', flush=True)
        if current == total: print()

logger = Logger(CONFIG.log_file, CONFIG.results_file)
GA_FEATURE_CACHE = {}

class GPUPatternAnalyzer:
    def __init__(self, cache_size: int, device: str):
        self.cache = {}
        self.cache_size = cache_size
        self.prime_list = list(primerange(2, 1000))
        self.device = torch.device(device)
        logger.log(f"Initialized GPUPatternAnalyzer on device: {self.device}")

    def compute_repetend(self, n: int, base: int, max_length: int = 10000) -> List[int]:
        """Fixed version of compute_repetend with correct digit calculation order"""
        cache_key = (n, base)
        if cache_key in self.cache: 
            return self.cache[cache_key]
        if math.gcd(n, base) != 1: 
            return []
        
        remainder, digits, seen = 1, [], {}
        
        while remainder not in seen and len(digits) < max_length:
            seen[remainder] = len(digits)
            # Calculate digit BEFORE updating remainder
            digit = (remainder * base) // n
            digits.append(digit)
            remainder = (remainder * base) % n
        
        # Extract the repeating part
        result = digits[seen.get(remainder, 0):] if remainder in seen else digits
        
        # Update cache
        if len(self.cache) >= self.cache_size: 
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = result
        
        # Also save to global pattern storage
        if 'REPETEND_PATTERNS' in globals():
            REPETEND_PATTERNS[(n, base)] = result
        
        return result

    def extract_pattern_features(self, n: int, base: int) -> Dict:
        pattern = self.compute_repetend(n, base)
        if not pattern: return {'valid': False}
        normalized = np.array(pattern) / (base - 1) if base > 1 else np.array(pattern)
        features = {
            'valid': True, 'n': n, 'base': base, 'length': len(pattern),
            'mean': np.mean(normalized), 'std': np.std(normalized),
            'skew': stats.skew(normalized) if len(normalized) > 2 else 0.0,
            'kurtosis': stats.kurtosis(normalized) if len(normalized) > 3 else 0.0,
            'unique_digits': len(set(pattern)), 'digit_entropy': self._digit_entropy(pattern),
            'transition_matrix': self._compute_transition_matrix(pattern, base),
            'transition_entropy': self._transition_entropy(pattern, base),
            'mean_jump': self._mean_digit_jump(pattern), 'max_jump': self._max_digit_jump(pattern),
            'autocorr_lag1': self._autocorr_at_lag(normalized, 1), 'autocorr_lag2': self._autocorr_at_lag(normalized, 2),
            'autocorr_decay': self._autocorr_decay_rate(normalized),
            'spectral_peak_count': self._count_spectral_peaks_gpu(normalized),
            'dominant_freq': self._dominant_frequency_gpu(normalized),
            'spectral_entropy': self._spectral_entropy_gpu(normalized),
            'spectral_energy_concentration': self._spectral_energy_concentration_gpu(normalized),
            'lempel_ziv': self._lempel_ziv_complexity(pattern),
            'run_count': self._count_runs(pattern), 'longest_run': self._longest_run(pattern),
            'recurring_2grams': self._count_recurring_ngrams(pattern, 2),
            'recurring_3grams': self._count_recurring_ngrams(pattern, 3),
            'multiplicative_order': self._multiplicative_order(base, n),
            'order_ratio': self._multiplicative_order(base, n) / totient(n) if totient(n) > 0 else 0.0
        }
        return features

    # Internal helper methods for feature extraction (prefixed with _)
    def _spectral_entropy_gpu(self, norm: np.ndarray) -> float:
        if len(norm) < 2: return 0.0
        s = torch.abs(torch.fft.fft(torch.tensor(norm, dtype=torch.float32, device=self.device)))[:len(norm)//2]
        p = s**2; p /= (torch.sum(p) + 1e-10); p = p[p > 1e-10]
        return -torch.sum(p * torch.log(p)).cpu().item() if len(p) > 0 else 0.0
    def _count_spectral_peaks_gpu(self, norm: np.ndarray) -> int:
        if len(norm) < 3: return 0
        s = torch.abs(torch.fft.fft(torch.tensor(norm, dtype=torch.float32, device=self.device)))[:len(norm)//2]
        if len(s) < 3: return 0
        t = torch.mean(s) + torch.std(s); sc = s.cpu().numpy()
        return sum(1 for i in range(1, len(sc) - 1) if sc[i] > sc[i-1] and sc[i] > sc[i+1] and sc[i] > t.item())
    def _dominant_frequency_gpu(self, norm: np.ndarray) -> int:
        if len(norm) < 2: return 0
        s = torch.abs(torch.fft.fft(torch.tensor(norm, dtype=torch.float32, device=self.device)))[:len(norm)//2]
        return torch.argmax(s).cpu().item() if len(s) > 0 else 0
    def _spectral_energy_concentration_gpu(self, norm: np.ndarray) -> float:
        if len(norm) < 2: return 0.0
        s = torch.abs(torch.fft.fft(torch.tensor(norm, dtype=torch.float32, device=self.device)))[:len(norm)//2]
        if len(s) < 5: return 0.0
        p = s**2; total_p = torch.sum(p)
        if total_p == 0: return 0.0
        top_k = max(1, len(s) // 10); top_e = torch.sum(torch.topk(p, top_k).values)
        return (top_e / total_p).cpu().item()
    def _digit_entropy(self, p: List[int]) -> float:
        if not p: return 0.0
        probs = np.array(list(Counter(p).values())) / len(p)
        return -np.sum(probs * np.log(probs + 1e-10))
    def _compute_transition_matrix(self, p: List[int], b: int) -> np.ndarray:
        if len(p) < 2: return np.zeros((b, b))
        t = np.zeros((b, b)); [t.__setitem__((p[i], p[i+1]), t[p[i], p[i+1]] + 1) for i in range(len(p) - 1)]
        return t / (t.sum(axis=1)[:, np.newaxis] + 1e-10)
    def _transition_entropy(self, p: List[int], b: int) -> float:
        probs = self._compute_transition_matrix(p, b).flatten(); probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
    def _mean_digit_jump(self, p: List[int]) -> float: return np.mean([abs(p[i+1] - p[i]) for i in range(len(p)-1)]) if len(p) > 1 else 0.0
    def _max_digit_jump(self, p: List[int]) -> int: return max([abs(p[i+1] - p[i]) for i in range(len(p)-1)]) if len(p) > 1 else 0
    def _autocorr_at_lag(self, n: np.ndarray, lag: int) -> float:
        if len(n) <= lag: return 0.0
        c = np.corrcoef(n[:-lag], n[lag:])[0, 1]
        return c if not np.isnan(c) else 0.0
    def _autocorr_decay_rate(self, n: np.ndarray) -> float:
        if len(n) < 10: return 0.0
        ac = [abs(self._autocorr_at_lag(n, lag)) for lag in range(1, min(10, len(n)//2))]
        if len(ac) < 2: return 0.0
        try: return -np.polyfit(np.arange(1, len(ac) + 1), np.log(np.array(ac) + 1e-10), 1)[0]
        except: return 0.0
    def _lempel_ziv_complexity(self, p: List[int]) -> int: # Corrected & Efficient
        if not p: return 0
        s = "".join(map(str, p)); n = len(s); d, lp, c = {s[0]}, s[0], 1
        for i in range(1, n):
            cp = lp + s[i]
            if cp in d: lp = cp
            else: d.add(cp); c += 1; lp = s[i]
        return c
    def _count_runs(self, p: List[int]) -> int: 
        return 1 + sum(1 for i in range(1, len(p)) if p[i] != p[i-1]) if len(p)>1 else 1
    def _longest_run(self, p: List[int]) -> int:
        if not p: return 0
        mr, cr = 1, 1
        for i in range(1, len(p)): cr = cr + 1 if p[i] == p[i-1] else 1; mr = max(mr, cr)
        return mr
    def _count_recurring_ngrams(self, p: List[int], n: int) -> int:
        if len(p) < n: return 0
        return sum(1 for c in Counter(tuple(p[i:i+n]) for i in range(len(p)-n+1)).values() if c > 1)
    def _multiplicative_order(self, a: int, n: int) -> int:
        if math.gcd(a, n) != 1: return 0
        o, c = 1, a % n
        while c != 1 and o < n: c = (c * a) % n; o += 1
        return o if c == 1 else 0

REPETEND_PATTERNS = {}  # Store all computed repetends

class GeneticFeatureEvolver:
    def __init__(self, analyzer: GPUPatternAnalyzer, config: Config):
        self.analyzer = analyzer
        self.config = config
        self.population = []; self.fitness_history = []
        self.best_individual = None; self.best_fitness = float('-inf'); self.generation = 0
        logger.log(f"Initialized GeneticFeatureEvolver with PopSize={config.population_size}, Elite={config.elite_size}")

    def create_individual(self) -> Dict:
        feature_names = list(self.analyzer.extract_pattern_features(3, 10).keys())
        feature_names = [f for f in feature_names if isinstance(self.analyzer.extract_pattern_features(3,10)[f], (int, float))]
        return {
            'id': random.randint(1000000, 9999999),
            'feature_weights': {f: random.random() for f in feature_names},
            'feature_active': {f: random.random() > 0.3 for f in feature_names},
            'bases': sorted(random.sample(self.analyzer.prime_list[:50], random.randint(5, 30))),
            'aggregation': random.choice(['mean', 'median', 'std', 'max', 'min']),
            'normalization': random.choice(['none', 'zscore', 'minmax', 'log']),
            'use_pca': random.random() > 0.5, 'pca_components': random.randint(3, 10),
            'fitness': 0, 'birth_generation': self.generation
        }

    def evaluate_fitness(self, individual: Dict, test_data: List[Tuple[int, Dict]]) -> float:
        features, targets = [], []
        for n, factors in test_data:
            feat = self._extract_features_from_cache(n, individual)
            if feat is not None and feat.size > 0: features.append(feat); targets.append(len(factors))
        if len(features) < 10: return 0.0
        features, targets = np.array(features), np.array(targets)
        min_len = min(len(f) for f in features); features = np.array([f[:min_len] for f in features])
        if individual['use_pca'] and features.shape[1] > individual['pca_components']:
            features = PCA(n_components=min(individual['pca_components'], features.shape[1])).fit_transform(features)
        summary = np.mean(features, axis=1) if features.shape[1] > 1 else features[:, 0]
        if np.std(summary) < 1e-9 or np.std(targets) < 1e-9: return 0.0
        corr = abs(np.corrcoef(summary, targets)[0, 1]); corr = 0.0 if np.isnan(corr) else corr
        unique_omega = np.unique(targets); sep = 0.0
        if len(unique_omega) > 1:
            try:
                f_stat, _ = stats.f_oneway(*[summary[targets == o] for o in unique_omega])
                sep = np.log(f_stat + 1) / 10 if not np.isnan(f_stat) else 0.0
            except: pass
        penalty = 0.01 * sum(individual['feature_active'].values())
        fitness = corr + 0.3 * sep - penalty
        return fitness if not np.isnan(fitness) else 0.0

    def _extract_features_from_cache(self, n: int, ind: Dict) -> Optional[np.ndarray]:
        all_feats = []
        for base in ind['bases']:
            feats = GA_FEATURE_CACHE.get((n, base))
            if feats and feats['valid']:
                sel = [feats[fname] * ind['feature_weights'][fname] for fname, act in ind['feature_active'].items()
                       if act and fname in feats and isinstance(feats[fname], (int, float))]
                if sel: all_feats.append(sel)
        if not all_feats: return None
        ag = np.array(all_feats); agg_map = {'mean':np.mean,'median':np.median,'std':np.std,'max':np.max,'min':np.min}
        agged = agg_map[ind['aggregation']](ag, axis=0)
        if ind['normalization'] == 'zscore': return (agged - np.mean(agged)) / (np.std(agged) + 1e-10)
        if ind['normalization'] == 'minmax':
            min_v, max_v = np.min(agged), np.max(agged)
            return (agged - min_v) / (max_v - min_v + 1e-10)
        if ind['normalization'] == 'log': return np.log(np.abs(agged) + 1)
        return agged

    def evolve(self, test_data: List[Tuple[int, Dict]], start_time: float):
        if not self.population:
            logger.log("Initializing genetic population..."); self.population = [self.create_individual() for _ in range(self.config.population_size)]

        for gen in range(self.generation, self.config.generations):
            elapsed_h = (time.time() - start_time) / 3600
            if elapsed_h > self.config.runtime_hours: logger.log("Runtime limit reached. Stopping evolution.", "WARNING"); break
            if os.path.exists(self.config.stop_file):
                logger.log(f"Stop file '{self.config.stop_file}' detected. Stopping evolution.", "WARNING"); os.remove(self.config.stop_file); break
            
            gen_start = time.time()
            logger.log(f"\n{'='*60}\nGENERATION {gen} (Elapsed: {elapsed_h:.2f}/{self.config.runtime_hours:.2f} hrs)\n{'='*60}")
            
            # NOTE: Fitness evaluation is now extremely fast due to pre-computation.
            for ind in self.population: ind['fitness'] = self.evaluate_fitness(ind, test_data)
            
            self.population.sort(key=lambda x: x.get('fitness', 0), reverse=True)
            
            if self.population[0]['fitness'] > self.best_fitness:
                self.best_fitness = self.population[0]['fitness']; self.best_individual = self.population[0].copy()
                active = [k for k, v in self.best_individual['feature_active'].items() if v]
                logger.result(f"NEW BEST at generation {gen}: fitness={self.best_fitness:.4f}")
                logger.result(f"  Active features ({len(active)}): {active[:5]}...")
                logger.result(f"  Strategy: {len(self.best_individual['bases'])} bases, agg={self.best_individual['aggregation']}, norm={self.best_individual['normalization']}")

            fit_vals = [ind.get('fitness', 0) for ind in self.population]
            self.fitness_history.append({'gen': gen, 'best': np.max(fit_vals), 'mean': np.mean(fit_vals), 'std': np.std(fit_vals)})
            logger.log(f"Generation {gen} Summary: Best={np.max(fit_vals):.4f}, Mean={np.mean(fit_vals):.4f} ± {np.std(fit_vals):.4f}")
            logger.log("Top 3 individuals:"); [logger.log(f"  {i+1}. ID={ind['id']}, fitness={ind['fitness']:.4f}, active={sum(ind['feature_active'].values())}") for i, ind in enumerate(self.population[:3])]
            
            if gen % self.config.checkpoint_interval == 0 and gen > self.generation: 
                self.save_checkpoint(gen)
            
            new_pop = self.population[:self.config.elite_size]
            while len(new_pop) < self.config.population_size:
                p1, p2 = self._tournament_select(), self._tournament_select()
                child = self._crossover(p1, p2) if random.random() < self.config.crossover_rate else random.choice(self.population[:20]).copy()
                new_pop.append(self._mutate(child))
            
            self.population = new_pop; self.generation = gen + 1
            logger.log(f"Generation {gen} completed in {time.time() - gen_start:.2f} seconds")

    def save_checkpoint(self, generation: int):
        """Enhanced checkpoint saving with pattern data"""
        filename = f"sieve_echo_checkpoint_gen{generation}.pkl"
        checkpoint_data = {
            'generation': generation,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'population': self.population,
            'config': self.config,
            'ga_feature_cache': GA_FEATURE_CACHE
        }
        
        # FIX: Definire include_patterns o rimuovere il controllo
        include_patterns = True  # <-- AGGIUNGERE QUESTA LINEA
        if include_patterns and 'REPETEND_PATTERNS' in globals():
            checkpoint_data['repetend_patterns'] = REPETEND_PATTERNS
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.log(f"Enhanced checkpoint saved to {filename} (with {len(REPETEND_PATTERNS)} patterns)")



    def _crossover(self, p1, p2):
        c = {'id':random.randint(1e6,1e7-1),'birth_generation':self.generation,'parent1_id':p1['id'],'parent2_id':p2['id'],'feature_weights':{},'feature_active':{}}
        for fname in p1['feature_weights']:
            src = p1 if random.random() < 0.5 else p2
            c['feature_weights'][fname], c['feature_active'][fname] = src['feature_weights'][fname], src['feature_active'][fname]
        bases = sorted(list(set(p1['bases'] + p2['bases']))); num = (len(p1['bases']) + len(p2['bases'])) // 2
        c['bases'] = sorted(random.sample(bases, min(num, len(bases)))) if bases else []
        for k in ['aggregation', 'normalization', 'use_pca', 'pca_components']: c[k] = random.choice([p1[k], p2[k]])
        return c
    
    def _mutate(self, ind):
        if random.random() > self.config.mutation_rate: return ind
        m = ind.copy()
        for fname in m['feature_weights']:
            if random.random() < 0.1: m['feature_weights'][fname] = max(0, min(1, m['feature_weights'][fname] + random.gauss(0, 0.2)))
            if random.random() < 0.05: m['feature_active'][fname] = not m['feature_active'][fname]
        if random.random() < 0.2:
            if len(m['bases']) > 3 and random.random() < 0.5: m['bases'].remove(random.choice(m['bases']))
            else:
                new_b = random.choice(self.analyzer.prime_list[:50])
                if new_b not in m['bases']: m['bases'] = sorted(m['bases'] + [new_b])
        if random.random() < 0.1: m['aggregation'] = random.choice(['mean', 'median', 'std', 'max', 'min'])
        return m
    def _tournament_select(self, size=5): return max(random.sample(self.population, min(size, len(self.population))), key=lambda x: x.get('fitness', 0))

class PatternDecompositionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, max_factors: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 64))
        self.omega_predictor = nn.Linear(64, max_factors)
        self.prime_predictor = nn.Linear(64, 100) # Predict presence of first 100 primes
        self.power_predictor = nn.Linear(64, 5) # Predict max power
        self.to(CONFIG.device)
    def forward(self, x): return self.omega_predictor(self.encoder(x)), self.prime_predictor(self.encoder(x)), self.power_predictor(self.encoder(x))

class SieveEchoExplorer:
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = GPUPatternAnalyzer(config.pattern_cache_size, config.device)
        self.genetic_evolver = GeneticFeatureEvolver(self.analyzer, config)
        self.device = torch.device(config.device)
        self.results = {}
        self.start_time = time.time()
        logger.log("="*80 + "\nSIEVE ECHO EXPLORER INITIALIZED\n" + "="*80)

    def run_complete_exploration(self):
        logger.log(f"Starting {self.config.runtime_hours}-hour exploration...")
        if os.path.exists(self.config.stop_file):
            logger.log(f"Removing old stop file '{self.config.stop_file}'", "WARNING"); os.remove(self.config.stop_file)
        phases = [
            ("Pattern Uniqueness", self.test_pattern_uniqueness),
            ("Pattern Inheritance", self.test_pattern_inheritance),
            ("Genetic Discovery", self.run_genetic_discovery),
            ("Neural Decomposition", self.train_pattern_decomposition_network)
        ]
        for name, func in phases:
            if (time.time() - self.start_time)/3600 < self.config.runtime_hours:
                logger.log(f"\n{'='*60}\nPHASE: {name}\n{'='*60}"); func()
            else: logger.log(f"Skipping phase '{name}' - time limit reached"); break
        self.generate_final_report()

    def precompute_features_for_ga(self, test_data: List[Tuple[int, Dict]]):
        global GA_FEATURE_CACHE
        if GA_FEATURE_CACHE: logger.log("GA Feature Cache already populated. Skipping pre-computation."); return
        logger.log("Starting feature pre-computation for Genetic Algorithm...")
        tasks = [(n, b) for n, _ in test_data for b in self.analyzer.prime_list[:50]]
        for i, (n, b) in enumerate(tasks):
            if (n, b) not in GA_FEATURE_CACHE: GA_FEATURE_CACHE[(n, b)] = self.analyzer.extract_pattern_features(n, b)
            if (i+1) % 1000 == 0: logger.progress(i+1, len(tasks), "Computing features...")
        logger.progress(len(tasks), len(tasks), "Feature pre-computation complete.")

    def run_genetic_discovery(self):
        test_data = self._prepare_test_data(min(10000, self.config.max_n), 1000)
        self.precompute_features_for_ga(test_data)
        logger.log("\nStarting genetic evolution...")
        self.genetic_evolver.evolve(test_data, self.start_time)
        if self.genetic_evolver.best_individual:
            self.results['genetic_best'] = {'fitness': self.genetic_evolver.best_fitness, 'individual': self.genetic_evolver.best_individual}

    def _prepare_test_data(self, max_n, size):
        logger.log(f"Preparing test dataset: max_n={max_n}, sample_size={size}")
        data = set()
        primes = list(primerange(2, min(max_n, 10000)))
        [data.add(p) for p in random.sample(primes, min(len(primes), size // 4))]
        [data.add(random.choice(primes[:100])**random.randint(2,4)) for _ in range(size//4)]
        while len(data) < size: data.add(random.randint(2, max_n))
        return [(n, factorint(n)) for n in data if n < max_n]

    def test_pattern_uniqueness(self):
        base, p_to_n, collisions = 10, {}, []
        test_range = min(5000, self.config.max_n)
        for n in range(2, test_range):
            if math.gcd(n, base) == 1:
                pt = tuple(self.analyzer.compute_repetend(n, base))
                if pt in p_to_n: collisions.append((p_to_n[pt], n))
                else: p_to_n[pt] = n
        self.results['pattern_uniqueness'] = {'tested': test_range-2, 'collisions': len(collisions)}
        logger.result(f"Pattern Uniqueness (base 10): Tested {test_range-2} numbers, found {len(collisions)} collisions.")
        if len(collisions) == 0: logger.result("✓ PATTERN UNIQUENESS CONFIRMED for tested range.")
        else: logger.result("✗ PATTERN UNIQUENESS VIOLATED.")

    def test_pattern_inheritance(self):
        scores, base, num_tests = [], 10, 200
        primes = list(primerange(3, 100))
        for i in range(num_tests):
            p1, p2 = random.choice(primes), random.choice(primes)
            if p1==p2 or math.gcd(p1*p2, base) != 1: continue
            f1 = self.analyzer.extract_pattern_features(p1, base)
            f2 = self.analyzer.extract_pattern_features(p2, base)
            fc = self.analyzer.extract_pattern_features(p1*p2, base)
            if not all([f1['valid'], f2['valid'], fc['valid']]): continue
            lcm = math.lcm(f1['length'], f2['length'])
            scores.append({'period_ratio': fc['length'] / lcm if lcm > 0 else 0})
            if i < 3: logger.log(f"Inheritance Example: p1={p1}, p2={p2}. Period Ratio: {scores[-1]['period_ratio']:.4f}")
        ratios = [s['period_ratio'] for s in scores]
        crt_compliant = sum(1 for r in ratios if abs(r-1.0) < 0.01)
        self.results['pattern_inheritance'] = {'tested': len(scores), 'mean_ratio': np.mean(ratios), 'crt_compliance': crt_compliant/len(scores)}
        logger.result(f"Pattern Inheritance: Mean period ratio={np.mean(ratios):.4f} (CRT compliant: {100*crt_compliant/len(scores):.1f}%)")

    def train_pattern_decomposition_network(self):
        if 'genetic_best' not in self.results: logger.log("No GA result available, skipping NN training."); return
        best_ind = self.results['genetic_best']['individual']
        logger.log(f"Training NN with features from best GA individual (fitness: {self.results['genetic_best']['fitness']:.4f})")
        data = self._prepare_test_data(min(20000, self.config.max_n), 5000)
        X, y_omega, y_primes = [], [], []
        evolver = self.genetic_evolver # Use an instance to call the method
        for i, (n, factors) in enumerate(data):
            logger.progress(i, len(data), "Extracting NN features")
            feats = evolver._extract_features_from_cache(n, best_ind)
            if feats is not None and feats.size > 0:
                X.append(feats); y_omega.append(len(factors))
                pv = np.zeros(100); [pv.__setitem__(p, 1) for p in factors if p < 100]; y_primes.append(pv)
        if len(X) < 100: logger.log("Insufficient data for NN training."); return
        min_len = min(len(x) for x in X); X = torch.tensor(np.array([x[:min_len] for x in X]), dtype=torch.float32)
        y_omega = torch.tensor(y_omega, dtype=torch.long); y_primes = torch.tensor(np.array(y_primes), dtype=torch.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9); X, y_omega, y_primes = X.to(self.device), y_omega.to(self.device), y_primes.to(self.device)
        
        model = PatternDecompositionNetwork(X.shape[1]).to(self.device); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_acc = 0
        for epoch in range(100):
            model.train(); optimizer.zero_grad()
            omega_p, prime_p, _ = model(X)
            loss = F.cross_entropy(omega_p, y_omega) + 0.5 * F.binary_cross_entropy_with_logits(prime_p, y_primes)
            loss.backward(); optimizer.step()
            if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(X)[0].argmax(1) == y_omega).float().mean().item()
                    logger.log(f"Epoch {epoch+1}/100: Loss={loss.item():.4f}, Omega Acc={acc:.3f}")
                    best_acc = max(best_acc, acc)
        self.results['neural_training'] = {'final_accuracy': acc, 'best_accuracy': best_acc, 'dataset_size': len(X)}
        logger.result(f"NN Training Complete: Final ω(n) prediction accuracy = {acc:.3f}")

        # After training loop completes:
        if model is not None:  # Add after training
            accuracy_info = {'final': acc, 'best': best_acc}
            model_path = self.save_neural_model(model, optimizer, accuracy_info, X.shape[1])
            self.results['neural_training']['model_path'] = model_path
            
            # Clean up GPU memory
            del model, optimizer, X, y_omega, y_primes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_neural_model(self, model, optimizer, accuracy_info, input_dim):  # <-- AGGIUNGERE 'self'
        """Save the trained neural network model"""
        model_path = 'sieve_echo_neural_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_accuracy': accuracy_info['final'],
            'best_accuracy': accuracy_info['best'],
            'input_dim': input_dim,
            'architecture': {
                'hidden_dim': 256,
                'max_factors': 10
            }
        }, model_path)
        logger.result(f"Neural network model saved to {model_path}")
        return model_path

    def generate_final_report(self):
        runtime = (time.time() - self.start_time) / 3600
        logger.log("\n" + "="*80 + "\nSIEVE ECHO CONJECTURE - FINAL REPORT\n" + "="*80)
        logger.result(f"Total runtime: {runtime:.2f} hours")

        if 'pattern_uniqueness' in self.results:
            res = self.results['pattern_uniqueness']
            logger.result(f"Uniqueness: {res['collisions']} collisions in {res['tested']} numbers.")
        
        if 'pattern_inheritance' in self.results:
            res = self.results['pattern_inheritance']
            logger.result(f"Inheritance: {res['crt_compliance']*100:.1f}% CRT compliance over {res['tested']} semiprimes.")
        
        if 'genetic_best' in self.results:
            logger.result("\n--- Genetic Discovery Final Summary ---")
            self._interpret_best_individual(self.results['genetic_best']['individual'])
        
        if 'neural_training' in self.results:
            res = self.results['neural_training']
            logger.result(f"Neural Net: Reached {res['final_accuracy']:.3f} accuracy predicting ω(n).")        
        
        logger.log("\n" + "="*80)
        logger.log("CONCLUSIONS:")
        logger.log("-" * 40)
        
        conclusions = []
        
        if 'pattern_uniqueness' in self.results:
            if self.results['pattern_uniqueness']['collisions'] == 0:
                conclusions.append("✓ Patterns appear to uniquely encode n")
            else:
                conclusions.append("✗ Pattern uniqueness violated in some cases")
        
        if 'pattern_inheritance' in self.results:
            # FIX: Usare 'mean_ratio' invece di 'mean_period_ratio'
            # E aggiungere controllo se la chiave esiste
            if 'mean_ratio' in self.results['pattern_inheritance']:
                if abs(self.results['pattern_inheritance']['mean_ratio'] - 1.0) < 0.1:
                    conclusions.append("✓ Composite patterns follow CRT inheritance")
                else:
                    conclusions.append("? Pattern inheritance shows unexpected behavior")
            else:
                logger.log("WARNING: 'mean_ratio' not found in pattern_inheritance results", "WARNING")
        
        if 'genetic_best' in self.results:
            if self.results['genetic_best']['fitness'] > 0.5:
                conclusions.append("✓ Pattern features correlate strongly with factorization")
            else:
                conclusions.append("? Pattern-factorization correlation is weak")
        
        if 'neural_training' in self.results:
            if self.results['neural_training']['final_accuracy'] > 0.7:
                conclusions.append("✓ Neural networks can predict ω(n) from patterns")
            else:
                conclusions.append("? Neural prediction accuracy is limited")
        
        for conclusion in conclusions:
            logger.result(conclusion)
        
        if len([c for c in conclusions if c.startswith('✓')]) >= 3:
            logger.result("\nThe Sieve Echo Conjecture is STRONGLY SUPPORTED by empirical evidence.")
        elif len([c for c in conclusions if c.startswith('✓')]) >= 2:
            logger.result("\nThe Sieve Echo Conjecture shows PROMISING SUPPORT from empirical evidence.")
        else:
            logger.result("\nThe Sieve Echo Conjecture requires FURTHER INVESTIGATION.")
        
        logger.log("="*80)
        
        # Save final results
        with open('sieve_echo_final_results.json', 'w') as f:
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if torch.is_tensor(obj):
                    return obj.cpu().numpy().tolist()
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)               

         # ADDITION: Save all collected data
        logger.log("\nSaving all collected data...")
        
        # 1. Save final checkpoint
        if self.genetic_evolver.generation > 0:
            self.genetic_evolver.save_checkpoint(self.genetic_evolver.generation)
            logger.result(f"Final GA checkpoint saved (generation {self.genetic_evolver.generation})")
        
        # 2. Save pattern cache
        pattern_cache_file = 'sieve_echo_pattern_cache.pkl'
        with open(pattern_cache_file, 'wb') as f:
            pickle.dump({
                'ga_feature_cache': GA_FEATURE_CACHE,
                'repetend_patterns': REPETEND_PATTERNS if 'REPETEND_PATTERNS' in globals() else {}
            }, f)
        logger.result(f"Pattern cache saved to {pattern_cache_file} ({len(GA_FEATURE_CACHE)} features, {len(REPETEND_PATTERNS)} patterns)")
        
        # 3. Save results with enhanced data
        enhanced_results = dict(self.results)
        enhanced_results['metadata'] = {
            'runtime_hours': runtime,
            'max_n': self.config.max_n,
            'ga_generations': self.genetic_evolver.generation,
            'pattern_count': len(GA_FEATURE_CACHE),
            'repetend_count': len(REPETEND_PATTERNS) if 'REPETEND_PATTERNS' in globals() else 0
        }
        
        # 4. Save as both JSON and pickle for compatibility
        with open('sieve_echo_final_results.json', 'w') as f:
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if torch.is_tensor(obj):
                    return obj.cpu().numpy().tolist()
                return obj
            json.dump(enhanced_results, f, indent=2, default=convert)
        
        with open('sieve_echo_final_results.pkl', 'wb') as f:
            pickle.dump(enhanced_results, f)
        
        logger.result(f"\nFinal results saved to:")
        logger.result(f"  - JSON: sieve_echo_final_results.json")
        logger.result(f"  - Pickle: sieve_echo_final_results.pkl")
        logger.result(f"  - Pattern cache: {pattern_cache_file}")
        logger.result(f"  - Detailed log: {CONFIG.log_file}")
        logger.result(f"  - Results summary: {CONFIG.results_file}")

    def _interpret_best_individual(self, ind: Dict):
        active = {k: ind['feature_weights'][k] for k, v in ind['feature_active'].items() if v}
        if not active: logger.result("GA found no optimal active features."); return
        sorted_feats = sorted(active.items(), key=lambda i: i[1], reverse=True)
        logger.result("Optimal strategy relies on these key features (ranked by importance):")
        [logger.result(f"  {i+1}. {f} (weight: {w:.3f})") for i, (f, w) in enumerate(sorted_feats[:5])]
        logger.result(f"Base Strategy: Use {len(ind['bases'])} primes from {min(ind['bases'])} to {max(ind['bases'])}.")
        logger.result(f"Data Strategy: Aggregate features via '{ind['aggregation']}', normalize with '{ind['normalization']}'. PCA used: {ind['use_pca']}.")

def main():
    parser = argparse.ArgumentParser(description='Sieve Echo Complete Framework v3.1')
    parser.add_argument('--hours', type=float, default=16.0, help='Runtime hours')
    parser.add_argument('--max_n', type=int, default=50000, help='Maximum n')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint file')
    args = parser.parse_args(); CONFIG.runtime_hours = args.hours; CONFIG.max_n = args.max_n
    
    explorer = SieveEchoExplorer(CONFIG)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.log(f"Loading checkpoint: {args.checkpoint}")
        try:
            with open(args.checkpoint, 'rb') as f:
                cp = pickle.load(f)
                explorer.genetic_evolver.population = cp.get('population', [])
                explorer.genetic_evolver.best_individual = cp.get('best_individual')
                explorer.genetic_evolver.best_fitness = cp.get('best_fitness', float('-inf'))
                explorer.genetic_evolver.generation = cp.get('generation', 0)
                explorer.genetic_evolver.fitness_history = cp.get('fitness_history', [])
                global GA_FEATURE_CACHE; GA_FEATURE_CACHE = cp.get('ga_feature_cache', {})
                logger.log(f"Checkpoint loaded. Resuming from generation {cp.get('generation', 0)}")
                
                # ADDITION: Load pattern data if available
                if 'repetend_patterns' in cp:
                    global REPETEND_PATTERNS
                    REPETEND_PATTERNS = cp['repetend_patterns']
                    logger.log(f"Loaded {len(REPETEND_PATTERNS)} repetend patterns")
                    
        except Exception as e:
            logger.log(f"Failed to load checkpoint: {e}", "ERROR")
    
    try:
        explorer.run_complete_exploration()
    except KeyboardInterrupt:
        logger.log("\nExploration interrupted. Saving all data...", "WARNING")
        explorer.generate_final_report()  # Use enhanced version
    except Exception as e:
        logger.log(f"An unexpected error occurred: {e}", "CRITICAL")
        logger.log(traceback.format_exc(), "ERROR")
        explorer.generate_final_report()  # Use enhanced version
    else:
        # Normal completion - ensure everything is saved
        explorer.generate_final_report()  # Use enhanced version

if __name__ == "__main__":
    main()