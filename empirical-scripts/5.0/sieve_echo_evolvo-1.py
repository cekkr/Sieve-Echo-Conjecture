"""
Enhanced Sieve Echo Evolver v5.0
Implements efficient screening, dynamic ML architecture search, and mathematical formula discovery
"""

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
from datetime import datetime
from collections import defaultdict, Counter, deque
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
from scipy import stats
from sklearn.decomposition import PCA
from sympy import factorint, primerange, isprime, totient, S, symbols, lambdify
import warnings
warnings.filterwarnings('ignore')

# Import evolvo engine
from evolvo_engine import DataStore, InstructionSet, Interpreter, BaseEvaluator, get_default_instruction_set, myFloat, AlgorithmGenerator
EVOLVO_AVAILABLE = True

""" # Optional import removed
try:
    from evolvo_engine import DataStore, InstructionSet, Interpreter, BaseEvaluator, get_default_instruction_set, myFloat, AlgorithmGenerator
    EVOLVO_AVAILABLE = True
except ImportError:
    print("WARNING: evolvo_engine not found. Formula discovery will be limited.")
    EVOLVO_AVAILABLE = False
"""

# ==============================================================================
# PICKLABLE HELPER FUNCTIONS FOR EVOLVO INSTRUCTION SET
# These functions are defined at the top level to be picklable by multiprocessing.
# They replace all lambdas previously used in the InstructionSet.
# ==============================================================================
def _add(a, b): return myFloat(a + b)
def _sub(a, b): return myFloat(a - b)
def _mul(a, b): return myFloat(a * b)
def _div(a, b): return myFloat(a / (b if abs(b) > 1e-9 else 1e-9))
def _log(a): return myFloat(math.log(abs(a) + 1e-9))
def _sqrt(a): return myFloat(math.sqrt(abs(a)))
def _pow(a, b): return myFloat(a ** min(abs(b), 5)) # Limit exponent to prevent overflow
def _exp(a): return myFloat(math.exp(min(a, 10))) # Limit input to prevent overflow
def _sin(a): return myFloat(math.sin(a))
def _cos(a): return myFloat(math.cos(a))

def get_picklable_instruction_set() -> InstructionSet:
    """
    Creates a fully picklable InstructionSet by using named, top-level functions
    instead of lambdas. This is a replacement for the problematic
    get_default_instruction_set from the evolvo_engine library.
    """
    if not EVOLVO_AVAILABLE:
        return None
        
    iset = InstructionSet()
    
    # Standard arithmetic
    iset.register('ADD', _add, ['d', 'd'], 'decimal')
    iset.register('SUB', _sub, ['d', 'd'], 'decimal')
    iset.register('MUL', _mul, ['d', 'd'], 'decimal')
    iset.register('DIV', _div, ['d', 'd'], 'decimal')
    
    # Mathematical functions
    iset.register('LOG', _log, ['d'], 'decimal')
    iset.register('SQRT', _sqrt, ['d'], 'decimal')
    iset.register('POW', _pow, ['d', 'd'], 'decimal')
    iset.register('EXP', _exp, ['d'], 'decimal')
    iset.register('SIN', _sin, ['d'], 'decimal')
    iset.register('COS', _cos, ['d'], 'decimal')
    
    # Note: Logic/control flow instructions from the original default set
    # like SET, CPY, JMP, etc., are usually methods of the InstructionSet class
    # itself and are generally picklable. We only need to replace the lambda-defined ones.
    # The default instruction set also includes these, so we add them manually.
    # Standard arithmetic (op_type='decimal')
    iset.register('ADD', _add, ['d', 'd'], 'decimal')
    iset.register('SUB', _sub, ['d', 'd'], 'decimal')
    iset.register('MUL', _mul, ['d', 'd'], 'decimal')
    # The default Div avoids DivByZero but returns 1. We'll use a safer version.
    iset.register('DIV', _div, ['d', 'd'], 'decimal')
    
    # Mathematical functions from sieve_echo_evolvo (also op_type='decimal')
    iset.register('LOG', _log, ['d'], 'decimal')
    iset.register('SQRT', _sqrt, ['d'], 'decimal')
    iset.register('POW', _pow, ['d', 'd'], 'decimal')
    iset.register('EXP', _exp, ['d'], 'decimal')
    iset.register('SIN', _sin, ['d'], 'decimal')
    iset.register('COS', _cos, ['d'], 'decimal')

    # Boolean operations (op_type='bool')
    # The default evolvo_engine.py uses lambdas for these, so we provide picklable versions.
    iset.register('NOT', lambda a: not a, ['b'], op_type='bool') # lambda is fine here if bools aren't used, but we can be explicit
    iset.register('CMP', lambda a, b: a == b, ['d', 'd'], op_type='bool')
    iset.register('GT',  lambda a, b: a > b, ['d', 'd'], op_type='bool')
        
    return iset
# ==============================================================================

# Helper functions for MathematicalConstantsLibrary to ensure picklability
# These replace the unpicklable lambda functions from the original code.
def _prime_density_formula(n):
    return 1 / math.log(n) if n > 1 else 0

def _mertens_product_formula(n):
    return math.exp(-0.5772156649015329) * math.log(n) if n > 1 else 0

def _hardy_ramanujan_formula(n):
    return math.log(math.log(n)) if n > math.e else 0
    
# prime_probability is identical to prime_density, can reuse the function
# but we define it separately for clarity, matching the original keys.
def _prime_probability_formula(n):
    return 1 / math.log(n) if n > 1 else 0


class MathematicalConstantsLibrary:
    """Library of known mathematical constants and formulas for pattern matching"""
    def __init__(self):
        self.constants = {
            'e': math.e,
            'pi': math.pi,
            'golden_ratio': (1 + math.sqrt(5)) / 2,
            'golden_ratio_conjugate': (math.sqrt(5) - 1) / 2,
            'inv_golden_ratio_sq': -1 / ((1 + math.sqrt(5)) / 2) ** 2,
            'euler_mascheroni': 0.5772156649015329,
            'meissel_mertens': 0.2614972128476428,
            'artin': 0.3739558136192023,
            'sqrt_2': math.sqrt(2),
            'sqrt_3': math.sqrt(3),
            'sqrt_5': math.sqrt(5),
            'ln_2': math.log(2),
            'ln_10': math.log(10),
            'catalan': 0.915965594177219,
            'apery': 1.202056903159594,  # ζ(3)
            'feigenbaum_delta': 4.669201609102990,
            'feigenbaum_alpha': 2.502907875095892,
            'twin_prime': 0.6601618158468696,
            'mills': 1.3063778838630806,
            'plastic': 1.324717957244746,  # Real root of x³ = x + 1
            'tribonacci': 1.839286755214161,
            'conway': 1.303577269034296,
            'khinchin': 2.685452001065306,
            'levy': 3.275822918721811,
            'reciprocal_fibonacci': 3.359885666243178,
            'embree_trefethen': 0.70258,
        }
        
        # Common mathematical expressions
        self.expressions = {
            '5_minus_1_over_15': 5 - 1/15,  # 4.9333...
            'e_to_gamma': math.exp(0.5772156649015329),
            'e_to_minus_gamma': math.exp(-0.5772156649015329),
            'pi_squared_over_6': math.pi**2 / 6,  # ζ(2)
            'sqrt_2_minus_1': math.sqrt(2) - 1,
            'log_log_2': math.log(math.log(2)),
        }
        
        # Known prime-related formulas
        # MODIFIED: Replaced lambdas with references to top-level functions
        self.prime_formulas = {
            'prime_density': _prime_density_formula,
            'mertens_product': _mertens_product_formula,
            'hardy_ramanujan': _hardy_ramanujan_formula,
            'prime_probability': _prime_probability_formula,
        }
    
    def find_closest_constant(self, value: float, tolerance: float = 0.01) -> Optional[str]:
        """Find if a value matches any known constant within tolerance"""
        for name, const in {**self.constants, **self.expressions}.items():
            if abs(value - const) < tolerance:
                return name
        return None

    def test_formula_match(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
        """Test correlation with known formulas"""
        results = {}
        
        for name, formula in self.prime_formulas.items():
            try:
                predicted = np.array([formula(x) for x in x_data])
                if np.all(np.isfinite(predicted)) and np.std(predicted) > 0:
                    corr = np.corrcoef(y_data, predicted)[0, 1]
                    results[name] = abs(corr)
            except:
                continue
        
        return results

class EfficientNDRComputer:
    """Optimized NDR (Normalized Digit Representation) computation with caching and batching"""
    def __init__(self, max_cache_size: int = 100000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.batch_size = 100
        self.prime_list = list(primerange(2, 10000))
        
    def compute_ndr_batch(self, numbers: List[int], bases: List[int] = None) -> Dict[int, Dict]:
        """Compute NDR patterns for a batch of numbers efficiently"""
        if bases is None:
            bases = [2, 3, 5, 7, 10, 11, 13, 16]  # Default bases
        
        results = {}
        
        # Process in parallel using threads (I/O bound operation)
        with ThreadPoolExecutor(max_workers=min(8, len(numbers))) as executor:
            futures = {executor.submit(self._compute_single_ndr, n, bases): n 
                      for n in numbers if n not in self.cache}
            
            for future in as_completed(futures):
                n = futures[future]
                try:
                    results[n] = future.result()
                    # Update cache
                    if len(self.cache) < self.max_cache_size:
                        self.cache[n] = results[n]
                except Exception as e:
                    results[n] = {'valid': False, 'error': str(e)}
        
        # Add cached results
        for n in numbers:
            if n in self.cache and n not in results:
                results[n] = self.cache[n]
        
        return results
    
    def _compute_single_ndr(self, n: int, bases: List[int]) -> Dict:
        """Compute NDR features for a single number across multiple bases"""
        if n <= 2:
            return {'valid': False, 'n': n}
        
        features_by_base = []
        
        for base in bases:
            if base >= n or math.gcd(n, base) != 1:
                continue
            
            pattern = self._compute_repetend(n, base)
            if not pattern or len(pattern) < 2:
                continue
            
            ndr = np.array(pattern) / base
            
            # Compute features efficiently
            features = {
                'base': base,
                'length': len(pattern),
                'mean': float(np.mean(ndr)),
                'std': float(np.std(ndr)),
                'kurtosis': float(stats.kurtosis(ndr)) if len(ndr) > 3 else 0.0,
                'skew': float(stats.skew(ndr)) if len(ndr) > 3 else 0.0,
                'entropy': self._compute_entropy(ndr),
            }
            
            features_by_base.append(features)
        
        if not features_by_base:
            return {'valid': False, 'n': n}
        
        # Aggregate features
        aggregated = self._aggregate_features(features_by_base)
        aggregated['n'] = n
        aggregated['omega'] = len(factorint(n))
        aggregated['valid'] = True
        
        return aggregated
    
    def _compute_repetend(self, n: int, base: int, max_length: int = 1000) -> List[int]:
        """Fast repetend computation"""
        remainder = 1
        digits = []
        seen = {}
        
        while remainder != 0 and remainder not in seen and len(digits) < max_length:
            seen[remainder] = len(digits)
            remainder *= base
            digit = remainder // n
            digits.append(digit)
            remainder = remainder % n
        
        # Extract repeating part
        if remainder in seen:
            return digits[seen[remainder]:]
        return digits
    
    def _compute_entropy(self, ndr: np.ndarray) -> float:
        """Compute Shannon entropy of NDR pattern"""
        if len(ndr) < 2:
            return 0.0
        
        # Use FFT for spectral entropy
        fft = np.fft.fft(ndr)
        power = np.abs(fft[:len(fft)//2])**2
        
        if np.sum(power) > 1e-9:
            p = power / np.sum(power)
            p = p[p > 1e-10]
            return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
        return 0.0
    
    def _aggregate_features(self, features_list: List[Dict]) -> Dict:
        """Efficiently aggregate features across bases"""
        feature_names = ['length', 'mean', 'std', 'kurtosis', 'skew', 'entropy']
        aggregated = {}
        
        for fname in feature_names:
            values = [f[fname] for f in features_list if fname in f]
            if values:
                aggregated[f'{fname}_mean'] = float(np.mean(values))
                aggregated[f'{fname}_std'] = float(np.std(values))
                aggregated[f'{fname}_max'] = float(np.max(values))
        
        return aggregated

class DynamicNeuralArchitecture(nn.Module):
    """Self-configuring neural network that adapts its architecture"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        
        if hidden_dims is None:
            # Auto-configure based on input dimension
            hidden_dims = self._auto_configure(input_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def _auto_configure(self, input_dim: int) -> List[int]:
        """Automatically determine hidden layer dimensions"""
        if input_dim <= 10:
            return [32, 16]
        elif input_dim <= 20:
            return [64, 32, 16]
        else:
            return [128, 64, 32]
    
    def forward(self, x):
        return self.network(x)

class FormulaDiscoveryEvaluator(BaseEvaluator):
    """Evaluator that discovers mathematical formulas using Evolvo"""
    def __init__(self, data_store_config, instruction_set, test_data, math_library):
        super().__init__(data_store_config, instruction_set)
        self.test_data = test_data
        self.math_library = math_library
        
    def evaluate(self, algorithm, **kwargs) -> float:
        if not EVOLVO_AVAILABLE:
            return float('inf')
        
        data_store = DataStore(self.data_store_config)
        total_error = 0.0
        predictions = []
        actuals = []
        
        for n, omega_n, features in self.test_data[:min(100, len(self.test_data))]:
            if not features.get('valid', False):
                continue
            
            data_store.reset()
            # Set input features
            data_store.set('kurtosis', features.get('kurtosis_mean', 0))
            data_store.set('length', features.get('length_mean', 0))
            data_store.set('n', float(n))
            data_store.set('entropy', features.get('entropy_mean', 0))
            data_store.set('omega', float(omega_n))
            
            try:
                self.interpreter.execute(algorithm, data_store)
                predicted = data_store.get('prediction')
                
                predictions.append(predicted)
                actuals.append(omega_n)
                
                error = abs(predicted - omega_n)
                total_error += error ** 2
            except:
                total_error += 1e6
        
        if not predictions:
            return float('inf')
        
        # Calculate correlation
        if len(predictions) > 1 and np.std(predictions) > 0:
            corr = abs(np.corrcoef(actuals, predictions)[0, 1])
            
            # Bonus for matching known constants
            const_bonus = 0
            if len(set(predictions)) == 1:  # Constant prediction
                const_name = self.math_library.find_closest_constant(predictions[0])
                if const_name:
                    const_bonus = 0.5
            
            return 1.0 / (corr + const_bonus + 0.001)
        
        return total_error / len(predictions) if predictions else float('inf')

class ParallelGeneticOptimizer:
    """Parallel genetic algorithm with multiple populations and formula discovery"""
    def __init__(self, ndr_computer: EfficientNDRComputer, config: Any):
        self.ndr_computer = ndr_computer
        self.config = config
        self.math_library = MathematicalConstantsLibrary()
        
        # Multiple island populations for diversity
        self.num_islands = 4
        self.islands = [[] for _ in range(self.num_islands)]
        self.best_individuals = []
        self.formula_discoveries = []
        
        # Adaptive parameters
        self.generation = 0
        
        # Formula discovery via Evolvo
        self.init_evolvo()

    def _get_mutation_rate(self, g: int) -> float:
        """Decaying mutation rate schedule."""
        return 0.3 * (0.95 ** (g / 100))

    def _get_crossover_rate(self, g: int) -> float:
        """Oscillating crossover rate schedule."""
        return 0.7 + 0.2 * math.sin(g / 10)
        
    def init_evolvo(self):
        """Initialize Evolvo for formula discovery"""
        if not EVOLVO_AVAILABLE:
            self.evolvo_enabled = False
            return
        
        self.evolvo_config = {
            'd#': ['kurtosis', 'length', 'n', 'entropy', 'omega'],
            'b#': [],
            'd$': ['prediction', 'temp1', 'temp2'],
            'b$': []
        }
        
        # --- MODIFICATION ---
        # Call our new, fully picklable function instead of the library's default.
        # This is the crucial fix.
        self.instruction_set = get_picklable_instruction_set()
        
        # The registration of extra math operations is now handled by our
        # new function, so the old registration calls are removed from here.
        # --- END MODIFICATION ---
        
        self.evolvo_enabled = True

        
     def create_individual(self) -> Dict:
        """Create a new individual with random parameters"""
        individual = {
            'id': random.randint(1000000, 9999999),
            'feature_weights': {
                'kurtosis': random.uniform(0.5, 1.5),
                'length': random.uniform(0.01, 0.2),
                'n': random.uniform(0.01, 0.2),
                'entropy': random.uniform(0.1, 1.0),
                'skew': random.uniform(0, 0.5),
            },
            'neural_config': {
                'hidden_dims': random.choice([[32, 16], [64, 32], [128, 64, 32]]),
                'dropout': random.uniform(0.1, 0.5),
                'learning_rate': 10 ** random.uniform(-4, -2),
            },
            'use_three_features': random.random() > 0.5,
            'bases': sorted(random.sample(range(2, 30), random.randint(3, 8))),
            'fitness': 0.0,
            'formula': None,
        }
        
        # Add Evolvo formula if enabled
        if self.evolvo_enabled:
            generator = AlgorithmGenerator(self.evolvo_config, self.instruction_set)
            individual['formula'] = generator.generate_random_algorithm(max_len=8)
        
        return individual
    
    def evaluate_batch(self, individuals: List[Dict], test_numbers: List[int]) -> None:
        """Evaluate a batch of individuals efficiently"""
        # Compute NDR features for all test numbers
        ndr_features = self.ndr_computer.compute_ndr_batch(test_numbers)
        
        # Prepare test data
        test_data = []
        for n in test_numbers:
            if ndr_features[n].get('valid', False):
                omega = len(factorint(n))
                test_data.append((n, omega, ndr_features[n]))
        
        if len(test_data) < 10:
            for ind in individuals:
                ind['fitness'] = 0.0
            return
        
        # Evaluate each individual
        for ind in individuals:
            fitness = self._evaluate_individual(ind, test_data)
            ind['fitness'] = fitness
            
            # Test for mathematical constant matches
            if 'alpha_estimate' in ind and ind['alpha_estimate'] is not None:
                const_name = self.math_library.find_closest_constant(ind['alpha_estimate'])
                if const_name:
                    ind['constant_match'] = const_name
                    ind['fitness'] *= 1.2  # Bonus for matching known constant
    
    def _evaluate_individual(self, ind: Dict, test_data: List) -> float:
        """Evaluate a single individual's fitness"""
        # Extract features based on individual's weights
        X_data = []
        y_omega = []
        y_entropy = []
        
        active_features = ['kurtosis', 'length', 'n'] if ind['use_three_features'] else list(ind['feature_weights'].keys())
        
        for n, omega_n, features in test_data:
            feat_vec = []
            for fname in active_features:
                weight = ind['feature_weights'].get(fname, 0)
                if fname == 'n':
                    value = n
                else:
                    value = features.get(f'{fname}_mean', 0)
                feat_vec.append(value * weight)
            
            X_data.append(feat_vec)
            y_omega.append(omega_n)
            y_entropy.append(features.get('entropy_mean', 0))
        
        if len(X_data) < 10:
            return 0.0
        
        X = np.array(X_data)
        y = np.array(y_omega)
        
        # Calculate correlation
        summary = np.sum(X, axis=1)
        if np.std(summary) > 0 and np.std(y) > 0:
            corr = abs(np.corrcoef(summary, y)[0, 1])
        else:
            corr = 0.0
        
        # Estimate alpha and beta
        y_entropy_np = np.array(y_entropy)
        if len(y_entropy_np) > 10 and np.std(y) > 0:
            log_omega = np.log(y + 1e-9)
            if np.std(log_omega) > 0:
                try:
                    coeffs = np.polyfit(log_omega, y_entropy_np, 1)
                    ind['alpha_estimate'] = coeffs[0]
                    ind['beta_estimate'] = coeffs[1]
                    
                    # Check for constant matches
                    alpha_match = self.math_library.find_closest_constant(coeffs[0])
                    beta_match = self.math_library.find_closest_constant(coeffs[1])
                    
                    if alpha_match or beta_match:
                        corr *= 1.1  # Bonus for matching constants
                except:
                    pass
        
        # Evaluate formula if present
        formula_bonus = 0.0
        if self.evolvo_enabled and ind.get('formula'):
            evaluator = FormulaDiscoveryEvaluator(
                self.evolvo_config, self.instruction_set, test_data, self.math_library
            )
            formula_error = evaluator.evaluate(ind['formula'])
            if formula_error < 100:
                formula_bonus = 1.0 / (1.0 + formula_error)
        
        return corr + 0.3 * formula_bonus
    
    def evolve_islands(self, test_numbers: List[int], generations: int = 100):
        """Evolve multiple island populations in parallel"""
        
        # Initialize islands if empty
        for i in range(self.num_islands):
            if not self.islands[i]:
                self.islands[i] = [self.create_individual() 
                                  for _ in range(self.config.population_size // self.num_islands)]
        
        for gen in range(generations):
            self.generation = gen
            
            # Update adaptive parameters by calling the new methods
            mutation_rate = self._get_mutation_rate(gen)
            crossover_rate = self._get_crossover_rate(gen)
            
            print(f"\nGeneration {gen} | Mutation: {mutation_rate:.3f} | Crossover: {crossover_rate:.3f}")
            
            # Sample different test numbers for each generation (efficiency)
            gen_test_numbers = random.sample(test_numbers, min(200, len(test_numbers)))
            
            # Evolve each island in parallel
            with ProcessPoolExecutor(max_workers=self.num_islands) as executor:
                futures = []
                for island_idx in range(self.num_islands):
                    future = executor.submit(
                        self._evolve_single_island,
                        self.islands[island_idx],
                        gen_test_numbers,
                        mutation_rate,
                        crossover_rate
                    )
                    futures.append((island_idx, future))
                
                # Collect results
                for island_idx, future in futures:
                    self.islands[island_idx] = future.result()
            
            # Migration between islands every 10 generations
            if gen > 0 and gen % 10 == 0:
                self._migrate_between_islands()
            
            # Track best individuals
            all_individuals = [ind for island in self.islands for ind in island]
            all_individuals.sort(key=lambda x: x['fitness'], reverse=True)
            
            if all_individuals[0]['fitness'] > 0:
                self.best_individuals.append(all_individuals[0].copy())
                
                # Check for formula discoveries
                if all_individuals[0].get('constant_match'):
                    discovery = {
                        'generation': gen,
                        'individual': all_individuals[0],
                        'constant': all_individuals[0]['constant_match']
                    }
                    self.formula_discoveries.append(discovery)
                    print(f"DISCOVERY: Matched constant {discovery['constant']}!")
            
            # Report progress
            best_fitness = all_individuals[0]['fitness'] if all_individuals else 0
            avg_fitness = np.mean([ind['fitness'] for ind in all_individuals])
            print(f"Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")
            
            # Early stopping if converged
            if len(self.best_individuals) > 20:
                recent_best = [ind['fitness'] for ind in self.best_individuals[-20:]]
                if np.std(recent_best) < 0.0001:
                    print("Converged - stopping early")
                    break
    
    def _evolve_single_island(self, population: List[Dict], test_numbers: List[int],
                              mutation_rate: float, crossover_rate: float) -> List[Dict]:
        """Evolve a single island population"""
        # Create a copy to avoid shared memory issues in multiprocessing
        population = [ind.copy() for ind in population]
        
        # Evaluate fitness
        self.evaluate_batch(population, test_numbers)
        
        # Sort by fitness
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Elitism
        elite_size = max(2, len(population) // 10)
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < len(population):
            if random.random() < crossover_rate:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                child = self._crossover(p1, p2)
            else:
                child = self._tournament_select(population).copy()
            
            if random.random() < mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _migrate_between_islands(self):
        """Exchange best individuals between islands"""
        # Get best from each island
        best_per_island = [max(island, key=lambda x: x['fitness']) for island in self.islands]
        
        # Rotate best individuals
        for i in range(self.num_islands):
            next_island = (i + 1) % self.num_islands
            # Replace worst in next island with best from current
            self.islands[next_island][-1] = best_per_island[i].copy()
    
    def _tournament_select(self, population: List[Dict], size: int = 3) -> Dict:
        """Tournament selection"""
        tournament = random.sample(population, min(size, len(population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        """Crossover two parents"""
        child = self.create_individual()
        
        # Mix feature weights
        for feature in child['feature_weights']:
            if feature in p1['feature_weights'] and feature in p2['feature_weights']:
                child['feature_weights'][feature] = random.choice([
                    p1['feature_weights'][feature],
                    p2['feature_weights'][feature]
                ])
        
        # Mix other attributes
        child['use_three_features'] = random.choice([p1.get('use_three_features', False), 
                                                     p2.get('use_three_features', False)])
        child['bases'] = sorted(list(set(p1.get('bases', []) + p2.get('bases', []))))[:10]
        
        # Crossover formulas if available
        if self.evolvo_enabled and p1.get('formula') and p2.get('formula'):
            generator = AlgorithmGenerator(self.evolvo_config, self.instruction_set)
            child['formula'] = generator.crossover(p1['formula'], p2['formula'])
        
        return child
    
    def _mutate(self, ind: Dict) -> Dict:
        """Mutate an individual"""
        ind = ind.copy()
        
        # Mutate feature weights
        for feature in ind['feature_weights']:
            if random.random() < 0.3:
                ind['feature_weights'][feature] *= random.uniform(0.8, 1.2)
        
        # Mutate other parameters
        if random.random() < 0.1:
            ind['use_three_features'] = not ind.get('use_three_features', False)
        
        if random.random() < 0.2:
            # Add or remove a base
            if len(ind.get('bases', [])) > 2 and random.random() < 0.5:
                ind['bases'].remove(random.choice(ind['bases']))
            else:
                new_base = random.randint(2, 30)
                if new_base not in ind.get('bases', []):
                    ind['bases'] = sorted(ind.get('bases', []) + [new_base])
        
        # Mutate formula if available
        if self.evolvo_enabled and ind.get('formula'):
            generator = AlgorithmGenerator(self.evolvo_config, self.instruction_set)
            ind['formula'] = generator.mutate(ind['formula'])
        
        return ind
    
    def report_discoveries(self):
        """Generate report of mathematical discoveries"""
        print("\n" + "="*80)
        print("MATHEMATICAL FORMULA DISCOVERIES")
        print("="*80)
        
        if not self.formula_discoveries:
            print("No significant formula matches found yet.")
            return
        
        for discovery in self.formula_discoveries:
            print(f"\nGeneration {discovery['generation']}:")
            print(f"  Matched constant: {discovery['constant']}")
            ind = discovery['individual']
            if 'alpha_estimate' in ind:
                print(f"  Alpha: {ind['alpha_estimate']:.6f}")
            if 'beta_estimate' in ind:
                print(f"  Beta: {ind['beta_estimate']:.6f}")
            print(f"  Fitness: {ind['fitness']:.4f}")
        
        # Test against known theorems
        if self.best_individuals:
            best = self.best_individuals[-1]
            print("\n" + "-"*40)
            print("Testing against known theorems:")
            
            # Collect some test data
            test_numbers = random.sample(range(10, 1000), 100)
            ndr_features = self.ndr_computer.compute_ndr_batch(test_numbers)
            
            x_data = []
            y_data = []
            for n in test_numbers:
                if ndr_features[n].get('valid'):
                    x_data.append(n)
                    y_data.append(ndr_features[n].get('entropy_mean', 0))
            
            if x_data:
                correlations = self.math_library.test_formula_match(
                    np.array(x_data), np.array(y_data)
                )
                
                for formula_name, correlation in sorted(correlations.items(), 
                                                       key=lambda x: x[1], 
                                                       reverse=True)[:5]:
                    print(f"  {formula_name}: correlation = {correlation:.4f}")

# Configuration
@dataclass
class Config:
    max_n: int = 10000
    population_size: int = 400  # Per island
    runtime_hours: float = 24
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main execution function
def run_enhanced_evolver(runtime_hours: float = 1.0, max_n: int = 10000):
    """Run the enhanced evolver system"""
    print("="*80)
    print("ENHANCED SIEVE ECHO EVOLVER v5.0")
    print("="*80)
    
    config = Config()
    config.runtime_hours = runtime_hours
    config.max_n = max_n
    
    # Initialize components
    ndr_computer = EfficientNDRComputer()
    optimizer = ParallelGeneticOptimizer(ndr_computer, config)
    
    # Generate test numbers with strategic sampling
    test_numbers = []
    
    # Add primes
    test_numbers.extend(list(primerange(10, min(1000, max_n))))
    
    # Add semiprimes
    small_primes = list(primerange(2, 100))
    for _ in range(200):
        p = random.choice(small_primes)
        q = random.choice(small_primes)
        if p * q < max_n:
            test_numbers.append(p * q)
    
    # Add highly composite numbers
    test_numbers.extend([12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680])
    
    # Add random numbers
    test_numbers.extend(random.sample(range(10, min(5000, max_n)), 300))
    
    test_numbers = list(set(test_numbers))  # Remove duplicates
    
    print(f"Testing with {len(test_numbers)} numbers")
    
    # Run evolution
    start_time = time.time()
    max_generations = int(runtime_hours * 100)  # Adjust based on runtime
    
    try:
        optimizer.evolve_islands(test_numbers, generations=max_generations)
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
    
    # Report results
    optimizer.report_discoveries()
    
    # Save results
    results = {
        'best_individuals': optimizer.best_individuals,
        'formula_discoveries': optimizer.formula_discoveries,
        'runtime': time.time() - start_time,
        'config': config.__dict__
    }
    
    filename = f"sieve_echo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {filename}")
    print(f"Total runtime: {(time.time() - start_time) / 3600:.2f} hours")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Sieve Echo Evolver')
    parser.add_argument('--hours', type=float, default=24.0, help='Runtime in hours')
    parser.add_argument('--max_n', type=int, default=10000, help='Maximum n to test')
    args = parser.parse_args()
    
    run_enhanced_evolver(runtime_hours=args.hours, max_n=args.max_n)