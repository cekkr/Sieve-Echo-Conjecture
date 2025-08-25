#!/usr/bin/env python3
"""
Unified Sieve Echo Conjecture Analysis System
Version 8.2 - Complete integration of all components

Key Components:
1. NDR (Normalized Digit Representation) - renamed from theta to avoid confusion
2. Evolvo genetic algorithm language for formula discovery
3. Neural architecture search for pattern recognition
4. Comprehensive mathematical pattern analysis
5. Proper handling of number types (float conversion for Rational)
"""

import numpy as np
import math
import random
import pickle
import json
import time
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# Core mathematical libraries
from sympy import (
    factorint, primerange, isprime, totient, divisors, mobius,
    primorial, factorial, nextprime, prevprime, prime, primepi,
    sqrt as sym_sqrt, log as sym_log, N as sym_N
)
from scipy import stats, signal, optimize, special, integrate
from scipy.fft import fft, ifft, fftfreq
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Global configuration for all analysis components"""
    # Data generation
    max_n: int = 1000000
    sample_size: int = 10000
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16])
    
    # NDR (formerly theta) analysis
    ndr_max_length: int = 10000
    base_invariance_threshold: float = 0.1
    
    # Evolvo genetic algorithm settings
    evolvo_enabled: bool = True
    evolvo_generations: int = 1000
    evolvo_population: int = 1000
    evolvo_max_algorithm_length: int = 60
    
    # Neural network settings
    nn_enabled: bool = True
    nn_hidden_dim: int = 512
    nn_learning_rate: float = 0.001
    nn_epochs: int = 1000
    
    # Genetic algorithm settings
    ga_enabled: bool = True
    ga_generations: int = 1000
    ga_population_size: int = 1000
    ga_elite_size: int = 10
    ga_crossover_rate: float = 0.7
    ga_mutation_rate: float = 0.3
    
    # Mathematical constants
    euler_gamma: float = 0.5772156649015329
    meissel_mertens: float = 0.2614972128476428
    
    # Analysis settings
    parallel_enabled: bool = True
    n_workers: int = cpu_count()
    save_plots: bool = True
    save_models: bool = True
    verbose: bool = True

CONFIG = Config()

# ==============================================================================
# EVOLVO GENETIC ALGORITHM LANGUAGE (Restored from evolvo_model.py)
# ==============================================================================

class DataStore:
    """Manages variables and constants for genetic algorithms"""
    def __init__(self, config: Dict):
        self.config = config
        self.stores = {}
        self.store_names = {}
        self.name_to_type_map = {}
        self.initial_values = {}
        
        for store_type, names in config.items():
            self.stores[store_type] = [None] * len(names)
            self.store_names[store_type] = list(names)
            for name in names:
                self.name_to_type_map[name] = store_type
        
        self.reset()
    
    def set_initial_value(self, name: str, value: Any):
        if name not in self.name_to_type_map:
            raise ValueError(f"'{name}' not defined in store configuration.")
        
        store_type = self.name_to_type_map[name]
        is_bool = store_type.startswith('b')
        
        # Convert to appropriate type
        if is_bool:
            value = bool(value)
        else:
            # Convert Rational to float
            value = float(sym_N(value))
        
        self.initial_values[name] = value
        self.set(name, value)
    
    def get_store_type_and_index(self, name: str) -> Tuple[str, int]:
        if name not in self.name_to_type_map:
            raise NameError(f"Store named '{name}' not found.")
        store_type = self.name_to_type_map[name]
        index = self.store_names[store_type].index(name)
        return store_type, index
    
    def set(self, name: str, value: Any):
        store_type, index = self.get_store_type_and_index(name)
        self.set_by_location(store_type, index, value)
    
    def get(self, name: str) -> Any:
        store_type, index = self.get_store_type_and_index(name)
        return self.get_by_location(store_type, index)
    
    def set_by_location(self, store_type: str, index: int, value: Any):
        is_bool = store_type.startswith('b')
        
        if is_bool and not isinstance(value, bool):
            value = bool(value)
        elif not is_bool:
            value = float(sym_N(value))
        
        if index >= len(self.stores[store_type]):
            if store_type.endswith('$'):
                self.stores[store_type].extend([None] * (index + 1 - len(self.stores[store_type])))
            else:
                raise IndexError(f"Cannot create new constants of type {store_type} at runtime.")
        
        self.stores[store_type][index] = value
    
    def get_by_location(self, store_type: str, index: int) -> Any:
        if index >= len(self.stores[store_type]):
            return False if store_type.startswith('b') else 0.0
        return self.stores[store_type][index]
    
    def reset(self):
        for store_type, names in self.store_names.items():
            is_bool = store_type.startswith('b')
            default_val = False if is_bool else 0.0
            self.stores[store_type] = [default_val] * len(names)
        
        for name, value in self.initial_values.items():
            self.set(name, value)

class InstructionSet:
    """Defines operations for genetic algorithms"""
    def __init__(self):
        self.operations = {}
        self.op_properties = {}
        self.op_types = defaultdict(list)
    
    def register(self, name: str, function: Callable, arg_types: List[str], op_type: str = "decimal"):
        self.operations[name] = function
        self.op_properties[name] = {'name': name, 'arg_count': len(arg_types), 'arg_types': arg_types}
        self.op_types[op_type].append(name)

def get_evolvo_instruction_set() -> InstructionSet:
    """Create comprehensive instruction set for genetic algorithms"""
    iset = InstructionSet()
    
    # Basic arithmetic
    iset.register('ADD', lambda a, b: float(a + b), ['d', 'd'], 'decimal')
    iset.register('SUB', lambda a, b: float(a - b), ['d', 'd'], 'decimal')
    iset.register('MUL', lambda a, b: float(a * b), ['d', 'd'], 'decimal')
    iset.register('DIV', lambda a, b: float(a / b) if abs(b) > 1e-9 else 0.0, ['d', 'd'], 'decimal')
    iset.register('MOD', lambda a, b: float(a % b) if abs(b) > 1e-9 else 0.0, ['d', 'd'], 'decimal')
    
    # Mathematical functions
    iset.register('LOG', lambda a: float(math.log(abs(a) + 1e-9)), ['d'], 'decimal')
    iset.register('LOG2', lambda a: float(math.log2(abs(a) + 1e-9)), ['d'], 'decimal')
    iset.register('LOG10', lambda a: float(math.log10(abs(a) + 1e-9)), ['d'], 'decimal')
    iset.register('SQRT', lambda a: float(math.sqrt(abs(a))), ['d'], 'decimal')
    iset.register('POW', lambda a, b: float(a ** min(abs(b), 10)), ['d', 'd'], 'decimal')
    iset.register('EXP', lambda a: float(math.exp(min(a, 10))), ['d'], 'decimal')
    
    # Trigonometric
    iset.register('SIN', lambda a: float(math.sin(a)), ['d'], 'decimal')
    iset.register('COS', lambda a: float(math.cos(a)), ['d'], 'decimal')
    iset.register('TAN', lambda a: float(math.tan(a % (math.pi/2 - 0.01))), ['d'], 'decimal')
    iset.register('ATAN', lambda a: float(math.atan(a)), ['d'], 'decimal')
    
    # Number theory
    iset.register('GCD', lambda a, b: float(math.gcd(int(abs(a)), int(abs(b)))), ['d', 'd'], 'decimal')
    iset.register('FLOOR', lambda a: float(math.floor(a)), ['d'], 'decimal')
    iset.register('CEIL', lambda a: float(math.ceil(a)), ['d'], 'decimal')
    iset.register('ROUND', lambda a: float(round(a)), ['d'], 'decimal')
    
    # Statistical
    iset.register('MIN', lambda a, b: float(min(a, b)), ['d', 'd'], 'decimal')
    iset.register('MAX', lambda a, b: float(max(a, b)), ['d', 'd'], 'decimal')
    iset.register('ABS', lambda a: float(abs(a)), ['d'], 'decimal')
    iset.register('SIGN', lambda a: float(1 if a > 0 else -1 if a < 0 else 0), ['d'], 'decimal')
    
    # Constants
    iset.register('PI', lambda: float(math.pi), [], 'decimal')
    iset.register('E', lambda: float(math.e), [], 'decimal')
    iset.register('PHI', lambda: float((1 + math.sqrt(5))/2), [], 'decimal')
    iset.register('EULER', lambda: float(0.5772156649015329), [], 'decimal')
    
    # Boolean operations
    iset.register('NOT', lambda a: not a, ['b'], 'bool')
    iset.register('AND', lambda a, b: a and b, ['b', 'b'], 'bool')
    iset.register('OR', lambda a, b: a or b, ['b', 'b'], 'bool')
    iset.register('CMP', lambda a, b: abs(a - b) < 1e-9, ['d', 'd'], 'bool')
    iset.register('GT', lambda a, b: a > b, ['d', 'd'], 'bool')
    iset.register('GTE', lambda a, b: a >= b, ['d', 'd'], 'bool')
    iset.register('LT', lambda a, b: a < b, ['d', 'd'], 'bool')
    iset.register('LTE', lambda a, b: a <= b, ['d', 'd'], 'bool')
    
    # Structural
    iset.register('IF', None, [], 'structural')
    iset.register('END', None, [], 'structural')
    
    return iset

class Interpreter:
    """Executes genetic algorithms"""
    def __init__(self, instruction_set: InstructionSet):
        self.iset = instruction_set
    
    def _to_bytecode(self, instructions: List) -> List:
        bytecode = []
        context_stack = [bytecode]
        
        for instr in instructions:
            if not instr:
                continue
            
            op = instr[0]
            if op == 'IF':
                if_instr = {'op': 'IF', 'condition': instr[1:3], 'body': []}
                context_stack[-1].append(if_instr)
                context_stack.append(if_instr['body'])
            elif op == 'END':
                if len(context_stack) > 1:
                    context_stack.pop()
            else:
                line_instr = {
                    'op': 'ASSIGN',
                    'target': instr[0:2],
                    'source_op': instr[2],
                    'args': [instr[3:5], instr[5:7]][:self.iset.op_properties[instr[2]]['arg_count']]
                }
                context_stack[-1].append(line_instr)
        
        return bytecode
    
    def execute(self, instructions: List, data_store: DataStore):
        bytecode = self._to_bytecode(instructions)
        self._execute_bytecode(bytecode, data_store)
    
    def _execute_bytecode(self, bytecode: List, data_store: DataStore):
        for line in bytecode:
            op = line['op']
            
            if op == 'IF':
                condition_val = data_store.get_by_location(*line['condition'])
                if condition_val:
                    self._execute_bytecode(line['body'], data_store)
                continue
            
            source_op = line['source_op']
            args = [data_store.get_by_location(*arg) for arg in line['args']]
            
            op_func = self.iset.operations[source_op]
            result = op_func(*args)
            
            data_store.set_by_location(*line['target'], result)

# ==============================================================================
# NDR (NORMALIZED DIGIT REPRESENTATION) CORE
# ==============================================================================

class NDRAnalyzer:
    """Core analyzer for Normalized Digit Representation patterns"""
    
    def __init__(self):
        self.cache = {}
        self.patterns_db = defaultdict(list)
    
    def compute_repetend(self, n: int, base: int, max_length: int = 10000) -> List[int]:
        """Compute repeating decimal pattern for 1/n in given base"""
        cache_key = (n, base)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if math.gcd(n, base) != 1:
            return []
        
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
            result = digits[seen[remainder]:]
        else:
            result = digits
        
        self.cache[cache_key] = result
        return result
    
    def compute_ndr(self, pattern: List[int], base: int) -> np.ndarray:
        """Convert pattern to Normalized Digit Representation"""
        if not pattern:
            return np.array([])
        return np.array(pattern) / base
    
    def compute_ndr_entropy(self, ndr: np.ndarray) -> float:
        """Compute spectral entropy of NDR pattern"""
        if len(ndr) < 2:
            return 0.0
        
        # Compute FFT
        fft_vals = np.abs(fft(ndr))[:len(ndr)//2]
        if len(fft_vals) == 0:
            return 0.0
        
        # Power spectrum
        power = fft_vals**2
        total_power = np.sum(power)
        if total_power == 0:
            return 0.0
        
        # Shannon entropy of power spectrum
        p_spectrum = power / total_power
        p_spectrum = p_spectrum[p_spectrum > 1e-10]
        if len(p_spectrum) == 0:
            return 0.0
        
        return -np.sum(p_spectrum * np.log(p_spectrum))
    
    def extract_comprehensive_features(self, n: int) -> Dict:
        """Extract all features for n across multiple bases"""
        features = {
            'n': n,
            'omega': len(factorint(n)),
            'Omega': sum(factorint(n).values()),
            'tau': len(divisors(n)),
            'sigma': sum(divisors(n)),
            'phi': int(totient(n)),
            'mu': int(mobius(n)),
            'is_prime': isprime(n),
            'is_semiprime': len(factorint(n)) == 2 and sum(factorint(n).values()) == 2,
            'is_prime_power': len(factorint(n)) == 1,
            'smallest_prime_factor': min(factorint(n).keys()) if factorint(n) else n,
            'largest_prime_factor': max(factorint(n).keys()) if factorint(n) else n,
        }
        
        # NDR features across multiple bases
        ndr_entropies = []
        pattern_lengths = []
        kurtosis_values = []
        skewness_values = []
        
        for base in CONFIG.test_bases:
            if math.gcd(n, base) != 1:
                continue
            
            pattern = self.compute_repetend(n, base)
            if not pattern:
                continue
            
            ndr = self.compute_ndr(pattern, base)
            
            # Base-specific features
            features[f'length_b{base}'] = len(pattern)
            features[f'mean_b{base}'] = float(np.mean(ndr))
            features[f'std_b{base}'] = float(np.std(ndr))
            
            if len(ndr) > 3:
                features[f'skew_b{base}'] = float(stats.skew(ndr))
                features[f'kurtosis_b{base}'] = float(stats.kurtosis(ndr))
                kurtosis_values.append(float(stats.kurtosis(ndr)))
                skewness_values.append(float(stats.skew(ndr)))
            
            # NDR entropy
            ndr_entropy = self.compute_ndr_entropy(ndr)
            features[f'ndr_entropy_b{base}'] = ndr_entropy
            ndr_entropies.append(ndr_entropy)
            
            # Pattern complexity
            features[f'unique_digits_b{base}'] = len(set(pattern))
            features[f'compression_ratio_b{base}'] = len(pattern) / max(1, len(set(pattern)))
            
            # Multiplicative order
            features[f'mult_order_b{base}'] = len(pattern)
            features[f'order_ratio_b{base}'] = len(pattern) / max(1, features['phi'])
            
            pattern_lengths.append(len(pattern))
        
        # Aggregate features
        if ndr_entropies:
            features['ndr_entropy_mean'] = float(np.mean(ndr_entropies))
            features['ndr_entropy_std'] = float(np.std(ndr_entropies))
            features['ndr_entropy_min'] = float(np.min(ndr_entropies))
            features['ndr_entropy_max'] = float(np.max(ndr_entropies))
        
        if pattern_lengths:
            features['length_mean'] = float(np.mean(pattern_lengths))
            features['length_std'] = float(np.std(pattern_lengths))
            features['length_gcd'] = int(np.gcd.reduce(pattern_lengths))
        
        if kurtosis_values:
            features['kurtosis_mean'] = float(np.mean(kurtosis_values))
            features['kurtosis_std'] = float(np.std(kurtosis_values))
        
        if skewness_values:
            features['skewness_mean'] = float(np.mean(skewness_values))
        
        return features

# ==============================================================================
# EVOLVO FORMULA DISCOVERER
# ==============================================================================

class EvolvoFormulaDiscoverer:
    """Discovers mathematical formulas using genetic programming"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.instruction_set = get_evolvo_instruction_set()
        self.best_algorithm = None
        self.best_fitness = float('inf')
        
        # Configure data store for genetic algorithms
        self.store_config = {
            'd#': ['n', 'ndr_entropy', 'kurtosis', 'length', 'phi', 'tau', 'sigma', 
                   'one', 'two', 'e', 'pi', 'golden', 'euler_gamma'],
            'b#': ['is_prime', 'true', 'false'],
            'd$': ['omega_pred', 'temp1', 'temp2', 'temp3', 'result'],
            'b$': ['condition', 'flag']
        }
    
    def create_evaluator(self, target: str = 'omega'):
        """Create evaluator for target variable"""
        class TargetEvaluator:
            def __init__(self, data, store_config, instruction_set, target):
                self.data = data
                self.store_config = store_config
                self.instruction_set = instruction_set
                self.interpreter = Interpreter(instruction_set)
                self.target = target
            
            def evaluate(self, algorithm):
                if len(algorithm) > CONFIG.evolvo_max_algorithm_length:
                    return float('inf')
                
                data_store = DataStore(self.store_config)
                total_error = 0.0
                count = 0
                
                for d in self.data[:min(100, len(self.data))]:
                    if self.target not in d:
                        continue
                    
                    # Set constants
                    data_store.reset()
                    data_store.set('n', float(d['n']))
                    data_store.set('ndr_entropy', float(d.get('ndr_entropy_mean', 0)))
                    data_store.set('kurtosis', float(d.get('kurtosis_mean', 0)))
                    data_store.set('length', float(d.get('length_mean', 0)))
                    data_store.set('phi', float(d.get('phi', 1)))
                    data_store.set('tau', float(d.get('tau', 1)))
                    data_store.set('sigma', float(d.get('sigma', 1)))
                    data_store.set('one', 1.0)
                    data_store.set('two', 2.0)
                    data_store.set('e', math.e)
                    data_store.set('pi', math.pi)
                    data_store.set('golden', 1.618033988749895)
                    data_store.set('euler_gamma', 0.5772156649015329)
                    data_store.set('is_prime', bool(d.get('is_prime', False)))
                    data_store.set('true', True)
                    data_store.set('false', False)
                    
                    try:
                        self.interpreter.execute(algorithm, data_store)
                        predicted = data_store.get('omega_pred')
                        actual = float(d[self.target])
                        error = (predicted - actual) ** 2
                        total_error += error
                        count += 1
                    except:
                        return float('inf')
                
                if count == 0:
                    return float('inf')
                
                mse = total_error / count
                complexity_penalty = len(algorithm) * 0.01
                return mse + complexity_penalty
        
        return TargetEvaluator(self.data, self.store_config, self.instruction_set, target)
    
    def generate_random_algorithm(self, max_length: int = 10) -> List:
        """Generate random valid algorithm"""
        algorithm = []
        ops = list(self.instruction_set.op_types['decimal'])
        
        for _ in range(random.randint(1, max_length)):
            op = random.choice(ops)
            
            # Target is always omega_pred
            target = ['d$', 0]
            
            # Get operation properties
            op_props = self.instruction_set.op_properties[op]
            arg_count = op_props['arg_count']
            
            if arg_count == 0:  # Constants
                instruction = target + [op]
            elif arg_count == 1:  # Unary
                arg_type = 'd#' if random.random() < 0.7 else 'd$'
                arg_idx = random.randint(0, 12 if arg_type == 'd#' else 4)
                instruction = target + [op, arg_type, arg_idx]
            else:  # Binary
                arg1_type = 'd#' if random.random() < 0.7 else 'd$'
                arg1_idx = random.randint(0, 12 if arg1_type == 'd#' else 4)
                arg2_type = 'd#' if random.random() < 0.7 else 'd$'
                arg2_idx = random.randint(0, 12 if arg2_type == 'd#' else 4)
                instruction = target + [op, arg1_type, arg1_idx, arg2_type, arg2_idx]
            
            algorithm.append(instruction)
        
        return algorithm
    
    def evolve_formula(self, target: str = 'omega', generations: int = None):
        """Evolve formula using genetic programming"""
        generations = generations or CONFIG.evolvo_generations
        evaluator = self.create_evaluator(target)
        
        # Initialize population
        population = []
        for _ in range(CONFIG.evolvo_population):
            algorithm = self.generate_random_algorithm()
            fitness = evaluator.evaluate(algorithm)
            population.append((algorithm, fitness))
        
        # Evolution loop
        for gen in range(generations):
            population.sort(key=lambda x: x[1])
            
            if population[0][1] < self.best_fitness:
                self.best_fitness = population[0][1]
                self.best_algorithm = population[0][0]
                print(f"Evolvo gen {gen}: best fitness = {self.best_fitness:.4f}")
            
            # Create next generation
            new_population = population[:CONFIG.evolvo_population // 5]  # Elitism
            
            while len(new_population) < CONFIG.evolvo_population:
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)
                
                if random.random() < CONFIG.ga_crossover_rate:
                    child = self.crossover(parent1[0], parent2[0])
                else:
                    child = parent1[0].copy()
                
                if random.random() < CONFIG.ga_mutation_rate:
                    child = self.mutate(child)
                
                fitness = evaluator.evaluate(child)
                new_population.append((child, fitness))
            
            population = new_population
        
        return self.best_algorithm
    
    def tournament_select(self, population: List, size: int = 3):
        tournament = random.sample(population, min(size, len(population)))
        return min(tournament, key=lambda x: x[1])
    
    def crossover(self, parent1: List, parent2: List) -> List:
        if not parent1 or not parent2:
            return parent1 or parent2
        
        point1 = random.randint(0, len(parent1))
        point2 = random.randint(0, len(parent2))
        return parent1[:point1] + parent2[point2:]
    
    def mutate(self, algorithm: List) -> List:
        if not algorithm:
            return algorithm
        
        mutated = algorithm.copy()
        mutation_type = random.choice(['modify', 'insert', 'delete'])

        try:
            if mutation_type == 'modify' and mutated:
                idx = random.randint(0, len(mutated) - 1)
                mutated[idx] = self.generate_random_algorithm(1)[0]
            elif mutation_type == 'insert':
                idx = random.randint(0, len(mutated))
                mutated.insert(idx, self.generate_random_algorithm(1)[0])
            elif mutation_type == 'delete' and len(mutated) > 1:
                idx = random.randint(0, len(mutated) - 1)
                mutated.pop(idx)
        except:
            print("mutate: mutation failed")
        
        return mutated

# ==============================================================================
# MATHEMATICAL PATTERN DISCOVERY
# ==============================================================================

class PatternDiscoveryEngine:
    """Discovers mathematical patterns using multiple approaches"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.discovered_patterns = []
        self.constants_lib = self._init_constants()
    
    def _init_constants(self) -> Dict[str, float]:
        """Initialize library of mathematical constants"""
        return {
            'e': math.e,
            'pi': math.pi,
            'phi': (1 + math.sqrt(5)) / 2,
            'phi_conjugate': (math.sqrt(5) - 1) / 2,
            'inv_phi_squared': -1 / ((1 + math.sqrt(5)) / 2) ** 2,
            'euler_gamma': 0.5772156649015329,
            'meissel_mertens': 0.2614972128476428,
            'sqrt_2': math.sqrt(2),
            'sqrt_3': math.sqrt(3),
            'sqrt_5': math.sqrt(5),
            'ln_2': math.log(2),
            'ln_10': math.log(10),
            'catalan': 0.915965594177219,
            'apery': 1.202056903159594,
            'twin_prime': 0.6601618158468696,
            'mills': 1.3063778838630806,
            'plastic': 1.324717957244746,
            'tribonacci': 1.839286755214161,
            '5_minus_1_over_15': 5 - 1/15,
            'negative_phi_squared_approx': -1.599,
        }
    
    def find_constant_matches(self, value: float, tolerance: float = 0.01) -> List[Dict]:
        """Find mathematical constants matching a value"""
        matches = []
        
        for name, const in self.constants_lib.items():
            error = abs(value - const)
            if error < tolerance:
                matches.append({
                    'constant': name,
                    'value': const,
                    'error': error
                })
        
        # Check combinations
        for name1, const1 in self.constants_lib.items():
            for name2, const2 in self.constants_lib.items():
                # Addition/subtraction
                for op, result in [('+', const1 + const2), ('-', const1 - const2)]:
                    error = abs(value - result)
                    if error < tolerance:
                        matches.append({
                            'constant': f"{name1} {op} {name2}",
                            'value': result,
                            'error': error
                        })
        
        return sorted(matches, key=lambda x: x['error'])[:5]
    
    def analyze_sieve_echo_law(self) -> Dict:
        """Analyze the core Sieve Echo relationship"""
        # Extract data for analysis
        ndr_entropies = []
        omega_values = []
        
        for d in self.data:
            if 'ndr_entropy_mean' in d and 'omega' in d:
                ndr_entropies.append(d['ndr_entropy_mean'])
                omega_values.append(d['omega'])
        
        if len(ndr_entropies) < 10:
            return {}
        
        # Fit linear model: H_NDR = α * log(ω) + β
        X = np.log(np.array(omega_values) + 1).reshape(-1, 1)
        y = np.array(ndr_entropies)
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = float(model.coef_[0])
        beta = float(model.intercept_)
        r2 = model.score(X, y)
        
        # Find constant matches
        alpha_matches = self.find_constant_matches(alpha)
        beta_matches = self.find_constant_matches(beta)
        
        result = {
            'law': f"H_NDR = {alpha:.4f} * log(ω) + {beta:.4f}",
            'alpha': alpha,
            'beta': beta,
            'r2': r2,
            'alpha_candidates': alpha_matches,
            'beta_candidates': beta_matches,
            'confirmed': r2 > 0.5
        }
        
        self.discovered_patterns.append(result)
        return result
    
    def test_riemann_zeta_connections(self) -> List[Dict]:
        """Test connections to Riemann zeta function"""
        connections = []
        
        for s in [2, 3, 4]:
            # Compute partial zeta sum
            max_n = min(1000, max(d['n'] for d in self.data))
            zeta_partial = sum(1/k**s for k in range(1, max_n + 1))
            
            # Test correlation with NDR entropy
            correlations = {}
            
            for feature in ['ndr_entropy_mean', 'kurtosis_mean', 'length_mean']:
                if all(feature in d for d in self.data[:100]):
                    X = [1/d['n']**s for d in self.data[:100]]
                    y = [d[feature] for d in self.data[:100]]
                    
                    if np.std(X) > 0 and np.std(y) > 0:
                        corr = np.corrcoef(X, y)[0, 1]
                        correlations[feature] = corr
            
            if correlations:
                connections.append({
                    's': s,
                    'zeta_value': zeta_partial,
                    'correlations': correlations,
                    'max_correlation': max(correlations.values(), key=abs)
                })
        
        return connections
    
    def test_prime_number_theorem_connections(self) -> Dict:
        """Test connections to Prime Number Theorem"""
        results = []
        
        for d in self.data:
            n = d['n']
            if n <= 2:
                continue
            
            # PNT approximation
            pi_n_approx = n / math.log(n)
            
            # Li(n) approximation
            li_n = self._logarithmic_integral(n)
            
            # Actual prime count
            actual_pi_n = int(primepi(n))
            
            # NDR entropy
            h_ndr = d.get('ndr_entropy_mean', 0)
            
            results.append({
                'n': n,
                'h_ndr': h_ndr,
                'pi_n': actual_pi_n,
                'pnt_approx': pi_n_approx,
                'li_approx': li_n,
                'pnt_error': abs(pi_n_approx - actual_pi_n),
                'li_error': abs(li_n - actual_pi_n),
                'pnt_entropy_product': h_ndr * math.log(n)
            })
        
        # Analyze correlations
        if results:
            correlations = {}
            h_vals = [r['h_ndr'] for r in results]
            
            for key in ['pnt_entropy_product', 'pnt_error', 'li_error']:
                vals = [r[key] for r in results]
                if np.std(vals) > 0 and np.std(h_vals) > 0:
                    correlations[key] = np.corrcoef(vals, h_vals)[0, 1]
            
            return {
                'correlations': correlations,
                'best_predictor': max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
            }
        
        return {}
    
    def _logarithmic_integral(self, x: float) -> float:
        """Compute logarithmic integral Li(x)"""
        if x <= 2:
            return 0
        result, _ = integrate.quad(lambda t: 1/np.log(t), 2, x)
        return result
    
    def test_modular_patterns(self) -> List[Dict]:
        """Test modular arithmetic patterns"""
        patterns = []
        
        for mod in [6, 12, 30, 210]:
            residue_classes = defaultdict(list)
            
            for d in self.data:
                residue = d['n'] % mod
                if 'ndr_entropy_mean' in d:
                    residue_classes[residue].append(d['ndr_entropy_mean'])
            
            if len(residue_classes) > 1:
                means = {r: np.mean(vals) for r, vals in residue_classes.items() if vals}
                variance = np.var(list(means.values()))
                
                if variance > 0.01:
                    patterns.append({
                        'modulus': mod,
                        'variance': variance,
                        'residue_means': means,
                        'significant': True
                    })
        
        return patterns
    
    def discover_all_patterns(self) -> Dict:
        """Run all pattern discovery methods"""
        results = {
            'sieve_echo_law': self.analyze_sieve_echo_law(),
            'riemann_zeta': self.test_riemann_zeta_connections(),
            'prime_number_theorem': self.test_prime_number_theorem_connections(),
            'modular_patterns': self.test_modular_patterns()
        }
        
        return results

# ==============================================================================
# MAIN ANALYSIS SYSTEM
# ==============================================================================

class UnifiedSieveEchoAnalyzer:
    """Unified analysis system for Sieve Echo Conjecture"""
    
    def __init__(self):
        self.ndr_analyzer = NDRAnalyzer()
        self.data = []
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def generate_dataset(self, max_n: int = None):
        """Generate comprehensive dataset"""
        max_n = max_n or CONFIG.max_n
        print(f"Generating dataset for n=2 to {max_n}...")
        
        self.data = []
        
        if CONFIG.parallel_enabled:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=CONFIG.n_workers) as executor:
                futures = []
                for n in range(2, min(max_n, CONFIG.sample_size)):
                    if n > 1:
                        futures.append(executor.submit(self.ndr_analyzer.extract_comprehensive_features, n))
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        features = future.result()
                        self.data.append(features)
                        if (i + 1) % 100 == 0:
                            print(f"Processed {i + 1} numbers...")
                    except Exception as e:
                        print(f"Error processing: {e}")
        else:
            # Sequential processing
            for n in range(2, min(max_n, CONFIG.sample_size)):
                if n > 1:
                    features = self.ndr_analyzer.extract_comprehensive_features(n)
                    self.data.append(features)
                    
                    if n % 100 == 0:
                        print(f"Processed up to n={n}")
        
        print(f"Generated {len(self.data)} data points")
    
    def test_base_invariance(self) -> Dict:
        """Test if NDR entropy is base-invariant"""
        print("\nTesting base invariance...")
        
        invariant_count = 0
        non_invariant_count = 0
        results = []
        
        for d in self.data[:100]:  # Test sample
            n = d['n']
            entropies = []
            
            for base in CONFIG.test_bases:
                key = f'ndr_entropy_b{base}'
                if key in d:
                    entropies.append(d[key])
            
            if len(entropies) >= 2:
                cv = np.std(entropies) / np.mean(entropies) if np.mean(entropies) > 0 else float('inf')
                is_invariant = cv < CONFIG.base_invariance_threshold
                
                if is_invariant:
                    invariant_count += 1
                else:
                    non_invariant_count += 1
                
                results.append({
                    'n': n,
                    'cv': cv,
                    'is_invariant': is_invariant,
                    'is_prime': d['is_prime']
                })
        
        # Analyze by prime status
        prime_invariant = sum(1 for r in results if r['is_invariant'] and r['is_prime'])
        prime_total = sum(1 for r in results if r['is_prime'])
        composite_invariant = sum(1 for r in results if r['is_invariant'] and not r['is_prime'])
        composite_total = sum(1 for r in results if not r['is_prime'])
        
        return {
            'invariant_rate': invariant_count / max(1, invariant_count + non_invariant_count),
            'prime_invariance_rate': prime_invariant / max(1, prime_total),
            'composite_invariance_rate': composite_invariant / max(1, composite_total),
            'mean_cv': np.mean([r['cv'] for r in results])
        }
    
    def run_evolvo_discovery(self):
        """Run Evolvo genetic formula discovery"""
        if not CONFIG.evolvo_enabled:
            return None
        
        print("\nRunning Evolvo formula discovery...")
        discoverer = EvolvoFormulaDiscoverer(self.data)
        
        # Discover formula for omega
        best_algorithm = discoverer.evolve_formula('omega', CONFIG.evolvo_generations)
        
        if best_algorithm:
            print(f"Best formula fitness: {discoverer.best_fitness:.4f}")
            
            # Decode algorithm to readable form
            formula_str = self._decode_algorithm(best_algorithm)
            print(f"Formula: {formula_str}")
            
            return {
                'algorithm': best_algorithm,
                'fitness': discoverer.best_fitness,
                'formula': formula_str
            }
        
        return None
    
    def _decode_algorithm(self, algorithm: List) -> str:
        """Convert algorithm to readable formula"""
        if not algorithm:
            return "empty"
        
        ops = []
        for instr in algorithm:
            if len(instr) >= 3:
                ops.append(instr[2])
        
        return " → ".join(ops) if ops else "?"
    
    def run_pattern_discovery(self):
        """Run comprehensive pattern discovery"""
        print("\nDiscovering mathematical patterns...")
        
        engine = PatternDiscoveryEngine(self.data)
        patterns = engine.discover_all_patterns()
        
        # Print key findings
        if 'sieve_echo_law' in patterns and patterns['sieve_echo_law']:
            law = patterns['sieve_echo_law']
            print(f"\nSieve Echo Law: {law['law']}")
            print(f"  R² = {law['r2']:.4f}")
            
            if law['alpha_candidates']:
                print(f"  α candidates: {law['alpha_candidates'][0]['constant']}")
            if law['beta_candidates']:
                print(f"  β candidates: {law['beta_candidates'][0]['constant']}")
        
        return patterns
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        if not CONFIG.save_plots:
            return
        
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. NDR Entropy vs Omega
        omega_vals = [d['omega'] for d in self.data if 'omega' in d and 'ndr_entropy_mean' in d]
        entropy_vals = [d['ndr_entropy_mean'] for d in self.data if 'omega' in d and 'ndr_entropy_mean' in d]
        
        if omega_vals and entropy_vals:
            axes[0, 0].scatter(omega_vals, entropy_vals, alpha=0.5)
            axes[0, 0].set_xlabel('ω(n)')
            axes[0, 0].set_ylabel('Mean NDR Entropy')
            axes[0, 0].set_title('Sieve Echo Relationship')
        
        # 2. Base Invariance
        n_vals = []
        cv_vals = []
        for d in self.data[:100]:
            entropies = [d[f'ndr_entropy_b{b}'] for b in CONFIG.test_bases 
                        if f'ndr_entropy_b{b}' in d]
            if len(entropies) >= 2:
                n_vals.append(d['n'])
                cv = np.std(entropies) / np.mean(entropies) if np.mean(entropies) > 0 else 0
                cv_vals.append(cv)
        
        if n_vals and cv_vals:
            axes[0, 1].scatter(n_vals, cv_vals, alpha=0.5)
            axes[0, 1].axhline(y=CONFIG.base_invariance_threshold, color='r', linestyle='--')
            axes[0, 1].set_xlabel('n')
            axes[0, 1].set_ylabel('Coefficient of Variation')
            axes[0, 1].set_title('Base Invariance Test')
        
        # 3. Kurtosis Distribution
        kurtosis_vals = [d['kurtosis_mean'] for d in self.data if 'kurtosis_mean' in d]
        if kurtosis_vals:
            axes[0, 2].hist(kurtosis_vals, bins=30, alpha=0.7)
            axes[0, 2].set_xlabel('Mean Kurtosis')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Kurtosis Distribution')
        
        # 4. Pattern Length vs n
        n_vals = [d['n'] for d in self.data if 'length_mean' in d]
        length_vals = [d['length_mean'] for d in self.data if 'length_mean' in d]
        
        if n_vals and length_vals:
            axes[1, 0].scatter(n_vals, length_vals, alpha=0.5)
            axes[1, 0].set_xlabel('n')
            axes[1, 0].set_ylabel('Mean Pattern Length')
            axes[1, 0].set_title('Pattern Length Distribution')
        
        # 5. Prime vs Composite Entropy
        prime_entropies = [d['ndr_entropy_mean'] for d in self.data 
                          if d.get('is_prime') and 'ndr_entropy_mean' in d]
        composite_entropies = [d['ndr_entropy_mean'] for d in self.data 
                               if not d.get('is_prime') and 'ndr_entropy_mean' in d]
        
        if prime_entropies and composite_entropies:
            axes[1, 1].hist([prime_entropies, composite_entropies], 
                           label=['Prime', 'Composite'], alpha=0.7, bins=20)
            axes[1, 1].set_xlabel('NDR Entropy')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Entropy: Primes vs Composites')
            axes[1, 1].legend()
        
        # 6. Multiplicative Order Ratio
        order_ratios = [d['order_ratio_b10'] for d in self.data if 'order_ratio_b10' in d]
        if order_ratios:
            axes[1, 2].hist(order_ratios, bins=30, alpha=0.7)
            axes[1, 2].set_xlabel('Order Ratio (ord_n(10) / φ(n))')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Multiplicative Order Ratio')
        
        plt.tight_layout()
        plt.savefig(f'sieve_echo_analysis_{self.timestamp}.png', dpi=150)
        plt.close()
        
        print(f"Saved visualization to sieve_echo_analysis_{self.timestamp}.png")
    
    def save_results(self):
        """Save all results"""
        print("\nSaving results...")
        
        # Prepare results dictionary
        save_data = {
            'timestamp': self.timestamp,
            'config': CONFIG.__dict__,
            'data_summary': {
                'total_numbers': len(self.data),
                'max_n': max(d['n'] for d in self.data) if self.data else 0,
                'features_per_number': len(self.data[0]) if self.data else 0
            },
            'results': self.results,
            'sample_data': self.data[:10]  # Save sample for verification
        }
        
        # Save JSON
        with open(f'sieve_echo_results_{self.timestamp}.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Save pickle for complete data
        with open(f'sieve_echo_complete_{self.timestamp}.pkl', 'wb') as f:
            pickle.dump({'data': self.data, 'results': self.results}, f)
        
        print(f"Results saved with timestamp {self.timestamp}")
    
    def run_complete_analysis(self):
        """Run complete unified analysis"""
        print("="*80)
        print("UNIFIED SIEVE ECHO CONJECTURE ANALYSIS")
        print("="*80)
        
        # Phase 1: Generate dataset
        self.generate_dataset()
        
        # Phase 2: Test base invariance
        invariance_results = self.test_base_invariance()
        self.results['base_invariance'] = invariance_results
        print(f"\nBase invariance rate: {invariance_results['invariant_rate']:.2%}")
        
        # Phase 3: Discover patterns
        patterns = self.run_pattern_discovery()
        self.results['patterns'] = patterns
        
        # Phase 4: Evolvo formula discovery
        if CONFIG.evolvo_enabled:
            evolvo_results = self.run_evolvo_discovery()
            self.results['evolvo'] = evolvo_results
        
        # Phase 5: Create visualizations
        self.create_visualizations()
        
        # Phase 6: Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point"""
    analyzer = UnifiedSieveEchoAnalyzer()
    
    try:
        analyzer.run_complete_analysis()
        return True
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        analyzer.save_results()
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)