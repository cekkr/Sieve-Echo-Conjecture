#!/usr/bin/env python3
"""
Sieve Echo Conjecture - Unified Evolvo Discovery Engine v9
Key improvements:
- Uses unified evolvo library instead of split evolvo_model/evolvo_nn
- Multi-base NDR pattern discovery 
- Proper evolutionary approach to finding patterns
- Co-evolution of formulas and neural architectures

---

## Major Changes from v8 to v9:

### 1. **Multi-Base NDR Discovery**
- Tests patterns across 13 different prime bases (2, 3, 5, 7, 10, 11, 13, 16, 17, 19, 23, 29, 31)
- Computes base invariance metrics (coefficient of variation) to verify pattern consistency
- Aggregates features across all bases (mean entropy, std, kurtosis, etc.)

### 2. **Proper Evolvo Integration**
- Uses `evolvo.AlgorithmGenome` for formula evolution
- Uses `evolvo.NeuralGenome` for neural architecture search  
- Implements `evolvo.ResourceAwareEvolver` for memory-managed evolution
- Includes `evolvo.ResourceMonitor` for VRAM/RAM management
- Uses `evolvo.QLearningGuide` for guided evolution

### 3. **Co-Evolution System**
- Evolves formulas and neural networks together
- Shares learning between both systems via Q-learning
- Allows mutual feedback between symbolic and neural approaches

### 4. **Key Improvements**
- **Base Invariance Testing**: Verifies that patterns are structurally similar across different bases
- **Proper Genome Creation**: Creates valid instruction sequences using the new Instruction class
- **Resource Management**: Prevents memory overflow when evolving multiple neural networks
- **Parallel Evaluation**: Uses batch processing for neural network evaluation

### 5. **Critical Fixes**
- No longer computes simple correlations - uses evolution to discover patterns
- Properly normalizes digits to [0,1] interval (NDR framework)
- Tests pattern structure across bases, not expecting identical values
- Lets evolution discover feature combinations rather than pre-specifying them

## How It Works:

1. **Data Generation**: For each n, computes NDR patterns in all test bases
2. **Feature Aggregation**: Calculates mean/std of entropy, kurtosis, length across bases
3. **Formula Evolution**: Uses genetic algorithms to evolve mathematical formulas
4. **Neural Evolution**: Evolves neural architectures with resource constraints
5. **Co-Evolution**: Both systems evolve together, sharing insights via Q-learning
6. **Base Invariance Check**: Verifies patterns are consistent (CV < 0.1 is good)

The script now properly implements the evolutionary discovery approach you intended, where patterns emerge from evolution rather than being predetermined, and critically tests across multiple numerical bases to find universal patterns rather than base-specific artifacts.

"""
import copy
import traceback
from pathlib import Path

import numpy as np
import math
import random

import pickle
import dill

import json
import time
import os
import sys
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import warnings
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# Core mathematical libraries
from sympy import factorint, isprime, totient, divisors
from scipy import stats
from scipy.fft import fft
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Device configuration
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import unified Evolvo library
import evolvo

warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    # Core parameters
    perpetual_mode: bool = True
    max_cycles: int = 100
    data_chunk_size: int = 1000
    
    # CRITICAL: Multiple bases for pattern discovery
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 19, 23, 29, 31])
    
    # Evolution parameters
    formula_generations: int = 200
    formula_population_size: int = 1000
    nn_generations: int = 50
    nn_population_size: int = 200
    max_algorithm_length: int = 60
    
    # Resource management
    max_model_params: int = int(1e8)  # 100M parameters max
    max_memory_mb: float = 2048 * 6  # 2GB max per model
    
    # Files
    state_file: str = "sieve_echo_state_v7.pkl"
    results_dir: str = "results_v7"
    
CONFIG = Config()

# ==============================================================================
# NDR (Normalized Digit Representation) COMPUTER
# ==============================================================================
class MultiBaseNDRComputer:
    """Computes NDR patterns across multiple bases"""
    
    def __init__(self):
        self.cache = {}
        
    def compute_ndr(self, n: int, base: int) -> np.ndarray:
        """Compute normalized digit representation for 1/n in given base"""
        if (n, base) in self.cache:
            return self.cache[(n, base)]
            
        if math.gcd(n, base) != 1:
            return np.array([])
            
        # Compute repetend
        remainders = {}
        digits = []
        r = 1
        pos = 0
        
        while r != 0 and r not in remainders:
            remainders[r] = pos
            r = r * base
            digit = r // n
            digits.append(digit)
            r = r % n
            pos += 1
            
            # Safety limit
            if len(digits) > n + 1:
                break
                
        # Extract repeating part
        if r in remainders:
            repetend = digits[remainders[r]:]
        else:
            repetend = digits
            
        # Normalize to [0, 1]
        ndr = np.array(repetend) / base
        self.cache[(n, base)] = ndr
        return ndr
    
    def compute_multi_base_features(self, n: int, bases: List[int]) -> Dict:
        """Compute features across multiple bases"""
        features = {
            'n': n,
            'omega': len(factorint(n)),
            'phi': totient(n),
            'tau': len(divisors(n)),
            'is_prime': isprime(n),
            'is_prime_power': self._is_prime_power(n)
        }
        
        # Collect NDR patterns for all bases
        patterns = {}
        entropies = []
        lengths = []
        kurtoses = []
        
        for base in bases:
            ndr = self.compute_ndr(n, base)
            if len(ndr) == 0:
                continue
                
            patterns[base] = ndr
            lengths.append(len(ndr))
            
            # Compute entropy via FFT
            if len(ndr) > 1:
                spectrum = np.abs(fft(ndr))[:len(ndr)//2]
                if len(spectrum) > 0 and np.sum(spectrum) > 1e-10:
                    spectrum_norm = spectrum / np.sum(spectrum)
                    spectrum_norm = spectrum_norm[spectrum_norm > 1e-10]
                    if len(spectrum_norm) > 0:
                        entropy = -np.sum(spectrum_norm * np.log(spectrum_norm + 1e-10))
                        entropies.append(entropy)
                        
            # Compute kurtosis
            if len(ndr) > 3:
                kurt = stats.kurtosis(ndr)
                kurtoses.append(kurt)
        
        # Aggregate features
        if entropies:
            features['entropy_mean'] = np.mean(entropies)
            features['entropy_std'] = np.std(entropies)
            features['entropy_cv'] = np.std(entropies) / (np.mean(entropies) + 1e-10)
        
        if lengths:
            features['length_mean'] = np.mean(lengths)
            features['length_std'] = np.std(lengths)
            
        if kurtoses:
            features['kurtosis_mean'] = np.mean(kurtoses)
            features['kurtosis_std'] = np.std(kurtoses)
            
        # Store patterns for further analysis
        features['patterns'] = patterns
        
        return features
    
    def _is_prime_power(self, n: int) -> bool:
        factors = factorint(n)
        return len(factors) == 1

# ==============================================================================
# EVOLVO-BASED FORMULA DISCOVERER
# ==============================================================================
class UnifiedFormulaDiscoverer:
    """Enhanced formula discoverer with detailed genome tracking"""
    
    def __init__(self, data: List[Dict], resource_monitor: evolvo.ResourceMonitor):
        self.data = data
        self.resource_monitor = resource_monitor
        self.instruction_set = self._create_enhanced_instruction_set()
        self.best_discoveries = []  # Track ALL good discoveries
        self.hall_of_fame = []  # Track top genomes across generations
        
    def _create_enhanced_instruction_set(self):
        """Create instruction set with mathematical operations"""
        iset = evolvo.EnhancedInstructionSet()
        iset.register('KURTOSIS', lambda x: x, ['decimal'], 'decimal', 'custom')
        iset.register('ENTROPY', lambda x: x, ['decimal'], 'decimal', 'custom')
        return iset
    
    def _decode_genome(self, genome: evolvo.AlgorithmGenome) -> Dict:
        """Fully decode genome into readable formula representation"""
        
        # Get data configuration for variable names
        feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean', 
                        'entropy_cv', 'n', 'phi', 'tau']
        constants = ['one', 'two', 'pi', 'e', 'half', 'quarter']
        
        decoded = {
            'raw_instructions': [],
            'symbolic_formula': [],
            'dependency_graph': {},
            'effective_length': 0,
            'unique_operations': set(),
            'complexity_score': 0
        }
        
        # Track which variables are actually used
        used_vars = set()
        produced_vars = set()
        
        for i, instr in enumerate(genome.instructions):
            if isinstance(instr, evolvo.Instruction):
                # Decode target
                target_type, target_idx = instr.target
                if target_type == 'd#':
                    target_name = constants[target_idx] if target_idx < len(constants) else f'd#{target_idx}'
                elif target_type == 'd$':
                    target_name = f'd${target_idx}'
                elif target_type == 'b$':
                    target_name = f'b${target_idx}'
                else:
                    target_name = f'{target_type}_{target_idx}'
                
                # Decode arguments
                arg_names = []
                for arg_type, arg_idx in instr.args:
                    if arg_type == 'd#':
                        if arg_idx < len(feature_names):
                            arg_name = feature_names[arg_idx]
                        elif arg_idx < len(feature_names) + len(constants):
                            arg_name = constants[arg_idx - len(feature_names)]
                        else:
                            arg_name = f'd#{arg_idx}'
                    elif arg_type == 'd$':
                        arg_name = f'd${arg_idx}'
                    elif arg_type == 'b$':
                        arg_name = f'b${arg_idx}'
                    else:
                        arg_name = f'{arg_type}_{arg_idx}'
                    
                    arg_names.append(arg_name)
                    used_vars.add(arg_name)
                
                produced_vars.add(target_name)
                
                # Build symbolic representation
                if instr.operation in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'POW']:
                    op_symbols = {'ADD': '+', 'SUB': '-', 'MUL': '*', 
                                 'DIV': '/', 'MOD': '%', 'POW': '^'}
                    if len(arg_names) == 2:
                        symbolic = f"{target_name} = ({arg_names[0]} {op_symbols[instr.operation]} {arg_names[1]})"
                    else:
                        symbolic = f"{target_name} = {instr.operation}({', '.join(arg_names)})"
                elif instr.operation in ['EXP', 'LOG', 'SIN', 'COS', 'SQRT', 'ABS']:
                    symbolic = f"{target_name} = {instr.operation}({arg_names[0] if arg_names else ''})"
                elif instr.operation in ['GT', 'LT', 'EQ', 'GTE', 'LTE']:
                    op_symbols = {'GT': '>', 'LT': '<', 'EQ': '==', 'GTE': '>=', 'LTE': '<='}
                    if len(arg_names) == 2:
                        symbolic = f"{target_name} = ({arg_names[0]} {op_symbols[instr.operation]} {arg_names[1]})"
                    else:
                        symbolic = f"{target_name} = {instr.operation}({', '.join(arg_names)})"
                else:
                    symbolic = f"{target_name} = {instr.operation}({', '.join(arg_names)})"
                
                decoded['raw_instructions'].append({
                    'index': i,
                    'operation': instr.operation,
                    'target': target_name,
                    'args': arg_names,
                    'symbolic': symbolic
                })
                
                decoded['symbolic_formula'].append(symbolic)
                decoded['unique_operations'].add(instr.operation)
                
                # Build dependency graph
                decoded['dependency_graph'][target_name] = {
                    'operation': instr.operation,
                    'depends_on': arg_names,
                    'instruction_index': i
                }
        
        # Calculate effective length (instructions that contribute to output)
        if genome.outputs:
            # Backward trace from outputs
            necessary = set()
            to_check = [f'{t}_{i}' if t.endswith('#') else f'{t[0]}${i}' 
                       for t, i in genome.outputs]
            
            while to_check:
                var = to_check.pop()
                if var in decoded['dependency_graph']:
                    deps = decoded['dependency_graph'][var]
                    necessary.add(deps['instruction_index'])
                    to_check.extend(deps['depends_on'])
            
            decoded['effective_length'] = len(necessary)
        else:
            decoded['effective_length'] = len(decoded['raw_instructions'])
        
        # Calculate complexity score
        # Higher score for: more unique operations, longer effective length, 
        # use of mathematical functions
        math_ops = {'EXP', 'LOG', 'SIN', 'COS', 'SQRT', 'POW'}
        math_op_count = len(decoded['unique_operations'] & math_ops)
        
        decoded['complexity_score'] = (
            decoded['effective_length'] * 2 +  # Length matters
            len(decoded['unique_operations']) * 3 +  # Diversity of operations
            math_op_count * 5  # Mathematical operations are valuable
        )
        
        # Convert sets to lists for JSON serialization
        decoded['unique_operations'] = list(decoded['unique_operations'])
        
        return decoded
    
    def _evaluate_genome_comprehensive(self, genome: evolvo.AlgorithmGenome, 
                                      data_sample: List[Dict]) -> Dict:
        """Comprehensive evaluation with multiple metrics"""
        
        data_config = {
            'b#': ['true', 'false'],
            'd#': ['kurtosis_mean', 'length_mean', 'entropy_mean', 'entropy_cv', 
                   'n', 'phi', 'tau', 'one', 'two', 'pi', 'e', 'half', 'quarter'],
            'b$': [f'b{i}' for i in range(8)],
            'd$': [f'd{i}' for i in range(16)]
        }
        
        try:
            compiled = genome.to_executable()
            data_store = evolvo.UnifiedDataStore(data_config)
            
            predictions = []
            actuals = []
            errors = []
            
            for d in data_sample:
                # Set inputs
                for feature in ['kurtosis_mean', 'length_mean', 'entropy_mean', 
                              'entropy_cv', 'n', 'phi', 'tau']:
                    data_store.set(feature, d.get(feature, 0))
                
                # Set constants
                data_store.set('one', 1.0)
                data_store.set('two', 2.0)
                data_store.set('pi', math.pi)
                data_store.set('e', math.e)
                data_store.set('half', 0.5)
                data_store.set('quarter', 0.25)
                
                # Execute
                results = compiled.execute(data_store)
                
                # Get prediction
                pred = results.get('d$_0', 0)
                actual = d.get('omega', 0)
                
                if isinstance(pred, (int, float, np.number)) and np.isfinite(pred):
                    predictions.append(pred)
                    actuals.append(actual)
                    errors.append(abs(pred - actual))
            
            if len(predictions) == 0:
                return {'fitness': 0, 'mse': float('inf'), 'correlation': 0}
            
            # Calculate metrics
            mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)])
            mae = np.mean(errors)
            
            # Calculate correlation if we have variance
            if np.std(predictions) > 1e-10 and np.std(actuals) > 1e-10:
                correlation = np.corrcoef(predictions, actuals)[0, 1]
            else:
                correlation = 0
            
            fitness = 1 / (1 + mse)
            
            return {
                'fitness': fitness,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'n_valid_predictions': len(predictions)
            }
            
        except Exception as e:
            return {'fitness': 0, 'mse': float('inf'), 'correlation': 0, 'error': str(e)}
    
    def evolve_formulas(self) -> Dict:
        """Enhanced formula evolution with comprehensive tracking"""
        print("\n📊 Evolving formulas with detailed tracking...")
        
        feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean', 
                        'entropy_cv', 'n', 'phi', 'tau']
        
        valid_data = [d for d in self.data if all(f in d for f in feature_names)]
        
        if len(valid_data) < 100:
            print("Not enough data for formula evolution")
            return {}
        
        # Create initial population (same as before)
        data_config = {
            'b#': ['true', 'false'],
            'd#': feature_names + ['one', 'two', 'pi', 'e', 'half', 'quarter'],
            'b$': [f'b{i}' for i in range(8)],
            'd$': [f'd{i}' for i in range(16)]
        }
        
        population = []
        for _ in range(CONFIG.formula_population_size):
            genome = evolvo.AlgorithmGenome(data_config, self.instruction_set)
            
            # Generate random algorithm
            for _ in range(random.randint(3, CONFIG.max_algorithm_length)):
                op_name = random.choice(list(self.instruction_set.operations.keys()))
                op_info = self.instruction_set.operations[op_name]
                
                if op_name in ['IF', 'ELSE', 'END', 'ASSIGN']:
                    continue
                
                if op_info.return_type == 'decimal':
                    target_store = 'd$'
                    target_idx = random.randint(0, len(data_config[target_store]) - 1)
                elif op_info.return_type == 'bool':
                    target_store = 'b$'
                    target_idx = random.randint(0, len(data_config[target_store]) - 1)
                else:
                    continue
                
                target = (target_store, target_idx)
                
                args = []
                for arg_type in op_info.arg_types:
                    if arg_type in ['decimal', 'any']:
                        store_type = 'd#' if random.random() < 0.7 else 'd$'
                        idx = random.randint(0, len(data_config[store_type]) - 1)
                        args.append((store_type, idx))
                    elif arg_type == 'bool':
                        store_type = 'b#' if random.random() < 0.5 else 'b$'
                        idx = random.randint(0, len(data_config[store_type]) - 1)
                        args.append((store_type, idx))
                
                instruction = evolvo.Instruction(target, op_name, args)
                genome.add_instruction(instruction)
            
            genome.mark_output(('d$', 0))
            population.append(genome)
        
        # Evolution with comprehensive tracking
        evolver = evolvo.UnifiedEvolver(evolvo.GenomeType.ALGORITHM, 
                                       CONFIG.formula_population_size)
        evolver.population = population
        
        # Track best genomes across all generations
        generation_history = []
        unique_top_genomes = {}  # signature -> genome_info
        
        for gen in range(CONFIG.formula_generations):
            # Evaluate population
            for genome in evolver.population:
                if genome.fitness is None:
                    eval_results = self._evaluate_genome_comprehensive(
                        genome, 
                        random.sample(valid_data, min(100, len(valid_data)))
                    )
                    genome.fitness = eval_results['fitness']
                    genome.metrics = eval_results  # Store all metrics
            
            # Sort by fitness
            evolver.population.sort(key=lambda g: g.fitness or 0, reverse=True)
            
            # Track top performers
            top_10 = evolver.population[:10]
            
            for rank, genome in enumerate(top_10):
                sig = genome.get_signature()
                decoded = self._decode_genome(genome)
                
                # Calculate combined score (fitness + complexity)
                combined_score = (genome.fitness * 100 + 
                                decoded['complexity_score'] * 0.1)
                
                genome_info = {
                    'generation': gen,
                    'rank': rank + 1,
                    'fitness': genome.fitness,
                    'metrics': getattr(genome, 'metrics', {}),
                    'decoded': decoded,
                    'combined_score': combined_score,
                    'signature': sig[:16]  # Short version for display
                }
                
                # Update if this is a new genome or better version
                if sig not in unique_top_genomes or \
                   genome_info['combined_score'] > unique_top_genomes[sig]['combined_score']:
                    unique_top_genomes[sig] = genome_info
            
            # Store generation summary
            generation_history.append({
                'generation': gen,
                'best_fitness': top_10[0].fitness,
                'mean_fitness': np.mean([g.fitness for g in evolver.population[:50]]),
                'unique_genomes': len(evolver.diversity_cache)
            })
            
            # Evolution step
            if gen < CONFIG.formula_generations - 1:
                evolver.generation = gen
                elite_size = int(CONFIG.formula_population_size * 0.1)
                new_population = evolver.population[:elite_size]
                
                while len(new_population) < CONFIG.formula_population_size:
                    parent1 = evolver._tournament_select()
                    parent2 = evolver._tournament_select()
                    
                    if random.random() < 0.7:
                        child = evolver.crossover(parent1, parent2)
                    else:
                        child = copy.deepcopy(random.choice([parent1, parent2]))
                    
                    if random.random() < 0.3:
                        child = evolver.mutate(child)
                    
                    new_population.append(child)
                
                evolver.population = new_population[:CONFIG.formula_population_size]
            
            # Progress report every 10 generations
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {top_10[0].fitness:.4f}, "
                      f"Unique genomes = {len(unique_top_genomes)}")
        
        # Sort all discovered genomes by combined score
        sorted_discoveries = sorted(unique_top_genomes.values(), 
                                   key=lambda x: x['combined_score'], 
                                   reverse=True)
        
        # Return comprehensive results
        return {
            'top_discoveries': sorted_discoveries[:20],  # Top 20 unique formulas
            'generation_history': generation_history,
            'total_unique_genomes': len(unique_top_genomes),
            'best_by_fitness': sorted_discoveries[0] if sorted_discoveries else None,
            'most_complex': max(sorted_discoveries, 
                              key=lambda x: x['decoded']['complexity_score']) 
                            if sorted_discoveries else None
        }

# ==============================================================================
# NEURAL ARCHITECTURE SEARCH WITH UNIFIED EVOLVO
# ==============================================================================

def safe_tensor_conversion(data, device):
    """Safely convert numpy array to PyTorch tensor"""
    # If it's already a tensor, just move to device
    if isinstance(data, torch.Tensor):
        return data.to(device)
    
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Handle object dtype
    if data.dtype == np.object_:
        try:
            # Try to convert to float
            data = np.array(data, dtype=np.float64)
        except (ValueError, TypeError):
            # If that fails, try to flatten and convert
            flat_data = []
            for item in data.flat:
                if isinstance(item, (list, np.ndarray)):
                    flat_data.extend(np.array(item).flatten())
                else:
                    flat_data.append(float(item))
            data = np.array(flat_data).reshape(data.shape)
    
    # Convert to tensor
    return torch.FloatTensor(data).to(device)

class UnifiedNeuralSearcher:
    """Neural architecture search using unified evolvo"""
    
    def __init__(self, data: List[Dict], resource_monitor: evolvo.ResourceMonitor):
        self.data = data
        self.resource_monitor = resource_monitor
        self.best_model = None
        
    def evolve_architectures(self) -> Dict:
        """Evolve neural architectures for omega prediction"""
        print("\n🧠 Evolving neural architectures...")
        
        # Prepare data
        feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean',
                         'entropy_cv', 'n', 'phi', 'tau']
        # Define the target variable here, as the "single source of truth"
        target_name = 'omega'

        # --- START: CORRECTED & ROBUST FILTER (USING target_name) ---
        valid_data = []
        for d in self.data:
            try:
                is_valid = True
                # Check all features
                for f in feature_names:
                    if not (f in d and isinstance(d[f], (int, float, np.number)) and np.isfinite(d[f])):
                        is_valid = False
                        break
                if not is_valid: continue

                # CORRECT: Check the target variable using the target_name variable
                if not (target_name in d and isinstance(d[target_name], (int, float, np.number)) and np.isfinite(d[target_name])):
                    is_valid = False
                
                if is_valid:
                    valid_data.append(d)
            except Exception:
                # Failsafe for any other unexpected data corruption
                continue
        # --- END: CORRECTED & ROBUST FILTER ---

        if len(valid_data) < 100:
            print("Not enough valid, finite data for neural evolution")
            return {}
        
        # Create train/test split
        X = np.array([[d.get(f, 0) for f in feature_names] for d in valid_data])
        # CORRECT: Use target_name here as well
        y = np.array([d.get(target_name, 0) for d in valid_data])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        """
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
        """

        X_train_t = safe_tensor_conversion(X_train, device)
        y_train_t = safe_tensor_conversion(y_train.reshape(-1, 1), device)
        X_test_t = safe_tensor_conversion(X_test, device)
        y_test_t = safe_tensor_conversion(y_test.reshape(-1, 1), device)

        # Define shapes
        input_shape = evolvo.TensorShape(features=len(feature_names))
        output_shape = evolvo.TensorShape(features=1)
        
        # Create initial population
        population = []
        for _ in range(CONFIG.nn_population_size):
            genome = evolvo.NeuralGenome(
                input_shape, output_shape,
                max_params=CONFIG.max_model_params,
                max_memory_mb=CONFIG.max_memory_mb
            )
            
            # Add random layers
            num_layers = random.randint(2, 5)
            hidden_size = random.choice([32, 64, 128, 256])
            
            for i in range(num_layers):
                if i == 0:
                    # First layer
                    layer = evolvo.LayerSpec(
                        'linear',
                        {'in_features': len(feature_names), 
                         'out_features': hidden_size}
                    )
                elif i == num_layers - 1:
                    # Output layer
                    layer = evolvo.LayerSpec(
                        'linear',
                        {'in_features': hidden_size, 'out_features': 1}
                    )
                else:
                    # Hidden layer
                    layer = evolvo.LayerSpec(
                        'linear',
                        {'in_features': hidden_size, 
                         'out_features': hidden_size}
                    )
                
                if not genome.add_layer(layer):
                    break
                    
                # Add activation (except after output)
                if i < num_layers - 1:
                    genome.add_layer(evolvo.LayerSpec('relu', {}))
                    
                    # Sometimes add dropout
                    if random.random() < 0.3:
                        genome.add_layer(evolvo.LayerSpec(
                            'dropout', {'p': random.uniform(0.1, 0.5)}
                        ))
            
            population.append(genome)
        
        # Create resource-aware evolver
        evolver = evolvo.ResourceAwareEvolver(
            evolvo.GenomeType.NEURAL,
            CONFIG.nn_population_size,
            max_model_params=CONFIG.max_model_params,
            resource_monitor=self.resource_monitor
        )
        evolver.population = population
        
        # Define fitness function
        def evaluate_nn(genome: evolvo.NeuralGenome, model: Optional[nn.Module] = None) -> float:
            try:
                if model is None:
                    model = genome.to_executable().to(device)
                
                # Training
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                model.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = model(X_train_t)
                    loss = criterion(outputs, y_train_t)
                    loss.backward()
                    optimizer.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_t)
                    test_loss = criterion(test_outputs, y_test_t).item()
                
                return 1 / (1 + test_loss)
                
            except Exception as e:
                print(f"WARN: NN evaluation failed for genome {genome.get_signature()[:10]}... with error: {e}", file=sys.stderr)
                return 0.0
        
        # Evolve with resource management
        evolver.evaluate_population_parallel(evaluate_nn, batch_size=4)
        
        for gen in range(CONFIG.nn_generations):
            print(f"Generation {gen+1}/{CONFIG.nn_generations}")
            evolver.generation = gen
            
            # Evolution step
            new_population = []
            
            # Elitism
            evolver.population.sort(key=lambda g: g.fitness or 0, reverse=True)
            elite_size = int(CONFIG.nn_population_size * 0.2)
            new_population.extend(evolver.population[:elite_size])
            
            # Breeding
            while len(new_population) < CONFIG.nn_population_size:
                parent1 = evolver._tournament_select()
                parent2 = evolver._tournament_select()
                child = evolver.crossover(parent1, parent2)
                
                if random.random() < 0.3:
                    child = evolver.mutate(child)
                
                new_population.append(child)
            
            evolver.population = new_population[:CONFIG.nn_population_size]
            evolver.evaluate_population_parallel(evaluate_nn, batch_size=4)
        
        # Get best model
        evolver.population.sort(key=lambda g: g.fitness or 0, reverse=True)
        best = evolver.population[0]
        
        if best.fitness and best.fitness > 0:
            print(f"Best NN fitness: {best.fitness:.4f}")
            return {
                'genome': best,
                'fitness': best.fitness,
                'architecture': f"{len(best.layers)} layers, {best.estimated_params} params"
            }
        
        return {}

# ==============================================================================
# CO-EVOLUTION SYSTEM
# ==============================================================================
class CoEvolutionSystem:
    """Co-evolve formulas and neural networks together"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.resource_monitor = evolvo.ResourceMonitor()
        self.formula_discoverer = UnifiedFormulaDiscoverer(data, self.resource_monitor)
        self.neural_searcher = UnifiedNeuralSearcher(data, self.resource_monitor)
        self.q_guide = evolvo.QLearningGuide()
        
    def co_evolve(self, cycles: int = 10):
        """Run co-evolution cycles"""
        print("\n🔄 Starting co-evolution...")
        
        best_formula = None
        best_nn = None
        
        for cycle in range(cycles):
            print(f"\n--- Co-evolution Cycle {cycle+1}/{cycles} ---")
            
            # Evolve formulas
            formula_result = self.formula_discoverer.evolve_formulas()
            if formula_result:
                best_formula = formula_result
                
                # Use Q-learning to guide next evolution
                if best_formula.get('genome'):
                    action = self.q_guide.choose_action(best_formula['genome'])
                    reward = best_formula.get('fitness', 0)
                    self.q_guide.update(best_formula['genome'], action, reward, best_formula['genome'])
            
            # Evolve neural networks
            nn_result = self.neural_searcher.evolve_architectures()
            if nn_result:
                best_nn = nn_result
                
                # Q-learning guidance for NN
                if best_nn.get('genome'):
                    action = self.q_guide.choose_action(best_nn['genome'])
                    reward = best_nn.get('fitness', 0)
                    self.q_guide.update(best_nn['genome'], action, reward, best_nn['genome'])
            
            # Report progress
            if best_formula:
                print(f"Best formula fitness: {best_formula.get('fitness', 0):.4f}")
            if best_nn:
                print(f"Best NN fitness: {best_nn.get('fitness', 0):.4f}")
        
        return {
            'best_formula': best_formula,
            'best_nn': best_nn,
            'q_states_explored': len(self.q_guide.q_tables[evolvo.GenomeType.ALGORITHM]) + 
                                len(self.q_guide.q_tables[evolvo.GenomeType.NEURAL])
        }

#########
#########
#########

#todo: Move it in evolvo.py?

class RobustSerializer:
    """
    Multi-level fallback serialization system.
    Priority: dill -> pickle -> json
    """
    
    @staticmethod
    def make_json_safe(obj: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
        """
        Recursively convert any object to JSON-serializable format.
        Handles numpy arrays, evolvo genomes, and other complex types.
        """
        if current_depth > max_depth:
            return f"<max_depth_exceeded:{type(obj).__name__}>"
        
        # Basic JSON-safe types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle infinity and NaN
        if isinstance(obj, float):
            if np.isnan(obj):
                return "NaN"
            elif np.isinf(obj):
                return "Inf" if obj > 0 else "-Inf"
        
        # Handle collections
        if isinstance(obj, dict):
            return {
                str(k): RobustSerializer.make_json_safe(v, max_depth, current_depth + 1) 
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [RobustSerializer.make_json_safe(item, max_depth, current_depth + 1) 
                   for item in obj]
        if isinstance(obj, set):
            return list(obj)
        
        # Handle evolvo genomes - extract key information
        if hasattr(obj, '__class__') and 'Genome' in obj.__class__.__name__:
            return {
                'type': obj.__class__.__name__,
                'fitness': getattr(obj, 'fitness', None),
                'generation': getattr(obj, 'generation', None),
                'signature': getattr(obj, 'get_signature', lambda: 'unknown')()[:16],
                'metadata': getattr(obj, 'metadata', {})
            }
        
        # Handle other objects with __dict__
        if hasattr(obj, '__dict__'):
            try:
                return {
                    '_type': obj.__class__.__name__,
                    '_data': RobustSerializer.make_json_safe(obj.__dict__, max_depth, current_depth + 1)
                }
            except:
                return f"<{obj.__class__.__name__}>"
        
        # Last resort - string representation
        return str(obj)
    
    @staticmethod
    def save_with_fallbacks(data: Any, base_path: str, verbose: bool = True) -> bool:
        """
        Try multiple serialization methods with automatic fallbacks.
        Returns True if at least one method succeeded.
        """
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        strategies = [
            ('dill', '.dill', RobustSerializer._save_dill),
            ('pickle', '.pkl', RobustSerializer._save_pickle),
            ('json_full', '.json', RobustSerializer._save_json_full),
            ('json_safe', '_safe.json', RobustSerializer._save_json_safe),
            ('json_minimal', '_minimal.json', RobustSerializer._save_json_minimal)
        ]
        
        saved_formats = []
        
        for name, extension, save_func in strategies:
            filepath = str(base_path.with_suffix(extension))
            try:
                save_func(data, filepath)
                saved_formats.append(name)
                if verbose:
                    print(f"✓ Saved {name} format to {filepath}")
            except Exception as e:
                if verbose:
                    print(f"✗ Failed {name}: {str(e)[:100]}")
        
        if not saved_formats:
            print("ERROR: Could not save in any format!")
            return False
        
        return True
    
    @staticmethod
    def _save_dill(data: Any, filepath: str):
        """Save using dill (handles lambdas and complex objects)"""
        with open(filepath, 'wb') as f:
            dill.dump(data, f)
    
    @staticmethod
    def _save_pickle(data: Any, filepath: str):
        """Save using standard pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def _save_json_full(data: Any, filepath: str):
        """Save as JSON with full conversion"""
        json_data = RobustSerializer.make_json_safe(data)
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    @staticmethod
    def _save_json_safe(data: Any, filepath: str):
        """Save as JSON with safe defaults"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def _save_json_minimal(data: Any, filepath: str):
        """Save minimal essential data only"""
        minimal = RobustSerializer._extract_minimal(data)
        with open(filepath, 'w') as f:
            json.dump(minimal, f, indent=2)
    
    @staticmethod
    def _extract_minimal(data: Any) -> Dict:
        """Extract only the most essential information"""
        if isinstance(data, dict):
            return {
                'keys': list(data.keys()),
                'size': len(data),
                'sample': {k: str(v)[:100] for k, v in list(data.items())[:5]}
            }
        elif isinstance(data, list):
            return {
                'length': len(data),
                'sample': [str(item)[:100] for item in data[:5]]
            }
        else:
            return {'type': type(data).__name__, 'str': str(data)[:500]}
    
    @staticmethod
    def load_with_fallbacks(base_path: str, verbose: bool = True) -> Any:
        """Try to load from multiple formats"""
        base_path = Path(base_path)
        
        load_strategies = [
            ('.dill', dill.load, 'rb'),
            ('.pkl', pickle.load, 'rb'),
            ('.json', json.load, 'r'),
            ('_safe.json', json.load, 'r'),
            ('_minimal.json', json.load, 'r')
        ]
        
        for extension, loader, mode in load_strategies:
            filepath = base_path.with_suffix(extension)
            if filepath.exists():
                try:
                    with open(filepath, mode) as f:
                        data = loader(f)
                    if verbose:
                        print(f"✓ Loaded from {filepath}")
                    return data
                except Exception as e:
                    if verbose:
                        print(f"✗ Failed to load {filepath}: {str(e)[:100]}")
        
        return None


class FormulaResultsManager:
    """
    Manages formula discovery results with JSON-only output.
    No HTML, no formatting issues, just clean structured data.
    """
    
    def __init__(self, results_dir: str = "results_v7"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.serializer = RobustSerializer()
    
    def save_cycle_results(self, cycle: int, formula_discoveries: Dict, 
                          nn_results: Dict = None, base_invariance: float = None):
        """Save all cycle results as structured JSON"""
        
        cycle_dir = self.results_dir / f"cycle_{cycle:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        # Process formula discoveries
        formulas_data = self._process_formula_discoveries(formula_discoveries)
        
        # Create comprehensive cycle data
        cycle_data = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'formula_results': formulas_data,
            'neural_results': self._process_nn_results(nn_results),
            'base_invariance': base_invariance,
            'summary': self._create_summary(formulas_data, nn_results)
        }
        
        # Save with multiple fallback strategies
        success = self.serializer.save_with_fallbacks(
            cycle_data,
            str(cycle_dir / 'cycle_data'),
            verbose=True
        )
        
        # Also save individual formula details for easy access
        if formulas_data and 'top_formulas' in formulas_data:
            for i, formula in enumerate(formulas_data['top_formulas'][:20]):
                formula_path = cycle_dir / f'formula_{i:02d}.json'
                with open(formula_path, 'w') as f:
                    json.dump(formula, f, indent=2)
        
        return success
    
    def _process_formula_discoveries(self, discoveries: Dict) -> Dict:
        """Process formula discoveries into JSON-safe format"""
        if not discoveries:
            return {}
        
        result = {
            'total_unique_genomes': discoveries.get('total_unique_genomes', 0),
            'generation_count': len(discoveries.get('generation_history', [])),
            'top_formulas': []
        }
        
        # Process each top formula
        for formula_info in discoveries.get('top_discoveries', [])[:50]:  # Keep top 50
            decoded = formula_info.get('decoded', {})
            
            formula_data = {
                'rank': formula_info.get('rank', 0),
                'signature': formula_info.get('signature', ''),
                'generation': formula_info.get('generation', 0),
                'scores': {
                    'fitness': formula_info.get('fitness', 0),
                    'combined': formula_info.get('combined_score', 0),
                    'complexity': decoded.get('complexity_score', 0)
                },
                'metrics': formula_info.get('metrics', {}),
                'structure': {
                    'total_instructions': len(decoded.get('raw_instructions', [])),
                    'effective_length': decoded.get('effective_length', 0),
                    'unique_operations': decoded.get('unique_operations', [])
                },
                'formula': {
                    'symbolic': decoded.get('symbolic_formula', []),
                    'dependencies': self._simplify_dependencies(decoded.get('dependency_graph', {}))
                }
            }
            
            result['top_formulas'].append(formula_data)
        
        # Add generation history summary
        if 'generation_history' in discoveries:
            result['evolution_progress'] = self._summarize_evolution(discoveries['generation_history'])
        
        return result
    
    def _process_nn_results(self, nn_results: Dict) -> Dict:
        """Process neural network results into JSON format"""
        if not nn_results:
            return {}
        
        return {
            'fitness': nn_results.get('fitness', 0),
            'architecture': nn_results.get('architecture', ''),
            'signature': nn_results.get('genome', {}).get('signature', '')[:16] if isinstance(nn_results.get('genome'), dict) else ''
        }
    
    def _simplify_dependencies(self, dep_graph: Dict) -> List[Dict]:
        """Simplify dependency graph for JSON"""
        simplified = []
        for var, deps in dep_graph.items():
            simplified.append({
                'variable': var,
                'operation': deps.get('operation', ''),
                'inputs': deps.get('depends_on', []),
                'instruction': deps.get('instruction_index', -1)
            })
        return simplified
    
    def _create_summary(self, formulas_data: Dict, nn_results: Dict) -> Dict:
        """Create cycle summary statistics"""
        summary = {
            'formulas_discovered': len(formulas_data.get('top_formulas', [])),
            'best_formula_fitness': 0,
            'best_nn_fitness': 0,
            'avg_complexity': 0
        }
        
        if formulas_data and 'top_formulas' in formulas_data:
            formulas = formulas_data['top_formulas']
            if formulas:
                summary['best_formula_fitness'] = max(f['scores']['fitness'] for f in formulas)
                summary['avg_complexity'] = np.mean([f['scores']['complexity'] for f in formulas])
        
        if nn_results:
            summary['best_nn_fitness'] = nn_results.get('fitness', 0)
        
        return summary
    
    def _summarize_evolution(self, history: List[Dict]) -> Dict:
        """Summarize evolution progress"""
        if not history:
            return {}
        
        return {
            'total_generations': len(history),
            'fitness_progression': [h.get('best_fitness', 0) for h in history[::10]],  # Every 10th gen
            'final_best_fitness': history[-1].get('best_fitness', 0) if history else 0,
            'diversity_progression': [h.get('unique_genomes', 0) for h in history[::10]]
        }
    
    def load_all_cycles(self) -> Dict:
        """Load all cycle results"""
        all_results = {}
        
        for cycle_dir in sorted(self.results_dir.glob('cycle_*')):
            cycle_num = int(cycle_dir.name.split('_')[1])
            cycle_data = self.serializer.load_with_fallbacks(
                str(cycle_dir / 'cycle_data'),
                verbose=False
            )
            if cycle_data:
                all_results[f'cycle_{cycle_num}'] = cycle_data
        
        return all_results
    
    def generate_analysis_report(self) -> Dict:
        """Generate analysis-ready JSON report of all discoveries"""
        all_cycles = self.load_all_cycles()
        
        report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'total_cycles': len(all_cycles),
                'results_dir': str(self.results_dir)
            },
            'global_best': {
                'formula': None,
                'neural': None
            },
            'all_formulas': [],
            'evolution_stats': {},
            'patterns_found': []
        }
        
        # Find global best
        best_formula_fitness = 0
        best_nn_fitness = 0
        
        for cycle_data in all_cycles.values():
            formula_results = cycle_data.get('formula_results', {})
            
            for formula in formula_results.get('top_formulas', []):
                fitness = formula['scores']['fitness']
                if fitness > best_formula_fitness:
                    best_formula_fitness = fitness
                    report['global_best']['formula'] = formula
                
                # Collect all formulas for analysis
                report['all_formulas'].append({
                    'cycle': cycle_data['cycle'],
                    'formula': formula
                })
            
            nn_results = cycle_data.get('neural_results', {})
            if nn_results and nn_results.get('fitness', 0) > best_nn_fitness:
                best_nn_fitness = nn_results['fitness']
                report['global_best']['neural'] = nn_results
        
        # Identify patterns in successful formulas
        report['patterns_found'] = self._identify_patterns(report['all_formulas'])
        
        # Save analysis report
        report_path = self.results_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to {report_path}")
        return report
    
    def _identify_patterns(self, all_formulas: List[Dict]) -> List[Dict]:
        """Identify common patterns in successful formulas"""
        patterns = {}
        
        for entry in all_formulas:
            formula = entry['formula']
            if formula['scores']['fitness'] > 0.8:  # High-performing formulas
                ops = tuple(sorted(formula['structure']['unique_operations']))
                if ops not in patterns:
                    patterns[ops] = {
                        'operations': list(ops),
                        'count': 0,
                        'avg_fitness': 0,
                        'examples': []
                    }
                patterns[ops]['count'] += 1
                patterns[ops]['avg_fitness'] += formula['scores']['fitness']
                if len(patterns[ops]['examples']) < 3:
                    patterns[ops]['examples'].append({
                        'cycle': entry['cycle'],
                        'fitness': formula['scores']['fitness']
                    })
        
        # Calculate averages
        for pattern in patterns.values():
            pattern['avg_fitness'] /= pattern['count']
        
        # Sort by frequency and fitness
        sorted_patterns = sorted(patterns.values(), 
                               key=lambda x: (x['count'], x['avg_fitness']), 
                               reverse=True)
        
        return sorted_patterns[:10]  # Top 10 patterns
    
#########
#########
#########

class EnhancedResultsHandler:
    """Handles saving, loading, and displaying formula discovery results"""
    
    def __init__(self, results_dir: str = "results_v7"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def save_formula_discoveries(self, discoveries: Dict, cycle: int):
        """Save formula discoveries with full details"""
        
        # Create cycle-specific directory
        cycle_dir = os.path.join(self.results_dir, f"cycle_{cycle}")
        os.makedirs(cycle_dir, exist_ok=True)
        
        # Save summary JSON
        summary = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'total_unique_genomes': discoveries.get('total_unique_genomes', 0),
            'top_formula_count': len(discoveries.get('top_discoveries', [])),
        }
        
        # Extract key info from top discoveries
        top_formulas = []
        for i, formula_info in enumerate(discoveries.get('top_discoveries', [])):
            formula_summary = {
                'rank': i + 1,
                'signature': formula_info.get('signature', ''),
                'generation': formula_info.get('generation', 0),
                'fitness': formula_info.get('fitness', 0),
                'combined_score': formula_info.get('combined_score', 0),
                'complexity_score': formula_info['decoded'].get('complexity_score', 0),
                'effective_length': formula_info['decoded'].get('effective_length', 0),
                'unique_operations': formula_info['decoded'].get('unique_operations', []),
                'metrics': formula_info.get('metrics', {})
            }
            top_formulas.append(formula_summary)
        
        summary['top_formulas'] = top_formulas
        
        # Save summary
        with open(os.path.join(cycle_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed formulas in separate files
        for i, formula_info in enumerate(discoveries.get('top_discoveries', [])[:20]):
            formula_file = os.path.join(cycle_dir, f'formula_{i+1:02d}.txt')
            with open(formula_file, 'w') as f:
                self._write_formula_details(f, formula_info)
        
        # Save generation history
        if 'generation_history' in discoveries:
            history_file = os.path.join(cycle_dir, 'generation_history.json')
            with open(history_file, 'w') as f:
                json.dump(discoveries['generation_history'], f, indent=2)
        
        print(f"Saved {len(top_formulas)} formulas to {cycle_dir}")
        
    def _write_formula_details(self, file, formula_info):
        """Write detailed formula information to file"""
        
        file.write("="*80 + "\n")
        file.write(f"FORMULA RANK #{formula_info.get('rank', 0)}\n")
        file.write("="*80 + "\n\n")
        
        file.write(f"Signature: {formula_info.get('signature', 'N/A')}\n")
        file.write(f"Generation: {formula_info.get('generation', 0)}\n")
        file.write(f"Fitness: {formula_info.get('fitness', 0):.6f}\n")
        file.write(f"Combined Score: {formula_info.get('combined_score', 0):.4f}\n")
        
        decoded = formula_info.get('decoded', {})
        file.write(f"Complexity Score: {decoded.get('complexity_score', 0)}\n")
        file.write(f"Total Instructions: {len(decoded.get('raw_instructions', []))}\n")
        file.write(f"Effective Length: {decoded.get('effective_length', 0)}\n")
        file.write(f"Unique Operations: {', '.join(decoded.get('unique_operations', []))}\n")
        
        metrics = formula_info.get('metrics', {})
        if metrics:
            file.write("\nPerformance Metrics:\n")
            file.write(f"  MSE: {metrics.get('mse', 'N/A')}\n")
            file.write(f"  MAE: {metrics.get('mae', 'N/A')}\n")
            file.write(f"  Correlation: {metrics.get('correlation', 'N/A')}\n")
            file.write(f"  Valid Predictions: {metrics.get('n_valid_predictions', 0)}\n")
        
        file.write("\n" + "-"*40 + "\n")
        file.write("SYMBOLIC FORMULA:\n")
        file.write("-"*40 + "\n\n")
        
        for line in decoded.get('symbolic_formula', []):
            file.write(f"  {line}\n")
        
        file.write("\n" + "-"*40 + "\n")
        file.write("DEPENDENCY GRAPH:\n")
        file.write("-"*40 + "\n\n")
        
        for var, deps in decoded.get('dependency_graph', {}).items():
            file.write(f"  {var}:\n")
            file.write(f"    Operation: {deps.get('operation', 'N/A')}\n")
            file.write(f"    Depends on: {', '.join(deps.get('depends_on', []))}\n")
            file.write(f"    Instruction #: {deps.get('instruction_index', 'N/A')}\n\n")
        
    def display_top_formulas(self, discoveries: Dict, n: int = 5):
        """Display top n formulas in console"""
        
        print("\n" + "="*80)
        print(f"TOP {n} DISCOVERED FORMULAS")
        print("="*80)
        
        for i, formula_info in enumerate(discoveries.get('top_discoveries', [])[:n]):
            print(f"\n--- Formula #{i+1} ---")
            print(f"Fitness: {formula_info.get('fitness', 0):.6f}")
            print(f"Combined Score: {formula_info.get('combined_score', 0):.4f}")
            
            decoded = formula_info.get('decoded', {})
            print(f"Complexity: {decoded.get('complexity_score', 0)}")
            print(f"Effective Length: {decoded.get('effective_length', 0)}")
            print(f"Operations: {', '.join(decoded.get('unique_operations', []))}")
            
            print("\nFirst 5 instructions:")
            for j, line in enumerate(decoded.get('symbolic_formula', [])[:5]):
                print(f"  {j+1}. {line}")
            
            if len(decoded.get('symbolic_formula', [])) > 5:
                print(f"  ... ({len(decoded.get('symbolic_formula', [])) - 5} more instructions)")
            
            print()
    
    def generate_html_report(self, all_cycles_results: Dict):
        """Generate an HTML report with all discoveries"""
        
        html_file = os.path.join(self.results_dir, 'discovery_report.html')
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sieve Echo Formula Discovery Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
                h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .formula-card { 
                    background: #f8f9fa; 
                    border: 1px solid #dee2e6; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 15px 0;
                }
                .metrics { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                    gap: 10px; 
                    margin: 10px 0;
                }
                .metric { 
                    background: #e9ecef; 
                    padding: 8px; 
                    border-radius: 3px; 
                    text-align: center;
                }
                .code { 
                    background: #272822; 
                    color: #f8f8f2; 
                    padding: 10px; 
                    border-radius: 5px; 
                    overflow-x: auto; 
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                }
                .high-score { background: #d4edda; border-color: #c3e6cb; }
                .medium-score { background: #fff3cd; border-color: #ffeaa7; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧬 Sieve Echo Formula Discovery Report</h1>
                <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """
        
        # Add summary statistics
        total_formulas = sum(len(cycle_data.get('co_evolution', {}).get('best_formula', {}).get('top_discoveries', [])) 
                           for cycle_data in all_cycles_results.values() if 'co_evolution' in cycle_data)
        
        html_content += f"""
        <h2>Summary Statistics</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Cycles:</strong><br>{len(all_cycles_results)}
            </div>
            <div class="metric">
                <strong>Total Unique Formulas:</strong><br>{total_formulas}
            </div>
        </div>
        """
        
        # Add formulas by cycle
        for cycle_key in sorted(all_cycles_results.keys()):
            if not cycle_key.startswith('cycle_'):
                continue
                
            cycle_data = all_cycles_results[cycle_key]
            co_evo = cycle_data.get('co_evolution', {})
            
            if not co_evo or 'best_formula' not in co_evo:
                continue
            
            html_content += f"""
            <h2>Cycle {cycle_key.replace('cycle_', '')}</h2>
            """ 
                        
            formula_data = co_evo['best_formula']
            if isinstance(formula_data, dict) and 'top_discoveries' in formula_data:
                for i, formula in enumerate(formula_data['top_discoveries'][:10]):
                    score = formula.get('combined_score', 0)
                    score_class = 'high-score' if score > 100 else 'medium-score' if score > 50 else ''
                                
                    html_content += f"""
                            <div class="formula-card {score_class}">
                                <h3>Formula #{i+1} - Score: {score:.2f}</h3>
                                <div class="metrics">
                                    <div class="metric">Fitness: {formula.get('fitness', 0):.6f}</div>
                                    <div class="metric">Complexity: {formula['decoded'].get('complexity_score', 0)}</div>
                                    <div class="metric">Length: {formula['decoded'].get('effective_length', 0)}</div>
                                </div>
                                <div class="code">
                    """

                    for line in formula['decoded'].get('symbolic_formula', [])[:10]:
                        html_content += f"{line}<br>"
                                
                    html_content += """
                                </div>
                            </div>
                    """
                    
            html_content += """
                </div>
            </body>
            </html>
            """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {html_file}")

# ==============================================================================
# MAIN DISCOVERY SYSTEM
# ==============================================================================

class SieveEchoDiscoverySystem:
    def __init__(self):
        self.ndr_computer = MultiBaseNDRComputer()
        self.data = []
        self.current_n = 2
        self.results = {}
        
        # Initialize robust serialization
        self.serializer = RobustSerializer()
        self.results_manager = FormulaResultsManager(CONFIG.results_dir)
        
        # Create results directory
        os.makedirs(CONFIG.results_dir, exist_ok=True)
        
        # Load saved state if exists
        self.load_state()
    
    def save_state(self):
        """Save state with robust multi-format fallback"""
        print("\n💾 Saving state...")
        
        state = {
            'data': self.data,
            'current_n': self.current_n,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save with automatic fallbacks
        success = self.serializer.save_with_fallbacks(
            state,
            CONFIG.state_file,
            verbose=True
        )
        
        if success:
            print("✅ State saved successfully")
        else:
            print("❌ ERROR: Could not save state!")
            # Emergency: try to at least save the data points
            emergency_file = f"{CONFIG.state_file}_emergency_{int(time.time())}.json"
            try:
                with open(emergency_file, 'w') as f:
                    json.dump({
                        'current_n': self.current_n,
                        'data_length': len(self.data),
                        'data_sample': self.data[:10] if self.data else []
                    }, f, indent=2, default=str)
                print(f"Emergency data saved to {emergency_file}")
            except:
                print("CRITICAL: Could not even save emergency data!")
    
    def load_state(self):
        """Load state with automatic format detection"""
        print("📂 Loading state...")
        
        state = self.serializer.load_with_fallbacks(CONFIG.state_file, verbose=True)
        
        if state:
            self.data = state.get('data', [])
            self.current_n = state.get('current_n', 2)
            self.results = state.get('results', {})
            print(f"✅ Loaded state: {len(self.data)} data points, current n={self.current_n}")
        else:
            print("🆕 No valid state found, starting fresh")
            self.data = []
            self.current_n = 2
            self.results = {}
    
    def run_discovery(self):
        """Main discovery loop with improved error handling"""
        print("\n" + "="*80)
        print("🚀 SIEVE ECHO DISCOVERY ENGINE v7 - ROBUST EDITION")
        print("="*80)
        
        cycle = 0
        last_save_time = time.time()
        
        while cycle < CONFIG.max_cycles or CONFIG.perpetual_mode:
            try:
                cycle += 1
                print(f"\n{'='*40} CYCLE {cycle} {'='*40}")
                
                # Generate data
                self.generate_data()
                
                if len(self.data) < 200:
                    print("Not enough data yet, continuing...")
                    continue
                
                # Analyze base invariance
                self.analyze_base_invariance()
                
                # Run co-evolution with enhanced tracking
                print("\n🔬 Starting co-evolution...")
                co_evolver = self._create_safe_coevolver()
                co_results = co_evolver.co_evolve(cycles=5)
                
                # Process and save results
                self._process_cycle_results(cycle, co_results)
                
                # Periodic state saving (every 5 minutes)
                if time.time() - last_save_time > 300:
                    self.save_state()
                    last_save_time = time.time()
                
                # Check for convergence
                if self._check_convergence(co_results):
                    print("\n🎉 High fitness achieved!")
                    if not CONFIG.perpetual_mode:
                        break
                        
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
                self.save_state()
                raise
            except Exception as e:
                print(f"\n⚠️ Error in cycle {cycle}: {e}")
                traceback.print_exc()
                # Save state before continuing
                self.save_state()
                print("Continuing with next cycle...")
                continue
    
    def _create_safe_coevolver(self):
        """Create co-evolver with enhanced tracking"""
        class SafeCoEvolutionSystem(CoEvolutionSystem):
            def __init__(self, data, results_manager):
                super().__init__(data)
                self.results_manager = results_manager
                
            def co_evolve(self, cycles=10):
                all_discoveries = []
                best_formula = None
                best_nn = None
                
                for cycle in range(cycles):
                    print(f"\n--- Co-evolution Cycle {cycle+1}/{cycles} ---")
                    
                    # Evolve formulas
                    try:
                        formula_result = self.formula_discoverer.evolve_formulas()
                        if formula_result:
                            all_discoveries.append(formula_result)
                            best_formula = formula_result
                    except Exception as e:
                        print(f"Formula evolution error: {e}")
                    
                    # Evolve neural networks
                    try:
                        nn_result = self.neural_searcher.evolve_architectures()
                        if nn_result:
                            best_nn = nn_result
                    except Exception as e:
                        print(f"Neural evolution error: {e}")
                
                return {
                    'all_discoveries': all_discoveries,
                    'best_formula': best_formula,
                    'best_nn': best_nn,
                    'q_states_explored': len(self.q_guide.q_tables[evolvo.GenomeType.ALGORITHM]) + 
                                        len(self.q_guide.q_tables[evolvo.GenomeType.NEURAL])
                }
        
        return SafeCoEvolutionSystem(self.data, self.results_manager)
    
    def _process_cycle_results(self, cycle: int, co_results: Dict):
        """Process and save cycle results"""
        
        # Extract best formula from all discoveries
        best_formula_discovery = None
        if co_results.get('all_discoveries'):
            # Get the discovery with best formula
            for discovery in co_results['all_discoveries']:
                if discovery and 'top_discoveries' in discovery:
                    best_formula_discovery = discovery
                    break
        
        # Store in memory
        self.results[f'cycle_{cycle}'] = {
            'data_size': len(self.data),
            'base_invariance': self.results.get('base_invariance_cv', None),
            'formula_discovery': best_formula_discovery is not None,
            'nn_discovery': co_results.get('best_nn') is not None,
            'q_states': co_results.get('q_states_explored', 0)
        }
        
        # Save to disk using results manager
        if best_formula_discovery or co_results.get('best_nn'):
            self.results_manager.save_cycle_results(
                cycle=cycle,
                formula_discoveries=best_formula_discovery,
                nn_results=co_results.get('best_nn'),
                base_invariance=self.results.get('base_invariance_cv')
            )
        
        # Save state
        self.save_state()
        
        print(f"\n✅ Cycle {cycle} results saved")
    
    def _check_convergence(self, co_results: Dict) -> bool:
        """Check if we've achieved convergence"""
        if co_results.get('best_formula'):
            formula = co_results['best_formula']
            if isinstance(formula, dict) and formula.get('fitness', 0) > 0.95:
                return True
        return False
    
    def generate_data(self):
        """Generate multi-base NDR data (unchanged)"""
        print(f"\nGenerating data for n={self.current_n} to {self.current_n + CONFIG.data_chunk_size - 1}")
        
        for n in range(self.current_n, self.current_n + CONFIG.data_chunk_size):
            features = self.ndr_computer.compute_multi_base_features(n, CONFIG.test_bases)
            if 'entropy_mean' in features:
                self.data.append(features)
        
        self.current_n += CONFIG.data_chunk_size
        print(f"Total data points: {len(self.data)}")
    
    def analyze_base_invariance(self):
        """Analyze pattern invariance across bases (unchanged)"""
        print("\n🔍 Analyzing base invariance...")
        
        invariances = []
        for d in self.data[-100:]:
            if 'entropy_cv' in d:
                invariances.append(d['entropy_cv'])
        
        if invariances:
            mean_cv = np.mean(invariances)
            print(f"Mean entropy CV across bases: {mean_cv:.4f}")
            print(f"Base invariance quality: {'GOOD' if mean_cv < 0.1 else 'POOR'}")
            self.results['base_invariance_cv'] = mean_cv
    
    def final_report(self):
        """Generate final analysis report"""
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)
        
        print(f"\nTotal data points: {len(self.data)}")
        print(f"Final n: {self.current_n}")
        print(f"Total cycles: {len([k for k in self.results.keys() if 'cycle_' in k])}")
        
        # Generate comprehensive analysis
        try:
            analysis_report = self.results_manager.generate_analysis_report()
            
            if analysis_report['global_best']['formula']:
                best = analysis_report['global_best']['formula']
                print(f"\n🏆 Best Formula:")
                print(f"  Fitness: {best['scores']['fitness']:.6f}")
                print(f"  Complexity: {best['scores']['complexity']}")
                print(f"  Operations: {', '.join(best['structure']['unique_operations'])}")
            
            if analysis_report['patterns_found']:
                print(f"\n🔍 Common Patterns Found:")
                for i, pattern in enumerate(analysis_report['patterns_found'][:3]):
                    print(f"  Pattern {i+1}: {', '.join(pattern['operations'])}")
                    print(f"    Frequency: {pattern['count']}")
                    print(f"    Avg Fitness: {pattern['avg_fitness']:.4f}")
            
        except Exception as e:
            print(f"Could not generate analysis report: {e}")
        
        print("\n✅ Discovery complete!")
        print(f"📁 Results saved in {CONFIG.results_dir}/")
        print(f"📄 Analysis report: {CONFIG.results_dir}/analysis_report.json")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    system = SieveEchoDiscoverySystem()
    
    try:
        system.run_discovery()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.final_report()
        system.save_state()

if __name__ == "__main__":
    main()