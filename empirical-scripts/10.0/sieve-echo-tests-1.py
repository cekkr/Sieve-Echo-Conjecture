#!/usr/bin/env python3
"""
Sieve Echo Conjecture - Unified Evolvo Discovery Engine v9
Key improvements:
- Uses unified evolvo library instead of split evolvo_model/evolvo_nn
- Multi-base NDR pattern discovery 
- Proper evolutionary approach to finding patterns
- Co-evolution of formulas and neural architectures

"""
import copy
import traceback
from pathlib import Path

import numpy as np
import math
import random

import json
import time
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

# Core mathematical libraries
from sympy import factorint, isprime, totient, divisors
from scipy import stats
from scipy.fft import fft
from sklearn.model_selection import train_test_split

# Device configuration
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import unified Evolvo library and its components
import evolvo
from evolvo import RobustSerializer, FormulaResultsManager

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
    max_memory_mb: float = 2048 * 6  # 12GB max total model memory footprint
    
    # Files
    state_file: str = "sieve_echo_state_v7.dill" # Use dill for better object support
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
            
        remainders = {}
        digits = []
        r = 1
        pos = 0
        
        while r != 0 and r not in remainders:
            remainders[r] = pos
            r *= base
            digits.append(r // n)
            r %= n
            pos += 1
            if len(digits) > n + 1: break # Safety limit
                
        repetend = digits[remainders.get(r, 0):]
        ndr = np.array(repetend) / (base -1) if base > 1 else np.array([]) # Normalize to [0,1]
        self.cache[(n, base)] = ndr
        return ndr
    
    def compute_multi_base_features(self, n: int, bases: List[int]) -> Dict:
        """Compute features across multiple bases"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features = {
                'n': n, 'omega': len(factorint(n)), 'phi': totient(n),
                'tau': len(divisors(n)), 'is_prime': isprime(n)
            }
            
            entropies, lengths, kurtoses = [], [], []
            for base in bases:
                ndr = self.compute_ndr(n, base)
                if len(ndr) == 0: continue
                
                lengths.append(len(ndr))
                if len(ndr) > 1:
                    spectrum = np.abs(fft(ndr))[:len(ndr)//2]
                    if np.sum(spectrum) > 1e-10:
                        p = spectrum / np.sum(spectrum)
                        p = p[p > 1e-10]
                        if len(p) > 0: entropies.append(-np.sum(p * np.log(p)))
                if len(ndr) > 3:
                    kurtoses.append(stats.kurtosis(ndr))

            for name, data in [('entropy', entropies), ('length', lengths), ('kurtosis', kurtoses)]:
                if data:
                    features[f'{name}_mean'] = np.mean(data)
                    features[f'{name}_std'] = np.std(data)
                    if name == 'entropy':
                        features['entropy_cv'] = np.std(data) / (np.mean(data) + 1e-10)
            return features

# ==============================================================================
# EVOLVO-BASED FORMULA DISCOVERER
# ==============================================================================
class UnifiedFormulaDiscoverer:
    """Enhanced formula discoverer with detailed genome tracking"""
    
    def __init__(self, data: List[Dict], resource_monitor: evolvo.ResourceMonitor):
        self.data = data
        self.resource_monitor = resource_monitor
        self.instruction_set = evolvo.EnhancedInstructionSet()
        self.feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean', 'entropy_cv', 'n', 'phi', 'tau']
        self.data_config = {
            'b#': ['true', 'false'],
            'd#': self.feature_names + ['one', 'two', 'pi', 'e', 'half', 'quarter'],
            'b$': [f'b{i}' for i in range(8)], 'd$': [f'd{i}' for i in range(16)]
        }

    def _decode_genome(self, genome: evolvo.AlgorithmGenome) -> Dict:
        """Fully decode genome into readable formula representation"""
        decoded = {'symbolic_formula': [], 'unique_operations': set()}
        for instr in genome.instructions:
            if isinstance(instr, evolvo.Instruction):
                target_type, target_idx = instr.target
                target_name = f'd${target_idx}' if target_type == 'd$' else f'b${target_idx}'
                
                arg_names = []
                for arg_type, arg_idx in instr.args:
                    if arg_type == 'd#':
                        arg_name = self.data_config['d#'][arg_idx]
                    else:
                        arg_name = f'd${arg_idx}'
                    arg_names.append(arg_name)
                
                op_symbols = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/', 'POW': '**'}
                if instr.operation in op_symbols and len(arg_names) == 2:
                    symbolic = f"{target_name} = ({arg_names[0]} {op_symbols[instr.operation]} {arg_names[1]})"
                else:
                    symbolic = f"{target_name} = {instr.operation}({', '.join(arg_names)})"
                
                decoded['symbolic_formula'].append(symbolic)
                decoded['unique_operations'].add(instr.operation)

        decoded['unique_operations'] = list(decoded['unique_operations'])
        return decoded
    
    def _evaluate_genome(self, genome: evolvo.AlgorithmGenome, data_sample: List[Dict]) -> float:
        """Evaluate a single genome and return its fitness."""
        try:
            compiled = genome.to_executable()
            data_store = evolvo.UnifiedDataStore(self.data_config)
            predictions, actuals = [], []
            
            for d in data_sample:
                for feature in self.feature_names:
                    data_store.set(feature, d.get(feature, 0))
                data_store.set('one', 1.0); data_store.set('two', 2.0); data_store.set('pi', math.pi)
                data_store.set('e', math.e); data_store.set('half', 0.5); data_store.set('quarter', 0.25)
                
                results = compiled.execute(data_store)
                pred = results.get('d$_0', 0)
                if np.isfinite(pred):
                    predictions.append(pred)
                    actuals.append(d.get('omega', 0))
            
            if not predictions: return 0.0
            
            mse = np.mean((np.array(predictions) - np.array(actuals))**2)
            return 1 / (1 + mse)
        except Exception:
            return 0.0
    
    def evolve_formulas(self) -> Dict:
        """Enhanced formula evolution with comprehensive tracking"""
        print("\n📊 Evolving formulas...")
        valid_data = [d for d in self.data if all(f in d for f in self.feature_names)]
        if len(valid_data) < 100:
            print("Not enough data for formula evolution.")
            return {}

        evolver = evolvo.UnifiedEvolver(evolvo.GenomeType.ALGORITHM, CONFIG.formula_population_size)
        
        # Create initial population
        for _ in range(CONFIG.formula_population_size):
            genome = evolvo.AlgorithmGenome(self.data_config, self.instruction_set)
            for _ in range(random.randint(5, CONFIG.max_algorithm_length)):
                # Simplified but valid instruction generation
                op_name = random.choice(['ADD', 'SUB', 'MUL', 'DIV', 'POW', 'LOG', 'SIN', 'COS', 'SQRT', 'ABS'])
                op_info = self.instruction_set.operations[op_name]
                target = ('d$', random.randint(0, len(self.data_config['d$']) - 1))
                args = []
                for _ in op_info.arg_types:
                    store = 'd#' if random.random() < 0.7 else 'd$'
                    idx = random.randint(0, len(self.data_config[store]) - 1)
                    args.append((store, idx))
                genome.add_instruction(evolvo.Instruction(target, op_name, args))
            genome.mark_output(('d$', 0))
            evolver.add_genome(genome)
            
        # Evolution loop
        evaluator = lambda g: self._evaluate_genome(g, random.sample(valid_data, min(200, len(valid_data))))
        evolver.evolve(CONFIG.formula_generations, evaluator)

        # Process results
        top_discoveries = []
        for rank, genome in enumerate(evolver.population[:20]):
            if genome.fitness is not None and genome.fitness > 0.1:
                decoded = self._decode_genome(genome)
                top_discoveries.append({
                    'rank': rank + 1, 'fitness': genome.fitness,
                    'signature': genome.get_signature(), 'decoded': decoded
                })
        
        return {'top_discoveries': top_discoveries}

# ==============================================================================
# NEURAL ARCHITECTURE SEARCH
# ==============================================================================

class UnifiedNeuralSearcher:
    """Neural architecture search using unified evolvo"""
    
    def __init__(self, data: List[Dict], resource_monitor: evolvo.ResourceMonitor):
        self.data = data
        self.resource_monitor = resource_monitor
        self.feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean', 'entropy_cv', 'n', 'phi', 'tau']
        self.target_name = 'omega'
        
    def evolve_architectures(self) -> Dict:
        """Evolve neural architectures for omega prediction"""
        print("\n🧠 Evolving neural architectures...")
        
        valid_data = [d for d in self.data if all(f in d and np.isfinite(d[f]) for f in self.feature_names) and self.target_name in d and np.isfinite(d[self.target_name])]
        if len(valid_data) < 100:
            print("Not enough valid data for neural evolution.")
            return {}
            
        X = np.array([[d[f] for f in self.feature_names] for d in valid_data])
        y = np.array([d[self.target_name] for d in valid_data]).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).to(device)

        input_shape = evolvo.TensorShape(features=len(self.feature_names))
        output_shape = evolvo.TensorShape(features=1)
        
        evolver = evolvo.ResourceAwareEvolver(
            evolvo.GenomeType.NEURAL, CONFIG.nn_population_size,
            max_model_params=CONFIG.max_model_params, resource_monitor=self.resource_monitor
        )

        for _ in range(CONFIG.nn_population_size):
            genome = evolvo.NeuralGenome(input_shape, output_shape, max_params=CONFIG.max_model_params, max_memory_mb=CONFIG.max_memory_mb)
            num_layers = random.randint(2, 5)
            hidden_size = random.choice([32, 64, 128, 256])
            
            current_features = len(self.feature_names)
            for i in range(num_layers):
                is_output_layer = (i == num_layers - 1)
                out_features = 1 if is_output_layer else hidden_size
                genome.add_layer(evolvo.LayerSpec('linear', {'in_features': current_features, 'out_features': out_features}))
                if not is_output_layer:
                    genome.add_layer(evolvo.LayerSpec('relu', {}))
                    if random.random() < 0.3:
                        genome.add_layer(evolvo.LayerSpec('dropout', {'p': random.uniform(0.1, 0.5)}))
                current_features = out_features
            evolver.add_genome(genome)
        
        def evaluate_nn(genome: evolvo.NeuralGenome, model: Optional[nn.Module] = None) -> float:
            try:
                if model is None: model = genome.to_executable().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                for _ in range(30): # Reduced epochs for faster evolution
                    optimizer.zero_grad()
                    loss = criterion(model(X_train_t), y_train_t)
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    test_loss = criterion(model(X_test_t), y_test_t).item()
                return 1 / (1 + test_loss)
            except Exception:
                return 0.0

        for gen in range(CONFIG.nn_generations):
            print(f"NN Generation {gen+1}/{CONFIG.nn_generations}")
            evolver.evaluate_population_parallel(evaluate_nn, batch_size=8) # Increased batch size
            evolver.evolve(generations=1, evaluator=evaluate_nn) # Use built-in evolve step

        best_genome = evolver.population[0]
        if best_genome.fitness and best_genome.fitness > 0:
            return {
                'genome': best_genome, 'fitness': best_genome.fitness,
                'architecture': f"{len(best_genome.layers)} layers, {best_genome.estimated_params} params"
            }
        return {}


# ==============================================================================
# CO-EVOLUTION & MAIN SYSTEM
# ==============================================================================
class SieveEchoDiscoverySystem:
    def __init__(self):
        self.ndr_computer = MultiBaseNDRComputer()
        self.data = []
        self.current_n = 2
        self.results = {}
        self.serializer = RobustSerializer()
        self.results_manager = FormulaResultsManager(CONFIG.results_dir)
        self.resource_monitor = evolvo.ResourceMonitor(max_memory_mb=CONFIG.max_memory_mb)
        self.load_state()

    def save_state(self):
        print("\n💾 Saving state...")
        state = {'data': self.data, 'current_n': self.current_n, 'results': self.results}
        self.serializer.save_with_fallbacks(state, CONFIG.state_file)
    
    def load_state(self):
        print("📂 Loading state...")
        state = self.serializer.load_with_fallbacks(CONFIG.state_file)
        if state:
            self.data = state.get('data', [])
            self.current_n = state.get('current_n', 2)
            self.results = state.get('results', {})
            print(f"✅ Loaded state: {len(self.data)} data points, n={self.current_n}")
        else:
            print("🆕 No valid state found, starting fresh.")

    def run_discovery(self):
        print("\n🚀 SIEVE ECHO DISCOVERY ENGINE v9 🚀")
        cycle = max([int(k.split('_')[-1]) for k in self.results.keys() if k.startswith('cycle_')] + [0])
        
        while cycle < CONFIG.max_cycles or CONFIG.perpetual_mode:
            try:
                cycle += 1
                print(f"\n{'='*30} CYCLE {cycle} {'='*30}")
                
                self.generate_data()
                if len(self.data) < 200: continue
                
                base_invariance = self.analyze_base_invariance()
                
                print("\n🔬 Starting co-evolution...")
                formula_discoverer = UnifiedFormulaDiscoverer(self.data, self.resource_monitor)
                neural_searcher = UnifiedNeuralSearcher(self.data, self.resource_monitor)

                formula_results = formula_discoverer.evolve_formulas()
                nn_results = neural_searcher.evolve_architectures()

                self.results[f'cycle_{cycle}'] = {'formula_results': formula_results, 'nn_results': nn_results}
                self.results_manager.save_cycle_results(cycle, formula_results, nn_results, base_invariance)
                self.save_state()
                        
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user.")
                break
            except Exception as e:
                print(f"\n⚠️ ERROR in cycle {cycle}: {e}")
                traceback.print_exc()
                self.save_state()
                continue
    
    def generate_data(self):
        print(f"\nGenerating data for n={self.current_n} to {self.current_n + CONFIG.data_chunk_size - 1}")
        start_n = self.current_n
        end_n = self.current_n + CONFIG.data_chunk_size
        
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(self.ndr_computer.compute_multi_base_features, n, CONFIG.test_bases): n for n in range(start_n, end_n)}
            for future in as_completed(futures):
                features = future.result()
                if features and 'entropy_mean' in features:
                    self.data.append(features)
        
        self.current_n = end_n
        print(f"Total data points: {len(self.data)}")
    
    def analyze_base_invariance(self) -> Optional[float]:
        print("\n🔍 Analyzing base invariance...")
        invariances = [d['entropy_cv'] for d in self.data[-200:] if 'entropy_cv' in d]
        if invariances:
            mean_cv = np.mean(invariances)
            print(f"Mean entropy CV across bases: {mean_cv:.4f} ({'GOOD' if mean_cv < 0.1 else 'POOR'})")
            return mean_cv
        return None

    def final_report(self):
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)
        # Simplified final report
        best_formula_fitness = 0
        best_nn_fitness = 0
        for cycle_results in self.results.values():
            if cycle_results.get('formula_results', {}).get('top_discoveries'):
                fitness = cycle_results['formula_results']['top_discoveries'][0]['fitness']
                if fitness > best_formula_fitness: best_formula_fitness = fitness
            if cycle_results.get('nn_results', {}).get('fitness'):
                 fitness = cycle_results['nn_results']['fitness']
                 if fitness > best_nn_fitness: best_nn_fitness = fitness
        print(f"🏆 Best Formula Fitness Found: {best_formula_fitness:.4f}")
        print(f"🏆 Best Neural Net Fitness Found: {best_nn_fitness:.4f}")
        print(f"📁 Results saved in {CONFIG.results_dir}/")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    system = SieveEchoDiscoverySystem()
    try:
        system.run_discovery()
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nExiting. Saving final state...")
        system.final_report()
        system.save_state()

if __name__ == "__main__":
    main()