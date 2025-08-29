#!/usr/bin/env python3
"""
Sieve Echo Conjecture - Unified Evolvo Discovery Engine v10
Key improvements:
- Moves utility classes (Serializer, ResultsManager) into the evolvo library.
- Implements a complexity bonus in the fitness function to reward longer, more diverse algorithms.
- Fixes serialization crashes by saving a curated state dictionary instead of entire class instances.
- Increases the number of saved top formulas per cycle to 50.
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
from concurrent.futures import ProcessPoolExecutor, as_completed # cpu_count doesn't exists

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
    perpetual_mode: bool = True
    max_cycles: int = 100
    data_chunk_size: int = 1000
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 10, 11, 13, 17, 19, 23, 29, 31])
    formula_generations: int = 200
    formula_population_size: int = 1000
    nn_generations: int = 50
    nn_population_size: int = 200
    max_algorithm_length: int = 60
    max_model_params: int = int(1e8)
    max_memory_mb: float = 4096 
    state_file: str = "sieve_echo_state_v10.json" # Use JSON for safe state
    results_dir: str = "results_v10"
    
CONFIG = Config()

# ==============================================================================
# NDR (Normalized Digit Representation) COMPUTER
# ==============================================================================
class MultiBaseNDRComputer:
    def __init__(self):
        self.cache = {}
        
    def compute_ndr(self, n: int, base: int) -> np.ndarray:
        if (n, base) in self.cache: return self.cache[(n, base)]
        if math.gcd(n, base) != 1: return np.array([])
        remainders, digits, r, pos = {}, [], 1, 0
        while r != 0 and r not in remainders:
            remainders[r] = pos
            r *= base
            digits.append(r // n)
            r %= n
            pos += 1
            if len(digits) > n + 1: break
        repetend = digits[remainders.get(r, 0):]
        ndr = np.array(repetend) / (base - 1) if base > 1 else np.array([])
        self.cache[(n, base)] = ndr
        return ndr
    
    def compute_multi_base_features(self, n: int, bases: List[int]) -> Dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features = {'n': n, 'omega': len(factorint(n)), 'phi': totient(n), 'tau': len(divisors(n))}
            data_points = defaultdict(list)
            for base in bases:
                ndr = self.compute_ndr(n, base)
                if len(ndr) > 3:
                    data_points['length'].append(len(ndr))
                    data_points['kurtosis'].append(stats.kurtosis(ndr))
                    spectrum = np.abs(fft(ndr))[:len(ndr)//2]
                    if np.sum(spectrum) > 1e-10:
                        p = spectrum / np.sum(spectrum)
                        data_points['entropy'].append(-np.sum(p[p > 1e-10] * np.log(p[p > 1e-10])))
            for name, data in data_points.items():
                if data:
                    features[f'{name}_mean'] = np.mean(data)
                    features[f'{name}_std'] = np.std(data)
                    if name == 'entropy': features['entropy_cv'] = np.std(data) / (np.mean(data) + 1e-10)
            return features

# ==============================================================================
# EVOLVO-BASED FORMULA DISCOVERER
# ==============================================================================
class UnifiedFormulaDiscoverer:
    def __init__(self, data: List[Dict]):
        self.data = data
        self.instruction_set = evolvo.EnhancedInstructionSet()
        self.feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean', 'entropy_cv', 'n', 'phi', 'tau']
        self.data_config = {
            'b#': [], 'd#': self.feature_names + ['one', 'pi', 'e'],
            'b$': [f'b{i}' for i in range(4)], 'd$': [f'd{i}' for i in range(8)]
        }

    def _decode_genome(self, genome: evolvo.AlgorithmGenome) -> Dict:
        decoded = {'symbolic_formula': [], 'unique_operations': set(), 'effective_length': len(genome.instructions)}
        for instr in genome.instructions:
            if isinstance(instr, evolvo.Instruction):
                target_type, target_idx = instr.target
                target_name = f'd${target_idx}' if target_type == 'd$' else f'b${target_idx}'
                arg_names = [self.data_config['d#'][i] if t == 'd#' else f'd${i}' for t, i in instr.args]
                op_symbols = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/', 'POW': '**'}
                op = instr.operation
                symbolic = f"{target_name} = {op}({', '.join(arg_names)})"
                if op in op_symbols and len(arg_names) == 2:
                    symbolic = f"{target_name} = ({arg_names[0]} {op_symbols[op]} {arg_names[1]})"
                decoded['symbolic_formula'].append(symbolic)
                decoded['unique_operations'].add(op)
        decoded['unique_operations'] = list(decoded['unique_operations'])
        return decoded
    
    def _evaluate_genome(self, genome: evolvo.AlgorithmGenome, data_sample: List[Dict]) -> float:
        try:
            compiled = genome.to_executable()
            data_store = evolvo.UnifiedDataStore(self.data_config)
            predictions, actuals = [], []
            
            for d in data_sample:
                for feature in self.feature_names: data_store.set(feature, d.get(feature, 0))
                data_store.set('one', 1.0); data_store.set('pi', math.pi); data_store.set('e', math.e)
                
                results = compiled.execute(data_store)
                pred = results.get('d$_0', 0)
                if np.isfinite(pred):
                    predictions.append(pred); actuals.append(d.get('omega', 0))
            
            if not predictions: return 0.0
            
            mse = np.mean((np.array(predictions) - np.array(actuals))**2)
            # --- START: COMPLEXITY BONUS ---
            complexity_score = len(genome.instructions) + len(set(i.operation for i in genome.instructions if isinstance(i, evolvo.Instruction))) * 2
            fitness = (1 / (1 + mse)) + complexity_score * 0.001
            # --- END: COMPLEXITY BONUS ---
            return fitness
        except Exception: return 0.0
    
    def evolve_formulas(self) -> Dict:
        print("\n📊 Evolving formulas...")
        valid_data = [d for d in self.data if all(f in d for f in self.feature_names)]
        if len(valid_data) < 100:
            print("Not enough data for formula evolution."); return {}

        evolver = evolvo.UnifiedEvolver(evolvo.GenomeType.ALGORITHM, CONFIG.formula_population_size)
        
        for _ in range(CONFIG.formula_population_size):
            genome = evolvo.AlgorithmGenome(self.data_config, self.instruction_set)
            for _ in range(random.randint(5, CONFIG.max_algorithm_length)):
                op_name = random.choice(list(self.instruction_set.operations.keys()))
                op_info = self.instruction_set.operations[op_name]
                if op_info.return_type != 'decimal': continue
                target = ('d$', random.randrange(len(self.data_config['d$'])))
                args = [('d#' if random.random() < 0.7 else 'd$', random.randrange(len(self.data_config['d#' if random.random() < 0.7 else 'd$']))) for _ in op_info.arg_types]
                genome.add_instruction(evolvo.Instruction(target, op_name, args))
            genome.mark_output(('d$', 0))
            evolver.add_genome(genome)
            
        evaluator = lambda g: self._evaluate_genome(g, random.sample(valid_data, min(200, len(valid_data))))
        evolver.evolve(CONFIG.formula_generations, evaluator)

        top_discoveries = []
        for rank, genome in enumerate(evolver.population[:50]): # Save top 50
            if genome.fitness and genome.fitness > 0.1:
                top_discoveries.append({'rank': rank + 1, 'fitness': genome.fitness, 'genome': genome})
        
        return {'top_discoveries': top_discoveries}

# ==============================================================================
# MAIN DISCOVERY SYSTEM
# ==============================================================================
class SieveEchoDiscoverySystem:
    def __init__(self):
        self.ndr_computer = MultiBaseNDRComputer()
        self.data = []
        self.current_n = 2
        self.results = {}
        self.serializer = RobustSerializer()
        self.results_manager = FormulaResultsManager(CONFIG.results_dir)
        self.load_state()

    def save_state(self):
        print("\n💾 Saving state...")
        
        # Process results to make them JSON-serializable
        serialized_results = {}
        for cycle_key, cycle_data in self.results.items():
            if not isinstance(cycle_data, dict):
                continue
                
            serialized_cycle = {}
            
            # Handle formula results
            if 'formula_results' in cycle_data and cycle_data['formula_results']:
                formula_results = cycle_data['formula_results']
                serialized_formulas = {
                    'top_discoveries': []
                }
                
                # Process each formula discovery
                if 'top_discoveries' in formula_results:
                    for discovery in formula_results['top_discoveries'][:50]:  # Keep top 50
                        formula_entry = {
                            'rank': discovery.get('rank', 0),
                            'fitness': discovery.get('fitness', 0)
                        }
                        
                        # Extract the actual algorithm from the genome
                        if 'genome' in discovery and discovery['genome'] is not None:
                            genome = discovery['genome']
                            
                            # Extract algorithm structure
                            if hasattr(genome, 'instructions'):
                                instructions_data = []
                                for instr in genome.instructions:
                                    if hasattr(instr, 'target'):  # Regular instruction
                                        instructions_data.append({
                                            'type': 'instruction',
                                            'target': instr.target,
                                            'operation': instr.operation,
                                            'args': instr.args
                                        })
                                    else:  # Control flow
                                        instructions_data.append(instr)
                                
                                formula_entry['algorithm'] = {
                                    'instructions': instructions_data,
                                    'outputs': list(genome.outputs) if hasattr(genome, 'outputs') else [],
                                    'signature': genome.get_signature() if hasattr(genome, 'get_signature') else None
                                }
                                
                                # Add data config if available
                                if hasattr(genome, 'data_config'):
                                    formula_entry['algorithm']['data_config'] = genome.data_config
                        
                        serialized_formulas['top_discoveries'].append(formula_entry)
                
                serialized_cycle['formula_results'] = serialized_formulas
            
            serialized_results[cycle_key] = serialized_cycle
        
        # Create complete state with all formula details
        state = {
            'current_n': self.current_n,
            'data': self.data[-1000:],  # Keep last 1000 data points to avoid huge files
            'results': serialized_results,  # Full results, not just summaries
            'total_cycles': len([k for k in self.results.keys() if k.startswith('cycle_')]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save main state file
        try:
            with open(CONFIG.state_file, 'w') as f:
                json.dump(state, f, default=str, indent=2)
            print(f"✅ State saved to {CONFIG.state_file} with {len(serialized_results)} cycles")
        except Exception as e:
            print(f"⚠️ Error saving state: {e}")
            # Try to save a minimal version
            minimal_state = {
                'current_n': self.current_n,
                'total_data_points': len(self.data),
                'total_cycles': len(serialized_results)
            }
            with open(CONFIG.state_file + '.minimal', 'w') as f:
                json.dump(minimal_state, f)
            print(f"✅ Minimal state saved to {CONFIG.state_file}.minimal")
    
    def load_state(self):
        print("📂 Loading state...")
        if Path(CONFIG.state_file).exists():
            try:
                with open(CONFIG.state_file, 'r') as f:
                    state = json.load(f)
                self.data = state.get('data', [])
                self.current_n = state.get('current_n', 2)
                # Results are not reloaded to avoid object complexity, they are saved each cycle.
                print(f"✅ Loaded state: {len(self.data)} data points, n={self.current_n}")
            except Exception as e:
                print(f"⚠️ Could not load state file, starting fresh. Error: {e}")
        else:
            print("🆕 No state file found, starting fresh.")

    def run_discovery(self):
        print("\n🚀 SIEVE ECHO DISCOVERY ENGINE v10 🚀")
        cycle = max([int(k.split('_')[-1]) for k in self.results.keys() if k.startswith('cycle_')] + [0])
        
        while cycle < CONFIG.max_cycles or CONFIG.perpetual_mode:
            try:
                cycle += 1
                print(f"\n{'='*30} CYCLE {cycle} {'='*30}")
                
                self.generate_data()
                if len(self.data) < 200: continue
                
                base_invariance = self.analyze_base_invariance()
                
                print("\n🔬 Starting formula evolution...")
                formula_discoverer = UnifiedFormulaDiscoverer(self.data)
                formula_results = formula_discoverer.evolve_formulas()
                
                # Process and decode formulas before storing
                if formula_results and 'top_discoveries' in formula_results:
                    for discovery in formula_results['top_discoveries']:
                        if 'genome' in discovery and discovery['genome']:
                            # Add decoded version to each discovery
                            try:
                                discovery['decoded'] = formula_discoverer._decode_genome(discovery['genome'])
                            except Exception as e:
                                print(f"Warning: Could not decode genome: {e}")
                                discovery['decoded'] = {}
                
                # Store results with all formula details
                self.results[f'cycle_{cycle}'] = {
                    'formula_results': formula_results,
                    'base_invariance': base_invariance,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to both locations for redundancy
                # 1. Save via FormulaResultsManager (creates individual formula files)
                self.results_manager.save_cycle_results(
                    cycle, 
                    formula_results, 
                    base_invariance=base_invariance,
                    discoverer_instance=formula_discoverer
                )
                
                # 2. Save complete state (includes all formulas in main state file)
                self.save_state()
                
                # Also save a cycle-specific backup
                cycle_backup_file = Path(CONFIG.results_dir) / f"cycle_{cycle:03d}" / "complete_state.json"
                cycle_backup_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(cycle_backup_file, 'w') as f:
                        json.dump({
                            'cycle': cycle,
                            'current_n': self.current_n,
                            'formula_results': self.serializer.make_json_safe(
                                formula_results, 
                                discoverer_instance=formula_discoverer
                            ),
                            'base_invariance': base_invariance
                        }, f, default=str, indent=2)
                    print(f"✅ Cycle backup saved to {cycle_backup_file}")
                except Exception as e:
                    print(f"⚠️ Could not save cycle backup: {e}")
                        
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user.")
                break
            except Exception:
                print(f"\n⚠️ ERROR in cycle {cycle}:")
                traceback.print_exc()
                self.save_state()
                continue
    
    def generate_data(self):
        print(f"\nGenerating data for n={self.current_n} to {self.current_n + CONFIG.data_chunk_size - 1}")
        start_n, end_n = self.current_n, self.current_n + CONFIG.data_chunk_size
        
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
        print("\n" + "="*80 + "\n📊 FINAL REPORT\n" + "="*80)
        print(f"Total data points: {len(self.data)}")
        print(f"Final n: {self.current_n}")
        print(f"📁 Results saved in {CONFIG.results_dir}/")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    system = SieveEchoDiscoverySystem()
    try:
        system.run_discovery()
    except Exception:
        print(f"\n❌ A critical error occurred:")
        traceback.print_exc()
    finally:
        print("\nExiting. Saving final state...")
        system.final_report()
        system.save_state()

if __name__ == "__main__":
    main()