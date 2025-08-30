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
from evolvo import EnhancedAlgorithmSerializer, RobustSerializer, FormulaResultsManager

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
    
    def evolve_formulas(self) -> Dict:
        """UPDATED: With immediate saving capability"""
        print("\n📊 Evolving formulas...")
        valid_data = [d for d in self.data if all(f in d for f in self.feature_names)]
        if len(valid_data) < 100:
            print("Not enough data for formula evolution.")
            return {}

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
        for rank, genome in enumerate(evolver.population[:50]):
            if genome.fitness and genome.fitness > 0.1:
                discovery = {'rank': rank + 1, 'fitness': genome.fitness, 'genome': genome}
                top_discoveries.append(discovery)
                
                # If we have access to the save method, use it immediately
                if hasattr(self, 'save_algorithm_immediately'):
                    cycle = getattr(self, 'current_cycle', 0)
                    self.save_algorithm_immediately(genome, cycle, rank + 1, self)
        
        return {'top_discoveries': top_discoveries}

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
    

##############################
##############################
##############################

class RealTimeSaver:
    """Save algorithms immediately as they're discovered"""
    
    def __init__(self, base_dir="realtime_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.saved_signatures = set()
    
    def save_algorithm_immediately(self, genome, cycle, rank, fitness, discoverer=None):
        """Save algorithm IMMEDIATELY upon discovery"""
        try:
            # Create unique filename
            signature = genome.get_signature()[:8] if hasattr(genome, 'get_signature') else 'unknown'
            
            # Skip if already saved
            if signature in self.saved_signatures:
                return False
            
            filename = f"cycle_{cycle:03d}_rank_{rank:03d}_fit_{fitness:.4f}_{signature}.json"
            filepath = self.base_dir / filename
            
            # Serialize with complete structure
            algorithm_data = EnhancedAlgorithmSerializer.serialize_genome_complete(genome)
            algorithm_data['cycle'] = cycle
            algorithm_data['rank'] = rank
            algorithm_data['fitness'] = fitness
            
            # Add decoded version if discoverer available
            if discoverer and hasattr(discoverer, '_decode_genome'):
                try:
                    algorithm_data['decoded'] = discoverer._decode_genome(genome)
                except:
                    pass
            
            # WRITE IMMEDIATELY
            with open(filepath, 'w') as f:
                json.dump(algorithm_data, f, indent=2, default=str)
            
            print(f"✓ SAVED: {filename} ({len(algorithm_data['instructions'])} instructions)")
            self.saved_signatures.add(signature)
            return True
            
        except Exception as e:
            print(f"✗ SAVE FAILED: {e}")
            return False

# FIX 3: MEMORY MANAGEMENT
class MemoryManager:
    """Prevent RAM over-usage"""
    
    def __init__(self, max_cache_size=10000, max_data_size=5000):
        self.max_cache_size = max_cache_size
        self.max_data_size = max_data_size
        self.gc_counter = 0
    
    def cleanup_ndr_cache(self, ndr_computer):
        """Clean NDR cache when it gets too large"""
        if len(ndr_computer.cache) > self.max_cache_size:
            # Keep only recent entries (LRU-style)
            keys_to_remove = list(ndr_computer.cache.keys())[:-self.max_cache_size//2]
            for key in keys_to_remove:
                del ndr_computer.cache[key]
            print(f"Cleaned NDR cache: {len(keys_to_remove)} entries removed")
    
    def trim_data_list(self, data_list):
        """Keep data list from growing infinitely"""
        if len(data_list) > self.max_data_size:
            # Keep most recent data
            trimmed = data_list[-self.max_data_size:]
            data_list.clear()
            data_list.extend(trimmed)
            print(f"Trimmed data list to {len(data_list)} entries")
    
    def cleanup_evolver(self, evolver):
        """Clean evolver's diversity cache"""
        if hasattr(evolver, 'diversity_cache') and len(evolver.diversity_cache) > self.max_cache_size:
            evolver.diversity_cache.clear()
            # Re-add current population signatures
            for genome in evolver.population:
                evolver.diversity_cache.add(genome.get_signature())
            print(f"Reset diversity cache to {len(evolver.diversity_cache)} entries")
    
    def force_gc(self):
        """Force garbage collection periodically"""
        self.gc_counter += 1
        if self.gc_counter % 10 == 0:
            gc.collect()
            print(f"Forced GC at counter {self.gc_counter}")

# FIX 4: PROPERLY INTEGRATE Q-LEARNING AND GPU
class EnhancedDiscoverySystem:
    """Enhanced system with all fixes integrated"""
    
    def __init__(self, original_system):
        self.system = original_system
        self.realtime_saver = RealTimeSaver()
        self.memory_manager = MemoryManager()
        self.q_learning = None  # Will initialize if needed
        
        # Enable GPU if available
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Enhanced system using device: {self.device}")
    
    def run_discovery_cycle_enhanced(self, cycle):
        """Enhanced discovery cycle with all fixes"""
        
        # MEMORY MANAGEMENT
        self.memory_manager.cleanup_ndr_cache(self.system.ndr_computer)
        self.memory_manager.trim_data_list(self.system.data)
        self.memory_manager.force_gc()
        
        # Generate data
        self.system.generate_data()
        
        # Run formula discovery
        formula_discoverer = UnifiedFormulaDiscoverer(self.system.data)
        formula_results = formula_discoverer.evolve_formulas()
        
        # IMMEDIATE SAVING - The critical fix!
        if formula_results and 'top_discoveries' in formula_results:
            saved_count = 0
            for idx, discovery in enumerate(formula_results['top_discoveries'][:50]):
                if 'genome' in discovery and discovery['genome']:
                    success = self.realtime_saver.save_algorithm_immediately(
                        discovery['genome'],
                        cycle,
                        discovery.get('rank', idx + 1),
                        discovery.get('fitness', 0),
                        formula_discoverer
                    )
                    if success:
                        saved_count += 1
            
            print(f"Cycle {cycle}: Saved {saved_count} algorithms in real-time")
        
        # Also save complete cycle data
        self.save_cycle_complete(cycle, formula_results, formula_discoverer)
        
        # Clean up evolver memory
        if hasattr(formula_discoverer, 'evolver'):
            self.memory_manager.cleanup_evolver(formula_discoverer.evolver)
        
        return formula_results
    
    def save_cycle_complete(self, cycle, formula_results, discoverer):
        """Save complete cycle data with verification"""
        cycle_dir = Path(f"results_verified/cycle_{cycle:03d}")
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete formulas
        if formula_results and 'top_discoveries' in formula_results:
            all_formulas = []
            for discovery in formula_results['top_discoveries'][:50]:
                if 'genome' in discovery and discovery['genome']:
                    formula_data = EnhancedAlgorithmSerializer.serialize_genome_complete(
                        discovery['genome'],
                        discoverer.data_config if hasattr(discoverer, 'data_config') else None
                    )
                    formula_data['rank'] = discovery.get('rank', 0)
                    formula_data['fitness'] = discovery.get('fitness', 0)
                    all_formulas.append(formula_data)
            
            # Save all formulas in one file for this cycle
            all_formulas_path = cycle_dir / 'all_formulas.json'
            with open(all_formulas_path, 'w') as f:
                json.dump(all_formulas, f, indent=2, default=str)
            
            print(f"✓ Saved {len(all_formulas)} complete formulas to {all_formulas_path}")
        
        # Save cycle summary
        summary = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'total_formulas': len(formula_results.get('top_discoveries', [])),
            'data_points': len(self.system.data),
            'current_n': self.system.current_n,
            'memory_usage_mb': self._get_memory_usage()
        }
        
        with open(cycle_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# INTEGRATION: Modify the main system's evolve_formulas method
def evolve_formulas_with_fixes(self):
    """Replacement method with all fixes"""
    print("\n📊 Evolving formulas with enhanced saving...")
    
    # Initialize real-time saver if not exists
    if not hasattr(self, 'realtime_saver'):
        self.realtime_saver = RealTimeSaver()
    
    # ... rest of evolution code ...
    # After getting top_discoveries:
    
    top_discoveries = []
    for rank, genome in enumerate(evolver.population[:50]):
        if genome.fitness and genome.fitness > 0.1:
            discovery = {'rank': rank + 1, 'fitness': genome.fitness, 'genome': genome}
            top_discoveries.append(discovery)
            
            # SAVE IMMEDIATELY!
            self.realtime_saver.save_algorithm_immediately(
                genome, 
                self.current_cycle if hasattr(self, 'current_cycle') else 0,
                rank + 1,
                genome.fitness,
                self
            )
    
    return {'top_discoveries': top_discoveries}

# ==============================================================================
# MAIN DISCOVERY SYSTEM
# ==============================================================================
class SieveEchoDiscoverySystem:
    """
    Replace these methods in your existing SieveEchoDiscoverySystem class
    """
    
    def __init__(self):
        """Updated __init__ with new components"""
        self.ndr_computer = MultiBaseNDRComputer()
        self.data = []
        self.current_n = 2
        self.results = {}
        self.serializer = RobustSerializer()
        self.results_manager = FormulaResultsManager(CONFIG.results_dir)
        
        # NEW: Real-time saving and memory management
        self.realtime_dir = Path("algorithms_realtime")
        self.realtime_dir.mkdir(parents=True, exist_ok=True)
        self.saved_signatures = set()
        self.max_cache_size = 10000
        self.max_data_size = 5000
        
        self.load_state()
    
    def save_algorithm_immediately(self, genome, cycle: int, rank: int, discoverer=None) -> bool:
        """NEW METHOD: Save algorithm as soon as it's discovered"""
        try:
            # Get unique signature
            signature = genome.get_signature()[:8] if hasattr(genome, 'get_signature') else 'unknown'
            
            # Skip if already saved
            if signature in self.saved_signatures:
                return False
            
            # Build complete algorithm data
            algorithm_data = {
                'cycle': cycle,
                'rank': rank,
                'fitness': genome.fitness,
                'signature': genome.get_signature() if hasattr(genome, 'get_signature') else None,
                'generation': genome.generation if hasattr(genome, 'generation') else 0,
                'timestamp': datetime.now().isoformat(),
                'instructions': [],
                'outputs': [],
                'data_config': {}
            }
            
            # Extract data config
            if hasattr(genome, 'data_config'):
                algorithm_data['data_config'] = genome.data_config
            
            # Extract all instructions with full detail
            if hasattr(genome, 'instructions'):
                for idx, instr in enumerate(genome.instructions):
                    if hasattr(instr, 'target'):  # Regular instruction
                        instr_data = {
                            'index': idx,
                            'type': 'instruction',
                            'target': {
                                'store': instr.target[0],
                                'index': instr.target[1],
                                'full': f"{instr.target[0]}[{instr.target[1]}]"
                            },
                            'operation': instr.operation,
                            'args': []
                        }
                        
                        # Extract arguments
                        for arg in instr.args:
                            instr_data['args'].append({
                                'store': arg[0],
                                'index': arg[1],
                                'full': f"{arg[0]}[{arg[1]}]"
                            })
                        
                        # Human-readable form
                        arg_str = ', '.join([a['full'] for a in instr_data['args']])
                        instr_data['readable'] = f"{instr_data['target']['full']} = {instr.operation}({arg_str})"
                        
                        algorithm_data['instructions'].append(instr_data)
                    else:  # Control flow
                        algorithm_data['instructions'].append({
                            'index': idx,
                            'type': 'control_flow',
                            'control_type': instr.get('type', 'unknown'),
                            'condition': instr.get('condition')
                        })
            
            # Extract outputs
            if hasattr(genome, 'outputs'):
                for out in genome.outputs:
                    algorithm_data['outputs'].append({
                        'store': out[0],
                        'index': out[1],
                        'full': f"{out[0]}[{out[1]}]"
                    })
            
            # Add decoded version if discoverer available
            if discoverer and hasattr(discoverer, '_decode_genome'):
                try:
                    algorithm_data['decoded'] = discoverer._decode_genome(genome)
                except Exception as e:
                    algorithm_data['decoded'] = {'error': str(e)}
            
            # Save to file immediately
            filename = f"cycle_{cycle:03d}_rank_{rank:03d}_fit_{algorithm_data['fitness']:.4f}_{signature}.json"
            filepath = self.realtime_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(algorithm_data, f, indent=2, default=str)
            
            print(f"  ✓ SAVED: {filename} ({len(algorithm_data['instructions'])} instructions)")
            self.saved_signatures.add(signature)
            
            # Also save a simplified version for quick access
            simple_path = self.realtime_dir / "latest" / f"c{cycle:03d}_r{rank:02d}.json"
            simple_path.parent.mkdir(exist_ok=True)
            with open(simple_path, 'w') as f:
                json.dump({
                    'fitness': algorithm_data['fitness'],
                    'instructions': [i['readable'] for i in algorithm_data['instructions'] if 'readable' in i]
                }, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to save algorithm: {e}")
            traceback.print_exc()
            return False
    
    def clean_memory(self):
        """NEW METHOD: Clean up memory to prevent RAM over-usage"""
        # Clean NDR cache
        if len(self.ndr_computer.cache) > self.max_cache_size:
            old_size = len(self.ndr_computer.cache)
            keys_to_keep = list(self.ndr_computer.cache.keys())[-self.max_cache_size//2:]
            new_cache = {k: self.ndr_computer.cache[k] for k in keys_to_keep}
            self.ndr_computer.cache = new_cache
            print(f"  Cleaned NDR cache: {old_size} → {len(self.ndr_computer.cache)} entries")
        
        # Trim data list
        if len(self.data) > self.max_data_size:
            old_size = len(self.data)
            self.data = self.data[-self.max_data_size:]
            print(f"  Trimmed data: {old_size} → {len(self.data)} entries")
        
        # Clean results history (keep only last 10 cycles in memory)
        if len(self.results) > 10:
            cycles_to_keep = sorted(self.results.keys())[-10:]
            new_results = {k: self.results[k] for k in cycles_to_keep}
            self.results = new_results
            print(f"  Cleaned results: keeping last 10 cycles")
        
        # Force garbage collection
        gc.collect()
    
    def run_discovery(self):
        """UPDATED: Main discovery loop with real-time saving and memory management"""
        print("\n🚀 SIEVE ECHO DISCOVERY ENGINE v10 - ENHANCED 🚀")
        print(f"Real-time results will be saved to: {self.realtime_dir}")
        
        cycle = max([int(k.split('_')[-1]) for k in self.results.keys() if k.startswith('cycle_')] + [0])
        
        while cycle < CONFIG.max_cycles or CONFIG.perpetual_mode:
            try:
                cycle += 1
                print(f"\n{'='*30} CYCLE {cycle} {'='*30}")
                
                # MEMORY CLEANUP at start of each cycle
                self.clean_memory()
                
                # Generate data
                self.generate_data()
                if len(self.data) < 200:
                    print("Not enough data yet, continuing...")
                    continue
                
                # Analyze base invariance
                base_invariance = self.analyze_base_invariance()
                
                # Formula evolution
                print("\n🔬 Starting formula evolution...")
                formula_discoverer = UnifiedFormulaDiscoverer(self.data)
                
                # Store current cycle for the discoverer to use
                formula_discoverer.current_cycle = cycle
                
                # Inject real-time saving into the discoverer
                formula_discoverer.save_algorithm_immediately = self.save_algorithm_immediately
                
                formula_results = self.evolve_formulas_with_realtime_saving(formula_discoverer, cycle)
                
                # Count how many were saved
                saved_count = 0
                if formula_results and 'top_discoveries' in formula_results:
                    for discovery in formula_results['top_discoveries']:
                        if discovery.get('saved', False):
                            saved_count += 1
                
                print(f"\n✅ Cycle {cycle} complete: {saved_count} algorithms saved in real-time")
                
                # Store cycle results (lighter version now since algorithms are saved separately)
                self.results[f'cycle_{cycle}'] = {
                    'formulas_saved': saved_count,
                    'base_invariance': base_invariance,
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(self.data),
                    'current_n': self.current_n
                }
                
                # Save state (now much lighter)
                self.save_state_light()
                
                # Also save cycle summary
                self.save_cycle_summary(cycle, formula_results, base_invariance)
                
                # Extra memory cleanup every 10 cycles
                if cycle % 10 == 0:
                    print("\n🧹 Deep memory cleanup...")
                    self.deep_clean_memory()
                        
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user.")
                self.save_final_report(cycle)
                break
            except Exception:
                print(f"\n⚠️ ERROR in cycle {cycle}:")
                traceback.print_exc()
                self.save_state_light()
                continue
        
        self.save_final_report(cycle)
    
    def evolve_formulas_with_realtime_saving(self, discoverer, cycle: int) -> Dict:
        """NEW METHOD: Wrapper for formula evolution with real-time saving"""
        # Run the original evolution
        formula_results = discoverer.evolve_formulas()
        
        # Now save each discovery immediately
        if formula_results and 'top_discoveries' in formula_results:
            for idx, discovery in enumerate(formula_results['top_discoveries'][:50]):
                if 'genome' in discovery and discovery['genome']:
                    saved = self.save_algorithm_immediately(
                        discovery['genome'],
                        cycle,
                        discovery.get('rank', idx + 1),
                        discoverer
                    )
                    discovery['saved'] = saved  # Mark as saved
        
        return formula_results
    
    def save_state_light(self):
        """UPDATED: Lighter state saving (algorithms are saved separately)"""
        print("\n💾 Saving light state...")
        
        state = {
            'current_n': self.current_n,
            'total_data_points': len(self.data),
            'total_cycles': len([k for k in self.results.keys() if k.startswith('cycle_')]),
            'cycle_summaries': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Only save summaries, not full results
        for cycle_key, cycle_data in self.results.items():
            if cycle_key.startswith('cycle_'):
                state['cycle_summaries'][cycle_key] = {
                    'formulas_saved': cycle_data.get('formulas_saved', 0),
                    'base_invariance': cycle_data.get('base_invariance'),
                    'timestamp': cycle_data.get('timestamp')
                }
        
        # Save just the state file
        try:
            with open(CONFIG.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"  ✓ Light state saved to {CONFIG.state_file}")
        except Exception as e:
            print(f"  ✗ Error saving state: {e}")
    
    def save_cycle_summary(self, cycle: int, formula_results: Dict, base_invariance: float):
        """NEW METHOD: Save cycle summary separately"""
        summary_dir = Path("cycle_summaries")
        summary_dir.mkdir(exist_ok=True)
        
        summary = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'formulas_discovered': len(formula_results.get('top_discoveries', [])) if formula_results else 0,
            'base_invariance': base_invariance,
            'data_points': len(self.data),
            'current_n': self.current_n,
            'memory_usage_mb': self._get_memory_usage(),
            'ndr_cache_size': len(self.ndr_computer.cache)
        }
        
        # Add top 3 fitness scores
        if formula_results and 'top_discoveries' in formula_results:
            top_fitnesses = []
            for d in formula_results['top_discoveries'][:3]:
                if 'fitness' in d:
                    top_fitnesses.append(d['fitness'])
            summary['top_fitnesses'] = top_fitnesses
        
        filepath = summary_dir / f"cycle_{cycle:03d}_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Cycle summary saved to {filepath}")
    
    def deep_clean_memory(self):
        """NEW METHOD: Deeper memory cleanup every N cycles"""
        # Clear all caches
        self.ndr_computer.cache.clear()
        print("  Cleared entire NDR cache")
        
        # Keep only essential data
        self.data = self.data[-1000:]
        print(f"  Kept only last 1000 data points")
        
        # Clear saved signatures older than 100 cycles
        self.saved_signatures.clear()
        
        # Force multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        print(f"  Memory usage: {self._get_memory_usage():.1f} MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def save_final_report(self, final_cycle: int):
        """NEW METHOD: Save comprehensive final report"""
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)
        
        # Count total saved algorithms
        saved_files = list(self.realtime_dir.glob("cycle_*.json"))
        
        report = {
            'final_cycle': final_cycle,
            'total_algorithms_saved': len(saved_files),
            'total_data_points': len(self.data),
            'final_n': self.current_n,
            'timestamp': datetime.now().isoformat(),
            'saved_algorithms_by_cycle': {}
        }
        
        # Count by cycle
        for cycle in range(1, final_cycle + 1):
            cycle_files = list(self.realtime_dir.glob(f"cycle_{cycle:03d}_*.json"))
            if cycle_files:
                report['saved_algorithms_by_cycle'][cycle] = len(cycle_files)
        
        # Save report
        report_path = self.realtime_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Total algorithms saved: {report['total_algorithms_saved']}")
        print(f"Results saved in: {self.realtime_dir}/")
        print(f"Final report: {report_path}")
        
        # Also print top algorithms found
        self.print_top_algorithms()
    
    def print_top_algorithms(self):
        """NEW METHOD: Print the best algorithms found"""
        print("\n🏆 TOP ALGORITHMS FOUND:")
        
        # Find all saved algorithms
        all_algorithms = []
        for algo_file in self.realtime_dir.glob("cycle_*_rank_*.json"):
            try:
                with open(algo_file, 'r') as f:
                    data = json.load(f)
                    all_algorithms.append({
                        'file': algo_file.name,
                        'fitness': data.get('fitness', 0),
                        'cycle': data.get('cycle', 0),
                        'instructions': len(data.get('instructions', []))
                    })
            except:
                continue
        
        # Sort by fitness
        all_algorithms.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Print top 5
        for i, algo in enumerate(all_algorithms[:5], 1):
            print(f"{i}. Fitness: {algo['fitness']:.6f} | Cycle: {algo['cycle']} | "
                  f"Instructions: {algo['instructions']} | File: {algo['file']}")

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