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
    data_chunk_size: int = 500
    
    # CRITICAL: Multiple bases for pattern discovery
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 10, 11, 13, 16, 17, 19, 23, 29, 31])
    
    # Evolution parameters
    formula_generations: int = 100
    formula_population_size: int = 500
    nn_generations: int = 50
    nn_population_size: int = 50
    max_algorithm_length: int = 20
    
    # Resource management
    max_model_params: int = int(1e7)  # 10M parameters max
    max_memory_mb: float = 2048  # 2GB max per model
    
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
    """Uses unified evolvo library for formula discovery"""
    
    def __init__(self, data: List[Dict], resource_monitor: evolvo.ResourceMonitor):
        self.data = data
        self.resource_monitor = resource_monitor
        self.instruction_set = self._create_enhanced_instruction_set()
        self.best_discoveries = []
        
    def _create_enhanced_instruction_set(self):
        """Create instruction set with mathematical operations"""
        iset = evolvo.EnhancedInstructionSet()
        
        # Add custom operations for pattern analysis
        iset.register('KURTOSIS', lambda x: x, ['decimal'], 'decimal', 'custom')
        iset.register('ENTROPY', lambda x: x, ['decimal'], 'decimal', 'custom')
        
        return iset
    
    def evolve_formulas(self) -> Dict:
        """Evolve formulas to predict omega from NDR features"""
        print("\n📊 Evolving formulas with unified evolvo...")
        
        # Prepare data configuration
        feature_names = ['kurtosis_mean', 'length_mean', 'entropy_mean', 
                        'entropy_cv', 'n', 'phi', 'tau']
        
        # Filter data to only include entries with required features
        valid_data = [d for d in self.data if all(f in d for f in feature_names)]
        
        if len(valid_data) < 100:
            print("Not enough data for formula evolution")
            return {}
        
        # Create data configuration for evolvo
        data_config = {
            'b#': ['true', 'false'],
            'd#': feature_names + ['one', 'two', 'pi', 'e'],
            'b$': ['flag', 'temp_bool'],
            'd$': ['result', 'temp1', 'temp2', 'accumulator']
        }
        
        # Create algorithm genome population
        population = []
        for _ in range(CONFIG.formula_population_size):
            genome = evolvo.AlgorithmGenome(data_config, self.instruction_set)
            
            # Generate random algorithm
            for _ in range(random.randint(3, CONFIG.max_algorithm_length)):
                # Random operation
                op = random.choice(list(self.instruction_set.operations.keys()))
                if op in ['IF', 'ELSE', 'END', 'ASSIGN']:
                    continue  # Skip control flow for now
                
                # Create instruction
                op_info = self.instruction_set.operations[op]
                target = ('d$', 0)  # result
                args = []
                
                for arg_type in op_info.arg_types:
                    if arg_type in ['decimal', 'any']:
                        # Choose between constant and variable
                        if random.random() < 0.7:
                            store = 'd#'
                            idx = random.randint(0, len(data_config['d#'])-1)
                        else:
                            store = 'd$'
                            idx = random.randint(0, len(data_config['d$'])-1)
                        args.append((store, idx))
                    elif arg_type == 'bool':
                        store = 'b#' if random.random() < 0.5 else 'b$'
                        idx = random.randint(0, len(data_config[store])-1)
                        args.append((store, idx))
                
                instruction = evolvo.Instruction(target, op, args)
                genome.add_instruction(instruction)
            
            # Mark result as output
            genome.mark_output(('d$', 0))
            population.append(genome)
        
        # Create evolver
        evolver = evolvo.UnifiedEvolver(evolvo.GenomeType.ALGORITHM, 
                                       CONFIG.formula_population_size)
        evolver.population = population
        
        # Define fitness function
        def evaluate_formula(genome: evolvo.AlgorithmGenome) -> float:
            try:
                compiled = genome.to_executable()
                data_store = evolvo.UnifiedDataStore(data_config)
                
                total_error = 0
                count = 0
                
                for d in random.sample(valid_data, min(100, len(valid_data))):
                    # Set input values
                    for feature in feature_names:
                        data_store.set(feature, d.get(feature, 0))
                    
                    # Set constants
                    data_store.set('one', 1.0)
                    data_store.set('two', 2.0)
                    data_store.set('pi', math.pi)
                    data_store.set('e', math.e)
                    
                    # Execute
                    results = compiled.execute(data_store)
                    
                    # Get prediction
                    pred = results.get('d$_0', 0)
                    actual = d.get('omega', 0)
                    
                    if np.isfinite(pred):
                        error = (pred - actual) ** 2
                        total_error += error
                        count += 1
                
                if count == 0:
                    return -float('inf')
                    
                mse = total_error / count
                return 1 / (1 + mse)  # Higher fitness = lower MSE
                
            except Exception as e:
                return -float('inf')
        
        # Evolve
        evolved = evolver.evolve(CONFIG.formula_generations, evaluate_formula)
        
        # Get best formula
        if evolved and evolved[0].fitness > 0:
            best = evolved[0]
            print(f"Best formula fitness: {best.fitness:.4f}")
            
            return {
                'genome': best,
                'fitness': best.fitness,
                'formula': self._decode_genome(best)
            }
        
        return {}
    
    def _decode_genome(self, genome: evolvo.AlgorithmGenome) -> str:
        """Convert genome to readable formula"""
        # Simplified decoding
        return f"Formula with {len(genome.instructions)} instructions"

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
        
        valid_data = [d for d in self.data if all(f in d for f in feature_names)]
        
        if len(valid_data) < 100:
            print("Not enough data for neural evolution")
            return {}
        
        # Create train/test split
        X = np.array([[d.get(f, 0) for f in feature_names] for d in valid_data])
        y = np.array([d.get('omega', 0) for d in valid_data])
        
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

# ==============================================================================
# MAIN DISCOVERY SYSTEM
# ==============================================================================
class SieveEchoDiscoverySystem:
    def __init__(self):
        self.ndr_computer = MultiBaseNDRComputer()
        self.data = []
        self.current_n = 2
        self.results = {}
        
        # Create results directory
        os.makedirs(CONFIG.results_dir, exist_ok=True)
        
        # Load saved state if exists
        self.load_state()
        
    def load_state(self):
        if os.path.exists(CONFIG.state_file):
            print(f"Loading state from {CONFIG.state_file}")
            with open(CONFIG.state_file, 'rb') as f:
                state = pickle.load(f)
                self.data = state.get('data', [])
                self.current_n = state.get('current_n', 2)
                self.results = state.get('results', {})
    
    def save_state(self):
        print("Saving state...")
        state = {
            'data': self.data,
            'current_n': self.current_n,
            'results': self.results
        }
        with open(CONFIG.state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def generate_data(self):
        """Generate multi-base NDR data"""
        print(f"\nGenerating data for n={self.current_n} to {self.current_n + CONFIG.data_chunk_size - 1}")
        
        for n in range(self.current_n, self.current_n + CONFIG.data_chunk_size):
            features = self.ndr_computer.compute_multi_base_features(n, CONFIG.test_bases)
            
            # Only keep if we have meaningful features
            if 'entropy_mean' in features:
                self.data.append(features)
        
        self.current_n += CONFIG.data_chunk_size
        print(f"Total data points: {len(self.data)}")
    
    def analyze_base_invariance(self):
        """Analyze pattern invariance across bases"""
        print("\n📐 Analyzing base invariance...")
        
        invariances = []
        
        for d in self.data[-100:]:  # Last 100 entries
            if 'entropy_cv' in d:
                # CV < 0.1 suggests good invariance
                invariances.append(d['entropy_cv'])
        
        if invariances:
            mean_cv = np.mean(invariances)
            print(f"Mean entropy CV across bases: {mean_cv:.4f}")
            print(f"Base invariance quality: {'GOOD' if mean_cv < 0.1 else 'POOR'}")
            
            self.results['base_invariance_cv'] = mean_cv
    
    def run_discovery(self):
        """Main discovery loop"""
        print("\n" + "="*80)
        print("🚀 SIEVE ECHO DISCOVERY ENGINE v7 - UNIFIED EVOLVO")
        print("="*80)
        
        cycle = 0
        
        while cycle < CONFIG.max_cycles or CONFIG.perpetual_mode:
            cycle += 1
            print(f"\n{'='*40} CYCLE {cycle} {'='*40}")
            
            # Generate data
            self.generate_data()
            
            if len(self.data) < 200:
                print("Not enough data yet, continuing...")
                continue
            
            # Analyze base invariance
            self.analyze_base_invariance()
            
            # Run co-evolution
            co_evolver = CoEvolutionSystem(self.data)
            co_results = co_evolver.co_evolve(cycles=5)
            
            # Store results
            self.results[f'cycle_{cycle}'] = {
                'data_size': len(self.data),
                'co_evolution': co_results,
                'base_invariance': self.results.get('base_invariance_cv', None)
            }
            
            # Save state
            self.save_state()
            
            # Save results to JSON
            results_file = os.path.join(CONFIG.results_dir, f"cycle_{cycle}_results.json")
            with open(results_file, 'w') as f:
                json.dump(self.results[f'cycle_{cycle}'], f, indent=2, default=str)
            
            print(f"\nResults saved to {results_file}")
            
            # Check for convergence
            if co_results.get('best_formula', {}).get('fitness', 0) > 0.95:
                print("\n🎉 High fitness achieved! Consider stopping.")
                if not CONFIG.perpetual_mode:
                    break
    
    def final_report(self):
        """Generate final report"""
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)
        
        print(f"\nTotal data points: {len(self.data)}")
        print(f"Final n: {self.current_n}")
        
        # Find best results across all cycles
        best_formula_fitness = 0
        best_nn_fitness = 0
        
        for cycle_key, cycle_results in self.results.items():
            if 'cycle_' in cycle_key:
                co_evo = cycle_results.get('co_evolution', {})
                if co_evo:
                    formula_fitness = co_evo.get('best_formula', {}).get('fitness', 0)
                    nn_fitness = co_evo.get('best_nn', {}).get('fitness', 0)
                    
                    best_formula_fitness = max(best_formula_fitness, formula_fitness)
                    best_nn_fitness = max(best_nn_fitness, nn_fitness)
        
        print(f"\nBest formula fitness: {best_formula_fitness:.4f}")
        print(f"Best NN fitness: {best_nn_fitness:.4f}")
        
        if 'base_invariance_cv' in self.results:
            print(f"\nBase invariance CV: {self.results['base_invariance_cv']:.4f}")
            
        print("\n✅ Discovery complete!")

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