#!/usr/bin/env python3
"""
Sieve Echo Pattern Discovery Through Evolution
Version 8.3 - Patterns matters
Uses evolvo_model and evolvo_nn to discover patterns without preconceptions
"""

import numpy as np
import math
import signal
import sys
import time
import json
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import random
from dataclasses import dataclass
from sympy import factorint, isprime, totient, divisors
from scipy.fft import fft
from scipy import stats

# Import the evolvo libraries properly
import evolvo_model as em
import evolvo_nn as en

# Set up graceful shutdown
class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, *args):
        self.kill_now = True
        print("\n🛑 Graceful shutdown initiated...")

killer = GracefulKiller()

# ==============================================================================
# NDR (Normalized Digit Representation) Core
# ==============================================================================

class NDRComputer:
    """Computes NDR patterns without assumptions about what they mean"""
    
    def __init__(self):
        self.cache = {}
        
    def compute_repetend(self, n: int, base: int, max_length: int = 10000) -> List[int]:
        """Get repeating pattern for 1/n in given base"""
        if (n, base) in self.cache:
            return self.cache[(n, base)]
        
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
        
        self.cache[(n, base)] = result
        return result
    
    def compute_ndr(self, n: int, base: int) -> np.ndarray:
        """Normalize digits to [0,1] interval"""
        pattern = self.compute_repetend(n, base)
        if not pattern:
            return np.array([])
        return np.array(pattern) / base
    
    def extract_raw_features(self, n: int, bases: List[int]) -> Dict:
        """Extract features without assumptions about importance"""
        features = {'n': n}
        
        # Basic number theory (let evolution decide if these matter)
        factors = factorint(n)
        features['omega'] = len(factors)
        features['Omega'] = sum(factors.values())
        features['tau'] = len(divisors(n))
        features['sigma'] = sum(divisors(n))
        features['phi'] = totient(n)
        features['is_prime'] = isprime(n)
        features['is_prime_power'] = len(factors) == 1
        features['smallest_factor'] = min(factors.keys()) if factors else n
        features['largest_factor'] = max(factors.keys()) if factors else n
        features['radical'] = np.prod(list(factors.keys()))
        
        # NDR features across bases (no assumptions about which matter)
        ndr_entropies = []
        ndr_lengths = []
        ndr_means = []
        ndr_stds = []
        ndr_kurtosis = []
        ndr_skewness = []
        fft_maxfreqs = []
        fft_powers = []
        
        for base in bases:
            ndr = self.compute_ndr(n, base)
            if len(ndr) == 0:
                continue
                
            # Basic statistics
            ndr_lengths.append(len(ndr))
            ndr_means.append(np.mean(ndr))
            ndr_stds.append(np.std(ndr))
            
            if len(ndr) > 3:
                ndr_kurtosis.append(stats.kurtosis(ndr))
                ndr_skewness.append(stats.skew(ndr))
            
            # Fourier features
            if len(ndr) > 1:
                fft_result = np.abs(fft(ndr))[:len(ndr)//2]
                if len(fft_result) > 0:
                    power = fft_result**2
                    total_power = np.sum(power)
                    
                    if total_power > 0:
                        p_norm = power / total_power
                        p_norm = p_norm[p_norm > 1e-10]
                        if len(p_norm) > 0:
                            entropy = -np.sum(p_norm * np.log(p_norm))
                            ndr_entropies.append(entropy)
                    
                    fft_maxfreqs.append(np.argmax(fft_result[1:]) + 1 if len(fft_result) > 1 else 0)
                    fft_powers.append(np.max(fft_result))
        
        # Store all collected features
        if ndr_entropies:
            features['h_mean'] = np.mean(ndr_entropies)
            features['h_std'] = np.std(ndr_entropies)
            features['h_min'] = np.min(ndr_entropies)
            features['h_max'] = np.max(ndr_entropies)
            features['h_range'] = np.max(ndr_entropies) - np.min(ndr_entropies)
        
        if ndr_lengths:
            features['len_mean'] = np.mean(ndr_lengths)
            features['len_std'] = np.std(ndr_lengths)
            features['len_gcd'] = np.gcd.reduce(ndr_lengths) if len(ndr_lengths) > 1 else ndr_lengths[0]
        
        if ndr_kurtosis:
            features['kurt_mean'] = np.mean(ndr_kurtosis)
            features['kurt_std'] = np.std(ndr_kurtosis)
        
        if ndr_skewness:
            features['skew_mean'] = np.mean(ndr_skewness)
        
        if fft_maxfreqs:
            features['freq_mean'] = np.mean(fft_maxfreqs)
            features['power_max'] = np.max(fft_powers)
        
        return features

# ==============================================================================
# Evolvo Formula Discovery System
# ==============================================================================

class NDRFormulaEvolver:
    """Uses evolvo_model to evolve formulas that predict omega from NDR features"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.instruction_set = self._create_instruction_set()
        self.best_algorithm = None
        self.best_fitness = float('inf')
        self.discovered_formulas = []
        
        # Evolvo configuration
        self.store_config = {
            'd#': ['h_mean', 'h_std', 'len_mean', 'kurt_mean', 'n', 'one', 'two', 'pi', 'e'],
            'b#': ['is_prime', 'is_prime_power'],
            'd$': ['result', 'temp1', 'temp2', 'temp3'],
            'b$': ['flag']
        }
    
    def _create_instruction_set(self):
        """Create instruction set with diverse operations"""
        iset = em.get_default_instruction_set()
        
        # Add more mathematical operations
        iset.register('SQRT', lambda a: math.sqrt(abs(a)) if a >= 0 else 0, ['d'], 'decimal')
        iset.register('LOG', lambda a: math.log(abs(a) + 1e-9), ['d'], 'decimal')
        iset.register('SIN', lambda a: math.sin(a), ['d'], 'decimal')
        iset.register('COS', lambda a: math.cos(a), ['d'], 'decimal')
        iset.register('ABS', lambda a: abs(a), ['d'], 'decimal')
        iset.register('MAX', lambda a, b: max(a, b), ['d', 'd'], 'decimal')
        iset.register('MIN', lambda a, b: min(a, b), ['d', 'd'], 'decimal')
        
        return iset
    
    def create_evaluator(self):
        """Create evaluator for omega prediction"""
        class OmegaEvaluator(em.BaseEvaluator):
            def __init__(self, data, store_config, instruction_set):
                super().__init__(store_config, instruction_set)
                self.data = data
            
            def evaluate(self, algorithm, **kwargs):
                if len(algorithm) > 50:  # Limit complexity
                    return float('inf')
                
                total_error = 0.0
                count = 0
                
                for d in self.data[:min(500, len(self.data))]:  # Sample
                    if 'omega' not in d:
                        continue
                    
                    data_store = em.DataStore(self.store_config)
                    
                    # Set constants from data
                    for key in ['h_mean', 'h_std', 'len_mean', 'kurt_mean', 'n']:
                        if key in d:
                            data_store.set(key, d[key])
                        else:
                            data_store.set(key, 0)
                    
                    data_store.set('one', 1.0)
                    data_store.set('two', 2.0)
                    data_store.set('pi', math.pi)
                    data_store.set('e', math.e)
                    data_store.set('is_prime', d.get('is_prime', False))
                    data_store.set('is_prime_power', d.get('is_prime_power', False))
                    
                    try:
                        self.interpreter.execute(algorithm, data_store)
                        predicted = data_store.get('result')
                        actual = d['omega']
                        error = (predicted - actual) ** 2
                        total_error += error
                        count += 1
                    except:
                        return float('inf')
                
                if count == 0:
                    return float('inf')
                
                mse = total_error / count
                complexity_penalty = len(algorithm) * 0.001
                return mse + complexity_penalty
        
        return OmegaEvaluator(self.data, self.store_config, self.instruction_set)
    
    def generate_random_algorithm(self, max_length: int = 10) -> List:
        """Generate random algorithm"""
        algorithm = []
        ops = list(self.instruction_set.operations.keys())
        ops = [op for op in ops if op not in ['IF', 'END', 'ASSIGN']]
        
        for _ in range(random.randint(2, max_length)):
            op = random.choice(ops)
            target = ['d$', 0]  # result
            
            prop = self.instruction_set.op_properties[op]
            instruction = target + [op]
            
            for arg_type in prop['arg_types']:
                if arg_type == 'd':
                    store = random.choice(['d#', 'd$'])
                    idx = random.randint(0, 8 if store == 'd#' else 3)
                else:  # bool
                    store = random.choice(['b#', 'b$'])
                    idx = random.randint(0, 1 if store == 'b#' else 0)
                instruction.extend([store, idx])
            
            algorithm.append(instruction)
        
        return algorithm
    
    def evolve(self, generations: int = 100, population_size: int = 50):
        """Evolve formulas to predict omega"""
        evaluator = self.create_evaluator()
        
        # Initialize population
        population = []
        for _ in range(population_size):
            alg = self.generate_random_algorithm()
            fitness = evaluator.evaluate(alg)
            population.append((alg, fitness))
        
        for gen in range(generations):
            if killer.kill_now:
                break
            
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            # Track best
            if population[0][1] < self.best_fitness:
                self.best_fitness = population[0][1]
                self.best_algorithm = population[0][0]
                
                formula = self.decode_algorithm(population[0][0])
                self.discovered_formulas.append({
                    'generation': gen,
                    'fitness': self.best_fitness,
                    'formula': formula,
                    'algorithm': population[0][0]
                })
                
                print(f"📊 Gen {gen}: New formula discovered! Fitness={self.best_fitness:.4f}")
                print(f"   Formula: {formula}")
            
            # Create next generation
            new_pop = population[:population_size//5]  # Elite
            
            while len(new_pop) < population_size:
                if random.random() < 0.7:  # Crossover
                    p1 = self.tournament_select(population)
                    p2 = self.tournament_select(population)
                    child = self.crossover(p1[0], p2[0])
                else:  # Mutation
                    parent = self.tournament_select(population)
                    child = self.mutate(parent[0])
                
                fitness = evaluator.evaluate(child)
                new_pop.append((child, fitness))
            
            population = new_pop
        
        return self.best_algorithm
    
    def tournament_select(self, population, size=3):
        """Tournament selection"""
        tournament = random.sample(population, min(size, len(population)))
        return min(tournament, key=lambda x: x[1])
    
    def crossover(self, p1: List, p2: List) -> List:
        """Single-point crossover"""
        if not p1 or not p2:
            return p1 or p2
        point1 = random.randint(0, len(p1))
        point2 = random.randint(0, len(p2))
        return p1[:point1] + p2[point2:]
    
    def mutate(self, algorithm: List) -> List:
        """Mutate algorithm"""
        if not algorithm:
            return algorithm
        
        mutated = algorithm.copy()
        
        if random.random() < 0.3 and len(mutated) > 1:
            # Remove instruction
            mutated.pop(random.randint(0, len(mutated)-1))
        elif random.random() < 0.5:
            # Modify instruction
            idx = random.randint(0, len(mutated)-1)
            mutated[idx] = self.generate_random_algorithm(1)[0]
        else:
            # Add instruction
            new_inst = self.generate_random_algorithm(1)[0]
            mutated.insert(random.randint(0, len(mutated)), new_inst)
        
        return mutated
    
    def decode_algorithm(self, algorithm: List) -> str:
        """Convert to readable formula"""
        if not algorithm:
            return "empty"
        ops = []
        for inst in algorithm:
            if len(inst) >= 3:
                ops.append(inst[2])
        return " → ".join(ops) if ops else "?"

# ==============================================================================
# Neural Architecture Evolution
# ==============================================================================

class NDRNeuralEvolver:
    """Uses evolvo_nn to evolve neural architectures"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.feature_names = self._get_feature_names()
        self.best_genome = None
        self.best_accuracy = 0
        
        # Define input/output shapes
        self.input_shape = en.TensorShape(features=len(self.feature_names))
        self.output_shape = en.TensorShape(features=1)  # Predict omega
    
    def _get_feature_names(self) -> List[str]:
        """Extract numeric feature names"""
        features = set()
        for d in self.data[:100]:
            for k, v in d.items():
                if isinstance(v, (int, float)) and not math.isnan(v) and k != 'omega':
                    features.add(k)
        return sorted(list(features))[:20]  # Limit features
    
    def prepare_data(self):
        """Prepare training data"""
        X, y = [], []
        for d in self.data:
            if 'omega' not in d:
                continue
            row = []
            valid = True
            for feat in self.feature_names:
                if feat in d:
                    row.append(d[feat])
                else:
                    valid = False
                    break
            if valid:
                X.append(row)
                y.append(d['omega'])
        
        return np.array(X), np.array(y)
    
    def evaluate_genome(self, genome: en.ModelGenome) -> float:
        """Evaluate a neural architecture"""
        try:
            import torch
            import torch.nn as nn
            from sklearn.model_selection import train_test_split
            
            X, y = self.prepare_data()
            if len(X) < 100:
                return 0.0
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.LongTensor(y_test)
            
            # Create model
            model = genome.to_pytorch_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Quick training
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_train).squeeze()
                loss = criterion(outputs, y_train.float())
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                predictions = model(X_test).squeeze()
                accuracy = 1.0 - torch.mean(torch.abs(predictions - y_test.float())).item()
            
            return max(0, accuracy)
            
        except Exception as e:
            return 0.0
    
    def evolve(self, generations: int = 20):
        """Evolve neural architectures"""
        evolver = en.AdvancedModelEvolver(
            self.input_shape,
            self.output_shape,
            task_type='regression'
        )
        
        population = evolver.evolve(
            generations=generations,
            population_size=20,
            fitness_func=self.evaluate_genome,
            multi_objective=False
        )
        
        if population:
            self.best_genome = population[0]
            self.best_accuracy = population[0].fitness or 0
            print(f"🧠 Best neural architecture: {self.best_accuracy:.3f} accuracy")
        
        return self.best_genome

# ==============================================================================
# Co-Evolution System
# ==============================================================================

class CoEvolutionSystem:
    """Co-evolves formulas and neural networks"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.formula_evolver = NDRFormulaEvolver(data)
        self.neural_evolver = NDRNeuralEvolver(data)
        self.discoveries = []
    
    def run(self, cycles: int = 10):
        """Run co-evolution cycles"""
        for cycle in range(cycles):
            if killer.kill_now:
                break
            
            print(f"\n🔄 Co-evolution cycle {cycle+1}/{cycles}")
            
            # Evolve formulas
            print("  📐 Evolving formulas...")
            self.formula_evolver.evolve(generations=50)
            
            # Evolve neural networks
            print("  🧬 Evolving neural architectures...")
            self.neural_evolver.evolve(generations=10)
            
            # Record discoveries
            self.discoveries.append({
                'cycle': cycle,
                'best_formula_fitness': self.formula_evolver.best_fitness,
                'best_nn_accuracy': self.neural_evolver.best_accuracy,
                'formulas_found': len(self.formula_evolver.discovered_formulas)
            })
            
            # Report progress
            print(f"  ✅ Cycle complete: Formula fitness={self.formula_evolver.best_fitness:.4f}, "
                  f"NN accuracy={self.neural_evolver.best_accuracy:.3f}")

# ==============================================================================
# Pattern Discovery Engine
# ==============================================================================

class FreePatternExplorer:
    """Explores patterns without preconceptions"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.unexpected_findings = []
    
    def explore(self, data: List[Dict]):
        """Look for any patterns, expected or not"""
        
        # 1. Look for patterns we DON'T expect
        self._find_unexpected_correlations(data)
        
        # 2. Look for non-linear relationships
        self._explore_nonlinear(data)
        
        # 3. Look for clustering patterns
        self._explore_clusters(data)
        
        # 4. Look for outliers that might be interesting
        self._find_outliers(data)
    
    def _find_unexpected_correlations(self, data):
        """Find correlations between features we wouldn't expect to relate"""
        features = list(data[0].keys()) if data else []
        
        for f1 in features:
            for f2 in features:
                if f1 >= f2:
                    continue
                
                vals1, vals2 = [], []
                for d in data:
                    if f1 in d and f2 in d:
                        v1, v2 = d[f1], d[f2]
                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            if not (math.isnan(v1) or math.isnan(v2)):
                                vals1.append(v1)
                                vals2.append(v2)
                
                if len(vals1) > 100:
                    corr = np.corrcoef(vals1, vals2)[0, 1]
                    if abs(corr) > 0.7:  # Strong correlation
                        self.patterns['correlations'].append({
                            'features': (f1, f2),
                            'correlation': corr,
                            'unexpected': True
                        })
                        
                        # Check if this is surprising
                        if not self._is_expected_correlation(f1, f2):
                            self.unexpected_findings.append({
                                'type': 'correlation',
                                'description': f"Unexpected correlation between {f1} and {f2}: {corr:.3f}"
                            })
    
    def _is_expected_correlation(self, f1: str, f2: str) -> bool:
        """Check if correlation is expected based on feature names"""
        # These are expected to correlate
        expected_pairs = [
            ('is_prime', 'omega'),
            ('h_mean', 'h_std'),
            ('len_mean', 'phi'),
            ('tau', 'sigma')
        ]
        
        for pair in expected_pairs:
            if (f1 in pair and f2 in pair):
                return True
        return False
    
    def _explore_nonlinear(self, data):
        """Look for non-linear patterns"""
        # Example: Look for logarithmic relationships
        for feat in ['h_mean', 'len_mean', 'kurt_mean']:
            if not all(feat in d for d in data[:100]):
                continue
            
            x = [d[feat] for d in data if feat in d and d[feat] > 0]
            y = [d['omega'] for d in data if feat in d and d[feat] > 0]
            
            if len(x) > 100:
                # Try log transform
                log_x = np.log(np.array(x) + 1e-9)
                corr = np.corrcoef(log_x, y)[0, 1]
                
                if abs(corr) > 0.5:
                    self.patterns['nonlinear'].append({
                        'feature': feat,
                        'transform': 'log',
                        'correlation': corr
                    })
    
    def _explore_clusters(self, data):
        """Look for natural clustering in the data"""
        # Simple clustering based on omega values
        omega_groups = defaultdict(list)
        for d in data:
            if 'omega' in d:
                omega_groups[d['omega']].append(d)
        
        # Analyze each cluster
        for omega, group in omega_groups.items():
            if len(group) < 10:
                continue
            
            # Find what's common in this cluster
            if 'h_mean' in group[0]:
                h_means = [g['h_mean'] for g in group if 'h_mean' in g]
                if h_means:
                    self.patterns['clusters'].append({
                        'omega': omega,
                        'size': len(group),
                        'h_mean_avg': np.mean(h_means),
                        'h_mean_std': np.std(h_means)
                    })
    
    def _find_outliers(self, data):
        """Find interesting outliers"""
        if not data:
            return
        
        for feat in ['h_mean', 'h_std', 'len_mean']:
            if feat not in data[0]:
                continue
            
            vals = [d[feat] for d in data if feat in d]
            if len(vals) < 100:
                continue
            
            mean, std = np.mean(vals), np.std(vals)
            
            for d in data:
                if feat in d:
                    z_score = abs((d[feat] - mean) / std) if std > 0 else 0
                    if z_score > 3:  # 3 sigma outlier
                        self.unexpected_findings.append({
                            'type': 'outlier',
                            'n': d['n'],
                            'feature': feat,
                            'value': d[feat],
                            'z_score': z_score
                        })

# ==============================================================================
# Main Discovery System
# ==============================================================================

class SieveEchoDiscoverySystem:
    """Main system that coordinates all discovery methods"""
    
    def __init__(self):
        self.ndr_computer = NDRComputer()
        self.data = []
        self.discoveries = {
            'formulas': [],
            'architectures': [],
            'patterns': [],
            'unexpected': [],
            'statistics': {}
        }
        self.start_time = time.time()
    
    def generate_data(self, n_max: int = 5000, bases: List[int] = None):
        """Generate NDR data for analysis"""
        if bases is None:
            # Use diverse prime bases
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        
        print(f"📊 Generating data for n=2 to {n_max} across {len(bases)} bases...")
        
        for n in range(2, min(n_max, 10000)):
            if killer.kill_now:
                break
            
            if n % 100 == 0:
                print(f"  Processing n={n}...")
            
            features = self.ndr_computer.extract_raw_features(n, bases)
            self.data.append(features)
        
        print(f"✅ Generated {len(self.data)} data points")
    
    def run_discovery(self):
        """Run all discovery methods"""
        print("\n🚀 Starting pattern discovery without preconceptions...\n")
        
        # 1. Generate data
        self.generate_data()
        
        if not self.data:
            print("❌ No data generated")
            return
        
        # 2. Basic statistical analysis
        print("\n📈 Basic statistical analysis...")
        self.analyze_basic_statistics()
        
        # 3. Free pattern exploration
        print("\n🔍 Exploring patterns freely...")
        explorer = FreePatternExplorer()
        explorer.explore(self.data)
        self.discoveries['patterns'] = explorer.patterns
        self.discoveries['unexpected'] = explorer.unexpected_findings
        
        # Report unexpected findings immediately
        if explorer.unexpected_findings:
            print(f"\n⚡ Found {len(explorer.unexpected_findings)} unexpected patterns!")
            for finding in explorer.unexpected_findings[:5]:
                print(f"  - {finding['description'] if 'description' in finding else finding}")
        
        # 4. Evolve formulas
        print("\n🧬 Evolving formulas to predict omega...")
        formula_evolver = NDRFormulaEvolver(self.data)
        formula_evolver.evolve(generations=100)
        self.discoveries['formulas'] = formula_evolver.discovered_formulas
        
        # 5. Evolve neural networks
        try:
            print("\n🧠 Evolving neural architectures...")
            neural_evolver = NDRNeuralEvolver(self.data)
            neural_evolver.evolve(generations=10)
            if neural_evolver.best_genome:
                self.discoveries['architectures'].append({
                    'accuracy': neural_evolver.best_accuracy,
                    'layers': len(neural_evolver.best_genome.layers)
                })
        except ImportError:
            print("  ⚠️ PyTorch not available, skipping neural evolution")
        
        # 6. Co-evolution
        print("\n🔄 Running co-evolution...")
        coevo = CoEvolutionSystem(self.data)
        coevo.run(cycles=5)
        
        # 7. Final analysis
        self.final_analysis()
    
    def analyze_basic_statistics(self):
        """Compute basic statistics without assumptions"""
        if not self.data:
            return
        
        # Just compute correlations with omega
        correlations = {}
        
        for key in self.data[0].keys():
            if key == 'omega':
                continue
            
            vals = []
            omegas = []
            
            for d in self.data:
                if key in d and 'omega' in d:
                    v = d[key]
                    if isinstance(v, (int, float)) and not math.isnan(v):
                        vals.append(v)
                        omegas.append(d['omega'])
            
            if len(vals) > 100:
                corr = np.corrcoef(vals, omegas)[0, 1]
                correlations[key] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        self.discoveries['statistics']['correlations'] = dict(sorted_corr[:20])
        
        print(f"  Top correlations with omega:")
        for feat, corr in sorted_corr[:5]:
            print(f"    {feat}: {corr:.3f}")
    
    def final_analysis(self):
        """Generate final analysis"""
        print("\n" + "="*60)
        print("📋 FINAL ANALYSIS")
        print("="*60)
        
        runtime = time.time() - self.start_time
        
        # Summary statistics
        print(f"\n⏱️ Runtime: {runtime:.1f} seconds")
        print(f"📊 Data points analyzed: {len(self.data)}")
        print(f"🧬 Formulas discovered: {len(self.discoveries['formulas'])}")
        print(f"⚡ Unexpected patterns: {len(self.discoveries['unexpected'])}")
        
        # Best formula
        if self.discoveries['formulas']:
            best = min(self.discoveries['formulas'], key=lambda x: x['fitness'])
            print(f"\n🏆 Best formula found:")
            print(f"   Fitness: {best['fitness']:.4f}")
            print(f"   Operations: {best['formula']}")
        
        # Most unexpected findings
        if self.discoveries['unexpected']:
            print(f"\n🎯 Most unexpected discoveries:")
            for finding in self.discoveries['unexpected'][:3]:
                if 'description' in finding:
                    print(f"   - {finding['description']}")
                else:
                    print(f"   - {finding['type']}: {finding}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save all discoveries"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'runtime': time.time() - self.start_time,
            'data_size': len(self.data),
            'discoveries': self.discoveries
        }
        
        # Save JSON
        with open(f'sieve_discoveries_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save pickle for complete data
        with open(f'sieve_discoveries_{timestamp}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Results saved to sieve_discoveries_{timestamp}.json")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("="*60)
    print("🔬 SIEVE ECHO PATTERN DISCOVERY")
    print("Using evolvo framework for unbiased exploration")
    print("="*60)
    
    system = SieveEchoDiscoverySystem()
    
    try:
        system.run_discovery()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        print("\n🏁 Generating final report...")
        system.final_analysis()
    
    print("\n✨ Discovery complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)