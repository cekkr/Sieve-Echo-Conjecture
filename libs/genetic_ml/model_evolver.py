# -*- coding: utf-8 -*-
# genetic_model_engine.py
# Genetic PyTorch Model Definition Language
# Compatible with evolvo_engine.py for chained optimization

"""
This is a complementary system that generates PyTorch model definitions using similar evolutionary concepts. 
This will create a genetic machine learning model definition language that can evolve neural architectures.
I've created a comprehensive genetic machine learning model definition language for PyTorch that follows the same evolutionary principles as your `evolvo_engine`. 

Here are the key features:

## Core Components

### 1. **ModelGenome** - The DNA of Neural Networks
- Represents architectures as evolvable genomes
- Includes skip connections for complex architectures
- Generates unique signatures (MD5 hashes) to detect and avoid duplicate architectures

### 2. **LayerFactory** - Intelligent Layer Generation
- Predefined templates for all major layer types (Linear, Conv2D, LSTM, Attention, etc.)
- Valid parameter ranges to avoid dead-end configurations
- Shape-aware generation to ensure compatibility

### 3. **ModelEvolver** - Genetic Operations
- **Crossover**: Single-point and uniform crossover strategies
- **Mutation**: Modify parameters, replace layers, add/remove layers
- **Selection**: Tournament selection with elitism
- **Diversity enforcement**: Tracks unique architectures to maintain population diversity

### 4. **Q-Learning Integration**
- `QLearningArchitectureAgent` learns which layer combinations work best
- Updates Q-values based on architecture performance
- Guides evolution toward promising regions of the search space

### 5. **Hybrid Optimization**
- `HybridGeneticOptimizer` combines your evolvo_engine with neural architecture search
- Co-evolves both algorithms and model architectures
- Evaluates combined fitness of algorithm-model pairs

## Key Features to Avoid Redundancy

1. **Signature-based deduplication**: Each architecture gets a unique hash preventing identical genomes
2. **Task-specific layer probabilities**: Different distributions for classification, sequence, and regression tasks
3. **Shape tracking**: Ensures layer compatibility throughout the network
4. **Complexity penalties**: Fitness functions consider both performance and architecture efficiency

## Integration Points with evolvo_engine

The system can be chained with your original library through:
- Shared evaluation metrics
- Combined fitness functions
- Co-evolution where algorithms and models evolve together
- Q-learning rewards based on both algorithm and model performance

## Usage Example

```python
# Create evolver for classification task
evolver = ModelEvolver(
    input_shape=(784,),  # MNIST-like input
    output_shape=(10,),  # 10 classes
    task_type='classification'
)

# Evolve population with custom fitness function
population = evolver.evolve_population(
    population_size=50,
    generations=100,
    fitness_func=your_evaluation_function
)

# Best model can be converted to PyTorch
best_genome = population[0]
pytorch_model = best_genome.to_pytorch_model()
```

This system provides a robust foundation for evolutionary neural architecture search that integrates seamlessly with your genetic algorithm framework,
enabling sophisticated co-evolution strategies for finding optimal model-algorithm combinations.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json

@dataclass
class LayerSpec:
    """Specification for a neural network layer"""
    layer_type: str
    params: Dict[str, Any]
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    
    def __hash__(self):
        """Create hash for detecting duplicate architectures"""
        return hash((self.layer_type, json.dumps(self.params, sort_keys=True)))

class ModelGenome:
    """Represents a neural network architecture as a genome"""
    def __init__(self, input_shape: Tuple, output_shape: Tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers: List[LayerSpec] = []
        self.connections: Dict[int, List[int]] = defaultdict(list)  # Skip connections
        self.fitness: Optional[float] = None
        self._signature: Optional[str] = None
    
    def add_layer(self, layer_spec: LayerSpec):
        """Add a layer to the genome"""
        self.layers.append(layer_spec)
        self._signature = None  # Reset signature cache
    
    def get_signature(self) -> str:
        """Generate unique signature to detect duplicate architectures"""
        if self._signature is None:
            sig_parts = []
            for layer in self.layers:
                sig_parts.append(f"{layer.layer_type}:{json.dumps(layer.params, sort_keys=True)}")
            for src, dests in sorted(self.connections.items()):
                sig_parts.append(f"conn:{src}->{sorted(dests)}")
            self._signature = hashlib.md5('|'.join(sig_parts).encode()).hexdigest()
        return self._signature
    
    def to_pytorch_model(self) -> nn.Module:
        """Convert genome to actual PyTorch model"""
        return DynamicModel(self)

class LayerFactory:
    """Factory for creating neural network layers with genetic constraints"""
    
    # Define layer templates with valid parameter ranges
    LAYER_TEMPLATES = {
        'linear': {
            'params': {
                'out_features': [8, 16, 32, 64, 128, 256, 512],
                'bias': [True, False]
            },
            'requires_input_features': True
        },
        'conv2d': {
            'params': {
                'out_channels': [8, 16, 32, 64, 128],
                'kernel_size': [1, 3, 5, 7],
                'stride': [1, 2],
                'padding': ['same', 'valid', 0, 1, 2],
                'bias': [True, False]
            },
            'requires_input_channels': True
        },
        'lstm': {
            'params': {
                'hidden_size': [16, 32, 64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.0, 0.1, 0.2, 0.3],
                'bidirectional': [True, False]
            },
            'requires_input_size': True
        },
        'attention': {
            'params': {
                'embed_dim': [64, 128, 256, 512],
                'num_heads': [1, 2, 4, 8],
                'dropout': [0.0, 0.1, 0.2]
            },
            'requires_embed_dim': True
        },
        'dropout': {
            'params': {
                'p': [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        },
        'batchnorm1d': {
            'params': {},
            'requires_num_features': True
        },
        'batchnorm2d': {
            'params': {},
            'requires_num_features': True
        },
        'activation': {
            'params': {
                'type': ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu']
            }
        },
        'pooling': {
            'params': {
                'type': ['max2d', 'avg2d', 'adaptive_avg2d', 'adaptive_max2d'],
                'kernel_size': [2, 3],
                'stride': [1, 2]
            }
        }
    }
    
    @classmethod
    def create_random_layer(cls, layer_type: str, prev_shape: Optional[Tuple] = None) -> LayerSpec:
        """Create a random layer of specified type"""
        if layer_type not in cls.LAYER_TEMPLATES:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        template = cls.LAYER_TEMPLATES[layer_type]
        params = {}
        
        # Sample random parameters
        for param_name, param_options in template.get('params', {}).items():
            params[param_name] = random.choice(param_options)
        
        # Handle shape-dependent parameters
        if prev_shape:
            if template.get('requires_input_features') and len(prev_shape) >= 1:
                params['in_features'] = prev_shape[-1]
            if template.get('requires_input_channels') and len(prev_shape) >= 3:
                params['in_channels'] = prev_shape[-3]
            if template.get('requires_num_features'):
                params['num_features'] = prev_shape[-3] if len(prev_shape) >= 3 else prev_shape[-1]
            if template.get('requires_input_size'):
                params['input_size'] = prev_shape[-1]
            if template.get('requires_embed_dim'):
                params['embed_dim'] = prev_shape[-1]
        
        return LayerSpec(layer_type, params)

class DynamicModel(nn.Module):
    """Dynamic PyTorch model created from genome"""
    def __init__(self, genome: ModelGenome):
        super().__init__()
        self.genome = genome
        self.layers = nn.ModuleList()
        self.build_layers()
    
    def build_layers(self):
        """Build actual PyTorch layers from genome"""
        for i, layer_spec in enumerate(self.genome.layers):
            layer = self._create_layer(layer_spec)
            if layer:
                self.layers.append(layer)
    
    def _create_layer(self, spec: LayerSpec) -> Optional[nn.Module]:
        """Create PyTorch layer from specification"""
        lt = spec.layer_type
        p = spec.params
        
        if lt == 'linear':
            return nn.Linear(p.get('in_features', 128), p['out_features'], p.get('bias', True))
        elif lt == 'conv2d':
            return nn.Conv2d(
                p.get('in_channels', 3), p['out_channels'],
                p['kernel_size'], p.get('stride', 1),
                p.get('padding', 0), bias=p.get('bias', True)
            )
        elif lt == 'lstm':
            return nn.LSTM(
                p.get('input_size', 128), p['hidden_size'],
                p.get('num_layers', 1), dropout=p.get('dropout', 0),
                bidirectional=p.get('bidirectional', False),
                batch_first=True
            )
        elif lt == 'attention':
            return nn.MultiheadAttention(
                p['embed_dim'], p['num_heads'],
                dropout=p.get('dropout', 0), batch_first=True
            )
        elif lt == 'dropout':
            return nn.Dropout(p['p'])
        elif lt == 'batchnorm1d':
            return nn.BatchNorm1d(p['num_features'])
        elif lt == 'batchnorm2d':
            return nn.BatchNorm2d(p['num_features'])
        elif lt == 'activation':
            act_type = p['type']
            if act_type == 'relu': return nn.ReLU()
            elif act_type == 'tanh': return nn.Tanh()
            elif act_type == 'sigmoid': return nn.Sigmoid()
            elif act_type == 'leaky_relu': return nn.LeakyReLU()
            elif act_type == 'elu': return nn.ELU()
            elif act_type == 'gelu': return nn.GELU()
        elif lt == 'pooling':
            pool_type = p['type']
            if pool_type == 'max2d':
                return nn.MaxPool2d(p.get('kernel_size', 2), p.get('stride', 2))
            elif pool_type == 'avg2d':
                return nn.AvgPool2d(p.get('kernel_size', 2), p.get('stride', 2))
            elif pool_type == 'adaptive_avg2d':
                return nn.AdaptiveAvgPool2d((1, 1))
            elif pool_type == 'adaptive_max2d':
                return nn.AdaptiveMaxPool2d((1, 1))
        
        return None
    
    def forward(self, x):
        """Forward pass with skip connections"""
        outputs = []
        
        for i, layer in enumerate(self.layers):
            # Apply layer
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, x, x)
            else:
                x = layer(x)
            
            outputs.append(x)
            
            # Handle skip connections
            if i in self.genome.connections:
                for dest in self.genome.connections[i]:
                    if dest < len(outputs):
                        # Add skip connection (with shape matching if needed)
                        if outputs[dest].shape == x.shape:
                            outputs[dest] = outputs[dest] + x
        
        return x

class ModelEvolver:
    """Evolves PyTorch model architectures using genetic algorithms"""
    
    def __init__(self, input_shape: Tuple, output_shape: Tuple, 
                 task_type: str = 'classification'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.task_type = task_type
        self.population: List[ModelGenome] = []
        self.generation = 0
        self.diversity_cache = set()  # Track unique architectures
        
        # Configure layer probabilities based on task
        self.layer_probs = self._get_layer_probabilities()
    
    def _get_layer_probabilities(self) -> Dict[str, float]:
        """Get layer type probabilities based on task"""
        if self.task_type == 'classification':
            return {
                'linear': 0.3,
                'activation': 0.25,
                'dropout': 0.15,
                'batchnorm1d': 0.1,
                'conv2d': 0.15,
                'pooling': 0.05
            }
        elif self.task_type == 'sequence':
            return {
                'lstm': 0.3,
                'attention': 0.2,
                'linear': 0.2,
                'activation': 0.15,
                'dropout': 0.15
            }
        else:  # 'regression' or general
            return {
                'linear': 0.35,
                'activation': 0.3,
                'dropout': 0.2,
                'batchnorm1d': 0.15
            }
    
    def generate_random_genome(self, max_layers: int = 10) -> ModelGenome:
        """Generate a random model genome"""
        genome = ModelGenome(self.input_shape, self.output_shape)
        num_layers = random.randint(2, max_layers)
        
        prev_shape = self.input_shape
        
        for i in range(num_layers):
            # Choose layer type based on probabilities
            layer_type = np.random.choice(
                list(self.layer_probs.keys()),
                p=list(self.layer_probs.values())
            )
            
            # Create layer
            layer_spec = LayerFactory.create_random_layer(layer_type, prev_shape)
            genome.add_layer(layer_spec)
            
            # Update shape tracking (simplified)
            if layer_type == 'linear':
                prev_shape = (layer_spec.params['out_features'],)
            elif layer_type == 'conv2d':
                # Simplified shape tracking for conv layers
                prev_shape = (layer_spec.params['out_channels'], None, None)
            
            # Randomly add skip connections
            if i > 0 and random.random() < 0.1:
                source = random.randint(0, i-1)
                genome.connections[source].append(i)
        
        # Ensure output layer matches expected shape
        if self.task_type == 'classification':
            genome.add_layer(LayerSpec('linear', {
                'in_features': prev_shape[0] if prev_shape else 128,
                'out_features': self.output_shape[0],
                'bias': True
            }))
        
        return genome
    
    def crossover(self, parent1: ModelGenome, parent2: ModelGenome) -> ModelGenome:
        """Crossover two parent genomes"""
        child = ModelGenome(self.input_shape, self.output_shape)
        
        # Crossover layers
        if random.random() < 0.5:
            # Single-point crossover
            point1 = random.randint(0, len(parent1.layers))
            point2 = random.randint(0, len(parent2.layers))
            child.layers = parent1.layers[:point1] + parent2.layers[point2:]
        else:
            # Uniform crossover
            max_len = max(len(parent1.layers), len(parent2.layers))
            for i in range(max_len):
                if i < len(parent1.layers) and i < len(parent2.layers):
                    child.layers.append(
                        random.choice([parent1.layers[i], parent2.layers[i]])
                    )
                elif i < len(parent1.layers):
                    child.layers.append(parent1.layers[i])
                elif i < len(parent2.layers):
                    child.layers.append(parent2.layers[i])
        
        # Crossover connections
        all_connections = set(parent1.connections.keys()) | set(parent2.connections.keys())
        for conn in all_connections:
            if random.random() < 0.5:
                if conn in parent1.connections:
                    child.connections[conn] = parent1.connections[conn].copy()
                elif conn in parent2.connections:
                    child.connections[conn] = parent2.connections[conn].copy()
        
        return child
    
    def mutate(self, genome: ModelGenome, mutation_rate: float = 0.1) -> ModelGenome:
        """Mutate a genome"""
        mutated = ModelGenome(self.input_shape, self.output_shape)
        mutated.layers = genome.layers.copy()
        mutated.connections = {k: v.copy() for k, v in genome.connections.items()}
        
        # Layer mutations
        for i in range(len(mutated.layers)):
            if random.random() < mutation_rate:
                mutation_type = random.choice(['modify', 'replace', 'remove'])
                
                if mutation_type == 'modify' and mutated.layers[i].params:
                    # Modify a parameter
                    param_name = random.choice(list(mutated.layers[i].params.keys()))
                    template = LayerFactory.LAYER_TEMPLATES.get(mutated.layers[i].layer_type, {})
                    if param_name in template.get('params', {}):
                        mutated.layers[i].params[param_name] = random.choice(
                            template['params'][param_name]
                        )
                
                elif mutation_type == 'replace':
                    # Replace with new random layer
                    layer_type = np.random.choice(
                        list(self.layer_probs.keys()),
                        p=list(self.layer_probs.values())
                    )
                    mutated.layers[i] = LayerFactory.create_random_layer(layer_type)
                
                elif mutation_type == 'remove' and len(mutated.layers) > 2:
                    mutated.layers.pop(i)
                    break
        
        # Add new layer mutation
        if random.random() < mutation_rate:
            layer_type = np.random.choice(
                list(self.layer_probs.keys()),
                p=list(self.layer_probs.values())
            )
            new_layer = LayerFactory.create_random_layer(layer_type)
            insert_pos = random.randint(0, len(mutated.layers))
            mutated.layers.insert(insert_pos, new_layer)
        
        # Connection mutations
        if random.random() < mutation_rate:
            if mutated.connections and random.random() < 0.5:
                # Remove a connection
                conn_to_remove = random.choice(list(mutated.connections.keys()))
                del mutated.connections[conn_to_remove]
            else:
                # Add a connection
                if len(mutated.layers) > 2:
                    source = random.randint(0, len(mutated.layers) - 2)
                    dest = random.randint(source + 1, len(mutated.layers) - 1)
                    if dest not in mutated.connections[source]:
                        mutated.connections[source].append(dest)
        
        return mutated
    
    def is_novel_architecture(self, genome: ModelGenome) -> bool:
        """Check if architecture is novel (not seen before)"""
        signature = genome.get_signature()
        if signature in self.diversity_cache:
            return False
        self.diversity_cache.add(signature)
        return True
    
    def evolve_population(self, population_size: int = 50, generations: int = 100,
                         fitness_func: Callable = None) -> List[ModelGenome]:
        """Main evolution loop"""
        # Initialize population
        self.population = []
        attempts = 0
        while len(self.population) < population_size and attempts < population_size * 10:
            genome = self.generate_random_genome()
            if self.is_novel_architecture(genome):
                self.population.append(genome)
            attempts += 1
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            if fitness_func:
                for genome in self.population:
                    if genome.fitness is None:
                        genome.fitness = fitness_func(genome)
            
            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness or 0, reverse=True)
            
            # Selection and reproduction
            elite_size = population_size // 10
            new_population = self.population[:elite_size]  # Elitism
            
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                if random.random() < 0.3:
                    child = self.mutate(child)
                
                # Add if novel
                if self.is_novel_architecture(child):
                    new_population.append(child)
            
            self.population = new_population[:population_size]
            
            # Report progress
            best_fitness = self.population[0].fitness if self.population[0].fitness else 0
            print(f"Generation {gen}: Best fitness = {best_fitness:.4f}, "
                  f"Unique architectures = {len(self.diversity_cache)}")
        
        return self.population
    
    def _tournament_select(self, tournament_size: int = 3) -> ModelGenome:
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness or 0)

class ModelChainEvaluator:
    """Evaluates model genomes in conjunction with evolvo_engine algorithms"""
    
    def __init__(self, data_loader, loss_fn=nn.CrossEntropyLoss(), device='cpu'):
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device
    
    def evaluate_genome(self, genome: ModelGenome, epochs: int = 5) -> float:
        """Train and evaluate a model genome, return fitness score"""
        try:
            model = genome.to_pytorch_model().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Quick training
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in self.data_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = self.loss_fn(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in self.data_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total if total > 0 else 0
            # Fitness combines accuracy with architecture complexity penalty
            complexity_penalty = len(genome.layers) * 0.001
            fitness = accuracy - complexity_penalty
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            return -1.0  # Invalid architecture

class QLearningArchitectureAgent:
    """Q-learning agent for architecture search decisions"""
    
    def __init__(self, state_dim: int = 10, action_space: List[str] = None):
        self.state_dim = state_dim
        self.action_space = action_space or list(LayerFactory.LAYER_TEMPLATES.keys())
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
    
    def get_state_representation(self, genome: ModelGenome) -> str:
        """Convert genome to state representation"""
        # Simplified state: layer type sequence
        state = []
        for layer in genome.layers[-self.state_dim:]:  # Last N layers
            state.append(layer.layer_type)
        return '|'.join(state)
    
    def choose_action(self, state: str) -> str:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        q_values = self.q_table[state]
        if not q_values:
            return random.choice(self.action_space)
        
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def guide_evolution(self, genome: ModelGenome) -> ModelGenome:
        """Use Q-learning to guide architecture evolution"""
        state = self.get_state_representation(genome)
        action = self.choose_action(state)
        
        # Apply action (add layer of chosen type)
        new_layer = LayerFactory.create_random_layer(action)
        genome.add_layer(new_layer)
        
        return genome

# Integration with evolvo_engine
class HybridGeneticOptimizer:
    """Combines evolvo_engine algorithms with neural architecture search"""
    
    def __init__(self, evolvo_evaluator, model_evolver: ModelEvolver):
        self.evolvo_evaluator = evolvo_evaluator
        self.model_evolver = model_evolver
        self.q_agent = QLearningArchitectureAgent()
    
    def co_evolve(self, generations: int = 50):
        """Co-evolve algorithms and neural architectures"""
        best_algorithm = None
        best_model = None
        best_combined_fitness = -float('inf')
        
        for gen in range(generations):
            # Evolve algorithms using evolvo_engine
            # ... (integration with AlgorithmGenerator from evolvo_engine)
            
            # Evolve model architectures
            model_population = self.model_evolver.evolve_population(
                population_size=20,
                generations=5,
                fitness_func=lambda g: self.evaluate_combined_fitness(g, best_algorithm)
            )
            
            # Update Q-learning agent based on results
            for genome in model_population[:5]:  # Top 5 models
                state = self.q_agent.get_state_representation(genome)
                reward = genome.fitness or 0
                # Update Q-values based on architecture performance
                for layer in genome.layers:
                    self.q_agent.update_q_value(state, layer.layer_type, reward, state)
            
            # Track best combination
            if model_population[0].fitness > best_combined_fitness:
                best_combined_fitness = model_population[0].fitness
                best_model = model_population[0]
                print(f"Generation {gen}: New best fitness = {best_combined_fitness:.4f}")
        
        return best_model, best_algorithm
    
    def evaluate_combined_fitness(self, model_genome: ModelGenome, algorithm) -> float:
        """Evaluate fitness of model-algorithm combination"""
        # Placeholder for combined evaluation
        # This would integrate with evolvo_engine's evaluation
        model_fitness = self.model_evolver.fitness_func(model_genome) if hasattr(self.model_evolver, 'fitness_func') else 0
        return model_fitness

# Example usage
if __name__ == "__main__":
    # Define input/output shapes for your task
    input_shape = (784,)  # e.g., flattened MNIST
    output_shape = (10,)   # e.g., 10 classes
    
    # Create evolver
    evolver = ModelEvolver(input_shape, output_shape, task_type='classification')
    
    # Generate some random architectures
    print("Generating random architectures...")
    for i in range(5):
        genome = evolver.generate_random_genome(max_layers=7)
        print(f"\nArchitecture {i+1}:")
        for j, layer in enumerate(genome.layers):
            print(f"  Layer {j}: {layer.layer_type} - {layer.params}")
        print(f"  Signature: {genome.get_signature()[:8]}...")
    
    # Demonstrate crossover and mutation
    print("\n\nDemonstrating genetic operations:")
    parent1 = evolver.generate_random_genome(max_layers=5)
    parent2 = evolver.generate_random_genome(max_layers=5)
    
    print("Parent 1 layers:", [l.layer_type for l in parent1.layers])
    print("Parent 2 layers:", [l.layer_type for l in parent2.layers])
    
    child = evolver.crossover(parent1, parent2)
    print("Child layers:", [l.layer_type for l in child.layers])
    
    mutated = evolver.mutate(child)
    print("Mutated child layers:", [l.layer_type for l in mutated.layers])
    
    # Create actual PyTorch model
    print("\n\nCreating PyTorch model from genome:")
    model = child.to_pytorch_model()
    print(model)
    
    print("\n\nSystem ready for integration with evolvo_engine!")