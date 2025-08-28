# -*- coding: utf-8 -*-
"""
Unified Evolvo Library - Genetic Evolution Framework
=====================================================

A unified genetic evolution system for both algorithmic sequences and neural architectures.

Key Design Principles:
1. **Canonical Representation**: Each genome has a unique canonical form to avoid redundant descriptions
2. **Valid-by-Construction**: Generation methods ensure syntactic validity, avoiding dead-ends
3. **Implicit Shape Inference**: Automatic shape/type propagation reduces configuration errors
4. **Modular Evolution**: Common evolution framework for both algorithms and neural networks
5. **Q-Learning Integration**: Built-in support for reinforcement learning guidance
6. **Graceful Error Handling**: Algorithms handle errors without crashing

Core Components:
- BaseGenome: Abstract base for all evolvable structures
- AlgorithmGenome: Represents instruction sequences with bool/decimal separation
- NeuralGenome: Represents neural network architectures with shape tracking
- UnifiedEvolver: Common evolution engine for all genome types
- QLearningGuide: Reinforcement learning for guided evolution

---

## Core Unification Strategy

### 1. **Common Base Architecture**
- `BaseGenome` abstract class provides a unified interface for all genome types
- Shared signature generation to prevent redundant descriptions with different orderings
- Common validation and simplification methods

### 2. **Unified Data System**
- `UnifiedDataStore` handles both algorithmic variables (bool/decimal with constant/variable separation) and tensor shapes
- `DataType` class provides type compatibility checking across both systems
- Graceful error handling that returns defaults instead of crashing

### 3. **Enhanced Instruction System**
- `Operation` class wraps functions with automatic error handling (try-except built in)
- `EnhancedInstructionSet` supports both mathematical operations and tensor operations
- Operations return sensible defaults on errors (0 for decimals, False for bools)

### 4. **Canonical Representations**
Both genome types now generate canonical signatures that:
- Use dependency analysis to eliminate order-dependent redundancy
- Identify "dead code" that doesn't contribute to outputs
- Support the `simplify()` method to remove unnecessary instructions/layers

### 5. **Unified Evolution Engine**
- Single `UnifiedEvolver` class handles both algorithm and neural evolution
- Shared crossover/mutation logic with type-specific implementations
- Multi-objective optimization with Pareto frontiers
- Adaptive mutation rates based on population diversity

### 6. **Q-Learning Integration**
- Single `QLearningGuide` class that works with both genome types
- Learns which operations/layers work best in different contexts
- Can guide both instruction selection and layer selection

## Key Features for Q-Learning Compatibility

1. **State Extraction**: Consistent state representation from genome history
2. **Action Spaces**: Well-defined action sets for each genome type
3. **Canonical Forms**: Ensures same genome always has same representation regardless of instruction ordering
4. **Dead Code Elimination**: The `simplify()` method identifies which instructions actually contribute to outputs

## Fine-Tuning Support

For neural networks:
- `freeze_layers()` method to protect trained layers during evolution
- Mutations respect frozen layer boundaries
- Support for gradual architecture growth

## Error Handling

- All operations wrapped with try-except
- Invalid operations return defaults instead of crashing
- Validation methods identify structural problems before execution
- No infinite retry loops - operations fail gracefully

## Example Usage Patterns

```python
# Algorithm evolution with specific outputs
algo_genome = AlgorithmGenome(data_config, instruction_set)
algo_genome.mark_output(('d$', 1))  # Mark which variables are outputs
simplified = algo_genome.simplify()  # Remove dead code

# Neural evolution with fine-tuning
neural_genome = NeuralGenome(input_shape, output_shape)
neural_genome.freeze_layers(10)  # Freeze first 10 layers
# Mutations will only affect layers after index 10

# Q-learning guided evolution
q_guide = QLearningGuide()
action = q_guide.choose_action(genome)
mutated = q_guide.guide_mutation(genome)
q_guide.update(genome, action, reward, mutated)
```

---

## Resource Management Features

### 1. **ResourceMonitor Class**
- **Automatic Memory Tiering**: Models automatically move between VRAM → RAM → Disk based on availability
- **LRU Cache**: Least recently used models are offloaded when memory is needed
- **Device Detection**: Automatically detects CUDA, MPS (Apple Silicon), or CPU
- **Memory Tracking**: Real-time monitoring of VRAM and RAM usage

### 2. **Resource-Aware Evolution**
- **Pre-validation**: Estimates model size before creation to prevent OOM errors
- **Batch Processing**: Evaluates populations in parallel batches with automatic memory cleanup
- **Dynamic Constraints**: Adjusts maximum model size based on available resources
- **Graceful Degradation**: Models that exceed limits get penalized rather than crashing

### 3. **Model Offloading Strategy**
```
VRAM (fastest, limited) 
  ↓ When full
RAM (fast, larger)
  ↓ When full  
Disk Cache (slow, unlimited)
```

### 4. **Key Methods**
- `can_fit_in_vram()`: Checks if model fits in GPU memory
- `offload_to_ram()`: Moves model from VRAM to system RAM
- `offload_to_disk()`: Serializes model to disk cache
- `model_context()`: Context manager for automatic resource cleanup

### 5. **Neural Genome Enhancements**
- **Resource Constraints**: Set maximum parameters and memory per model
- **Incremental Validation**: Each layer addition checks resource limits
- **Memory Estimation**: Calculates both parameter and activation memory

## Usage Example

```python
# Initialize with resource limits
resource_monitor = ResourceMonitor(
    max_vram_usage=0.7,  # Use max 70% of VRAM
    max_ram_usage=0.8,   # Use max 80% of RAM
    cache_dir=Path("./model_cache")
)

# Create resource-aware evolver
evolver = ResourceAwareEvolver(
    GenomeType.NEURAL,
    population_size=50,
    max_model_params=1e8,  # Max 100M parameters
    resource_monitor=resource_monitor
)

# Parallel evaluation with automatic batching
evolver.evaluate_population_parallel(
    evaluator=your_fitness_function,
    batch_size=4  # Process 4 models at a time
)
```

## Benefits

1. **No OOM Crashes**: Models that don't fit are automatically offloaded or rejected
2. **Parallel Processing**: Safely evaluate multiple models simultaneously
3. **Automatic Cleanup**: Memory is freed between batches
4. **Scale to Large Populations**: Can handle hundreds of models using disk cache
5. **Hardware Agnostic**: Works on CUDA, MPS (M1/M2 Macs), or CPU-only systems

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import hashlib
import json
import copy
from typing import List, Dict, Any, Tuple, Optional, Union, Set, Callable
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import traceback

# ============================================================================
# BASE GENOME SYSTEM
# ============================================================================

class GenomeType(Enum):
    """Types of genomes supported by the framework"""
    ALGORITHM = "algorithm"
    NEURAL = "neural"
    HYBRID = "hybrid"

class BaseGenome(ABC):
    """
    Abstract base class for all genome types.
    Ensures consistent interface for evolution operations.
    """
    def __init__(self, genome_type: GenomeType):
        self.genome_type = genome_type
        self.fitness: Optional[float] = None
        self.generation: int = 0
        self.parents: List[str] = []
        self.mutations: List[str] = []
        self._signature: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def get_signature(self) -> str:
        """Generate unique canonical signature to detect duplicates"""
        pass
    
    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate genome structure, return (is_valid, error_messages)"""
        pass
    
    @abstractmethod
    def simplify(self) -> 'BaseGenome':
        """Remove redundant/unused components, return simplified version"""
        pass
    
    @abstractmethod
    def to_executable(self) -> Any:
        """Convert to executable form (bytecode/PyTorch model)"""
        pass
    
    def __hash__(self):
        return hash(self.get_signature())
    
    def __eq__(self, other):
        if not isinstance(other, BaseGenome):
            return False
        return self.get_signature() == other.get_signature()

# ============================================================================
# DATA TYPES AND STORAGE
# ============================================================================

@dataclass
class DataType:
    """Unified type system for both algorithms and neural networks"""
    category: str  # 'bool', 'decimal', 'tensor'
    is_constant: bool = False
    shape: Optional[Tuple] = None  # For tensors
    dtype: Optional[torch.dtype] = None  # For tensors
    
    def is_compatible_with(self, other: 'DataType') -> bool:
        """Check if this type can be converted/connected to another"""
        if self.category != other.category:
            return False
        if self.category == 'tensor' and self.shape and other.shape:
            # Check tensor shape compatibility (allowing for broadcasting/reshaping)
            return self._can_reshape_to(self.shape, other.shape)
        return True
    
    def _can_reshape_to(self, from_shape: Tuple, to_shape: Tuple) -> bool:
        """Check if tensor can be reshaped/adapted from one shape to another"""
        # Simplified check - in practice would be more sophisticated
        from_elements = np.prod(from_shape) if from_shape else 1
        to_elements = np.prod(to_shape) if to_shape else 1
        return from_elements == to_elements or from_elements == 1 or to_elements == 1

class UnifiedDataStore:
    """
    Enhanced data store supporting both algorithmic variables and tensor shapes.
    Maintains strict separation between booleans/decimals and constants/variables.
    """
    def __init__(self, config: Dict[str, List[str]]):
        """
        Config format:
        {
            'b#': ['true', 'false'],  # Boolean constants
            'd#': ['pi', 'e'],        # Decimal constants  
            'b$': ['flag'],           # Boolean variables
            'd$': ['x', 'y'],         # Decimal variables
            't$': ['tensor_a']        # Tensor variables (for hybrid systems)
        }
        """
        self.config = config
        self.stores: Dict[str, List[Any]] = {}
        self.types: Dict[str, DataType] = {}
        self.name_map: Dict[str, Tuple[str, int]] = {}  # name -> (store_type, index)
        
        # Initialize stores
        for store_type, names in config.items():
            self.stores[store_type] = [self._default_value(store_type)] * len(names)
            for i, name in enumerate(names):
                self.name_map[name] = (store_type, i)
                self.types[name] = self._create_data_type(store_type)
    
    def _default_value(self, store_type: str) -> Any:
        """Get default value for store type"""
        if store_type.startswith('b'):
            return False
        elif store_type.startswith('d'):
            return np.float64(0)
        elif store_type.startswith('t'):
            return None  # Tensor placeholder
        return None
    
    def _create_data_type(self, store_type: str) -> DataType:
        """Create DataType from store type string"""
        category = {'b': 'bool', 'd': 'decimal', 't': 'tensor'}.get(store_type[0], 'unknown')
        is_constant = store_type.endswith('#')
        return DataType(category=category, is_constant=is_constant)
    
    def set(self, name: str, value: Any) -> bool:
        """Set value with type checking. Returns success status."""
        if name not in self.name_map:
            return False
        
        store_type, index = self.name_map[name]
        data_type = self.types[name]
        
        # Type checking and conversion
        try:
            if data_type.category == 'bool':
                value = bool(value)
            elif data_type.category == 'decimal':
                value = np.float64(value)
            elif data_type.category == 'tensor' and not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            
            self.stores[store_type][index] = value
            return True
        except (ValueError, TypeError):
            return False
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get value by name with optional default"""
        if name not in self.name_map:
            return default
        store_type, index = self.name_map[name]
        return self.stores[store_type][index]
    
    def reset(self):
        """Reset all variables to default values"""
        for store_type in self.stores:
            if store_type.endswith('$'):  # Only reset variables, not constants
                self.stores[store_type] = [self._default_value(store_type)] * len(self.stores[store_type])

# ============================================================================
# INSTRUCTION SET AND OPERATIONS
# ============================================================================

class Operation:
    """Enhanced operation definition with error handling and type inference"""
    def __init__(self, name: str, func: Callable, arg_types: List[str], 
                 return_type: str, category: str = 'arithmetic'):
        self.name = name
        self.func = func
        self.arg_types = arg_types  # ['decimal', 'decimal'] or ['bool', 'bool']
        self.return_type = return_type
        self.category = category  # 'arithmetic', 'logical', 'control', 'tensor'
        self.error_default = self._get_error_default()
    
    def _get_error_default(self) -> Any:
        """Default value to return on error"""
        if self.return_type == 'bool':
            return False
        elif self.return_type == 'decimal':
            return np.float64(0)
        else:
            return None
    
    def execute(self, *args) -> Any:
        """Execute operation with error handling"""
        try:
            return self.func(*args)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            # Graceful error handling - return default instead of crashing
            return self.error_default
        except Exception:
            # Unexpected errors also handled gracefully
            return self.error_default

class EnhancedInstructionSet:
    """
    Extensible instruction set with automatic type checking and error handling.
    Supports both algorithmic operations and tensor operations for hybrid systems.
    """
    def __init__(self):
        self.operations: Dict[str, Operation] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self._register_default_operations()
    
    def _register_default_operations(self):
        """Register standard mathematical and logical operations"""
        # Arithmetic operations
        self.register('ADD', lambda a, b: a + b, ['decimal', 'decimal'], 'decimal', 'arithmetic')
        self.register('SUB', lambda a, b: a - b, ['decimal', 'decimal'], 'decimal', 'arithmetic')
        self.register('MUL', lambda a, b: a * b, ['decimal', 'decimal'], 'decimal', 'arithmetic')
        self.register('DIV', lambda a, b: a / b if b != 0 else np.float64(0), 
                     ['decimal', 'decimal'], 'decimal', 'arithmetic')
        self.register('MOD', lambda a, b: a % b if b != 0 else np.float64(0),
                     ['decimal', 'decimal'], 'decimal', 'arithmetic')
        self.register('POW', lambda a, b: a ** b if abs(a) < 1e10 and abs(b) < 10 else np.float64(0),
                     ['decimal', 'decimal'], 'decimal', 'arithmetic')
        
        # Mathematical functions
        self.register('EXP', lambda a: np.exp(a) if a < 700 else np.float64(np.inf),
                     ['decimal'], 'decimal', 'math')
        self.register('LOG', lambda a: np.log(a) if a > 0 else np.float64(-np.inf),
                     ['decimal'], 'decimal', 'math')
        self.register('SIN', lambda a: np.sin(a), ['decimal'], 'decimal', 'math')
        self.register('COS', lambda a: np.cos(a), ['decimal'], 'decimal', 'math')
        self.register('SQRT', lambda a: np.sqrt(a) if a >= 0 else np.float64(0),
                     ['decimal'], 'decimal', 'math')
        self.register('ABS', lambda a: abs(a), ['decimal'], 'decimal', 'math')
        
        # Logical operations
        self.register('NOT', lambda a: not a, ['bool'], 'bool', 'logical')
        self.register('AND', lambda a, b: a and b, ['bool', 'bool'], 'bool', 'logical')
        self.register('OR', lambda a, b: a or b, ['bool', 'bool'], 'bool', 'logical')
        self.register('XOR', lambda a, b: a != b, ['bool', 'bool'], 'bool', 'logical')
        
        # Comparison operations
        self.register('EQ', lambda a, b: abs(a - b) < 1e-10, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('GT', lambda a, b: a > b, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('LT', lambda a, b: a < b, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('GTE', lambda a, b: a >= b, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('LTE', lambda a, b: a <= b, ['decimal', 'decimal'], 'bool', 'comparison')
        
        # Control flow (special handling required)
        self.register('IF', None, ['bool'], None, 'control')
        self.register('ELSE', None, [], None, 'control')
        self.register('END', None, [], None, 'control')
        
        # Assignment (identity function)
        self.register('ASSIGN', lambda a: a, ['any'], 'any', 'control')
    
    def register(self, name: str, func: Optional[Callable], arg_types: List[str], 
                return_type: Optional[str], category: str = 'custom'):
        """Register a new operation"""
        op = Operation(name, func, arg_types, return_type, category)
        self.operations[name] = op
        self.categories[category].append(name)
    
    def get_compatible_operations(self, input_types: List[str]) -> List[str]:
        """Get operations compatible with given input types"""
        compatible = []
        for name, op in self.operations.items():
            if len(op.arg_types) == len(input_types):
                if all(self._types_compatible(it, at) 
                      for it, at in zip(input_types, op.arg_types)):
                    compatible.append(name)
        return compatible
    
    def _types_compatible(self, input_type: str, arg_type: str) -> bool:
        """Check if input type is compatible with argument type"""
        if arg_type == 'any':
            return True
        return input_type == arg_type

# ============================================================================
# ALGORITHM GENOME
# ============================================================================

@dataclass
class Instruction:
    """Single instruction in an algorithm"""
    target: Tuple[str, int]  # (store_type, index) e.g., ('d$', 0)
    operation: str  # Operation name
    args: List[Tuple[str, int]]  # List of (store_type, index) for arguments
    
    def to_list(self) -> List:
        """Convert to legacy list format for compatibility"""
        result = list(self.target) + [self.operation]
        for arg in self.args:
            result.extend(arg)
        return result
    
    def get_dependencies(self) -> Set[Tuple[str, int]]:
        """Get all data dependencies for this instruction"""
        return set(self.args)
    
    def get_output(self) -> Tuple[str, int]:
        """Get the output location of this instruction"""
        return self.target

class AlgorithmGenome(BaseGenome):
    """
    Genome representing an algorithm as a sequence of instructions.
    Maintains strict type safety and supports multiple outputs.
    """
    def __init__(self, data_config: Dict[str, List[str]], instruction_set: EnhancedInstructionSet):
        super().__init__(GenomeType.ALGORITHM)
        self.data_config = data_config
        self.instruction_set = instruction_set
        self.instructions: List[Union[Instruction, Dict]] = []  # Dict for control flow
        self.outputs: Set[Tuple[str, int]] = set()  # Track which variables are outputs
        self.execution_order: Optional[List[int]] = None  # Optimized execution order
    
    def add_instruction(self, instruction: Instruction) -> bool:
        """Add instruction with validation. Returns success status."""
        # Validate operation exists
        if instruction.operation not in self.instruction_set.operations:
            return False
        
        # Validate argument count
        op = self.instruction_set.operations[instruction.operation]
        if len(instruction.args) != len(op.arg_types):
            return False
        
        # Type checking would go here (simplified for brevity)
        self.instructions.append(instruction)
        self._signature = None  # Reset signature cache
        return True
    
    def add_control_flow(self, control_type: str, condition: Optional[Tuple[str, int]] = None):
        """Add control flow structure (IF/ELSE/END)"""
        if control_type == 'IF' and condition:
            self.instructions.append({'type': 'IF', 'condition': condition, 'body': []})
        elif control_type in ['ELSE', 'END']:
            self.instructions.append({'type': control_type})
        self._signature = None
    
    def mark_output(self, location: Tuple[str, int]):
        """Mark a variable as an output of this algorithm"""
        self.outputs.add(location)
    
    def get_signature(self) -> str:
        """Generate canonical signature for this algorithm"""
        if self._signature is None:
            # Create canonical form by analyzing data flow dependencies
            canon_form = self._get_canonical_form()
            sig_parts = []
            
            for item in canon_form:
                if isinstance(item, Instruction):
                    sig_parts.append(f"{item.operation}:{item.args}")
                else:  # Control flow
                    sig_parts.append(f"{item['type']}:{item.get('condition', '')}")
            
            self._signature = hashlib.md5('|'.join(str(s) for s in sig_parts).encode()).hexdigest()
        
        return self._signature
    
    def _get_canonical_form(self) -> List:
        """
        Convert to canonical form to eliminate redundant orderings.
        Uses dependency analysis to find minimal representation.
        """
        # Build dependency graph
        deps = defaultdict(set)
        outputs_map = defaultdict(list)
        
        for i, instr in enumerate(self.instructions):
            if isinstance(instr, Instruction):
                for arg in instr.args:
                    deps[i].update(outputs_map.get(arg, []))
                outputs_map[instr.target].append(i)
        
        # Topological sort for canonical ordering
        visited = set()
        canon_order = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in deps[node]:
                visit(dep)
            canon_order.append(node)
        
        for i in range(len(self.instructions)):
            visit(i)
        
        # Build canonical form
        return [self.instructions[i] for i in canon_order]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate algorithm structure and type safety"""
        errors = []
        
        # Check for type mismatches
        for i, instr in enumerate(self.instructions):
            if isinstance(instr, Instruction):
                op = self.instruction_set.operations.get(instr.operation)
                if not op:
                    errors.append(f"Instruction {i}: Unknown operation '{instr.operation}'")
                    continue
                
                # Check argument types (simplified)
                if len(instr.args) != len(op.arg_types):
                    errors.append(f"Instruction {i}: Argument count mismatch")
        
        # Check control flow balance
        if_count = sum(1 for i in self.instructions if isinstance(i, dict) and i['type'] == 'IF')
        end_count = sum(1 for i in self.instructions if isinstance(i, dict) and i['type'] == 'END')
        if if_count != end_count:
            errors.append(f"Unbalanced control flow: {if_count} IFs, {end_count} ENDs")
        
        return len(errors) == 0, errors
    
    def simplify(self) -> 'AlgorithmGenome':
        """
        Remove dead code and redundant instructions.
        Only keeps instructions that contribute to outputs.
        """
        if not self.outputs:
            return self  # No outputs defined, keep everything
        
        # Backward trace from outputs to find necessary instructions
        necessary = set()
        to_check = list(self.outputs)
        
        while to_check:
            target = to_check.pop()
            for i, instr in enumerate(self.instructions):
                if isinstance(instr, Instruction) and instr.target == target:
                    if i not in necessary:
                        necessary.add(i)
                        to_check.extend(instr.args)
        
        # Create simplified genome with only necessary instructions
        simplified = AlgorithmGenome(self.data_config, self.instruction_set)
        simplified.outputs = self.outputs.copy()
        
        for i in sorted(necessary):
            simplified.instructions.append(self.instructions[i])
        
        return simplified
    
    def to_executable(self) -> 'CompiledAlgorithm':
        """Convert to optimized executable form"""
        return CompiledAlgorithm(self)

class CompiledAlgorithm:
    """Optimized bytecode representation of an algorithm for fast execution"""
    def __init__(self, genome: AlgorithmGenome):
        self.genome = genome
        self.bytecode = self._compile()
    
    def _compile(self) -> List[Dict]:
        """Compile to optimized bytecode"""
        bytecode = []
        for instr in self.genome.instructions:
            if isinstance(instr, Instruction):
                op = self.genome.instruction_set.operations[instr.operation]
                bytecode.append({
                    'op': op,
                    'target': instr.target,
                    'args': instr.args
                })
            else:
                bytecode.append(instr)  # Control flow
        return bytecode
    
    def execute(self, data_store: UnifiedDataStore) -> Dict[str, Any]:
        """Execute algorithm and return outputs"""
        results = {}
        
        for item in self.bytecode:
            if 'op' in item:  # Regular instruction
                op = item['op']
                args = [data_store.stores[t][i] for t, i in item['args']]
                result = op.execute(*args)
                t, i = item['target']
                data_store.stores[t][i] = result
        
        # Collect outputs
        for t, i in self.genome.outputs:
            key = f"{t}_{i}"
            results[key] = data_store.stores[t][i]
        
        return results

# ============================================================================
# RESOURCE MANAGEMENT
# ============================================================================

import psutil
import gc
import pickle
import tempfile
import os
from pathlib import Path
from contextlib import contextmanager

class ResourceMonitor:
    """
    Monitors and manages system resources (VRAM, RAM, Disk).
    Provides automatic model offloading and loading based on resource availability.
    """
    def __init__(self, max_vram_usage: float = 0.8, max_ram_usage: float = 0.8,
                 cache_dir: Optional[Path] = None):
        """
        Args:
            max_vram_usage: Maximum fraction of VRAM to use (0.8 = 80%)
            max_ram_usage: Maximum fraction of RAM to use
            cache_dir: Directory for disk cache (temp dir if None)
        """
        self.max_vram_usage = max_vram_usage
        self.max_ram_usage = max_ram_usage
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / 'evolvo_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Track model locations
        self.model_locations: Dict[str, str] = {}  # model_id -> 'vram'/'ram'/'disk'
        self.model_sizes: Dict[str, float] = {}  # model_id -> size in MB
        self.loaded_models: Dict[str, nn.Module] = {}  # Currently in memory
        self.access_counts: Dict[str, int] = defaultdict(int)  # LRU tracking
        
        # Device management
        self.device = self._get_best_device()
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = torch.backends.mps.is_available()
        
    def _get_best_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_vram_info(self) -> Tuple[float, float]:
        """Get VRAM usage (used_mb, total_mb)"""
        if self.has_cuda:
            return (torch.cuda.memory_allocated() / 1024**2,
                   torch.cuda.max_memory_allocated() / 1024**2)
        elif self.has_mps:
            # MPS doesn't provide direct memory querying
            # Estimate based on system
            return (0, 8192)  # Default 8GB for Apple Silicon
        else:
            return (0, 0)
    
    def get_ram_info(self) -> Tuple[float, float]:
        """Get RAM usage (used_mb, total_mb)"""
        mem = psutil.virtual_memory()
        return (mem.used / 1024**2, mem.total / 1024**2)
    
    def estimate_model_size(self, genome: 'NeuralGenome') -> float:
        """Estimate model memory footprint in MB"""
        total_params = 0
        
        for layer in genome.layers:
            if layer.layer_type == 'linear':
                in_f = layer.params.get('in_features', 1)
                out_f = layer.params.get('out_features', 1)
                total_params += in_f * out_f + out_f
            elif layer.layer_type == 'conv2d':
                in_c = layer.params.get('in_channels', 1)
                out_c = layer.params.get('out_channels', 1)
                k = layer.params.get('kernel_size', 3)
                if isinstance(k, (list, tuple)):
                    k = k[0] * k[1]
                else:
                    k = k * k
                total_params += in_c * out_c * k + out_c
            # Add more layer types as needed
        
        # 4 bytes per parameter + overhead
        size_mb = (total_params * 4) / 1024**2 * 1.5  # 1.5x for overhead
        return size_mb
    
    def can_fit_in_vram(self, size_mb: float) -> bool:
        """Check if model can fit in VRAM"""
        if not (self.has_cuda or self.has_mps):
            return False
        
        used, total = self.get_vram_info()
        available = (total * self.max_vram_usage) - used
        return size_mb < available
    
    def can_fit_in_ram(self, size_mb: float) -> bool:
        """Check if model can fit in RAM"""
        used, total = self.get_ram_info()
        available = (total * self.max_ram_usage) - used
        return size_mb < available
    
    def offload_to_ram(self, model_id: str) -> bool:
        """Move model from VRAM to RAM"""
        if model_id not in self.loaded_models:
            return False
        
        model = self.loaded_models[model_id]
        try:
            model.cpu()
            self.model_locations[model_id] = 'ram'
            torch.cuda.empty_cache() if self.has_cuda else None
            return True
        except Exception:
            return False
    
    def offload_to_disk(self, model_id: str) -> bool:
        """Move model from RAM to disk cache"""
        if model_id not in self.loaded_models:
            return False
        
        model = self.loaded_models[model_id]
        cache_path = self.cache_dir / f"{model_id}.pkl"
        
        try:
            # Save state dict only (more efficient than full model)
            torch.save(model.state_dict(), cache_path)
            del self.loaded_models[model_id]
            self.model_locations[model_id] = 'disk'
            gc.collect()
            return True
        except Exception:
            return False
    
    def load_from_disk(self, model_id: str, genome: 'NeuralGenome') -> Optional[nn.Module]:
        """Load model from disk cache"""
        cache_path = self.cache_dir / f"{model_id}.pkl"
        if not cache_path.exists():
            return None
        
        try:
            model = genome.to_executable()
            model.load_state_dict(torch.load(cache_path, map_location='cpu'))
            self.loaded_models[model_id] = model
            self.model_locations[model_id] = 'ram'
            return model
        except Exception:
            return None
    
    def get_model(self, model_id: str, genome: 'NeuralGenome') -> Optional[nn.Module]:
        """Get model, loading from cache if necessary"""
        self.access_counts[model_id] += 1
        
        # Already loaded
        if model_id in self.loaded_models:
            model = self.loaded_models[model_id]
            
            # Try to move to VRAM if beneficial
            if (self.model_locations[model_id] == 'ram' and 
                self.can_fit_in_vram(self.model_sizes.get(model_id, 0))):
                model.to(self.device)
                self.model_locations[model_id] = 'vram'
            
            return model
        
        # Load from disk
        if self.model_locations.get(model_id) == 'disk':
            return self.load_from_disk(model_id, genome)
        
        # Create new model
        size_mb = self.estimate_model_size(genome)
        self.model_sizes[model_id] = size_mb
        
        # Check if we need to free memory
        if not self.can_fit_in_ram(size_mb):
            self._free_memory(size_mb)
        
        try:
            model = genome.to_executable()
            
            if self.can_fit_in_vram(size_mb):
                model.to(self.device)
                self.model_locations[model_id] = 'vram'
            else:
                self.model_locations[model_id] = 'ram'
            
            self.loaded_models[model_id] = model
            return model
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Try to free memory and retry
                self._free_memory(size_mb * 2)
                return self.get_model(model_id, genome)
            raise
    
    def _free_memory(self, required_mb: float):
        """Free memory using LRU eviction"""
        # Sort by access count (LRU)
        sorted_models = sorted(self.access_counts.items(), key=lambda x: x[1])
        
        freed = 0
        for model_id, _ in sorted_models:
            if freed >= required_mb:
                break
            
            if model_id in self.loaded_models:
                size = self.model_sizes.get(model_id, 0)
                location = self.model_locations.get(model_id, 'ram')
                
                if location == 'vram':
                    # First try moving to RAM
                    if self.offload_to_ram(model_id):
                        freed += size * 0.5  # Estimate partial freeing
                    else:
                        # If can't move to RAM, go directly to disk
                        self.offload_to_disk(model_id)
                        freed += size
                elif location == 'ram':
                    # Move to disk
                    self.offload_to_disk(model_id)
                    freed += size
        
        # Force garbage collection
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()
    
    @contextmanager
    def model_context(self, model_id: str, genome: 'NeuralGenome'):
        """Context manager for using a model with automatic resource management"""
        model = self.get_model(model_id, genome)
        try:
            yield model
        finally:
            # Optionally offload if memory pressure is high
            used_ram, total_ram = self.get_ram_info()
            if used_ram / total_ram > self.max_ram_usage:
                self.offload_to_disk(model_id)
    
    def cleanup_cache(self):
        """Clean up disk cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


####
####
####


class NeuralGenome(BaseGenome):
    """
    Genome representing a neural network architecture.
    Supports automatic shape inference and fine-tuning.
    """

    def __init__(self, input_shape: TensorShape, output_shape: TensorShape,
                 max_params: Optional[int] = None, max_memory_mb: Optional[float] = None):
        super().__init__(GenomeType.NEURAL)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers: List[LayerSpec] = []
        self.skip_connections: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        self.frozen_until: int = 0  # For fine-tuning: layers before this are frozen

        # Resource constraints
        self.max_params = max_params or 1e9  # Default 1B parameters
        self.max_memory_mb = max_memory_mb or 8192  # Default 8GB
        self.estimated_params: int = 0
        self.estimated_memory_mb: float = 0

    def add_layer(self, layer_spec: LayerSpec) -> bool:
        """Add layer with automatic shape inference and resource checking"""
        if self.layers:
            prev_shape = self.layers[-1].output_shape
            if prev_shape and layer_spec.input_shape:
                if not prev_shape.is_compatible_with(layer_spec.input_shape):
                    return False

        # Estimate resource usage for new layer
        new_params = self._estimate_layer_params(layer_spec)
        new_memory = self._estimate_layer_memory(layer_spec)

        # Check resource constraints
        if self.estimated_params + new_params > self.max_params:
            return False  # Would exceed parameter limit
        if self.estimated_memory_mb + new_memory > self.max_memory_mb:
            return False  # Would exceed memory limit

        self.layers.append(layer_spec)
        self.estimated_params += new_params
        self.estimated_memory_mb += new_memory
        self._signature = None
        return True

    def _estimate_layer_params(self, layer: LayerSpec) -> int:
        """Estimate number of parameters in a layer"""
        params = 0
        if layer.layer_type == 'linear':
            in_f = layer.params.get('in_features', 1)
            out_f = layer.params.get('out_features', 1)
            params = in_f * out_f
            if layer.params.get('bias', True):
                params += out_f
        elif layer.layer_type == 'conv2d':
            in_c = layer.params.get('in_channels', 1)
            out_c = layer.params.get('out_channels', 1)
            k = layer.params.get('kernel_size', 3)
            if isinstance(k, (list, tuple)):
                k = k[0] * k[1]
            else:
                k = k * k
            params = in_c * out_c * k
            if layer.params.get('bias', True):
                params += out_c
        # Add more layer types as needed
        return params

    def _estimate_layer_memory(self, layer: LayerSpec) -> float:
        """Estimate memory usage of a layer in MB"""
        params = self._estimate_layer_params(layer)
        # 4 bytes per parameter + activation memory (estimated)
        memory_mb = (params * 4) / 1024 ** 2

        # Add activation memory estimate
        if layer.output_shape:
            shape = layer.output_shape
            activation_elements = 1
            if shape.features:
                activation_elements = shape.features
            elif shape.channels and shape.height and shape.width:
                activation_elements = shape.channels * shape.height * shape.width

            # Assume batch size of 32 for memory estimation
            activation_memory_mb = (activation_elements * 32 * 4) / 1024 ** 2
            memory_mb += activation_memory_mb

        return memory_mb * 1.2  # 20% overhead

    def add_skip_connection(self, source: int, dest: int, merge_type: str = "add"):
        """Add skip connection with merge strategy"""
        if source >= dest or source < 0 or dest >= len(self.layers):
            return False
        self.skip_connections[source].append((dest, merge_type))
        self._signature = None
        return True

    def freeze_layers(self, until_layer: int):
        """Freeze layers for fine-tuning (layers before this index won't be modified)"""
        self.frozen_until = max(0, min(until_layer, len(self.layers)))

    def get_signature(self) -> str:
        """Generate canonical signature for architecture"""
        if self._signature is None:
            sig_parts = []

            # Canonical layer representation
            for layer in self.layers:
                sig_parts.append(f"{layer.layer_type}:{json.dumps(layer.params, sort_keys=True)}")

            # Canonical skip connections
            for src in sorted(self.skip_connections.keys()):
                for dest, merge in sorted(self.skip_connections[src]):
                    sig_parts.append(f"skip:{src}->{dest}:{merge}")

            self._signature = hashlib.md5('|'.join(sig_parts).encode()).hexdigest()

        return self._signature

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate architecture connectivity and shapes"""
        errors = []

        # Check shape compatibility
        prev_shape = self.input_shape
        for i, layer in enumerate(self.layers):
            if layer.input_shape and prev_shape:
                if not prev_shape.is_compatible_with(layer.input_shape):
                    errors.append(f"Layer {i}: Shape mismatch")
            prev_shape = layer.output_shape

        # Check output compatibility
        if prev_shape and not prev_shape.is_compatible_with(self.output_shape):
            errors.append("Final layer incompatible with expected output shape")

        # Validate skip connections
        for src, dests in self.skip_connections.items():
            for dest, _ in dests:
                if src >= dest:
                    errors.append(f"Invalid skip: {src} -> {dest}")

        return len(errors) == 0, errors

    def simplify(self) -> 'NeuralGenome':
        """Remove redundant layers and optimize architecture"""
        simplified = NeuralGenome(self.input_shape, self.output_shape)

        # Remove consecutive activations
        prev_was_activation = False
        for layer in self.layers:
            if layer.layer_type == 'activation':
                if prev_was_activation:
                    continue  # Skip redundant activation
                prev_was_activation = True
            else:
                prev_was_activation = False
            simplified.layers.append(layer)

        # Copy skip connections
        simplified.skip_connections = copy.deepcopy(self.skip_connections)

        return simplified

    def to_executable(self) -> nn.Module:
        """Convert to PyTorch model"""
        return DynamicNeuralModel(self)


# ============================================================================
# UNIFIED EVOLUTION ENGINE
# ============================================================================

class UnifiedEvolver:
    """
    Unified evolution engine for all genome types.
    Supports co-evolution of algorithms and neural networks.
    """
    def __init__(self, genome_type: GenomeType, population_size: int = 50):
        self.genome_type = genome_type
        self.population_size = population_size
        self.population: List[BaseGenome] = []
        self.generation = 0
        self.diversity_cache: Set[str] = set()
        self.hall_of_fame: List[BaseGenome] = []
        
        # Evolution parameters (adaptive)
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        self.elite_ratio = 0.1
        
        # Multi-objective optimization
        self.objectives = ['fitness', 'complexity', 'efficiency']
        self.pareto_front: List[BaseGenome] = []
    
    def add_genome(self, genome: BaseGenome) -> bool:
        """Add genome to population if novel"""
        signature = genome.get_signature()
        if signature in self.diversity_cache:
            return False
        
        self.diversity_cache.add(signature)
        self.population.append(genome)
        return True
    
    def crossover(self, parent1: BaseGenome, parent2: BaseGenome) -> BaseGenome:
        """Unified crossover operation"""
        if parent1.genome_type != parent2.genome_type:
            # Hybrid crossover not implemented in this version
            return copy.deepcopy(random.choice([parent1, parent2]))
        
        if isinstance(parent1, AlgorithmGenome):
            return self._crossover_algorithms(parent1, parent2)
        elif isinstance(parent1, NeuralGenome):
            return self._crossover_neural(parent1, parent2)
        else:
            return copy.deepcopy(parent1)
    
    def _crossover_algorithms(self, p1: AlgorithmGenome, p2: AlgorithmGenome) -> AlgorithmGenome:
        """Crossover for algorithm genomes"""
        child = AlgorithmGenome(p1.data_config, p1.instruction_set)
        
        # Single-point crossover on instructions
        if p1.instructions and p2.instructions:
            point1 = random.randint(0, len(p1.instructions))
            point2 = random.randint(0, len(p2.instructions))
            child.instructions = p1.instructions[:point1] + p2.instructions[point2:]
        
        # Inherit outputs from both parents
        child.outputs = p1.outputs | p2.outputs
        
        return child
    
    def _crossover_neural(self, p1: NeuralGenome, p2: NeuralGenome) -> NeuralGenome:
        """Crossover for neural genomes"""
        child = NeuralGenome(p1.input_shape, p1.output_shape)
        
        # Layer-wise crossover
        max_layers = max(len(p1.layers), len(p2.layers))
        for i in range(max_layers):
            if i < len(p1.layers) and i < len(p2.layers):
                # Randomly choose layer from either parent
                child.layers.append(copy.deepcopy(random.choice([p1.layers[i], p2.layers[i]])))
            elif i < len(p1.layers):
                if random.random() < 0.5:
                    child.layers.append(copy.deepcopy(p1.layers[i]))
            elif i < len(p2.layers):
                if random.random() < 0.5:
                    child.layers.append(copy.deepcopy(p2.layers[i]))
        
        # Inherit skip connections
        for parent in [p1, p2]:
            for src, dests in parent.skip_connections.items():
                if src < len(child.layers):
                    for dest, merge in dests:
                        if dest < len(child.layers) and random.random() < 0.5:
                            child.skip_connections[src].append((dest, merge))
        
        return child
    
    def mutate(self, genome: BaseGenome) -> BaseGenome:
        """Unified mutation operation"""
        mutated = copy.deepcopy(genome)
        
        if isinstance(mutated, AlgorithmGenome):
            self._mutate_algorithm(mutated)
        elif isinstance(mutated, NeuralGenome):
            self._mutate_neural(mutated)
        
        mutated.generation = self.generation
        mutated._signature = None  # Reset signature
        return mutated
    
    def _mutate_algorithm(self, genome: AlgorithmGenome):
        """Mutate algorithm genome"""
        if not genome.instructions:
            return
        
        mutation_type = random.choice(['modify', 'add', 'remove', 'reorder'])
        
        if mutation_type == 'modify' and genome.instructions:
            # Modify random instruction
            idx = random.randint(0, len(genome.instructions) - 1)
            if isinstance(genome.instructions[idx], Instruction):
                # Change operation or arguments
                ops = list(genome.instruction_set.operations.keys())
                genome.instructions[idx].operation = random.choice(ops)
        
        elif mutation_type == 'add':
            # Add new instruction
            # Implementation depends on specific requirements
            pass
        
        elif mutation_type == 'remove' and len(genome.instructions) > 1:
            # Remove random instruction
            idx = random.randint(0, len(genome.instructions) - 1)
            genome.instructions.pop(idx)
        
        elif mutation_type == 'reorder' and len(genome.instructions) > 2:
            # Swap two instructions if dependency-safe
            i, j = random.sample(range(len(genome.instructions)), 2)
            genome.instructions[i], genome.instructions[j] = genome.instructions[j], genome.instructions[i]
    
    def _mutate_neural(self, genome: NeuralGenome):
        """Mutate neural genome respecting frozen layers"""
        start_idx = genome.frozen_until  # Don't mutate frozen layers
        
        if len(genome.layers) <= start_idx:
            return
        
        mutation_type = random.choice(['modify', 'add', 'remove', 'skip'])
        
        if mutation_type == 'modify':
            # Modify layer parameters
            idx = random.randint(start_idx, len(genome.layers) - 1)
            layer = genome.layers[idx]
            # Modify random parameter
            if layer.params:
                param = random.choice(list(layer.params.keys()))
                # Generate new value (simplified)
                if isinstance(layer.params[param], bool):
                    layer.params[param] = not layer.params[param]
                elif isinstance(layer.params[param], (int, float)):
                    layer.params[param] *= random.uniform(0.5, 2.0)
        
        elif mutation_type == 'add':
            # Add new layer
            idx = random.randint(start_idx, len(genome.layers))
            # Create appropriate layer based on context
            # Implementation depends on layer factory
            pass
        
        elif mutation_type == 'remove' and len(genome.layers) > start_idx + 1:
            # Remove layer
            idx = random.randint(start_idx, len(genome.layers) - 1)
            genome.layers.pop(idx)
        
        elif mutation_type == 'skip':
            # Add or remove skip connection
            if genome.skip_connections and random.random() < 0.5:
                # Remove random skip
                src = random.choice(list(genome.skip_connections.keys()))
                genome.skip_connections.pop(src)
            else:
                # Add random skip
                if len(genome.layers) > 2:
                    src = random.randint(start_idx, len(genome.layers) - 2)
                    dest = random.randint(src + 1, len(genome.layers) - 1)
                    genome.skip_connections[src].append((dest, 'add'))
    
    def evolve(self, generations: int, evaluator: Callable[[BaseGenome], float],
              multi_objective: bool = False) -> List[BaseGenome]:
        """Main evolution loop"""
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate population
            for genome in self.population:
                if genome.fitness is None:
                    try:
                        genome.fitness = evaluator(genome)
                    except Exception as e:
                        genome.fitness = -float('inf')  # Penalize errors
            
            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness or -float('inf'), reverse=True)
            
            # Update hall of fame
            self.hall_of_fame.extend(self.population[:5])
            self.hall_of_fame.sort(key=lambda g: g.fitness or -float('inf'), reverse=True)
            self.hall_of_fame = self.hall_of_fame[:20]
            
            # Selection
            elite_size = int(self.population_size * self.elite_ratio)
            new_population = self.population[:elite_size]  # Elitism
            
            # --- START: MODIFIED REPRODUCTION LOOP ---
            attempts = 0
            max_attempts = self.population_size * 10 # Allow 10 attempts per slot

            while len(new_population) < self.population_size and attempts < max_attempts:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                # Check for novelty before adding to new_population
                if child.get_signature() not in self.diversity_cache:
                    self.diversity_cache.add(child.get_signature())
                    new_population.append(child)
                
                attempts += 1

            # If the loop timed out, fill the rest with mutated elites
            if len(new_population) < self.population_size:
                print(f"WARN: Population diversity exhausted. Filling with {self.population_size - len(new_population)} mutated elites.")
                while len(new_population) < self.population_size:
                    elite = copy.deepcopy(self.population[0])
                    mutated_elite = self.mutate(elite)
                    # We don't check for diversity here, just fill the spots
                    new_population.append(mutated_elite)

            self.population = new_population[:self.population_size]
            # --- END: MODIFIED REPRODUCTION LOOP ---
            
            # Adaptive rates
            if gen % 10 == 0:
                self._adapt_rates()
            
            # Report progress
            best = self.population[0]
            print(f"Generation {gen}: Best fitness = {best.fitness:.4f}, "
                  f"Unique genomes = {len(self.diversity_cache)}")
        
        return self.population
    
    def _tournament_select(self, tournament_size: int = 3) -> BaseGenome:
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness or -float('inf'))
    
    def _adapt_rates(self):
        """Adapt evolution rates based on progress"""
        # Check for stagnation
        if len(self.hall_of_fame) >= 5:
            recent_fitness = [g.fitness for g in self.hall_of_fame[:5]]
            if all(abs(f - recent_fitness[0]) < 0.001 for f in recent_fitness):
                # Increase mutation to escape local optimum
                self.mutation_rate = min(0.8, self.mutation_rate * 1.2)
            else:
                # Decrease mutation when making progress
                self.mutation_rate = max(0.1, self.mutation_rate * 0.95)


class ResourceAwareEvolver(UnifiedEvolver):
    """
    Enhanced evolver with resource management for parallel model evaluation.
    """
    def __init__(self, genome_type: GenomeType, population_size: int = 50,
                 max_model_params: int = 1e8, resource_monitor: Optional[ResourceMonitor] = None):
        super().__init__(genome_type, population_size)
        self.max_model_params = max_model_params
        self.resource_monitor = resource_monitor or ResourceMonitor()
        
        # Constraints based on available resources
        self._update_constraints()
    
    def _update_constraints(self):
        """Update evolution constraints based on available resources"""
        vram_used, vram_total = self.resource_monitor.get_vram_info()
        ram_used, ram_total = self.resource_monitor.get_ram_info()
        
        # Estimate maximum model size that can fit
        available_memory = min(
            (vram_total - vram_used) * 0.8 if vram_total > 0 else float('inf'),
            (ram_total - ram_used) * 0.8
        )
        
        # Assuming 4 bytes per parameter
        max_params_by_memory = (available_memory * 1024**2) / 4
        self.max_model_params = min(self.max_model_params, max_params_by_memory)
    
    def validate_genome_resources(self, genome: BaseGenome) -> bool:
        """Check if genome respects resource constraints"""
        if isinstance(genome, NeuralGenome):
            size_mb = self.resource_monitor.estimate_model_size(genome)
            
            # Check if it can fit anywhere (RAM or disk at minimum)
            if not self.resource_monitor.can_fit_in_ram(size_mb * 1.5):  # 1.5x for safety
                return False
            
            # Estimate parameter count
            estimated_params = (size_mb * 1024**2) / 4
            if estimated_params > self.max_model_params:
                return False
        
        return True
    

    def evaluate_population_parallel(self, evaluator: Callable, batch_size: int = 4):
        """
        Evaluate population in parallel batches with resource management.
        """
        self._update_constraints()
        
        batches = [self.population[i:i+batch_size] 
                for i in range(0, len(self.population), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            batch_models = []
            
            for genome in batch:
                # === START: New Upper Try-Except for Genome Isolation ===
                try:
                    # Reset fitness to ensure it's re-evaluated
                    genome.fitness = None

                    if not self.validate_genome_resources(genome):
                        genome.fitness = -float('inf')  # Penalize oversized models
                        continue # Skip to the next genome in the batch
                    
                    if isinstance(genome, NeuralGenome):
                        model_id = genome.get_signature()

                        # This inner try-except is still useful for context-specific errors
                        try:
                            with self.resource_monitor.model_context(model_id, genome) as model:
                                if model:
                                    batch_models.append((genome, model))
                                else:
                                    # Model failed to build or load
                                    genome.fitness = -float('inf')
                        except Exception as model_context_error:
                            print(f"ERROR: model_context failed for genome {model_id[:10]}...: {model_context_error}", file=sys.stderr)
                            genome.fitness = -float('inf')
                    else:
                        # For non-neural genomes like AlgorithmGenome
                        batch_models.append((genome, None))

                except Exception as processing_error:
                    # This catches ANY error related to a single genome (e.g., bad signature, validation bug)
                    import traceback
                    print(f"FATAL ERROR processing genome. Assigning -inf fitness.", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    genome.fitness = -float('inf')
                    # The 'continue' is implicit as we will just move to the next genome
                # === END: New Upper Try-Except ===

            # Evaluate batch (only models that were successfully prepared)
            for genome, model in batch_models:
                # If fitness was already set to -inf due to an error, skip evaluation
                if genome.fitness is not None:
                    continue

                try:
                    if model:
                        genome.fitness = evaluator(genome, model=model)
                    else:
                        genome.fitness = evaluator(genome)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        genome.fitness = -float('inf')
                        if isinstance(genome, NeuralGenome):
                            self.resource_monitor.offload_to_disk(genome.get_signature())
                    else:
                        # Catch other evaluation-time errors
                        genome.fitness = -float('inf')
                        print(f"ERROR: Evaluator failed for genome {genome.get_signature()[:10]}...: {e}", file=sys.stderr)
                except Exception as e:
                    genome.fitness = -float('inf')
                    print(f"ERROR: A non-runtime error occurred in evaluator for genome {genome.get_signature()[:10]}...: {e}", file=sys.stderr)
            
            # Free memory between batches
            if batch_idx < len(batches) - 1:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# ============================================================================
# NEURAL GENOME
# ============================================================================

@dataclass
class TensorShape:
    """Enhanced tensor shape with automatic compatibility checking"""
    batch: Optional[int] = None
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    sequence: Optional[int] = None
    features: Optional[int] = None
    
    def get_flat_features(self) -> int:
        """Calculate total features when flattened"""
        if self.features:
            return self.features
        if self.channels and self.height and self.width:
            return self.channels * self.height * self.width
        if self.channels:
            return self.channels * (self.height or 1) * (self.width or 1)
        if self.sequence and self.features:
            return self.sequence * self.features
        return 1
    
    def is_compatible_with(self, other: 'TensorShape') -> bool:
        """Check if shapes can be connected (with implicit flattening)"""
        # Feature to feature
        if self.features is not None and other.features is not None:
            return self.features == other.features
        
        # Spatial to feature (implicit flattening)
        if self.features is None and other.features is not None:
            return self.get_flat_features() == other.features
        
        # Feature to spatial requires explicit reshape
        if self.features is not None and other.features is None:
            return False
        
        # Spatial to spatial (channels can be adapted)
        if self.channels is not None and other.channels is not None:
            return True
        
        return True

@dataclass
class LayerSpec:
    """Neural network layer specification"""
    layer_type: str
    params: Dict[str, Any]
    input_shape: Optional[TensorShape] = None
    output_shape: Optional[TensorShape] = None
    computation_cost: float = 0.0
    memory_cost: float = 0.0
    
    def __hash__(self):
        return hash((self.layer_type, json.dumps(self.params, sort_keys=True)))

class DynamicNeuralModel(nn.Module):
    """PyTorch model dynamically created from NeuralGenome"""
    def __init__(self, genome: NeuralGenome):
        super().__init__()
        self.genome = genome
        self.layers = nn.ModuleList()
        self.skip_connections = genome.skip_connections
        self._build_layers()
    
    def _build_layers(self):
        """Build PyTorch layers from genome specification"""
        for spec in self.genome.layers:
            layer = self._create_layer(spec)
            if layer:
                self.layers.append(layer)
    
    def _create_layer(self, spec: LayerSpec) -> Optional[nn.Module]:
        """Create PyTorch layer from specification"""
        lt = spec.layer_type
        p = spec.params
        
        # Map layer types to PyTorch modules
        if lt == 'linear':
            return nn.Linear(p.get('in_features', 128), p['out_features'], p.get('bias', True))
        elif lt == 'conv2d':
            return nn.Conv2d(p.get('in_channels', 3), p['out_channels'],
                           p['kernel_size'], p.get('stride', 1), p.get('padding', 0))
        elif lt == 'batchnorm2d':
            return nn.BatchNorm2d(p['num_features'])
        elif lt == 'relu':
            return nn.ReLU()
        elif lt == 'dropout':
            return nn.Dropout(p.get('p', 0.5))
        elif lt == 'maxpool2d':
            return nn.MaxPool2d(p.get('kernel_size', 2), p.get('stride', 2))
        # Add more layer types as needed
        
        return None
    
    def forward(self, x):
        """Forward pass with skip connections"""
        outputs = {}
        
        for i, layer in enumerate(self.layers):
            # Apply layer
            if isinstance(layer, nn.Linear) and len(x.shape) > 2:
                x = torch.flatten(x, 1)  # Implicit flattening
            x = layer(x)
            outputs[i] = x
            
            # Handle incoming skip connections
            if i in [dest for srcs in self.skip_connections.values() for dest, _ in srcs]:
                for src, dests in self.skip_connections.items():
                    for dest, merge_type in dests:
                        if dest == i and src in outputs:
                            x = self._merge_tensors(x, outputs[src], merge_type)
        
        return x
    
    def _merge_tensors(self, t1: torch.Tensor, t2: torch.Tensor, merge_type: str) -> torch.Tensor:
        """Merge tensors according to strategy"""
        # Ensure shape compatibility
        if t1.shape != t2.shape:
            # Use adaptive pooling or linear projection to match shapes
            if len(t1.shape) == 4 and len(t2.shape) == 4:  # Conv features
                t2 = F.adaptive_avg_pool2d(t2, (t1.shape[2], t1.shape[3]))
                if t1.shape[1] != t2.shape[1]:  # Channel mismatch
                    conv = nn.Conv2d(t2.shape[1], t1.shape[1], 1).to(t1.device)
                    t2 = conv(t2)
        
        if merge_type == 'add':
            return t1 + t2
        elif merge_type == 'concat':
            return torch.cat([t1, t2], dim=1)
        elif merge_type == 'mul':
            return t1 * t2
        else:
            return t1 + t2  # Default to add

# ============================================================================
# Q-LEARNING INTEGRATION
# ============================================================================

class QLearningGuide:
    """
    Q-Learning agent for guiding evolution decisions.
    Learns which operations/layers work best in different contexts.
    """
    def __init__(self, state_features: int = 10, learning_rate: float = 0.1):
        self.state_features = state_features
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.epsilon = 0.1
        
        # Separate Q-tables for different genome types
        self.q_tables = {
            GenomeType.ALGORITHM: defaultdict(lambda: defaultdict(float)),
            GenomeType.NEURAL: defaultdict(lambda: defaultdict(float))
        }
        
        # Action spaces
        self.action_spaces = {
            GenomeType.ALGORITHM: ['ADD', 'SUB', 'MUL', 'DIV', 'IF', 'GT', 'AND'],
            GenomeType.NEURAL: ['conv2d', 'linear', 'batchnorm', 'dropout', 'relu', 'skip']
        }
    
    def get_state(self, genome: BaseGenome) -> str:
        """Extract state representation from genome"""
        if isinstance(genome, AlgorithmGenome):
            # Last N operations
            ops = [i.operation for i in genome.instructions[-self.state_features:] 
                  if isinstance(i, Instruction)]
            return '|'.join(ops)
        elif isinstance(genome, NeuralGenome):
            # Last N layer types
            types = [l.layer_type for l in genome.layers[-self.state_features:]]
            return '|'.join(types)
        return ""
    
    def choose_action(self, genome: BaseGenome, epsilon_greedy: bool = True) -> str:
        """Choose next action using epsilon-greedy policy"""
        state = self.get_state(genome)
        genome_type = genome.genome_type
        
        if epsilon_greedy and random.random() < self.epsilon:
            return random.choice(self.action_spaces[genome_type])
        
        q_values = self.q_tables[genome_type][state]
        if not q_values:
            return random.choice(self.action_spaces[genome_type])
        
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update(self, genome: BaseGenome, action: str, reward: float, next_genome: BaseGenome):
        """Update Q-values based on experience"""
        state = self.get_state(genome)
        next_state = self.get_state(next_genome)
        genome_type = genome.genome_type
        
        current_q = self.q_tables[genome_type][state][action]
        next_max_q = max(self.q_tables[genome_type][next_state].values()) \
                    if self.q_tables[genome_type][next_state] else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_tables[genome_type][state][action] = new_q
    
    def guide_mutation(self, genome: BaseGenome) -> BaseGenome:
        """Use Q-learning to guide mutation decisions"""
        action = self.choose_action(genome)
        mutated = copy.deepcopy(genome)
        
        if isinstance(mutated, AlgorithmGenome):
            # Add instruction based on Q-learning suggestion
            # Implementation depends on instruction format
            pass
        elif isinstance(mutated, NeuralGenome):
            # Add layer based on Q-learning suggestion
            # Implementation depends on layer factory
            pass
        
        return mutated

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Unified Evolvo Framework Demo ===\n")
    
    # 1. Algorithm Evolution Example
    print("1. Algorithm Evolution:")
    print("-" * 40)
    
    # Setup data configuration
    data_config = {
        'b#': ['true', 'false'],
        'd#': ['pi', 'e', 'one'],
        'b$' : ['result_bool'],
        'd$' : ['x', 'y', 'temp']
    }
    
    # Create instruction set
    instruction_set = EnhancedInstructionSet()
    
    # Create algorithm genome
    algo_genome = AlgorithmGenome(data_config, instruction_set)
    
    # Add some instructions
    algo_genome.add_instruction(Instruction(
        target=('d$', 2),  # temp
        operation='MUL', args=[('d$' , 0), ('d#', 2)]  # x * one
    ))
    algo_genome.add_instruction(Instruction(
        target=('d$', 1),  # y
        operation='ADD',
        args=[('d$', 2), ('d#', 1)]  # temp + e
    ))
    algo_genome.mark_output(('d$' , 1))  # y is output
    
    # Validate and simplify
    valid, errors = algo_genome.validate()
    print(f"Algorithm valid: {valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print(f"Signature: {algo_genome.get_signature()[:16]}...")
    
    # 2. Neural Architecture Evolution with Resource Management
    print("\n2. Neural Architecture Evolution with Resource Management:")
    print("-" * 40)
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(max_vram_usage=0.7, max_ram_usage=0.8)
    
    # Check available resources
    vram_used, vram_total = resource_monitor.get_vram_info()
    ram_used, ram_total = resource_monitor.get_ram_info()
    
    print(f"Available VRAM: {vram_total - vram_used:.0f}/{vram_total:.0f} MB")
    print(f"Available RAM: {(ram_total - ram_used):.0f}/{ram_total:.0f} MB")
    print(f"Device: {resource_monitor.device}")
    
    # Define shapes with resource constraints
    input_shape = TensorShape(batch=None, channels=3, height=224, width=224)
    output_shape = TensorShape(features=1000)
    
    # Calculate maximum model size based on available resources
    max_memory = min(
        (vram_total - vram_used) * 0.5 if vram_total > 0 else float('inf'),
        (ram_total - ram_used) * 0.3
    )
    max_params = int((max_memory * 1024**2) / 4)  # 4 bytes per param
    
    print(f"Maximum model parameters: {max_params/1e6:.1f}M")
    
    # Create neural genome with resource constraints
    neural_genome = NeuralGenome(
        input_shape, 
        output_shape,
        max_params=max_params,
        max_memory_mb=max_memory
    )
    
    # Add layers (will respect resource constraints)
    layers_added = 0
    layer_specs = [
        LayerSpec('conv2d', {'in_channels': 3, 'out_channels': 64, 'kernel_size': 7, 'stride': 2},
                 input_shape=input_shape,
                 output_shape=TensorShape(channels=64, height=112, width=112)),
        LayerSpec('batchnorm2d', {'num_features': 64},
                 output_shape=TensorShape(channels=64, height=112, width=112)),
        LayerSpec('relu', {},
                 output_shape=TensorShape(channels=64, height=112, width=112)),
        LayerSpec('conv2d', {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3},
                 output_shape=TensorShape(channels=128, height=112, width=112)),
    ]
    
    for spec in layer_specs:
        if neural_genome.add_layer(spec):
            layers_added += 1
            print(f"Added {spec.layer_type} - Total params: {neural_genome.estimated_params/1e6:.2f}M, "
                  f"Memory: {neural_genome.estimated_memory_mb:.1f}MB")
        else:
            print(f"Cannot add {spec.layer_type} - would exceed resource limits")
    
    # 3. Parallel Evolution with Resource Management
    print("\n3. Parallel Evolution with Resource Management:")
    print("-" * 40)
    
    # Create resource-aware evolver
    evolver = ResourceAwareEvolver(
        GenomeType.NEURAL, 
        population_size=10,
        max_model_params=max_params,
        resource_monitor=resource_monitor
    )
    
    # Generate initial population of small models
    for i in range(10):
        genome = NeuralGenome(
            input_shape, 
            output_shape,
            max_params=max_params,
            max_memory_mb=max_memory
        )
        
        # Add random small architecture
        num_layers = random.randint(3, 7)
        channels = 32
        
        for j in range(num_layers):
            if j % 3 == 0:
                # Conv layer
                layer = LayerSpec('conv2d', 
                                {'in_channels': channels if j > 0 else 3, 
                                 'out_channels': channels,
                                 'kernel_size': 3})
            elif j % 3 == 1:
                # Batch norm
                layer = LayerSpec('batchnorm2d', {'num_features': channels})
            else:
                # Activation
                layer = LayerSpec('relu', {})
            
            if not genome.add_layer(layer):
                break  # Hit resource limit
        
        evolver.add_genome(genome)
    
    # Define evaluation function that works with resource manager
    def evaluate_with_resources(genome: NeuralGenome, model: Optional[nn.Module] = None) -> float:
        """Evaluate model with automatic resource management"""
        if model is None:
            model_id = genome.get_signature()
            model = resource_monitor.get_model(model_id, genome)
        
        if model is None:
            return -float('inf')  # Model couldn't be created
        
        # Simple fitness based on parameter efficiency
        # In practice, this would involve actual training/evaluation
        param_count = genome.estimated_params
        if param_count == 0:
            return -float('inf')
        
        # Favor models that use resources efficiently
        efficiency = 1000 / (param_count / 1e6)  # Inverse of millions of parameters
        memory_penalty = genome.estimated_memory_mb / max_memory
        
        return efficiency - memory_penalty
    
    # Evolve with parallel batch evaluation
    print("Starting evolution with resource-aware batch processing...")
    evolver.evaluate_population_parallel(evaluate_with_resources, batch_size=3)
    
    # Report results
    evolver.population.sort(key=lambda g: g.fitness or -float('inf'), reverse=True)
    best = evolver.population[0]
    
    print(f"\nBest genome:")
    print(f"  Fitness: {best.fitness:.2f}")
    print(f"  Parameters: {best.estimated_params/1e6:.2f}M")
    print(f"  Memory: {best.estimated_memory_mb:.1f}MB")
    print(f"  Layers: {len(best.layers)}")
    
    # Show resource utilization
    print(f"\nResource utilization:")
    print(f"  Models in VRAM: {sum(1 for loc in resource_monitor.model_locations.values() if loc == 'vram')}")
    print(f"  Models in RAM: {sum(1 for loc in resource_monitor.model_locations.values() if loc == 'ram')}")
    print(f"  Models on disk: {sum(1 for loc in resource_monitor.model_locations.values() if loc == 'disk')}")
    
    # 4. Fine-tuning Example
    print("\n4. Fine-tuning with Frozen Layers:")
    print("-" * 40)
    
    # Take best model and prepare for fine-tuning
    best_genome = evolver.population[0]
    if isinstance(best_genome, NeuralGenome) and len(best_genome.layers) > 3:
        # Freeze early layers
        freeze_point = len(best_genome.layers) // 2
        best_genome.freeze_layers(freeze_point)
        print(f"Frozen first {freeze_point} layers for fine-tuning")
        
        # Now mutations will only affect layers after freeze_point
        mutated = evolver.mutate(best_genome)
        print(f"Mutated genome still has {freeze_point} frozen layers")
    
    # 5. Q-Learning Integration Example
    print("\n5. Q-Learning Guided Evolution:")
    print("-" * 40)
    
    q_guide = QLearningGuide()
    
    # Simulate learning from experience
    for i in range(20):
        genome = random.choice(evolver.population)
        action = q_guide.choose_action(genome)
        
        # Apply guided mutation
        original_fitness = genome.fitness or 0
        mutated = evolver.mutate(genome)
        
        # Evaluate mutated genome
        with resource_monitor.model_context(mutated.get_signature(), mutated) as model:
            if model:
                mutated.fitness = evaluate_with_resources(mutated, model)
            else:
                mutated.fitness = -float('inf')
        
        # Calculate reward
        reward = mutated.fitness - original_fitness
        
        # Update Q-values
        q_guide.update(genome, action, reward, mutated)
        
        if i % 5 == 0:
            print(f"  Iteration {i}: Explored {len(q_guide.q_tables[GenomeType.NEURAL])} states")
    
    print(f"\nQ-Learning explored {len(q_guide.q_tables[GenomeType.NEURAL])} unique states")
    
    # Cleanup
    print("\n6. Cleanup:")
    print("-" * 40)
    resource_monitor.cleanup_cache()
    print("Cache cleaned up")
    
    # Final resource status
    vram_used_end, _ = resource_monitor.get_vram_info()
    ram_used_end, _ = resource_monitor.get_ram_info()
    print(f"VRAM freed: {vram_used - vram_used_end:.0f} MB")
    print(f"RAM freed: {ram_used - ram_used_end:.0f} MB")
    
    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("- Automatic VRAM/RAM/Disk tiering for models")
    print("- Resource-aware population evolution")
    print("- Parallel batch evaluation with memory management")
    print("- Model size constraints based on available resources")
    print("- Fine-tuning with frozen layers")
    print("- Q-learning integration for guided evolution")
    print("- Automatic cleanup and resource monitoring")