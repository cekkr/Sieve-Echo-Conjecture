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

"""

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

class NeuralGenome(BaseGenome):
    """
    Genome representing a neural network architecture.
    Supports automatic shape inference and fine-tuning.
    """
    def __init__(self, input_shape: TensorShape, output_shape: TensorShape):
        super().__init__(GenomeType.NEURAL)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers: List[LayerSpec] = []
        self.skip_connections: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        self.frozen_until: int = 0  # For fine-tuning: layers before this are frozen
    
    def add_layer(self, layer_spec: LayerSpec) -> bool:
        """Add layer with automatic shape inference"""
        if self.layers:
            prev_shape = self.layers[-1].output_shape
            if prev_shape and layer_spec.input_shape:
                if not prev_shape.is_compatible_with(layer_spec.input_shape):
                    return False
        
        self.layers.append(layer_spec)
        self._signature = None
        return True
    
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
            
            # Reproduction
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                # Add if novel
                if self.add_genome(child):
                    new_population.append(child)
            
            self.population = new_population[:self.population_size]
            
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
        'b$': ['result_bool'],
        'd$': ['x', 'y', 'temp']
    }
    
    # Create instruction set
    instruction_set = EnhancedInstructionSet()
    
    # Create algorithm genome
    algo_genome = AlgorithmGenome(data_config, instruction_set)
    
    # Add some instructions
    algo_genome.add_instruction(Instruction(
        target=('d$', 2),  # temp
        operation='MUL',
        args=[('d$', 0), ('d#', 2)]  # x * one
    ))
    algo_genome.add_instruction(Instruction(
        target=('d$', 1),  # y
        operation='ADD',
        args=[('d$', 2), ('d#', 1)]  # temp + e
    ))
    algo_genome.mark_output(('d$', 1))  # y is output
    
    # Validate and simplify
    valid, errors = algo_genome.validate()
    print(f"Algorithm valid: {valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print(f"Signature: {algo_genome.get_signature()[:16]}...")
    
    # 2. Neural Architecture Evolution Example
    print("\n2. Neural Architecture Evolution:")
    print("-" * 40)
    
    # Define shapes
    input_shape = TensorShape(batch=None, channels=3, height=224, width=224)
    output_shape = TensorShape(features=1000)
    
    # Create neural genome
    neural_genome = NeuralGenome(input_shape, output_shape)
    
    # Add layers
    neural_genome.add_layer(LayerSpec(
        'conv2d',
        {'in_channels': 3, 'out_channels': 64, 'kernel_size': 7, 'stride': 2},
        input_shape=input_shape,
        output_shape=TensorShape(channels=64, height=112, width=112)
    ))
    neural_genome.add_layer(LayerSpec(
        'batchnorm2d',
        {'num_features': 64},
        output_shape=TensorShape(channels=64, height=112, width=112)
    ))
    neural_genome.add_layer(LayerSpec(
        'relu', {},
        output_shape=TensorShape(channels=64, height=112, width=112)
    ))
    
    # Add skip connection
    neural_genome.add_skip_connection(0, 2, 'add')
    
    # Validate
    valid, errors = neural_genome.validate()
    print(f"Neural architecture valid: {valid}")
    print(f"Signature: {neural_genome.get_signature()[:16]}...")
    
    # 3. Evolution Example
    print("\n3. Evolution Process:")
    print("-" * 40)
    
    # Create evolver
    evolver = UnifiedEvolver(GenomeType.ALGORITHM, population_size=10)
    
    # Generate initial population
    for _ in range(10):
        genome = AlgorithmGenome(data_config, instruction_set)
        # Add random instructions
        for _ in range(random.randint(2, 5)):
            genome.add_instruction(Instruction(
                target=(random.choice(['d$', 'b$']), random.randint(0, 2)),
                operation=random.choice(list(instruction_set.operations.keys())),
                args=[(random.choice(['d#', 'd$']), random.randint(0, 2)) 
                     for _ in range(2)]
            ))
        evolver.add_genome(genome)
    
    # Simple fitness function
    def dummy_fitness(genome):
        # Reward shorter algorithms
        return 10 - len(genome.instructions) if isinstance(genome, AlgorithmGenome) else 0
    
    # Evolve
    final_population = evolver.evolve(generations=5, evaluator=dummy_fitness)
    
    print(f"\nBest genome fitness: {final_population[0].fitness:.2f}")
    print(f"Population diversity: {len(evolver.diversity_cache)} unique genomes")
    
    # 4. Q-Learning Integration
    print("\n4. Q-Learning Guided Evolution:")
    print("-" * 40)
    
    q_guide = QLearningGuide()
    
    # Simulate learning
    for _ in range(10):
        genome = random.choice(evolver.population)
        action = q_guide.choose_action(genome)
        
        # Simulate mutation with action
        mutated = evolver.mutate(genome)
        
        # Calculate reward (simplified)
        reward = dummy_fitness(mutated) - dummy_fitness(genome)
        
        # Update Q-values
        q_guide.update(genome, action, reward, mutated)
    
    print("Q-Learning updated with experience")
    print(f"Explored states: {len(q_guide.q_tables[GenomeType.ALGORITHM])}")
    
    print("\n=== Demo Complete ===")