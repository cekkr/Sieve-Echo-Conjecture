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
- RobustSerializer: Advanced serialization with fallbacks (dill, pickle, json)
- FormulaResultsManager: Manages and saves experimental results robustly.

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
import sys
import psutil
import gc
import pickle
import tempfile
import os
from pathlib import Path
from contextlib import contextmanager
import dill
from datetime import datetime


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
            return self._can_reshape_to(self.shape, other.shape)
        return True
    
    def _can_reshape_to(self, from_shape: Tuple, to_shape: Tuple) -> bool:
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
        self.name_map: Dict[str, Tuple[str, int]] = {}
        for store_type, names in config.items():
            self.stores[store_type] = [self._default_value(store_type)] * len(names)
            for i, name in enumerate(names):
                self.name_map[name] = (store_type, i)
                self.types[name] = self._create_data_type(store_type)
    
    def _default_value(self, store_type: str) -> Any:
        if store_type.startswith('b'): return False
        if store_type.startswith('d'): return np.float64(0)
        return None
    
    def _create_data_type(self, store_type: str) -> DataType:
        category = {'b': 'bool', 'd': 'decimal', 't': 'tensor'}.get(store_type[0], 'unknown')
        is_constant = store_type.endswith('#')
        return DataType(category=category, is_constant=is_constant)
    
    def set(self, name: str, value: Any) -> bool:
        if name not in self.name_map: return False
        store_type, index = self.name_map[name]
        data_type = self.types[name]
        try:
            if data_type.category == 'bool': value = bool(value)
            elif data_type.category == 'decimal': value = np.float64(value)
            self.stores[store_type][index] = value
            return True
        except (ValueError, TypeError): return False
    
    def get(self, name: str, default: Any = None) -> Any:
        if name not in self.name_map: return default
        store_type, index = self.name_map[name]
        return self.stores[store_type][index]
    
    def reset(self):
        for store_type in self.stores:
            if store_type.endswith('$'):
                self.stores[store_type] = [self._default_value(store_type)] * len(self.stores[store_type])

    

# ============================================================================
# INSTRUCTION SET AND OPERATIONS
# ============================================================================

class Operation:
    def __init__(self, name: str, func: Callable, arg_types: List[str], 
                 return_type: str, category: str = 'arithmetic'):
        self.name = name
        self.func = func
        self.arg_types = arg_types
        self.return_type = return_type
        self.category = category
        self.error_default = self._get_error_default()
    
    def _get_error_default(self) -> Any:
        if self.return_type == 'bool': return False
        if self.return_type == 'decimal': return np.float64(0)
        return None
    
    def execute(self, *args) -> Any:
        try:
            if self.func is None: return self.error_default
            return self.func(*args)
        except Exception: return self.error_default


class EnhancedInstructionSet:
    """
    Extensible instruction set with automatic type checking and error handling.
    Supports both algorithmic operations and tensor operations for hybrid systems.
    """
    def __init__(self):
        self.operations: Dict[str, Operation] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self._register_default_operations()

    # --- Robust Mathematical Helper Functions ---
    # These functions ensure numerical stability by checking for invalid inputs (NaN, inf)
    # and catching overflows, returning a default value (0.0) instead of crashing.

    def _safe_add(self, a, b): return self._safe_op(lambda x, y: x + y, a, b)
    def _safe_sub(self, a, b): return self._safe_op(lambda x, y: x - y, a, b)
    def _safe_mul(self, a, b): return self._safe_op(lambda x, y: x * y, a, b)   
    def _safe_mod(self, a, b):
        if abs(np.float64(b)) < 1e-9: return np.float64(0)
        return self._safe_op(np.fmod, a, b)
   
    def _safe_exp(self, a):
        a_f = np.float64(a)
        if a_f > 700: return np.float64(np.inf) # Consistent with numpy
        return self._safe_op(np.exp, a_f)
   
    def _safe_sin(self, a): return self._safe_op(np.sin, a)
    def _safe_cos(self, a): return self._safe_op(np.cos, a)
    
    def _safe_abs(self, a): return self._safe_op(np.abs, a)
    
    # --- Comparison helpers to ensure type safety ---
    def _safe_eq(self, a, b): return abs(np.float64(a) - np.float64(b)) < 1e-9
    def _safe_gt(self, a, b): return np.float64(a) > np.float64(b)
    def _safe_lt(self, a, b): return np.float64(a) < np.float64(b)
    def _safe_gte(self, a, b): return np.float64(a) >= np.float64(b)
    def _safe_lte(self, a, b): return np.float64(a) <= np.float64(b)

    def _register_default_operations(self):
        """Register standard mathematical and logical operations"""
        # Arithmetic operations
        self.register('ADD', lambda a, b: self._safe_op(lambda x,y: x+y, a, b), ['decimal', 'decimal'], 'decimal')
        self.register('SUB', lambda a, b: self._safe_op(lambda x,y: x-y, a, b), ['decimal', 'decimal'], 'decimal')
        self.register('MUL', lambda a, b: self._safe_op(lambda x,y: x*y, a, b), ['decimal', 'decimal'], 'decimal')
        self.register('DIV', self._safe_div, ['decimal', 'decimal'], 'decimal')
        self.register('POW', self._safe_pow, ['decimal', 'decimal'], 'decimal')
        self.register('LOG', self._safe_log, ['decimal'], 'decimal')
        self.register('SQRT', self._safe_sqrt, ['decimal'], 'decimal')
        self.register('SIN', lambda a: self._safe_op(np.sin, a), ['decimal'], 'decimal')
        self.register('COS', lambda a: self._safe_op(np.cos, a), ['decimal'], 'decimal')
        self.register('ABS', lambda a: self._safe_op(np.abs, a), ['decimal'], 'decimal')
        
        # Mathematical functions
        self.register('EXP', self._safe_exp, ['decimal'], 'decimal', 'math')
        
        # Logical operations
        self.register('NOT', lambda a: not a, ['bool'], 'bool', 'logical')
        self.register('AND', lambda a, b: a and b, ['bool', 'bool'], 'bool', 'logical')
        self.register('OR', lambda a, b: a or b, ['bool', 'bool'], 'bool', 'logical')
        self.register('XOR', lambda a, b: a != b, ['bool', 'bool'], 'bool', 'logical')
        
        # Comparison operations
        self.register('EQ', self._safe_eq, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('GT', self._safe_gt, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('LT', self._safe_lt, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('GTE', self._safe_gte, ['decimal', 'decimal'], 'bool', 'comparison')
        self.register('LTE', self._safe_lte, ['decimal', 'decimal'], 'bool', 'comparison')
        
        # Control flow (special handling required)
        self.register('IF', None, ['bool'], None, 'control')
        self.register('ELSE', None, [], None, 'control')
        self.register('END', None, [], None, 'control')
        
        # Assignment (identity function)
        self.register('ASSIGN', lambda a: a, ['any'], 'any', 'control')
    
    def _safe_op(self, func, *args):
        try:
            args_float = [np.float64(arg) for arg in args]
            if not all(np.isfinite(arg) for arg in args_float): return np.float64(0)
            with np.errstate(all='raise'):
                result = func(*args_float)
                return result if np.isfinite(result) else np.float64(0)
        except (FloatingPointError, ValueError, TypeError): return np.float64(0)

    def _safe_div(self, a, b):
        if abs(np.float64(b)) < 1e-9: return np.float64(0)
        return self._safe_op(lambda x, y: x / y, a, b)
    def _safe_log(self, a):
        a_f = np.float64(a)
        if a_f <= 1e-9: return np.float64(-np.inf)
        return self._safe_op(np.log, a_f)
    def _safe_sqrt(self, a):
        a_f = np.float64(a)
        if a_f < 0: return np.float64(0)
        return self._safe_op(np.sqrt, a_f)
    def _safe_pow(self, a, b):
        a_f, b_f = np.float64(a), np.float64(b)
        if not (np.isfinite(a_f) and np.isfinite(b_f)): return np.float64(0)
        if a_f < 0 and b_f % 1 != 0: return np.float64(0)
        if abs(b_f) > 50: return np.float64(0)
        return self._safe_op(np.power, a_f, b_f)
    
    def register(self, name: str, func: Optional[Callable], arg_types: List[str], 
                return_type: Optional[str], category: str = 'custom'):
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
    target: Tuple[str, int]
    operation: str
    args: List[Tuple[str, int]]
    def get_dependencies(self) -> Set[Tuple[str, int]]: return set(self.args)
    def get_output(self) -> Tuple[str, int]: return self.target


class CompiledAlgorithm:
    def __init__(self, genome: AlgorithmGenome):
        self.genome = genome
    
    def execute(self, data_store: UnifiedDataStore) -> Dict[str, Any]:
        for instr in self.genome.instructions:
            op = self.genome.instruction_set.operations[instr.operation]
            args = [data_store.get(f"{t}_{i}") for t, i in instr.args]
            result = op.execute(*args)
            data_store.set(f"{instr.target[0]}_{instr.target[1]}", result)
        return {f"{t}_{i}": data_store.get(f"{t}_{i}") for t, i in self.genome.outputs}

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
        if op.func is not None and len(instruction.args) != len(op.arg_types):
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
                if op.func is not None and len(instr.args) != len(op.arg_types):
                    errors.append(f"Instruction {i}: Argument count mismatch for {op.name}")
        
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
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
    def _get_best_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_vram_info(self) -> Tuple[float, float]:
        """Get VRAM usage (used_mb, total_mb)"""
        if self.has_cuda:
            return (torch.cuda.memory_allocated() / 1024**2,
                   torch.cuda.get_device_properties(0).total_memory / 1024**2)
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
        total_params = genome.estimated_params
        
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
            if self.has_cuda:
                torch.cuda.empty_cache()
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
            if "out of memory" in str(e).lower():
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
            try:
                cache_file.unlink()
            except OSError:
                pass


# ============================================================================
# NEURAL GENOME & DYNAMIC MODEL
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

    def __init__(self, input_shape: 'TensorShape', output_shape: 'TensorShape',
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

    def add_layer(self, layer_spec: 'LayerSpec') -> bool:
        """Add layer with automatic shape inference and resource checking"""
        if self.layers:
            prev_shape = self.layers[-1].output_shape
            if prev_shape and layer_spec.input_shape:
                if not prev_shape.is_compatible_with(layer_spec.input_shape):
                    return False

        # Estimate resource usage for new layer
        new_params = self._estimate_layer_params(layer_spec)
        new_memory = self._estimate_layer_memory(new_params, layer_spec.output_shape)

        # Check resource constraints
        if self.estimated_params + new_params > self.max_params:
            return False
        if self.estimated_memory_mb + new_memory > self.max_memory_mb:
            return False

        self.layers.append(layer_spec)
        self.estimated_params += new_params
        self.estimated_memory_mb += new_memory
        self._signature = None
        return True

    def _estimate_layer_params(self, layer: 'LayerSpec') -> int:
        """Estimate number of parameters in a layer"""
        params = 0
        lt = layer.layer_type
        p = layer.params
        if lt == 'linear':
            params = p.get('in_features', 1) * p.get('out_features', 1)
            if p.get('bias', True): params += p.get('out_features', 1)
        elif lt == 'conv2d':
            k = p.get('kernel_size', 3)
            k_size = k*k if isinstance(k, int) else k[0]*k[1]
            params = p.get('in_channels', 1) * p.get('out_channels', 1) * k_size
            if p.get('bias', True): params += p.get('out_channels', 1)
        elif lt == 'batchnorm2d':
            params = 2 * p.get('num_features', 1) # weight and bias
        return params

    def _estimate_layer_memory(self, params: int, output_shape: Optional[TensorShape]) -> float:
        """Estimate memory usage of a layer in MB"""
        memory_mb = (params * 4) / 1024 ** 2 # 4 bytes per parameter

        if output_shape:
            activation_elements = output_shape.get_flat_features()
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
            for layer in self.layers:
                sig_parts.append(f"{layer.layer_type}:{json.dumps(layer.params, sort_keys=True)}")
            for src in sorted(self.skip_connections.keys()):
                for dest, merge in sorted(self.skip_connections[src]):
                    sig_parts.append(f"skip:{src}->{dest}:{merge}")
            self._signature = hashlib.md5('|'.join(sig_parts).encode()).hexdigest()
        return self._signature

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate architecture connectivity and shapes"""
        errors = []
        prev_shape = self.input_shape
        for i, layer in enumerate(self.layers):
            if layer.input_shape and prev_shape:
                if not prev_shape.is_compatible_with(layer.input_shape):
                    errors.append(f"Layer {i}: Shape mismatch")
            prev_shape = layer.output_shape

        if prev_shape and not prev_shape.is_compatible_with(self.output_shape):
            errors.append("Final layer incompatible with expected output shape")

        for src, dests in self.skip_connections.items():
            for dest, _ in dests:
                if src >= dest: errors.append(f"Invalid skip: {src} -> {dest}")

        return len(errors) == 0, errors

    def simplify(self) -> 'NeuralGenome':
        """Remove redundant layers and optimize architecture"""
        simplified = NeuralGenome(self.input_shape, self.output_shape)
        prev_was_activation = False
        for layer in self.layers:
            is_activation = layer.layer_type in ['relu', 'sigmoid', 'tanh']
            if is_activation:
                if prev_was_activation: continue
                prev_was_activation = True
            else:
                prev_was_activation = False
            simplified.add_layer(copy.deepcopy(layer))

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
        
        try:
            if lt == 'linear':
                return nn.Linear(int(p['in_features']), int(p['out_features']), p.get('bias', True))
            elif lt == 'conv2d':
                return nn.Conv2d(int(p['in_channels']), int(p['out_channels']),
                               int(p['kernel_size']), int(p.get('stride', 1)), int(p.get('padding', 0)))
            elif lt == 'batchnorm2d':
                return nn.BatchNorm2d(int(p['num_features']))
            elif lt == 'relu':
                return nn.ReLU()
            elif lt == 'dropout':
                return nn.Dropout(float(p.get('p', 0.5)))
            elif lt == 'maxpool2d':
                return nn.MaxPool2d(int(p.get('kernel_size', 2)), int(p.get('stride', 2)))
        except (TypeError, KeyError, ValueError) as e:
             print(f"ERROR creating layer {lt} with params {p}: {e}", file=sys.stderr)
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
            parent_layer = None
            if i < len(p1.layers) and i < len(p2.layers):
                parent_layer = random.choice([p1.layers[i], p2.layers[i]])
            elif i < len(p1.layers):
                if random.random() < 0.5: parent_layer = p1.layers[i]
            elif i < len(p2.layers):
                if random.random() < 0.5: parent_layer = p2.layers[i]
            
            if parent_layer:
                child.add_layer(copy.deepcopy(parent_layer))
        
        # Inherit skip connections
        for parent in [p1, p2]:
            for src, dests in parent.skip_connections.items():
                if src < len(child.layers):
                    for dest, merge in dests:
                        if dest < len(child.layers) and random.random() < 0.5:
                            child.add_skip_connection(src, dest, merge)
        
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

        mutation_type = random.choice(['modify', 'add', 'remove', 'reorder'])        

        if not genome.instructions and mutation_type != 'add':
            return
        
        if mutation_type == 'modify' and genome.instructions:
            idx = random.randint(0, len(genome.instructions) - 1)
            if isinstance(genome.instructions[idx], Instruction):
                old_op = genome.instruction_set.operations[genome.instructions[idx].operation]
                compatible_ops = [
                    op_name for op_name, op in genome.instruction_set.operations.items()
                    if len(op.arg_types) == len(old_op.arg_types) and op.return_type == old_op.return_type
                ]
                if compatible_ops:
                    genome.instructions[idx].operation = random.choice(compatible_ops)

        elif mutation_type == 'add':
            op_name = random.choice(list(genome.instruction_set.operations.keys()))
            op_info = genome.instruction_set.operations[op_name]

            if op_name not in ['IF', 'ELSE', 'END', 'ASSIGN']:
                if op_info.return_type == 'decimal': target_store = 'd$'
                elif op_info.return_type == 'bool': target_store = 'b$'
                else: return

                if not genome.data_config.get(target_store): return

                target_idx = random.randint(0, len(genome.data_config[target_store]) - 1)
                target = (target_store, target_idx)
                
                args = []
                for arg_type in op_info.arg_types:
                    if arg_type in ['decimal', 'any']: store_type = 'd#' if random.random() < 0.7 else 'd$'
                    elif arg_type == 'bool': store_type = 'b#' if random.random() < 0.5 else 'b$'
                    else: continue
                    
                    if not genome.data_config.get(store_type): continue
                    idx = random.randint(0, len(genome.data_config[store_type]) - 1)
                    args.append((store_type, idx))

                if len(args) == len(op_info.arg_types):
                    new_instruction = Instruction(target, op_name, args)
                    insert_pos = random.randint(0, len(genome.instructions))
                    genome.instructions.insert(insert_pos, new_instruction)
        
        elif mutation_type == 'remove' and len(genome.instructions) > 1:
            idx = random.randint(0, len(genome.instructions) - 1)
            genome.instructions.pop(idx)
        
        elif mutation_type == 'reorder' and len(genome.instructions) > 2:
            i, j = random.sample(range(len(genome.instructions)), 2)
            genome.instructions[i], genome.instructions[j] = genome.instructions[j], genome.instructions[i]
    
    def _mutate_neural(self, genome: NeuralGenome):
        """Mutate neural genome respecting frozen layers"""
        start_idx = genome.frozen_until
        
        if not genome.layers:
            # Handle empty genome: only 'add' mutation is possible
            mutation_type = 'add'
        else:
            mutation_type = random.choice(['modify', 'add', 'remove', 'skip'])

        if mutation_type == 'modify' and len(genome.layers) > start_idx:
            idx = random.randint(start_idx, len(genome.layers) - 1)
            layer = genome.layers[idx]
            if layer.params:
                param_key = random.choice(list(layer.params.keys()))
                original_value = layer.params[param_key]
                
                # --- START: TYPE-AWARE MUTATION (BUG FIX) ---
                if isinstance(original_value, bool):
                    layer.params[param_key] = not original_value
                elif isinstance(original_value, int):
                    new_val = int(round(original_value * random.uniform(0.75, 1.25)))
                    layer.params[param_key] = max(1, new_val) # Ensure it's at least 1
                elif isinstance(original_value, float):
                    layer.params[param_key] *= random.uniform(0.75, 1.25)
                # --- END: TYPE-AWARE MUTATION ---

        elif mutation_type == 'add':
            idx = random.randint(start_idx, len(genome.layers))
            # (Simplified layer creation logic)
            new_layer = LayerSpec('relu', {})
            genome.layers.insert(idx, new_layer)
        
        elif mutation_type == 'remove' and len(genome.layers) > start_idx + 1:
            idx = random.randint(start_idx, len(genome.layers) - 1)
            genome.layers.pop(idx)
        
        elif mutation_type == 'skip':
            if genome.skip_connections and random.random() < 0.5:
                if genome.skip_connections:
                    src = random.choice(list(genome.skip_connections.keys()))
                    if genome.skip_connections[src]:
                        genome.skip_connections[src].pop(random.randrange(len(genome.skip_connections[src])))
                        if not genome.skip_connections[src]:
                            del genome.skip_connections[src]
            else:
                if len(genome.layers) > start_idx + 1:
                    src, dest = sorted(random.sample(range(start_idx, len(genome.layers)), 2))
                    genome.add_skip_connection(src, dest, 'add')

        # Recalculate estimated params/memory after mutation
        genome.estimated_params = sum(genome._estimate_layer_params(l) for l in genome.layers)
        genome.estimated_memory_mb = sum(genome._estimate_layer_memory(genome._estimate_layer_params(l), l.output_shape) for l in genome.layers)
    
        def evolve(self, generations: int, evaluator: Callable[[BaseGenome], float],
              adaptive_config: Optional[Dict] = None,
              generation_callback: Optional[Callable] = None,
              callback_top_k: int = 0) -> List[BaseGenome]:
            """
            MODIFIED: Main evolution loop with adaptive early stopping and real-time callbacks.
            
            Args:
                generations: Maximum number of generations to run.
                evaluator: The fitness evaluation function.
                adaptive_config (dict): Optional config for early stopping.
                    - 'enabled' (bool): Whether to use adaptive generations.
                    - 'min_generations' (int): Minimum generations to run.
                    - 'stagnation_window' (int): How many generations to look back for progress.
                    - 'stagnation_threshold' (float): The minimum fitness improvement required.
                generation_callback (Callable): A function to call for top genomes every generation.
                    It receives (genome, generation, rank).
                callback_top_k (int): How many of the top genomes to pass to the callback.
            """
            
            fitness_history = []

            for gen in range(generations):
                self.generation = gen
                
                # Evaluate any unevaluated genomes
                for genome in self.population:
                    if genome.fitness is None:
                        try:
                            genome.fitness = evaluator(genome)
                        except Exception:
                            genome.fitness = -float('inf')
                
                self.population.sort(key=lambda g: g.fitness or -float('inf'), reverse=True)
                
                # --- NEW: Real-time callback for top genomes ---
                if generation_callback and callback_top_k > 0:
                    for i in range(min(callback_top_k, len(self.population))):
                        generation_callback(self.population[i], gen, i + 1)
                
                # Update Hall of Fame
                self.hall_of_fame.extend(self.population[:5])
                self.hall_of_fame.sort(key=lambda g: g.fitness or -float('inf'), reverse=True)
                self.hall_of_fame = self.hall_of_fame[:20]

                # --- NEW: Adaptive Early Stopping Logic ---
                if adaptive_config and adaptive_config.get('enabled', False):
                    best_fitness = self.population[0].fitness if self.population else -float('inf')
                    fitness_history.append(best_fitness)
                    
                    min_gens = adaptive_config.get('min_generations', 50)
                    if gen > min_gens:
                        window = adaptive_config.get('stagnation_window', 15)
                        threshold = adaptive_config.get('stagnation_threshold', 0.0001)
                        
                        if len(fitness_history) > window:
                            past_fitness = fitness_history[-window]
                            improvement = best_fitness - past_fitness
                            if improvement < threshold:
                                print(f"\nEvolution stagnated at generation {gen}. Best fitness={best_fitness:.4f}. Stopping early.")
                                break # Exit the loop

                elite_size = int(self.population_size * self.elite_ratio)
                new_population = self.population[:elite_size]
                
                # Generate new population
                attempts = 0
                max_attempts = self.population_size * 10
                while len(new_population) < self.population_size and attempts < max_attempts:
                    try:
                        parent1 = self._tournament_select()
                        parent2 = self._tournament_select()
                    except ValueError: break
                    
                    child = self.crossover(parent1, parent2) if random.random() < self.crossover_rate else copy.deepcopy(random.choice([parent1, parent2]))
                    if random.random() < self.mutation_rate:
                        child = self.mutate(child)

                    if child.get_signature() not in self.diversity_cache:
                        self.diversity_cache.add(child.get_signature())
                        new_population.append(child)
                    attempts += 1

                if len(new_population) < self.population_size and self.population:
                    print(f"WARN: Filling with {self.population_size - len(new_population)} mutated elites.")
                    while len(new_population) < self.population_size:
                        new_population.append(self.mutate(copy.deepcopy(self.population[0])))

                self.population = new_population[:self.population_size]
                
                if gen % 10 == 0: self._adapt_rates()
                
                if self.population:
                    best = self.population[0]
                    print(f"Gen {gen:03d}: Best Fitness={best.fitness:.4f} | Pop Size={len(self.population)} | Unique Genomes={len(self.diversity_cache)}")
            
            return self.population
    
    def _tournament_select(self, tournament_size: int = 3) -> BaseGenome:
        """Tournament selection"""
        if not self.population:
            raise ValueError("Cannot select from an empty population.")
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness or -float('inf'))
    
    def _adapt_rates(self):
        """Adapt evolution rates based on progress"""
        if len(self.hall_of_fame) >= 5:
            recent_fitness = [g.fitness for g in self.hall_of_fame[:5] if g.fitness is not None]
            if not recent_fitness: return
            
            if all(abs(f - recent_fitness[0]) < 1e-4 for f in recent_fitness):
                self.mutation_rate = min(0.8, self.mutation_rate * 1.2)
            else:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.95)

# ============================================================================
# RESOURCE-AWARE EVOLVER
# ============================================================================

class ResourceAwareEvolver(UnifiedEvolver):
    """
    Enhanced evolver with resource management for parallel model evaluation.
    """
    def __init__(self, genome_type: GenomeType, population_size: int = 50,
                 max_model_params: int = 1e8, resource_monitor: Optional[ResourceMonitor] = None):
        super().__init__(genome_type, population_size)
        self.max_model_params = max_model_params
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self._update_constraints()
    
    def _update_constraints(self):
        """Update evolution constraints based on available resources"""
        _, vram_total = self.resource_monitor.get_vram_info()
        _, ram_total = self.resource_monitor.get_ram_info()
        available_mem_mb = min(vram_total * 0.8, ram_total * 0.8)
        max_params_by_mem = (available_mem_mb * 1024**2) / 8 # 8 bytes for param+grad
        self.max_model_params = min(self.max_model_params, max_params_by_mem)
    
    def validate_genome_resources(self, genome: BaseGenome) -> bool:
        """Check if genome respects resource constraints"""
        if isinstance(genome, NeuralGenome):
            if genome.estimated_params > self.max_model_params: return False
            size_mb = self.resource_monitor.estimate_model_size(genome)
            if not self.resource_monitor.can_fit_in_ram(size_mb * 1.5): return False
        return True
    
    def evaluate_population_parallel(self, evaluator: Callable, batch_size: int = 4):
        """
        Evaluate population in parallel batches with resource management.
        """
        self._update_constraints()
        batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            batch_models = []
            for genome in batch:
                try:
                    genome.fitness = None
                    if not self.validate_genome_resources(genome):
                        genome.fitness = -float('inf')
                        continue
                    
                    if isinstance(genome, NeuralGenome):
                        model_id = genome.get_signature()
                        with self.resource_monitor.model_context(model_id, genome) as model:
                            if model: batch_models.append((genome, model))
                            else: genome.fitness = -float('inf')
                    else:
                        batch_models.append((genome, None))
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    genome.fitness = -float('inf')

            for genome, model in batch_models:
                if genome.fitness is not None: continue
                try:
                    genome.fitness = evaluator(genome, model=model) if model else evaluator(genome)
                except Exception as e:
                    genome.fitness = -float('inf')
                    if "out of memory" in str(e).lower() and isinstance(genome, NeuralGenome):
                        self.resource_monitor.offload_to_disk(genome.get_signature())
                    else:
                        print(f"ERROR: Evaluator failed for {genome.get_signature()[:10]}: {e}", file=sys.stderr)

            if batch_idx < len(batches) - 1:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

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
        
        self.q_tables = {
            GenomeType.ALGORITHM: defaultdict(lambda: defaultdict(float)),
            GenomeType.NEURAL: defaultdict(lambda: defaultdict(float))
        }
        
        self.action_spaces = {
            GenomeType.ALGORITHM: ['ADD', 'SUB', 'MUL', 'DIV', 'IF', 'GT', 'AND'],
            GenomeType.NEURAL: ['conv2d', 'linear', 'batchnorm', 'dropout', 'relu', 'skip']
        }
    
    def get_state(self, genome: BaseGenome) -> str:
        """Extract state representation from genome"""
        if isinstance(genome, AlgorithmGenome):
            ops = [i.operation for i in genome.instructions[-self.state_features:] if isinstance(i, Instruction)]
            return '|'.join(ops)
        elif isinstance(genome, NeuralGenome):
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
        next_max_q = max(self.q_tables[genome_type][next_state].values()) if self.q_tables[genome_type][next_state] else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_tables[genome_type][state][action] = new_q
    
    def guide_mutation(self, genome: BaseGenome) -> BaseGenome:
        """Use Q-learning to guide mutation decisions"""
        # (Simplified example of guided mutation)
        action = self.choose_action(genome)
        mutated = copy.deepcopy(genome)
        
        if isinstance(mutated, NeuralGenome) and action in ['conv2d', 'linear', 'relu']:
            new_layer = LayerSpec(action, {}) # Params would be more complex
            mutated.layers.append(new_layer)
        return mutated


# ============================================================================
# SERIALIZATION & RESULTS MANAGEMENT
# ============================================================================

class EnhancedAlgorithmSerializer:
    """Ensures complete algorithm structure is captured and saved"""
    
    @staticmethod
    def serialize_genome_complete(genome, data_config=None):
        """Extract COMPLETE algorithm structure from genome"""
        result = {
            'signature': genome.get_signature() if hasattr(genome, 'get_signature') else None,
            'fitness': genome.fitness,
            'generation': genome.generation if hasattr(genome, 'generation') else 0,
            'instructions': [],
            'outputs': [],
            'data_config': data_config or (genome.data_config if hasattr(genome, 'data_config') else {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract every instruction with full detail
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
                    
                    # Detailed argument extraction
                    for arg in instr.args:
                        instr_data['args'].append({
                            'store': arg[0],
                            'index': arg[1],
                            'full': f"{arg[0]}[{arg[1]}]"
                        })
                    
                    # Add human-readable form
                    arg_str = ', '.join([a['full'] for a in instr_data['args']])
                    instr_data['readable'] = f"{instr_data['target']['full']} = {instr.operation}({arg_str})"
                    
                    result['instructions'].append(instr_data)
                else:  # Control flow
                    result['instructions'].append({
                        'index': idx,
                        'type': 'control_flow',
                        'control_type': instr.get('type', 'unknown'),
                        'condition': instr.get('condition')
                    })
        
        # Extract outputs
        if hasattr(genome, 'outputs'):
            for out in genome.outputs:
                result['outputs'].append({
                    'store': out[0],
                    'index': out[1],
                    'full': f"{out[0]}[{out[1]}]"
                })
        
        return result

class RobustSerializer:
    """
    Multi-level fallback serialization system.
    Priority: dill -> pickle -> json
    """
    
    @staticmethod
    def make_json_safe(obj: Any, max_depth: int = 10, current_depth: int = 0, discoverer_instance=None) -> Any:
        if current_depth > max_depth: return f"<max_depth_exceeded:{type(obj).__name__}>"
        if obj is None or isinstance(obj, (bool, int, float, str)): return obj
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, float):
            if np.isnan(obj): return "NaN"
            if np.isinf(obj): return "Inf" if obj > 0 else "-Inf"
        if isinstance(obj, dict):
            return {str(k): RobustSerializer.make_json_safe(v, max_depth, current_depth + 1, discoverer_instance) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RobustSerializer.make_json_safe(item, max_depth, current_depth + 1, discoverer_instance) for item in obj]
        if isinstance(obj, set): return list(obj)
        
        # --- START: ENHANCED GENOME SERIALIZATION ---
        if isinstance(obj, AlgorithmGenome) and discoverer_instance:
             # Use the discoverer's own method to get a rich, serializable dict
             return discoverer_instance._decode_genome(obj)
        # --- END: ENHANCED GENOME SERIALIZATION ---
        
        if isinstance(obj, BaseGenome):
            return {
                'type': obj.__class__.__name__,
                'fitness': obj.fitness,
                'signature': obj.get_signature()[:16],
            }
        if hasattr(obj, '__dict__'):
            try:
                return {'_type': obj.__class__.__name__, '_data': RobustSerializer.make_json_safe(obj.__dict__, max_depth, current_depth + 1, discoverer_instance)}
            except:
                return f"<{obj.__class__.__name__}>"
        return str(obj)
    
    @staticmethod
    def save_with_fallbacks(data: Any, base_path: Union[str, Path], verbose: bool = True, discoverer_instance=None) -> bool:
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        strategies = [
            ('dill', '.dill', lambda d, p: dill.dump(d, open(p, 'wb'))),
            ('pickle', '.pkl', lambda d, p: pickle.dump(d, open(p, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)),
            ('json', '.json', lambda d, p: json.dump(RobustSerializer.make_json_safe(d, discoverer_instance=discoverer_instance), open(p, 'w'), indent=2)),
        ]
        saved = False
        for name, ext, func in strategies:
            filepath = base_path.with_suffix(ext)
            try:
                func(data, filepath)
                if verbose: print(f"✓ Saved {name} to {filepath}")
                saved = True
            except Exception as e:
                if verbose: print(f"✗ Failed {name}: {str(e)[:150]}")
        return saved
    
    @staticmethod
    def load_with_fallbacks(base_path: Union[str, Path], verbose: bool = True) -> Any:
        base_path = Path(base_path)
        # Prioritize dill/pickle as they preserve object types
        strategies = [('.dill', dill.load, 'rb'), ('.pkl', pickle.load, 'rb'), ('.json', json.load, 'r')]
        for ext, loader, mode in strategies:
            filepath = base_path.with_suffix(ext)
            if filepath.exists():
                try:
                    with open(filepath, mode) as f:
                        data = loader(f)
                    if verbose: print(f"✓ Loaded from {filepath}")
                    return data
                except Exception as e:
                    if verbose: print(f"✗ Failed to load {filepath}: {str(e)[:100]}")
        return None

class FormulaResultsManager:
    """Manages formula discovery results with JSON-only output."""
    def __init__(self, results_dir: str = "results_v7"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.serializer = RobustSerializer()
    
    def save_cycle_results(self, cycle: int, formula_discoveries: Dict, 
                      nn_results: Dict = None, base_invariance: float = None, discoverer_instance=None):
        """Save cycle results with proper algorithm serialization."""
        cycle_dir = self.results_dir / f"cycle_{cycle:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        cycle_data = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'formula_results': formula_discoveries,
            'neural_results': self._process_nn_results(nn_results),
            'base_invariance': base_invariance,
        }
        
        # Save main cycle data with all results
        self.serializer.save_with_fallbacks(cycle_data, cycle_dir / 'cycle_data', discoverer_instance=discoverer_instance)
        
        # Save top formulas individually with full algorithm structure
        if formula_discoveries and 'top_discoveries' in formula_discoveries:
            for i, discovery in enumerate(formula_discoveries['top_discoveries'][:50]):
                # Create properly structured formula data
                formula_data = {
                    'rank': discovery.get('rank', i + 1),
                    'fitness': discovery.get('fitness', 0),
                    'timestamp': datetime.now().isoformat(),
                }
                
                # Extract and serialize the complete algorithm structure
                if 'genome' in discovery and discovery['genome'] is not None:
                    genome = discovery['genome']
                    
                    # Build the complete algorithm representation
                    if hasattr(genome, 'instructions'):
                        instructions_data = []
                        for idx, instr in enumerate(genome.instructions):
                            if hasattr(instr, 'target'):  # Regular instruction
                                instructions_data.append({
                                    'index': idx,
                                    'type': 'instruction',
                                    'target': {'store': instr.target[0], 'index': instr.target[1]},
                                    'operation': instr.operation,
                                    'args': [{'store': arg[0], 'index': arg[1]} for arg in instr.args]
                                })
                            else:  # Control flow structure
                                instructions_data.append({
                                    'index': idx,
                                    'type': 'control_flow',
                                    'control_type': instr.get('type'),
                                    'condition': instr.get('condition')
                                })
                        
                        # Include data configuration for reconstruction
                        data_config = {}
                        if hasattr(genome, 'data_config'):
                            data_config = genome.data_config
                        
                        formula_data['algorithm'] = {
                            'instructions': instructions_data,
                            'outputs': [{'store': out[0], 'index': out[1]} for out in genome.outputs] if hasattr(genome, 'outputs') else [],
                            'signature': genome.get_signature() if hasattr(genome, 'get_signature') else None,
                            'data_config': data_config,
                            'instruction_count': len(instructions_data)
                        }
                    
                    # Add human-readable decoded version
                    if discoverer_instance and hasattr(discoverer_instance, '_decode_genome'):
                        try:
                            decoded = discoverer_instance._decode_genome(genome)
                            formula_data['decoded'] = decoded
                        except Exception as e:
                            print(f"Warning: Could not decode genome {i}: {e}")
                            formula_data['decoded'] = {'error': str(e)}
                
                # Save the complete formula with algorithm structure
                formula_path = cycle_dir / f'formula_{i:02d}.json'
                try:
                    with open(formula_path, 'w') as f:
                        json.dump(self.serializer.make_json_safe(formula_data, discoverer_instance=discoverer_instance), 
                                f, indent=2, default=str)
                    print(f"  ✓ Saved formula {i:02d} with {formula_data.get('algorithm', {}).get('instruction_count', 0)} instructions")
                except Exception as e:
                    print(f"  ✗ Failed to save formula {i:02d}: {e}")

    def _process_nn_results(self, r: Dict) -> Dict:
        if not r: return {}
        return {'fitness': r.get('fitness', 0), 'architecture': r.get('architecture', '')}
    
    def _process_formula_discoveries(self, d: Dict) -> Dict:
        if not d: return {}
        result = {
            'total_unique_genomes': d.get('total_unique_genomes', 0),
            'top_formulas': []
        }
        for info in d.get('top_discoveries', [])[:50]:
            decoded = info.get('decoded', {})
            result['top_formulas'].append({
                'rank': info.get('rank', 0), 'signature': info.get('signature', ''),
                'scores': {'fitness': info.get('fitness', 0), 'combined': info.get('combined_score', 0)},
                'metrics': info.get('metrics', {}),
                'structure': {'effective_length': decoded.get('effective_length', 0), 'ops': decoded.get('unique_operations', [])},
                'formula': {'symbolic': decoded.get('symbolic_formula', [])}
            })
        if 'generation_history' in d:
            result['evolution_progress'] = {'total_generations': len(d['generation_history'])}
        return result   

    def _create_summary(self, formulas_data: Dict, nn_results: Dict) -> Dict:
        summary = {'best_formula_fitness': 0, 'best_nn_fitness': 0}
        if formulas_data and formulas_data.get('top_formulas'):
            summary['best_formula_fitness'] = max(f['scores']['fitness'] for f in formulas_data['top_formulas'])
        if nn_results:
            summary['best_nn_fitness'] = nn_results.get('fitness', 0)
        return summary