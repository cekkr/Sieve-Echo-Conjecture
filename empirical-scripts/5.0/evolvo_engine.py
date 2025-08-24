# -*- coding: utf-8 -*-
# evolvo_engine.py
# Version 1.1: Added AlgorithmGenerator and minor optimizations.

import math
import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Any, Callable, Tuple

myFloat = np.double

class DataStore:
    def __init__(self, config: Dict[str, List[str]]):
        self.config = config
        self.stores: Dict[str, List[Any]] = {}
        self.store_names: Dict[str, List[str]] = {}
        self.name_to_type_map: Dict[str, str] = {}
        self.initial_values: Dict[str, Any] = {}
        for store_type, names in config.items():
            self.stores[store_type] = [None] * len(names)
            self.store_names[store_type] = list(names)
            for i, name in enumerate(names):
                self.name_to_type_map[name] = (store_type, i)
        self.reset()

    def set_initial_value(self, name: str, value: Any):
        if name not in self.name_to_type_map:
            raise ValueError(f"'{name}' not defined in store configuration.")
        self.initial_values[name] = value
        self.set(name, value)

    def get_store_type_and_index(self, name: str) -> Tuple[str, int]:
        try:
            return self.name_to_type_map[name]
        except KeyError:
            raise NameError(f"Store named '{name}' not found.")

    def set(self, name: str, value: Any):
        store_type, index = self.get_store_type_and_index(name)
        self.set_by_location(store_type, index, value)

    def get(self, name: str) -> Any:
        store_type, index = self.get_store_type_and_index(name)
        return self.get_by_location(store_type, index)

    def set_by_location(self, store_type: str, index: int, value: Any):
        is_bool = store_type.startswith('b')
        if is_bool: value = bool(value)
        elif not isinstance(value, myFloat): value = myFloat(value)
        self.stores[store_type][index] = value

    def get_by_location(self, store_type: str, index: int) -> Any:
        return self.stores[store_type][index]

    def reset(self):
        for store_type, names in self.store_names.items():
            is_bool = store_type.startswith('b')
            default_val = False if is_bool else myFloat(0)
            self.stores[store_type] = [default_val] * len(names)
        for name, value in self.initial_values.items(): self.set(name, value)

class InstructionSet:
    def __init__(self):
        self.operations: Dict[str, Callable] = {}
        self.op_properties: Dict[str, Dict[str, Any]] = {}
        self.op_types: Dict[str, List[str]] = defaultdict(list)

    def register(self, name: str, function: Callable, arg_types: List[str], op_type: str = "decimal"):
        self.operations[name] = function
        self.op_properties[name] = {'name': name, 'arg_count': len(arg_types), 'arg_types': arg_types}
        self.op_types[op_type].append(name)

def get_default_instruction_set() -> InstructionSet:
    iset = InstructionSet()
    # Decimal Ops
    iset.register('ADD', lambda a, b: a + b, ['d', 'd'], op_type='decimal')
    iset.register('SUB', lambda a, b: a - b, ['d', 'd'], op_type='decimal')
    iset.register('MUL', lambda a, b: a * b, ['d', 'd'], op_type='decimal')
    iset.register('DIV', lambda a, b: a / b if b != 0 else myFloat(1), ['d', 'd'], op_type='decimal') # Avoid DivByZero
    # Boolean Ops
    iset.register('NOT', lambda a: not a, ['b'], op_type='bool')
    iset.register('CMP', lambda a, b: a == b, ['d', 'd'], op_type='bool')
    iset.register('GT',  lambda a, b: a > b, ['d', 'd'], op_type='bool')
    return iset

class Interpreter:
    def __init__(self, instruction_set: InstructionSet):
        self.iset = instruction_set

    def _to_bytecode(self, instructions: List[List[Any]]) -> List[Dict]:
        bytecode = []
        for instr in instructions:
            if not instr or len(instr) < 3: continue
            op_name = instr[2]
            op_props = self.iset.op_properties.get(op_name)
            if not op_props: continue # Skip invalid op
            
            line_instr = {
                'target': (instr[0], instr[1]),
                'op_func': self.iset.operations[op_name],
                'args': [(instr[3], instr[4])]
            }
            if op_props['arg_count'] == 2:
                line_instr['args'].append((instr[5], instr[6]))
            bytecode.append(line_instr)
        return bytecode

    def execute(self, instructions: List[List[Any]], data_store: DataStore):
        bytecode = self._to_bytecode(instructions)
        for line in bytecode:
            # Resolve arguments
            arg_vals = [data_store.get_by_location(*arg) for arg in line['args']]
            # Get result from the operation
            result = line['op_func'](*arg_vals)
            # Assign result to target
            data_store.set_by_location(*line['target'], result)

class BaseEvaluator:
    def __init__(self, data_store_config: Dict, instruction_set: InstructionSet):
        self.data_store_config = data_store_config
        self.instruction_set = instruction_set
        self.interpreter = Interpreter(self.instruction_set)

    def evaluate(self, algorithm: List, **kwargs) -> float:
        raise NotImplementedError("You must implement the 'evaluate' method.")

class AlgorithmGenerator:
    """Generates and manipulates algorithms for genetic programming."""
    def __init__(self, data_store_config: Dict, instruction_set: InstructionSet):
        self.config = data_store_config
        self.iset = instruction_set
        self.d_vars = self.config.get('d$', [])
        self.d_consts = self.config.get('d#', [])
        self.b_vars = self.config.get('b$', [])
        self.b_consts = self.config.get('b#', [])
        self.decimal_ops = self.iset.op_types['decimal']
        self.bool_ops = self.iset.op_types['bool']

    def _get_random_arg(self, arg_type: str) -> Tuple[str, int]:
        if arg_type == 'd':
            source_type = random.choice(['d$', 'd#'])
            source_list = self.d_vars if source_type == 'd$' else self.d_consts
            if not source_list: return self._get_random_arg('d') # Recurse if empty
            return source_type, random.randrange(len(source_list))
        else: # arg_type == 'b'
            source_type = random.choice(['b$', 'b#'])
            source_list = self.b_vars if source_type == 'b$' else self.b_consts
            if not source_list: return self._get_random_arg('b')
            return source_type, random.randrange(len(source_list))

    def generate_random_instruction(self) -> List:
        # Decide if the output is bool or decimal
        if self.b_vars and random.random() < 0.3: # Less frequent boolean ops
            target_type, op_list = 'b$', self.bool_ops
            target_index = random.randrange(len(self.b_vars))
        else:
            if not self.d_vars: return [] # Cannot create if no decimal variables exist
            target_type, op_list = 'd$', self.decimal_ops
            target_index = random.randrange(len(self.d_vars))

        op_name = random.choice(op_list)
        op_props = self.iset.op_properties[op_name]
        
        instr = [target_type, target_index, op_name]
        for arg_type in op_props['arg_types']:
            st, si = self._get_random_arg(arg_type)
            instr.extend([st, si])
        return instr

    def generate_random_algorithm(self, max_len: int = 15) -> List[List]:
        return [self.generate_random_instruction() for _ in range(random.randint(1, max_len))]

    def crossover(self, p1: List[List], p2: List[List]) -> List[List]:
        if not p1 or not p2: return p1 or p2
        point1 = random.randint(0, len(p1))
        point2 = random.randint(0, len(p2))
        return p1[:point1] + p2[point2:]

    def mutate(self, algorithm: List[List]) -> List[List]:
        if not algorithm: return [self.generate_random_instruction()]
        idx = random.randrange(len(algorithm))
        mutation_type = random.random()
        if mutation_type < 0.5: # Mutate instruction
            algorithm[idx] = self.generate_random_instruction()
        elif mutation_type < 0.75 and len(algorithm) > 1: # Delete instruction
            algorithm.pop(idx)
        else: # Add instruction
            algorithm.insert(idx, self.generate_random_instruction())
        return algorithm