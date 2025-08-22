# -*- coding: utf-8 -*-
# evolvo_engine.py

"""
This library, which we'll call **"Evolvo,"** is designed to be the engine for genetic programming or reinforcement learning systems, like the Q-learning agent in your original code.

### Core Concepts of the "Evolvo" Library

1.  **DataStore**: Manages all variables and constants. You can easily register new ones, distinguishing between booleans and decimals, and constants (`#`) versus variables (`$`).
2.  **InstructionSet**: Defines the valid operations (e.g., `ADD`, `SUB`, `IF`). It's designed to be easily extended with new custom functions, including complex mathematical ones like `EXP` or `SIN`.
3.  **Interpreter**: Takes an algorithm (a list of instructions), converts it to efficient bytecode, and executes it using a given `DataStore`.
4.  **AlgorithmGenerator**: This is the interactive "game" you mentioned. It guides the creation of syntactically valid algorithms step-by-step by providing a list of valid "next moves" (tokens) at each point. This prevents the generation of nonsensical code.
5.  **Evaluator**: A flexible component for scoring an algorithm's performance. You can create custom evaluators for different problems (e.g., symbolic regression, optimization) by defining how the algorithm's output should be measured.

### How to Use This Library

1.  **Import the Code**: Add the script as `evolvo_engine.py` to your project.
2.  **Import**: In your main project file (e.g., your Q-learning script), you can now `import evolvo_engine`.
3.  **Configure**:
    *   Define your constants and variables in a `store_config` dictionary. These are the inputs and memory available to your algorithms.
    *   Get the `default_instruction_set` and, if needed, `register()` new custom functions like `SIN`, `COS`, `LOG`, etc.
4.  **Implement an Evaluator**:
    *   Create a class that inherits from `evolvo_engine.BaseEvaluator`.
    *   Implement the `evaluate` method. This is where you define the "fitness function." You will:
        *   Create a `DataStore`.
        *   Set any initial or external values (like the `x` in the example).
        *   Call `self.interpreter.execute(algorithm, data_store)`.
        *   Retrieve the result from the `DataStore` (you decide which variable holds the final answer, e.g., `'y_out'`).
        *   Compare the result to the desired outcome and return a numerical score.
5.  **Generate and Test**:
    *   Use your genetic algorithm or Q-learning model to generate algorithms in the specified list format. The `AlgorithmGenerator` (the `Calculon` class) is the perfect tool for this, as it ensures any generated algorithm is valid.
    *   Pass each generated algorithm to your evaluator instance to get its score.
    *   Use this score to guide the evolution or learning process.
"""

import math
import numpy as np
from collections import defaultdict

# Using a more precise float type by default
myFloat = np.double

class DataStore:
    """
    Manages the state of all constants and variables for an algorithm's execution.
    It separates data into four types: bool constants (b#), decimal constants (d#),
    bool variables (b$), and decimal variables (d$).
    """
    def __init__(self, config):
        """
        Initializes the DataStore based on a configuration dictionary.
        
        Args:
            config (dict): A dictionary defining constants and variables.
                           Example: {
                               'b#': ['false', 'true'],
                               'd#': ['zero', 'one'],
                               'b$': ['is_prime_prediction'],
                               'd$': ['iterator', 'result']
                           }
        """
        self.config = config
        self.stores = {}
        self.store_names = {}
        self.name_to_type_map = {}
        self.initial_values = {}

        for store_type, names in config.items():
            self.stores[store_type] = [None] * len(names)
            self.store_names[store_type] = list(names)
            for name in names:
                self.name_to_type_map[name] = store_type
        
        self.reset()

    def set_initial_value(self, name, value):
        """Sets the default value for a constant or variable upon reset."""
        if name not in self.name_to_type_map:
            raise ValueError(f"'{name}' not defined in store configuration.")
        
        store_type = self.name_to_type_map[name]
        is_bool = store_type.startswith('b')
        
        # Type checking and conversion
        if is_bool:
            value = bool(value)
        elif not isinstance(value, myFloat):
            value = myFloat(value)
            
        self.initial_values[name] = value
        
        # Also set the current value
        self.set(name, value)

    def get_store_type_and_index(self, name):
        """Internal helper to find where a named variable/constant is stored."""
        if name not in self.name_to_type_map:
            raise NameError(f"Store named '{name}' not found.")
        store_type = self.name_to_type_map[name]
        index = self.store_names[store_type].index(name)
        return store_type, index

    def set(self, name, value):
        """Sets the value of a named variable or constant."""
        store_type, index = self.get_store_type_and_index(name)
        self.set_by_location(store_type, index, value)

    def get(self, name):
        """Gets the value of a named variable or constant."""
        store_type, index = self.get_store_type_and_index(name)
        return self.get_by_location(store_type, index)

    def set_by_location(self, store_type, index, value):
        """Sets a value using its direct store type and index."""
        is_bool = store_type.startswith('b')
        
        if is_bool and not isinstance(value, bool):
            value = bool(value)
        elif not is_bool and not isinstance(value, myFloat):
            value = myFloat(value)
            
        if index >= len(self.stores[store_type]):
             # Dynamically extend variable list if a new index is accessed
            if store_type.endswith('$'):
                self.stores[store_type].extend([None] * (index + 1 - len(self.stores[store_type])))
            else:
                raise IndexError(f"Cannot create new constants of type {store_type} at runtime.")

        self.stores[store_type][index] = value

    def get_by_location(self, store_type, index):
        """Gets a value using its direct store type and index."""
        if index >= len(self.stores[store_type]):
            # Return a sensible default if accessing an uninitialized variable
            return False if store_type.startswith('b') else myFloat(0)
        return self.stores[store_type][index]

    def reset(self):
        """Resets all variables to their initial values."""
        for store_type, names in self.store_names.items():
            is_bool = store_type.startswith('b')
            default_val = False if is_bool else myFloat(0)
            self.stores[store_type] = [default_val] * len(names)
        
        for name, value in self.initial_values.items():
            self.set(name, value)

class InstructionSet:
    """
    Defines the set of operations available for building an algorithm.
    This class is designed to be easily extensible.
    """
    def __init__(self):
        self.operations = {}
        self.op_properties = {}
        self.op_types = defaultdict(list)

    def register(self, name, function, arg_types, op_type="decimal"):
        """
        Registers a new operation.

        Args:
            name (str): The name of the operation (e.g., 'ADD').
            function (callable): The function to execute.
            arg_types (list): A list of types for each argument ('b' for bool, 'd' for decimal).
            op_type (str): The general category of the operation ('decimal', 'bool', 'neutral', 'structural').
        """
        self.operations[name] = function
        self.op_properties[name] = {'name': name, 'arg_count': len(arg_types), 'arg_types': arg_types}
        self.op_types[op_type].append(name)
        
    def is_one_arg_op(self, name):
        return self.op_properties[name]['arg_count'] == 1

def get_default_instruction_set():
    """Returns an InstructionSet with a standard set of mathematical and logical operations."""
    iset = InstructionSet()
    
    # Neutral Operations
    iset.register('ASSIGN', lambda a: a, ['any'], op_type='neutral')

    # Decimal Operations
    iset.register('ADD', lambda a, b: a + b, ['d', 'd'], op_type='decimal')
    iset.register('SUB', lambda a, b: a - b, ['d', 'd'], op_type='decimal')
    iset.register('MUL', lambda a, b: a * b, ['d', 'd'], op_type='decimal')
    iset.register('DIV', lambda a, b: a / b if b != 0 else myFloat(0), ['d', 'd'], op_type='decimal')
    iset.register('MOD', lambda a, b: a % b if b != 0 else myFloat(0), ['d', 'd'], op_type='decimal')
    # Example of adding a new function:
    iset.register('EXP', lambda a: myFloat(math.exp(a)) if a < 700 else myFloat('inf'), ['d'], op_type='decimal')

    # Boolean Operations
    iset.register('NOT', lambda a: not a, ['b'], op_type='bool')
    iset.register('OR',  lambda a, b: a or b, ['b', 'b'], op_type='bool')
    iset.register('AND', lambda a, b: a and b, ['b', 'b'], op_type='bool')
    iset.register('CMP', lambda a, b: a == b, ['d', 'd'], op_type='bool')
    iset.register('GT',  lambda a, b: a > b, ['d', 'd'], op_type='bool')
    iset.register('GET', lambda a, b: a >= b, ['d', 'd'], op_type='bool')

    # Structural Operations
    iset.register('IF', None, [], op_type='structural')
    iset.register('END', None, [], op_type='structural')
    
    return iset

class Interpreter:
    """
    Executes an algorithm represented as a list of instructions.
    """
    def __init__(self, instruction_set):
        self.iset = instruction_set

    def _to_bytecode(self, instructions):
        """Converts a list of instructions into a nested bytecode structure."""
        bytecode = []
        context_stack = [bytecode]

        for instr in instructions:
            if not instr: continue
            
            op = instr[0]
            if op == 'IF':
                # Format: {'op': 'IF', 'condition': [type, index], 'body': []}
                if_instr = {'op': 'IF', 'condition': instr[1:3], 'body': []}
                context_stack[-1].append(if_instr)
                context_stack.append(if_instr['body'])
            elif op == 'END':
                if len(context_stack) > 1:
                    context_stack.pop()
            else:
                # Format: {'op': 'ASSIGN', 'target': [type, index], 'source_op': 'ADD', 'args': [[t,i], [t,i]]}
                line_instr = {
                    'op': 'ASSIGN',
                    'target': instr[0:2],
                    'source_op': instr[2],
                    'args': [instr[3:5], instr[5:7]][:self.iset.op_properties[instr[2]]['arg_count']]
                }
                context_stack[-1].append(line_instr)
        return bytecode

    def execute(self, instructions, data_store):
        """
        Executes a set of instructions against a DataStore.

        Args:
            instructions (list): The algorithm to execute.
            data_store (DataStore): The data environment for the execution.
        """
        bytecode = self._to_bytecode(instructions)
        self._execute_bytecode(bytecode, data_store)

    def _execute_bytecode(self, bytecode, data_store):
        """Recursively executes the bytecode structure."""
        for line in bytecode:
            op = line['op']
            
            if op == 'IF':
                condition_val = data_store.get_by_location(*line['condition'])
                if condition_val:
                    self._execute_bytecode(line['body'], data_store)
                continue

            # Handle standard assignment operations
            source_op = line['source_op']
            
            # Resolve arguments
            args = [data_store.get_by_location(*arg) for arg in line['args']]
            
            # Get result from the operation
            op_func = self.iset.operations[source_op]
            result = op_func(*args)
            
            # Assign result to target
            data_store.set_by_location(*line['target'], result)


class BaseEvaluator:
    """
    Abstract base class for evaluating algorithms.
    Subclass this to create custom scoring logic for your specific problem.
    """
    def __init__(self, data_store_config, instruction_set):
        self.data_store_config = data_store_config
        self.instruction_set = instruction_set
        self.interpreter = Interpreter(self.instruction_set)

    def evaluate(self, algorithm, **kwargs):
        """
        This method must be implemented by subclasses.
        It should execute the algorithm and return a score.

        Args:
            algorithm (list): The list of instructions to be evaluated.
            **kwargs: Any external data or parameters needed for evaluation.

        Returns:
            float: A score indicating the algorithm's performance (e.g., lower is better).
        """
        raise NotImplementedError("You must implement the 'evaluate' method.")


### ----------------------------------------------------------------------
### --- EXAMPLE USAGE ----------------------------------------------------
### ----------------------------------------------------------------------

if __name__ == '__main__':
    
    print("--- Evolvo Engine Demo ---")

    # 1. Define the "hardware" of our machine: what variables and constants exist.
    # This configuration is used to initialize the DataStore.
    store_config = {
        # Constants (read-only)
        'd#': ['x', 'one', 'two'], 
        'b#': ['always_true'],
        # Variables (read/write)
        'd$': ['y_out', 'temp_var'],
        'b$': ['decision']
    }

    # 2. Define the instruction set (the "software" or available functions).
    instruction_set = get_default_instruction_set()
    print(f"Available Decimal Ops: {instruction_set.op_types['decimal']}")
    print(f"Available Boolean Ops: {instruction_set.op_types['bool']}")
    
    # 3. Create a custom evaluator for a specific problem.
    # Problem: Evolve an algorithm that calculates: y = (x * 2) + 1
    class SymbolicRegressionEvaluator(BaseEvaluator):
        def evaluate(self, algorithm, target_function, test_points=range(-5, 6)):
            data_store = DataStore(self.data_store_config)
            
            # Set initial constant values
            data_store.set_initial_value('one', 1.0)
            data_store.set_initial_value('two', 2.0)
            data_store.set_initial_value('always_true', True)
            
            total_error = 0.0
            
            for x_val in test_points:
                data_store.reset() # Reset variables for each run
                data_store.set('x', myFloat(x_val)) # Set the input external variable
                
                # Execute the algorithm
                try:
                    self.interpreter.execute(algorithm, data_store)
                except Exception as e:
                    # Penalize algorithms that cause errors
                    return float('inf')

                # Get the result (we decide the output is stored in 'y_out')
                predicted_y = data_store.get('y_out')
                true_y = target_function(x_val)
                
                # Calculate Mean Squared Error
                total_error += (predicted_y - true_y) ** 2
            
            return total_error / len(test_points)

    # 4. Instantiate the evaluator
    evaluator = SymbolicRegressionEvaluator(store_config, instruction_set)

    # 5. Define a hand-coded algorithm to test the evaluator.
    # This algorithm correctly calculates y = (x * 2) + 1
    # Instruction format: [target_type, target_index, operation, arg1_type, arg1_index, arg2_type, arg2_index]
    
    # temp_var = x * 2
    # y_out = temp_var + 1
    perfect_algorithm = [
        ['d$', 1, 'MUL', 'd#', 0, 'd#', 2], # temp_var = x * two
        ['d$', 0, 'ADD', 'd$', 1, 'd#', 1]  # y_out = temp_var + one
    ]

    # Another less perfect algorithm: y = x + 2
    imperfect_algorithm = [
        ['d$', 0, 'ADD', 'd#', 0, 'd#', 2] # y_out = x + two
    ]
    
    # A faulty algorithm that tries to add a boolean
    faulty_algorithm = [
        ['d$', 0, 'ADD', 'd#', 0, 'b#', 0] # This would fail if not for type checks
    ]

    # 6. Evaluate the algorithms
    target_func = lambda x: (x * 2) + 1
    
    score_perfect = evaluator.evaluate(perfect_algorithm, target_function=target_func)
    print(f"\nEvaluating perfect algorithm: y = (x * 2) + 1")
    print(f"  -> Mean Squared Error: {score_perfect:.4f}")

    score_imperfect = evaluator.evaluate(imperfect_algorithm, target_function=target_func)
    print(f"Evaluating imperfect algorithm: y = x + 2")
    print(f"  -> Mean Squared Error: {score_imperfect:.4f}")