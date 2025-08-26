# -*- coding: utf-8 -*-
# evolvo_model_enhanced.py
# Enhanced Genetic PyTorch Model Definition Language
# Advanced neural architecture search with modern ML techniques

"""
Enhanced version with:
1. Advanced shape tracking and validation
2. Modern layer types (Transformer, LayerNorm, etc.)
3. Multi-objective optimization (Pareto frontiers)
4. Architecture constraints and patterns
5. Gradient flow analysis
6. Neural Architecture Search (NAS) techniques
7. Architecture pruning and compression
8. Better skip connection handling
9. Hardware-aware optimization
10. Advanced mutation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import copy
from enum import Enum
from abc import ABC, abstractmethod

# Enhanced shape tracking
@dataclass
class TensorShape:
    """Advanced tensor shape representation"""
    batch: Optional[int] = None  # Batch dimension (usually None/variable)
    channels: Optional[int] = None  # Channel dimension for conv layers
    height: Optional[int] = None  # Height for 2D data
    width: Optional[int] = None  # Width for 2D data
    sequence: Optional[int] = None  # Sequence length for RNNs
    features: Optional[int] = None  # Feature dimension for linear layers

    ### FIX START ###
    # The original check was too simplistic and failed on valid conv->linear transitions.
    def is_compatible_with(self, other: 'TensorShape') -> bool:
        """
        Check if this shape (as an output) can connect to other shape (as an input).
        Handles implicit flattening for spatial-to-feature transitions.
        """
        # Case 1: Both are feature vectors (e.g., Linear -> Linear)
        if self.features is not None and other.features is not None:
            return self.features == other.features

        # Case 2: This is spatial (e.g., Conv output), other is a feature vector (e.g., Linear input)
        # This is the key fix to allow connections that require flattening.
        if self.features is None and other.features is not None:
            return self.get_flat_features() == other.features
        
        # Case 3: This is a feature vector, other is spatial (requires explicit Unflatten/Reshape)
        if self.features is not None and other.features is None:
            return False  # Disallow this implicit transition for safety

        # Case 4: Both are spatial (e.g., Conv -> Conv)
        # The LayerFactory will handle adapting the in_channels, so this check can be permissive.
        if self.channels is not None and other.channels is not None:
            return True

        # Default to true for ambiguous or compatible cases (e.g., connecting to an activation)
        return True
    ### FIX END ###

    def get_flat_features(self) -> int:
        """Get flattened feature count"""
        if self.features:
            return self.features
        if self.channels and self.height and self.width:
            return self.channels * self.height * self.width
        # Handle cases where H/W might be None but C is present
        if self.channels:
            return self.channels * (self.height or 1) * (self.width or 1)
        if self.sequence and self.features:
            return self.sequence * self.features
        return 1

@dataclass
class LayerSpec:
    """Enhanced layer specification with shape tracking"""
    layer_type: str
    params: Dict[str, Any]
    input_shape: Optional[TensorShape] = None
    output_shape: Optional[TensorShape] = None
    computation_cost: float = 0.0  # FLOPs estimate
    memory_cost: float = 0.0  # Memory in MB
    
    def __hash__(self):
        return hash((self.layer_type, json.dumps(self.params, sort_keys=True)))

class ArchitecturePattern(Enum):
    """Common architecture patterns"""
    RESIDUAL = "residual"
    DENSE = "dense"
    INCEPTION = "inception"
    BOTTLENECK = "bottleneck"
    SQUEEZE_EXCITE = "squeeze_excite"
    ATTENTION = "attention"
    MOBILE = "mobile"

class ModelGenome:
    """Enhanced genome with advanced features"""
    def __init__(self, input_shape: TensorShape, output_shape: TensorShape, 
                 constraints: Optional[Dict] = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers: List[LayerSpec] = []
        self.connections: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # (dest, merge_type)
        self.patterns: List[ArchitecturePattern] = []  # Architecture patterns used
        self.constraints = constraints or {}
        
        # Performance metrics
        self.fitness: Optional[float] = None
        self.accuracy: Optional[float] = None
        self.latency: Optional[float] = None
        self.memory_usage: Optional[float] = None
        self.flops: Optional[float] = None
        
        # Genetic metadata
        self.generation: int = 0
        self.parents: List[str] = []
        self.mutations: List[str] = []
        self._signature: Optional[str] = None
    
    def add_layer(self, layer_spec: LayerSpec):
        """Add layer with shape validation"""
        if self.layers and layer_spec.input_shape:
            # Validate shape compatibility
            prev_output = self.layers[-1].output_shape
            # The input_shape of the new layer spec is the same as the prev_output
            # This check is a sanity check. The more robust check is now in is_compatible_with
            if prev_output and not prev_output.is_compatible_with(layer_spec.input_shape):
                # This error should no longer be raised due to the fixes, but is kept as a safeguard.
                raise ValueError(f"Shape mismatch: {prev_output} -> {layer_spec.input_shape}")
        
        self.layers.append(layer_spec)
        self._signature = None
    
    def add_skip_connection(self, source: int, dest: int, merge_type: str = "add"):
        """Add skip connection with merge type"""
        if source >= dest:
            raise ValueError("Skip connection must go forward")
        self.connections[source].append((dest, merge_type))
        self._signature = None
    
    def get_signature(self) -> str:
        """Generate unique signature"""
        if self._signature is None:
            sig_parts = []
            for layer in self.layers:
                sig_parts.append(f"{layer.layer_type}:{json.dumps(layer.params, sort_keys=True)}")
            for src, dests in sorted(self.connections.items()):
                for dest, merge in dests:
                    sig_parts.append(f"conn:{src}->{dest}:{merge}")
            self._signature = hashlib.md5('|'.join(sig_parts).encode()).hexdigest()
        return self._signature
    
    def estimate_complexity(self) -> Dict[str, float]:
        """Estimate computational complexity"""
        total_flops = 0
        total_params = 0
        total_memory = 0
        
        for layer in self.layers:
            total_flops += layer.computation_cost
            total_memory += layer.memory_cost
            
            # Estimate parameters
            if layer.layer_type == 'linear':
                in_f = layer.params.get('in_features', 1)
                out_f = layer.params.get('out_features', 1)
                total_params += in_f * out_f
                if layer.params.get('bias', True):
                    total_params += out_f
            elif layer.layer_type == 'conv2d':
                in_c = layer.params.get('in_channels', 1)
                out_c = layer.params.get('out_channels', 1)
                k = layer.params.get('kernel_size', 3)
                if isinstance(k, (list, tuple)):
                    k = k[0] * k[1]
                else:
                    k = k * k
                total_params += in_c * out_c * k
                if layer.params.get('bias', True):
                    total_params += out_c
        
        self.flops = total_flops
        self.memory_usage = total_memory
        
        return {
            'flops': total_flops,
            'params': total_params,
            'memory_mb': total_memory
        }
    
    def to_pytorch_model(self) -> nn.Module:
        """Convert to PyTorch model"""
        return AdvancedDynamicModel(self)

class LayerFactory:
    """Enhanced factory with modern layer types"""
    
    LAYER_TEMPLATES = {
        # Basic layers
        'linear': {
            'params': {
                'out_features': [8, 16, 32, 64, 128, 256, 512, 1024],
                'bias': [True, False]
            },
            'requires_input_features': True
        },
        
        # Convolutional layers
        'conv2d': {
            'params': {
                'out_channels': [8, 16, 32, 64, 128, 256],
                'kernel_size': [1, 3, 5, 7],
                'stride': [1, 2],
                'padding': ['same', 'valid', 0, 1, 2, 3],
                'groups': [1, 2, 4, 8],  # For grouped/depthwise convolutions
                'dilation': [1, 2],
                'bias': [True, False]
            },
            'requires_input_channels': True
        },
        
        'conv1d': {
            'params': {
                'out_channels': [8, 16, 32, 64, 128],
                'kernel_size': [1, 3, 5, 7, 9],
                'stride': [1, 2],
                'padding': ['same', 'valid', 0, 1, 2],
                'bias': [True, False]
            },
            'requires_input_channels': True
        },
        
        # Modern normalization layers
        'batchnorm1d': {
            'params': {'momentum': [0.1, 0.01], 'eps': [1e-5, 1e-3]},
            'requires_num_features': True
        },
        
        'batchnorm2d': {
            'params': {'momentum': [0.1, 0.01], 'eps': [1e-5, 1e-3]},
            'requires_num_features': True
        },
        
        'layernorm': {
            'params': {'eps': [1e-5, 1e-6]},
            'requires_normalized_shape': True
        },
        
        'groupnorm': {
            'params': {'num_groups': [1, 2, 4, 8, 16, 32]},
            'requires_num_channels': True
        },
        
        'instancenorm2d': {
            'params': {'eps': [1e-5], 'momentum': [0.1]},
            'requires_num_features': True
        },
        
        # Recurrent layers
        'lstm': {
            'params': {
                'hidden_size': [16, 32, 64, 128, 256, 512],
                'num_layers': [1, 2, 3, 4],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
                'bidirectional': [True, False]
            },
            'requires_input_size': True
        },
        
        'gru': {
            'params': {
                'hidden_size': [16, 32, 64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.0, 0.1, 0.2, 0.3],
                'bidirectional': [True, False]
            },
            'requires_input_size': True
        },
        
        # Attention mechanisms
        'multihead_attention': {
            'params': {
                'embed_dim': [64, 128, 256, 512],
                'num_heads': [1, 2, 4, 8, 16],
                'dropout': [0.0, 0.1, 0.2],
                'batch_first': [True]
            },
            'requires_embed_dim': True
        },
        
        'self_attention': {
            'params': {
                'embed_dim': [64, 128, 256, 512],
                'num_heads': [1, 2, 4, 8]
            },
            'requires_embed_dim': True
        },
        
        # Transformer blocks
        'transformer_encoder': {
            'params': {
                'd_model': [128, 256, 512],
                'nhead': [2, 4, 8],
                'num_layers': [1, 2, 3, 4, 6],
                'dim_feedforward': [512, 1024, 2048],
                'dropout': [0.0, 0.1, 0.2]
            },
            'requires_d_model': True
        },
        
        # Activation functions
        'activation': {
            'params': {
                'type': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu', 
                        'swish', 'mish', 'hardswish', 'tanh', 'sigmoid', 
                        'softplus', 'softsign']
            }
        },
        
        # Pooling layers
        'pooling': {
            'params': {
                'type': ['max2d', 'avg2d', 'adaptive_avg2d', 'adaptive_max2d',
                        'max1d', 'avg1d', 'global_avg', 'global_max'],
                'kernel_size': [2, 3, 4],
                'stride': [1, 2, 3]
            }
        },
        
        # Dropout variants
        'dropout': {
            'params': {
                'p': [0.1, 0.2, 0.3, 0.4, 0.5],
                'type': ['standard', 'spatial', 'alpha']
            }
        },
        
        # Modern architectural components
        'squeeze_excite': {
            'params': {
                'reduction': [4, 8, 16]
            },
            'requires_channels': True
        },
        
        'residual_block': {
            'params': {
                'channels': [16, 32, 64, 128],
                'stride': [1, 2]
            },
            'requires_input_channels': True
        },
        
        'mobile_block': {  # MobileNet-style block
            'params': {
                'expansion': [1, 2, 4, 6],
                'out_channels': [16, 32, 64, 128],
                'stride': [1, 2]
            },
            'requires_input_channels': True
        }
    }
    
    @classmethod
    def create_layer(cls, layer_type: str, prev_shape: Optional[TensorShape] = None,
                    specific_params: Optional[Dict] = None) -> LayerSpec:
        """Create layer with specific or random parameters"""
        if layer_type not in cls.LAYER_TEMPLATES:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        template = cls.LAYER_TEMPLATES[layer_type]
        params = specific_params or {}
        
        # Fill in random parameters for missing ones
        for param_name, param_options in template.get('params', {}).items():
            if param_name not in params:
                params[param_name] = random.choice(param_options)
        
        # Handle shape-dependent parameters
        if prev_shape:
            params = cls._adapt_params_to_shape(layer_type, params, prev_shape, template)
        
        # Calculate output shape
        output_shape = cls._calculate_output_shape(layer_type, params, prev_shape)
        
        # Estimate computational cost
        computation_cost = cls._estimate_computation_cost(layer_type, params, prev_shape)
        memory_cost = cls._estimate_memory_cost(layer_type, params)
        
        return LayerSpec(
            layer_type=layer_type,
            params=params,
            input_shape=prev_shape,
            output_shape=output_shape,
            computation_cost=computation_cost,
            memory_cost=memory_cost
        )
    
    @classmethod
    def _adapt_params_to_shape(cls, layer_type: str, params: Dict, 
                               shape: TensorShape, template: Dict) -> Dict:
        """Adapt parameters to input shape"""
        params = params.copy()
        
        ### FIX START ###
        # This logic now correctly infers in_features from a spatial shape.
        if template.get('requires_input_features'):
            # This layer needs a 1D feature vector. Flatten if necessary.
            if shape.features is None:
                params['in_features'] = shape.get_flat_features()
            else:
                params['in_features'] = shape.features
        ### FIX END ###

        if template.get('requires_input_channels') and shape.channels:
            params['in_channels'] = shape.channels
        if template.get('requires_num_features'):
            if shape.channels:
                params['num_features'] = shape.channels
            elif shape.features:
                params['num_features'] = shape.features
        if template.get('requires_input_size') and shape.features:
            params['input_size'] = shape.features
        if template.get('requires_embed_dim') and shape.features:
            params['embed_dim'] = shape.features
        if template.get('requires_d_model') and shape.features:
            params['d_model'] = shape.features
        if template.get('requires_num_channels') and shape.channels:
            params['num_channels'] = shape.channels
        if template.get('requires_normalized_shape'):
            if shape.features:
                params['normalized_shape'] = [shape.features]
            elif shape.channels and shape.height and shape.width:
                params['normalized_shape'] = [shape.channels, shape.height, shape.width]
        
        return params
    
    @classmethod
    def _calculate_output_shape(cls, layer_type: str, params: Dict,
                               input_shape: Optional[TensorShape]) -> TensorShape:
        """Calculate output shape for a layer"""
        if not input_shape:
            return TensorShape()

        ### FIX START ###
        # Refactored to create clean shape objects from scratch, preventing inconsistent states
        # where a shape has both `features` and `channels`.
        
        # Start with a clean slate, only copying batch dim
        output = TensorShape(batch=input_shape.batch)
        
        if layer_type == 'linear':
            # This is a feature-producing layer
            output.features = params.get('out_features', 1)

        elif layer_type in ['conv2d', 'residual_block', 'mobile_block']:
            # These are spatial-producing layers
            output.channels = params.get('out_channels', params.get('channels', 1))
            output.height = input_shape.height
            output.width = input_shape.width
            stride = params.get('stride', 1)
            if isinstance(stride, (list, tuple)): stride = stride[0]
            if output.height: output.height //= stride
            if output.width: output.width //= stride
        
        elif layer_type in ['lstm', 'gru']:
            # These operate on sequences and features
            output.sequence = input_shape.sequence
            output.features = params.get('hidden_size', 1)
            if params.get('bidirectional', False):
                output.features *= 2

        elif layer_type in ['multihead_attention', 'self_attention', 'transformer_encoder']:
             output.sequence = input_shape.sequence
             output.features = params.get('embed_dim', params.get('d_model', input_shape.features))

        elif layer_type == 'pooling':
            # Pooling preserves channels but changes spatial dims
            output.channels = input_shape.channels
            output.height = input_shape.height
            output.width = input_shape.width
            pool_type = params.get('type', 'max2d')
            if 'global' in pool_type or 'adaptive' in pool_type:
                output.height = 1
                output.width = 1
            elif '2d' in pool_type:
                stride = params.get('stride', params.get('kernel_size', 2))
                if output.height: output.height //= stride
                if output.width: output.width //= stride
        
        elif layer_type in ['batchnorm1d', 'batchnorm2d', 'layernorm', 'groupnorm',
                            'instancenorm2d', 'activation', 'dropout', 'squeeze_excite']:
            # These layers preserve shape, so a deepcopy is safe and correct.
            return copy.deepcopy(input_shape)

        else:
            # Default for unknown layers is to preserve shape
            return copy.deepcopy(input_shape)

        return output
        ### FIX END ###

    @classmethod
    def _estimate_computation_cost(cls, layer_type: str, params: Dict,
                                  input_shape: Optional[TensorShape]) -> float:
        """Estimate FLOPs for a layer"""
        if not input_shape:
            return 0.0
        
        flops = 0.0
        
        if layer_type == 'linear':
            in_f = params.get('in_features', 1)
            out_f = params.get('out_features', 1)
            flops = 2 * in_f * out_f  # Multiply-accumulate operations
        
        elif layer_type == 'conv2d':
            in_c = params.get('in_channels', 1)
            out_c = params.get('out_channels', 1)
            k = params.get('kernel_size', 3)
            if isinstance(k, (list, tuple)):
                k = k[0] * k[1]
            else:
                k = k * k
            
            h = input_shape.height or 32
            w = input_shape.width or 32
            stride = params.get('stride', 1)
            if isinstance(stride, (list, tuple)):
                stride = stride[0]
            
            out_h = h // stride
            out_w = w // stride
            
            flops = 2 * in_c * out_c * k * out_h * out_w
        
        return flops / 1e6  # Convert to MFLOPs
    
    @classmethod
    def _estimate_memory_cost(cls, layer_type: str, params: Dict) -> float:
        """Estimate memory usage in MB"""
        memory = 0.0
        
        if layer_type == 'linear':
            in_f = params.get('in_features', 1)
            out_f = params.get('out_features', 1)
            memory = (in_f * out_f + out_f) * 4  # 4 bytes per float32
        
        elif layer_type == 'conv2d':
            in_c = params.get('in_channels', 1)
            out_c = params.get('out_channels', 1)
            k = params.get('kernel_size', 3)
            if isinstance(k, (list, tuple)):
                k = k[0] * k[1]
            else:
                k = k * k
            memory = (in_c * out_c * k + out_c) * 4
        
        return memory / 1e6  # Convert to MB

class ArchitectureConstraints:
    """Defines constraints for valid architectures"""
    
    def __init__(self):
        self.max_depth = 100
        self.max_parameters = 1e9  # 1 billion parameters
        self.max_memory_mb = 8000  # 8GB
        self.max_flops = 1e12  # 1 TFLOPs
        self.required_patterns = []
        self.forbidden_sequences = []
        self.layer_limits = {}
    
    def add_layer_limit(self, layer_type: str, max_count: int):
        """Limit the number of specific layer types"""
        self.layer_limits[layer_type] = max_count
    
    def add_required_pattern(self, pattern: ArchitecturePattern):
        """Require certain architectural patterns"""
        self.required_patterns.append(pattern)
    
    def validate(self, genome: ModelGenome) -> Tuple[bool, List[str]]:
        """Validate a genome against constraints"""
        violations = []
        
        # Check depth
        if len(genome.layers) > self.max_depth:
            violations.append(f"Exceeds max depth: {len(genome.layers)} > {self.max_depth}")
        
        # Check complexity
        complexity = genome.estimate_complexity()
        if complexity['params'] > self.max_parameters:
            violations.append(f"Too many parameters: {complexity['params']:.2e} > {self.max_parameters:.2e}")
        if complexity['memory_mb'] > self.max_memory_mb:
            violations.append(f"Too much memory: {complexity['memory_mb']:.1f}MB > {self.max_memory_mb}MB")
        if complexity['flops'] > self.max_flops:
            violations.append(f"Too many FLOPs: {complexity['flops']:.2e} > {self.max_flops:.2e}")
        
        # Check layer limits
        layer_counts = defaultdict(int)
        for layer in genome.layers:
            layer_counts[layer.layer_type] += 1
        
        for layer_type, max_count in self.layer_limits.items():
            if layer_counts[layer_type] > max_count:
                violations.append(f"Too many {layer_type} layers: {layer_counts[layer_type]} > {max_count}")
        
        # Check required patterns
        for pattern in self.required_patterns:
            if pattern not in genome.patterns:
                violations.append(f"Missing required pattern: {pattern.value}")
        
        return len(violations) == 0, violations

class GradientFlowAnalyzer:
    """Analyzes gradient flow through architecture"""
    
    @staticmethod
    def analyze(genome: ModelGenome) -> Dict[str, Any]:
        """Analyze potential gradient flow issues"""
        analysis = {
            'max_depth': len(genome.layers),
            'skip_connections': len(genome.connections),
            'potential_vanishing': False,
            'potential_exploding': False,
            'bottlenecks': [],
            'dead_ends': []
        }
        
        # Check for vanishing gradients
        activation_count = sum(1 for l in genome.layers if l.layer_type == 'activation')
        if activation_count > 10:
            has_norm = any(l.layer_type in ['batchnorm1d', 'batchnorm2d', 'layernorm', 'groupnorm'] 
                          for l in genome.layers)
            has_skip = len(genome.connections) > 0
            
            if not has_norm and not has_skip:
                analysis['potential_vanishing'] = True
        
        # Check for exploding gradients
        conv_count = sum(1 for l in genome.layers if 'conv' in l.layer_type)
        if conv_count > 20 and len(genome.connections) < conv_count // 4:
            analysis['potential_exploding'] = True
        
        # Find bottlenecks (sudden feature reduction)
        for i in range(1, len(genome.layers)):
            prev_shape = genome.layers[i-1].output_shape
            curr_shape = genome.layers[i].output_shape
            
            if prev_shape and curr_shape:
                prev_features = prev_shape.get_flat_features()
                curr_features = curr_shape.get_flat_features()
                
                if curr_features > 0 and prev_features > 0 and curr_features < prev_features * 0.1:  # 90% reduction
                    analysis['bottlenecks'].append(i)
        
        return analysis

class AdvancedDynamicModel(nn.Module):
    """Enhanced dynamic model with advanced features"""
    
    def __init__(self, genome: ModelGenome):
        super().__init__()
        self.genome = genome
        self.layers = nn.ModuleList()
        self.skip_connections = genome.connections
        self.build_layers()
    
    def build_layers(self):
        """Build PyTorch layers from genome"""
        for i, layer_spec in enumerate(self.genome.layers):
            layer = self._create_layer(layer_spec)
            if layer:
                self.layers.append(layer)
    
    def _create_layer(self, spec: LayerSpec) -> Optional[nn.Module]:
        """Create PyTorch layer from specification"""
        lt = spec.layer_type
        p = spec.params
        
        # Basic layers
        if lt == 'linear':
            return nn.Linear(p.get('in_features', 128), p['out_features'], p.get('bias', True))
        
        # Convolutional layers
        elif lt == 'conv2d':
            return nn.Conv2d(
                p.get('in_channels', 3), p['out_channels'],
                p['kernel_size'], p.get('stride', 1),
                p.get('padding', 0), 
                dilation=p.get('dilation', 1),
                groups=p.get('groups', 1),
                bias=p.get('bias', True)
            )
        
        elif lt == 'conv1d':
            return nn.Conv1d(
                p.get('in_channels', 3), p['out_channels'],
                p['kernel_size'], p.get('stride', 1),
                p.get('padding', 0), bias=p.get('bias', True)
            )
        
        # Normalization layers
        elif lt == 'batchnorm1d':
            return nn.BatchNorm1d(p['num_features'], momentum=p.get('momentum', 0.1),
                                eps=p.get('eps', 1e-5))
        elif lt == 'batchnorm2d':
            return nn.BatchNorm2d(p['num_features'], momentum=p.get('momentum', 0.1),
                                eps=p.get('eps', 1e-5))
        elif lt == 'layernorm':
            return nn.LayerNorm(p['normalized_shape'], eps=p.get('eps', 1e-5))
        elif lt == 'groupnorm':
            return nn.GroupNorm(p['num_groups'], p['num_channels'], eps=p.get('eps', 1e-5))
        elif lt == 'instancenorm2d':
            return nn.InstanceNorm2d(p['num_features'], eps=p.get('eps', 1e-5),
                                    momentum=p.get('momentum', 0.1))
        
        # Recurrent layers
        elif lt == 'lstm':
            return nn.LSTM(
                p.get('input_size', 128), p['hidden_size'],
                p.get('num_layers', 1), dropout=p.get('dropout', 0),
                bidirectional=p.get('bidirectional', False),
                batch_first=True
            )
        elif lt == 'gru':
            return nn.GRU(
                p.get('input_size', 128), p['hidden_size'],
                p.get('num_layers', 1), dropout=p.get('dropout', 0),
                bidirectional=p.get('bidirectional', False),
                batch_first=True
            )
        
        # Attention layers
        elif lt == 'multihead_attention':
            return nn.MultiheadAttention(
                p['embed_dim'], p['num_heads'],
                dropout=p.get('dropout', 0), 
                batch_first=p.get('batch_first', True)
            )
        elif lt == 'self_attention':
            return SelfAttention(p['embed_dim'], p['num_heads'])
        
        # Transformer
        elif lt == 'transformer_encoder':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=p['d_model'],
                nhead=p['nhead'],
                dim_feedforward=p.get('dim_feedforward', 2048),
                dropout=p.get('dropout', 0.1),
                batch_first=True
            )
            return nn.TransformerEncoder(encoder_layer, p.get('num_layers', 1))
        
        # Activation functions
        elif lt == 'activation':
            act_type = p['type']
            if act_type == 'relu': return nn.ReLU()
            elif act_type == 'leaky_relu': return nn.LeakyReLU(0.01)
            elif act_type == 'elu': return nn.ELU()
            elif act_type == 'selu': return nn.SELU()
            elif act_type == 'gelu': return nn.GELU()
            elif act_type == 'swish': return Swish()
            elif act_type == 'mish': return Mish()
            elif act_type == 'hardswish': return nn.Hardswish()
            elif act_type == 'tanh': return nn.Tanh()
            elif act_type == 'sigmoid': return nn.Sigmoid()
            elif act_type == 'softplus': return nn.Softplus()
            elif act_type == 'softsign': return nn.Softsign()
        
        # Pooling layers
        elif lt == 'pooling':
            pool_type = p['type']
            k = p.get('kernel_size', 2)
            s = p.get('stride', 2)
            
            if pool_type == 'max2d': return nn.MaxPool2d(k, s)
            elif pool_type == 'avg2d': return nn.AvgPool2d(k, s)
            elif pool_type == 'adaptive_avg2d': return nn.AdaptiveAvgPool2d((1, 1))
            elif pool_type == 'adaptive_max2d': return nn.AdaptiveMaxPool2d((1, 1))
            elif pool_type == 'max1d': return nn.MaxPool1d(k, s)
            elif pool_type == 'avg1d': return nn.AvgPool1d(k, s)
            elif pool_type == 'global_avg': return GlobalAvgPool()
            elif pool_type == 'global_max': return GlobalMaxPool()
        
        # Dropout
        elif lt == 'dropout':
            dropout_type = p.get('type', 'standard')
            if dropout_type == 'standard':
                return nn.Dropout(p['p'])
            elif dropout_type == 'spatial':
                return nn.Dropout2d(p['p'])
            elif dropout_type == 'alpha':
                return nn.AlphaDropout(p['p'])
        
        # Modern architectural components
        elif lt == 'squeeze_excite':
            return SqueezeExcite(p.get('channels', 64), p.get('reduction', 4))
        elif lt == 'residual_block':
            return ResidualBlock(p.get('in_channels', 64), p.get('channels', 64), 
                                p.get('stride', 1))
        elif lt == 'mobile_block':
            return MobileBlock(p.get('in_channels', 32), p.get('out_channels', 64),
                             p.get('expansion', 4), p.get('stride', 1))
        
        return None
    
    def forward(self, x):
        """Forward pass with advanced skip connections"""
        
        # Move input to same device as model
        x = x.to(next(self.parameters()).device)
        outputs = {}
        current_tensor = x

        for i, layer_spec in enumerate(self.genome.layers):
            layer = self.layers[i]
            
            # Check for incoming skip connections to this layer
            # (This is more complex, for now we handle outgoing skips)

            # Apply layer
            # Handle implicit flatten
            if isinstance(layer, nn.Linear) and len(current_tensor.shape) > 2:
                current_tensor = torch.flatten(current_tensor, 1)

            if isinstance(layer, (nn.LSTM, nn.GRU)):
                current_tensor, _ = layer(current_tensor)
            elif isinstance(layer, nn.MultiheadAttention):
                current_tensor, _ = layer(current_tensor, current_tensor, current_tensor)
            else:
                current_tensor = layer(current_tensor)
            
            outputs[i] = current_tensor
            
            # Handle outgoing skip connections from this layer
            if i in self.skip_connections:
                for dest, merge_type in self.skip_connections[i]:
                    if dest < len(self.genome.layers) and dest in outputs:
                        # The destination tensor already exists, so we merge into it
                        outputs[dest] = self._merge_tensors(outputs[dest], current_tensor, merge_type)
        
        return outputs.get(len(self.layers) - 1, x)

    def _merge_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor, 
                      merge_type: str) -> torch.Tensor:
        """Merge two tensors with shape matching"""
        # Ensure compatible shapes
        if tensor1.shape != tensor2.shape:
            # Try to match shapes
            s1 = tensor1.shape
            s2 = tensor2.shape
            
            # Try to pad/pool to match spatial dimensions
            if len(s1) == 4 and len(s2) == 4 and (s1[2:] != s2[2:]):
                target_h, target_w = s1[2], s1[3]
                adaptive_pool = nn.AdaptiveAvgPool2d((target_h, target_w)).to(tensor2.device)
                tensor2_adapted = adaptive_pool(tensor2)
            else:
                tensor2_adapted = tensor2

            # Use 1x1 conv or linear to match channels/features
            if s1[1] != tensor2_adapted.shape[1]:
                if len(s1) == 4:  # Conv features
                    adapter = nn.Conv2d(tensor2_adapted.shape[1], s1[1], 1).to(tensor2.device)
                else:  # Linear features
                    adapter = nn.Linear(tensor2_adapted.shape[-1], s1[-1]).to(tensor2.device)
                tensor2_adapted = adapter(tensor2_adapted)
            tensor2 = tensor2_adapted
        
        if merge_type == 'add':
            return tensor1 + tensor2
        elif merge_type == 'concat':
            return torch.cat([tensor1, tensor2], dim=1) # Concat on channel/feature dim
        elif merge_type == 'mul':
            return tensor1 * tensor2
        elif merge_type == 'max':
            return torch.max(tensor1, tensor2)
        else:
            return tensor1 + tensor2  # Default to add

# Custom layers and modules
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if len(x.shape) == 4:
            x = self.pool(x)
            return torch.flatten(x, 1)
        elif len(x.shape) == 3: # (batch, seq, features)
            return x.mean(dim=1)
        else:
            return x

class GlobalMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        if len(x.shape) == 4:
            x = self.pool(x)
            return torch.flatten(x, 1)
        elif len(x.shape) == 3:
            return x.max(dim=1)[0]
        else:
            return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        return self.attention(x, x, x)[0]

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        super().__init__()
        hidden = in_channels * expansion
        
        self.conv1 = nn.Conv2d(in_channels, hidden, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.use_residual = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.use_residual:
            return x + out
        return out

class MultiObjectiveEvolver:
    """Multi-objective optimization using Pareto frontiers"""
    
    def __init__(self, objectives: List[str] = ['accuracy', 'latency', 'memory']):
        self.objectives = objectives
        self.pareto_front = []
    
    def dominates(self, genome1: ModelGenome, genome2: ModelGenome) -> bool:
        """Check if genome1 dominates genome2"""
        better_in_one = False
        for obj in self.objectives:
            val1 = getattr(genome1, obj, 0)
            val2 = getattr(genome2, obj, 0)
            
            if obj in ['latency', 'memory_usage', 'flops']:  # Minimize
                if val1 > val2:
                    return False
                if val1 < val2:
                    better_in_one = True
            else:  # Maximize
                if val1 < val2:
                    return False
                if val1 > val2:
                    better_in_one = True
        
        return better_in_one
    
    def update_pareto_front(self, population: List[ModelGenome]):
        """Update Pareto front with non-dominated solutions"""
        self.pareto_front = []
        
        for genome in population:
            is_dominated = False
            for other in population:
                if other != genome and self.dominates(other, genome):
                    is_dominated = True
                    break
            
            if not is_dominated:
                self.pareto_front.append(genome)
    
    def crowding_distance(self, population: List[ModelGenome]) -> Dict[ModelGenome, float]:
        """Calculate crowding distance for diversity preservation"""
        distances = {g: 0.0 for g in population}
        n = len(population)
        
        if n <= 2:
            for g in population:
                distances[g] = float('inf')
            return distances
        
        for obj in self.objectives:
            # Sort by objective
            sorted_pop = sorted(population, key=lambda g: getattr(g, obj, 0))
            
            # Boundary points get infinite distance
            distances[sorted_pop[0]] = float('inf')
            distances[sorted_pop[-1]] = float('inf')
            
            # Calculate distances for interior points
            obj_range = getattr(sorted_pop[-1], obj, 0) - getattr(sorted_pop[0], obj, 0)
            if obj_range > 0:
                for i in range(1, n-1):
                    dist = (getattr(sorted_pop[i+1], obj, 0) - 
                           getattr(sorted_pop[i-1], obj, 0)) / obj_range
                    distances[sorted_pop[i]] += dist
        
        return distances

class AdvancedModelEvolver:
    """Enhanced model evolver with advanced strategies"""
    
    def __init__(self, input_shape: TensorShape, output_shape: TensorShape,
                 task_type: str = 'classification', constraints: Optional[ArchitectureConstraints] = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.task_type = task_type
        self.constraints = constraints or ArchitectureConstraints()
        self.population: List[ModelGenome] = []
        self.generation = 0
        self.diversity_cache = set()
        self.hall_of_fame = []  # Best architectures ever found
        self.multi_objective = MultiObjectiveEvolver()
        self.gradient_analyzer = GradientFlowAnalyzer()
        
        # Advanced mutation strategies
        self.mutation_strategies = {
            'layer_mutation': 0.3,
            'connection_mutation': 0.2,
            'pattern_insertion': 0.2,
            'pruning': 0.15,
            'regularization': 0.15
        }
    
    def generate_population(self, size: int) -> List[ModelGenome]:
        """Generate diverse initial population"""
        population = []
        strategies = ['random', 'residual', 'dense', 'mobile', 'transformer']
        
        while len(population) < size:
            try:
                strategy = strategies[len(population) % len(strategies)]
                genome = self.generate_architecture(strategy)
                
                if self.is_novel_architecture(genome):
                    population.append(genome)
            except Exception as e:
                print(f"Warning: Failed to generate a valid architecture with strategy '{strategy}'. Error: {e}")
                # Try next strategy
                continue
        
        return population
    
    def generate_architecture(self, strategy: str) -> ModelGenome:
        """Generate architecture using specific strategy"""
        genome = ModelGenome(self.input_shape, self.output_shape, self.constraints.__dict__)
        
        if strategy == 'residual':
            genome = self._generate_residual_architecture(genome)
        elif strategy == 'dense':
            genome = self._generate_dense_architecture(genome)
        elif strategy == 'mobile':
            genome = self._generate_mobile_architecture(genome)
        elif strategy == 'transformer':
            genome = self._generate_transformer_architecture(genome)
        else:
            genome = self._generate_random_architecture(genome)
        
        return genome
    
    def _generate_residual_architecture(self, genome: ModelGenome) -> ModelGenome:
        """Generate ResNet-style architecture"""
        genome.patterns.append(ArchitecturePattern.RESIDUAL)
        prev_shape = self.input_shape
        
        # Initial conv
        layer = LayerFactory.create_layer('conv2d', prev_shape, 
                                         {'out_channels': 64, 'kernel_size': 7, 'stride': 2})
        genome.add_layer(layer)
        prev_shape = layer.output_shape
        
        # Residual blocks
        for i in range(random.randint(3, 8)):
            layer = LayerFactory.create_layer('residual_block', prev_shape)
            genome.add_layer(layer)
            
            # Add skip connection every 2 blocks
            if i > 0 and i % 2 == 0:
                genome.add_skip_connection(len(genome.layers)-3, len(genome.layers)-1)
            
            prev_shape = layer.output_shape
        
        ### FIX START ###
        # This section is now much cleaner. It relies on the robust LayerFactory to handle
        # the transition from the pooling layer's spatial output to the linear layer's feature input.
        
        # Global pooling
        pool_layer = LayerFactory.create_layer('pooling', prev_shape, {'type': 'global_avg'})
        genome.add_layer(pool_layer)
        prev_shape = pool_layer.output_shape

        # Classifier
        # The factory will now automatically infer `in_features` from the flattened `prev_shape`.
        linear_layer = LayerFactory.create_layer('linear', prev_shape,
                                                 {'out_features': self.output_shape.features})
        genome.add_layer(linear_layer)
        ### FIX END ###
        
        return genome
    
    def _generate_dense_architecture(self, genome: ModelGenome) -> ModelGenome:
        """Generate DenseNet-style architecture"""
        genome.patterns.append(ArchitecturePattern.DENSE)
        # Implementation similar to residual but with dense connections
        return self._generate_random_architecture(genome)  # Simplified
    
    def _generate_mobile_architecture(self, genome: ModelGenome) -> ModelGenome:
        """Generate MobileNet-style architecture"""
        genome.patterns.append(ArchitecturePattern.MOBILE)
        prev_shape = self.input_shape
        
        # Mobile blocks
        for i in range(random.randint(5, 12)):
            expansion = random.choice([1, 2, 4, 6])
            out_channels = random.choice([16, 32, 64, 128])
            stride = 2 if i % 3 == 0 else 1
            
            layer = LayerFactory.create_layer('mobile_block', prev_shape,
                                             {'expansion': expansion, 
                                              'out_channels': out_channels,
                                              'stride': stride})
            genome.add_layer(layer)
            prev_shape = layer.output_shape
        
        # Final layers
        pool_layer = LayerFactory.create_layer('pooling', prev_shape, {'type': 'global_avg'})
        genome.add_layer(pool_layer)
        prev_shape = pool_layer.output_shape
        linear_layer = LayerFactory.create_layer('linear', prev_shape,
                                                 {'out_features': self.output_shape.features})
        genome.add_layer(linear_layer)

        return genome
    
    def _generate_transformer_architecture(self, genome: ModelGenome) -> ModelGenome:
        """Generate Transformer-style architecture"""
        genome.patterns.append(ArchitecturePattern.ATTENTION)
        prev_shape = self.input_shape
        
        # Transformer encoder
        layer = LayerFactory.create_layer('transformer_encoder', prev_shape,
                                         {'d_model': 256, 'nhead': 8, 'num_layers': 4})
        genome.add_layer(layer)
        prev_shape = layer.output_shape

        # Final layers
        pool_layer = LayerFactory.create_layer('pooling', prev_shape, {'type': 'global_avg'})
        genome.add_layer(pool_layer)
        prev_shape = pool_layer.output_shape
        linear_layer = LayerFactory.create_layer('linear', prev_shape,
                                                 {'out_features': self.output_shape.features})
        genome.add_layer(linear_layer)
        
        return genome
    
    def _generate_random_architecture(self, genome: ModelGenome) -> ModelGenome:
        """Generate random architecture"""
        prev_shape = self.input_shape
        num_layers = random.randint(5, 20)
        
        for _ in range(num_layers):
            layer_type = self._choose_layer_type(prev_shape)
            layer = LayerFactory.create_layer(layer_type, prev_shape)
            genome.add_layer(layer)
            prev_shape = layer.output_shape
            
            # Random skip connections
            if len(genome.layers) > 3 and random.random() < 0.2:
                source = random.randint(0, len(genome.layers)-3)
                genome.add_skip_connection(source, len(genome.layers)-1)

        # Ensure final layer is appropriate
        if prev_shape.features is None: # If last layer was spatial
            pool_layer = LayerFactory.create_layer('pooling', prev_shape, {'type': 'global_avg'})
            genome.add_layer(pool_layer)
            prev_shape = pool_layer.output_shape

        if prev_shape.features != self.output_shape.features:
            linear_layer = LayerFactory.create_layer('linear', prev_shape,
                                                    {'out_features': self.output_shape.features})
            genome.add_layer(linear_layer)

        return genome
    
    def _choose_layer_type(self, current_shape: TensorShape) -> str:
        """Choose layer type based on task and current tensor shape"""
        is_spatial = current_shape.channels is not None
        is_sequence = current_shape.sequence is not None
        
        if self.task_type == 'classification':
            weights = {
                'conv2d': 0.3 if is_spatial else 0.0,
                'linear': 0.2 if not is_spatial else 0.05, # Can add after flattening
                'activation': 0.2,
                'batchnorm2d': 0.1 if is_spatial else 0.0,
                'batchnorm1d': 0.1 if not is_spatial else 0.0,
                'dropout': 0.1,
                'pooling': 0.1 if is_spatial else 0.0,
                'residual_block': 0.1 if is_spatial else 0.0,
            }
        elif self.task_type == 'sequence':
            weights = {
                'lstm': 0.25 if not is_spatial else 0.0,
                'gru': 0.15 if not is_spatial else 0.0,
                'multihead_attention': 0.2,
                'linear': 0.15,
                'activation': 0.15,
                'dropout': 0.1,
                'layernorm': 0.1,
            }
        else:
            weights = {
                'linear': 0.35,
                'activation': 0.25,
                'dropout': 0.15,
                'batchnorm1d': 0.15,
                'layernorm': 0.1
            }
        
        # Filter out impossible choices and normalize
        possible_layers = {k: v for k, v in weights.items() if v > 0}
        total_weight = sum(possible_layers.values())
        if total_weight == 0: return 'linear' # Fallback
        
        return random.choices(
            list(possible_layers.keys()),
            weights=[v / total_weight for v in possible_layers.values()]
        )[0]
    
    def mutate_advanced(self, genome: ModelGenome) -> ModelGenome:
        """Advanced mutation with multiple strategies"""
        mutated = copy.deepcopy(genome)
        mutated.generation = self.generation
        
        # Choose mutation strategy
        strategy = random.choices(
            list(self.mutation_strategies.keys()),
            weights=list(self.mutation_strategies.values())
        )[0]
        
        if strategy == 'layer_mutation':
            mutated = self._mutate_layers(mutated)
        elif strategy == 'connection_mutation':
            mutated = self._mutate_connections(mutated)
        elif strategy == 'pattern_insertion':
            mutated = self._insert_pattern(mutated)
        elif strategy == 'pruning':
            mutated = self._prune_architecture(mutated)
        elif strategy == 'regularization':
            mutated = self._add_regularization(mutated)
        
        mutated.mutations.append(strategy)
        return mutated
    
    def _mutate_layers(self, genome: ModelGenome) -> ModelGenome:
        """Mutate individual layers"""
        if not genome.layers:
            return genome
        
        idx = random.randint(0, len(genome.layers)-1)
        mutation_type = random.choice(['replace', 'modify', 'duplicate', 'add', 'remove'])
        
        if mutation_type == 'replace':
            prev_shape = genome.layers[idx-1].output_shape if idx > 0 else genome.input_shape
            new_layer_type = self._choose_layer_type(prev_shape)
            new_layer = LayerFactory.create_layer(new_layer_type, prev_shape)
            genome.layers[idx] = new_layer
        elif mutation_type == 'modify' and genome.layers[idx].params:
            # Modify a random parameter
            param_name = random.choice(list(genome.layers[idx].params.keys()))
            template = LayerFactory.LAYER_TEMPLATES.get(genome.layers[idx].layer_type, {})
            if param_name in template.get('params', {}):
                genome.layers[idx].params[param_name] = random.choice(
                    template['params'][param_name]
                )
        elif mutation_type == 'duplicate' and idx < len(genome.layers) -1: # Don't duplicate final layer
            genome.layers.insert(idx+1, copy.deepcopy(genome.layers[idx]))
        elif mutation_type == 'add':
            prev_shape = genome.layers[idx].output_shape
            new_layer_type = self._choose_layer_type(prev_shape)
            new_layer = LayerFactory.create_layer(new_layer_type, prev_shape)
            genome.layers.insert(idx + 1, new_layer)
        elif mutation_type == 'remove' and len(genome.layers) > 3:
            genome.layers.pop(idx)

        # After any structural change, we must rebuild subsequent layers to ensure shape compatibility
        self._rebuild_from_index(genome, 0)
        
        return genome

    def _rebuild_from_index(self, genome: ModelGenome, start_index: int):
        """Rebuilds layers from a given index to fix shape inconsistencies after mutation."""
        prev_shape = genome.layers[start_index - 1].output_shape if start_index > 0 else genome.input_shape
        for i in range(start_index, len(genome.layers)):
            spec = genome.layers[i]
            # Create a new layer of the same type and params, but adapted to the new `prev_shape`
            new_spec = LayerFactory.create_layer(spec.layer_type, prev_shape, spec.params)
            genome.layers[i] = new_spec
            prev_shape = new_spec.output_shape
    
    def _mutate_connections(self, genome: ModelGenome) -> ModelGenome:
        """Mutate skip connections"""
        if len(genome.layers) < 3:
            return genome
        
        if genome.connections and random.random() < 0.5:
            # Remove a connection
            source = random.choice(list(genome.connections.keys()))
            genome.connections.pop(source)
        else:
            # Add a connection
            source = random.randint(0, len(genome.layers)-3)
            dest = random.randint(source+2, len(genome.layers)-1)
            merge_type = random.choice(['add', 'concat', 'mul'])
            genome.add_skip_connection(source, dest, merge_type)
        
        return genome
    
    def _insert_pattern(self, genome: ModelGenome) -> ModelGenome:
        """Insert architectural pattern"""
        patterns = [p for p in ArchitecturePattern if p not in genome.patterns]
        if not patterns:
            return genome
        
        pattern = random.choice(patterns)
        insert_pos = random.randint(0, len(genome.layers))
        
        if pattern == ArchitecturePattern.SQUEEZE_EXCITE:
            # Insert SE block
            if genome.layers and genome.layers[0].output_shape.channels:
                se_layer = LayerFactory.create_layer(
                    'squeeze_excite',
                    genome.layers[min(insert_pos, len(genome.layers)-1)].output_shape
                )
                genome.layers.insert(insert_pos, se_layer)
                genome.patterns.append(pattern)
        
        self._rebuild_from_index(genome, insert_pos)
        return genome
    
    def _prune_architecture(self, genome: ModelGenome) -> ModelGenome:
        """Prune unnecessary layers"""
        if len(genome.layers) <= 5:
            return genome
        
        # Analyze gradient flow
        analysis = self.gradient_analyzer.analyze(genome)
        
        # Remove layers that might cause issues
        layers_to_remove = []
        
        # Remove bottlenecks
        layers_to_remove.extend(analysis['bottlenecks'])
        
        # Remove redundant activations
        for i in range(len(genome.layers)-1):
            if (genome.layers[i].layer_type == 'activation' and
                genome.layers[i+1].layer_type == 'activation'):
                layers_to_remove.append(i)
        
        # Remove layers
        rebuilt = False
        for idx in sorted(set(layers_to_remove), reverse=True):
            if len(genome.layers) > 5:  # Keep minimum layers
                genome.layers.pop(idx)
                rebuilt = True
        
        if rebuilt:
             self._rebuild_from_index(genome, 0)
        return genome
    
    def _add_regularization(self, genome: ModelGenome) -> ModelGenome:
        """Add regularization layers"""
        # Find positions after conv/linear layers
        positions = []
        for i, layer in enumerate(genome.layers):
            if layer.layer_type in ['conv2d', 'linear', 'conv1d']:
                positions.append(i+1)
        
        if positions:
            pos = random.choice(positions)
            reg_type = random.choice(['dropout', 'batchnorm2d', 'layernorm'])
            
            if pos < len(genome.layers):
                prev_shape = genome.layers[pos-1].output_shape
            else:
                prev_shape = genome.layers[-1].output_shape
            
            # Ensure chosen regularization is compatible with shape
            if reg_type == 'batchnorm2d' and prev_shape.channels is None:
                reg_type = 'batchnorm1d' if prev_shape.features is not None else 'dropout'

            reg_layer = LayerFactory.create_layer(reg_type, prev_shape)
            genome.layers.insert(pos, reg_layer)
            self._rebuild_from_index(genome, pos)
        
        return genome
    
    def crossover_advanced(self, parent1: ModelGenome, parent2: ModelGenome) -> ModelGenome:
        """Advanced crossover with pattern preservation"""
        child = ModelGenome(self.input_shape, self.output_shape, self.constraints.__dict__)
        child.generation = self.generation
        child.parents = [parent1.get_signature()[:8], parent2.get_signature()[:8]]
        
        # Inherit patterns
        child.patterns = list(set(parent1.patterns + parent2.patterns))
        
        # Choose crossover strategy
        strategy = random.choice(['uniform', 'pattern_based', 'layer_blocks'])
        
        if strategy == 'uniform':
            # Uniform crossover
            max_len = max(len(parent1.layers), len(parent2.layers))
            for i in range(max_len):
                if i < len(parent1.layers) and i < len(parent2.layers):
                    child.layers.append(
                        copy.deepcopy(random.choice([parent1.layers[i], parent2.layers[i]]))
                    )
                elif i < len(parent1.layers):
                    if random.random() < 0.5:
                        child.layers.append(copy.deepcopy(parent1.layers[i]))
                elif i < len(parent2.layers):
                    if random.random() < 0.5:
                        child.layers.append(copy.deepcopy(parent2.layers[i]))
        
        elif strategy == 'pattern_based':
            # Preserve architectural patterns
            # Take pattern blocks from each parent
            if parent1.patterns and random.random() < 0.5:
                child.layers = copy.deepcopy(parent1.layers[:len(parent1.layers)//2])
            else:
                child.layers = copy.deepcopy(parent2.layers[:len(parent2.layers)//2])
            
            if parent2.patterns and random.random() < 0.5:
                child.layers.extend(copy.deepcopy(parent2.layers[len(parent2.layers)//2:]))
            else:
                child.layers.extend(copy.deepcopy(parent1.layers[len(parent1.layers)//2:]))
        
        else:  # layer_blocks
            # Take contiguous blocks from each parent
            block_size = 3
            use_parent1 = True
            i = 0
            
            while i < max(len(parent1.layers), len(parent2.layers)):
                parent = parent1 if use_parent1 else parent2
                if i < len(parent.layers):
                    child.layers.extend(
                        copy.deepcopy(parent.layers[i:i+block_size])
                    )
                i += block_size
                use_parent1 = not use_parent1
        
        # Inherit connections
        for source, dests in parent1.connections.items():
            if source < len(child.layers) - 1:
                for dest, merge_type in dests:
                    if dest < len(child.layers) and random.random() < 0.5:
                        child.add_skip_connection(source, dest, merge_type)
        
        for source, dests in parent2.connections.items():
            if source < len(child.layers) - 1:
                for dest, merge_type in dests:
                    if dest < len(child.layers) and random.random() < 0.5:
                        child.add_skip_connection(source, dest, merge_type)
        
        # Rebuild to ensure validity
        self._rebuild_from_index(child, 0)
        return child
    
    def is_novel_architecture(self, genome: ModelGenome) -> bool:
        """Check if architecture is novel"""
        signature = genome.get_signature()
        if signature in self.diversity_cache:
            return False
        self.diversity_cache.add(signature)
        return True
    
    def evolve(self, generations: int = 100, population_size: int = 50,
              fitness_func: Optional[callable] = None,
              multi_objective: bool = True) -> List[ModelGenome]:
        """Main evolution loop with advanced strategies"""
        
        # Initialize population
        if not self.population:
            self.population = self.generate_population(population_size)
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            if fitness_func:
                for genome in self.population:
                    if genome.fitness is None:
                        try:
                            metrics = fitness_func(genome)
                            if isinstance(metrics, dict):
                                genome.fitness = metrics.get('fitness', 0)
                                genome.accuracy = metrics.get('accuracy', 0)
                                genome.latency = metrics.get('latency', float('inf'))
                                genome.memory_usage = metrics.get('memory', float('inf'))
                            else:
                                genome.fitness = metrics
                        except Exception as e:
                            print(f"Fitness evaluation failed for a genome: {e}. Assigning poor fitness.")
                            genome.fitness = -1.0 # Penalize failing architectures
            
            # Multi-objective optimization
            if multi_objective:
                self.multi_objective.update_pareto_front(self.population)
                crowding = self.multi_objective.crowding_distance(self.population)
            
            # Sort population
            if multi_objective and self.multi_objective.pareto_front:
                # Sort by Pareto rank and crowding distance
                self.population.sort(
                    key=lambda g: (
                        g not in self.multi_objective.pareto_front,
                        -crowding.get(g, 0)
                    )
                )
            else:
                # Sort by fitness
                self.population.sort(key=lambda g: g.fitness or 0, reverse=True)
            
            # Update hall of fame
            self.hall_of_fame.extend(self.population[:5])
            self.hall_of_fame.sort(key=lambda g: g.fitness or 0, reverse=True)
            self.hall_of_fame = [g for g in self.hall_of_fame if g.fitness is not None and g.fitness > -1.0] # Clean up
            self.hall_of_fame = self.hall_of_fame[:20]  # Keep top 20
            
            # Selection and reproduction
            elite_size = max(2, population_size // 10)
            new_population = self.population[:elite_size]  # Elitism
            
            # Add some hall of fame members
            if self.hall_of_fame:
                new_population.extend(random.sample(
                    self.hall_of_fame,
                    min(3, len(self.hall_of_fame))
                ))
            
            attempts = 0
            while len(new_population) < population_size and attempts < population_size * 5:
                attempts += 1
                try:
                    # Tournament selection
                    parent1 = self._tournament_select()
                    parent2 = self._tournament_select()
                    
                    # Crossover
                    child = self.crossover_advanced(parent1, parent2)
                    
                    # Mutation
                    if random.random() < 0.8: # Higher mutation rate for exploration
                        child = self.mutate_advanced(child)
                    
                    # Validate constraints
                    valid, violations = self.constraints.validate(child)
                    
                    # Add if valid and novel
                    if valid and self.is_novel_architecture(child):
                        new_population.append(child)
                except Exception as e:
                    print(f"Warning: Error during reproduction: {e}")
                    continue
            
            # If we failed to generate enough, fill with random new ones
            if len(new_population) < population_size:
                new_population.extend(self.generate_population(population_size - len(new_population)))

            self.population = new_population[:population_size]
            
            # Report progress
            best = self.population[0]
            print(f"Gen {gen}: Best fitness={best.fitness:.4f}, "
                  f"Pareto front size={len(self.multi_objective.pareto_front)}, "
                  f"Unique architectures={len(self.diversity_cache)}")
            
            # Adaptive mutation rate
            if gen > 10 and gen % 10 == 0:
                self._adapt_mutation_rates()
        
        return self.population
    
    def _tournament_select(self, tournament_size: int = 3) -> ModelGenome:
        """Tournament selection"""
        contenders = [g for g in self.population if g.fitness is not None]
        if not contenders: return random.choice(self.population)
        tournament = random.sample(contenders, min(tournament_size, len(contenders)))
        return max(tournament, key=lambda g: g.fitness or -float('inf'))
    
    def _adapt_mutation_rates(self):
        """Adapt mutation rates based on population diversity"""
        # Calculate population diversity
        stagnation = len(self.hall_of_fame) > 0 and \
                     all(g.get_signature() == self.hall_of_fame[0].get_signature() for g in self.population[:5])
        
        if stagnation:  # Low diversity or stuck at local optimum
            # Increase mutation rates
            print("Stagnation detected. Increasing mutation rates.")
            for key in self.mutation_strategies:
                self.mutation_strategies[key] *= 1.2
        else:  # Good diversity
            # Decrease mutation rates
            for key in self.mutation_strategies:
                self.mutation_strategies[key] *= 0.95
        
        # Normalize
        total = sum(self.mutation_strategies.values())
        for key in self.mutation_strategies:
            self.mutation_strategies[key] /= total

# Example usage
if __name__ == "__main__":
    print("Enhanced Evolvo Model Library Demo")
    print("="*50)
    
    # Define shapes
    input_shape = TensorShape(batch=None, channels=3, height=32, width=32)
    output_shape = TensorShape(features=10)  # CIFAR-10 classes
    
    # Create constraints
    constraints = ArchitectureConstraints()
    constraints.max_depth = 30
    constraints.max_parameters = 5e6  # 5M parameters
    constraints.add_layer_limit('conv2d', 15)
    
    # Create evolver
    evolver = AdvancedModelEvolver(
        input_shape, output_shape,
        task_type='classification',
        constraints=constraints
    )
    
    # Generate some architectures
    print("\nGenerating architectures with different strategies:")
    for strategy in ['residual', 'mobile', 'random']:
        try:
            genome = evolver.generate_architecture(strategy)
            print(f"\n{strategy.upper()} Architecture:")
            print(f"  Layers: {len(genome.layers)}")
            print(f"  Patterns: {[p.value for p in genome.patterns]}")
            
            complexity = genome.estimate_complexity()
            print(f"  Estimated params: {complexity['params']:.2e}")
            print(f"  Estimated FLOPs: {complexity['flops']:.2e} MFLOPs")
            
            # Validate
            valid, violations = constraints.validate(genome)
            print(f"  Valid: {valid}")
            if violations:
                print(f"  Violations: {violations}")
            
            # Analyze gradient flow
            analysis = GradientFlowAnalyzer.analyze(genome)
            print(f"  Gradient flow analysis:")
            print(f"    Potential vanishing: {analysis['potential_vanishing']}")
            print(f"    Potential exploding: {analysis['potential_exploding']}")
            print(f"    Bottlenecks: {analysis['bottlenecks']}")
            
            # Test model creation
            model = genome.to_pytorch_model()
            print("  PyTorch model created successfully.")
            # Test forward pass
            dummy_input = torch.randn(2, 3, 32, 32)
            output = model(dummy_input)
            print(f"  Forward pass successful. Output shape: {output.shape}")

        except Exception as e:
            print(f"\nERROR generating '{strategy}' architecture: {e}")

    print("\n" + "="*50)
    print("Enhanced system ready for advanced neural architecture search!")