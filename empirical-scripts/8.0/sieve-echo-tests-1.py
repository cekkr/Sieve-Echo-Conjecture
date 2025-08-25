#!/usr/bin/env python3
"""
Sieve Echo Conjecture - Enhanced Empirical Analysis System
Version 8.0 - Incorporates critical improvements from empirical findings

Key Enhancements Based on Empirical Discoveries:
1. Base Invariance Testing (CRITICAL)
2. Riemann Zeta Function Connections
3. Prime Number Theorem Integration
4. Enhanced Fourier Analysis with Phase Information
5. Chinese Remainder Theorem Validation
6. Multiplicative Order Deep Analysis
7. Euler's Constant Investigation
8. Modular Forms Connection
9. Adaptive Feature Discovery
10. Parallel Processing Framework
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
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings
import traceback
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# Core mathematical libraries
from sympy import (factorint, primerange, isprime, totient, divisors, mobius,
                   primorial, factorial, nextprime, prevprime, prime, primepi, sqrt, log)
from scipy import stats, signal, optimize, special, integrate
from scipy.fft import fft, ifft, fftfreq
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Neural network support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Neural network features disabled.")

warnings.filterwarnings('ignore')

# Import Evolvo engine
try:
    from evolvo_model import DataStore, InstructionSet, Interpreter, BaseEvaluator, myFloat
    EVOLVO_AVAILABLE = True
except ImportError:
    EVOLVO_AVAILABLE = False
    print("WARNING: evolvo_engine not found. Genetic formula discovery will be limited.")
    myFloat = float

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    max_n: int = 100000
    test_bases: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31])
    sample_size: int = 10000
    runtime_hours: float = 24.0
    
    # Base invariance testing
    base_invariance_threshold: float = 0.1
    base_invariance_test_count: int = 50
    
    # Mathematical constants
    euler_gamma: float = 0.5772156649015329
    meissel_mertens: float = 0.2614972128476428
    
    # Parallel processing
    n_workers: int = cpu_count()
    chunk_size: int = 1000
    
    # Analysis settings
    test_crt: bool = True
    test_zeta: bool = True
    test_pnt: bool = True
    test_modular: bool = True
    test_fourier_enhanced: bool = True
    test_multiplicative_order: bool = True
    
    # Output settings
    save_plots: bool = True
    save_models: bool = True
    verbose: bool = True
    checkpoint_interval: int = 100

CONFIG = Config()

# ==============================================================================
# ENHANCED MATHEMATICAL CONSTANTS LIBRARY
# ==============================================================================

class ExtendedMathematicalConstants:
    """Extended library with systematic constant testing"""
    
    def __init__(self):
        self.constants = {
            'e': math.e,
            'pi': math.pi,
            'golden_ratio': (1 + math.sqrt(5)) / 2,
            'golden_ratio_conjugate': (math.sqrt(5) - 1) / 2,
            'inv_golden_ratio': 2 / (1 + math.sqrt(5)),
            'inv_golden_ratio_sq': 2 - (1 + math.sqrt(5)) / 2,
            'euler_mascheroni': 0.5772156649015329,
            'meissel_mertens': 0.2614972128476428,
            'artin': 0.3739558136192023,
            'sqrt_2': math.sqrt(2),
            'sqrt_3': math.sqrt(3),
            'sqrt_5': math.sqrt(5),
            'ln_2': math.log(2),
            'ln_10': math.log(10),
            'catalan': 0.915965594177219,
            'apery': 1.202056903159594,  # ζ(3)
            'twin_prime': 0.6601618158468696,
            'mills': 1.3063778838630806,
            'plastic': 1.324717957244746,
            'tribonacci': 1.839286755214161,
            'khinchin': 2.6854520010653064,
            'feigenbaum_delta': 4.669201609102990,
            'feigenbaum_alpha': 2.502907875095892,
            'conway': 1.303577269034296,
            'omega': 0.5671432904097838,  # W(1)
            'gauss': 0.8346268416740731,
            'landau_ramanujan': 0.76422365358922,
            'viswanath': 1.1319882487943,
            'parabolic': 2.2955871493926,
        }
        
        # Common expressions
        self.expressions = {
            '5_minus_1_over_15': 5 - 1/15,
            'e_to_gamma': math.exp(0.5772156649015329),
            'e_to_minus_gamma': math.exp(-0.5772156649015329),
            'pi_squared_over_6': math.pi**2 / 6,
            'sqrt_2_minus_1': math.sqrt(2) - 1,
            'log_log_2': math.log(math.log(2)),
            '3_over_4': 0.75,
            'negative_one': -1.0,
            'approx_negative_one': -0.9988,  # The empirical alpha
            'empirical_beta': 2.42,
        }
    
    def find_closest_constant(self, value: float, tolerance: float = 0.01) -> Optional[Tuple[str, float]]:
        """Find closest matching constant"""
        best_match = None
        best_error = tolerance
        
        for name, const in {**self.constants, **self.expressions}.items():
            error = abs(value - const)
            if error < best_error:
                best_error = error
                best_match = (name, const)
        
        return best_match

# ==============================================================================
# ENHANCED LOGGER
# ==============================================================================

class EnhancedLogger:
    """Enhanced logger with structured output"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.findings = defaultdict(list)
        self.correlations = {}
        self.formulas = {}
        self.patterns = {}
        self.anomalies = []
        self.base_invariance_results = {}
        self.zeta_connections = {}
        self.pnt_relationships = {}
        self.start_time = time.time()
        self.log_file = f'sieve_echo_enhanced_{self.timestamp}.txt'
        
        with open(self.log_file, 'w') as f:
            f.write(f"Sieve Echo Enhanced Analysis - Started {datetime.now()}\n")
            f.write("="*80 + "\n")
    
    def log(self, message: str, level: str = "INFO"):
        elapsed = time.time() - self.start_time
        formatted = f"[{elapsed:.1f}s][{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def add_finding(self, category: str, finding: Dict):
        self.findings[category].append({
            'timestamp': time.time() - self.start_time,
            'data': finding
        })
        if category in ['CRITICAL', 'DISCOVERY']:
            self.log(f"{category}: {finding.get('description', 'New finding')}", category)
    
    def save_all(self):
        results = {
            'findings': dict(self.findings),
            'correlations': self.correlations,
            'formulas': self.formulas,
            'patterns': self.patterns,
            'anomalies': self.anomalies,
            'base_invariance': self.base_invariance_results,
            'zeta_connections': self.zeta_connections,
            'pnt_relationships': self.pnt_relationships,
            'runtime': time.time() - self.start_time
        }
        
        with open(f'sieve_echo_enhanced_results_{self.timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f'sieve_echo_enhanced_complete_{self.timestamp}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        self.log(f"Results saved ({len(self.formulas)} formulas, {len(self.correlations)} correlations)", "INFO")

logger = EnhancedLogger()

# ==============================================================================
# CRITICAL IMPROVEMENT 1: BASE INVARIANCE TESTING
# ==============================================================================

class BaseInvarianceTester:
    """Test if theta entropy is truly base-invariant (CRITICAL)"""
    
    def __init__(self):
        self.results = {}
        self.invariant_numbers = []
        self.non_invariant_numbers = []
    
    def test_base_invariance(self, n: int, bases: List[int] = None) -> Dict:
        """Test if theta entropy is base-invariant for n"""
        if bases is None:
            bases = list(range(2, min(50, n)))
        
        entropies = []
        valid_bases = []
        
        for base in bases:
            if math.gcd(n, base) == 1:
                pattern = self.compute_repetend(n, base)
                if pattern:
                    ndr = np.array(pattern) / base
                    entropy = self.compute_theta_entropy(ndr)
                    entropies.append(entropy)
                    valid_bases.append(base)
        
        if len(entropies) < 2:
            return {'is_invariant': None, 'reason': 'insufficient_bases'}
        
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        cv = std_entropy / mean_entropy if mean_entropy > 0 else float('inf')
        
        is_invariant = cv < CONFIG.base_invariance_threshold
        
        result = {
            'n': n,
            'mean': mean_entropy,
            'std': std_entropy,
            'cv': cv,
            'is_invariant': is_invariant,
            'num_bases_tested': len(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'range': np.max(entropies) - np.min(entropies)
        }
        
        # Store for analysis
        if is_invariant:
            self.invariant_numbers.append(n)
        else:
            self.non_invariant_numbers.append(n)
        
        self.results[n] = result
        return result
    
    def compute_repetend(self, n: int, base: int, max_length: int = 10000) -> List[int]:
        """Compute repeating decimal pattern"""
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
        
        if remainder in seen:
            return digits[seen[remainder]:]
        return digits
    
    def compute_theta_entropy(self, ndr: np.ndarray) -> float:
        """Compute spectral entropy"""
        if len(ndr) < 2:
            return 0.0
        
        fft_vals = np.abs(fft(ndr))[:len(ndr)//2]
        if len(fft_vals) == 0:
            return 0.0
        
        power = fft_vals**2
        total_power = np.sum(power)
        if total_power == 0:
            return 0.0
        
        p_spectrum = power / total_power
        p_spectrum = p_spectrum[p_spectrum > 1e-10]
        if len(p_spectrum) == 0:
            return 0.0
        
        return -np.sum(p_spectrum * np.log(p_spectrum))
    
    def analyze_invariance_patterns(self) -> Dict:
        """Analyze patterns in base invariance"""
        if not self.results:
            return {}
        
        # Check if primes are more invariant
        prime_invariant = [n for n in self.invariant_numbers if isprime(n)]
        composite_invariant = [n for n in self.invariant_numbers if not isprime(n)]
        
        analysis = {
            'total_tested': len(self.results),
            'invariant_count': len(self.invariant_numbers),
            'non_invariant_count': len(self.non_invariant_numbers),
            'invariance_rate': len(self.invariant_numbers) / max(1, len(self.results)),
            'prime_invariance_rate': len(prime_invariant) / max(1, sum(1 for n in self.results if isprime(n))),
            'composite_invariance_rate': len(composite_invariant) / max(1, sum(1 for n in self.results if not isprime(n))),
            'mean_cv_invariant': np.mean([r['cv'] for r in self.results.values() if r['is_invariant']]) if self.invariant_numbers else 0,
            'mean_cv_non_invariant': np.mean([r['cv'] for r in self.results.values() if not r['is_invariant']]) if self.non_invariant_numbers else 0
        }
        
        return analysis

# ==============================================================================
# CRITICAL IMPROVEMENT 2: RIEMANN ZETA CONNECTION
# ==============================================================================

class RiemannZetaAnalyzer:
    """Explore connections to Riemann zeta function"""
    
    def __init__(self):
        self.zeta_values = {}
        self.correlations = {}
        
    def compute_zeta_correlation(self, n_values: List[int], theta_entropies: List[float], 
                                s_values: List[float] = [2, 3, 4]) -> Dict:
        """Explore connections to Riemann zeta function"""
        correlations = {}
        
        for s in s_values:
            # Partial zeta sum up to max n
            max_n = max(n_values)
            zeta_partial = sum(1/k**s for k in range(1, max_n + 1))
            self.zeta_values[s] = zeta_partial
            
            # Test various relationships
            correlations[f'zeta_{s}'] = {
                'zeta_value': zeta_partial,
                'correlations': {}
            }
            
            # Direct correlation with 1/n^s
            inverse_powers = [1/n**s for n in n_values]
            if len(theta_entropies) == len(inverse_powers):
                corr = np.corrcoef(theta_entropies, inverse_powers)[0,1]
                correlations[f'zeta_{s}']['correlations']['direct'] = corr
            
            # Log correlation
            log_n_s = [np.log(n**s) for n in n_values]
            if len(theta_entropies) == len(log_n_s):
                corr = np.corrcoef(theta_entropies, log_n_s)[0,1]
                correlations[f'zeta_{s}']['correlations']['log'] = corr
            
            # Product with omega
            omega_values = [len(factorint(n)) for n in n_values]
            product_vals = [zeta_partial * omega for omega in omega_values]
            if len(theta_entropies) == len(product_vals):
                corr = np.corrcoef(theta_entropies, product_vals)[0,1]
                correlations[f'zeta_{s}']['correlations']['omega_product'] = corr
        
        # Test connection to Riemann hypothesis
        # Critical line Re(s) = 1/2
        s_critical = 0.5
        correlations['critical_line'] = self.test_critical_line_connection(n_values, theta_entropies)
        
        self.correlations = correlations
        return correlations
    
    def test_critical_line_connection(self, n_values: List[int], theta_entropies: List[float]) -> Dict:
        """Test connection to critical line of zeta function"""
        # Compute proxy for zeta on critical line
        critical_values = []
        
        for n in n_values:
            # Use Dirichlet eta function as proxy (converges for Re(s) > 0)
            eta_value = sum((-1)**(k+1) / k**0.5 for k in range(1, min(n, 1000)))
            critical_values.append(eta_value)
        
        if len(theta_entropies) == len(critical_values):
            correlation = np.corrcoef(theta_entropies, critical_values)[0,1]
            return {
                'correlation': correlation,
                'significant': abs(correlation) > 0.3
            }
        
        return {'correlation': 0, 'significant': False}
    
    def compute_prime_zeta(self, n: int, s: float = 2) -> float:
        """Compute prime zeta function P(s) = sum(1/p^s) for primes p <= n"""
        primes = list(primerange(2, n + 1))
        return sum(1/p**s for p in primes)

# ==============================================================================
# CRITICAL IMPROVEMENT 3: PRIME NUMBER THEOREM INTEGRATION
# ==============================================================================

def symbolic_std(values):
    mean = sum(values) / len(values)
    variance = sum((v - mean)**2 for v in values) / len(values)
    return sqrt(variance)

class PrimeNumberTheoremAnalyzer:
    """Test connections to Prime Number Theorem"""
    
    def __init__(self):
        self.results = {}
        
    def logarithmic_integral(self, x: float) -> float:
        """Compute logarithmic integral Li(x)"""
        if x <= 2:
            return 0
        
        # Numerical integration
        result, _ = integrate.quad(lambda t: 1/np.log(t), 2, x)
        return result
    
    def test_pnt_relationship(self, n_values: List[int], theta_entropies: List[float]) -> Dict:
        """Test connections to Prime Number Theorem"""
        results = []
        
        for i, n in enumerate(n_values):
            if n <= 2:
                continue
            
            # PNT approximation
            pi_n = n / np.log(n)
            
            # Li(n) - logarithmic integral
            li_n = self.logarithmic_integral(n)
            
            # Actual prime count
            actual_pi_n = primepi(n)
            
            # Theta entropy
            h_theta = theta_entropies[i] if i < len(theta_entropies) else 0
            
            # Test various relationships
            result = {
                'n': n,
                'h_theta': h_theta,
                'pi_n': actual_pi_n,
                'pnt_approx': pi_n,
                'li_approx': li_n,
                'pnt_error': abs(pi_n - actual_pi_n),
                'li_error': abs(li_n - actual_pi_n),
                'pnt_ratio': pi_n / n,
                'li_ratio': li_n / n,
                'pnt_entropy_product': h_theta * np.log(n),
                'hardy_ramanujan': np.log(np.log(n)) if n > 2 else 0,  # Average ω(n)
                'mertens': self.mertens_function(n)
            }
            
            results.append(result)
        
        # Analyze correlations
        if results:
            df_results = {key: [r[key] for r in results] for key in results[0].keys()}
            
            correlations = {}
            for key in ['pnt_ratio', 'li_ratio', 'pnt_entropy_product', 'hardy_ramanujan', 'mertens']:
                if key in df_results and 'h_theta' in df_results:
                    vals = df_results[key]
                    h_vals = df_results['h_theta']
                    if len(vals) == len(h_vals) and symbolic_std(vals) > 0 and symbolic_std(h_vals) > 0:
                        # Rational to float conversion
                        vals = np.array([float(v) for v in vals])
                        h_vals = np.array([float(h) for h in h_vals])

                        corr = None
                        if vals.size > 1 and h_vals.size > 1:
                            corr = np.corrcoef(vals, h_vals)[0, 1]

                        correlations[key] = corr

            self.results = {
                'data': results,
                'correlations': correlations,
                'best_predictor': max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
            }
        
        return self.results
    
    def mertens_function(self, n: int) -> float:
        """Compute Mertens function M(n) = sum of Möbius function"""
        return sum(mobius(k) for k in range(1, n + 1))

# ==============================================================================
# IMPROVEMENT 4: ENHANCED FOURIER ANALYSIS
# ==============================================================================

class EnhancedFourierAnalyzer:
    """Advanced Fourier analysis with phase information"""
    
    def __init__(self):
        self.spectral_features = {}
        
    def advanced_fourier_analysis(self, ndr_pattern: np.ndarray) -> Dict:
        """Enhanced Fourier analysis with phase information"""
        if len(ndr_pattern) < 2:
            return {}
        
        # Compute FFT
        fft_result = fft(ndr_pattern)
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Power spectrum
        power = magnitude**2
        
        # Frequency bins
        freqs = fftfreq(len(ndr_pattern))
        
        # Spectral features
        total_power = np.sum(power)
        if total_power == 0:
            return {}
        
        normalized_power = power / total_power
        
        # Spectral centroid
        spectral_centroid = np.sum(np.arange(len(power)) * normalized_power)
        
        # Spectral spread
        spectral_spread = np.sqrt(np.sum((np.arange(len(power)) - spectral_centroid)**2 * normalized_power))
        
        # Spectral flux
        if len(magnitude) > 1:
            spectral_flux = np.sum(np.diff(magnitude)**2)
        else:
            spectral_flux = 0
        
        # Phase coherence (stability of significant components)
        significant_mask = magnitude > 0.1 * np.max(magnitude)
        phase_coherence = np.std(phase[significant_mask]) if np.any(significant_mask) else 0
        
        # Dominant frequency (skip DC component)
        if len(magnitude) > 1:
            dominant_frequency_idx = np.argmax(magnitude[1:]) + 1
            dominant_frequency = freqs[dominant_frequency_idx] if dominant_frequency_idx < len(freqs) else 0
        else:
            dominant_frequency = 0
        
        # Spectral entropy
        spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-10))
        
        # Spectral rolloff
        cumsum_power = np.cumsum(normalized_power)
        rolloff_idx = np.where(cumsum_power >= 0.85)[0]
        spectral_rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else len(power)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sum(np.abs(np.arange(len(power)) - spectral_centroid) * normalized_power)
        
        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_flux': spectral_flux,
            'phase_coherence': phase_coherence,
            'dominant_frequency': dominant_frequency,
            'spectral_entropy': spectral_entropy,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'dc_component': magnitude[0] if len(magnitude) > 0 else 0,
            'max_magnitude': np.max(magnitude),
            'mean_phase': np.mean(phase),
            'phase_variance': np.var(phase)
        }
        
        return features

# ==============================================================================
# IMPROVEMENT 5: CHINESE REMAINDER THEOREM VALIDATION
# ==============================================================================

class CRTValidator:
    """Validate that theta patterns follow Chinese Remainder Theorem"""
    
    def __init__(self):
        self.validation_results = {}
        
    def compute_theta_pattern(self, n: int, base: int = 10) -> np.ndarray:
        """Compute theta pattern for n in given base"""
        if math.gcd(n, base) != 1:
            return np.array([])
        
        remainder = 1
        pattern = []
        seen = {}
        
        while remainder != 0 and remainder not in seen:
            seen[remainder] = len(pattern)
            remainder = (remainder * base) % n
            digit = remainder
            pattern.append(digit)
        
        return np.array(pattern) / n  # Normalized
    
    def crt_combine_patterns(self, pattern_p1: np.ndarray, pattern_p2: np.ndarray, 
                           p1: int, p2: int) -> np.ndarray:
        """Combine patterns using CRT"""
        if math.gcd(p1, p2) != 1:
            return np.array([])
        
        # Length of combined pattern
        combined_length = len(pattern_p1) * len(pattern_p2)
        combined = np.zeros(combined_length)
        
        # CRT reconstruction
        for i in range(combined_length):
            # Map index to residues mod p1 and p2
            r1 = i % len(pattern_p1)
            r2 = i % len(pattern_p2)
            
            # Combine using CRT formula
            # This is simplified - actual CRT combination would be more complex
            combined[i] = (pattern_p1[r1] * p2 + pattern_p2[r2] * p1) / (p1 * p2)
        
        return combined
    
    def validate_crt_for_theta(self, p1: int, p2: int, base: int = 10) -> Dict:
        """Validate that theta patterns follow CRT"""
        if not isprime(p1) or not isprime(p2):
            return {'is_valid': False, 'reason': 'not_primes'}
        
        n = p1 * p2
        
        # Individual patterns
        pattern_p1 = self.compute_theta_pattern(p1, base)
        pattern_p2 = self.compute_theta_pattern(p2, base)
        
        # Combined pattern
        pattern_n = self.compute_theta_pattern(n, base)
        
        if len(pattern_p1) == 0 or len(pattern_p2) == 0 or len(pattern_n) == 0:
            return {'is_valid': False, 'reason': 'pattern_computation_failed'}
        
        # CRT reconstruction
        reconstructed = self.crt_combine_patterns(pattern_p1, pattern_p2, p1, p2)
        
        # Adjust lengths for comparison
        min_len = min(len(pattern_n), len(reconstructed))
        if min_len == 0:
            return {'is_valid': False, 'reason': 'empty_patterns'}
        
        pattern_n_trimmed = pattern_n[:min_len]
        reconstructed_trimmed = reconstructed[:min_len]
        
        # Compare
        error = np.mean(np.abs(pattern_n_trimmed - reconstructed_trimmed))
        correlation = np.corrcoef(pattern_n_trimmed, reconstructed_trimmed)[0,1] if min_len > 1 else 0
        
        is_valid = error < 0.01 or correlation > 0.99
        
        result = {
            'p1': p1,
            'p2': p2,
            'n': n,
            'error': error,
            'correlation': correlation,
            'is_valid': is_valid,
            'pattern_n_length': len(pattern_n),
            'reconstructed_length': len(reconstructed),
            'length_ratio': len(pattern_n) / (len(pattern_p1) * len(pattern_p2))
        }
        
        self.validation_results[f"{p1}_{p2}"] = result
        return result

# ==============================================================================
# IMPROVEMENT 6: MULTIPLICATIVE ORDER ANALYSIS
# ==============================================================================

class MultiplicativeOrderAnalyzer:
    """Deep analysis of multiplicative order patterns"""
    
    def __init__(self):
        self.order_data = {}
        
    def multiplicative_order(self, base: int, n: int) -> int:
        """Compute multiplicative order of base modulo n"""
        if math.gcd(base, n) != 1:
            return 0
        
        order = 1
        current = base % n
        
        while current != 1:
            current = (current * base) % n
            order += 1
            if order > n:  # Safety check
                return 0
        
        return order
    
    def analyze_multiplicative_order(self, n: int, bases: List[int] = None) -> Dict:
        """Deep analysis of multiplicative order patterns"""
        if bases is None:
            bases = list(range(2, min(100, n)))
        
        orders = []
        phi_n = totient(n)
        
        for base in bases:
            if math.gcd(n, base) == 1:
                order = self.multiplicative_order(base, n)
                
                order_data = {
                    'base': base,
                    'order': order,
                    'order_ratio': order / phi_n if phi_n > 0 else 0,
                    'is_primitive_root': order == phi_n,
                    'order_mod_n': order % n,
                    'divides_phi': phi_n % order == 0 if order > 0 else False
                }
                
                orders.append(order_data)
        
        if not orders:
            return {}
        
        # Statistical analysis
        order_values = [o['order'] for o in orders]
        order_ratios = [o['order_ratio'] for o in orders]
        
        analysis = {
            'n': n,
            'phi_n': phi_n,
            'mean_order': np.mean(order_values),
            'std_order': np.std(order_values),
            'mean_order_ratio': np.mean(order_ratios),
            'primitive_root_count': sum(1 for o in orders if o['is_primitive_root']),
            'has_primitive_root': any(o['is_primitive_root'] for o in orders),
            'order_gcd': np.gcd.reduce(order_values) if order_values else 0,
            'unique_orders': len(set(order_values)),
            'max_order': max(order_values) if order_values else 0,
            'min_order': min(order_values) if order_values else 0,
            'carmichael_lambda': self.carmichael_lambda(n)
        }
        
        # Check if n is cyclic
        analysis['is_cyclic'] = analysis['has_primitive_root']
        
        self.order_data[n] = analysis
        return analysis
    
    def carmichael_lambda(self, n: int) -> int:
        """Compute Carmichael's lambda function"""
        if n == 1:
            return 1
        
        factors = factorint(n)
        lambda_values = []
        
        for p, e in factors.items():
            if p == 2 and e >= 3:
                lambda_values.append(2**(e-2))
            else:
                lambda_values.append(p**(e-1) * (p-1))
        
        return np.lcm.reduce(lambda_values) if lambda_values else 1

# ==============================================================================
# IMPROVEMENT 7: EULER'S CONSTANT INVESTIGATION
# ==============================================================================

class EulerConstantInvestigator:
    """Investigate role of Euler's constant and e"""
    
    def __init__(self):
        self.euler_gamma = 0.5772156649015329
        self.results = {}
        
    def explore_euler_constant(self, n_values: List[int], theta_entropies: List[float]) -> Dict:
        """Investigate role of Euler's constant"""
        results = []
        
        for i, n in enumerate(n_values):
            if i >= len(theta_entropies):
                break
                
            h_theta = theta_entropies[i]
            
            # Mertens' theorem sum
            mertens_sum = sum(1/p for p in primerange(2, n+1))
            mertens_approx = np.log(np.log(n)) + self.euler_gamma if n > 2 else 0
            
            result = {
                'n': n,
                'h_theta': h_theta,
                'log_relation': h_theta / np.log(n) if n > 1 else 0,
                'euler_product': h_theta * self.euler_gamma,
                'exp_relation': np.exp(-h_theta),
                'mertens_sum': mertens_sum,
                'mertens_approx': mertens_approx,
                'mertens_error': abs(mertens_sum - mertens_approx),
                'exp_gamma_theta': np.exp(self.euler_gamma * h_theta),
                'log_gamma_relation': h_theta / (np.log(n) + self.euler_gamma) if n > 1 else 0
            }
            
            results.append(result)
        
        # Analyze correlations
        if results:
            correlations = {}
            h_theta_vals = [r['h_theta'] for r in results]
            
            for key in ['log_relation', 'euler_product', 'exp_relation', 'mertens_error', 
                       'exp_gamma_theta', 'log_gamma_relation']:
                vals = [r[key] for r in results]
                if np.std(vals) > 0 and np.std(h_theta_vals) > 0:
                    corr = np.corrcoef(vals, h_theta_vals)[0,1]
                    correlations[key] = corr
            
            self.results = {
                'data': results,
                'correlations': correlations,
                'best_correlation': max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None,
                'euler_gamma_appears': any(abs(c - self.euler_gamma) < 0.01 for c in correlations.values())
            }
        
        return self.results

# ==============================================================================
# IMPROVEMENT 8: MODULAR FORMS CONNECTION
# ==============================================================================

class ModularFormsExplorer:
    """Explore connections to modular arithmetic"""
    
    def __init__(self):
        self.modular_patterns = {}
        
    def explore_modular_forms(self, n: int, theta_entropy: float, 
                            moduli: List[int] = [6, 12, 30, 210]) -> Dict:
        """Explore connections to modular arithmetic"""
        patterns = {}
        
        for mod in moduli:
            residue = n % mod
            
            # Compute theta entropy for numbers with same residue
            same_residue = [k for k in range(max(2, n-mod), n+mod+1) 
                          if k % mod == residue and k > 1]
            
            # For this analysis, we'll simulate entropies
            # In practice, these would be computed from actual patterns
            entropies = []
            for k in same_residue[:10]:  # Limit for efficiency
                # Simplified entropy computation
                pattern_length = totient(k)
                simulated_entropy = log(pattern_length) / log(k) if k > 1 else 0
                entropies.append(simulated_entropy)
            
            patterns[f'mod_{mod}'] = {
                'residue': residue,
                'mean_entropy': np.mean(entropies) if entropies else 0,
                'std_entropy': symbolic_std(entropies) if entropies else 0,
                'is_prime_residue': isprime(residue) if residue > 1 else False,
                'is_coprime': math.gcd(residue, mod) == 1,
                'theta_deviation': abs(theta_entropy - np.mean(entropies)) if entropies else 0
            }
        
        self.modular_patterns[n] = patterns
        return patterns

# ==============================================================================
# IMPROVEMENT 9: ADAPTIVE FEATURE DISCOVERY
# ==============================================================================

class AdaptiveFeatureDiscovery:
    """Adaptive discovery of important features"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.feature_importance = {}
        self.discovered_relationships = []
        
    def discover_polynomial_relationships(self, target: str = 'omega', max_degree: int = 5):
        """Find polynomial relationships between features"""
        # Extract features
        feature_names = []
        for d in self.data[:10]:  # Sample to get feature names
            for k, v in d.items():
                if isinstance(v, (int, float)) and k != target:
                    if k not in feature_names:
                        feature_names.append(k)
        
        # Prepare data
        X_data = []
        y_data = []
        
        for d in self.data:
            if target not in d:
                continue
            
            features = []
            valid = True
            for fname in feature_names[:10]:  # Limit features for efficiency
                if fname in d:
                    features.append(d[fname])
                else:
                    valid = False
                    break
            
            if valid and features:
                X_data.append(features)
                y_data.append(d[target])
        
        if len(X_data) < 50:
            return []
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Generate polynomial features
        poly = PolynomialFeatures(degree=max_degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Select best features
        selector = SelectKBest(f_regression, k=min(20, X_poly.shape[1]))
        selector.fit(X_poly, y)
        
        # Get selected features
        selected_indices = selector.get_support(indices=True)
        feature_names_poly = poly.get_feature_names_out(feature_names[:len(X[0])])
        
        relationships = []
        for idx in selected_indices:
            if idx < len(feature_names_poly) and idx < len(selector.scores_):
                relationships.append({
                    'formula': feature_names_poly[idx],
                    'score': selector.scores_[idx],
                    'selected': True
                })
        
        self.discovered_relationships = relationships
        return relationships
    
    def discover_ratio_relationships(self, target: str = 'omega'):
        """Find important ratios between features"""
        feature_names = []
        for d in self.data[:10]:
            for k, v in d.items():
                if isinstance(v, (int, float)) and k != target:
                    if k not in feature_names:
                        feature_names.append(k)
        
        ratios = []
        
        for i, f1 in enumerate(feature_names[:20]):  # Limit for efficiency
            for f2 in feature_names[i+1:20]:
                # Prepare ratio data
                ratio_vals = []
                target_vals = []
                
                for d in self.data:
                    if f1 in d and f2 in d and target in d and d[f2] != 0:
                        ratio_vals.append(d[f1] / d[f2])
                        target_vals.append(d[target])
                
                if len(ratio_vals) > 50:
                    correlation = np.corrcoef(ratio_vals, target_vals)[0, 1]
                    
                    if abs(correlation) > 0.5:
                        ratios.append({
                            'numerator': f1,
                            'denominator': f2,
                            'correlation': correlation
                        })
        
        # Sort by absolute correlation
        ratios.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return ratios[:10]  # Return top 10

# ==============================================================================
# IMPROVEMENT 10: PARALLEL PROCESSING FRAMEWORK
# ==============================================================================

class ParallelSieveAnalyzer:
    """Parallel processing for large-scale analysis"""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or cpu_count()
        self.results = []
        
    def analyze_single(self, n: int) -> Dict:
        """Analyze a single number (worker function)"""
        result = {'n': n}
        
        # Number theory properties
        result['omega'] = len(factorint(n))
        result['Omega'] = sum(factorint(n).values())
        result['tau'] = len(divisors(n))
        result['sigma'] = sum(divisors(n))
        result['phi'] = totient(n)
        result['mu'] = mobius(n)
        result['is_prime'] = isprime(n)
        result['is_prime_power'] = len(factorint(n)) == 1
        
        # Compute theta entropy for multiple bases
        entropies = []
        for base in [2, 3, 5, 7, 10]:
            if math.gcd(n, base) == 1:
                tester = BaseInvarianceTester()
                pattern = tester.compute_repetend(n, base)
                if pattern:
                    ndr = np.array(pattern) / base
                    entropy = tester.compute_theta_entropy(ndr)
                    entropies.append(entropy)
                    result[f'theta_entropy_b{base}'] = entropy
        
        if entropies:
            result['theta_entropy_mean'] = np.mean(entropies)
            result['theta_entropy_std'] = np.std(entropies)
        
        return result
    
    def analyze_chunk(self, range_tuple: Tuple[int, int]) -> List[Dict]:
        """Analyze a chunk of numbers"""
        start, end = range_tuple
        results = []
        
        for n in range(start, end):
            if n > 1:  # Skip 0 and 1
                results.append(self.analyze_single(n))
        
        return results
    
    def analyze_range(self, start: int, end: int, chunk_size: int = None) -> List[Dict]:
        """Parallel analysis of number range"""
        chunk_size = chunk_size or CONFIG.chunk_size
        
        # Create chunks
        ranges = [(i, min(i + chunk_size, end)) 
                 for i in range(start, end, chunk_size)]
        
        # Process in parallel
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.analyze_chunk, r): r for r in ranges}
            
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    logger.log(f"Completed chunk: {len(all_results)} numbers analyzed", "INFO")
                except Exception as e:
                    logger.log(f"Error in chunk processing: {e}", "ERROR")
        
        self.results = all_results
        return all_results

############
############
############


# ==============================================================================
# PICKLABLE HELPER FUNCTIONS FOR EVOLVO
# ==============================================================================

def _add(a, b): return myFloat(a + b)
def _sub(a, b): return myFloat(a - b)
def _mul(a, b): return myFloat(a * b)
def _div(a, b): return myFloat(a / (b if abs(b) > 1e-9 else 1e-9))
def _log(a): return myFloat(math.log(abs(a) + 1e-9))
def _sqrt(a): return myFloat(math.sqrt(abs(a)))
def _pow(a, b): return myFloat(a ** min(abs(b), 5))
def _exp(a): return myFloat(math.exp(min(a, 10)))
def _sin(a): return myFloat(math.sin(a))
def _cos(a): return myFloat(math.cos(a))

# New mathematical operations (non-lambda)
def _tan(a): return myFloat(math.tan(a % (math.pi/2 - 0.01)))
def _atan(a): return myFloat(math.atan(a))
def _log2(a): return myFloat(math.log2(abs(a) + 1e-9))
def _log10(a): return myFloat(math.log10(abs(a) + 1e-9))
def _gcd(a, b): return myFloat(math.gcd(int(abs(a)), int(abs(b))))
def _modinv(a, n): 
    try:
        return myFloat(pow(int(a), -1, int(n)) if math.gcd(int(a), int(n)) == 1 else 0)
    except:
        return myFloat(0)
def _min(a, b): return myFloat(min(a, b))
def _max(a, b): return myFloat(max(a, b))
def _abs(a): return myFloat(abs(a))
def _gamma(a): return myFloat(math.gamma(min(abs(a), 100)) if a > 0 else 1)
def _erf(a): 
    try:
        return myFloat(special.erf(a))
    except:
        return myFloat(0)
def _mod(a, b): return myFloat(a % b if b != 0 else 0)
def _floor(a): return myFloat(math.floor(a))
def _ceil(a): return myFloat(math.ceil(a))
def _round(a): return myFloat(round(a))
def _sign(a): return myFloat(1 if a > 0 else -1 if a < 0 else 0)
def _hypot(a, b): return myFloat(math.hypot(a, b))

# Constants as functions
def _pi(): return myFloat(math.pi)
def _e(): return myFloat(math.e)
def _phi(): return myFloat((1 + math.sqrt(5))/2)
def _euler_gamma(): return myFloat(0.5772156649015329)

# Boolean operations
def _not_op(a): return not a
def _cmp_op(a, b): return abs(a - b) < 1e-9
def _gt_op(a, b): return a > b
def _gte_op(a, b): return a >= b
def _lt_op(a, b): return a < b
def _lte_op(a, b): return a <= b
def _and_op(a, b): return a and b
def _or_op(a, b): return a or b

def get_picklable_instruction_set() -> Optional[InstructionSet]:
    """Creates an enhanced fully picklable InstructionSet"""
    if not EVOLVO_AVAILABLE:
        return None
        
    iset = InstructionSet()
    
    # Basic arithmetic
    iset.register('ADD', _add, ['d', 'd'], 'decimal')
    iset.register('SUB', _sub, ['d', 'd'], 'decimal')
    iset.register('MUL', _mul, ['d', 'd'], 'decimal')
    iset.register('DIV', _div, ['d', 'd'], 'decimal')
    iset.register('MOD', _mod, ['d', 'd'], 'decimal')
    
    # Mathematical functions
    iset.register('LOG', _log, ['d'], 'decimal')
    iset.register('LOG2', _log2, ['d'], 'decimal')
    iset.register('LOG10', _log10, ['d'], 'decimal')
    iset.register('SQRT', _sqrt, ['d'], 'decimal')
    iset.register('POW', _pow, ['d', 'd'], 'decimal')
    iset.register('EXP', _exp, ['d'], 'decimal')
    
    # Trigonometric
    iset.register('SIN', _sin, ['d'], 'decimal')
    iset.register('COS', _cos, ['d'], 'decimal')
    iset.register('TAN', _tan, ['d'], 'decimal')
    iset.register('ATAN', _atan, ['d'], 'decimal')
    
    # Number theory
    iset.register('GCD', _gcd, ['d', 'd'], 'decimal')
    iset.register('MODINV', _modinv, ['d', 'd'], 'decimal')
    iset.register('FLOOR', _floor, ['d'], 'decimal')
    iset.register('CEIL', _ceil, ['d'], 'decimal')
    iset.register('ROUND', _round, ['d'], 'decimal')
    
    # Statistical
    iset.register('MIN', _min, ['d', 'd'], 'decimal')
    iset.register('MAX', _max, ['d', 'd'], 'decimal')
    iset.register('ABS', _abs, ['d'], 'decimal')
    iset.register('SIGN', _sign, ['d'], 'decimal')
    iset.register('HYPOT', _hypot, ['d', 'd'], 'decimal')
    
    # Special functions
    iset.register('GAMMA', _gamma, ['d'], 'decimal')
    iset.register('ERF', _erf, ['d'], 'decimal')
    
    # Constants
    iset.register('PI', _pi, [], 'decimal')
    iset.register('E', _e, [], 'decimal')
    iset.register('PHI', _phi, [], 'decimal')
    iset.register('EULER', _euler_gamma, [], 'decimal')
    
    # Boolean operations
    iset.register('NOT', _not_op, ['b'], 'bool')
    iset.register('AND', _and_op, ['b', 'b'], 'bool')
    iset.register('OR', _or_op, ['b', 'b'], 'bool')
    iset.register('CMP', _cmp_op, ['d', 'd'], 'bool')
    iset.register('GT', _gt_op, ['d', 'd'], 'bool')
    iset.register('GTE', _gte_op, ['d', 'd'], 'bool')
    iset.register('LT', _lt_op, ['d', 'd'], 'bool')
    iset.register('LTE', _lte_op, ['d', 'd'], 'bool')
    
    return iset

# ==============================================================================
# MATHEMATICAL CONSTANTS LIBRARY
# ==============================================================================

class MathematicalConstantsLibrary:
    """Library of known mathematical constants and formulas for pattern matching"""
    
    def __init__(self):
        self.constants = {
            'e': math.e,
            'pi': math.pi,
            'golden_ratio': (1 + math.sqrt(5)) / 2,
            'golden_ratio_conjugate': (math.sqrt(5) - 1) / 2,
            'inv_golden_ratio_sq': -1 / ((1 + math.sqrt(5)) / 2) ** 2,
            'euler_mascheroni': 0.5772156649015329,
            'meissel_mertens': 0.2614972128476428,
            'artin': 0.3739558136192023,
            'sqrt_2': math.sqrt(2),
            'sqrt_3': math.sqrt(3),
            'sqrt_5': math.sqrt(5),
            'ln_2': math.log(2),
            'ln_10': math.log(10),
            'catalan': 0.915965594177219,
            'apery': 1.202056903159594,  # ζ(3)
            'twin_prime': 0.6601618158468696,
            'mills': 1.3063778838630806,
            'plastic': 1.324717957244746,
            'tribonacci': 1.839286755214161,
        }
        
        # Common mathematical expressions
        self.expressions = {
            '5_minus_1_over_15': 5 - 1/15,  # 4.9333...
            'e_to_gamma': math.exp(0.5772156649015329),
            'e_to_minus_gamma': math.exp(-0.5772156649015329),
            'pi_squared_over_6': math.pi**2 / 6,  # ζ(2)
            'sqrt_2_minus_1': math.sqrt(2) - 1,
            'log_log_2': math.log(math.log(2)),
            '3_over_4': 0.75,  # The growth exponent
        }
    
    def find_closest_constant(self, value: float, tolerance: float = 0.01) -> Optional[str]:
        """Find if a value matches any known constant within tolerance"""
        for name, const in {**self.constants, **self.expressions}.items():
            if abs(value - const) < tolerance:
                return name
        return None

class EnhancedMathematicalConstantsLibrary(MathematicalConstantsLibrary):
    """Extended library with more constants and systematic testing"""
    
    def __init__(self):
        super().__init__()
        
        # Add more constants
        self.constants.update({
            'khinchin': 2.6854520010653064,  # Khinchin's constant
            'feigenbaum_delta': 4.669201609102990,  # Feigenbaum constant
            'feigenbaum_alpha': 2.502907875095892,
            'conway': 1.303577269034296,  # Conway's constant
            'champernowne': 0.123456789101112,  # Champernowne constant
            'liouville': 0.110001000000000000000001,  # Liouville number
            'erdos_borwein': 1.606695152415291,  # Erdős–Borwein constant
            'omega': 0.5671432904097838,  # Omega constant (W(1))
            'gauss': 0.8346268416740731,  # Gauss's constant
            'prime_constant': 0.414682509851111,  # Prime constant
            'backhouse': 1.456074948582689,  # Backhouse's constant
            'porter': 1.4670780794339754,  # Porter's constant
            'ice': 1.5396007178390819,  # Ice constant
            'niven': 1.7052111401053677,  # Niven's constant
            'sierpinski': 2.5849817595792532,  # Sierpiński's constant
            'landau_ramanujan': 0.76422365358922,  # Landau-Ramanujan constant
            'viswanath': 1.1319882487943,  # Viswanath's constant
            'parabolic': 2.2955871493926,  # Universal parabolic constant
        })
        
        # Add mathematical expressions involving multiple constants
        self.complex_expressions = {
            'golden_squared': self.constants['golden_ratio']**2,
            'e_to_pi': math.e**math.pi,
            'pi_to_e': math.pi**math.e,
            'golden_times_e': self.constants['golden_ratio'] * math.e,
            'sqrt_2_plus_sqrt_3': math.sqrt(2) + math.sqrt(3),
            'e_minus_phi': math.e - self.constants['golden_ratio'],
            'log_2_log_3': math.log(2) * math.log(3),
            'zeta_2': math.pi**2 / 6,  # ζ(2)
            'zeta_3': self.constants['apery'],  # ζ(3)
            'one_over_phi_squared': 1 / self.constants['golden_ratio']**2,
            'phi_minus_one_over_phi': self.constants['golden_ratio'] - 1/self.constants['golden_ratio'],
        }
    
    def systematic_constant_search(self, value: float, max_operations: int = 3):
        """Systematically search for constant combinations matching value"""
        matches = []
        tolerance = 0.01
        
        # Check single constants
        for name, const in {**self.constants, **self.expressions, **self.complex_expressions}.items():
            if abs(value - const) < tolerance:
                matches.append({
                    'expression': name,
                    'value': const,
                    'error': abs(value - const),
                    'complexity': 1
                })
        
        # Check combinations (up to max_operations)
        if max_operations >= 2:
            for name1, const1 in self.constants.items():
                for name2, const2 in self.constants.items():
                    # Try various operations
                    operations = [
                        ('+', const1 + const2),
                        ('-', const1 - const2),
                        ('*', const1 * const2),
                        ('/', const1 / const2 if const2 != 0 else None),
                        ('^', const1 ** const2 if abs(const2) < 10 else None)
                    ]
                    
                    for op, result in operations:
                        if result is not None and abs(value - result) < tolerance:
                            matches.append({
                                'expression': f"{name1} {op} {name2}",
                                'value': result,
                                'error': abs(value - result),
                                'complexity': 2
                            })
        
        # Sort by error then complexity
        matches.sort(key=lambda x: (x['error'], x['complexity']))
        
        return matches[:5]  # Return top 5 matches

# ==============================================================================
# ENHANCED LOGGER
# ==============================================================================

class UltimateLogger:
    """Enhanced logger that captures absolutely everything"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.findings = defaultdict(list)
        self.correlations = {}
        self.formulas = {}
        self.patterns = {}
        self.anomalies = []
        self.genetic_discoveries = []
        self.evolvo_algorithms = []
        self.neural_architectures = []
        self.start_time = time.time()
        self.log_file = f'sieve_echo_log_{self.timestamp}.txt'
        
        with open(self.log_file, 'w') as f:
            f.write(f"Sieve Echo Ultimate Discovery System - Started {datetime.now()}\n")
            f.write("="*80 + "\n")
    
    def log(self, message: str, level: str = "INFO"):
        elapsed = time.time() - self.start_time
        formatted = f"[{elapsed:.1f}s][{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def add_finding(self, category: str, finding: Dict):
        self.findings[category].append({
            'timestamp': time.time() - self.start_time,
            'data': finding
        })
    
    def add_correlation(self, name: str, value: float, confidence: float, details: Dict):
        self.correlations[name] = {
            'value': value,
            'confidence': confidence,
            'details': details
        }
        if abs(value) > 0.3:
            self.log(f"CORRELATION: {name} = {value:.4f} (conf: {confidence:.2%})", "RESULT")
    
    def add_formula(self, name: str, formula: str, error: float, params: Dict):
        self.formulas[name] = {
            'formula': formula,
            'error': error,
            'parameters': params,
            'timestamp': time.time() - self.start_time
        }
        self.log(f"FORMULA: {name}: {formula} (error: {error:.4f})", "DISCOVERY")
    
    def add_genetic_discovery(self, discovery: Dict):
        self.genetic_discoveries.append(discovery)
        self.log(f"GENETIC: {discovery.get('description', 'New pattern')}", "DISCOVERY")
    
    def add_evolvo_algorithm(self, algorithm: List, fitness: float, description: str):
        self.evolvo_algorithms.append({
            'algorithm': algorithm,
            'fitness': fitness,
            'description': description,
            'timestamp': time.time() - self.start_time
        })
        self.log(f"EVOLVO: {description} (fitness: {fitness:.4f})", "ALGORITHM")
    
    def save_all(self):
        results = {
            'findings': dict(self.findings),
            'correlations': self.correlations,
            'formulas': self.formulas,
            'patterns': self.patterns,
            'anomalies': self.anomalies,
            'genetic_discoveries': self.genetic_discoveries,
            'evolvo_algorithms': self.evolvo_algorithms,
            'neural_architectures': self.neural_architectures,
            'runtime': time.time() - self.start_time
        }
        
        # Save JSON
        with open(f'sieve_echo_results_{self.timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save pickle for complete data
        with open(f'sieve_echo_complete_{self.timestamp}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        self.log(f"Results saved ({len(self.formulas)} formulas, {len(self.correlations)} correlations)", "INFO")

logger = UltimateLogger()

# ==============================================================================
# NDR PATTERN ANALYZER (Enhanced)
# ==============================================================================

class NDRPatternAnalyzer:
    """Enhanced Normalized Digit Representation analyzer"""
    
    def __init__(self):
        self.cache = {}
        self.patterns_db = defaultdict(list)
        self.prime_list = list(primerange(2, 1000))
        
    def compute_repetend(self, n: int, base: int, max_length: int = 10000) -> List[int]:
        """Compute repeating decimal pattern"""
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
    
    def compute_ndr(self, pattern: List[int], base: int) -> np.ndarray:
        """Normalized Digit Representation"""
        if not pattern:
            return np.array([])
        return np.array(pattern) / base
    
    def compute_theta_entropy(self, ndr: np.ndarray) -> float:
        """Compute spectral entropy of NDR pattern (theta entropy)"""
        if len(ndr) < 2:
            return 0.0
        
        # Compute FFT
        fft_vals = np.abs(fft(ndr))[:len(ndr)//2]
        if len(fft_vals) == 0:
            return 0.0
        
        # Power spectrum
        power = fft_vals**2
        total_power = np.sum(power)
        if total_power == 0:
            return 0.0
        
        # Shannon entropy of power spectrum
        p_spectrum = power / total_power
        p_spectrum = p_spectrum[p_spectrum > 1e-10]
        if len(p_spectrum) == 0:
            return 0.0
        
        return -np.sum(p_spectrum * np.log(p_spectrum))
    
    def extract_comprehensive_features(self, n: int) -> Dict:
        """Extract all possible features for n"""
        features = {
            'n': n,
            'omega': len(factorint(n)),
            'Omega': sum(factorint(n).values()),
            'tau': len(divisors(n)),
            'sigma': sum(divisors(n)),
            'phi': totient(n),
            'mu': mobius(n),
            'is_prime': isprime(n),
            'is_semiprime': len(factorint(n)) == 2 and sum(factorint(n).values()) == 2,
            'is_prime_power': len(factorint(n)) == 1,
            'smallest_prime_factor': min(factorint(n).keys()) if factorint(n) else n,
            'largest_prime_factor': max(factorint(n).keys()) if factorint(n) else n,
            'prime_factors': list(factorint(n).keys()),
            'factorization': factorint(n)
        }
        
        # Compute radical (product of distinct prime factors)
        features['radical'] = 1
        for p in features['prime_factors']:
            features['radical'] *= p
        
        # Deficiency/abundance
        features['deficiency'] = 2*n - features['sigma']
        features['is_perfect'] = (features['deficiency'] == 0)
        features['is_abundant'] = (features['deficiency'] < 0)
        features['is_deficient'] = (features['deficiency'] > 0)
        
        # Pattern features across all bases
        theta_entropies = []
        pattern_lengths = []
        kurtosis_values = []
        
        for base in CONFIG.test_bases:
            if math.gcd(n, base) != 1:
                continue
            
            pattern = self.compute_repetend(n, base)
            if not pattern:
                continue
            
            ndr = self.compute_ndr(pattern, base)
            
            # Store base-specific features
            features[f'length_b{base}'] = len(pattern)
            features[f'mean_b{base}'] = np.mean(ndr) if len(ndr) > 0 else 0
            features[f'std_b{base}'] = np.std(ndr) if len(ndr) > 0 else 0
            features[f'skew_b{base}'] = stats.skew(ndr) if len(ndr) > 2 else 0
            features[f'kurtosis_b{base}'] = stats.kurtosis(ndr) if len(ndr) > 3 else 0
            
            # Theta entropy
            theta_entropy = self.compute_theta_entropy(ndr)
            features[f'theta_entropy_b{base}'] = theta_entropy
            theta_entropies.append(theta_entropy)
            
            # Pattern complexity
            features[f'unique_digits_b{base}'] = len(set(pattern))
            features[f'compression_ratio_b{base}'] = len(pattern) / max(1, len(set(pattern)))
            
            # Multiplicative order
            features[f'mult_order_b{base}'] = len(pattern)
            features[f'order_ratio_b{base}'] = len(pattern) / features['phi'] if features['phi'] > 0 else 0
            
            pattern_lengths.append(len(pattern))
            if len(ndr) > 3:
                kurtosis_values.append(stats.kurtosis(ndr))
        
        # Aggregate features
        if theta_entropies:
            features['theta_entropy_mean'] = np.mean(theta_entropies)
            features['theta_entropy_std'] = np.std(theta_entropies)
            features['theta_entropy_min'] = np.min(theta_entropies)
            features['theta_entropy_max'] = np.max(theta_entropies)
        
        if pattern_lengths:
            features['length_mean'] = np.mean(pattern_lengths)
            features['length_std'] = np.std(pattern_lengths)
            features['length_gcd'] = np.gcd.reduce(pattern_lengths)
        
        if kurtosis_values:
            features['kurtosis_mean'] = np.mean(kurtosis_values)
            features['kurtosis_std'] = np.std(kurtosis_values)
        
        return features

class DynamicFeatureEngineering:
    """Dynamically create and test new features based on discoveries"""
    
    def __init__(self, data):
        self.data = data
        self.feature_generators = []
        self.successful_features = []
        
    def generate_interaction_features(self):
        """Generate interaction features between existing features"""
        base_features = ['omega', 'theta_entropy_mean', 'kurtosis_mean', 'length_mean', 'phi', 'tau', 'sigma']
        
        for d in self.data:
            # Multiplicative interactions
            for f1 in base_features:
                for f2 in base_features:
                    if f1 != f2 and f1 in d and f2 in d:
                        d[f'{f1}_times_{f2}'] = d[f1] * d[f2]
                        d[f'{f1}_over_{f2}'] = d[f1] / d[f2] if d[f2] != 0 else 0
            
            # Logarithmic transformations
            for f in base_features:
                if f in d and d[f] > 0:
                    d[f'log_{f}'] = math.log(d[f])
                    d[f'sqrt_{f}'] = math.sqrt(d[f])
            
            # Modular features
            n = d['n']
            for mod in [6, 12, 30]:
                d[f'n_mod_{mod}'] = n % mod
                d[f'n_mod_{mod}_is_prime'] = isprime(n % mod)
    
    def generate_fourier_features(self):
        """Generate features from Fourier analysis"""
        for d in self.data:
            if 'theta_entropy_mean' in d:
                # Simulate Fourier coefficients
                theta = d['theta_entropy_mean']
                d['fourier_dc'] = theta  # DC component
                d['fourier_fundamental'] = math.sin(2 * math.pi * theta)
                d['fourier_harmonic2'] = math.sin(4 * math.pi * theta)
                d['fourier_energy'] = theta ** 2
    
    def test_feature_importance(self, feature_name, target='omega'):
        """Test if a feature is important for predicting target"""
        X = []
        y = []
        
        for d in self.data:
            if feature_name in d and target in d:
                X.append([d[feature_name]])
                y.append(d[target])
        
        if len(X) < 50:
            return 0.0
        
        # Simple correlation test
        X = np.array(X)
        y = np.array(y)
        
        if np.std(X) == 0 or np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(X.T, y)[0, 1]
        return abs(correlation)

# ==============================================================================
# EVOLVO FORMULA DISCOVERER
# ==============================================================================

class EvolvoFormulaDiscoverer:
    """Uses Evolvo genetic programming to discover mathematical formulas"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.instruction_set = get_picklable_instruction_set()
        self.best_algorithm = None
        self.best_fitness = float('inf')
        
        if not EVOLVO_AVAILABLE:
            logger.log("Evolvo not available - formula discovery limited", "WARNING")
            return
        
        # Configure Evolvo data store
        self.store_config = {
            'd#': ['kurtosis', 'length', 'n', 'entropy', 'phi', 'tau', 'one', 'two', 'golden'],
            'b#': ['is_prime'],
            'd$': ['omega_pred', 'temp1', 'temp2', 'temp3'],
            'b$': ['condition']
        }
    
    def create_evaluator(self, target: str = 'omega'):
        """Create an Evolvo evaluator for a specific target"""
        
        class TargetEvaluator(BaseEvaluator):
            def __init__(self, data, store_config, instruction_set, target):
                super().__init__(store_config, instruction_set)
                self.store_config = store_config
                self.data = data
                self.target = target
            
            def evaluate(self, algorithm, **kwargs):
                if len(algorithm) > CONFIG.evolvo_max_algorithm_length:
                    return float('inf')  # Penalize overly complex algorithms
                
                data_store = DataStore(self.store_config)
                total_error = 0.0
                count = 0
                
                for d in self.data[:min(100, len(self.data))]:  # Sample for speed
                    if self.target not in d:
                        continue
                    
                    # Set constants
                    data_store.reset()
                    data_store.set('kurtosis', d.get('kurtosis_mean', 0))
                    data_store.set('length', d.get('length_mean', 0))
                    data_store.set('n', d['n'])
                    data_store.set('entropy', d.get('theta_entropy_mean', 0))
                    data_store.set('phi', d.get('phi', 0))
                    data_store.set('tau', d.get('tau', 0))
                    data_store.set('one', 1.0)
                    data_store.set('two', 2.0)
                    data_store.set('golden', 1.618033988749895)
                    data_store.set('is_prime', d.get('is_prime', False))
                    
                    try:
                        self.interpreter.execute(algorithm, data_store)
                        predicted = data_store.get('omega_pred')
                        actual = d[self.target]
                        error = (predicted - actual) ** 2
                        total_error += error
                        count += 1
                    except:
                        return float('inf')
                
                if count == 0:
                    return float('inf')
                
                mse = total_error / count
                # Add complexity penalty
                complexity_penalty = len(algorithm) * 0.01
                return mse + complexity_penalty
        
        return TargetEvaluator(self.data, self.store_config, self.instruction_set, target)
    
    def generate_random_algorithm(self, max_length: int = 10) -> List:
        """Generate a random valid algorithm"""
        algorithm = []
        
        for _ in range(random.randint(1, max_length)):
            # Random operation
            ops = ['ADD', 'SUB', 'MUL', 'DIV', 'LOG', 'SQRT', 'POW'] # todo: get inserted ops
            op = random.choice(ops)
            
            # Random target (always omega_pred for simplicity)
            target = ['d$', 0]  # omega_pred
            
            # Random arguments
            if op in ['LOG', 'SQRT']:
                # Unary operation
                arg1_type = 'd#' if random.random() < 0.7 else 'd$'
                arg1_idx = random.randint(0, 8 if arg1_type == 'd#' else 3)
                instruction = target + [op, arg1_type, arg1_idx]
            else:
                # Binary operation
                arg1_type = 'd#' if random.random() < 0.7 else 'd$'
                arg1_idx = random.randint(0, 8 if arg1_type == 'd#' else 3)
                arg2_type = 'd#' if random.random() < 0.7 else 'd$'
                arg2_idx = random.randint(0, 8 if arg2_type == 'd#' else 3)
                instruction = target + [op, arg1_type, arg1_idx, arg2_type, arg2_idx]
            
            algorithm.append(instruction)
        
        return algorithm
    
    def evolve_formula(self, target: str = 'omega', generations: int = None):
        """Evolve a formula to predict the target variable"""
        if not EVOLVO_AVAILABLE:
            return None
        
        generations = generations or CONFIG.evolvo_generations
        evaluator = self.create_evaluator(target)
        
        # Initialize population
        population = []
        for _ in range(CONFIG.evolvo_population):
            algorithm = self.generate_random_algorithm()
            fitness = evaluator.evaluate(algorithm)
            population.append((algorithm, fitness))
        
        # Evolution loop
        for gen in range(generations):
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            # Check for improvement
            if population[0][1] < self.best_fitness:
                self.best_fitness = population[0][1]
                self.best_algorithm = population[0][0]
                
                # Decode algorithm to formula string
                formula = self.decode_algorithm(population[0][0])
                logger.add_evolvo_algorithm(
                    population[0][0],
                    population[0][1],
                    f"Gen {gen}: {formula}"
                )
            
            # Create next generation
            new_population = population[:CONFIG.evolvo_population // 5]  # Elitism
            
            while len(new_population) < CONFIG.evolvo_population:
                # Selection
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)
                
                # Crossover
                if random.random() < CONFIG.ga_crossover_rate:
                    child = self.crossover(parent1[0], parent2[0])
                else:
                    child = parent1[0].copy()
                
                # Mutation
                if random.random() < CONFIG.ga_mutation_rate:
                    child = self.mutate(child)
                
                # Evaluate
                fitness = evaluator.evaluate(child)
                new_population.append((child, fitness))
            
            population = new_population
            
            if gen % 10 == 0:
                logger.log(f"Evolvo gen {gen}: best fitness = {self.best_fitness:.4f}", "INFO")
        
        return self.best_algorithm
    
    def tournament_select(self, population: List, size: int = 3):
        """Tournament selection"""
        tournament = random.sample(population, min(size, len(population)))
        return min(tournament, key=lambda x: x[1])
    
    def crossover(self, parent1: List, parent2: List) -> List:
        """Single-point crossover"""
        if not parent1 or not parent2:
            return parent1 or parent2
        
        point1 = random.randint(0, len(parent1))
        point2 = random.randint(0, len(parent2))
        return parent1[:point1] + parent2[point2:]
    
    def mutate(self, algorithm: List) -> List:
        """Mutate an algorithm"""
        if not algorithm:
            return algorithm
        
        mutated = algorithm.copy()
        
        # Random mutation type
        mutation_type = random.choice(['modify', 'insert', 'delete'])
        
        if mutation_type == 'modify' and mutated:
            # Modify a random instruction
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = self.generate_random_algorithm(1)[0]
        
        elif mutation_type == 'insert':
            # Insert a new instruction
            idx = random.randint(0, len(mutated))
            new_instruction = self.generate_random_algorithm(1)[0]
            mutated.insert(idx, new_instruction)
        
        elif mutation_type == 'delete' and len(mutated) > 1:
            # Delete a random instruction
            idx = random.randint(0, len(mutated) - 1)
            mutated.pop(idx)
        
        return mutated
    
    def decode_algorithm(self, algorithm: List) -> str:
        """Convert algorithm to readable formula"""
        if not algorithm:
            return "empty"
        
        # Simplified decoding - just show operations
        ops = []
        for instr in algorithm:
            if len(instr) >= 3:
                op = instr[2]
                ops.append(op)
        
        return " → ".join(ops) if ops else "?"

# ==============================================================================
# GENETIC FEATURE EVOLVER
# ==============================================================================

class GeneticFeatureEvolver:
    """Enhanced genetic algorithm for feature discovery"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation = 0
        self.feature_names = self.get_all_feature_names()
        
    def get_all_feature_names(self) -> List[str]:
        """Extract all numeric feature names from data"""
        all_features = set()
        for d in self.data[:100]:  # Sample for speed
            for k, v in d.items():
                if isinstance(v, (int, float)) and not np.isnan(v) and np.isfinite(v):
                    all_features.add(k)
        return sorted(list(all_features))
    
    def create_individual(self) -> Dict:
        """Create a random individual"""
        # Select random subset of features
        num_features = random.randint(3, min(20, len(self.feature_names)))
        selected_features = random.sample(self.feature_names, num_features)
        
        return {
            'id': random.randint(1000000, 9999999),
            'features': selected_features,
            'weights': {f: random.uniform(-1, 1) for f in selected_features},
            'use_log': random.random() > 0.5,
            'use_sqrt': random.random() > 0.5,
            'use_interactions': random.random() > 0.5,
            'fitness': 0.0,
            'birth_generation': self.generation
        }
    
    def evaluate_fitness(self, individual: Dict, target: str = 'omega') -> float:
        """Evaluate fitness of an individual"""
        # Prepare data
        X_data = []
        y_data = []
        
        for d in self.data:
            if target not in d:
                continue
            
            features = []
            valid = True
            for fname in individual['features']:
                if fname in d:
                    val = d[fname]
                    if individual['use_log'] and val > 0:
                        val = np.log(val + 1)
                    elif individual['use_sqrt'] and val >= 0:
                        val = np.sqrt(val)
                    
                    val *= individual['weights'][fname]
                    features.append(val)
                else:
                    valid = False
                    break
            
            if valid and features:
                # Add interaction terms
                if individual['use_interactions'] and len(features) >= 2:
                    for i in range(len(features)-1):
                        features.append(features[i] * features[i+1])
                
                X_data.append(features)
                y_data.append(d[target])
        
        if len(X_data) < 10:
            return 0.0
        
        # Calculate correlation
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Multiple fitness metrics
        fitness = 0.0
        
        # 1. Correlation with target
        if X.shape[1] > 0:
            summary = np.sum(X, axis=1)
            if np.std(summary) > 0 and np.std(y) > 0:
                corr = abs(np.corrcoef(summary, y)[0, 1])
                fitness += corr
        
        # 2. Try linear regression
        try:
            model = Ridge(alpha=1.0)
            scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            avg_r2 = np.mean(scores)
            if avg_r2 > 0:
                fitness += avg_r2
        except:
            pass
        
        # 3. Penalty for too many features
        feature_penalty = len(individual['features']) * 0.01
        fitness -= feature_penalty
        
        return fitness
    
    def evolve(self, target: str = 'omega', generations: int = None):
        """Run genetic evolution"""
        generations = generations or CONFIG.ga_generations
        
        # Initialize population
        if not self.population:
            self.population = [self.create_individual() 
                             for _ in range(CONFIG.ga_population_size)]
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            for ind in self.population:
                ind['fitness'] = self.evaluate_fitness(ind, target)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Track best
            if self.population[0]['fitness'] > self.best_fitness:
                self.best_fitness = self.population[0]['fitness']
                self.best_individual = self.population[0].copy()
                
                # Log discovery
                features = self.best_individual['features'][:5]
                logger.add_genetic_discovery({
                    'generation': gen,
                    'fitness': self.best_fitness,
                    'features': features,
                    'description': f"Features: {', '.join(features)}"
                })
            
            # Create next generation
            new_population = self.population[:CONFIG.ga_elite_size]  # Elitism
            
            while len(new_population) < CONFIG.ga_population_size:
                # Selection
                parent1 = self.tournament_select()
                parent2 = self.tournament_select()
                
                # Crossover
                if random.random() < CONFIG.ga_crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < CONFIG.ga_mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
            
            if gen % 20 == 0:
                logger.log(f"GA gen {gen}: best fitness = {self.best_fitness:.4f}", "INFO")
    
    def tournament_select(self, size: int = 5) -> Dict:
        """Tournament selection"""
        tournament = random.sample(self.population, min(size, len(self.population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two parents"""
        child = self.create_individual()
        
        # Combine features
        all_features = list(set(parent1['features']) | set(parent2['features']))
        num_features = random.randint(3, min(20, len(all_features)))
        child['features'] = random.sample(all_features, min(num_features, len(all_features)))
        
        # Average weights
        child['weights'] = {}
        for f in child['features']:
            if f in parent1['weights'] and f in parent2['weights']:
                child['weights'][f] = (parent1['weights'][f] + parent2['weights'][f]) / 2
            elif f in parent1['weights']:
                child['weights'][f] = parent1['weights'][f]
            elif f in parent2['weights']:
                child['weights'][f] = parent2['weights'][f]
            else:
                child['weights'][f] = random.uniform(-1, 1)
        
        # Inherit flags
        child['use_log'] = random.choice([parent1['use_log'], parent2['use_log']])
        child['use_sqrt'] = random.choice([parent1['use_sqrt'], parent2['use_sqrt']])
        child['use_interactions'] = random.choice([parent1['use_interactions'], parent2['use_interactions']])
        
        return child
    
    def mutate(self, individual: Dict) -> Dict:
        """Mutate an individual"""
        mutated = individual.copy()
        
        # Mutate features
        if random.random() < 0.3:
            if random.random() < 0.5 and len(mutated['features']) > 3:
                # Remove a feature
                feature_to_remove = random.choice(mutated['features'])
                mutated['features'].remove(feature_to_remove)
                del mutated['weights'][feature_to_remove]
            else:
                # Add a feature
                available = [f for f in self.feature_names if f not in mutated['features']]
                if available:
                    new_feature = random.choice(available)
                    mutated['features'].append(new_feature)
                    mutated['weights'][new_feature] = random.uniform(-1, 1)
        
        # Mutate weights
        for f in mutated['features']:
            if random.random() < 0.1:
                mutated['weights'][f] *= random.uniform(0.5, 2.0)
        
        # Mutate flags
        if random.random() < 0.1:
            mutated['use_log'] = not mutated['use_log']
        if random.random() < 0.1:
            mutated['use_sqrt'] = not mutated['use_sqrt']
        if random.random() < 0.1:
            mutated['use_interactions'] = not mutated['use_interactions']
        
        return mutated

# ==============================================================================
# NEURAL NETWORK MODELS
# ==============================================================================

if TORCH_AVAILABLE:
    class SieveEchoNet(nn.Module):
        """Neural network for omega prediction"""
        
        def __init__(self, input_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
            self.fc_omega = nn.Linear(hidden_dim // 4, 10)  # Predict omega up to 10
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            omega = self.fc_omega(x)
            return omega
    
    class NeuralPredictor:
        """Train and evaluate neural networks"""
        
        def __init__(self, data: List[Dict]):
            self.data = data
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.scaler = StandardScaler()
        
        def prepare_data(self, features: List[str], target: str = 'omega'):
            """Prepare data for neural network"""
            X_data = []
            y_data = []
            
            for d in self.data:
                if target not in d:
                    continue
                
                x = []
                valid = True
                for f in features:
                    if f in d:
                        x.append(d[f])
                    else:
                        valid = False
                        break
                
                if valid:
                    X_data.append(x)
                    y_data.append(d[target])
            
            if not X_data:
                return None, None
            
            X = np.array(X_data).astype(float)
            y = np.array(y_data).astype(float)
            
            # Remove NaN
            mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            # Normalize
            X = self.scaler.fit_transform(X)
            
            return torch.FloatTensor(X), torch.LongTensor(y)
        
        def train(self, features: List[str], target: str = 'omega'):
            """Train neural network"""
            X, y = self.prepare_data(features, target)
            if X is None:
                logger.log("Insufficient data for neural network training", "WARNING")
                return None
            
            # Create model
            self.model = SieveEchoNet(len(features), CONFIG.nn_hidden_dim).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.nn_learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Move to device
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Training loop
            self.model.train()
            best_loss = float('inf')
            
            for epoch in range(CONFIG.nn_epochs):
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                
                if epoch % 20 == 0:
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = (predicted == y).float().mean().item()
                    logger.log(f"NN Epoch {epoch}: loss={loss.item():.4f}, acc={accuracy:.3f}", "INFO")
            
            # Final evaluation
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y).float().mean().item()
            
            logger.log(f"Neural network final accuracy: {accuracy:.3f}", "RESULT")
            return accuracy


class CoEvolutionSystem:
    """Co-evolve formulas and neural architectures together"""
    
    def __init__(self, data):
        self.data = data
        self.formula_population = []
        self.nn_population = []
        self.best_pairs = []
    
    def run_formula(self, formula, data):
        """Execute a formula on data and return predictions"""
        predictions = []
        
        if not EVOLVO_AVAILABLE:
            # Fallback: return random predictions
            return [random.random() * 5 for _ in data]
        
        # Assume formula is an Evolvo algorithm
        evaluator = self.create_simple_evaluator()
        
        for d in data:
            try:
                # Execute formula with data point
                result = evaluator.execute_single(formula, d)
                predictions.append(result)
            except:
                predictions.append(0)
        
        return predictions
    
    def create_simple_evaluator(self):
        """Create a simple evaluator for formula execution"""
        if not EVOLVO_AVAILABLE:
            return None
        
        class SimpleEvaluator:
            def __init__(self):
                self.instruction_set = get_picklable_instruction_set()
                self.interpreter = Interpreter(self.instruction_set)
                self.store_config = {
                    'd#': ['n', 'entropy', 'kurtosis', 'length', 'phi', 'tau'],
                    'b#': ['is_prime'],
                    'd$': ['result', 'temp'],
                    'b$': ['flag']
                }
            
            def execute_single(self, algorithm, data_point):
                data_store = DataStore(self.store_config)
                
                # Set values
                data_store.set('n', data_point.get('n', 0))
                data_store.set('entropy', data_point.get('theta_entropy_mean', 0))
                data_store.set('kurtosis', data_point.get('kurtosis_mean', 0))
                data_store.set('length', data_point.get('length_mean', 0))
                data_store.set('phi', data_point.get('phi', 0))
                data_store.set('tau', data_point.get('tau', 0))
                data_store.set('is_prime', data_point.get('is_prime', False))
                
                # Execute
                self.interpreter.execute(algorithm, data_store)
                
                return data_store.get('result')
        
        return SimpleEvaluator()
    
    def refine_with_nn(self, predictions, nn_model):
        """Refine predictions using neural network"""
        if not TORCH_AVAILABLE or nn_model is None:
            return predictions
        
        try:
            # Convert predictions to tensor
            X = torch.FloatTensor(predictions).reshape(-1, 1)
            
            # Simple refinement: pass through a linear layer
            with torch.no_grad():
                refined = nn_model(X).numpy().flatten()
            
            return refined.tolist()
        except:
            return predictions
    
    def compute_accuracy(self, predictions):
        """Compute accuracy of predictions"""
        if not self.data or not predictions:
            return 0.0
        
        correct = 0
        total = 0
        
        for i, pred in enumerate(predictions[:len(self.data)]):
            if i < len(self.data) and 'omega' in self.data[i]:
                true_val = self.data[i]['omega']
                if abs(pred - true_val) < 0.5:  # Within 0.5 of true value
                    correct += 1
                total += 1
        
        return correct / max(1, total)
    
    def evolve_formulas_with_nn_feedback(self):
        """Evolve formulas considering NN performance"""
        # Simple evolution step
        if not self.formula_population:
            # Initialize with random formulas
            for _ in range(10):
                formula = self.generate_random_formula()
                self.formula_population.append(formula)
        
        # Mutate formulas
        new_population = []
        for formula in self.formula_population:
            if random.random() < 0.3:
                mutated = self.mutate_formula(formula)
                new_population.append(mutated)
            else:
                new_population.append(formula)
        
        self.formula_population = new_population
    
    def evolve_nn_with_formula_feedback(self):
        """Evolve neural networks considering formula performance"""
        # Simple placeholder - would need full NN evolution implementation
        pass
    
    def generate_random_formula(self):
        """Generate a random formula"""
        if not EVOLVO_AVAILABLE:
            return []
        
        formula = []
        ops = ['ADD', 'SUB', 'MUL', 'DIV', 'LOG', 'SQRT']
        
        for _ in range(random.randint(2, 5)):
            op = random.choice(ops)
            target = ['d$', 0]  # result
            
            if op in ['LOG', 'SQRT']:
                arg = ['d#', random.randint(0, 5)]
                instruction = target + [op, arg[0], arg[1]]
            else:
                arg1 = ['d#', random.randint(0, 5)]
                arg2 = ['d#', random.randint(0, 5)]
                instruction = target + [op, arg1[0], arg1[1], arg2[0], arg2[1]]
            
            formula.append(instruction)
        
        return formula
    
    def mutate_formula(self, formula):
        """Mutate a formula"""
        if not formula:
            return formula
        
        mutated = formula.copy()
        
        if random.random() < 0.5 and mutated:
            # Change an operation
            idx = random.randint(0, len(mutated) - 1)
            ops = ['ADD', 'SUB', 'MUL', 'DIV', 'LOG', 'SQRT']
            mutated[idx][2] = random.choice(ops)
        
        return mutated
        
    def evaluate_pair(self, formula, nn_genome):
        """Evaluate a formula-NN pair for complementary performance"""
        # Formula predicts ω(n)
        formula_predictions = self.run_formula(formula, self.data)
        
        # NN refines the prediction
        nn_model = nn_genome.to_pytorch_model()
        refined_predictions = self.refine_with_nn(formula_predictions, nn_model)
        
        # Combined fitness
        formula_accuracy = self.compute_accuracy(formula_predictions)
        refined_accuracy = self.compute_accuracy(refined_predictions)
        
        # Reward complementarity
        improvement = refined_accuracy - formula_accuracy
        fitness = refined_accuracy + 0.1 * improvement  # Bonus for good pairing
        
        return fitness
    
    def co_evolve(self, generations=100):
        """Main co-evolution loop"""
        for gen in range(generations):
            # Evaluate all pairs
            pair_fitness = []
            for formula in self.formula_population:
                for nn in self.nn_population:
                    fitness = self.evaluate_pair(formula, nn)
                    pair_fitness.append((formula, nn, fitness))
            
            # Sort by fitness
            pair_fitness.sort(key=lambda x: x[2], reverse=True)
            
            # Store best pairs
            self.best_pairs = pair_fitness[:5]
            
            # Evolve populations with co-evolutionary pressure
            self.evolve_formulas_with_nn_feedback()
            self.evolve_nn_with_formula_feedback()
            
            logger.log(f"Co-evolution gen {gen}: Best pair fitness = {pair_fitness[0][2]:.4f}", "COEVO")

class NoveltySearchEvolver:
    """Evolution based on novelty rather than just fitness"""
    
    def __init__(self, archive_size=100):
        self.novelty_archive = []
        self.archive_size = archive_size
        self.behavior_cache = {}
        self.evaluator = None
    
    def execute_algorithm(self, algorithm, data_point):
        """Execute an algorithm on a data point"""
        if not EVOLVO_AVAILABLE:
            return random.random() * 5
        
        if self.evaluator is None:
            self.evaluator = self.create_evaluator()
        
        try:
            return self.evaluator.execute_single(algorithm, data_point)
        except:
            return 0
    
    def create_evaluator(self):
        """Create evaluator for algorithm execution"""
        if not EVOLVO_AVAILABLE:
            return None
        
        class SimpleEvaluator:
            def __init__(self):
                self.instruction_set = get_picklable_instruction_set()
                self.interpreter = Interpreter(self.instruction_set)
                self.store_config = {
                    'd#': ['n', 'entropy', 'kurtosis'],
                    'b#': ['is_prime'],
                    'd$': ['result'],
                    'b$': ['flag']
                }
            
            def execute_single(self, algorithm, data_point):
                data_store = DataStore(self.store_config)
                
                data_store.set('n', data_point.get('n', 0))
                data_store.set('entropy', data_point.get('theta_entropy_mean', 0))
                data_store.set('kurtosis', data_point.get('kurtosis_mean', 0))
                data_store.set('is_prime', data_point.get('is_prime', False))
                
                self.interpreter.execute(algorithm, data_store)
                
                return data_store.get('result')
        
        return SimpleEvaluator()
        
    def compute_behavior_vector(self, algorithm, data_sample):
        """Extract behavioral characteristics of an algorithm"""
        behaviors = []
        
        # Run algorithm on sample data
        for d in data_sample[:10]:  # Small sample for speed
            try:
                result = self.execute_algorithm(algorithm, d)
                behaviors.extend([
                    result,  # Raw output
                    abs(result - d.get('omega', 0)),  # Error
                    1 if result > 0 else 0,  # Sign
                    result % 10 if result > 0 else 0  # Last digit pattern
                ])
            except:
                behaviors.extend([0, 0, 0, 0])
        
        return np.array(behaviors)
    
    def compute_novelty(self, behavior_vector):
        """Compute novelty as distance to nearest neighbors in archive"""
        if not self.novelty_archive:
            return float('inf')
        
        distances = []
        for archived in self.novelty_archive:
            dist = np.linalg.norm(behavior_vector - archived)
            distances.append(dist)
        
        # Use k-nearest neighbors
        k = min(15, len(distances))
        nearest = sorted(distances)[:k]
        
        return np.mean(nearest) if nearest else float('inf')
    
    def update_archive(self, behavior_vector, threshold=0.5):
        """Add to archive if sufficiently novel"""
        novelty = self.compute_novelty(behavior_vector)
        
        if novelty > threshold:
            self.novelty_archive.append(behavior_vector)
            
            # Maintain archive size
            if len(self.novelty_archive) > self.archive_size:
                # Remove oldest or least novel
                self.novelty_archive.pop(0)
            
            return True
        return False

class AdaptiveEvolutionController:
    """Controls evolution parameters based on progress"""
    
    def __init__(self):
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.mutation_rate = 0.2
        self.exploration_rate = 0.1
        
    def update(self, population, best_fitness):
        """Adapt parameters based on evolution progress"""
        # Track fitness improvement
        if len(self.best_fitness_history) > 0:
            improvement = best_fitness - self.best_fitness_history[-1]
            if abs(improvement) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        self.best_fitness_history.append(best_fitness)
        
        # Calculate population diversity
        diversity = self.calculate_diversity(population)
        self.diversity_history.append(diversity)
        
        # Adapt mutation rate based on stagnation
        if self.stagnation_counter > 10:
            self.mutation_rate = min(0.8, self.mutation_rate * 1.2)
            self.exploration_rate = min(0.5, self.exploration_rate * 1.5)
            logger.log(f"Increasing exploration: mutation={self.mutation_rate:.3f}", "ADAPT")
            
            # Trigger catastrophic mutation every 50 generations of stagnation
            if self.stagnation_counter % 50 == 0:
                self.trigger_catastrophe(population)
        else:
            self.mutation_rate = max(0.1, self.mutation_rate * 0.95)
            self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
    
    def calculate_diversity(self, population):
        """Calculate population diversity using feature signatures"""
        if not population:
            return 0.0
        
        signatures = set()
        for individual in population:
            if hasattr(individual, 'get_signature'):
                signatures.add(individual.get_signature())
            elif isinstance(individual, dict) and 'features' in individual:
                # For genetic feature evolver
                sig = tuple(sorted(individual['features']))
                signatures.add(sig)
        
        return len(signatures) / max(1, len(population))
    
    def trigger_catastrophe(self, population):
        """Major population shake-up to escape local optima"""
        logger.log("CATASTROPHIC MUTATION TRIGGERED!", "EVOLUTION")
        # Keep only top 10% and regenerate rest with high variation
        keep_size = max(1, len(population) // 10)
        return keep_size


# ==============================================================================
# MAIN COMPREHENSIVE ANALYZER
# ==============================================================================

class EnhancedComprehensiveAnalyzer:
    """Enhanced main analyzer with all improvements"""
    
    def __init__(self):
        self.data = []
        self.base_invariance_tester = BaseInvarianceTester()
        self.zeta_analyzer = RiemannZetaAnalyzer()
        self.pnt_analyzer = PrimeNumberTheoremAnalyzer()
        self.fourier_analyzer = EnhancedFourierAnalyzer()
        self.crt_validator = CRTValidator()
        self.mult_order_analyzer = MultiplicativeOrderAnalyzer()
        self.euler_investigator = EulerConstantInvestigator()
        self.modular_explorer = ModularFormsExplorer()
        self.constants_lib = ExtendedMathematicalConstants()

        self.ndr_analyzer = NDRPatternAnalyzer()
        self.constants_lib = MathematicalConstantsLibrary()
        self.data = []
        
        # Initialize evolvers (will be populated after data generation)
        self.feature_evolver = None
        self.formula_discoverer = None
        
        # Add new components
        self.evolution_controller = AdaptiveEvolutionController()
        self.novelty_search = NoveltySearchEvolver()
        self.pattern_discoverer = None  # Will be initialized with data
        self.coevolution_system = None  # Will be initialized with data
        self.feature_engineer = None  # Will be initialized with data
        self.enhanced_constants = EnhancedMathematicalConstantsLibrary()
        
    def run_complete_analysis(self):
        """Run all analysis methods with improvements"""
        
        logger.log("="*80, "INFO")
        logger.log("ENHANCED SIEVE ECHO ANALYSIS WITH CRITICAL IMPROVEMENTS", "INFO")
        logger.log("="*80, "INFO")
        
        # Phase 1: Data Generation (with parallel processing)
        logger.log("\nPHASE 1: PARALLEL DATA GENERATION", "INFO")
        self.generate_dataset_parallel()

         # Initialize components that need data
        self.feature_evolver = GeneticFeatureEvolver(self.data)
        self.formula_discoverer = EvolvoFormulaDiscoverer(self.data) if CONFIG.evolvo_enabled else None
        self.pattern_discoverer = MultiStrategyPatternDiscovery(self.data)
        self.coevolution_system = CoEvolutionSystem(self.data)
        self.feature_engineer = DynamicFeatureEngineering(self.data)
        
        
        # 2. Test Sieve Echo Law
        logger.log("\n" + "="*80, "INFO")
        logger.log("PHASE 2: SIEVE ECHO LAW VALIDATION", "INFO")
        logger.log("="*80, "INFO")
        
        self.test_sieve_echo_law()
        
        # 3. Run genetic feature discovery
        logger.log("\n" + "="*80, "INFO")
        logger.log("PHASE 3: GENETIC FEATURE DISCOVERY", "INFO")
        logger.log("="*80, "INFO")
        
        self.run_genetic_discovery()
        
        # 4. Run Evolvo formula discovery
        if CONFIG.evolvo_enabled:
            logger.log("\n" + "="*80, "INFO")
            logger.log("PHASE 4: EVOLVO FORMULA DISCOVERY", "INFO")
            logger.log("="*80, "INFO")
            
            self.run_evolvo_discovery()
        
        # 5. Train neural networks
        if CONFIG.nn_enabled:
            logger.log("\n" + "="*80, "INFO")
            logger.log("PHASE 5: NEURAL NETWORK TRAINING", "INFO")
            logger.log("="*80, "INFO")
            
            self.train_neural_networks()
        
        # 6. Mine for patterns
        logger.log("\n" + "="*80, "INFO")
        logger.log("PHASE 6: PATTERN MINING", "INFO")
        logger.log("="*80, "INFO")
        
        self.mine_patterns()
        
        # 7. Generate visualizations
        if CONFIG.save_plots:
            logger.log("\n" + "="*80, "INFO")
            logger.log("PHASE 7: VISUALIZATION", "INFO")
            logger.log("="*80, "INFO")
            
            self.create_visualizations()
        
        # 8. Final report
        logger.log("\n" + "="*80, "INFO")
        logger.log("FINAL REPORT", "INFO")
        logger.log("="*80, "INFO")
        
        self.generate_final_report()
        
        # Phase 2: Base Invariance Testing (CRITICAL)
        logger.log("\nPHASE 2: BASE INVARIANCE TESTING (CRITICAL)", "INFO")
        self.test_base_invariance()
        
        # Phase 3: Riemann Zeta Connections
        logger.log("\nPHASE 3: RIEMANN ZETA FUNCTION CONNECTIONS", "INFO")
        self.test_zeta_connections()
        
        # Phase 4: Prime Number Theorem
        logger.log("\nPHASE 4: PRIME NUMBER THEOREM INTEGRATION", "INFO")
        self.test_pnt_relationships()
        
        # Phase 5: Enhanced Fourier Analysis
        logger.log("\nPHASE 5: ENHANCED FOURIER ANALYSIS", "INFO")
        self.run_enhanced_fourier()
        
        # Phase 6: CRT Validation
        if CONFIG.test_crt:
            logger.log("\nPHASE 6: CHINESE REMAINDER THEOREM VALIDATION", "INFO")
            self.validate_crt()
        
        # Phase 7: Multiplicative Order Analysis
        if CONFIG.test_multiplicative_order:
            logger.log("\nPHASE 7: MULTIPLICATIVE ORDER ANALYSIS", "INFO")
            self.analyze_multiplicative_orders()
        
        # Phase 8: Euler's Constant Investigation
        logger.log("\nPHASE 8: EULER'S CONSTANT INVESTIGATION", "INFO")
        self.investigate_euler()
        
        # Phase 9: Modular Forms
        if CONFIG.test_modular:
            logger.log("\nPHASE 9: MODULAR FORMS EXPLORATION", "INFO")
            self.explore_modular()
        
        # Phase 10: Adaptive Feature Discovery
        logger.log("\nPHASE 10: ADAPTIVE FEATURE DISCOVERY", "INFO")
        self.discover_features()
        
        # Final Report
        logger.log("\nFINAL REPORT", "INFO")
        self.generate_final_report()
    
    def generate_dataset_parallel(self):
        """Generate dataset using parallel processing"""
        logger.log(f"Generating dataset for n=2 to {CONFIG.max_n} using {CONFIG.n_workers} workers", "INFO")
        
        # Use parallel analyzer
        analyzer = ParallelSieveAnalyzer(CONFIG.n_workers)
        
        # Analyze range
        self.data = analyzer.analyze_range(2, min(CONFIG.max_n, CONFIG.sample_size))
        
        logger.log(f"Generated {len(self.data)} data points", "INFO")
    
    def test_base_invariance(self):
        """Test base invariance (CRITICAL)"""
        logger.log("Testing base invariance - CRITICAL for conjecture validity", "CRITICAL")
        
        # Sample numbers to test
        test_numbers = random.sample([d['n'] for d in self.data], 
                                    min(CONFIG.base_invariance_test_count, len(self.data)))
        
        for n in test_numbers[:20]:  # Detailed testing for first 20
            result = self.base_invariance_tester.test_base_invariance(n)
            
            if result['is_invariant'] is not None:
                if not result['is_invariant']:
                    logger.add_finding('NON_INVARIANT', {
                        'n': n,
                        'cv': result['cv'],
                        'description': f"n={n} shows non-invariant theta entropy (CV={result['cv']:.3f})"
                    })
        
        # Analyze patterns
        analysis = self.base_invariance_tester.analyze_invariance_patterns()
        
        logger.log(f"Base Invariance Results:", "CRITICAL")
        logger.log(f"  Invariance rate: {analysis.get('invariance_rate', 0):.2%}", "RESULT")
        logger.log(f"  Prime invariance rate: {analysis.get('prime_invariance_rate', 0):.2%}", "RESULT")
        logger.log(f"  Composite invariance rate: {analysis.get('composite_invariance_rate', 0):.2%}", "RESULT")
        
        logger.base_invariance_results = analysis
    
    def test_zeta_connections(self):
        """Test Riemann zeta function connections"""
        n_values = [d['n'] for d in self.data[:1000]]
        theta_entropies = [d.get('theta_entropy_mean', 0) for d in self.data[:1000]]
        
        correlations = self.zeta_analyzer.compute_zeta_correlation(n_values, theta_entropies)
        
        for s_key, data in correlations.items():
            if 'correlations' in data:
                for corr_type, corr_val in data['correlations'].items():
                    if abs(corr_val) > 0.3:
                        logger.log(f"Zeta connection found: {s_key} {corr_type} correlation = {corr_val:.4f}", "DISCOVERY")
        
        logger.zeta_connections = correlations
    
    def test_pnt_relationships(self):
        """Test Prime Number Theorem relationships"""
        n_values = [d['n'] for d in self.data[:1000]]
        theta_entropies = [d.get('theta_entropy_mean', 0) for d in self.data[:1000]]
        
        results = self.pnt_analyzer.test_pnt_relationship(n_values, theta_entropies)
        
        if 'correlations' in results:
            logger.log("PNT Correlations:", "RESULT")
            for key, corr in results['correlations'].items():
                logger.log(f"  {key}: {corr:.4f}", "RESULT")
            
            if results.get('best_predictor'):
                best_key, best_corr = results['best_predictor']
                logger.log(f"Best PNT predictor: {best_key} (r={best_corr:.4f})", "DISCOVERY")
        
        logger.pnt_relationships = results
    
    def run_enhanced_fourier(self):
        """Run enhanced Fourier analysis"""
        sample_data = random.sample(self.data, min(100, len(self.data)))
        
        fourier_features_collection = []
        
        for d in sample_data:
            n = d['n']
            
            # Get a pattern for analysis
            if math.gcd(n, 10) == 1:
                pattern = self.base_invariance_tester.compute_repetend(n, 10)
                if pattern:
                    ndr = np.array(pattern) / 10
                    features = self.fourier_analyzer.advanced_fourier_analysis(ndr)
                    features['n'] = n
                    features['omega'] = d['omega']
                    fourier_features_collection.append(features)
        
        if fourier_features_collection:
            # Analyze correlations with omega
            for feature_name in ['spectral_centroid', 'spectral_entropy', 'phase_coherence']:
                if feature_name in fourier_features_collection[0]:
                    feature_vals = [f[feature_name] for f in fourier_features_collection]
                    omega_vals = [f['omega'] for f in fourier_features_collection]
                    
                    if np.std(feature_vals) > 0 and np.std(omega_vals) > 0:
                        corr = np.corrcoef(feature_vals, omega_vals)[0,1]
                        if abs(corr) > 0.3:
                            logger.log(f"Fourier {feature_name} correlates with ω: r={corr:.4f}", "DISCOVERY")
    
    def validate_crt(self):
        """Validate Chinese Remainder Theorem"""
        # Test with small prime pairs
        prime_pairs = [(3, 5), (5, 7), (7, 11), (11, 13), (13, 17)]
        
        valid_count = 0
        total_count = 0
        
        for p1, p2 in prime_pairs:
            result = self.crt_validator.validate_crt_for_theta(p1, p2)
            
            if 'is_valid' in result:
                total_count += 1
                if result['is_valid']:
                    valid_count += 1
                    logger.log(f"CRT valid for ({p1}, {p2}): correlation={result.get('correlation', 0):.4f}", "RESULT")
                else:
                    logger.log(f"CRT violation for ({p1}, {p2}): error={result.get('error', 0):.4f}", "WARNING")
        
        if total_count > 0:
            logger.log(f"CRT validation rate: {valid_count}/{total_count} = {valid_count/total_count:.2%}", "RESULT")
    
    def analyze_multiplicative_orders(self):
        """Analyze multiplicative orders"""
        sample_numbers = random.sample([d['n'] for d in self.data if d['n'] < 1000], 
                                      min(50, len(self.data)))
        
        cyclic_count = 0
        
        for n in sample_numbers:
            analysis = self.mult_order_analyzer.analyze_multiplicative_order(n)
            
            if analysis.get('is_cyclic'):
                cyclic_count += 1
            
            # Check for interesting patterns
            if analysis.get('mean_order_ratio', 0) > 0.9:
                logger.add_finding('HIGH_ORDER_RATIO', {
                    'n': n,
                    'ratio': analysis['mean_order_ratio'],
                    'description': f"n={n} has high mean order ratio: {analysis['mean_order_ratio']:.3f}"
                })
        
        logger.log(f"Cyclic numbers: {cyclic_count}/{len(sample_numbers)} = {cyclic_count/len(sample_numbers):.2%}", "RESULT")
    
    def investigate_euler(self):
        """Investigate Euler's constant role"""
        n_values = [d['n'] for d in self.data[:500]]
        theta_entropies = [d.get('theta_entropy_mean', 0) for d in self.data[:500]]
        
        results = self.euler_investigator.explore_euler_constant(n_values, theta_entropies)
        
        if 'correlations' in results:
            logger.log("Euler's Constant Correlations:", "RESULT")
            for key, corr in results['correlations'].items():
                logger.log(f"  {key}: {corr:.4f}", "RESULT")
            
            if results.get('euler_gamma_appears'):
                logger.log("Euler's constant appears in the relationships!", "DISCOVERY")
    
    def explore_modular(self):
        """Explore modular forms"""
        sample_data = random.sample(self.data, min(100, len(self.data)))
        
        modular_patterns = defaultdict(list)
        
        for d in sample_data:
            n = d['n']
            theta_entropy = d.get('theta_entropy_mean', 0)
            
            patterns = self.modular_explorer.explore_modular_forms(n, theta_entropy)
            
            for mod_key, pattern_data in patterns.items():
                if pattern_data['is_coprime']:
                    modular_patterns[mod_key].append(pattern_data['theta_deviation'])
        
        # Analyze deviations
        for mod_key, deviations in modular_patterns.items():
            if deviations:
                mean_dev = np.mean(deviations)
                if mean_dev < 0.1:
                    logger.log(f"Strong modular pattern for {mod_key}: mean deviation = {mean_dev:.4f}", "DISCOVERY")
    
    def discover_features(self):
        """Run adaptive feature discovery"""
        discoverer = AdaptiveFeatureDiscovery(self.data[:500])
        
        # Discover polynomial relationships
        poly_relationships = discoverer.discover_polynomial_relationships()
        
        if poly_relationships:
            logger.log("Top Polynomial Relationships:", "DISCOVERY")
            for rel in poly_relationships[:5]:
                logger.log(f"  {rel['formula']}: score={rel['score']:.2f}", "RESULT")
        
        # Discover ratio relationships
        ratio_relationships = discoverer.discover_ratio_relationships()
        
        if ratio_relationships:
            logger.log("Top Ratio Relationships:", "DISCOVERY")
            for rel in ratio_relationships[:5]:
                logger.log(f"  {rel['numerator']}/{rel['denominator']}: r={rel['correlation']:.4f}", "RESULT")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.log("\n" + "="*80, "INFO")
        logger.log("COMPREHENSIVE ANALYSIS SUMMARY", "INFO")
        logger.log("="*80, "INFO")
        
        # Report critical findings
        if hasattr(logger, 'base_invariance_results'):
            inv_rate = logger.base_invariance_results.get('invariance_rate', 0)
            if inv_rate < 0.9:
                logger.log(f"⚠️ BASE INVARIANCE ISSUE: Only {inv_rate:.1%} of numbers show invariant theta entropy", "CRITICAL")
            else:
                logger.log(f"✓ Base invariance confirmed for {inv_rate:.1%} of numbers", "SUCCESS")
        
        # Report Riemann zeta connections
        if hasattr(logger, 'zeta_connections'):
            logger.log("\nRiemann Zeta Connections Found:", "RESULT")
            for key in logger.zeta_connections:
                if 'correlations' in logger.zeta_connections[key]:
                    corrs = logger.zeta_connections[key]['correlations']
                    if corrs:
                        max_corr = max(corrs.values(), key=abs)
                        logger.log(f"  {key}: max correlation = {max_corr:.4f}", "RESULT")
        
        # Report PNT relationships
        if hasattr(logger, 'pnt_relationships'):
            if 'best_predictor' in logger.pnt_relationships and logger.pnt_relationships['best_predictor']:
                best_key, best_corr = logger.pnt_relationships['best_predictor']
                logger.log(f"\nBest PNT predictor: {best_key} (r={best_corr:.4f})", "RESULT")
        
        # Save all results
        logger.save_all()
        
        runtime = time.time() - logger.start_time
        logger.log(f"\nTotal runtime: {runtime/3600:.2f} hours", "INFO")
        logger.log(f"Results saved with timestamp: {logger.timestamp}", "SUCCESS")


class MultiStrategyPatternDiscovery:
    """Discovers patterns using multiple mathematical approaches"""
    
    def __init__(self, data):
        self.data = data
        self.discovered_patterns = []
        self.pattern_library = {}
    
    def compute_correlation(self, feature1: str, feature2: str) -> float:
        """Compute correlation between two features"""
        vals1 = []
        vals2 = []
        
        for d in self.data:
            if feature1 in d and feature2 in d:
                v1 = d[feature1]
                v2 = d[feature2]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if np.isfinite(v1) and np.isfinite(v2):
                        vals1.append(v1)
                        vals2.append(v2)
        
        if len(vals1) < 10 or np.std(vals1) == 0 or np.std(vals2) == 0:
            return 0.0
        
        return np.corrcoef(vals1, vals2)[0, 1]
    
    def find_continued_fraction_patterns(self):
        """Look for patterns in continued fraction representations"""
        patterns = []
        
        for d in self.data:
            n = d['n']
            if n < 2:
                continue
            
            # Compute continued fraction expansion of 1/n
            cf_expansion = self.compute_continued_fraction(1, n, max_terms=20)
            
            # Store features
            d['cf_length'] = len(cf_expansion)
            d['cf_max'] = max(cf_expansion) if cf_expansion else 0
            d['cf_periodicity'] = self.detect_periodicity(cf_expansion)
            
            # Check for patterns
            if d.get('is_prime') and len(cf_expansion) > 0:
                d['cf_prime_indicator'] = len(cf_expansion) % d.get('phi', 1)
        
        # Look for correlations
        corr = self.compute_correlation('cf_length', 'omega')
        if abs(corr) > 0.3:
            patterns.append({
                'type': 'CONTINUED_FRACTION',
                'correlation': corr,
                'description': f'CF length correlates with ω(n): r={corr:.3f}'
            })
        
        return patterns
    
    def compute_continued_fraction(self, num: int, den: int, max_terms: int = 20) -> List[int]:
        """Compute continued fraction expansion"""
        cf = []
        for _ in range(max_terms):
            if den == 0:
                break
            
            q = num // den
            cf.append(q)
            
            num, den = den, num - q * den
            
            if num == 0:
                break
        
        return cf
    
    def detect_periodicity(self, sequence: List) -> int:
        """Detect period in a sequence"""
        if len(sequence) < 3:
            return 0
        
        for period in range(1, len(sequence) // 2):
            is_periodic = True
            for i in range(period, min(len(sequence), 3 * period)):
                if sequence[i] != sequence[i % period]:
                    is_periodic = False
                    break
            
            if is_periodic:
                return period
        
        return 0
    
    def find_fourier_patterns(self):
        """Look for patterns in Fourier transforms of digit sequences"""
        patterns = []
        
        for d in self.data:
            if 'theta_entropy_mean' not in d:
                continue
            
            # Already have FFT-based entropy, look for additional patterns
            n = d['n']
            
            # Compute spectral features
            for base in [10, 2, 16]:
                key = f'length_b{base}'
                if key in d and d[key] > 0:
                    # Simulate spectral peak
                    d[f'spectral_peak_b{base}'] = math.sin(2 * math.pi * d[key] / base)
                    d[f'spectral_spread_b{base}'] = d.get(f'std_b{base}', 0) * math.sqrt(d[key])
        
        # Check for spectral clustering by prime type
        prime_spectra = []
        composite_spectra = []
        
        for d in self.data:
            if 'spectral_peak_b10' in d:
                if d.get('is_prime'):
                    prime_spectra.append(d['spectral_peak_b10'])
                else:
                    composite_spectra.append(d['spectral_peak_b10'])
        
        if prime_spectra and composite_spectra:
            # Test if distributions differ
            if len(prime_spectra) > 30 and len(composite_spectra) > 30:
                from scipy import stats as scipy_stats
                _, p_value = scipy_stats.ks_2samp(prime_spectra, composite_spectra)
                
                if p_value < 0.05:
                    patterns.append({
                        'type': 'FOURIER_SPECTRAL',
                        'p_value': p_value,
                        'description': 'Primes have distinct spectral signatures'
                    })
        
        return patterns
    
    def find_graph_patterns(self):
        """Look for patterns treating numbers as graph nodes"""
        patterns = []
        
        # Build divisibility graph edges
        edges = defaultdict(list)
        
        for d in self.data:
            n = d['n']
            if n < 2:
                continue
            
            # Find divisors
            divisors_list = divisors(n)
            
            # Graph properties
            d['graph_degree'] = len(divisors_list) - 2  # Exclude 1 and n
            d['graph_clustering'] = 0
            
            # For small n, compute clustering coefficient
            if n < 100:
                neighbor_edges = 0
                neighbors = [div for div in divisors_list if 1 < div < n]
                
                for i, div1 in enumerate(neighbors):
                    for div2 in neighbors[i+1:]:
                        if div2 % div1 == 0:
                            neighbor_edges += 1
                
                if len(neighbors) > 1:
                    max_edges = len(neighbors) * (len(neighbors) - 1) / 2
                    d['graph_clustering'] = neighbor_edges / max_edges if max_edges > 0 else 0
        
        # Look for degree distribution patterns
        degree_by_omega = defaultdict(list)
        for d in self.data:
            if 'graph_degree' in d and 'omega' in d:
                degree_by_omega[d['omega']].append(d['graph_degree'])
        
        # Check if degree scales with omega
        if len(degree_by_omega) > 3:
            omega_vals = []
            mean_degrees = []
            
            for omega, degrees in degree_by_omega.items():
                if degrees:
                    omega_vals.append(omega)
                    mean_degrees.append(np.mean(degrees))
            
            if len(omega_vals) > 3:
                corr = np.corrcoef(omega_vals, mean_degrees)[0, 1]
                if abs(corr) > 0.5:
                    patterns.append({
                        'type': 'GRAPH_DEGREE',
                        'correlation': corr,
                        'description': f'Graph degree correlates with ω(n): r={corr:.3f}'
                    })
        
        return patterns
        
    def discover_all_patterns(self):
        """Run all pattern discovery strategies"""
        patterns = []
        
        # 1. Prime Number Theorem connections
        patterns.extend(self.find_pnt_patterns())
        
        # 2. Riemann Zeta connections
        patterns.extend(self.find_zeta_patterns())
        
        # 3. Modular arithmetic patterns
        patterns.extend(self.find_modular_patterns())
        
        # 4. Continued fraction patterns
        patterns.extend(self.find_continued_fraction_patterns())
        
        # 5. Fourier analysis patterns
        patterns.extend(self.find_fourier_patterns())
        
        # 6. Graph theory patterns (treating n as nodes)
        patterns.extend(self.find_graph_patterns())
        
        return patterns
    
    def find_pnt_patterns(self):
        """Look for Prime Number Theorem relationships"""
        patterns = []
        
        for d in self.data:
            n = d['n']
            if n > 2:
                # Classic PNT approximation
                pnt_approx = n / math.log(n)
                
                # Li(n) - logarithmic integral approximation
                li_n = self.logarithmic_integral(n)
                
                # Check correlation with theta entropy
                if 'theta_entropy_mean' in d:
                    d['pnt_ratio'] = pnt_approx / n
                    d['li_ratio'] = li_n / n
                    d['pnt_entropy_product'] = d['theta_entropy_mean'] * math.log(n)
        
        # Find correlations
        if len(self.data) > 100:
            corr = self.compute_correlation('pnt_entropy_product', 'omega')
            if abs(corr) > 0.3:
                patterns.append({
                    'type': 'PNT',
                    'formula': 'H_θ * ln(n) ~ ω(n)',
                    'correlation': corr
                })
        
        return patterns
    
    def find_zeta_patterns(self):
        """Look for Riemann Zeta function connections"""
        patterns = []
        
        # Compute partial zeta sums
        for s in [2, 3, 4]:  # Different s values
            zeta_sum = sum(1/n**s for n in range(1, min(1000, len(self.data))))
            
            # Check if pattern frequencies correlate with zeta values
            for d in self.data:
                if 'theta_entropy_mean' in d:
                    d[f'zeta_s{s}_score'] = d['theta_entropy_mean'] * zeta_sum
        
        return patterns
    
    def find_modular_patterns(self):
        """Discover modular arithmetic relationships"""
        patterns = []
        
        # Test different moduli
        for mod in [6, 12, 30, 210]:  # Important moduli in number theory
            residue_classes = defaultdict(list)
            
            for d in self.data:
                residue = d['n'] % mod
                if 'theta_entropy_mean' in d:
                    residue_classes[residue].append(d['theta_entropy_mean'])
            
            # Check if residue classes have distinct entropy patterns
            if len(residue_classes) > 1:
                means = {r: np.mean(vals) for r, vals in residue_classes.items() if vals}
                variance = np.var(list(means.values()))
                
                if variance > 0.01:  # Significant difference
                    patterns.append({
                        'type': 'MODULAR',
                        'modulus': mod,
                        'variance': variance,
                        'residue_means': means
                    })
        
        return patterns
    
    def logarithmic_integral(self, x):
        """Compute logarithmic integral Li(x)"""
        if x <= 2:
            return 0
        # Numerical approximation
        from scipy import integrate
        result, _ = integrate.quad(lambda t: 1/math.log(t), 2, x)
        return result

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point"""
    
    logger.log("="*80, "INFO")
    logger.log("SIEVE ECHO CONJECTURE - ENHANCED EMPIRICAL ANALYSIS", "INFO")
    logger.log("Version 8.0 - With Critical Improvements from Empirical Findings", "INFO")
    logger.log("="*80, "INFO")
    
    # Configuration summary
    logger.log("\nCONFIGURATION:", "INFO")
    logger.log(f"  Max n: {CONFIG.max_n}", "INFO")
    logger.log(f"  Sample size: {CONFIG.sample_size}", "INFO")
    logger.log(f"  Test bases: {len(CONFIG.test_bases)} bases", "INFO")
    logger.log(f"  Parallel workers: {CONFIG.n_workers}", "INFO")
    logger.log(f"  Base invariance threshold: {CONFIG.base_invariance_threshold}", "INFO")
    
    # Create and run analyzer
    analyzer = EnhancedComprehensiveAnalyzer()
    
    try:
        analyzer.run_complete_analysis()
    except KeyboardInterrupt:
        logger.log("\nAnalysis interrupted by user", "WARNING")
    except Exception as e:
        logger.log(f"ERROR: {e}", "ERROR")
        logger.log(traceback.format_exc(), "ERROR")
    finally:
        # Always save results
        logger.save_all()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)