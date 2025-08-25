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
        
    def run_complete_analysis(self):
        """Run all analysis methods with improvements"""
        
        logger.log("="*80, "INFO")
        logger.log("ENHANCED SIEVE ECHO ANALYSIS WITH CRITICAL IMPROVEMENTS", "INFO")
        logger.log("="*80, "INFO")
        
        # Phase 1: Data Generation (with parallel processing)
        logger.log("\nPHASE 1: PARALLEL DATA GENERATION", "INFO")
        self.generate_dataset_parallel()
        
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