#!/usr/bin/env python3
"""
Diagnostic script to test NDR feature extraction
Run this to verify the fixes work before running the full script
"""

import numpy as np
import torch
from sympy import factorint, primerange, isprime, totient
import math
from scipy import stats

def test_ndr_extraction():
    """Test if NDR features can be extracted properly"""
    
    print("Testing NDR feature extraction...")
    print("-" * 50)
    
    # Test parameters
    test_numbers = [7, 11, 15, 21, 30, 42, 60, 100, 210]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    valid_count = 0
    invalid_count = 0
    
    for n in test_numbers:
        print(f"\nTesting n = {n}:")
        
        # Find valid bases (must be < n and coprime to n)
        valid_bases = []
        for base in range(2, min(n, 17)):
            if math.gcd(n, base) == 1:
                valid_bases.append(base)
        
        print(f"  Valid bases: {valid_bases[:10]}...")
        
        if not valid_bases:
            print(f"  ❌ No valid bases found for n={n}")
            invalid_count += 1
            continue
        
        # Try to compute repetend for first valid base
        base = valid_bases[0]
        
        # Compute repetend
        remainder = 1
        digits = []
        seen = {}
        max_length = 1000
        
        while remainder != 0 and remainder not in seen and len(digits) < max_length:
            seen[remainder] = len(digits)
            remainder *= base
            digit = remainder // n
            digits.append(digit)
            remainder = remainder % n
        
        if remainder in seen:
            pattern = digits[seen[remainder]:]
        else:
            pattern = digits
        
        if pattern:
            print(f"  ✓ Repetend found with length {len(pattern)} in base {base}")
            
            # Convert to NDR
            ndr = torch.tensor(pattern, dtype=torch.float32, device=device) / base
            
            if len(ndr) >= 2:
                # Calculate basic features
                ndr_cpu = ndr.cpu().numpy()
                
                # FFT entropy
                fft = np.fft.fft(ndr_cpu)
                power_spectrum = np.abs(fft)**2
                
                if np.sum(power_spectrum) > 1e-9:
                    p = power_spectrum / np.sum(power_spectrum)
                    p = p[p > 1e-10]
                    entropy = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
                else:
                    entropy = 0.0
                
                # Statistics
                mean_val = float(np.mean(ndr_cpu))
                std_val = float(np.std(ndr_cpu))
                
                if len(ndr_cpu) > 3:
                    skew_val = float(stats.skew(ndr_cpu))
                    kurt_val = float(stats.kurtosis(ndr_cpu))
                else:
                    skew_val = 0.0
                    kurt_val = 0.0
                
                omega_n = len(factorint(n))
                
                print(f"  Features extracted successfully:")
                print(f"    - Length: {len(pattern)}")
                print(f"    - Entropy: {entropy:.4f}")
                print(f"    - Mean: {mean_val:.4f}")
                print(f"    - Std: {std_val:.4f}")
                print(f"    - Kurtosis: {kurt_val:.4f}")
                print(f"    - ω(n): {omega_n}")
                
                valid_count += 1
            else:
                print(f"  ⚠ Pattern too short for feature extraction")
                invalid_count += 1
        else:
            print(f"  ❌ No repetend found")
            invalid_count += 1
    
    print("\n" + "=" * 50)
    print(f"SUMMARY:")
    print(f"  Valid extractions: {valid_count}/{len(test_numbers)}")
    print(f"  Invalid extractions: {invalid_count}/{len(test_numbers)}")
    print(f"  Success rate: {valid_count/len(test_numbers)*100:.1f}%")
    
    if valid_count > len(test_numbers) * 0.5:
        print("\n✅ Feature extraction is working properly!")
        print("You can now run the full script.")
    else:
        print("\n❌ Feature extraction is failing.")
        print("Please check the implementation.")
    
    return valid_count > 0

if __name__ == "__main__":
    success = test_ndr_extraction()
    exit(0 if success else 1)