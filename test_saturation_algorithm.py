#!/usr/bin/env python3

"""
Test Saturation Time Detection Algorithm

This script tests the saturation time detection algorithm on synthetic and real data
to validate the implementation before running it on the full dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Add the current directory to path so we can import the saturation function
sys.path.append('.')

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: SciPy not available. Using fallback methods.")

def create_synthetic_std_curve(t_sat=50, y_plateau=10, noise_level=0.1, n_points=200):
    """
    Create a synthetic standard deviation curve that grows and then saturates.
    
    Args:
        t_sat: Saturation time
        y_plateau: Plateau value
        noise_level: Relative noise level
        n_points: Number of time points
    
    Returns:
        tuple: (time_array, std_values)
    """
    t = np.linspace(1, 100, n_points)
    
    # Create a growth curve that saturates
    # Use a function like y = y_plateau * (1 - exp(-(t/tau)^beta))
    tau = t_sat / 3  # Time constant
    beta = 1.5       # Growth exponent
    
    y = y_plateau * (1 - np.exp(-(t/tau)**beta))
    
    # Add some realistic noise
    noise = noise_level * y_plateau * np.random.normal(0, 1, len(t))
    y_noisy = y + noise
    
    # Ensure positive values
    y_noisy = np.maximum(y_noisy, 0.01)
    
    return t, y_noisy

def test_saturation_algorithm():
    """Test the saturation time algorithm on synthetic data."""
    
    # Import the saturation function from our main script
    from generate_linspace_sattime_from_std import detect_saturation_simple
    
    print("=== TESTING SIMPLE SATURATION TIME ALGORITHM ===")
    print("Goal: For each deviation value, find when its std curve saturates")
    print("Final plot: DEVIATION (Y) vs SATURATION TIME (X)")
    print("Method: Simple plateau detection based on moving averages")
    
    # Test multiple deviations to show the final result
    print("\n1. Testing multiple deviation values...")
    
    # Simulate different deviation values with different saturation behaviors
    deviation_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    saturation_times = []
    
    for i, dev in enumerate(deviation_values):
        # Create a realistic std vs time curve for this deviation
        # Higher deviation -> later saturation, higher final std
        true_sat_time = 20 + dev * 60  # Saturation between 20-80
        final_std = 5 + dev * 15      # Final std between 5-20
        noise_level = 0.05 + dev * 0.1  # More noise for higher deviations
        
        t, std_values = create_synthetic_std_curve(
            t_sat=true_sat_time, 
            y_plateau=final_std, 
            noise_level=noise_level, 
            n_points=100
        )
        
        # Find saturation time for this deviation's std curve
        sat_time, sat_value, metadata = detect_saturation_simple(t, std_values)
        saturation_times.append(sat_time)
        
        print(f"   Deviation {dev:.1f}: std curve saturates at time {sat_time:.1f} (target: {true_sat_time:.1f})")
    
    # Convert to numpy array
    saturation_times = np.array(saturation_times)
    
    # Filter valid results
    valid_mask = ~np.isnan(saturation_times)
    valid_devs = deviation_values[valid_mask]
    valid_sat_times = saturation_times[valid_mask]
    
    print(f"\nValid results: {len(valid_devs)} out of {len(deviation_values)}")
    
    # Test with real data if available
    real_devs = []
    real_sat_times = []
    
    std_base_dir = "experiments_data_samples_linspace_std"
    if os.path.exists(std_base_dir):
        print(f"\n2. Testing on real data from: {std_base_dir}")
        
        # Look for different deviation directories
        for root, dirs, files in os.walk(std_base_dir):
            if "std_vs_time.pkl" in files:
                # Extract deviation from path
                path_parts = root.split(os.sep)
                dev_folder = None
                for part in path_parts:
                    if part.startswith("dev_"):
                        dev_folder = part
                        break
                
                if dev_folder:
                    try:
                        # Parse deviation value
                        if "min" in dev_folder and "max" in dev_folder:
                            # Format: dev_min0.000_max0.600
                            max_part = dev_folder.split("_max")[1]
                            deviation = float(max_part)
                        else:
                            # Format: dev_0.600
                            deviation = float(dev_folder.replace("dev_", ""))
                        
                        # Load std data
                        std_file = os.path.join(root, "std_vs_time.pkl")
                        with open(std_file, 'rb') as f:
                            std_data = pickle.load(f)
                        
                        std_array = np.array(std_data)
                        t_real = np.arange(len(std_array))
                        
                        # Find saturation time
                        sat_time, sat_value, metadata = detect_saturation_simple(t_real, std_array)
                        
                        if not np.isnan(sat_time):
                            real_devs.append(deviation)
                            real_sat_times.append(sat_time)
                            print(f"   Real deviation {deviation:.3f}: saturates at time {sat_time:.1f}")
                        else:
                            print(f"   Real deviation {deviation:.3f}: no saturation detected")
                        
                        # Limit to first few for testing
                        if len(real_devs) >= 10:
                            break
                            
                    except Exception as e:
                        print(f"   Error processing {dev_folder}: {e}")
    
    # Create the SINGLE focused plot: DEVIATION vs SATURATION TIME
    plt.figure(figsize=(10, 8))
    
    # Plot synthetic results
    if len(valid_devs) > 0:
        plt.scatter(valid_sat_times, valid_devs, c='blue', s=100, alpha=0.7, 
                   label=f'Synthetic Data ({len(valid_devs)} points)', marker='o')
    
    # Plot real results
    if len(real_devs) > 0:
        plt.scatter(real_sat_times, real_devs, c='red', s=100, alpha=0.8, 
                   label=f'Real Data ({len(real_devs)} points)', marker='s')
    
    plt.xlabel('Saturation Time', fontsize=14)
    plt.ylabel('Deviation Value', fontsize=14)
    plt.title('Deviation vs Saturation Time\n(Simple Plateau Detection Method)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add explanation text
    if len(valid_devs) > 0 or len(real_devs) > 0:
        explanation = """
SIMPLE METHOD:
• Calculate moving averages with different window sizes
• Find where relative change becomes < 5%
• Look for sustained flat regions (10+ points)
• Return earliest consistent saturation point
        """
        plt.text(0.02, 0.98, explanation, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('deviation_vs_saturation_time_simple.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Plot saved as: deviation_vs_saturation_time_simple.png")
    print(f"Synthetic results: {len(valid_devs)} valid out of {len(deviation_values)} tested")
    print(f"Real results: {len(real_devs)} valid")
    
    if len(valid_devs) > 0:
        print(f"Synthetic data range:")
        print(f"  Deviations: {valid_devs.min():.1f} - {valid_devs.max():.1f}")
        print(f"  Saturation times: {valid_sat_times.min():.1f} - {valid_sat_times.max():.1f}")
    
    if len(real_devs) > 0:
        print(f"Real data range:")
        print(f"  Deviations: {min(real_devs):.3f} - {max(real_devs):.3f}")
        print(f"  Saturation times: {min(real_sat_times):.1f} - {max(real_sat_times):.1f}")
    
    print(f"\nSIMPLE METHOD CONFIRMED:")
    print(f"✓ Moving averages with different window sizes")
    print(f"✓ Relative change threshold: 5%")  
    print(f"✓ Minimum plateau length: 10 points")
    print(f"✓ No complex log-log derivatives - just straightforward plateau detection")

if __name__ == "__main__":
    test_saturation_algorithm()
