#!/usr/bin/env python3

"""
Test script to check if the standard deviation data can be loaded from real experiment files
"""

import sys
import os
import numpy as np
import pickle

# Add current directory to path
sys.path.append('.')

def test_std_data_loading():
    """Test loading standard deviation data from experiment files"""
    
    print("=== TESTING STD DATA LOADING ===")
    
    # Configuration from the main script
    N = 106
    steps = N//4
    devs = [
        0,              # No noise
        (np.pi/15, 0.1),     # max_dev=0.1, min_dev=0.01 (range [0.01, 0.1])
        (np.pi/6, 0.2),     # max_dev=0.5, min_dev=0.1 (range [0.1, 0.5]) 
        (np.pi/3, 0.5),     # max_dev=1.0, min_dev=0.5 (range [0.5, 1.0])
        (2*np.pi/3, 0.5)     # max_dev=10.0, min_dev=1.0 (range [1.0, 10.0])
    ]
    theta = np.pi/3
    
    print(f"N = {N}, steps = {steps}, theta = {theta:.4f}")
    print(f"Deviations: {devs}")
    
    # Try to import the required modules
    try:
        from smart_loading_static import get_experiment_dir
        print("[OK] smart_loading_static imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import smart_loading_static: {e}")
        return False
    
    # Try to load std data for each deviation
    std_base_dir = "experiments_data_samples_std"
    stds = []
    
    for i, dev in enumerate(devs):
        print(f"\n--- Processing deviation {i+1}/{len(devs)}: {dev} ---")
        
        # Handle new deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # New format: (max_dev, min_factor) or legacy (min, max)
            if dev[1] <= 1.0 and dev[1] >= 0.0:
                # New format
                max_dev, min_factor = dev
                has_noise = max_dev > 0
                dev_str = f"max{max_dev:.3f}_min{max_dev * min_factor:.3f}"
            else:
                # Legacy format
                min_val, max_val = dev
                has_noise = max_val > 0
                dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            # Single value format
            has_noise = dev > 0
            dev_str = f"{dev:.3f}"
        
        print(f"  Dev string: {dev_str}")
        print(f"  Has noise: {has_noise}")
        
        # Setup std data directory structure
        def dummy_tesselation_func(N):
            return None
            
        noise_params = [dev] if has_noise else [0]
        try:
            std_dir = get_experiment_dir(dummy_tesselation_func, has_noise, N, 
                                       noise_params=noise_params, noise_type="static_noise", 
                                       base_dir=std_base_dir, theta=theta)
            print(f"  Std directory: {std_dir}")
            
            std_filepath = os.path.join(std_dir, "std_vs_time.pkl")
            print(f"  Std file path: {std_filepath}")
            
            # Try to load std data
            if os.path.exists(std_filepath):
                try:
                    with open(std_filepath, 'rb') as f:
                        std_values = pickle.load(f)
                    print(f"  [OK] Loaded std data: {len(std_values)} time steps")
                    if len(std_values) > 0:
                        print(f"       First few values: {std_values[:5]}")
                        print(f"       Last few values: {std_values[-5:]}")
                        print(f"       Final std: {std_values[-1]:.6f}")
                    stds.append(std_values)
                except Exception as e:
                    print(f"  [ERROR] Could not load std data: {e}")
                    stds.append([])
            else:
                print(f"  [WARNING] Std file does not exist")
                stds.append([])
                
        except Exception as e:
            print(f"  [ERROR] Error getting std directory: {e}")
            stds.append([])
    
    print(f"\n=== SUMMARY ===")
    print(f"Total deviations: {len(devs)}")
    print(f"Loaded std data for: {sum(1 for std in stds if len(std) > 0)} deviations")
    
    # Check if we have any data for plotting
    if any(len(std) > 0 for std in stds):
        print("[OK] Standard deviation data is available for plotting")
        
        # Test plotting
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            print("\n[PLOT] Creating plot with real data...")
            
            plt.figure(figsize=(12, 8))
            
            plotted_count = 0
            for i, (std_values, dev) in enumerate(zip(stds, devs)):
                if len(std_values) > 0:
                    time_steps = list(range(len(std_values)))
                    
                    # Format dev for display
                    if isinstance(dev, tuple):
                        if dev[1] <= 1.0:
                            dev_display = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                        else:
                            dev_display = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
                    else:
                        dev_display = f"{dev:.3f}"
                    
                    # Plot with log-log scale
                    filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                    if filtered_data:
                        filtered_times, filtered_stds = zip(*filtered_data)
                        plt.loglog(filtered_times, filtered_stds, 
                                 label=f'Static deviation = {dev_display}', 
                                 marker='o', markersize=3, linewidth=2)
                        plotted_count += 1
            
            if plotted_count > 0:
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                plt.title('Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations', fontsize=14)
                plt.grid(True, alpha=0.3, which="both", ls="-")
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                print(f"[OK] Plot created with {plotted_count} data series")
                plt.show()
                
                return True
            else:
                print("[WARNING] No data could be plotted")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error creating plot: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("[WARNING] No standard deviation data is available for plotting")
        print("This could mean:")
        print("1. The experiment hasn't been run yet")
        print("2. The std data files are in a different location")
        print("3. The data format has changed")
        return False

if __name__ == "__main__":
    success = test_std_data_loading()
    if success:
        print("\n[SUCCESS] Data loading and plotting test completed successfully!")
    else:
        print("\n[FAILED] Data loading and plotting test failed!")
