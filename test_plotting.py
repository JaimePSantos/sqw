#!/usr/bin/env python3

"""
Test script to isolate the plotting issue in static_local_logged_mp.py
"""

import sys
import os
import numpy as np
import pickle

# Add current directory to path
sys.path.append('.')

def test_plotting():
    """Test the plotting functionality"""
    
    # Configuration
    ENABLE_PLOTTING = True
    USE_LOGLOG_PLOT = True
    SAVE_FIGURES = False
    
    print("=== TESTING PLOTTING FUNCTIONALITY ===")
    
    # Try importing matplotlib
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for Windows
        import matplotlib.pyplot as plt
        print("[OK] matplotlib imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import matplotlib: {e}")
        return False
    
    # Try creating some dummy data
    try:
        # Dummy standard deviation data
        stds = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # dev = 0
            [0.2, 0.4, 0.6, 0.8, 1.0],  # dev = 0.1
            [0.3, 0.6, 0.9, 1.2, 1.5],  # dev = 0.5
        ]
        devs = [0, 0.1, 0.5]
        
        print(f"[OK] Created dummy data: {len(stds)} deviation values")
        
        # Test plotting
        if ENABLE_PLOTTING:
            print("\n[PLOT] Creating standard deviation vs time plot...")
            
            if len(stds) > 0 and any(len(std) > 0 for std in stds):
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                for i, (std_values, dev) in enumerate(zip(stds, devs)):
                    if len(std_values) > 0:
                        time_steps = list(range(len(std_values)))
                        
                        # Filter out zero values for log-log plot
                        if USE_LOGLOG_PLOT:
                            # Remove zero values which can't be plotted on log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                            if filtered_data:
                                filtered_times, filtered_stds = zip(*filtered_data)
                                plt.loglog(filtered_times, filtered_stds, 
                                         label=f'Static deviation = {dev:.3f}', 
                                         marker='o', markersize=3, linewidth=2)
                        else:
                            plt.plot(time_steps, std_values, 
                                   label=f'Static deviation = {dev:.3f}', 
                                   marker='o', markersize=3, linewidth=2)
                
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                
                if USE_LOGLOG_PLOT:
                    plt.title('Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3, which="both", ls="-")  # Grid for both major and minor ticks
                    plot_filename = "test_static_noise_std_vs_time_loglog.png"
                else:
                    plt.title('Standard Deviation vs Time for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plot_filename = "test_static_noise_std_vs_time.png"
                
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # Save the plot (if enabled)
                if SAVE_FIGURES:
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"[OK] Plot saved as '{plot_filename}'")
                
                # Show the plot
                print(f"[OK] Displaying plot...")
                plt.show()
                
                plot_type = "log-log" if USE_LOGLOG_PLOT else "linear"
                saved_status = " and saved" if SAVE_FIGURES else ""
                print(f"[OK] Standard deviation plot displayed{saved_status}! (Scale: {plot_type})")
                
                return True
            else:
                print("[WARNING] No standard deviation data available for plotting")
                return False
                
    except Exception as e:
        print(f"[ERROR] Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plotting()
    if success:
        print("\n[SUCCESS] Plotting test completed successfully!")
    else:
        print("\n[FAILED] Plotting test failed!")
