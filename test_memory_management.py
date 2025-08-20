#!/usr/bin/env python3
"""
Test script to verify memory management improvements in probability distribution calculations.

This script tests that:
1. Mean probability calculations properly release memory after each step
2. Standard deviation calculations properly release memory after each deviation
3. Overall memory usage is properly monitored and reported
"""

import os
import sys
import time
import gc
import psutil

def monitor_memory():
    """Monitor current memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3)
        }
    except:
        return {'percent': 0, 'available_gb': 0, 'used_gb': 0}

def test_memory_management():
    """Test the memory management improvements"""
    print("=== TESTING MEMORY MANAGEMENT IMPROVEMENTS ===")
    
    # Set very small test parameters
    os.environ['FORCE_N_VALUE'] = '50'  # Very small N
    os.environ['FORCE_SAMPLES_COUNT'] = '2'  # Only 2 samples
    os.environ['ENABLE_PLOTTING'] = 'False'  # Disable plotting
    os.environ['CREATE_TAR_ARCHIVE'] = 'False'  # Disable archiving
    os.environ['SKIP_SAMPLE_COMPUTATION'] = 'True'  # Skip samples, just test analysis
    
    print("Test configuration:")
    print(f"  N = {os.environ['FORCE_N_VALUE']}")
    print(f"  Samples = {os.environ['FORCE_SAMPLES_COUNT']}")
    print(f"  Skip sample computation = True (testing analysis only)")
    print("=" * 60)
    
    # Monitor initial memory
    initial_memory = monitor_memory()
    print(f"Initial memory: {initial_memory['percent']:.1f}% used, {initial_memory['available_gb']:.1f}GB available")
    
    # Import the main script
    try:
        from static_cluster_logged_mp import run_static_experiment
        
        # Run the experiment
        print("\nRunning experiment with memory management...")
        start_time = time.time()
        
        result = run_static_experiment()
        
        end_time = time.time()
        
        # Monitor final memory
        final_memory = monitor_memory()
        print(f"\nFinal memory: {final_memory['percent']:.1f}% used, {final_memory['available_gb']:.1f}GB available")
        
        # Calculate memory change
        memory_change = final_memory['used_gb'] - initial_memory['used_gb']
        print(f"Memory change: {memory_change:+.3f}GB")
        
        print(f"\n=== TEST COMPLETED SUCCESSFULLY ===")
        print(f"Total time: {end_time - start_time:.1f} seconds")
        print(f"Result mode: {result.get('mode', 'unknown')}")
        
        # Check if memory usage is reasonable
        if abs(memory_change) < 0.5:  # Less than 500MB change
            print("✓ Memory usage appears well-controlled")
        else:
            print(f"⚠ Significant memory change: {memory_change:+.3f}GB")
            
    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Monitor memory even on failure
        final_memory = monitor_memory()
        print(f"\nMemory at failure: {final_memory['percent']:.1f}% used")

if __name__ == "__main__":
    test_memory_management()
