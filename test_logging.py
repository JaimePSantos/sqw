#!/usr/bin/env python3

"""
Test version of dynamic probdist generation with smaller parameters to verify logging
"""

import os
import sys
import math
import pickle
import gc
import time
import signal
import logging
import multiprocessing as mp
import traceback

# Test parameters - small values for quick verification
N = 100
steps = 20
samples = 5
base_theta = math.pi/3
devs = [0, 0.2]  # Just two deviations for quick test

# Test the logging functionality
if __name__ == "__main__":
    print("Testing step-by-step logging functionality...")
    print(f"Parameters: N={N}, steps={steps}, samples={samples}")
    
    # Create a simple logger for testing
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("test")
    
    # Simulate the step processing loop with the new logging
    actual_steps = steps + 1  # 0 to steps inclusive
    computed_steps = 0
    skipped_steps = 0
    
    print(f"Starting simulation of {actual_steps} steps...")
    
    for step_idx in range(actual_steps):
        # Log every 100th step or first/last step (with our small test, this will show first and last)
        if step_idx % 100 == 0 or step_idx == actual_steps - 1:
            logger.info(f"Processing step {step_idx}/{actual_steps - 1} (computing probability distribution for time step {step_idx})")
        
        # Simulate some processing
        time.sleep(0.1)  # Small delay to simulate work
        
        # Simulate success/skip logic
        was_skipped = step_idx % 3 == 0  # Every 3rd step is "skipped"
        if was_skipped:
            skipped_steps += 1
        else:
            computed_steps += 1
        
        # Progress summary every 100 steps or at the end (for our small test, just at the end)
        if step_idx % 100 == 0 or step_idx == actual_steps - 1:
            logger.info(f"Progress: {step_idx + 1}/{actual_steps} steps processed ({computed_steps} computed, {skipped_steps} skipped)")
    
    print(f"\nTest completed! Final results: {computed_steps} computed, {skipped_steps} skipped")
    print("The new logging format shows:")
    print("1. Which step is being processed (every 100th step)")
    print("2. What time step it represents")
    print("3. Running count of computed vs skipped steps")
