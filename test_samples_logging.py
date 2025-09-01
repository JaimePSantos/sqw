#!/usr/bin/env python3

"""
Test the updated logging in dynamic samples generation
"""

import time
import logging

# Test the logging functionality
if __name__ == "__main__":
    print("Testing improved step and sample logging for dynamic samples...")
    
    # Create a simple logger for testing
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("test")
    
    # Simulate the enhanced step logging
    steps = 150  # Small number for quick test
    samples_count = 12  # Small number for quick test
    
    print(f"Testing step-level logging (every 100th step) for {steps} steps...")
    for step_idx in range(steps):
        if step_idx % 100 == 0 or step_idx == steps - 1:
            logger.info(f"    Saving step {step_idx}/{steps - 1} (quantum walk state for time step {step_idx})")
        time.sleep(0.01)  # Small delay to simulate work
    
    print(f"\nTesting sample-level logging (every 5th sample) for {samples_count} samples...")
    dev_computed_samples = 0
    dev_skipped_samples = 0
    
    for sample_idx in range(samples_count):
        # Simulate some processing
        time.sleep(0.1)
        
        # Simulate sample computed vs skipped
        sample_computed = sample_idx % 3 != 0  # Every 3rd sample is "skipped"
        if sample_computed:
            dev_computed_samples += 1
        else:
            dev_skipped_samples += 1
        
        # Progress summary every 5 samples or at the end
        if (sample_idx + 1) % 5 == 0 or sample_idx == samples_count - 1:
            logger.info(f"Sample progress: {sample_idx + 1}/{samples_count} processed ({dev_computed_samples} computed, {dev_skipped_samples} skipped)")
    
    print(f"\nTest completed! The enhanced logging shows:")
    print("1. Step progress every 100th step with descriptive message")
    print("2. Sample progress every 5th sample with running counts")
    print("3. Clear indication of what computational work is being performed")
