#!/usr/bin/env python3
"""
Script to compute mean probability distributions from raw quantum state samples.
Scans experiments_data_samples, loads each sample for each step, converts to probability,
and writes the mean probability distribution for each step to experiments_data_samples_probDist
with a matching folder structure.
"""

import os
import pickle
import numpy as np
from pathlib import Path

def amp2prob(state):
    """Convert quantum state amplitudes to probability distribution (|amplitude|^2)."""
    return np.abs(state) ** 2

def process_dev_folder(source_exp_dir, target_exp_dir, steps, samples):
    os.makedirs(target_exp_dir, exist_ok=True)
    for step_idx in range(steps):
        step_dir = os.path.join(source_exp_dir, f"step_{step_idx}")
        prob_distributions = []
        valid_samples = 0
        for sample_idx in range(samples):
            filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
            filepath = os.path.join(step_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    state = pickle.load(f)
                prob_dist = amp2prob(state)
                prob_distributions.append(prob_dist)
                valid_samples += 1
            else:
                print(f"Warning: Sample file not found: {filepath}")
        if valid_samples > 0:
            mean_prob_dist = np.mean(prob_distributions, axis=0)
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(target_exp_dir, mean_filename)
            with open(mean_filepath, "wb") as f:
                pickle.dump(mean_prob_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
            if step_idx % 50 == 0:
                print(f"Processed {step_idx+1} / {steps} steps in {target_exp_dir}")
        else:
            print(f"No valid samples found for step {step_idx} in {source_exp_dir}")

def main():
    source_base = Path("experiments_data_samples")
    target_base = Path("experiments_data_samples_probDist")
    steps = 500  # Set to your number of steps
    samples = 10 # Set to your number of samples per step

    for root, dirs, files in os.walk(source_base):
        # Look for leaf folders containing step_0, step_1, ...
        if any(d.startswith("step_") for d in dirs):
            rel_path = Path(root).relative_to(source_base)
            source_exp_dir = Path(root)
            target_exp_dir = target_base / rel_path
            print(f"Processing: {source_exp_dir} -> {target_exp_dir}")
            process_dev_folder(str(source_exp_dir), str(target_exp_dir), steps, samples)

if __name__ == "__main__":
    main()
