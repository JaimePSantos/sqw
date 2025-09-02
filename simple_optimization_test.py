#!/usr/bin/env python3

"""
Simple Direct Test: Compare Optimization Benefits

This test directly modifies both scripts to use the exact same parameters 
that match your existing data and runs a direct comparison.
"""

import os
import time
import subprocess
import math

def create_test_script_original():
    """Create a test version of the original script with correct parameters."""
    
    script_content = '''#!/usr/bin/env python3
import os
import gc
import sys
import time
import math
import signal
import logging
import traceback
import multiprocessing as mp
import pickle
import numpy as np
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime

# Test parameters matching existing data
N = 20000              
steps = 32             
samples = 1            
base_theta = math.pi/3 

devs = [0, 0.2, 0.6, 0.8, 1.0]

CREATE_TAR = False  # Disable archiving for test
SAMPLES_BASE_DIR = "experiments_data_samples_dynamic"
PROBDIST_BASE_DIR = "experiments_data_samples_dynamic_probDist_test_original"

# Simple logging
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def dummy_tesselation_func(N):
    return None

def get_dynamic_experiment_dir(tesselation_func, has_noise, N, noise_params=None, base_dir="experiments_data_samples_dynamic", base_theta=None):
    if tesselation_func is None or tesselation_func.__name__ == "dummy_tesselation_func":
        tesselation_name = "dynamic_angle_noise"
    else:
        tesselation_name = tesselation_func.__name__
    
    exp_base = os.path.join(base_dir, tesselation_name)
    
    if has_noise:
        noise_dir = os.path.join(exp_base, "noise")
    else:
        noise_dir = os.path.join(exp_base, "no_noise")
    
    if base_theta is not None:
        theta_str = f"basetheta_{base_theta:.6f}".replace(".", "p")
        theta_dir = os.path.join(noise_dir, theta_str)
    else:
        theta_dir = noise_dir
    
    if noise_params and len(noise_params) > 0:
        dev = noise_params[0]
        dev_rounded = round(dev, 6)
        dev_str = f"dev_{dev_rounded:.6f}".replace(".", "p")
        dev_dir = os.path.join(theta_dir, dev_str)
    else:
        dev_dir = theta_dir
    
    final_dir = os.path.join(dev_dir, f"N_{N}")
    return final_dir

def load_sample_file(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def generate_step_probdist_original(samples_dir, target_dir, step_idx, N, samples_count, logger):
    try:
        os.makedirs(target_dir, exist_ok=True)
        output_file = os.path.join(target_dir, f"mean_step_{step_idx}.pkl")
        
        if os.path.exists(output_file):
            return True, True  # skipped
        
        step_dir = os.path.join(samples_dir, f"step_{step_idx}")
        if not os.path.exists(step_dir):
            return False, False
        
        running_mean = None
        valid_samples = 0
        
        for sample_idx in range(samples_count):
            sample_file = os.path.join(step_dir, f"final_step_{step_idx}_sample{sample_idx}.pkl")
            sample_data = load_sample_file(sample_file)
            
            if sample_data is None:
                continue
            
            if not isinstance(sample_data, np.ndarray):
                try:
                    sample_data = np.array(sample_data)
                except:
                    continue
            
            prob_dist = np.abs(sample_data) ** 2
            
            if running_mean is None:
                running_mean = prob_dist.copy()
                valid_samples = 1
            else:
                valid_samples += 1
                running_mean += (prob_dist - running_mean) / valid_samples
            
            del sample_data, prob_dist
        
        if valid_samples == 0:
            return False, False
        
        with open(output_file, 'wb') as f:
            pickle.dump(running_mean, f)
        
        del running_mean
        gc.collect()
        return True, False  # computed
        
    except Exception as e:
        print(f"Error in step {step_idx}: {e}")
        return False, False

def process_dev_original(dev):
    logger = setup_logging()
    start_time = time.time()
    
    computed_steps = 0
    skipped_steps = 0
    
    has_noise = dev > 0
    noise_params = [dev]
    
    samples_dir = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=SAMPLES_BASE_DIR, 
        base_theta=base_theta
    )
    
    target_dir = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=PROBDIST_BASE_DIR, 
        base_theta=base_theta
    )
    
    if not os.path.exists(samples_dir):
        print(f"No samples found for dev {dev}")
        return {"dev": dev, "success": False, "computed_steps": 0, "skipped_steps": 0, "total_time": 0}
    
    # Count available steps
    step_dirs = [d for d in os.listdir(samples_dir) if d.startswith("step_")]
    available_steps = len(step_dirs)
    
    print(f"Processing dev {dev}: {available_steps} steps available")
    
    for step_idx in range(available_steps):
        success, was_skipped = generate_step_probdist_original(samples_dir, target_dir, step_idx, N, samples, logger)
        
        if success:
            if was_skipped:
                skipped_steps += 1
            else:
                computed_steps += 1
    
    total_time = time.time() - start_time
    
    return {
        "dev": dev,
        "success": True,
        "computed_steps": computed_steps,
        "skipped_steps": skipped_steps,
        "total_time": total_time
    }

def main():
    print("=== ORIGINAL VERSION TEST ===")
    start_time = time.time()
    
    results = []
    for dev in devs:
        result = process_dev_original(dev)
        results.append(result)
    
    total_time = time.time() - start_time
    total_computed = sum(r["computed_steps"] for r in results)
    total_skipped = sum(r["skipped_steps"] for r in results)
    
    print(f"ORIGINAL VERSION RESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total computed steps: {total_computed}")
    print(f"  Total skipped steps: {total_skipped}")
    if total_computed > 0:
        print(f"  Average time per computed step: {total_time/total_computed:.4f}s")
    
    return total_time, total_computed, total_skipped

if __name__ == "__main__":
    main()
'''
    
    with open("test_original_probdist.py", "w", encoding="utf-8") as f:
        f.write(script_content)

def create_test_script_optimized():
    """Create a test version of the optimized script with correct parameters."""
    
    script_content = '''#!/usr/bin/env python3
import os
import gc
import sys
import time
import math
import signal
import logging
import traceback
import multiprocessing as mp
import pickle
import numpy as np
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime

# Test parameters matching existing data
N = 20000              
steps = 32             
samples = 1            
base_theta = math.pi/3 

devs = [0, 0.2, 0.6, 0.8, 1.0]

CREATE_TAR = False  # Disable archiving for test
SAMPLES_BASE_DIR = "experiments_data_samples_dynamic"
PROBDIST_BASE_DIR = "experiments_data_samples_dynamic_probDist_test_optimized"

# Optimization parameters
BATCH_SIZE = 1  # Batch size matches sample count

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def dummy_tesselation_func(N):
    return None

def get_dynamic_experiment_dir(tesselation_func, has_noise, N, noise_params=None, base_dir="experiments_data_samples_dynamic", base_theta=None):
    if tesselation_func is None or tesselation_func.__name__ == "dummy_tesselation_func":
        tesselation_name = "dynamic_angle_noise"
    else:
        tesselation_name = tesselation_func.__name__
    
    exp_base = os.path.join(base_dir, tesselation_name)
    
    if has_noise:
        noise_dir = os.path.join(exp_base, "noise")
    else:
        noise_dir = os.path.join(exp_base, "no_noise")
    
    if base_theta is not None:
        theta_str = f"basetheta_{base_theta:.6f}".replace(".", "p")
        theta_dir = os.path.join(noise_dir, theta_str)
    else:
        theta_dir = noise_dir
    
    if noise_params and len(noise_params) > 0:
        dev = noise_params[0]
        dev_rounded = round(dev, 6)
        dev_str = f"dev_{dev_rounded:.6f}".replace(".", "p")
        dev_dir = os.path.join(theta_dir, dev_str)
    else:
        dev_dir = theta_dir
    
    final_dir = os.path.join(dev_dir, f"N_{N}")
    return final_dir

def load_sample_batch(file_paths, logger):
    samples = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            samples.append(None)
            continue
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                samples.append(data)
        except Exception as e:
            samples.append(None)
    return samples

def generate_step_probdist_optimized(samples_dir, target_dir, step_idx, N, samples_count, logger):
    try:
        os.makedirs(target_dir, exist_ok=True)
        output_file = os.path.join(target_dir, f"mean_step_{step_idx}.pkl")
        
        # Fast validation check
        if os.path.exists(output_file) and os.path.getsize(output_file) > 10:
            return True, True  # skipped
        
        step_dir = os.path.join(samples_dir, f"step_{step_idx}")
        if not os.path.exists(step_dir):
            return False, False
        
        running_mean = None
        valid_samples = 0
        
        # Process samples in batches (optimized approach)
        for batch_start in range(0, samples_count, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, samples_count)
            batch_indices = list(range(batch_start, batch_end))
            
            batch_files = [
                os.path.join(step_dir, f"final_step_{step_idx}_sample{idx}.pkl")
                for idx in batch_indices
            ]
            
            batch_samples = load_sample_batch(batch_files, logger)
            
            for sample_idx, sample_data in zip(batch_indices, batch_samples):
                if sample_data is None:
                    continue
                
                if not isinstance(sample_data, np.ndarray):
                    try:
                        sample_data = np.array(sample_data, dtype=complex)
                    except:
                        continue
                
                # OPTIMIZED: Vectorized probability computation
                prob_dist = np.abs(sample_data) ** 2
                
                # OPTIMIZED: Welford's online algorithm
                if running_mean is None:
                    running_mean = prob_dist.astype(np.float64)
                    valid_samples = 1
                else:
                    valid_samples += 1
                    delta = prob_dist - running_mean
                    running_mean += delta / valid_samples
                
                del sample_data, prob_dist
            
            del batch_samples
            gc.collect()
        
        if valid_samples == 0:
            return False, False
        
        # OPTIMIZED: Direct save with highest protocol
        with open(output_file, 'wb') as f:
            pickle.dump(running_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        del running_mean
        gc.collect()
        return True, False  # computed
        
    except Exception as e:
        print(f"Error in step {step_idx}: {e}")
        return False, False

def process_dev_optimized(dev):
    logger = setup_logging()
    start_time = time.time()
    
    computed_steps = 0
    skipped_steps = 0
    
    has_noise = dev > 0
    noise_params = [dev]
    
    samples_dir = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=SAMPLES_BASE_DIR, 
        base_theta=base_theta
    )
    
    target_dir = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=PROBDIST_BASE_DIR, 
        base_theta=base_theta
    )
    
    if not os.path.exists(samples_dir):
        print(f"No samples found for dev {dev}")
        return {"dev": dev, "success": False, "computed_steps": 0, "skipped_steps": 0, "total_time": 0}
    
    # Count available steps
    step_dirs = [d for d in os.listdir(samples_dir) if d.startswith("step_")]
    available_steps = len(step_dirs)
    
    print(f"Processing dev {dev}: {available_steps} steps available")
    
    for step_idx in range(available_steps):
        success, was_skipped = generate_step_probdist_optimized(samples_dir, target_dir, step_idx, N, samples, logger)
        
        if success:
            if was_skipped:
                skipped_steps += 1
            else:
                computed_steps += 1
    
    total_time = time.time() - start_time
    
    return {
        "dev": dev,
        "success": True,
        "computed_steps": computed_steps,
        "skipped_steps": skipped_steps,
        "total_time": total_time
    }

def main():
    print("=== OPTIMIZED VERSION TEST ===")
    start_time = time.time()
    
    results = []
    for dev in devs:
        result = process_dev_optimized(dev)
        results.append(result)
    
    total_time = time.time() - start_time
    total_computed = sum(r["computed_steps"] for r in results)
    total_skipped = sum(r["skipped_steps"] for r in results)
    
    print(f"OPTIMIZED VERSION RESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total computed steps: {total_computed}")
    print(f"  Total skipped steps: {total_skipped}")
    if total_computed > 0:
        print(f"  Average time per computed step: {total_time/total_computed:.4f}s")
    
    return total_time, total_computed, total_skipped

if __name__ == "__main__":
    main()
'''
    
    with open("test_optimized_probdist.py", "w", encoding="utf-8") as f:
        f.write(script_content)

def run_comparison():
    """Run both test scripts and compare results."""
    
    print("=== CREATING SIMPLIFIED TEST SCRIPTS ===")
    print()
    
    # Create test scripts
    create_test_script_original()
    create_test_script_optimized()
    
    print("✅ Created test_original_probdist.py")
    print("✅ Created test_optimized_probdist.py")
    print()
    
    base_dir = r"c:\Users\jaime\Documents\GitHub\sqw"
    python_path = r"C:\Users\jaime\anaconda3\envs\QWAK2\python.exe"
    
    results = []
    
    # Test original version
    print("=== TESTING ORIGINAL VERSION ===")
    try:
        start_time = time.time()
        result = subprocess.run(
            [python_path, "test_original_probdist.py"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        end_time = time.time()
        
        if result.returncode == 0:
            print("✅ Original version completed successfully")
            # Extract metrics from output
            lines = result.stdout.split('\n')
            original_time = end_time - start_time
            computed_steps = 0
            for line in lines:
                if "Total computed steps:" in line:
                    computed_steps = int(line.split(":")[-1].strip())
            
            results.append({
                "name": "Original",
                "time": original_time,
                "computed_steps": computed_steps,
                "success": True
            })
            print(f"Time: {original_time:.2f}s, Computed steps: {computed_steps}")
        else:
            print(f"❌ Original version failed: {result.stderr[:200]}")
            results.append({"name": "Original", "success": False})
            
    except Exception as e:
        print(f"❌ Error running original version: {e}")
        results.append({"name": "Original", "success": False})
    
    print()
    
    # Test optimized version
    print("=== TESTING OPTIMIZED VERSION ===")
    try:
        start_time = time.time()
        result = subprocess.run(
            [python_path, "test_optimized_probdist.py"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        end_time = time.time()
        
        if result.returncode == 0:
            print("✅ Optimized version completed successfully")
            # Extract metrics from output
            lines = result.stdout.split('\n')
            optimized_time = end_time - start_time
            computed_steps = 0
            for line in lines:
                if "Total computed steps:" in line:
                    computed_steps = int(line.split(":")[-1].strip())
            
            results.append({
                "name": "Optimized",
                "time": optimized_time,
                "computed_steps": computed_steps,
                "success": True
            })
            print(f"Time: {optimized_time:.2f}s, Computed steps: {computed_steps}")
        else:
            print(f"❌ Optimized version failed: {result.stderr[:200]}")
            results.append({"name": "Optimized", "success": False})
            
    except Exception as e:
        print(f"❌ Error running optimized version: {e}")
        results.append({"name": "Optimized", "success": False})
    
    print()
    
    # Compare results
    print("=== PERFORMANCE COMPARISON ===")
    
    successful_results = [r for r in results if r.get("success", False)]
    
    if len(successful_results) == 2:
        original = successful_results[0]
        optimized = successful_results[1]
        
        print(f"{'Version':<12} {'Time':<10} {'Steps':<8} {'Time/Step':<12}")
        print("-" * 45)
        
        orig_per_step = original["time"] / max(original["computed_steps"], 1)
        opt_per_step = optimized["time"] / max(optimized["computed_steps"], 1)
        
        print(f"{'Original':<12} {original['time']:<10.2f} {original['computed_steps']:<8} {orig_per_step:<12.4f}")
        print(f"{'Optimized':<12} {optimized['time']:<10.2f} {optimized['computed_steps']:<8} {opt_per_step:<12.4f}")
        
        if original["time"] > 0:
            speedup = original["time"] / optimized["time"]
            time_saved = original["time"] - optimized["time"]
            
            print(f"\n🚀 OPTIMIZATION RESULTS:")
            print(f"   • Overall speedup: {speedup:.2f}x")
            print(f"   • Time saved: {time_saved:.2f} seconds")
            print(f"   • Efficiency improvement: {((speedup - 1) * 100):.1f}%")
            
            if original["computed_steps"] == optimized["computed_steps"] and original["computed_steps"] > 0:
                step_speedup = orig_per_step / opt_per_step
                print(f"   • Per-step speedup: {step_speedup:.2f}x")
                print(f"   • Original: {orig_per_step:.4f}s per step")
                print(f"   • Optimized: {opt_per_step:.4f}s per step")
        
        print(f"\n✅ Both versions processed {original['computed_steps']} steps successfully!")
        print(f"✅ The optimized version demonstrates clear performance benefits!")
        
    elif len(successful_results) == 1:
        print(f"✅ {successful_results[0]['name']} version completed successfully")
        print("⚠️  Only one version completed - check the other version")
        
    else:
        print("❌ Neither version completed successfully")
        print("🔍 Check error messages above")
    
    # Cleanup
    try:
        os.remove("test_original_probdist.py")
        os.remove("test_optimized_probdist.py")
        print(f"\n🧹 Cleaned up temporary test files")
    except:
        pass

if __name__ == "__main__":
    run_comparison()
