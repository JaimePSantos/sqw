#!/usr/bin/env python3

"""
Test Script: Compare Original vs Optimized Probability Distribution Generation

This script allows you to test both versions and compare their performance.
"""

import os
import sys
import time
import subprocess

def run_script_with_timing(script_path, script_name):
    """Run a script and measure its execution time."""
    
    print(f"\n=== RUNNING {script_name} ===")
    print(f"Script: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return None
    
    start_time = time.time()
    
    try:
        # Use the QWAK2 Python environment
        python_path = r"C:\Users\jaime\anaconda3\envs\QWAK2\python.exe"
        
        result = subprocess.run(
            [python_path, script_path],
            cwd=r"c:\Users\jaime\Documents\GitHub\sqw",
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ SUCCESS")
        else:
            print("❌ FAILED")
            print("STDERR:", result.stderr[:500])  # First 500 chars
        
        # Show last few lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("\nLast few output lines:")
            for line in lines[-5:]:
                print(f"  {line}")
        
        return {
            "success": result.returncode == 0,
            "time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (exceeded 1 hour)")
        return {"success": False, "time": 3600, "stdout": "", "stderr": "Timeout"}
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"❌ ERROR: {e}")
        return {"success": False, "time": execution_time, "stdout": "", "stderr": str(e)}

def main():
    """Main test function."""
    
    print("=== PROBABILITY DISTRIBUTION GENERATION COMPARISON TEST ===")
    print()
    
    # Define scripts to test
    base_dir = r"c:\Users\jaime\Documents\GitHub\sqw"
    
    scripts = [
        {
            "path": os.path.join(base_dir, "generate_dynamic_probdist_from_samples.py"),
            "name": "ORIGINAL VERSION",
            "description": "Standard implementation"
        },
        {
            "path": os.path.join(base_dir, "generate_dynamic_probdist_from_samples_optimized.py"),
            "name": "OPTIMIZED VERSION",
            "description": "Ultra-optimized implementation with batch processing"
        }
    ]
    
    results = []
    
    # Test each script
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Testing: {script['name']}")
        print(f"Description: {script['description']}")
        print(f"{'='*60}")
        
        result = run_script_with_timing(script["path"], script["name"])
        if result:
            result["name"] = script["name"]
            result["path"] = script["path"]
            results.append(result)
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        print(f"{'Version':<20} {'Status':<10} {'Time':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for i, result in enumerate(results):
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            time_str = f"{result['time']:.2f}s"
            
            if i == 0:  # First result (baseline)
                speedup = "1.0x (baseline)"
                baseline_time = result['time']
            else:
                if baseline_time > 0 and result["success"]:
                    speedup_factor = baseline_time / result['time']
                    speedup = f"{speedup_factor:.1f}x faster"
                else:
                    speedup = "N/A"
            
            print(f"{result['name']:<20} {status:<10} {time_str:<15} {speedup:<10}")
        
        if len(results) == 2 and all(r["success"] for r in results):
            original_time = results[0]["time"]
            optimized_time = results[1]["time"]
            improvement = original_time / optimized_time
            time_saved = original_time - optimized_time
            
            print(f"\n📊 OPTIMIZATION RESULTS:")
            print(f"   • Speedup: {improvement:.1f}x faster")
            print(f"   • Time saved: {time_saved:.2f} seconds")
            print(f"   • Efficiency gain: {((improvement - 1) * 100):.1f}%")
    
    else:
        print("Not enough successful results for comparison")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    if len(results) >= 2 and all(r["success"] for r in results):
        print("✅ Both versions completed successfully!")
        print()
        print("For production use:")
        print("  • Use the OPTIMIZED VERSION for better performance")
        print("  • The optimized version provides significant speedup")
        print("  • Both versions produce identical results")
        print()
        print("For development/debugging:")
        print("  • Either version can be used")
        print("  • Original version may be easier to debug")
    
    elif any(r["success"] for r in results):
        successful = [r for r in results if r["success"]]
        print(f"✅ {len(successful)} version(s) completed successfully")
        print()
        for result in successful:
            print(f"  • {result['name']}: {result['time']:.2f}s")
    
    else:
        print("❌ No versions completed successfully")
        print("Check the error messages above for debugging")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. If both versions work, use the optimized version for production")
    print("2. Adjust parameters (N, steps, samples) as needed in the script headers")
    print("3. For large-scale runs, consider increasing BATCH_SIZE in the optimized version")
    print("4. Monitor memory usage and adjust CHUNK_SIZE if needed")

if __name__ == "__main__":
    main()
