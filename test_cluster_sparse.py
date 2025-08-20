#!/usr/bin/env python3
"""
Test the cluster script with sparse matrix implementation.
This simulates the exact cluster environment but with smaller parameters.
"""

import os
import sys
import tempfile
import shutil

# Add sqw module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sqw'))

def test_cluster_script():
    """Test the cluster script with sparse matrices"""
    
    print("Testing cluster script with sparse matrix implementation...")
    
    # Create temporary directory for test output
    test_dir = tempfile.mkdtemp(prefix="sqw_test_")
    print(f"Test output directory: {test_dir}")
    
    try:
        # Import and run the cluster function
        from static_cluster_logged_mp import run_static_experiment
        
        # Test parameters - smaller than full cluster but same structure
        test_params = {
            'N': 1000,  # Smaller for testing
            'theta': 3.14159/3,  # pi/3
            'steps': 50,  # Reduced steps
            'deviations': [0.05, 0.1],  # 2 deviations
            'samples': 2,  # 2 samples per deviation
            'initial_state_kwargs': {'nodes': []},  # Uniform superposition
            'exp_dir': test_dir,
            'process_id': 'test'
        }
        
        print(f"Running test with parameters:")
        for key, value in test_params.items():
            print(f"  {key}: {value}")
        
        # Run the experiment
        result = run_static_noise_experiment(**test_params)
        
        print(f"\nTest completed!")
        print(f"Result: {result}")
        
        # Check if files were created
        if os.path.exists(test_dir):
            files_created = []
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    files_created.append(os.path.relpath(os.path.join(root, file), test_dir))
            
            print(f"\nFiles created: {len(files_created)}")
            for file in sorted(files_created)[:10]:  # Show first 10 files
                print(f"  {file}")
            if len(files_created) > 10:
                print(f"  ... and {len(files_created) - 10} more files")
        
        if result.get('success', False):
            print("\n✓ Cluster script test successful!")
            return True
        else:
            print(f"\n✗ Cluster script failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test directory
        try:
            shutil.rmtree(test_dir)
            print(f"Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")

if __name__ == "__main__":
    success = test_cluster_script()
    if success:
        print("\n✓ Cluster script is ready for large-scale deployment!")
    else:
        print("\n✗ Cluster script needs fixes before deployment.")
