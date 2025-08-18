#!/usr/bin/env python3

"""
Small test run to verify the multiprocessing fixes work.
This uses smaller parameters to test the functionality without memory issues.
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

# Import and temporarily modify the parameters for testing
print("Loading multiprocessing script for testing...")

# Read the original file
with open('static_cluster_logged_mp.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Create a test version with smaller parameters
test_content = content.replace(
    'N = 20000  # System size', 
    'N = 100  # System size (TEST: reduced for quick test)'
).replace(
    'steps = min(N//4, 1000)  # Time steps - limit to 1000 to prevent memory issues',
    'steps = 10  # Time steps (TEST: reduced for quick test)'
).replace(
    'samples = 5  # Samples per deviation - changed from 1 to 5',
    'samples = 1  # Samples per deviation (TEST: reduced for quick test)'
).replace(
    'devs = [0, 0.1, 0.5, 1, 10]  # List of static noise deviation values',
    'devs = [0, 0.1]  # List of static noise deviation values (TEST: reduced for quick test)'
)

# Write test version
with open('test_static_cluster_logged_mp.py', 'w', encoding='utf-8') as f:
    f.write(test_content)

print("Created test_static_cluster_logged_mp.py with smaller parameters:")
print("  N = 100")
print("  steps = 10") 
print("  samples = 1")
print("  devs = [0, 0.1]")
print("\nThis should run quickly and verify the fixes work.")
print("Run with: C:\\Users\\jaime\\anaconda3\\envs\\QWAK2\\python.exe test_static_cluster_logged_mp.py")
