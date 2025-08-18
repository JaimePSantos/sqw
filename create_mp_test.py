#!/usr/bin/env python3

"""
Small multiprocessing test to verify the fixed implementation works.
Uses smaller parameters to test quickly.
"""

import sys
import os

# Read the main script and create a test version
print("Creating small multiprocessing test...")

with open('static_cluster_logged_mp.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Create test version with smaller parameters
test_content = content.replace(
    'N = 20000  # System size', 
    'N = 500  # System size (TEST: reduced for quick test)'
).replace(
    'steps = N//4  # Time steps - now we can handle the full computation with streaming',
    'steps = 50  # Time steps (TEST: reduced for quick test)'
).replace(
    'samples = 5  # Samples per deviation - changed from 1 to 5',
    'samples = 1  # Samples per deviation (TEST: reduced for quick test)'
).replace(
    'devs = [0, 0.1, 0.5, 1, 10]  # List of static noise deviation values',
    'devs = [0, 0.1]  # List of static noise deviation values (TEST: reduced for quick test)'
)

# Write test version
with open('test_static_multiprocessing.py', 'w', encoding='utf-8') as f:
    f.write(test_content)

print("Created test_static_multiprocessing.py with parameters:")
print("  N = 500")
print("  steps = 50") 
print("  samples = 1")
print("  devs = [0, 0.1]")
print("\nThis should complete quickly and verify the streaming multiprocessing works.")
print("Run with: C:\\Users\\jaime\\anaconda3\\envs\\QWAK2\\python.exe test_static_multiprocessing.py")
