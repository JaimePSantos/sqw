#!/usr/bin/env python3

"""
Test script to verify that deviation tuple interpretation is fixed.
This should show that (0, 0.8) is correctly interpreted as min=0, max=0.8.
"""

import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_loading_static import format_deviation_for_filename

def test_deviation_formats():
    """Test different deviation formats to ensure correct interpretation."""
    
    print("Testing deviation format interpretation:")
    print("=" * 50)
    
    # Test cases that should work correctly now
    test_cases = [
        (0, "Single value: 0 (no noise)"),
        (0.5, "Single value: 0.5"),
        ((0, 0), "Tuple: (0, 0) - no noise"),
        ((0, 0.2), "Tuple: (0, 0.2) - small noise range"),
        ((0, 0.6), "Tuple: (0, 0.6) - medium noise range"),  
        ((0, 0.8), "Tuple: (0, 0.8) - medium noise range"),
        ((0, 1), "Tuple: (0, 1) - large noise range"),
        ((0.1, 0.5), "Tuple: (0.1, 0.5) - custom range"),
    ]
    
    for dev, description in test_cases:
        formatted = format_deviation_for_filename(dev, use_legacy_format=False)
        print(f"{description:<40} -> {formatted}")
    
    print("\n" + "=" * 50)
    print("Key verification:")
    print("(0, 0.8) should show as 'min0.000_max0.800' (NOT 'max0.000_min0.000')")
    
    # The problematic case that was incorrectly interpreted before
    result = format_deviation_for_filename((0, 0.8), use_legacy_format=False)
    if result == "min0.000_max0.800":
        print("✅ FIXED: (0, 0.8) correctly interpreted as min=0, max=0.8")
    else:
        print(f"❌ STILL BROKEN: (0, 0.8) incorrectly gives: {result}")
    
    return result == "min0.000_max0.800"

if __name__ == "__main__":
    success = test_deviation_formats()
    if success:
        print("\n🎉 All tests passed! The deviation interpretation bug is fixed.")
    else:
        print("\n💥 Tests failed! There are still issues with deviation interpretation.")
    
    sys.exit(0 if success else 1)
