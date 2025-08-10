#!/usr/bin/env python3
"""
Demonstration of improved deviation ranges in static noise experiments
"""

import numpy as np
import matplotlib.pyplot as plt
from StaggeredQW_static_noise import staggered_qwalk_with_noise

def demonstrate_deviation_ranges():
    """
    Demonstrate the new deviation range functionality
    """
    print("🔬 Static Noise Deviation Ranges Demonstration")
    print("=" * 50)
    
    N = 6
    theta = np.pi / 4
    steps = 1
    init_nodes = [N//2]
    
    # Test different deviation range formats
    test_cases = [
        ("No noise", 0.0),                    # Single value: backward compatibility
        ("Symmetric ±0.1", 0.1),              # Single value: creates [-0.1, +0.1]
        ("Exact range [0.0, 0.15]", (0.0, 0.15)),  # Tuple: exact range
        ("Negative range [-0.2, -0.05]", (-0.2, -0.05)),  # Negative range
        ("Mixed range [-0.1, +0.2]", (-0.1, 0.2))   # Mixed range
    ]
    
    print(f"System parameters: N={N}, theta={theta:.4f}, steps={steps}")
    print(f"Initial nodes: {init_nodes}")
    print()
    
    results = []
    
    for name, deviation_range in test_cases:
        print(f"🎯 Testing: {name}")
        print(f"   Deviation range: {deviation_range}")
        
        # Run experiment
        probabilities, _, _, _, _, red_noise, blue_noise = staggered_qwalk_with_noise(
            N, theta, steps, init_nodes, deviation_range
        )
        
        results.append({
            'name': name,
            'deviation_range': deviation_range,
            'probabilities': probabilities,
            'red_noise': red_noise,
            'blue_noise': blue_noise
        })
        
        # Show noise parameters generated
        print(f"   Red tessellation noise: {[f'{x:.4f}' for x in red_noise]}")
        print(f"   Blue tessellation noise: {[f'{x:.4f}' for x in blue_noise]}")
        print(f"   Max probability: {np.max(probabilities):.4f} at node {np.argmax(probabilities)}")
        print()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Probability distributions
    positions = np.arange(N)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        ax1.plot(positions, result['probabilities'].flatten(), 
                'o-', color=colors[i], label=result['name'], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Node Position')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability Distributions for Different Deviation Ranges')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise parameter ranges
    for i, result in enumerate(results):
        if result['name'] != "No noise":
            red_min, red_max = min(result['red_noise']), max(result['red_noise'])
            blue_min, blue_max = min(result['blue_noise']), max(result['blue_noise'])
            
            # Plot red noise range
            ax2.scatter([i-0.1], [(red_min + red_max)/2], 
                       color='red', s=100, alpha=0.7, label='Red' if i == 1 else "")
            ax2.plot([i-0.1, i-0.1], [red_min, red_max], 'r-', linewidth=3, alpha=0.7)
            
            # Plot blue noise range
            ax2.scatter([i+0.1], [(blue_min + blue_max)/2], 
                       color='blue', s=100, alpha=0.7, label='Blue' if i == 1 else "")
            ax2.plot([i+0.1, i+0.1], [blue_min, blue_max], 'b-', linewidth=3, alpha=0.7)
    
    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Noise Parameter Value')
    ax2.set_title('Actual Noise Parameter Ranges Generated')
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([r['name'] for r in results], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deviation_ranges_demonstration.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved to 'deviation_ranges_demonstration.png'")
    plt.show()
    
    # Summary
    print("\n=== Summary ===")
    print("✅ Backward compatibility: Single values still work (create symmetric ranges)")
    print("✅ New functionality: Tuples (min, max) allow exact range specification")
    print("✅ Flexible ranges: Support positive, negative, and mixed ranges")
    print("✅ Independent tessellations: Red and blue get separate random noise from same range")
    print("\nThe new system provides much more control over the noise characteristics!")

if __name__ == "__main__":
    demonstrate_deviation_ranges()
