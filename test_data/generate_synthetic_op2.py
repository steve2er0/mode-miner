#!/usr/bin/env python3
"""Generate synthetic modal data for testing Wavemap.

This script creates a synthetic OP2-like data structure that can be used
to test the mode animation without needing a real NASTRAN run.

Usage:
    python generate_synthetic_op2.py
    
This will create synthetic_modes.npz containing modal data matching
the cantilever_plate.bdf model.
"""

import numpy as np
from pathlib import Path


def generate_cantilever_modes():
    """Generate synthetic mode shapes for a cantilever plate.
    
    Creates approximate mode shapes based on analytical beam modes.
    """
    # Grid layout: 11 columns (x) x 5 rows (y) = 55 nodes
    nx, ny = 11, 5
    n_nodes = nx * ny
    n_modes = 6
    
    # Node IDs (1-55)
    node_ids = np.arange(1, n_nodes + 1, dtype=np.int64)
    
    # Node coordinates
    x = np.tile(np.linspace(0, 1.0, nx), ny)
    y = np.repeat(np.linspace(0, 0.4, ny), nx)
    z = np.zeros(n_nodes)
    
    # Approximate frequencies for a cantilever plate (Hz)
    # These are rough estimates for illustration
    frequencies = np.array([15.2, 42.8, 95.3, 118.6, 186.4, 232.1])
    
    # Generate mode shapes
    # Shape: (n_modes, n_nodes, 6) for [T1, T2, T3, R1, R2, R3]
    eigenvectors = np.zeros((n_modes, n_nodes, 6))
    
    # Mode 1: First bending (Z displacement varies with x^2)
    phi1_z = (x / 1.0) ** 2
    eigenvectors[0, :, 2] = phi1_z / np.max(np.abs(phi1_z))
    
    # Mode 2: First torsion (Z varies with x * y)
    phi2_z = (x / 1.0) ** 1.5 * (y - 0.2)
    eigenvectors[1, :, 2] = phi2_z / np.max(np.abs(phi2_z))
    
    # Mode 3: Second bending
    phi3_z = (x / 1.0) ** 2 * np.sin(1.5 * np.pi * x)
    eigenvectors[2, :, 2] = phi3_z / np.max(np.abs(phi3_z))
    
    # Mode 4: First in-plane bending (Y displacement)
    phi4_y = (x / 1.0) ** 2
    eigenvectors[3, :, 1] = phi4_y / np.max(np.abs(phi4_y))
    
    # Mode 5: Second torsion
    phi5_z = (x / 1.0) ** 2 * np.sin(np.pi * y / 0.4)
    eigenvectors[4, :, 2] = phi5_z / np.max(np.abs(phi5_z))
    
    # Mode 6: Third bending
    phi6_z = (x / 1.0) ** 2 * np.sin(2.5 * np.pi * x)
    eigenvectors[5, :, 2] = phi6_z / np.max(np.abs(phi6_z))
    
    return {
        'node_ids': node_ids,
        'frequencies': frequencies,
        'eigenvectors': eigenvectors,
    }


def main():
    """Generate and save synthetic modal data."""
    output_dir = Path(__file__).parent
    
    print("Generating synthetic modal data for cantilever plate...")
    data = generate_cantilever_modes()
    
    output_file = output_dir / 'synthetic_modes.npz'
    np.savez(
        output_file,
        node_ids=data['node_ids'],
        frequencies=data['frequencies'],
        eigenvectors=data['eigenvectors']
    )
    
    print(f"Saved to: {output_file}")
    print(f"  Nodes: {len(data['node_ids'])}")
    print(f"  Modes: {len(data['frequencies'])}")
    print(f"  Frequencies: {data['frequencies']} Hz")


if __name__ == '__main__':
    main()

