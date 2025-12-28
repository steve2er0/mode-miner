#!/usr/bin/env python3
"""Run Mode Miner with synthetic test data.

This script loads the cantilever plate BDF and synthetic modal data
for testing the visualization without needing a real OP2 file.
"""

import sys
import os
from pathlib import Path

# Configure Qt API before any Qt imports
os.environ['QT_API'] = 'pyside6'

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from mode_miner.ui.main_window import MainWindow
from mode_miner.ingest.bdf_reader import load_bdf_mesh
from mode_miner.model.modal_model import ModalModel
from mode_miner.model.dof_map import DOFMap


def generate_synthetic_modes(node_ids: np.ndarray, node_coords: np.ndarray):
    """Generate synthetic mode shapes for testing."""
    n_nodes = len(node_ids)
    n_modes = 6
    
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    
    # Normalize x to 0-1 range
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Approximate frequencies (Hz)
    frequencies = np.array([15.2, 42.8, 95.3, 118.6, 186.4, 232.1])
    
    # Generate mode shapes: (n_modes, n_nodes, 6)
    eigenvectors = np.zeros((n_modes, n_nodes, 6))
    
    # Mode 1: First bending (Z)
    phi = x_norm ** 2
    eigenvectors[0, :, 2] = phi / np.max(np.abs(phi) + 1e-10)
    
    # Mode 2: First torsion (Z varies with x * y)
    phi = x_norm ** 1.5 * (y_norm - 0.5)
    eigenvectors[1, :, 2] = phi / np.max(np.abs(phi) + 1e-10)
    
    # Mode 3: Second bending
    phi = x_norm ** 2 * np.sin(1.5 * np.pi * x_norm)
    eigenvectors[2, :, 2] = phi / np.max(np.abs(phi) + 1e-10)
    
    # Mode 4: In-plane bending (Y)
    phi = x_norm ** 2
    eigenvectors[3, :, 1] = phi / np.max(np.abs(phi) + 1e-10)
    
    # Mode 5: Second torsion
    phi = x_norm ** 2 * np.sin(np.pi * y_norm)
    eigenvectors[4, :, 2] = phi / np.max(np.abs(phi) + 1e-10)
    
    # Mode 6: Third bending
    phi = x_norm ** 2 * np.sin(2.5 * np.pi * x_norm)
    eigenvectors[5, :, 2] = phi / np.max(np.abs(phi) + 1e-10)
    
    return ModalModel(
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        dof_map=DOFMap(node_ids),
        is_mass_normalized=True
    )


def generate_synthetic_frf(modal_model: ModalModel, freq_range=(0.1, 300), n_points=500):
    """Generate synthetic FRF data for testing."""
    frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
    
    # Simple synthetic FRF with peaks at mode frequencies
    magnitude = np.ones_like(frequencies) * 1e-6
    
    damping = 0.02  # 2% damping
    for mode_freq in modal_model.frequencies:
        # Add resonance peak
        r = frequencies / mode_freq
        H = 1 / np.sqrt((1 - r**2)**2 + (2*damping*r)**2)
        magnitude += H * 1e-5
    
    # Find peaks
    from scipy.signal import find_peaks
    peak_indices, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1)
    peaks = [(frequencies[i], magnitude[i]) for i in peak_indices]
    
    return frequencies, magnitude, peaks


def load_test_data(window):
    """Load test data after window is ready."""
    test_bdf = Path(__file__).parent / "test_data" / "cantilever_plate.bdf"
    
    if not test_bdf.exists():
        print(f"Test BDF not found: {test_bdf}")
        return
    
    print(f"Loading test BDF: {test_bdf}", flush=True)
    try:
        # Load BDF with raw object for tree
        from pyNastran.bdf.bdf import BDF
        bdf = BDF()
        bdf.read_bdf(str(test_bdf))
        window._bdf_raw = bdf
        
        bdf_data = load_bdf_mesh(str(test_bdf))
        window._bdf_data = bdf_data
        window._mesh_view.set_mesh(bdf_data)
        window._model_tree.set_bdf_data(bdf_data, bdf)
        window._dof_selector.set_valid_nodes(bdf_data.node_ids)
        
        # Generate synthetic modes
        print("Generating synthetic modal data...", flush=True)
        modal_model = generate_synthetic_modes(
            bdf_data.node_ids, 
            bdf_data.node_coords
        )
        window._modal_model = modal_model
        window._mesh_view.set_modal_model(modal_model)
        window._mode_list.set_modal_model(modal_model)
        
        # Generate synthetic FRF
        print("Generating synthetic FRF...", flush=True)
        frequencies, magnitude, peaks = generate_synthetic_frf(modal_model)
        window._frf_viewer.set_frf_data(frequencies, magnitude, peaks=peaks)
        
        n_nodes = len(bdf_data.node_ids)
        n_cells = bdf_data.mesh.n_cells
        n_modes = modal_model.n_modes
        print(f"Loaded: {n_nodes} nodes, {n_cells} elements, {n_modes} modes", flush=True)
        window._update_status(
            f"Test data: {n_nodes} nodes, {n_cells} elements, {n_modes} modes"
        )
        
    except Exception as e:
        print(f"Error loading test data: {e}", flush=True)
        import traceback
        traceback.print_exc()


def main():
    """Launch Mode Miner with test data."""
    # Enable high-DPI support on macOS
    if sys.platform == 'darwin':
        os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')
    
    app = QApplication(sys.argv)
    app.setApplicationName("Mode Miner - Test Mode")
    
    window = MainWindow()
    window.setWindowTitle("Mode Miner - Test Mode")
    window.show()
    app.processEvents()
    
    print("Window shown, loading test data...", flush=True)
    
    # Defer data loading
    QTimer.singleShot(500, lambda: load_test_data(window))
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
