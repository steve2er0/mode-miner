"""Main application window with 5-panel layout."""

from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QSplitter, QPushButton,
    QLabel, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from .mesh_view import MeshView
from .mode_list import ModeListWidget
from .model_tree import ModelTreeWidget
from .dof_selector import DOFSelectorWidget
from .frf_viewer import FRFViewerWidget
from ..ingest.bdf_reader import load_bdf_mesh, BDFData
from ..ingest.op2_reader import load_op2_modes
from ..model.modal_model import ModalModel


class MainWindow(QMainWindow):
    """Main Mode Miner application window.
    
    Layout:
    ┌─────────┬────────────────────────┬──────────┐
    │         │                        │          │
    │  Model  │   3D Model Viewer /    │   Mode   │
    │  Tree   │     Mode Viewer        │   List   │
    │         │                        │          │
    │         ├────────────────────────┤          │
    │         │                        │          │
    ├─────────┤  FRF / Response Viewer │          │
    │ DOF Sel │                        │          │
    └─────────┴────────────────────────┴──────────┘
    """
    
    def __init__(self):
        super().__init__()
        
        self._bdf_data: Optional[BDFData] = None
        self._bdf_raw = None  # Raw BDF object for tree
        self._modal_model: Optional[ModalModel] = None
        self._last_frf_result = None  # Store last FRF computation result
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("Mode Miner")
        self.resize(1600, 1000)
        
        # Apply dark theme
        self.setStyleSheet(self._get_stylesheet())
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Main horizontal splitter (left | center | right)
        self._main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self._main_splitter)
        
        # === LEFT PANEL (Model Tree + DOF Selector) ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        # Left vertical splitter
        left_splitter = QSplitter(Qt.Vertical)
        
        # Model Tree (top of left panel)
        self._model_tree = ModelTreeWidget()
        left_splitter.addWidget(self._model_tree)
        
        # DOF Selector (bottom of left panel)
        self._dof_selector = DOFSelectorWidget()
        left_splitter.addWidget(self._dof_selector)
        
        left_splitter.setSizes([600, 300])
        left_layout.addWidget(left_splitter)
        
        self._main_splitter.addWidget(left_panel)
        
        # === CENTER PANEL (3D Viewer + FRF Viewer) ===
        center_splitter = QSplitter(Qt.Vertical)
        
        # 3D Mesh View (top)
        self._mesh_view = MeshView()
        center_splitter.addWidget(self._mesh_view)
        
        # FRF Viewer (bottom)
        self._frf_viewer = FRFViewerWidget()
        center_splitter.addWidget(self._frf_viewer)
        
        center_splitter.setSizes([600, 250])
        self._main_splitter.addWidget(center_splitter)
        
        # === RIGHT PANEL (Mode List) ===
        self._mode_list = ModeListWidget()
        self._main_splitter.addWidget(self._mode_list)
        
        # Set main splitter sizes (left: 220, center: stretch, right: 200)
        self._main_splitter.setSizes([220, 1000, 200])
    
    def _get_stylesheet(self) -> str:
        """Get the application stylesheet."""
        return """
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                color: #e0e0e0;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            }
            QMenuBar {
                background-color: #16213e;
                color: #e0e0e0;
                padding: 4px;
            }
            QMenuBar::item:selected {
                background-color: #0f3460;
            }
            QMenu {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
            }
            QMenu::item:selected {
                background-color: #0f3460;
            }
            QPushButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1a4a7a;
            }
            QPushButton:pressed {
                background-color: #0a2540;
            }
            QPushButton:disabled {
                background-color: #2a2a3a;
                color: #606060;
            }
            QStatusBar {
                background-color: #16213e;
                color: #a0a0a0;
            }
            QSplitter::handle {
                background-color: #4a4a6a;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
        """
    
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        load_bdf_action = QAction("Load &BDF...", self)
        load_bdf_action.setShortcut("Ctrl+B")
        load_bdf_action.triggered.connect(self._on_load_bdf)
        file_menu.addAction(load_bdf_action)
        
        load_op2_action = QAction("Load &OP2...", self)
        load_op2_action.setShortcut("Ctrl+O")
        load_op2_action.triggered.connect(self._on_load_op2)
        file_menu.addAction(load_op2_action)
        
        generate_modes_action = QAction("&Generate Test Modes", self)
        generate_modes_action.setShortcut("Ctrl+G")
        generate_modes_action.triggered.connect(self._on_generate_test_modes)
        file_menu.addAction(generate_modes_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        reset_camera_action = QAction("&Reset Camera", self)
        reset_camera_action.setShortcut("R")
        reset_camera_action.triggered.connect(self._on_reset_camera)
        view_menu.addAction(reset_camera_action)
        
        stop_anim_action = QAction("&Stop Animation", self)
        stop_anim_action.setShortcut("Escape")
        stop_anim_action.triggered.connect(self._on_stop_animation)
        view_menu.addAction(stop_anim_action)
    
    def _setup_statusbar(self):
        """Setup the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._update_status("Ready - Load a BDF file to begin")
    
    def _connect_signals(self):
        """Connect widget signals."""
        # Model tree -> highlight elements and grids in 3D
        self._model_tree.grid_selected.connect(self._on_grids_selected)
        self._model_tree.element_selected.connect(self._on_elements_selected)
        self._model_tree.elements_highlight_requested.connect(self._on_elements_highlight)
        self._model_tree.clear_highlight_requested.connect(self._on_clear_all_overlays)
        
        # DOF selector -> update force/response markers
        self._dof_selector.input_dof_changed.connect(self._on_input_dof_changed)
        self._dof_selector.response_dof_changed.connect(self._on_response_dof_changed)
        self._dof_selector.compute_requested.connect(self._on_compute_frf)
        
        # Mode list -> animate mode
        self._mode_list.mode_selected.connect(self._on_mode_selected)
        
        # FRF viewer -> filter modes
        self._frf_viewer.peak_selected.connect(self._on_peak_selected)
    
    def _update_status(self, message: str):
        """Update status bar message."""
        self._statusbar.showMessage(message)
    
    def _on_load_bdf(self):
        """Handle Load BDF action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open BDF File",
            "",
            "BDF Files (*.bdf *.dat *.nas);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self._update_status(f"Loading {Path(file_path).name}...")
            
            # Clear existing modal data (node IDs will change)
            self._modal_model = None
            self._last_frf_result = None
            self._mode_list.clear()
            self._frf_viewer.clear()
            
            # Load with pyNastran
            from pyNastran.bdf.bdf import BDF
            bdf = BDF()
            bdf.read_bdf(file_path)
            self._bdf_raw = bdf
            
            # Load mesh data
            self._bdf_data = load_bdf_mesh(file_path)
            
            # Update UI
            self._mesh_view.set_mesh(self._bdf_data)
            self._model_tree.set_bdf_data(self._bdf_data, bdf)
            self._dof_selector.set_valid_nodes(self._bdf_data.node_ids)
            
            n_nodes = len(self._bdf_data.node_ids)
            n_cells = self._bdf_data.mesh.n_cells
            self._update_status(f"Loaded: {n_nodes} nodes, {n_cells} elements (load OP2 for modes)")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading BDF",
                f"Failed to load BDF file:\n{str(e)}"
            )
            self._update_status("Error loading BDF")
    
    def _on_load_op2(self):
        """Handle Load OP2 action."""
        if self._bdf_data is None:
            QMessageBox.warning(
                self,
                "No BDF Loaded",
                "Please load a BDF file first."
            )
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open OP2 File",
            "",
            "OP2 Files (*.op2);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self._update_status(f"Loading {Path(file_path).name}...")
            self._modal_model = load_op2_modes(
                file_path,
                node_ids=self._bdf_data.node_ids
            )
            
            self._mesh_view.set_modal_model(self._modal_model)
            self._mode_list.set_modal_model(self._modal_model)
            
            n_modes = self._modal_model.n_modes
            freq_range = (
                f"{self._modal_model.frequencies[0]:.2f} - "
                f"{self._modal_model.frequencies[-1]:.2f} Hz"
            )
            self._update_status(f"Loaded: {n_modes} modes ({freq_range})")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading OP2",
                f"Failed to load OP2 file:\n{str(e)}"
            )
            self._update_status("Error loading OP2")
    
    def _on_generate_test_modes(self):
        """Generate synthetic mode shapes for testing."""
        if self._bdf_data is None:
            QMessageBox.warning(
                self,
                "No BDF Loaded",
                "Please load a BDF file first."
            )
            return
        
        try:
            import numpy as np
            from ..model.modal_model import ModalModel
            from ..model.dof_map import DOFMap
            
            node_ids = self._bdf_data.node_ids
            node_coords = self._bdf_data.node_coords
            n_nodes = len(node_ids)
            n_modes = 6
            
            # Normalize coordinates
            x = node_coords[:, 0]
            y = node_coords[:, 1]
            z = node_coords[:, 2]
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            z_range = z.max() - z.min()
            x_norm = (x - x.min()) / (x_range + 1e-10)
            y_norm = (y - y.min()) / (y_range + 1e-10)
            z_norm = (z - z.min()) / (z_range + 1e-10) if z_range > 0.01 else np.zeros_like(z)
            
            # Synthetic frequencies (Hz)
            frequencies = np.array([15.2, 42.8, 95.3, 118.6, 186.4, 232.1])
            
            # Generate mode shapes: (n_modes, n_nodes, 6)
            # Use sinusoidal shapes that have non-zero values everywhere
            eigenvectors = np.zeros((n_modes, n_nodes, 6))
            
            # Mode 1: First bending (Z) - sin shape
            phi = np.sin(0.5 * np.pi * x_norm)
            eigenvectors[0, :, 2] = phi / (np.max(np.abs(phi)) + 1e-10)
            
            # Mode 2: First torsion - varies with y
            phi = np.sin(0.5 * np.pi * x_norm) * np.sin(np.pi * y_norm)
            eigenvectors[1, :, 2] = phi / (np.max(np.abs(phi)) + 1e-10)
            
            # Mode 3: Second bending
            phi = np.sin(1.0 * np.pi * x_norm)
            eigenvectors[2, :, 2] = phi / (np.max(np.abs(phi)) + 1e-10)
            
            # Mode 4: In-plane bending (Y)
            phi = np.sin(0.5 * np.pi * x_norm)
            eigenvectors[3, :, 1] = phi / (np.max(np.abs(phi)) + 1e-10)
            
            # Mode 5: Second torsion
            phi = np.sin(1.0 * np.pi * x_norm) * np.sin(np.pi * y_norm)
            eigenvectors[4, :, 2] = phi / (np.max(np.abs(phi)) + 1e-10)
            
            # Mode 6: Third bending
            phi = np.sin(1.5 * np.pi * x_norm)
            eigenvectors[5, :, 2] = phi / (np.max(np.abs(phi)) + 1e-10)
            
            print(f"[Modes] Generated {n_modes} synthetic modes for {n_nodes} nodes", flush=True)
            print(f"[Modes] Eigenvector shape: {eigenvectors.shape}", flush=True)
            print(f"[Modes] Max eigenvector values per mode: {[np.max(np.abs(eigenvectors[i])) for i in range(n_modes)]}", flush=True)
            
            self._modal_model = ModalModel(
                frequencies=frequencies,
                eigenvectors=eigenvectors,
                dof_map=DOFMap(node_ids),
                is_mass_normalized=True
            )
            
            self._mesh_view.set_modal_model(self._modal_model)
            self._mode_list.set_modal_model(self._modal_model)
            
            freq_range = f"{frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz"
            self._update_status(f"Generated {n_modes} synthetic modes ({freq_range})")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error Generating Modes",
                f"Failed to generate synthetic modes:\n{str(e)}"
            )
    
    def _on_grids_selected(self, grid_ids):
        """Handle grid selection from model tree (list of IDs)."""
        if not grid_ids:
            return
        
        # Show markers at selected grids
        self._mesh_view.set_selected_grids(grid_ids)
        
        if len(grid_ids) == 1:
            self._update_status(f"Selected Grid {grid_ids[0]}")
        else:
            self._update_status(f"Selected {len(grid_ids)} grids")
    
    def _on_elements_selected(self, element_ids):
        """Handle element selection from model tree (list of IDs)."""
        if not element_ids:
            return
        
        # Highlight selected elements
        self._mesh_view.highlight_elements(set(element_ids))
        
        if len(element_ids) == 1:
            self._update_status(f"Selected Element {element_ids[0]}")
        else:
            self._update_status(f"Selected {len(element_ids)} elements")
    
    def _on_elements_highlight(self, element_ids):
        """Handle element highlight request from model tree."""
        n_elems = len(element_ids)
        self._update_status(f"Highlighting {n_elems} elements")
        self._mesh_view.highlight_elements(element_ids)
    
    def _on_clear_all_overlays(self):
        """Handle clear all overlays request."""
        self._mesh_view.clear_all_overlays()
        self._update_status("Selection cleared")
    
    def _on_input_dof_changed(self, dof):
        """Handle input DOF change - update force arrow."""
        if dof:
            dof_labels = ['Tx','Ty','Tz','Rx','Ry','Rz']
            self._update_status(f"Input: Node {dof.grid_id}, {dof_labels[dof.component-1]}")
            
            # Validate grid exists
            if self._mesh_view.is_valid_grid(dof.grid_id):
                self._mesh_view.set_force_marker(dof.grid_id, dof.component)
            else:
                self._mesh_view.clear_force_marker()
                self._dof_selector.set_error(f"Input node {dof.grid_id} not in model")
        else:
            self._mesh_view.clear_force_marker()
    
    def _on_response_dof_changed(self, dof):
        """Handle response DOF change - update response marker."""
        if dof:
            dof_labels = ['Tx','Ty','Tz','Rx','Ry','Rz']
            self._update_status(f"Response: Node {dof.grid_id}, {dof_labels[dof.component-1]}")
            
            # Validate grid exists
            if self._mesh_view.is_valid_grid(dof.grid_id):
                self._mesh_view.set_response_marker(dof.grid_id, dof.component)
            else:
                self._mesh_view.clear_response_marker()
                self._dof_selector.set_error(f"Response node {dof.grid_id} not in model")
        else:
            self._mesh_view.clear_response_marker()
    
    def _on_compute_frf(self):
        """Handle FRF computation request."""
        print(f"[FRF] Compute requested", flush=True)
        
        if self._modal_model is None:
            print("[FRF] No modal model loaded", flush=True)
            QMessageBox.warning(
                self,
                "No Modal Data",
                "Please load an OP2 file first."
            )
            return
        
        input_dof = self._dof_selector.get_input_dof()
        response_dof = self._dof_selector.get_response_dof()
        
        print(f"[FRF] Input DOF: {input_dof}, Response DOF: {response_dof}", flush=True)
        
        if input_dof is None:
            self._dof_selector.set_error("Please enter an input node ID")
            return
        if response_dof is None:
            self._dof_selector.set_error("Please enter a response node ID")
            return
        
        self._update_status("Computing FRF...")
        self._dof_selector.clear_error()
        
        # Get frequency range and damping
        freq_min, freq_max, freq_step = self._dof_selector.get_freq_range()
        damping = self._dof_selector.get_damping()
        
        print(f"[FRF] Freq: {freq_min}-{freq_max} Hz, step={freq_step}, damping={damping}", flush=True)
        
        try:
            from ..compute.frf import compute_mobility_frf
            
            result = compute_mobility_frf(
                modal_model=self._modal_model,
                input_dof=input_dof,
                response_dof=response_dof,
                freq_min=freq_min,
                freq_max=freq_max,
                freq_step=freq_step,
                damping=damping
            )
            
            print(f"[FRF] Computed: {len(result.frequencies)} points, {len(result.peaks)} peaks", flush=True)
            print(f"[FRF] Magnitude range: {result.magnitude.min():.2e} - {result.magnitude.max():.2e}", flush=True)
            
            # Store result for peak selection
            self._last_frf_result = result
            
            # Update FRF viewer
            self._frf_viewer.set_frf_data(
                result.frequencies,
                result.magnitude,
                result.phase,
                result.peaks
            )
            
            n_peaks = len(result.peaks)
            self._update_status(
                f"FRF computed: {len(result.frequencies)} points, {n_peaks} peaks detected"
            )
            
        except ValueError as e:
            print(f"[FRF] ValueError: {e}", flush=True)
            self._dof_selector.set_error(str(e))
            self._update_status(f"FRF error: {e}")
        except Exception as e:
            print(f"[FRF] Exception: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._dof_selector.set_error(f"Error: {e}")
            self._update_status(f"FRF error: {e}")
    
    def _on_mode_selected(self, mode_index: int):
        """Handle mode selection from list."""
        if self._modal_model is not None:
            freq = self._modal_model.frequencies[mode_index]
            self._update_status(f"Animating Mode {mode_index + 1}: {freq:.2f} Hz")
            self._mesh_view.animate_mode(mode_index)
    
    def _on_peak_selected(self, freq: float, magnitude: float):
        """Handle peak selection from FRF viewer."""
        self._update_status(f"Peak selected: {freq:.2f} Hz")
        
        if self._modal_model is None:
            return
        
        contributions = []
        
        # Use stored FRF result to get DOFs for contribution calculation
        if hasattr(self, '_last_frf_result') and self._last_frf_result is not None:
            try:
                from ..compute.frf import compute_mode_contributions
                
                contributions = compute_mode_contributions(
                    modal_model=self._modal_model,
                    input_dof=self._last_frf_result.input_dof,
                    response_dof=self._last_frf_result.response_dof,
                    target_freq=freq,
                    damping=self._last_frf_result.damping,
                    top_n=5
                )
                
            except Exception as e:
                print(f"[Peak] Error computing contributions: {e}", flush=True)
                contributions = []
        
        # Fallback to simple nearest mode if no contributions
        if not contributions:
            mode_freqs = self._modal_model.frequencies
            closest_idx = int(abs(mode_freqs - freq).argmin())
            contributions = [(closest_idx, 100.0)]
        
        # Update mode list with contributing modes
        self._mode_list.set_peak_filtered_modes(freq, contributions)
        
        # Animate the dominant mode (first in list, highest contribution)
        if contributions:
            dominant_mode_idx = contributions[0][0]
            dominant_contrib = contributions[0][1]
            mode_freq = self._modal_model.frequencies[dominant_mode_idx]
            
            self._update_status(
                f"Peak @ {freq:.1f} Hz → Mode {dominant_mode_idx + 1} "
                f"({mode_freq:.1f} Hz, {dominant_contrib:.1f}%)"
            )
            
            # Select the mode in the list and animate it
            self._mode_list.select_mode(dominant_mode_idx)
            self._mesh_view.animate_mode(dominant_mode_idx)
    
    def _on_stop_animation(self):
        """Handle stop animation."""
        self._mesh_view.stop_animation()
        self._mode_list.clear_selection()
        self._update_status("Animation stopped")
    
    def _on_reset_camera(self):
        """Reset camera to default view."""
        if self._bdf_data is not None:
            self._mesh_view.set_mesh(self._bdf_data)
