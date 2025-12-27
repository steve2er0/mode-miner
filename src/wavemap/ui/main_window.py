"""Main application window."""

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
from ..ingest.bdf_reader import load_bdf_mesh, BDFData
from ..ingest.op2_reader import load_op2_modes
from ..model.modal_model import ModalModel


class MainWindow(QMainWindow):
    """Main Wavemap application window."""
    
    def __init__(self):
        super().__init__()
        
        self._bdf_data: Optional[BDFData] = None
        self._modal_model: Optional[ModalModel] = None
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("Wavemap")
        self.resize(1400, 900)
        
        # Apply dark theme
        self.setStyleSheet("""
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
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1a4a7a;
            }
            QPushButton:pressed {
                background-color: #0a2540;
            }
            QStatusBar {
                background-color: #16213e;
                color: #a0a0a0;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        # Load buttons
        self._load_bdf_btn = QPushButton("Load BDF")
        self._load_bdf_btn.clicked.connect(self._on_load_bdf)
        left_layout.addWidget(self._load_bdf_btn)
        
        self._load_op2_btn = QPushButton("Load OP2")
        self._load_op2_btn.clicked.connect(self._on_load_op2)
        left_layout.addWidget(self._load_op2_btn)
        
        # Mode list
        self._mode_list = ModeListWidget()
        self._mode_list.mode_selected.connect(self._on_mode_selected)
        left_layout.addWidget(self._mode_list)
        
        # Animation controls
        self._stop_btn = QPushButton("Stop Animation")
        self._stop_btn.clicked.connect(self._on_stop_animation)
        left_layout.addWidget(self._stop_btn)
        
        splitter.addWidget(left_panel)
        
        # Right panel - mesh view
        self._mesh_view = MeshView()
        splitter.addWidget(self._mesh_view)
        
        # Set splitter sizes (20% left, 80% right)
        splitter.setSizes([280, 1120])
    
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
    
    def _setup_statusbar(self):
        """Setup the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._update_status("Ready - Load a BDF file to begin")
    
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
            self._bdf_data = load_bdf_mesh(file_path)
            self._mesh_view.set_mesh(self._bdf_data)
            
            n_nodes = len(self._bdf_data.node_ids)
            n_cells = self._bdf_data.mesh.n_cells
            self._update_status(
                f"Loaded: {n_nodes} nodes, {n_cells} elements"
            )
            
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
    
    def _on_mode_selected(self, mode_index: int):
        """Handle mode selection from list.
        
        Args:
            mode_index: 0-based mode index
        """
        if self._modal_model is not None:
            freq = self._modal_model.frequencies[mode_index]
            self._update_status(f"Animating Mode {mode_index + 1}: {freq:.2f} Hz")
            self._mesh_view.animate_mode(mode_index)
    
    def _on_stop_animation(self):
        """Handle stop animation button."""
        self._mesh_view.stop_animation()
        self._mode_list.clear_selection()
        self._update_status("Animation stopped")
    
    def _on_reset_camera(self):
        """Reset camera to default view."""
        # Re-render mesh to reset view
        if self._bdf_data is not None:
            self._mesh_view.set_mesh(self._bdf_data)

