"""Mesh viewer widget using matplotlib 3D."""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..ingest.bdf_reader import BDFData
from ..model.modal_model import ModalModel


class MeshView(QWidget):
    """3D mesh visualization widget using matplotlib.
    
    Displays the structural mesh and animates mode shapes.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure and canvas
        self._figure = Figure(facecolor='#1a1a2e')
        self._canvas = FigureCanvas(self._figure)
        self._layout.addWidget(self._canvas)
        
        # Create 3D axes
        self._ax = self._figure.add_subplot(111, projection='3d')
        self._setup_axes()
        
        # Mesh data
        self._bdf_data: Optional[BDFData] = None
        self._modal_model: Optional[ModalModel] = None
        self._poly_collection = None
        self._faces = None  # Store face indices for animation
        
        # Animation state
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_step)
        self._animation_phase = 0.0
        self._animation_mode_index: Optional[int] = None
        self._animation_scale = 1.0
        self._base_points: Optional[np.ndarray] = None
        self._mode_displacements: Optional[np.ndarray] = None
        
        # Animation parameters
        self._animation_fps = 20
        self._animation_speed = 1.0  # cycles per second
        
        self._canvas.draw()
    
    def _setup_axes(self):
        """Configure axes appearance."""
        self._ax.set_facecolor('#1a1a2e')
        self._ax.set_xlabel('X', color='white')
        self._ax.set_ylabel('Y', color='white')
        self._ax.set_zlabel('Z', color='white')
        self._ax.tick_params(colors='white')
        
        # Make panes transparent
        self._ax.xaxis.pane.fill = False
        self._ax.yaxis.pane.fill = False
        self._ax.zaxis.pane.fill = False
        
        # Make grid lines subtle
        self._ax.xaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.4, 0.3)
        self._ax.yaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.4, 0.3)
        self._ax.zaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.4, 0.3)
    
    def set_mesh(self, bdf_data: BDFData):
        """Set the mesh to display.
        
        Args:
            bdf_data: BDF data containing mesh
        """
        self.stop_animation()
        
        self._bdf_data = bdf_data
        self._base_points = bdf_data.node_coords.copy()
        
        # Extract faces from PyVista mesh
        self._faces = self._extract_faces(bdf_data.mesh)
        
        # Render the mesh
        self._render_mesh(self._base_points)
    
    def _extract_faces(self, mesh):
        """Extract face vertex indices from PyVista mesh."""
        faces = []
        
        if mesh.n_cells == 0:
            return faces
        
        # Get cell connectivity
        cells = mesh.cells
        i = 0
        while i < len(cells):
            n_verts = cells[i]
            face_indices = cells[i+1:i+1+n_verts]
            faces.append(face_indices)
            i += n_verts + 1
        
        return faces
    
    def _render_mesh(self, points: np.ndarray):
        """Render the mesh with given vertex positions."""
        self._ax.clear()
        self._setup_axes()
        
        if self._faces is None or len(self._faces) == 0:
            self._canvas.draw()
            return
        
        # Build polygon vertices
        verts = []
        for face_idx in self._faces:
            face_verts = points[face_idx]
            verts.append(face_verts)
        
        # Create polygon collection
        self._poly_collection = Poly3DCollection(
            verts,
            facecolors='steelblue',
            edgecolors='white',
            linewidths=0.5,
            alpha=0.9
        )
        self._ax.add_collection3d(self._poly_collection)
        
        # Set axis limits
        margin = 0.1
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # Ensure some Z range even for flat meshes
        z_range = z_max - z_min
        if z_range < 0.01:
            z_center = (z_max + z_min) / 2
            max_range = max(x_max - x_min, y_max - y_min)
            z_min = z_center - max_range / 4
            z_max = z_center + max_range / 4
        
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin
        z_margin = (z_max - z_min) * margin
        
        self._ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self._ax.set_ylim(y_min - y_margin, y_max + y_margin)
        self._ax.set_zlim(z_min - z_margin, z_max + z_margin)
        
        # Set equal aspect ratio
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        self._ax.set_box_aspect([
            (x_max - x_min) / max_range,
            (y_max - y_min) / max_range,
            (z_max - z_min) / max_range
        ])
        
        self._canvas.draw()
    
    def set_modal_model(self, modal_model: ModalModel):
        """Set the modal model for animation.
        
        Args:
            modal_model: Modal analysis results
        """
        self._modal_model = modal_model
    
    def animate_mode(self, mode_index: int, scale: Optional[float] = None):
        """Start animating a mode shape.
        
        Args:
            mode_index: 0-based mode index
            scale: Displacement scale factor. If None, auto-scales.
        """
        if self._modal_model is None or self._bdf_data is None:
            return
        
        if mode_index < 0 or mode_index >= self._modal_model.n_modes:
            return
        
        self.stop_animation()
        
        # Get translation displacements for this mode
        self._mode_displacements = self._modal_model.get_translation_shape(mode_index)
        
        # Auto-scale if not specified
        if scale is None:
            scale = self._compute_auto_scale()
        
        self._animation_scale = scale
        self._animation_mode_index = mode_index
        self._animation_phase = 0.0
        
        # Start animation timer
        interval_ms = int(1000 / self._animation_fps)
        self._animation_timer.start(interval_ms)
    
    def stop_animation(self):
        """Stop any running animation."""
        self._animation_timer.stop()
        self._animation_mode_index = None
        
        # Reset mesh to base position
        if self._bdf_data is not None and self._base_points is not None:
            self._render_mesh(self._base_points)
    
    def _compute_auto_scale(self) -> float:
        """Compute automatic scale factor for mode animation."""
        if self._mode_displacements is None or self._base_points is None:
            return 1.0
        
        # Get model bounding box size
        bbox_size = np.max(self._base_points, axis=0) - np.min(self._base_points, axis=0)
        model_size = np.max(bbox_size)
        
        # Get max displacement magnitude
        disp_magnitude = np.linalg.norm(self._mode_displacements, axis=1)
        max_disp = np.max(disp_magnitude)
        
        if max_disp < 1e-12:
            return 1.0
        
        # Target 15% of model size for good visibility
        target_disp = 0.15 * model_size
        return target_disp / max_disp
    
    def _animation_step(self):
        """Perform one animation frame."""
        if self._mode_displacements is None or self._base_points is None:
            return
        
        # Update phase
        phase_increment = (2 * np.pi * self._animation_speed) / self._animation_fps
        self._animation_phase += phase_increment
        
        # Compute displaced positions
        scale_factor = self._animation_scale * np.sin(self._animation_phase)
        displaced = self._base_points + scale_factor * self._mode_displacements
        
        self._update_mesh_points(displaced)
    
    def _update_mesh_points(self, points: np.ndarray):
        """Update mesh vertex positions for animation."""
        if self._poly_collection is None or self._faces is None:
            return
        
        # Build new polygon vertices
        verts = []
        for face_idx in self._faces:
            face_verts = points[face_idx]
            verts.append(face_verts)
        
        # Update the collection
        self._poly_collection.set_verts(verts)
        self._canvas.draw_idle()
    
    @property
    def is_animating(self) -> bool:
        """Whether animation is currently running."""
        return self._animation_timer.isActive()
