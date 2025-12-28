"""Mesh viewer widget using matplotlib 3D with overlays for markers and arrows."""

from typing import Optional, Set, List, Tuple
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..ingest.bdf_reader import BDFData
from ..model.modal_model import ModalModel


# DOF component to direction vector mapping (1-indexed)
DOF_DIRECTIONS = {
    1: np.array([1.0, 0.0, 0.0]),  # Tx -> +X
    2: np.array([0.0, 1.0, 0.0]),  # Ty -> +Y
    3: np.array([0.0, 0.0, 1.0]),  # Tz -> +Z
    4: np.array([1.0, 0.0, 0.0]),  # Rx -> rotation about X
    5: np.array([0.0, 1.0, 0.0]),  # Ry -> rotation about Y
    6: np.array([0.0, 0.0, 1.0]),  # Rz -> rotation about Z
}


class MeshView(QWidget):
    """3D mesh visualization widget with element highlighting and markers.
    
    Displays the structural mesh, animates mode shapes, and supports:
    - Element highlighting
    - Grid point markers
    - Force direction arrows
    - Response location markers
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
        self._highlight_collection = None
        self._faces = None
        self._face_to_element: dict = {}
        
        # Node ID to coordinate index mapping
        self._node_id_to_idx: dict = {}
        
        # Animation state
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_step)
        self._animation_phase = 0.0
        self._animation_mode_index: Optional[int] = None
        self._animation_scale = 1.0
        self._base_points: Optional[np.ndarray] = None
        self._mode_displacements: Optional[np.ndarray] = None
        self._animation_fps = 20
        self._animation_speed = 1.0
        
        # Overlay state
        self._highlighted_elements: Set[int] = set()
        self._selected_grids: List[int] = []
        self._force_grid_id: Optional[int] = None
        self._force_component: Optional[int] = None
        self._response_grid_id: Optional[int] = None
        self._response_component: Optional[int] = None
        
        # Model bounding box for scaling
        self._model_diagonal = 1.0
        
        self._canvas.draw()
    
    def _setup_axes(self):
        """Configure axes appearance."""
        self._ax.set_facecolor('#16213e')
        self._ax.set_xlabel('X', color='white', fontsize=10)
        self._ax.set_ylabel('Y', color='white', fontsize=10)
        self._ax.set_zlabel('Z', color='white', fontsize=10)
        self._ax.tick_params(colors='white', labelsize=9)
        
        self._ax.xaxis.pane.fill = False
        self._ax.yaxis.pane.fill = False
        self._ax.zaxis.pane.fill = False
        
        self._ax.xaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.4, 0.3)
        self._ax.yaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.4, 0.3)
        self._ax.zaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.4, 0.3)
    
    def set_mesh(self, bdf_data: BDFData):
        """Set the mesh to display."""
        self.stop_animation()
        self.clear_all_overlays()
        
        self._bdf_data = bdf_data
        self._base_points = bdf_data.node_coords.copy()
        
        # Build node ID to index mapping
        self._node_id_to_idx = {
            int(nid): idx for idx, nid in enumerate(bdf_data.node_ids)
        }
        
        # Compute model bounding box diagonal for scaling
        bbox = np.max(self._base_points, axis=0) - np.min(self._base_points, axis=0)
        self._model_diagonal = np.linalg.norm(bbox)
        if self._model_diagonal < 1e-10:
            self._model_diagonal = 1.0
        
        # Extract faces with element mapping
        self._faces, self._face_to_element = self._extract_faces_with_mapping(bdf_data)
        
        self._render_full()
    
    def _extract_faces_with_mapping(self, bdf_data: BDFData):
        """Extract face vertex indices with element ID mapping."""
        faces = []
        face_to_elem = {}
        
        mesh = bdf_data.mesh
        if mesh.n_cells == 0:
            return faces, face_to_elem
        
        cells = mesh.cells
        i = 0
        face_idx = 0
        cell_idx = 0
        
        while i < len(cells):
            n_verts = cells[i]
            face_indices = cells[i+1:i+1+n_verts]
            faces.append(face_indices)
            
            if cell_idx in bdf_data.cell_idx_to_element_id:
                face_to_elem[face_idx] = bdf_data.cell_idx_to_element_id[cell_idx]
            
            i += n_verts + 1
            face_idx += 1
            cell_idx += 1
        
        return faces, face_to_elem
    
    def _render_full(self):
        """Full render including mesh and all overlays."""
        points = self._base_points
        if points is None:
            return
        
        self._ax.clear()
        self._setup_axes()
        
        # Render mesh
        self._render_mesh_faces(points)
        
        # Render overlays
        self._render_grid_markers(points)
        self._render_force_arrow(points)
        self._render_response_marker(points)
        
        self._set_axis_limits(points)
        self._canvas.draw()
    
    def _render_mesh_faces(self, points: np.ndarray):
        """Render the mesh faces with highlighting."""
        if self._faces is None or len(self._faces) == 0:
            return
        
        normal_verts = []
        highlight_verts = []
        
        for i, face_idx in enumerate(self._faces):
            face_verts = points[face_idx]
            elem_id = self._face_to_element.get(i)
            
            if elem_id is not None and elem_id in self._highlighted_elements:
                highlight_verts.append(face_verts)
            else:
                normal_verts.append(face_verts)
        
        # Normal elements
        if normal_verts:
            self._poly_collection = Poly3DCollection(
                normal_verts,
                facecolors='steelblue',
                edgecolors='#4a4a6a',
                linewidths=0.3,
                alpha=0.7
            )
            self._ax.add_collection3d(self._poly_collection)
        
        # Highlighted elements
        if highlight_verts:
            self._highlight_collection = Poly3DCollection(
                highlight_verts,
                facecolors='#ff6b6b',
                edgecolors='white',
                linewidths=1.0,
                alpha=0.95
            )
            self._ax.add_collection3d(self._highlight_collection)
    
    def _render_grid_markers(self, points: np.ndarray):
        """Render spherical markers at selected grid points."""
        if not self._selected_grids:
            return
        
        marker_coords = []
        for grid_id in self._selected_grids:
            if grid_id in self._node_id_to_idx:
                idx = self._node_id_to_idx[grid_id]
                marker_coords.append(points[idx])
        
        if marker_coords:
            coords = np.array(marker_coords)
            marker_size = max(50, 200 * (self._model_diagonal / 10))
            self._ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c='#00ff88', s=marker_size, marker='o',
                edgecolors='white', linewidths=1.5,
                alpha=0.9, zorder=10
            )
    
    def _render_force_arrow(self, points: np.ndarray):
        """Render purple arrow at force input location."""
        if self._force_grid_id is None or self._force_component is None:
            return
        
        if self._force_grid_id not in self._node_id_to_idx:
            return
        
        idx = self._node_id_to_idx[self._force_grid_id]
        origin = points[idx]
        
        # Arrow length = 5% of model diagonal
        arrow_length = 0.08 * self._model_diagonal
        
        # Get direction from component
        if self._force_component in DOF_DIRECTIONS:
            direction = DOF_DIRECTIONS[self._force_component]
        else:
            direction = np.array([0, 0, 1])
        
        end = origin + direction * arrow_length
        
        # Check if rotational DOF
        is_rotation = self._force_component in [4, 5, 6]
        color = '#9933ff' if not is_rotation else '#cc66ff'
        
        # Draw arrow using quiver
        self._ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0] * arrow_length,
            direction[1] * arrow_length,
            direction[2] * arrow_length,
            color=color, arrow_length_ratio=0.3,
            linewidth=3, alpha=0.95, zorder=15
        )
        
        # Add force label
        label_offset = direction * arrow_length * 1.2
        dof_labels = ['', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        label = dof_labels[self._force_component] if self._force_component <= 6 else 'F'
        self._ax.text(
            origin[0] + label_offset[0],
            origin[1] + label_offset[1],
            origin[2] + label_offset[2],
            label, color=color, fontsize=10, fontweight='bold',
            zorder=16
        )
    
    def _render_response_marker(self, points: np.ndarray):
        """Render response location marker (diamond/star shape)."""
        if self._response_grid_id is None or self._response_component is None:
            return
        
        if self._response_grid_id not in self._node_id_to_idx:
            return
        
        idx = self._node_id_to_idx[self._response_grid_id]
        pos = points[idx]
        
        marker_size = max(100, 300 * (self._model_diagonal / 10))
        
        # Use diamond marker for response
        self._ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            c='#ffcc00', s=marker_size, marker='D',
            edgecolors='white', linewidths=2,
            alpha=0.95, zorder=12
        )
        
        # Optionally show small direction indicator for translation DOFs
        if self._response_component in [1, 2, 3]:
            direction = DOF_DIRECTIONS[self._response_component]
            arrow_length = 0.04 * self._model_diagonal
            
            self._ax.quiver(
                pos[0], pos[1], pos[2],
                direction[0] * arrow_length,
                direction[1] * arrow_length,
                direction[2] * arrow_length,
                color='#ffcc00', arrow_length_ratio=0.4,
                linewidth=2, alpha=0.8, zorder=13
            )
        
        # Add response label
        dof_labels = ['', 'vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz']
        label = dof_labels[self._response_component] if self._response_component <= 6 else 'v'
        offset = 0.03 * self._model_diagonal
        self._ax.text(
            pos[0] + offset, pos[1] + offset, pos[2] + offset,
            label, color='#ffcc00', fontsize=10, fontweight='bold',
            zorder=14
        )
    
    def _set_axis_limits(self, points: np.ndarray):
        """Set axis limits based on points."""
        margin = 0.15
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
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
        
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range > 0:
            self._ax.set_box_aspect([
                (x_max - x_min + 2*x_margin) / (max_range + 2*max(x_margin, y_margin, z_margin)),
                (y_max - y_min + 2*y_margin) / (max_range + 2*max(x_margin, y_margin, z_margin)),
                (z_max - z_min + 2*z_margin) / (max_range + 2*max(x_margin, y_margin, z_margin))
            ])
    
    # === Public API for overlays ===
    
    def highlight_elements(self, element_ids: Set[int]):
        """Highlight a set of elements."""
        self._highlighted_elements = element_ids
        self._render_full()
    
    def clear_highlight(self):
        """Clear element highlighting."""
        self._highlighted_elements = set()
        self._render_full()
    
    def set_selected_grids(self, grid_ids: List[int]):
        """Set grid points to mark with spheres."""
        self._selected_grids = grid_ids
        self._render_full()
    
    def clear_selected_grids(self):
        """Clear grid markers."""
        self._selected_grids = []
        self._render_full()
    
    def set_force_marker(self, grid_id: Optional[int], component: Optional[int]):
        """Set force arrow location and direction.
        
        Args:
            grid_id: Grid point ID for force location (None to clear)
            component: DOF component 1-6 (None to clear)
        """
        self._force_grid_id = grid_id
        self._force_component = component
        self._render_full()
    
    def clear_force_marker(self):
        """Remove force arrow."""
        self._force_grid_id = None
        self._force_component = None
        self._render_full()
    
    def set_response_marker(self, grid_id: Optional[int], component: Optional[int]):
        """Set response marker location and direction.
        
        Args:
            grid_id: Grid point ID for response location (None to clear)
            component: DOF component 1-6 (None to clear)
        """
        self._response_grid_id = grid_id
        self._response_component = component
        self._render_full()
    
    def clear_response_marker(self):
        """Remove response marker."""
        self._response_grid_id = None
        self._response_component = None
        self._render_full()
    
    def clear_all_overlays(self):
        """Clear all overlays (highlights, markers, arrows)."""
        self._highlighted_elements = set()
        self._selected_grids = []
        self._force_grid_id = None
        self._force_component = None
        self._response_grid_id = None
        self._response_component = None
        if self._base_points is not None:
            self._render_full()
    
    def get_grid_position(self, grid_id: int) -> Optional[np.ndarray]:
        """Get the XYZ position of a grid point."""
        if grid_id in self._node_id_to_idx and self._base_points is not None:
            return self._base_points[self._node_id_to_idx[grid_id]].copy()
        return None
    
    def is_valid_grid(self, grid_id: int) -> bool:
        """Check if a grid ID exists in the model."""
        return grid_id in self._node_id_to_idx
    
    # === Modal animation ===
    
    def set_modal_model(self, modal_model: ModalModel):
        """Set the modal model for animation."""
        self._modal_model = modal_model
    
    def animate_mode(self, mode_index: int, scale: Optional[float] = None):
        """Start animating a mode shape."""
        if self._modal_model is None or self._bdf_data is None:
            return
        
        if mode_index < 0 or mode_index >= self._modal_model.n_modes:
            return
        
        self.stop_animation()
        
        self._mode_displacements = self._modal_model.get_translation_shape(mode_index)
        
        if scale is None:
            scale = self._compute_auto_scale()
        
        self._animation_scale = scale
        self._animation_mode_index = mode_index
        self._animation_phase = 0.0
        
        interval_ms = int(1000 / self._animation_fps)
        self._animation_timer.start(interval_ms)
    
    def stop_animation(self):
        """Stop any running animation."""
        self._animation_timer.stop()
        self._animation_mode_index = None
        
        if self._bdf_data is not None and self._base_points is not None:
            self._render_full()
    
    def _compute_auto_scale(self) -> float:
        """Compute automatic scale factor for mode animation."""
        if self._mode_displacements is None or self._base_points is None:
            return 1.0
        
        disp_magnitude = np.linalg.norm(self._mode_displacements, axis=1)
        max_disp = np.max(disp_magnitude)
        
        if max_disp < 1e-12:
            return 1.0
        
        target_disp = 0.15 * self._model_diagonal
        return target_disp / max_disp
    
    def _animation_step(self):
        """Perform one animation frame."""
        if self._mode_displacements is None or self._base_points is None:
            return
        
        phase_increment = (2 * np.pi * self._animation_speed) / self._animation_fps
        self._animation_phase += phase_increment
        
        scale_factor = self._animation_scale * np.sin(self._animation_phase)
        displaced = self._base_points + scale_factor * self._mode_displacements
        
        self._update_mesh_points(displaced)
    
    def _update_mesh_points(self, points: np.ndarray):
        """Update mesh vertex positions for animation (fast path)."""
        if self._poly_collection is None or self._faces is None:
            return
        
        normal_verts = []
        highlight_verts = []
        
        for i, face_idx in enumerate(self._faces):
            face_verts = points[face_idx]
            elem_id = self._face_to_element.get(i)
            
            if elem_id is not None and elem_id in self._highlighted_elements:
                highlight_verts.append(face_verts)
            else:
                normal_verts.append(face_verts)
        
        if normal_verts:
            self._poly_collection.set_verts(normal_verts)
        if highlight_verts and self._highlight_collection:
            self._highlight_collection.set_verts(highlight_verts)
        
        self._canvas.draw_idle()
    
    @property
    def is_animating(self) -> bool:
        """Whether animation is currently running."""
        return self._animation_timer.isActive()
