"""DOF selection panel with frequency and damping controls."""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont, QIntValidator, QDoubleValidator

from ..model.dof_map import NodeDOF


DOF_LABELS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


class DOFSelectorWidget(QWidget):
    """Panel for selecting input/response DOFs and FRF parameters.
    
    Signals:
        input_dof_changed: Emitted when input DOF selection changes
        response_dof_changed: Emitted when response DOF selection changes
        compute_requested: Emitted when user requests FRF computation
    """
    
    input_dof_changed = Signal(object)
    response_dof_changed = Signal(object)
    compute_requested = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._valid_node_ids = set()
        self._constrained_dofs = set()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        
        # Header
        header = QLabel("Input / Response")
        header.setFont(QFont("", -1, QFont.Bold))
        header.setStyleSheet("color: #e0e0e0; font-size: 13px;")
        layout.addWidget(header)
        
        # Input DOF group
        input_group = QGroupBox("Input (Force)")
        input_group.setStyleSheet(self._group_style())
        input_layout = QFormLayout(input_group)
        
        self._input_node = QLineEdit()
        self._input_node.setPlaceholderText("Node ID")
        self._input_node.setValidator(QIntValidator(1, 99999999))
        self._input_node.setStyleSheet(self._input_style())
        self._input_node.textChanged.connect(self._on_input_changed)
        input_layout.addRow("Node:", self._input_node)
        
        self._input_dof = QComboBox()
        self._input_dof.addItems(DOF_LABELS)
        self._input_dof.setStyleSheet(self._combo_style())
        self._input_dof.currentIndexChanged.connect(self._on_input_changed)
        input_layout.addRow("DOF:", self._input_dof)
        
        layout.addWidget(input_group)
        
        # Response DOF group
        response_group = QGroupBox("Response (Velocity)")
        response_group.setStyleSheet(self._group_style())
        response_layout = QFormLayout(response_group)
        
        self._response_node = QLineEdit()
        self._response_node.setPlaceholderText("Node ID")
        self._response_node.setValidator(QIntValidator(1, 99999999))
        self._response_node.setStyleSheet(self._input_style())
        self._response_node.textChanged.connect(self._on_response_changed)
        response_layout.addRow("Node:", self._response_node)
        
        self._response_dof = QComboBox()
        self._response_dof.addItems(DOF_LABELS)
        self._response_dof.setStyleSheet(self._combo_style())
        self._response_dof.currentIndexChanged.connect(self._on_response_changed)
        response_layout.addRow("DOF:", self._response_dof)
        
        layout.addWidget(response_group)
        
        # Frequency range group
        freq_group = QGroupBox("Frequency Range")
        freq_group.setStyleSheet(self._group_style())
        freq_layout = QFormLayout(freq_group)
        
        self._freq_min = QDoubleSpinBox()
        self._freq_min.setRange(0.1, 10000)
        self._freq_min.setValue(1.0)
        self._freq_min.setSuffix(" Hz")
        self._freq_min.setStyleSheet(self._spinbox_style())
        freq_layout.addRow("Min:", self._freq_min)
        
        self._freq_max = QDoubleSpinBox()
        self._freq_max.setRange(1, 50000)
        self._freq_max.setValue(500.0)
        self._freq_max.setSuffix(" Hz")
        self._freq_max.setStyleSheet(self._spinbox_style())
        freq_layout.addRow("Max:", self._freq_max)
        
        self._freq_step = QDoubleSpinBox()
        self._freq_step.setRange(0.01, 100)
        self._freq_step.setValue(0.5)
        self._freq_step.setSuffix(" Hz")
        self._freq_step.setDecimals(2)
        self._freq_step.setStyleSheet(self._spinbox_style())
        freq_layout.addRow("Step:", self._freq_step)
        
        layout.addWidget(freq_group)
        
        # Damping group
        damping_group = QGroupBox("Damping")
        damping_group.setStyleSheet(self._group_style())
        damping_layout = QFormLayout(damping_group)
        
        self._damping = QDoubleSpinBox()
        self._damping.setRange(0.001, 0.5)
        self._damping.setValue(0.02)
        self._damping.setDecimals(3)
        self._damping.setSingleStep(0.005)
        self._damping.setStyleSheet(self._spinbox_style())
        damping_layout.addRow("ζ (ratio):", self._damping)
        
        # Show percentage
        self._damping_pct = QLabel("= 2.0%")
        self._damping_pct.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self._damping.valueChanged.connect(
            lambda v: self._damping_pct.setText(f"= {v*100:.1f}%")
        )
        damping_layout.addRow("", self._damping_pct)
        
        layout.addWidget(damping_group)
        
        # Compute button
        self._compute_btn = QPushButton("Compute FRF")
        self._compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1a4a7a;
            }
            QPushButton:disabled {
                background-color: #2a2a3a;
                color: #606060;
            }
        """)
        self._compute_btn.clicked.connect(self.compute_requested.emit)
        layout.addWidget(self._compute_btn)
        
        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        
        layout.addStretch()
    
    def _group_style(self) -> str:
        return """
            QGroupBox {
                color: #c0c0c0;
                border: 1px solid #4a4a6a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """
    
    def _input_style(self) -> str:
        return """
            QLineEdit {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus {
                border-color: #0f4c75;
            }
        """
    
    def _combo_style(self) -> str:
        return """
            QComboBox {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #16213e;
                color: #e0e0e0;
                selection-background-color: #0f3460;
            }
        """
    
    def _spinbox_style(self) -> str:
        return """
            QDoubleSpinBox {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
                padding: 4px;
            }
            QDoubleSpinBox:focus {
                border-color: #0f4c75;
            }
        """
    
    def set_valid_nodes(self, node_ids):
        """Set the valid node IDs for validation."""
        self._valid_node_ids = set(node_ids)
    
    def set_constrained_dofs(self, constrained: set):
        """Set the constrained DOFs for warnings."""
        self._constrained_dofs = constrained
    
    def set_input_node(self, node_id: int):
        """Set the input node ID (e.g., from 3D picker)."""
        self._input_node.setText(str(node_id))
    
    def set_response_node(self, node_id: int):
        """Set the response node ID (e.g., from 3D picker)."""
        self._response_node.setText(str(node_id))
    
    def get_input_dof(self) -> Optional[NodeDOF]:
        """Get the current input DOF selection."""
        try:
            node_id = int(self._input_node.text())
            component = self._input_dof.currentIndex() + 1
            return NodeDOF(node_id, component)
        except (ValueError, TypeError):
            return None
    
    def get_response_dof(self) -> Optional[NodeDOF]:
        """Get the current response DOF selection."""
        try:
            node_id = int(self._response_node.text())
            component = self._response_dof.currentIndex() + 1
            return NodeDOF(node_id, component)
        except (ValueError, TypeError):
            return None
    
    def get_freq_range(self) -> tuple:
        """Get frequency range parameters."""
        return (
            self._freq_min.value(),
            self._freq_max.value(),
            self._freq_step.value()
        )
    
    def get_damping(self) -> float:
        """Get damping ratio."""
        return self._damping.value()
    
    def _on_input_changed(self):
        """Handle input DOF change."""
        dof = self.get_input_dof()
        self._validate_and_update()
        self.input_dof_changed.emit(dof)
    
    def _on_response_changed(self):
        """Handle response DOF change."""
        dof = self.get_response_dof()
        self._validate_and_update()
        self.response_dof_changed.emit(dof)
    
    def _validate_and_update(self):
        """Validate selections and update status."""
        messages = []
        
        input_dof = self.get_input_dof()
        response_dof = self.get_response_dof()
        
        if input_dof:
            if self._valid_node_ids and input_dof.grid_id not in self._valid_node_ids:
                messages.append(f"Input node {input_dof.grid_id} not in model")
            elif (input_dof.grid_id, input_dof.component) in self._constrained_dofs:
                messages.append(f"Warning: Input DOF is constrained")
        
        if response_dof:
            if self._valid_node_ids and response_dof.grid_id not in self._valid_node_ids:
                messages.append(f"Response node {response_dof.grid_id} not in model")
            elif (response_dof.grid_id, response_dof.component) in self._constrained_dofs:
                messages.append(f"Warning: Response DOF is constrained")
        
        self._status.setText("\n".join(messages))
        
        can_compute = input_dof is not None and response_dof is not None
        self._compute_btn.setEnabled(can_compute)
    
    def set_error(self, message: str):
        """Display an error message."""
        self._status.setText(message)
        self._status.setStyleSheet("color: #ff6b6b; font-size: 11px;")
    
    def clear_error(self):
        """Clear the error message."""
        self._status.setText("")
