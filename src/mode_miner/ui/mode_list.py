"""Mode list widget for displaying and selecting modes."""

from typing import Optional, List, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QLabel, QHeaderView, QPushButton, QHBoxLayout
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QColor

from ..model.modal_model import ModalModel


class ModeListWidget(QWidget):
    """Widget displaying list of modes with frequencies and contributions.
    
    Supports two modes:
    1. Full Mode List - shows all modes
    2. Peak-Filtered - shows only top-N contributing modes for a peak
    
    Signals:
        mode_selected: Emitted when a mode is selected, with 0-based index
    """
    
    mode_selected = Signal(int)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._modal_model: Optional[ModalModel] = None
        self._is_filtered = False
        self._filtered_modes: List[Tuple[int, float, float]] = []  # (index, freq, contribution%)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header_layout = QHBoxLayout()
        
        self._header = QLabel("Mode List")
        self._header.setFont(QFont("", -1, QFont.Bold))
        self._header.setStyleSheet("color: #e0e0e0; font-size: 13px;")
        header_layout.addWidget(self._header)
        
        header_layout.addStretch()
        
        # Show all button (hidden by default)
        self._show_all_btn = QPushButton("Show All")
        self._show_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1a4a7a;
            }
        """)
        self._show_all_btn.clicked.connect(self.show_all_modes)
        self._show_all_btn.hide()
        header_layout.addWidget(self._show_all_btn)
        
        layout.addLayout(header_layout)
        
        # Filter indicator
        self._filter_label = QLabel("")
        self._filter_label.setStyleSheet("color: #ffcc00; font-size: 11px;")
        self._filter_label.hide()
        layout.addWidget(self._filter_label)
        
        # Mode table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Mode", "Freq (Hz)", "Contrib %"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 4px;
                gridline-color: #3a3a5a;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #0f3460;
            }
            QTableWidget::item:hover {
                background-color: #1a1a4e;
            }
            QHeaderView::section {
                background-color: #0f3460;
                color: #e0e0e0;
                padding: 4px;
                border: none;
                border-bottom: 1px solid #4a4a6a;
            }
        """)
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table)
    
    def set_modal_model(self, modal_model: ModalModel):
        """Set the modal model and populate the list.
        
        Args:
            modal_model: Modal analysis results
        """
        self._modal_model = modal_model
        self._is_filtered = False
        self._populate_full_list()
    
    def _populate_full_list(self):
        """Populate table with all modes."""
        self._table.setRowCount(0)
        self._show_all_btn.hide()
        self._filter_label.hide()
        self._header.setText("Mode List")
        
        if self._modal_model is None:
            return
        
        # Hide contribution column for full list
        self._table.setColumnHidden(2, True)
        
        for i, freq in enumerate(self._modal_model.frequencies):
            row = self._table.rowCount()
            self._table.insertRow(row)
            
            # Mode number (1-based)
            mode_item = QTableWidgetItem(str(i + 1))
            mode_item.setData(Qt.UserRole, i)  # Store 0-based index
            mode_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, mode_item)
            
            # Frequency
            freq_item = QTableWidgetItem(f"{freq:.2f}")
            freq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, freq_item)
            
            # Empty contribution
            self._table.setItem(row, 2, QTableWidgetItem(""))
    
    def set_peak_filtered_modes(
        self, 
        peak_freq: float,
        contributions: List[Tuple[int, float]]  # List of (mode_index, contribution%)
    ):
        """Show only modes contributing to a specific peak.
        
        Args:
            peak_freq: The peak frequency
            contributions: List of (mode_index, contribution_percent) sorted by contribution
        """
        if self._modal_model is None:
            return
        
        self._is_filtered = True
        self._table.setRowCount(0)
        
        # Show contribution column
        self._table.setColumnHidden(2, False)
        
        # Update header
        self._header.setText("Contributing Modes")
        self._filter_label.setText(f"Peak @ {peak_freq:.2f} Hz")
        self._filter_label.show()
        self._show_all_btn.show()
        
        for mode_idx, contrib in contributions:
            freq = self._modal_model.frequencies[mode_idx]
            
            row = self._table.rowCount()
            self._table.insertRow(row)
            
            # Mode number
            mode_item = QTableWidgetItem(str(mode_idx + 1))
            mode_item.setData(Qt.UserRole, mode_idx)
            mode_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, mode_item)
            
            # Frequency
            freq_item = QTableWidgetItem(f"{freq:.2f}")
            freq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, freq_item)
            
            # Contribution
            contrib_item = QTableWidgetItem(f"{contrib:.1f}%")
            contrib_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # Color code by contribution
            if contrib > 50:
                contrib_item.setForeground(QColor("#ff6b6b"))
            elif contrib > 20:
                contrib_item.setForeground(QColor("#ffcc00"))
            else:
                contrib_item.setForeground(QColor("#a0a0a0"))
            
            self._table.setItem(row, 2, contrib_item)
    
    def show_all_modes(self):
        """Switch back to showing all modes."""
        self._is_filtered = False
        self._populate_full_list()
    
    def _on_cell_clicked(self, row: int, column: int):
        """Handle table cell click."""
        item = self._table.item(row, 0)
        if item:
            mode_idx = item.data(Qt.UserRole)
            if mode_idx is not None:
                self.mode_selected.emit(mode_idx)
    
    def select_mode(self, mode_index: int):
        """Programmatically select a mode.
        
        Args:
            mode_index: 0-based mode index
        """
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.data(Qt.UserRole) == mode_index:
                self._table.selectRow(row)
                break
    
    def clear_selection(self):
        """Clear the current selection."""
        self._table.clearSelection()
    
    def clear(self):
        """Clear all mode data."""
        self._modal_model = None
        self._is_filtered = False
        self._table.setRowCount(0)
        self._show_all_btn.hide()
        self._filter_label.hide()
        self._header.setText("Mode List")
    
    @property
    def is_filtered(self) -> bool:
        """Whether the list is currently peak-filtered."""
        return self._is_filtered
    
    @property
    def selected_mode_index(self) -> Optional[int]:
        """Get currently selected mode index (0-based)."""
        items = self._table.selectedItems()
        if items:
            return items[0].data(Qt.UserRole)
        return None
