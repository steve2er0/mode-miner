"""Mode list widget for displaying and selecting modes."""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel
)
from PySide6.QtCore import Signal

from ..model.modal_model import ModalModel


class ModeListWidget(QWidget):
    """Widget displaying list of modes with frequencies.
    
    Signals:
        mode_selected: Emitted when a mode is selected, with 0-based index
    """
    
    mode_selected = Signal(int)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._modal_model: Optional[ModalModel] = None
        
        # Setup UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Header
        header = QLabel("Modes")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)
        
        # Mode list
        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
            }
            QListWidget::item:selected {
                background-color: #0f3460;
            }
            QListWidget::item:hover {
                background-color: #1a1a4e;
            }
        """)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)
    
    def set_modal_model(self, modal_model: ModalModel):
        """Set the modal model and populate the list.
        
        Args:
            modal_model: Modal analysis results
        """
        self._modal_model = modal_model
        self._populate_list()
    
    def _populate_list(self):
        """Populate the list with mode entries."""
        self._list.clear()
        
        if self._modal_model is None:
            return
        
        for i, freq in enumerate(self._modal_model.frequencies):
            mode_num = i + 1  # 1-based for display
            text = f"Mode {mode_num}: {freq:.2f} Hz"
            item = QListWidgetItem(text)
            item.setData(256, i)  # Store 0-based index
            self._list.addItem(item)
    
    def _on_selection_changed(self, row: int):
        """Handle list selection change.
        
        Args:
            row: Selected row index
        """
        if row >= 0:
            self.mode_selected.emit(row)
    
    def clear_selection(self):
        """Clear the current selection."""
        self._list.clearSelection()
    
    @property
    def selected_mode_index(self) -> Optional[int]:
        """Get currently selected mode index (0-based)."""
        row = self._list.currentRow()
        return row if row >= 0 else None

