"""Model Tree panel for BDF structure navigation with element highlighting."""

from typing import Optional, Dict, Set, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
    QLabel, QPushButton, QHBoxLayout, QAbstractItemView
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from ..ingest.bdf_reader import BDFData


class ModelTreeWidget(QWidget):
    """Tree view showing model structure with material/property drill-down.
    
    Signals:
        grid_selected: Emitted when grid IDs are selected (list of grid IDs)
        element_selected: Emitted when element IDs are selected (list of element IDs)
        elements_highlight_requested: Emitted with set of element IDs to highlight
        clear_highlight_requested: Emitted when highlight should be cleared
    """
    
    grid_selected = Signal(object)  # List[int] of grid IDs
    element_selected = Signal(object)  # List[int] of element IDs
    elements_highlight_requested = Signal(object)  # Set[int] of element IDs
    clear_highlight_requested = Signal()
    
    # Max items to show per category before truncating
    MAX_ITEMS_DISPLAY = 100
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._bdf_data: Optional[BDFData] = None
        self._bdf_raw = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header with clear button
        header_layout = QHBoxLayout()
        
        header = QLabel("Model Tree")
        header.setFont(QFont("", -1, QFont.Bold))
        header.setStyleSheet("color: #e0e0e0; font-size: 13px; padding: 4px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        self._clear_btn = QPushButton("Clear Selection")
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1a4a7a;
            }
        """)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        self._clear_btn.hide()
        header_layout.addWidget(self._clear_btn)
        
        layout.addLayout(header_layout)
        
        # Tree widget with multi-select support
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #4a4a6a;
                border-radius: 4px;
            }
            QTreeWidget::item {
                padding: 2px 4px;
            }
            QTreeWidget::item:selected {
                background-color: #0f3460;
            }
            QTreeWidget::item:hover {
                background-color: #1a1a4e;
            }
            QTreeWidget::branch {
                background-color: #16213e;
            }
        """)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)
    
    def set_bdf_data(self, bdf_data: BDFData, bdf_raw=None):
        """Populate tree from BDF data.
        
        Args:
            bdf_data: Parsed BDF data with mappings
            bdf_raw: Raw pyNastran BDF object for metadata
        """
        self._bdf_data = bdf_data
        self._bdf_raw = bdf_raw
        self._populate_tree()
    
    def _populate_tree(self):
        """Populate the tree with model structure."""
        self._tree.clear()
        
        if self._bdf_data is None:
            return
        
        # === MATERIALS ===
        if self._bdf_raw and hasattr(self._bdf_raw, 'materials') and self._bdf_raw.materials:
            materials_item = QTreeWidgetItem([f"Materials ({len(self._bdf_raw.materials)})"])
            materials_item.setData(0, 256, "materials_root")
            self._tree.addTopLevelItem(materials_item)
            
            for mid, mat in sorted(self._bdf_raw.materials.items()):
                elem_count = len(self._bdf_data.material_to_elements.get(mid, set()))
                child = QTreeWidgetItem([f"{mat.type} {mid} ({elem_count} elems)"])
                child.setData(0, 256, "material")
                child.setData(0, 257, mid)
                materials_item.addChild(child)
            
            materials_item.setExpanded(True)
        
        # === PROPERTIES ===
        if self._bdf_raw and hasattr(self._bdf_raw, 'properties') and self._bdf_raw.properties:
            props_item = QTreeWidgetItem([f"Properties ({len(self._bdf_raw.properties)})"])
            props_item.setData(0, 256, "properties_root")
            self._tree.addTopLevelItem(props_item)
            
            for pid, prop in sorted(self._bdf_raw.properties.items()):
                elem_count = len(self._bdf_data.property_to_elements.get(pid, set()))
                mids = self._bdf_data.property_to_material.get(pid, set())
                mid_str = f" -> MID {','.join(map(str, sorted(mids)))}" if mids else ""
                
                child = QTreeWidgetItem([f"{prop.type} {pid} ({elem_count} elems){mid_str}"])
                child.setData(0, 256, "property")
                child.setData(0, 257, pid)
                props_item.addChild(child)
            
            props_item.setExpanded(True)
        
        # === GRIDS ===
        n_grids = len(self._bdf_data.node_ids)
        grids_item = QTreeWidgetItem([f"Grids ({n_grids})"])
        grids_item.setData(0, 256, "grids_root")
        self._tree.addTopLevelItem(grids_item)
        
        # Add grids (limited for performance)
        for i, nid in enumerate(self._bdf_data.node_ids):
            if i >= self.MAX_ITEMS_DISPLAY:
                more = QTreeWidgetItem([f"... and {n_grids - self.MAX_ITEMS_DISPLAY} more"])
                more.setData(0, 256, "more_grids")
                grids_item.addChild(more)
                break
            child = QTreeWidgetItem([f"Grid {nid}"])
            child.setData(0, 256, "grid")
            child.setData(0, 257, int(nid))
            grids_item.addChild(child)
        
        # === ELEMENTS ===
        n_elements = self._bdf_data.mesh.n_cells
        elements_item = QTreeWidgetItem([f"Elements ({n_elements})"])
        elements_item.setData(0, 256, "elements_root")
        self._tree.addTopLevelItem(elements_item)
        
        # Group elements by type with individual IDs
        if self._bdf_raw:
            elem_by_type: Dict[str, List[int]] = {}
            for eid, elem in self._bdf_raw.elements.items():
                if elem.type not in elem_by_type:
                    elem_by_type[elem.type] = []
                elem_by_type[elem.type].append(eid)
            
            for etype in sorted(elem_by_type.keys()):
                eids = sorted(elem_by_type[etype])
                type_item = QTreeWidgetItem([f"{etype} ({len(eids)})"])
                type_item.setData(0, 256, "element_type")
                type_item.setData(0, 257, etype)
                elements_item.addChild(type_item)
                
                # Add individual element IDs (limited for performance)
                for i, eid in enumerate(eids):
                    if i >= self.MAX_ITEMS_DISPLAY:
                        more = QTreeWidgetItem([f"... and {len(eids) - self.MAX_ITEMS_DISPLAY} more"])
                        more.setData(0, 256, "more_elements")
                        type_item.addChild(more)
                        break
                    child = QTreeWidgetItem([f"EID {eid}"])
                    child.setData(0, 256, "element")
                    child.setData(0, 257, eid)
                    type_item.addChild(child)
        
        # === COORDINATE SYSTEMS ===
        if self._bdf_raw and hasattr(self._bdf_raw, 'coords') and len(self._bdf_raw.coords) > 1:
            coords_item = QTreeWidgetItem([f"Coord Systems ({len(self._bdf_raw.coords)})"])
            coords_item.setData(0, 256, "coords_root")
            self._tree.addTopLevelItem(coords_item)
            
            for cid, coord in sorted(self._bdf_raw.coords.items()):
                child = QTreeWidgetItem([f"{coord.type} {cid}"])
                coords_item.addChild(child)
        
        # === CONSTRAINTS ===
        n_spcs = 0
        if self._bdf_raw:
            n_spcs = len(getattr(self._bdf_raw, 'spcs', {})) + len(getattr(self._bdf_raw, 'spc1s', {}))
        if n_spcs > 0:
            spcs_item = QTreeWidgetItem([f"Constraints ({n_spcs})"])
            spcs_item.setData(0, 256, "constraints_root")
            self._tree.addTopLevelItem(spcs_item)
    
    def _on_selection_changed(self):
        """Handle tree selection change (supports multi-select)."""
        selected_items = self._tree.selectedItems()
        
        if not selected_items:
            return
        
        # Collect selected items by type
        selected_grids: List[int] = []
        selected_elements: List[int] = []
        highlight_elements: Set[int] = set()
        
        for item in selected_items:
            item_type = item.data(0, 256)
            item_id = item.data(0, 257)
            
            if item_type == "grid" and item_id is not None:
                selected_grids.append(item_id)
            
            elif item_type == "element" and item_id is not None:
                selected_elements.append(item_id)
                highlight_elements.add(item_id)
            
            elif item_type == "element_type" and item_id is not None:
                # Select all elements of this type
                if self._bdf_raw:
                    for eid, elem in self._bdf_raw.elements.items():
                        if elem.type == item_id:
                            highlight_elements.add(eid)
            
            elif item_type == "material" and item_id is not None:
                if self._bdf_data:
                    elem_ids = self._bdf_data.material_to_elements.get(item_id, set())
                    highlight_elements.update(elem_ids)
            
            elif item_type == "property" and item_id is not None:
                if self._bdf_data:
                    elem_ids = self._bdf_data.property_to_elements.get(item_id, set())
                    highlight_elements.update(elem_ids)
        
        # Emit signals based on what was selected
        if selected_grids:
            self._clear_btn.show()
            self.grid_selected.emit(selected_grids)
        
        if selected_elements:
            self._clear_btn.show()
            self.element_selected.emit(selected_elements)
        
        if highlight_elements:
            self._clear_btn.show()
            self.elements_highlight_requested.emit(highlight_elements)
    
    def _on_clear_clicked(self):
        """Handle clear button click."""
        self._tree.clearSelection()
        self._clear_btn.hide()
        self.clear_highlight_requested.emit()
    
    def clear(self):
        """Clear the tree."""
        self._tree.clear()
        self._bdf_data = None
        self._bdf_raw = None
        self._clear_btn.hide()
