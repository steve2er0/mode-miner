"""BDF file reader for mesh extraction with element mappings."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pyvista as pv

from pyNastran.bdf.bdf import BDF


# PyVista cell type codes
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14


@dataclass
class BDFData:
    """Container for BDF mesh data with element mappings.
    
    Attributes:
        node_ids: Array of grid point IDs in order
        node_coords: Array of node coordinates, shape (n_nodes, 3)
        mesh: PyVista UnstructuredGrid for visualization
        element_id_to_cell_idx: Mapping from NASTRAN element ID to VTK cell index
        cell_idx_to_element_id: Mapping from VTK cell index to NASTRAN element ID
        property_to_elements: Mapping from property ID to set of element IDs
        material_to_elements: Mapping from material ID to set of element IDs
        property_to_material: Mapping from property ID to material ID(s)
    """
    node_ids: np.ndarray
    node_coords: np.ndarray
    mesh: pv.UnstructuredGrid
    element_id_to_cell_idx: Dict[int, int] = field(default_factory=dict)
    cell_idx_to_element_id: Dict[int, int] = field(default_factory=dict)
    property_to_elements: Dict[int, Set[int]] = field(default_factory=dict)
    material_to_elements: Dict[int, Set[int]] = field(default_factory=dict)
    property_to_material: Dict[int, Set[int]] = field(default_factory=dict)


def load_bdf_mesh(bdf_path: str) -> BDFData:
    """Load a BDF file and extract mesh data with mappings.
    
    Args:
        bdf_path: Path to BDF file
        
    Returns:
        BDFData containing node info, PyVista mesh, and element mappings
    """
    bdf = BDF()
    bdf.read_bdf(bdf_path, punch=False)
    
    # Extract nodes
    node_ids, node_coords = _extract_nodes(bdf)
    
    # Build node ID to index mapping
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    # Extract elements and build mesh with mappings
    mesh, elem_to_cell, cell_to_elem = _build_mesh(bdf, node_coords, node_id_to_idx)
    
    # Build property and material mappings
    prop_to_elem, mat_to_elem, prop_to_mat = _build_property_material_maps(bdf)
    
    return BDFData(
        node_ids=node_ids,
        node_coords=node_coords,
        mesh=mesh,
        element_id_to_cell_idx=elem_to_cell,
        cell_idx_to_element_id=cell_to_elem,
        property_to_elements=prop_to_elem,
        material_to_elements=mat_to_elem,
        property_to_material=prop_to_mat
    )


def _extract_nodes(bdf: BDF) -> Tuple[np.ndarray, np.ndarray]:
    """Extract node IDs and coordinates from BDF."""
    n_nodes = len(bdf.nodes)
    node_ids = np.zeros(n_nodes, dtype=np.int64)
    node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
    
    for i, (nid, node) in enumerate(sorted(bdf.nodes.items())):
        node_ids[i] = nid
        node_coords[i] = node.get_position()
    
    return node_ids, node_coords


def _build_mesh(
    bdf: BDF, 
    node_coords: np.ndarray,
    node_id_to_idx: Dict[int, int]
) -> Tuple[pv.UnstructuredGrid, Dict[int, int], Dict[int, int]]:
    """Build PyVista mesh from BDF elements with ID mappings."""
    cells = []
    cell_types = []
    elem_to_cell = {}  # element_id -> cell_idx
    cell_to_elem = {}  # cell_idx -> element_id
    
    # Element type handlers
    element_handlers = {
        'CTRIA3': (VTK_TRIANGLE, lambda e: e.nodes),
        'CTRIA6': (VTK_TRIANGLE, lambda e: e.nodes[:3]),
        'CQUAD4': (VTK_QUAD, lambda e: e.nodes),
        'CQUAD8': (VTK_QUAD, lambda e: e.nodes[:4]),
        'CTETRA': (VTK_TETRA, lambda e: e.nodes[:4]),
        'CHEXA': (VTK_HEXAHEDRON, lambda e: e.nodes[:8]),
        'CPENTA': (VTK_WEDGE, lambda e: e.nodes[:6]),
        'CPYRAM': (VTK_PYRAMID, lambda e: e.nodes[:5]),
    }
    
    cell_idx = 0
    for eid, elem in bdf.elements.items():
        elem_type = elem.type
        
        if elem_type in element_handlers:
            vtk_type, get_nodes = element_handlers[elem_type]
            elem_nodes = get_nodes(elem)
            
            try:
                indices = [node_id_to_idx[nid] for nid in elem_nodes]
                cells.append([len(indices)] + indices)
                cell_types.append(vtk_type)
                
                # Store mappings
                elem_to_cell[eid] = cell_idx
                cell_to_elem[cell_idx] = eid
                cell_idx += 1
                
            except KeyError:
                continue
    
    if not cells:
        return pv.UnstructuredGrid(), {}, {}
    
    cells_flat = []
    for cell in cells:
        cells_flat.extend(cell)
    
    mesh = pv.UnstructuredGrid(
        np.array(cells_flat, dtype=np.int64),
        np.array(cell_types, dtype=np.uint8),
        node_coords
    )
    
    return mesh, elem_to_cell, cell_to_elem


def _build_property_material_maps(bdf: BDF) -> Tuple[
    Dict[int, Set[int]], 
    Dict[int, Set[int]], 
    Dict[int, Set[int]]
]:
    """Build property->elements and material->elements mappings.
    
    Returns:
        Tuple of (property_to_elements, material_to_elements, property_to_material)
    """
    prop_to_elem: Dict[int, Set[int]] = {}
    prop_to_mat: Dict[int, Set[int]] = {}
    
    # First, map properties to materials
    for pid, prop in bdf.properties.items():
        prop_to_mat[pid] = set()
        
        # PSHELL has MID1, MID2, MID3, MID4
        if prop.type == 'PSHELL':
            for attr in ['mid1', 'mid2', 'mid3', 'mid4']:
                mid = getattr(prop, attr, None)
                if mid is not None and mid > 0:
                    prop_to_mat[pid].add(mid)
        
        # PSOLID has MID
        elif prop.type == 'PSOLID':
            if hasattr(prop, 'mid') and prop.mid:
                prop_to_mat[pid].add(prop.mid)
        
        # PBAR, PBEAM have MID
        elif prop.type in ('PBAR', 'PBEAM', 'PBARL', 'PBEAML'):
            if hasattr(prop, 'mid') and prop.mid:
                prop_to_mat[pid].add(prop.mid)
        
        # PCOMP has multiple MID layers
        elif prop.type == 'PCOMP':
            if hasattr(prop, 'mids'):
                for mid in prop.mids:
                    if mid is not None and mid > 0:
                        prop_to_mat[pid].add(mid)
    
    # Map elements to properties
    for eid, elem in bdf.elements.items():
        if hasattr(elem, 'pid') and elem.pid:
            pid = elem.pid
            if pid not in prop_to_elem:
                prop_to_elem[pid] = set()
            prop_to_elem[pid].add(eid)
    
    # Build material to elements mapping
    mat_to_elem: Dict[int, Set[int]] = {}
    for pid, mids in prop_to_mat.items():
        elem_ids = prop_to_elem.get(pid, set())
        for mid in mids:
            if mid not in mat_to_elem:
                mat_to_elem[mid] = set()
            mat_to_elem[mid].update(elem_ids)
    
    return prop_to_elem, mat_to_elem, prop_to_mat


def get_element_subset_mesh(
    bdf_data: BDFData, 
    element_ids: Set[int]
) -> Optional[pv.UnstructuredGrid]:
    """Extract a subset mesh containing only specified elements.
    
    Args:
        bdf_data: BDF data with mappings
        element_ids: Set of NASTRAN element IDs to extract
        
    Returns:
        PyVista mesh containing only the specified elements, or None if empty
    """
    if not element_ids:
        return None
    
    # Convert element IDs to cell indices
    cell_indices = []
    for eid in element_ids:
        if eid in bdf_data.element_id_to_cell_idx:
            cell_indices.append(bdf_data.element_id_to_cell_idx[eid])
    
    if not cell_indices:
        return None
    
    # Extract subset using PyVista
    return bdf_data.mesh.extract_cells(cell_indices)
