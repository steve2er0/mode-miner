"""BDF file reader for mesh extraction."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
    """Container for BDF mesh data.
    
    Attributes:
        node_ids: Array of grid point IDs in order
        node_coords: Array of node coordinates, shape (n_nodes, 3)
        mesh: PyVista UnstructuredGrid for visualization
    """
    node_ids: np.ndarray
    node_coords: np.ndarray
    mesh: pv.UnstructuredGrid


def load_bdf_mesh(bdf_path: str) -> BDFData:
    """Load a BDF file and extract mesh data.
    
    Args:
        bdf_path: Path to BDF file
        
    Returns:
        BDFData containing node info and PyVista mesh
    """
    bdf = BDF()
    bdf.read_bdf(bdf_path, punch=False)
    
    # Extract nodes
    node_ids, node_coords = _extract_nodes(bdf)
    
    # Build node ID to index mapping
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    # Extract elements and build mesh
    mesh = _build_mesh(bdf, node_coords, node_id_to_idx)
    
    return BDFData(
        node_ids=node_ids,
        node_coords=node_coords,
        mesh=mesh
    )


def _extract_nodes(bdf: BDF) -> Tuple[np.ndarray, np.ndarray]:
    """Extract node IDs and coordinates from BDF.
    
    Args:
        bdf: Parsed BDF object
        
    Returns:
        Tuple of (node_ids, node_coords)
    """
    n_nodes = len(bdf.nodes)
    node_ids = np.zeros(n_nodes, dtype=np.int64)
    node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
    
    for i, (nid, node) in enumerate(sorted(bdf.nodes.items())):
        node_ids[i] = nid
        # Get position in basic coordinate system
        node_coords[i] = node.get_position()
    
    return node_ids, node_coords


def _build_mesh(
    bdf: BDF, 
    node_coords: np.ndarray,
    node_id_to_idx: Dict[int, int]
) -> pv.UnstructuredGrid:
    """Build PyVista mesh from BDF elements.
    
    Args:
        bdf: Parsed BDF object
        node_coords: Node coordinate array
        node_id_to_idx: Mapping from node ID to array index
        
    Returns:
        PyVista UnstructuredGrid
    """
    cells = []
    cell_types = []
    
    # Element type handlers
    element_handlers = {
        'CTRIA3': (VTK_TRIANGLE, lambda e: e.nodes),
        'CTRIA6': (VTK_TRIANGLE, lambda e: e.nodes[:3]),  # Use corner nodes
        'CQUAD4': (VTK_QUAD, lambda e: e.nodes),
        'CQUAD8': (VTK_QUAD, lambda e: e.nodes[:4]),  # Use corner nodes
        'CTETRA': (VTK_TETRA, lambda e: e.nodes[:4]),
        'CHEXA': (VTK_HEXAHEDRON, lambda e: e.nodes[:8]),
        'CPENTA': (VTK_WEDGE, lambda e: e.nodes[:6]),
        'CPYRAM': (VTK_PYRAMID, lambda e: e.nodes[:5]),
    }
    
    for eid, elem in bdf.elements.items():
        elem_type = elem.type
        
        if elem_type in element_handlers:
            vtk_type, get_nodes = element_handlers[elem_type]
            elem_nodes = get_nodes(elem)
            
            # Convert node IDs to indices
            try:
                indices = [node_id_to_idx[nid] for nid in elem_nodes]
                cells.append([len(indices)] + indices)
                cell_types.append(vtk_type)
            except KeyError:
                # Skip elements with missing nodes
                continue
    
    if not cells:
        # Return empty mesh if no elements
        return pv.UnstructuredGrid()
    
    # Flatten cells list
    cells_flat = []
    for cell in cells:
        cells_flat.extend(cell)
    
    return pv.UnstructuredGrid(
        np.array(cells_flat, dtype=np.int64),
        np.array(cell_types, dtype=np.uint8),
        node_coords
    )

