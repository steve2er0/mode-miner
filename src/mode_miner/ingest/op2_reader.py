"""OP2 file reader for modal results."""

from typing import Optional
import numpy as np

from pyNastran.op2.op2 import OP2

from ..model.modal_model import ModalModel
from ..model.dof_map import DOFMap


def load_op2_modes(
    op2_path: str,
    subcase: int = 1,
    node_ids: Optional[np.ndarray] = None
) -> ModalModel:
    """Load modal analysis results from OP2 file.
    
    Args:
        op2_path: Path to OP2 file
        subcase: Subcase ID to extract (default 1)
        node_ids: Optional array of node IDs to match ordering.
                  If None, uses order from OP2 file.
        
    Returns:
        ModalModel containing frequencies and eigenvectors
        
    Raises:
        ValueError: If no eigenvector results found
    """
    op2 = OP2()
    op2.read_op2(op2_path)
    
    # Get eigenvector results
    if subcase not in op2.eigenvectors:
        available = list(op2.eigenvectors.keys())
        raise ValueError(
            f"Subcase {subcase} not found in OP2. Available: {available}"
        )
    
    eigenvector = op2.eigenvectors[subcase]
    
    # Extract frequencies
    # eigenvector.eigns contains eigenvalues (omega^2)
    # eigenvector.freqs contains frequencies in Hz if available
    if hasattr(eigenvector, 'freqs') and eigenvector.freqs is not None:
        frequencies = np.array(eigenvector.freqs)
    else:
        # Compute from eigenvalues: f = sqrt(eigenvalue) / (2*pi)
        eigenvalues = np.array(eigenvector.eigns)
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    
    # Get node ordering from OP2
    op2_node_ids = eigenvector.node_gridtype[:, 0].astype(np.int64)
    
    # eigenvector.data has shape (n_modes, n_nodes, 6)
    # Components are [T1, T2, T3, R1, R2, R3]
    raw_data = eigenvector.data  # complex or real
    
    # Take real part if complex (should be real for SOL 103)
    if np.iscomplexobj(raw_data):
        raw_data = raw_data.real
    
    n_modes = raw_data.shape[0]
    n_nodes_op2 = raw_data.shape[1]
    
    if node_ids is not None:
        # Reorder to match provided node_ids
        eigenvectors = _reorder_eigenvectors(
            raw_data, op2_node_ids, node_ids
        )
        dof_map = DOFMap(node_ids)
    else:
        eigenvectors = raw_data
        dof_map = DOFMap(op2_node_ids)
    
    # Check for mass normalization
    # pyNastran doesn't directly expose this, assume mass-normalized
    # In practice, would need to check NASTRAN PARAM settings
    
    return ModalModel(
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        dof_map=dof_map,
        is_mass_normalized=True
    )


def _reorder_eigenvectors(
    data: np.ndarray,
    op2_node_ids: np.ndarray,
    target_node_ids: np.ndarray
) -> np.ndarray:
    """Reorder eigenvector data to match target node ordering.
    
    Args:
        data: Eigenvector data, shape (n_modes, n_nodes, 6)
        op2_node_ids: Node IDs in OP2 order
        target_node_ids: Desired node ID ordering
        
    Returns:
        Reordered eigenvector array
    """
    n_modes = data.shape[0]
    n_nodes = len(target_node_ids)
    
    # Build OP2 node ID to index mapping
    op2_id_to_idx = {nid: idx for idx, nid in enumerate(op2_node_ids)}
    
    # Create reordered array
    reordered = np.zeros((n_modes, n_nodes, 6), dtype=data.dtype)
    
    for i, nid in enumerate(target_node_ids):
        if nid in op2_id_to_idx:
            reordered[:, i, :] = data[:, op2_id_to_idx[nid], :]
        # Missing nodes get zero displacement
    
    return reordered

