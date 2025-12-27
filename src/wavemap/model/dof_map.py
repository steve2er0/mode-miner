"""DOF mapping utilities for modal analysis."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class NodeDOF:
    """Represents a single degree of freedom at a node.
    
    Attributes:
        grid_id: NASTRAN grid point ID
        component: DOF component (1-6: T1, T2, T3, R1, R2, R3)
    """
    grid_id: int
    component: int
    
    def __post_init__(self):
        if not 1 <= self.component <= 6:
            raise ValueError(f"Component must be 1-6, got {self.component}")
    
    def __repr__(self) -> str:
        return f"NodeDOF({self.grid_id}, {self.component})"


class DOFMap:
    """Maps between node DOFs and array indices.
    
    Provides bidirectional mapping between (grid_id, component) pairs
    and flat array indices used in eigenvector storage.
    """
    
    def __init__(self, node_ids: np.ndarray):
        """Initialize DOF map from array of node IDs.
        
        Args:
            node_ids: Array of grid point IDs in order
        """
        self._node_ids = np.asarray(node_ids)
        self._n_nodes = len(self._node_ids)
        
        # Build node_id -> index mapping
        self._node_to_idx: Dict[int, int] = {
            int(nid): idx for idx, nid in enumerate(self._node_ids)
        }
    
    @property
    def node_ids(self) -> np.ndarray:
        """Array of node IDs in order."""
        return self._node_ids
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self._n_nodes
    
    @property
    def n_dofs(self) -> int:
        """Total number of DOFs (6 per node)."""
        return self._n_nodes * 6
    
    def node_index(self, grid_id: int) -> int:
        """Get array index for a node ID.
        
        Args:
            grid_id: NASTRAN grid point ID
            
        Returns:
            Index into node arrays
            
        Raises:
            KeyError: If grid_id not in map
        """
        return self._node_to_idx[grid_id]
    
    def dof_index(self, dof: NodeDOF) -> int:
        """Get flat array index for a DOF.
        
        Args:
            dof: NodeDOF specifying grid and component
            
        Returns:
            Index into flattened DOF array
        """
        node_idx = self.node_index(dof.grid_id)
        return node_idx * 6 + (dof.component - 1)
    
    def dof_from_index(self, idx: int) -> NodeDOF:
        """Get NodeDOF from flat array index.
        
        Args:
            idx: Index into flattened DOF array
            
        Returns:
            NodeDOF for that index
        """
        node_idx = idx // 6
        component = (idx % 6) + 1
        return NodeDOF(int(self._node_ids[node_idx]), component)
    
    def get_translation_indices(self, grid_id: int) -> Tuple[int, int, int]:
        """Get indices for translation DOFs (T1, T2, T3) of a node.
        
        Args:
            grid_id: NASTRAN grid point ID
            
        Returns:
            Tuple of (T1_idx, T2_idx, T3_idx)
        """
        base = self.node_index(grid_id) * 6
        return (base, base + 1, base + 2)

