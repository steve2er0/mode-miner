"""Modal model data structure."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .dof_map import DOFMap


@dataclass
class ModalModel:
    """Container for modal analysis results.
    
    Attributes:
        frequencies: Array of natural frequencies in Hz, shape (n_modes,)
        eigenvectors: Mode shape matrix, shape (n_modes, n_nodes, 6)
                      Last axis is [T1, T2, T3, R1, R2, R3]
        dof_map: Mapping between node IDs and array indices
        modal_mass: Optional modal mass array, shape (n_modes,)
                    If None, eigenvectors are assumed mass-normalized
        is_mass_normalized: Whether eigenvectors are mass-normalized
    """
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    dof_map: DOFMap
    modal_mass: Optional[np.ndarray] = None
    is_mass_normalized: bool = True
    
    def __post_init__(self):
        """Validate array shapes."""
        n_modes = len(self.frequencies)
        if self.eigenvectors.shape[0] != n_modes:
            raise ValueError(
                f"Eigenvector count {self.eigenvectors.shape[0]} "
                f"doesn't match frequency count {n_modes}"
            )
        if self.eigenvectors.shape[1] != self.dof_map.n_nodes:
            raise ValueError(
                f"Eigenvector node count {self.eigenvectors.shape[1]} "
                f"doesn't match DOF map node count {self.dof_map.n_nodes}"
            )
        if self.eigenvectors.shape[2] != 6:
            raise ValueError(
                f"Eigenvector DOF count must be 6, got {self.eigenvectors.shape[2]}"
            )
        if self.modal_mass is not None and len(self.modal_mass) != n_modes:
            raise ValueError(
                f"Modal mass count {len(self.modal_mass)} "
                f"doesn't match mode count {n_modes}"
            )
    
    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return len(self.frequencies)
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.dof_map.n_nodes
    
    def get_mode_shape(self, mode_index: int) -> np.ndarray:
        """Get displacement shape for a mode.
        
        Args:
            mode_index: 0-based mode index
            
        Returns:
            Array of shape (n_nodes, 6) with displacements
        """
        return self.eigenvectors[mode_index]
    
    def get_translation_shape(self, mode_index: int) -> np.ndarray:
        """Get translation-only shape for a mode.
        
        Args:
            mode_index: 0-based mode index
            
        Returns:
            Array of shape (n_nodes, 3) with T1, T2, T3 displacements
        """
        return self.eigenvectors[mode_index, :, :3]

