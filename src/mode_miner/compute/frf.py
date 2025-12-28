"""FRF computation using modal superposition."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks

from ..model.modal_model import ModalModel
from ..model.dof_map import NodeDOF


@dataclass
class FRFResult:
    """Container for FRF computation results.
    
    Attributes:
        frequencies: Frequency array (Hz)
        magnitude: FRF magnitude array (v/F)
        phase: FRF phase array (degrees)
        peaks: List of (frequency, magnitude) peak locations
        input_dof: Input DOF used
        response_dof: Response DOF used
        damping: Modal damping ratio used
    """
    frequencies: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    peaks: List[Tuple[float, float]]
    input_dof: NodeDOF
    response_dof: NodeDOF
    damping: float


def compute_mobility_frf(
    modal_model: ModalModel,
    input_dof: NodeDOF,
    response_dof: NodeDOF,
    freq_min: float = 1.0,
    freq_max: float = 500.0,
    freq_step: float = 0.5,
    damping: float = 0.02
) -> FRFResult:
    """Compute SISO mobility FRF (v/F) using modal superposition.
    
    Computes:
        H_xf(ω) = Σ_r [ φ_i,r * φ_j,r / (m_r * (ω_r² - ω² + i*2*ζ*ω_r*ω)) ]
        H_vf(ω) = i*ω * H_xf(ω)  (mobility = velocity / force)
    
    Args:
        modal_model: Modal analysis results with eigenvectors
        input_dof: Force input DOF (node, component)
        response_dof: Response DOF (node, component)
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
        freq_step: Frequency step (Hz)
        damping: Modal damping ratio (0.02 = 2%)
        
    Returns:
        FRFResult containing frequencies, magnitude, phase, and peaks
        
    Raises:
        ValueError: If DOFs not found or normalization unknown
    """
    # Validate normalization
    if not modal_model.is_mass_normalized and modal_model.modal_mass is None:
        raise ValueError(
            "Cannot compute FRF: Modal normalization unknown. "
            "Modes must be mass-normalized or modal masses must be provided."
        )
    
    # Get DOF indices
    try:
        input_node_idx = modal_model.dof_map.node_index(input_dof.grid_id)
        response_node_idx = modal_model.dof_map.node_index(response_dof.grid_id)
    except KeyError as e:
        raise ValueError(f"DOF not found in model: {e}")
    
    # Component index (0-5 for T1,T2,T3,R1,R2,R3)
    input_comp = input_dof.component - 1
    response_comp = response_dof.component - 1
    
    # Extract mode shape coefficients at input and response DOFs
    # eigenvectors shape: (n_modes, n_nodes, 6)
    phi_input = modal_model.eigenvectors[:, input_node_idx, input_comp]  # (n_modes,)
    phi_response = modal_model.eigenvectors[:, response_node_idx, response_comp]  # (n_modes,)
    
    # Debug output
    print(f"[FRF] Input node idx: {input_node_idx}, comp: {input_comp}", flush=True)
    print(f"[FRF] Response node idx: {response_node_idx}, comp: {response_comp}", flush=True)
    print(f"[FRF] phi_input: {phi_input}", flush=True)
    print(f"[FRF] phi_response: {phi_response}", flush=True)
    
    # Natural frequencies (rad/s)
    omega_n = 2 * np.pi * modal_model.frequencies  # (n_modes,)
    
    # Modal masses (1.0 for mass-normalized modes)
    if modal_model.is_mass_normalized:
        modal_mass = np.ones(modal_model.n_modes)
    else:
        modal_mass = modal_model.modal_mass
    
    # Create frequency array
    frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
    omega = 2 * np.pi * frequencies  # rad/s
    
    # Compute receptance H_xf (displacement / force)
    # H_xf(ω) = Σ_r [ φ_i,r * φ_j,r / (m_r * (ω_r² - ω² + i*2*ζ*ω_r*ω)) ]
    H_xf = np.zeros(len(frequencies), dtype=complex)
    
    for r in range(modal_model.n_modes):
        omega_r = omega_n[r]
        m_r = modal_mass[r]
        phi_i = phi_input[r]
        phi_j = phi_response[r]
        
        # Denominator: m_r * (ω_r² - ω² + i*2*ζ*ω_r*ω)
        denom = m_r * (omega_r**2 - omega**2 + 1j * 2 * damping * omega_r * omega)
        
        # Avoid division by zero at very low frequencies
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        
        H_xf += (phi_i * phi_j) / denom
    
    # Compute mobility H_vf = i*ω * H_xf (velocity / force)
    H_vf = 1j * omega * H_xf
    
    # Magnitude and phase
    magnitude = np.abs(H_vf)
    phase = np.angle(H_vf, deg=True)
    
    # Detect peaks
    peaks = _detect_peaks(frequencies, magnitude)
    
    return FRFResult(
        frequencies=frequencies,
        magnitude=magnitude,
        phase=phase,
        peaks=peaks,
        input_dof=input_dof,
        response_dof=response_dof,
        damping=damping
    )


def _detect_peaks(
    frequencies: np.ndarray, 
    magnitude: np.ndarray,
    prominence_factor: float = 0.1
) -> List[Tuple[float, float]]:
    """Detect peaks in FRF magnitude.
    
    Args:
        frequencies: Frequency array
        magnitude: Magnitude array
        prominence_factor: Minimum prominence as fraction of max magnitude
        
    Returns:
        List of (frequency, magnitude) tuples at peaks
    """
    if len(magnitude) < 3:
        return []
    
    # Use log magnitude for better peak detection
    log_mag = np.log10(magnitude + 1e-30)
    
    # Find peaks with prominence
    min_prominence = prominence_factor * (np.max(log_mag) - np.min(log_mag))
    peak_indices, properties = find_peaks(log_mag, prominence=min_prominence)
    
    peaks = [(frequencies[i], magnitude[i]) for i in peak_indices]
    
    return peaks


def compute_mode_contributions(
    modal_model: ModalModel,
    input_dof: NodeDOF,
    response_dof: NodeDOF,
    target_freq: float,
    damping: float = 0.02,
    top_n: int = 5
) -> List[Tuple[int, float]]:
    """Compute per-mode contributions at a specific frequency.
    
    Args:
        modal_model: Modal analysis results
        input_dof: Force input DOF
        response_dof: Response DOF
        target_freq: Frequency at which to evaluate contributions (Hz)
        damping: Modal damping ratio
        top_n: Number of top contributing modes to return
        
    Returns:
        List of (mode_index, contribution_percent) sorted by contribution
    """
    try:
        input_node_idx = modal_model.dof_map.node_index(input_dof.grid_id)
        response_node_idx = modal_model.dof_map.node_index(response_dof.grid_id)
    except KeyError:
        return []
    
    input_comp = input_dof.component - 1
    response_comp = response_dof.component - 1
    
    phi_input = modal_model.eigenvectors[:, input_node_idx, input_comp]
    phi_response = modal_model.eigenvectors[:, response_node_idx, response_comp]
    
    omega_n = 2 * np.pi * modal_model.frequencies
    omega = 2 * np.pi * target_freq
    
    if modal_model.is_mass_normalized:
        modal_mass = np.ones(modal_model.n_modes)
    else:
        modal_mass = modal_model.modal_mass
    
    # Compute individual mode contributions
    contributions = []
    for r in range(modal_model.n_modes):
        omega_r = omega_n[r]
        m_r = modal_mass[r]
        phi_i = phi_input[r]
        phi_j = phi_response[r]
        
        denom = m_r * (omega_r**2 - omega**2 + 1j * 2 * damping * omega_r * omega)
        if np.abs(denom) < 1e-30:
            continue
        
        H_r = (phi_i * phi_j) / denom
        H_vf_r = 1j * omega * H_r
        
        contributions.append((r, np.abs(H_vf_r)))
    
    # Normalize to percentages
    total = sum(c[1] for c in contributions)
    if total > 0:
        contributions = [(idx, 100.0 * val / total) for idx, val in contributions]
    
    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    return contributions[:top_n]

