#!/usr/bin/env python3
"""
refine_shell_mesh.py - Nastran BDF Shell Mesh Refinement Tool

A CLI tool to refine (increase mesh density of) Nastran BDF files containing
2D shell elements (CQUAD4 and CTRIA3). The tool preserves all original GRID IDs
and element IDs, only adding new nodes and elements as needed.

USAGE:
    python refine_shell_mesh.py --in model.bdf --out model_refined.bdf --target 2.5 --max-passes 8

OPTIONS:
    --in, -i          Input BDF file path (required)
    --out, -o         Output BDF file path (required)
    --target, -t      Target maximum edge length (required)
    --max-passes, -m  Maximum refinement passes (default: 10)
    --pids, -p        Comma-separated list of property IDs to refine (optional)
    --eid-range, -e   Element ID range to refine, format: START:END (optional)
    --verbose, -v     Enable verbose logging

DEPENDENCIES:
    pip install pyNastran

NOTES:
    - All original GRID and element IDs are preserved unchanged
    - New GRIDs start at max(existing_grid_id) + 1
    - New elements start at max(existing_eid) + 1
    - Edge midpoints are shared between adjacent elements (conformal mesh)
    - Element property IDs (PID) and orientation fields are preserved
    - Element-based loads (PLOAD4, etc.) are NOT remapped in this version

Author: Mesh Refinement Tool v1.0
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from pyNastran.bdf.bdf import BDF
    from pyNastran.bdf.cards.elements.shell import CQUAD4, CTRIA3
    from pyNastran.bdf.cards.nodes import GRID
except ImportError:
    print("ERROR: pyNastran is required. Install with: pip install pyNastran")
    sys.exit(1)


# Type aliases for clarity
NodeId = int
ElementId = int
EdgeKey = Tuple[NodeId, NodeId]  # Sorted tuple of node IDs
Coord3D = np.ndarray  # Shape (3,)


@dataclass
class RefinementStats:
    """Statistics for a single refinement pass."""
    elements_split: int = 0
    nodes_added: int = 0
    elements_added: int = 0
    elements_removed: int = 0


@dataclass
class IdAllocator:
    """
    Efficient ID allocator that maintains counters for new GRID and element IDs.
    Avoids O(N) max() calls in loops.
    """
    next_grid_id: NodeId
    next_element_id: ElementId
    
    def allocate_grid_id(self) -> NodeId:
        """Allocate and return the next available GRID ID."""
        gid = self.next_grid_id
        self.next_grid_id += 1
        return gid
    
    def allocate_element_id(self) -> ElementId:
        """Allocate and return the next available element ID."""
        eid = self.next_element_id
        self.next_element_id += 1
        return eid


@dataclass
class EdgeCache:
    """
    Cache for edge midpoint nodes to ensure conformal mesh.
    Key: sorted tuple of (node_id_1, node_id_2)
    Value: midpoint node ID
    """
    _cache: Dict[EdgeKey, NodeId] = field(default_factory=dict)
    
    @staticmethod
    def make_edge_key(n1: NodeId, n2: NodeId) -> EdgeKey:
        """Create a canonical edge key (sorted tuple)."""
        return (min(n1, n2), max(n1, n2))
    
    def get_midpoint(self, n1: NodeId, n2: NodeId) -> Optional[NodeId]:
        """Get cached midpoint node ID for an edge, or None if not cached."""
        key = self.make_edge_key(n1, n2)
        return self._cache.get(key)
    
    def set_midpoint(self, n1: NodeId, n2: NodeId, mid_id: NodeId) -> None:
        """Cache the midpoint node ID for an edge."""
        key = self.make_edge_key(n1, n2)
        self._cache[key] = mid_id
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear the cache for a new pass."""
        self._cache.clear()


def get_grid_xyz(model: BDF, nid: NodeId) -> Coord3D:
    """
    Get the XYZ coordinates of a GRID node.
    
    Args:
        model: pyNastran BDF model
        nid: Node ID
        
    Returns:
        numpy array of shape (3,) with X, Y, Z coordinates
    """
    grid = model.nodes[nid]
    return np.array(grid.xyz, dtype=float)


def compute_edge_length(model: BDF, n1: NodeId, n2: NodeId) -> float:
    """
    Compute the Euclidean distance between two nodes.
    
    Args:
        model: pyNastran BDF model
        n1: First node ID
        n2: Second node ID
        
    Returns:
        Edge length as float
    """
    xyz1 = get_grid_xyz(model, n1)
    xyz2 = get_grid_xyz(model, n2)
    return float(np.linalg.norm(xyz2 - xyz1))


def compute_max_edge_length_quad(model: BDF, nodes: List[NodeId]) -> float:
    """
    Compute the maximum edge length of a CQUAD4.
    
    Args:
        model: pyNastran BDF model
        nodes: List of 4 node IDs [n1, n2, n3, n4]
        
    Returns:
        Maximum edge length
    """
    n1, n2, n3, n4 = nodes
    edges = [(n1, n2), (n2, n3), (n3, n4), (n4, n1)]
    return max(compute_edge_length(model, a, b) for a, b in edges)


def compute_max_edge_length_tri(model: BDF, nodes: List[NodeId]) -> float:
    """
    Compute the maximum edge length of a CTRIA3.
    
    Args:
        model: pyNastran BDF model
        nodes: List of 3 node IDs [n1, n2, n3]
        
    Returns:
        Maximum edge length
    """
    n1, n2, n3 = nodes
    edges = [(n1, n2), (n2, n3), (n3, n1)]
    return max(compute_edge_length(model, a, b) for a, b in edges)


def compute_midpoint(model: BDF, n1: NodeId, n2: NodeId) -> Coord3D:
    """
    Compute the midpoint coordinates between two nodes.
    
    Args:
        model: pyNastran BDF model
        n1: First node ID
        n2: Second node ID
        
    Returns:
        Midpoint coordinates as numpy array
    """
    xyz1 = get_grid_xyz(model, n1)
    xyz2 = get_grid_xyz(model, n2)
    return (xyz1 + xyz2) / 2.0


def compute_centroid_quad(model: BDF, nodes: List[NodeId]) -> Coord3D:
    """
    Compute the centroid of a CQUAD4.
    
    Args:
        model: pyNastran BDF model
        nodes: List of 4 node IDs
        
    Returns:
        Centroid coordinates as numpy array
    """
    coords = [get_grid_xyz(model, nid) for nid in nodes]
    return sum(coords) / 4.0


def get_or_create_midpoint_node(
    model: BDF,
    n1: NodeId,
    n2: NodeId,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    stats: RefinementStats
) -> NodeId:
    """
    Get an existing midpoint node from cache, or create a new one.
    
    This ensures conformal mesh by reusing midpoint nodes on shared edges.
    
    Args:
        model: pyNastran BDF model
        n1: First node ID of edge
        n2: Second node ID of edge
        edge_cache: Cache of edge midpoints
        id_alloc: ID allocator
        stats: Statistics tracker
        
    Returns:
        Node ID of the midpoint (existing or newly created)
    """
    # Check cache first
    cached_mid = edge_cache.get_midpoint(n1, n2)
    if cached_mid is not None:
        return cached_mid
    
    # Create new midpoint node
    new_nid = id_alloc.allocate_grid_id()
    xyz = compute_midpoint(model, n1, n2)
    
    # Get coordinate system from one of the parent nodes (typically 0)
    parent_grid = model.nodes[n1]
    cp = parent_grid.cp
    cd = parent_grid.cd
    ps = parent_grid.ps
    seid = parent_grid.seid
    
    # Add the new GRID card
    model.add_grid(new_nid, xyz, cp=cp, cd=cd, ps=ps, seid=seid)
    
    # Cache and track
    edge_cache.set_midpoint(n1, n2, new_nid)
    stats.nodes_added += 1
    
    return new_nid


def create_centroid_node(
    model: BDF,
    nodes: List[NodeId],
    id_alloc: IdAllocator,
    stats: RefinementStats
) -> NodeId:
    """
    Create a centroid node for a CQUAD4 element.
    
    Centroid nodes are NOT shared between elements.
    
    Args:
        model: pyNastran BDF model
        nodes: List of corner node IDs
        id_alloc: ID allocator
        stats: Statistics tracker
        
    Returns:
        Node ID of the newly created centroid node
    """
    new_nid = id_alloc.allocate_grid_id()
    xyz = compute_centroid_quad(model, nodes)
    
    # Get coordinate system from first corner node
    parent_grid = model.nodes[nodes[0]]
    cp = parent_grid.cp
    cd = parent_grid.cd
    ps = parent_grid.ps
    seid = parent_grid.seid
    
    model.add_grid(new_nid, xyz, cp=cp, cd=cd, ps=ps, seid=seid)
    stats.nodes_added += 1
    
    return new_nid


def split_cquad4(
    model: BDF,
    elem: CQUAD4,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    stats: RefinementStats,
    elements_to_remove: Set[ElementId],
    new_elements: List[dict]
) -> None:
    """
    Split a CQUAD4 into 4 child CQUAD4 elements.
    
    Split pattern:
        n4----n34----n3
        |      |      |
        |  Q4  |  Q3  |
        |      |      |
       n41----nc----n23
        |      |      |
        |  Q1  |  Q2  |
        |      |      |
        n1----n12----n2
    
    Child quads (counter-clockwise from n1):
        Q1: [n1,  n12, nc,  n41]
        Q2: [n12, n2,  n23, nc ]
        Q3: [nc,  n23, n3,  n34]
        Q4: [n41, nc,  n34, n4 ]
    
    Args:
        model: pyNastran BDF model
        elem: CQUAD4 element to split
        edge_cache: Edge midpoint cache
        id_alloc: ID allocator
        stats: Statistics tracker
        elements_to_remove: Set to add original element ID to
        new_elements: List to append new element definitions to
    """
    eid = elem.eid
    pid = elem.pid
    n1, n2, n3, n4 = elem.nodes
    
    # Get orientation fields
    theta_mcid = elem.theta_mcid
    zoffset = elem.zoffset
    tflag = elem.tflag
    t1 = elem.T1
    t2 = elem.T2
    t3 = elem.T3
    t4 = elem.T4
    
    # Get or create midpoint nodes (shared via edge cache)
    n12 = get_or_create_midpoint_node(model, n1, n2, edge_cache, id_alloc, stats)
    n23 = get_or_create_midpoint_node(model, n2, n3, edge_cache, id_alloc, stats)
    n34 = get_or_create_midpoint_node(model, n3, n4, edge_cache, id_alloc, stats)
    n41 = get_or_create_midpoint_node(model, n4, n1, edge_cache, id_alloc, stats)
    
    # Create centroid node (not shared)
    nc = create_centroid_node(model, [n1, n2, n3, n4], id_alloc, stats)
    
    # Define the 4 child quads
    child_quads = [
        [n1, n12, nc, n41],   # Q1
        [n12, n2, n23, nc],   # Q2
        [nc, n23, n3, n34],   # Q3
        [n41, nc, n34, n4],   # Q4
    ]
    
    # Mark original element for removal
    elements_to_remove.add(eid)
    stats.elements_split += 1
    
    # Queue new elements for creation
    for child_nodes in child_quads:
        new_eid = id_alloc.allocate_element_id()
        new_elements.append({
            'type': 'CQUAD4',
            'eid': new_eid,
            'pid': pid,
            'nodes': child_nodes,
            'theta_mcid': theta_mcid,
            'zoffset': zoffset,
            'tflag': tflag,
            'T1': t1,
            'T2': t2,
            'T3': t3,
            'T4': t4,
        })
        stats.elements_added += 1


def split_ctria3(
    model: BDF,
    elem: CTRIA3,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    stats: RefinementStats,
    elements_to_remove: Set[ElementId],
    new_elements: List[dict]
) -> None:
    """
    Split a CTRIA3 into 4 child CTRIA3 elements.
    
    Split pattern:
            n3
           /  \\
          /    \\
        n31----n23
        /  \\  /  \\
       /    \\/    \\
      n1----n12----n2
    
    Child tris:
        T1: [n1,  n12, n31]  (corner at n1)
        T2: [n12, n2,  n23]  (corner at n2)
        T3: [n31, n23, n3 ]  (corner at n3)
        T4: [n12, n23, n31]  (central tri)
    
    Args:
        model: pyNastran BDF model
        elem: CTRIA3 element to split
        edge_cache: Edge midpoint cache
        id_alloc: ID allocator
        stats: Statistics tracker
        elements_to_remove: Set to add original element ID to
        new_elements: List to append new element definitions to
    """
    eid = elem.eid
    pid = elem.pid
    n1, n2, n3 = elem.nodes
    
    # Get orientation fields
    theta_mcid = elem.theta_mcid
    zoffset = elem.zoffset
    tflag = elem.tflag
    t1 = elem.T1
    t2 = elem.T2
    t3 = elem.T3
    
    # Get or create midpoint nodes (shared via edge cache)
    n12 = get_or_create_midpoint_node(model, n1, n2, edge_cache, id_alloc, stats)
    n23 = get_or_create_midpoint_node(model, n2, n3, edge_cache, id_alloc, stats)
    n31 = get_or_create_midpoint_node(model, n3, n1, edge_cache, id_alloc, stats)
    
    # Define the 4 child tris
    child_tris = [
        [n1, n12, n31],   # T1 - corner at n1
        [n12, n2, n23],   # T2 - corner at n2
        [n31, n23, n3],   # T3 - corner at n3
        [n12, n23, n31],  # T4 - central tri
    ]
    
    # Mark original element for removal
    elements_to_remove.add(eid)
    stats.elements_split += 1
    
    # Queue new elements for creation
    for child_nodes in child_tris:
        new_eid = id_alloc.allocate_element_id()
        new_elements.append({
            'type': 'CTRIA3',
            'eid': new_eid,
            'pid': pid,
            'nodes': child_nodes,
            'theta_mcid': theta_mcid,
            'zoffset': zoffset,
            'tflag': tflag,
            'T1': t1,
            'T2': t2,
            'T3': t3,
        })
        stats.elements_added += 1


def should_refine_element(
    elem_id: ElementId,
    elem_pid: int,
    pid_filter: Optional[Set[int]],
    eid_range: Optional[Tuple[int, int]]
) -> bool:
    """
    Check if an element should be considered for refinement based on filters.
    
    Args:
        elem_id: Element ID
        elem_pid: Element's property ID
        pid_filter: Set of PIDs to include, or None for all
        eid_range: (min_eid, max_eid) range, or None for all
        
    Returns:
        True if element passes all filters
    """
    if pid_filter is not None and elem_pid not in pid_filter:
        return False
    if eid_range is not None:
        min_eid, max_eid = eid_range
        if not (min_eid <= elem_id <= max_eid):
            return False
    return True


def run_refinement_pass(
    model: BDF,
    target_edge_length: float,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    pid_filter: Optional[Set[int]] = None,
    eid_range: Optional[Tuple[int, int]] = None,
    logger: Optional[logging.Logger] = None
) -> RefinementStats:
    """
    Execute one refinement pass over the model.
    
    Args:
        model: pyNastran BDF model
        target_edge_length: Maximum allowed edge length
        edge_cache: Edge midpoint cache (cleared at start of pass)
        id_alloc: ID allocator
        pid_filter: Optional set of PIDs to refine
        eid_range: Optional (min, max) element ID range
        logger: Optional logger for verbose output
        
    Returns:
        Statistics for this pass
    """
    stats = RefinementStats()
    elements_to_remove: Set[ElementId] = set()
    new_elements: List[dict] = []
    
    # Clear edge cache for this pass (edges from previous passes are already nodes)
    edge_cache.clear()
    
    # Get current element IDs (copy to avoid modification during iteration)
    current_eids = list(model.elements.keys())
    
    for eid in current_eids:
        elem = model.elements[eid]
        
        # Handle CQUAD4
        if isinstance(elem, CQUAD4):
            if not should_refine_element(eid, elem.pid, pid_filter, eid_range):
                continue
            
            max_edge = compute_max_edge_length_quad(model, list(elem.nodes))
            if max_edge > target_edge_length:
                split_cquad4(model, elem, edge_cache, id_alloc, stats,
                           elements_to_remove, new_elements)
        
        # Handle CTRIA3
        elif isinstance(elem, CTRIA3):
            if not should_refine_element(eid, elem.pid, pid_filter, eid_range):
                continue
            
            max_edge = compute_max_edge_length_tri(model, list(elem.nodes))
            if max_edge > target_edge_length:
                split_ctria3(model, elem, edge_cache, id_alloc, stats,
                           elements_to_remove, new_elements)
    
    # Remove original elements
    for eid in elements_to_remove:
        del model.elements[eid]
    stats.elements_removed = len(elements_to_remove)
    
    # Add new elements
    for elem_def in new_elements:
        if elem_def['type'] == 'CQUAD4':
            model.add_cquad4(
                eid=elem_def['eid'],
                pid=elem_def['pid'],
                nids=elem_def['nodes'],
                theta_mcid=elem_def['theta_mcid'],
                zoffset=elem_def['zoffset'],
                tflag=elem_def['tflag'],
                T1=elem_def['T1'],
                T2=elem_def['T2'],
                T3=elem_def['T3'],
                T4=elem_def['T4'],
            )
        elif elem_def['type'] == 'CTRIA3':
            model.add_ctria3(
                eid=elem_def['eid'],
                pid=elem_def['pid'],
                nids=elem_def['nodes'],
                theta_mcid=elem_def['theta_mcid'],
                zoffset=elem_def['zoffset'],
                tflag=elem_def['tflag'],
                T1=elem_def['T1'],
                T2=elem_def['T2'],
                T3=elem_def['T3'],
            )
    
    return stats


def refine_mesh(
    input_file: str,
    output_file: str,
    target_edge_length: float,
    max_passes: int = 10,
    pids: Optional[List[int]] = None,
    eid_range: Optional[Tuple[int, int]] = None,
    verbose: bool = False
) -> Dict:
    """
    Main mesh refinement function.
    
    Args:
        input_file: Path to input BDF file
        output_file: Path to output BDF file
        target_edge_length: Target maximum edge length
        max_passes: Maximum number of refinement passes
        pids: Optional list of property IDs to refine
        eid_range: Optional (min, max) element ID range to refine
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with summary statistics
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Reading BDF: {input_file}")
    
    # Read BDF without cross-referencing
    model = BDF(debug=False)
    model.read_bdf(input_file, xref=False)
    
    # Initial counts
    initial_nodes = len(model.nodes)
    initial_elements = len(model.elements)
    
    # Count element types
    initial_quads = sum(1 for e in model.elements.values() if isinstance(e, CQUAD4))
    initial_tris = sum(1 for e in model.elements.values() if isinstance(e, CTRIA3))
    
    logger.info(f"Initial mesh: {initial_nodes} nodes, {initial_elements} elements "
                f"({initial_quads} CQUAD4, {initial_tris} CTRIA3)")
    logger.info(f"Target edge length: {target_edge_length}")
    logger.info(f"Max passes: {max_passes}")
    
    if pids:
        logger.info(f"Filtering PIDs: {pids}")
    if eid_range:
        logger.info(f"Filtering EID range: {eid_range[0]} to {eid_range[1]}")
    
    # Initialize ID allocators
    max_grid_id = max(model.nodes.keys()) if model.nodes else 0
    max_elem_id = max(model.elements.keys()) if model.elements else 0
    
    id_alloc = IdAllocator(
        next_grid_id=max_grid_id + 1,
        next_element_id=max_elem_id + 1
    )
    
    edge_cache = EdgeCache()
    pid_filter = set(pids) if pids else None
    
    # Track total statistics
    total_stats = {
        'passes': 0,
        'total_elements_split': 0,
        'total_nodes_added': 0,
        'total_elements_added': 0,
    }
    
    # Refinement loop
    for pass_num in range(1, max_passes + 1):
        logger.info(f"\n--- Pass {pass_num} ---")
        
        stats = run_refinement_pass(
            model=model,
            target_edge_length=target_edge_length,
            edge_cache=edge_cache,
            id_alloc=id_alloc,
            pid_filter=pid_filter,
            eid_range=eid_range,
            logger=logger
        )
        
        if stats.elements_split == 0:
            logger.info("No elements needed refinement. Stopping.")
            break
        
        logger.info(f"  Elements split: {stats.elements_split}")
        logger.info(f"  Nodes added: {stats.nodes_added}")
        logger.info(f"  Elements added: {stats.elements_added}")
        logger.info(f"  Elements removed: {stats.elements_removed}")
        
        total_stats['passes'] += 1
        total_stats['total_elements_split'] += stats.elements_split
        total_stats['total_nodes_added'] += stats.nodes_added
        total_stats['total_elements_added'] += stats.elements_added
    else:
        logger.warning(f"Reached maximum passes ({max_passes}). Some elements may still exceed target.")
    
    # Final counts
    final_nodes = len(model.nodes)
    final_elements = len(model.elements)
    final_quads = sum(1 for e in model.elements.values() if isinstance(e, CQUAD4))
    final_tris = sum(1 for e in model.elements.values() if isinstance(e, CTRIA3))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("REFINEMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Passes completed: {total_stats['passes']}")
    logger.info(f"Nodes: {initial_nodes} -> {final_nodes} (+{final_nodes - initial_nodes})")
    logger.info(f"Elements: {initial_elements} -> {final_elements}")
    logger.info(f"  CQUAD4: {initial_quads} -> {final_quads}")
    logger.info(f"  CTRIA3: {initial_tris} -> {final_tris}")
    
    # Sanity checks
    logger.info("\n--- Sanity Checks ---")
    
    # Check 1: Original nodes preserved
    # (We never delete nodes, only add, so this is guaranteed)
    logger.info("✓ Original GRID IDs preserved (by design)")
    
    # Check 2: Element count math
    # For quads: each split removes 1, adds 4 (net +3)
    # For tris: each split removes 1, adds 4 (net +3)
    expected_element_change = total_stats['total_elements_split'] * 3
    actual_element_change = final_elements - initial_elements
    if expected_element_change == actual_element_change:
        logger.info(f"✓ Element count change matches expected ({actual_element_change})")
    else:
        logger.warning(f"✗ Element count mismatch: expected +{expected_element_change}, got +{actual_element_change}")
    
    # Check 3: Verify no remaining violations (sample check)
    violations = 0
    for eid, elem in model.elements.items():
        if isinstance(elem, CQUAD4):
            if should_refine_element(eid, elem.pid, pid_filter, eid_range):
                max_edge = compute_max_edge_length_quad(model, list(elem.nodes))
                if max_edge > target_edge_length:
                    violations += 1
        elif isinstance(elem, CTRIA3):
            if should_refine_element(eid, elem.pid, pid_filter, eid_range):
                max_edge = compute_max_edge_length_tri(model, list(elem.nodes))
                if max_edge > target_edge_length:
                    violations += 1
    
    if violations == 0:
        logger.info("✓ All elements meet target edge length")
    else:
        logger.warning(f"✗ {violations} elements still exceed target edge length")
    
    # Write output
    logger.info(f"\nWriting refined BDF: {output_file}")
    
    # NOTE: Element-based loads (PLOAD4, PLOAD2, etc.) are NOT remapped.
    # To add load remapping, one would need to:
    # 1. Before refinement, collect all element-based loads
    # 2. Track parent->child element mappings
    # 3. After refinement, create new load cards for child elements
    # This is left as a future enhancement.
    
    model.write_bdf(output_file)
    logger.info("Done.")
    
    total_stats['initial_nodes'] = initial_nodes
    total_stats['final_nodes'] = final_nodes
    total_stats['initial_elements'] = initial_elements
    total_stats['final_elements'] = final_elements
    total_stats['violations'] = violations
    
    return total_stats


def parse_eid_range(s: str) -> Tuple[int, int]:
    """Parse an element ID range string like '1000:5000'."""
    parts = s.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid EID range format: {s}. Expected START:END")
    return (int(parts[0]), int(parts[1]))


def parse_pids(s: str) -> List[int]:
    """Parse a comma-separated list of PIDs."""
    return [int(p.strip()) for p in s.split(',')]


# =============================================================================
# UNIT TEST / VALIDATION
# =============================================================================

def create_synthetic_test_bdf() -> str:
    """
    Create a minimal synthetic BDF for testing.
    Returns the BDF content as a string.
    
    Creates a simple 2x1 mesh of two CQUAD4 elements:
    
        4-------5-------6
        |       |       |
        |  E1   |  E2   |
        |       |       |
        1-------2-------3
    
    With edge length = 10.0 units
    """
    bdf_content = """$ Synthetic test BDF for mesh refinement validation
$ Two CQUAD4 elements, edge length = 10.0
SOL 101
CEND
BEGIN BULK
$ Nodes
GRID           1       0      0.      0.      0.       0
GRID           2       0     10.      0.      0.       0
GRID           3       0     20.      0.      0.       0
GRID           4       0      0.     10.      0.       0
GRID           5       0     10.     10.      0.       0
GRID           6       0     20.     10.      0.       0
$ Elements
CQUAD4         1       1       1       2       5       4
CQUAD4         2       1       2       3       6       5
$ Property
PSHELL         1       1     0.1       1               1
$ Material
MAT1           1  1.0+7            0.3
ENDDATA
"""
    return bdf_content


def run_unit_test() -> bool:
    """
    Run a simple unit test to validate the refinement logic.
    
    Returns:
        True if test passes, False otherwise
    """
    import tempfile
    import os
    
    print("\n" + "="*60)
    print("UNIT TEST: Single refinement pass validation")
    print("="*60)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.bdf")
        output_path = os.path.join(tmpdir, "test_output.bdf")
        
        # Write synthetic BDF
        bdf_content = create_synthetic_test_bdf()
        with open(input_path, 'w') as f:
            f.write(bdf_content)
        
        print(f"Created synthetic BDF with 2 CQUAD4 elements (edge length=10.0)")
        print(f"Target edge length: 6.0 (should trigger one split pass)")
        
        # Run refinement
        try:
            stats = refine_mesh(
                input_file=input_path,
                output_file=output_path,
                target_edge_length=6.0,
                max_passes=1,
                verbose=False
            )
            
            # Validate results
            # Initial: 6 nodes, 2 elements
            # After 1 pass with target=6.0:
            #   - Both quads have max edge = 10.0 > 6.0, so both split
            #   - Each quad adds: 4 midside nodes (but shared edge has 1 shared) + 1 centroid
            #   - Shared edge between E1 and E2: nodes 2-5
            #   - Total new nodes: 4 (E1 midsides) + 3 (E2 midsides, minus shared) + 2 (centroids) = 9
            #   - Actually: edge 1-2, 2-5, 5-4, 4-1 for E1 = 4 midside
            #              edge 2-3, 3-6, 6-5, 5-2 for E2 = 4 midside, but 2-5 shared = 3 new
            #   - Total new nodes: 4 + 3 + 2 = 9
            # Final: 6 + 9 = 15 nodes
            # Each quad becomes 4 quads: 2*4 = 8 elements
            
            passed = True
            
            if stats['final_nodes'] != 15:
                print(f"✗ FAIL: Expected 15 nodes, got {stats['final_nodes']}")
                passed = False
            else:
                print(f"✓ PASS: Node count correct (15)")
            
            if stats['final_elements'] != 8:
                print(f"✗ FAIL: Expected 8 elements, got {stats['final_elements']}")
                passed = False
            else:
                print(f"✓ PASS: Element count correct (8)")
            
            if stats['passes'] != 1:
                print(f"✗ FAIL: Expected 1 pass, got {stats['passes']}")
                passed = False
            else:
                print(f"✓ PASS: Pass count correct (1)")
            
            # Verify output file exists and can be read
            model = BDF(debug=False)
            model.read_bdf(output_path, xref=False)
            print(f"✓ PASS: Output BDF readable by pyNastran")
            
            # Check that original node IDs 1-6 still exist
            original_nids = {1, 2, 3, 4, 5, 6}
            if original_nids.issubset(set(model.nodes.keys())):
                print(f"✓ PASS: Original node IDs preserved")
            else:
                print(f"✗ FAIL: Original node IDs not preserved")
                passed = False
            
            print("\n" + ("TEST PASSED" if passed else "TEST FAILED"))
            return passed
            
        except Exception as e:
            print(f"✗ FAIL: Exception during test: {e}")
            import traceback
            traceback.print_exc()
            return False


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Refine Nastran BDF shell mesh to target edge length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5
  python refine_shell_mesh.py -i model.bdf -o refined.bdf -t 2.5 -m 8 -p 10,12,15
  python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5 --eid-range 1000:5000
  python refine_shell_mesh.py --test  # Run unit test
        """
    )
    
    parser.add_argument('--in', '-i', dest='input_file', 
                        help='Input BDF file path')
    parser.add_argument('--out', '-o', dest='output_file',
                        help='Output BDF file path')
    parser.add_argument('--target', '-t', type=float,
                        help='Target maximum edge length')
    parser.add_argument('--max-passes', '-m', type=int, default=10,
                        help='Maximum refinement passes (default: 10)')
    parser.add_argument('--pids', '-p', type=str,
                        help='Comma-separated list of property IDs to refine')
    parser.add_argument('--eid-range', '-e', type=str,
                        help='Element ID range to refine (format: START:END)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--test', action='store_true',
                        help='Run unit test with synthetic BDF')
    
    args = parser.parse_args()
    
    # Run unit test if requested
    if args.test:
        success = run_unit_test()
        sys.exit(0 if success else 1)
    
    # Validate required arguments for normal operation
    if not args.input_file:
        parser.error("--in/-i is required (or use --test for unit test)")
    if not args.output_file:
        parser.error("--out/-o is required")
    if args.target is None:
        parser.error("--target/-t is required")
    
    # Parse optional filters
    pids = parse_pids(args.pids) if args.pids else None
    eid_range = parse_eid_range(args.eid_range) if args.eid_range else None
    
    # Run refinement
    try:
        refine_mesh(
            input_file=args.input_file,
            output_file=args.output_file,
            target_edge_length=args.target,
            max_passes=args.max_passes,
            pids=pids,
            eid_range=eid_range,
            verbose=args.verbose
        )
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
