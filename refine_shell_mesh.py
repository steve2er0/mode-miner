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

import argparse
import logging
import math
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from pyNastran.bdf.bdf import BDF
    from pyNastran.bdf.cards.elements.shell import CQUAD4, CTRIA3
    from pyNastran.bdf.cards.elements.bars import CBAR
    from pyNastran.bdf.cards.elements.beam import CBEAM
    from pyNastran.bdf.cards.elements.rods import CROD
    from pyNastran.bdf.cards.nodes import GRID
except ImportError:
    print("ERROR: pyNastran is required. Install with: pip install pyNastran")
    sys.exit(1)


# Type aliases for clarity
NodeId = int
ElementId = int
EdgeKey = Tuple[NodeId, NodeId]  # Sorted tuple of node IDs
Coord3D = np.ndarray  # Shape (3,)


class RefinementStats:
    """Statistics for a single refinement pass."""
    
    def __init__(self):
        # type: () -> None
        self.elements_split = 0
        self.nodes_added = 0
        self.elements_added = 0
        self.elements_removed = 0


class IdAllocator:
    """
    Efficient ID allocator that maintains counters for new GRID and element IDs.
    Avoids O(N) max() calls in loops.
    """
    
    def __init__(self, next_grid_id, next_element_id):
        # type: (NodeId, ElementId) -> None
        self.next_grid_id = next_grid_id
        self.next_element_id = next_element_id
    
    def allocate_grid_id(self):
        # type: () -> NodeId
        """Allocate and return the next available GRID ID."""
        gid = self.next_grid_id
        self.next_grid_id += 1
        return gid
    
    def allocate_element_id(self):
        # type: () -> ElementId
        """Allocate and return the next available element ID."""
        eid = self.next_element_id
        self.next_element_id += 1
        return eid


class EdgeCache:
    """
    Cache for edge midpoint nodes to ensure conformal mesh.
    Key: sorted tuple of (node_id_1, node_id_2)
    Value: midpoint node ID
    """
    
    def __init__(self):
        # type: () -> None
        self._cache = {}  # type: Dict[EdgeKey, NodeId]
    
    @staticmethod
    def make_edge_key(n1, n2):
        # type: (NodeId, NodeId) -> EdgeKey
        """Create a canonical edge key (sorted tuple)."""
        return (min(n1, n2), max(n1, n2))
    
    def get_midpoint(self, n1, n2):
        # type: (NodeId, NodeId) -> Optional[NodeId]
        """Get cached midpoint node ID for an edge, or None if not cached."""
        key = self.make_edge_key(n1, n2)
        return self._cache.get(key)
    
    def set_midpoint(self, n1, n2, mid_id):
        # type: (NodeId, NodeId, NodeId) -> None
        """Cache the midpoint node ID for an edge."""
        key = self.make_edge_key(n1, n2)
        self._cache[key] = mid_id
    
    def __len__(self):
        # type: () -> int
        return len(self._cache)
    
    def clear(self):
        # type: () -> None
        """Clear the cache for a new pass."""
        self._cache.clear()


def get_grid_xyz(model: BDF, nid: NodeId) -> Coord3D:
    """
    Get the XYZ coordinates of a GRID node in global coordinate system.
    
    Args:
        model: pyNastran BDF model (must be cross-referenced)
        nid: Node ID
        
    Returns:
        numpy array of shape (3,) with X, Y, Z coordinates in global CS
    """
    grid = model.nodes[nid]
    # Use get_position() to get global coordinates (handles CP transformation)
    # This requires the model to be cross-referenced (xref=True)
    try:
        return grid.get_position()
    except AttributeError:
        # Fallback for non-cross-referenced models
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
    
    # Inherit the analysis coordinate system (CD) from the parent nodes
    # If both parents have the same CD, use that; otherwise use CD=0
    grid1 = model.nodes[n1]
    grid2 = model.nodes[n2]
    cd1 = getattr(grid1, 'cd', 0) if hasattr(grid1, 'cd') else 0
    cd2 = getattr(grid2, 'cd', 0) if hasattr(grid2, 'cd') else 0
    cd = cd1 if cd1 == cd2 else 0
    
    # Create new node with inherited CD (analysis coordinate system)
    # CP=0 since xyz is computed in global coordinates via get_position()
    model.add_grid(new_nid, xyz, cp=0, cd=cd, ps=0, seid=0)
    
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
    
    # Create new node in basic/global coordinate system (CP=0, CD=0)
    model.add_grid(new_nid, xyz, cp=0, cd=0, ps=0, seid=0)
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


def compute_bar_length(model: BDF, nodes: List[NodeId]) -> float:
    """
    Compute the length of a bar/beam element.
    
    Args:
        model: pyNastran BDF model
        nodes: List of 2 node IDs [GA, GB]
        
    Returns:
        Element length
    """
    return compute_edge_length(model, nodes[0], nodes[1])


def split_cbar(
    model: BDF,
    elem: CBAR,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    stats: RefinementStats,
    elements_to_remove: Set[ElementId],
    new_elements: List[dict]
) -> None:
    """
    Split a CBAR into 2 child CBAR elements.
    
    Split pattern:
        GA -------- GB    becomes    GA ---- NM ---- GB
        
    Child bars:
        B1: [GA, NM]  (first half)
        B2: [NM, GB]  (second half)
    
    Args:
        model: pyNastran BDF model
        elem: CBAR element to split
        edge_cache: Edge midpoint cache
        id_alloc: ID allocator
        stats: Statistics tracker
        elements_to_remove: Set to add original element ID to
        new_elements: List to append new element definitions to
    """
    eid = elem.eid
    pid = elem.pid
    ga, gb = elem.nodes[:2]  # CBAR nodes are [GA, GB]
    
    # Get orientation - could be G0 (node ID) or X vector
    g0 = elem.g0
    # Ensure X vector is copied correctly as [x1, x2, x3]
    # pyNastran may return this as numpy array or in different formats
    x = None
    if elem.x is not None:
        try:
            x_raw = elem.x
            if hasattr(x_raw, 'tolist'):
                x = x_raw.tolist()  # numpy array -> list
            elif isinstance(x_raw, (list, tuple)):
                x = list(x_raw)
            else:
                x = [float(x_raw[0]), float(x_raw[1]), float(x_raw[2])]
        except (IndexError, TypeError):
            x = None
    
    # Log parent orientation for debugging
    import logging
    logger = logging.getLogger(__name__)
    if g0 is not None:
        logger.info(f"CBAR {eid}: Splitting - parent uses G0 orientation (g0={g0}), x={x}")
    else:
        logger.info(f"CBAR {eid}: Splitting - parent uses X vector orientation: x={x} (X1={x[0] if x else None}, X2={x[1] if x else None}, X3={x[2] if x else None})")
    
    # Get optional fields
    offt = getattr(elem, 'offt', 'GGG')
    pa = getattr(elem, 'pa', 0)
    pb = getattr(elem, 'pb', 0)
    
    # Extract offset vectors WA and WB, converting to list format
    wa = None
    wb = None
    wa_raw = getattr(elem, 'wa', None)
    wb_raw = getattr(elem, 'wb', None)
    if wa_raw is not None:
        try:
            if hasattr(wa_raw, 'tolist'):
                wa = wa_raw.tolist()
            elif isinstance(wa_raw, (list, tuple)):
                wa = list(wa_raw)
            else:
                wa = [float(wa_raw[0]), float(wa_raw[1]), float(wa_raw[2])]
        except (IndexError, TypeError):
            wa = None
    if wb_raw is not None:
        try:
            if hasattr(wb_raw, 'tolist'):
                wb = wb_raw.tolist()
            elif isinstance(wb_raw, (list, tuple)):
                wb = list(wb_raw)
            else:
                wb = [float(wb_raw[0]), float(wb_raw[1]), float(wb_raw[2])]
        except (IndexError, TypeError):
            wb = None
    
    logger.info(f"CBAR {eid}: Parent offsets wa={wa}, wb={wb}")
    
    # Get or create midpoint node (shared via edge cache)
    nm = get_or_create_midpoint_node(model, ga, gb, edge_cache, id_alloc, stats)
    
    # Validate orientation - G0 cannot be one of the child element's nodes
    # If G0 is the midpoint or one of the original nodes, we have a problem
    if g0 is not None and g0 in (ga, gb, nm):
        # G0 conflicts with element nodes - convert to X vector instead
        # Calculate the orientation vector from parent element
        logger.warning(f"CBAR {eid}: G0={g0} conflicts with element nodes. "
                      f"Converting to X vector orientation.")
        try:
            # Get positions of the nodes to compute a perpendicular vector
            ga_pos = get_grid_xyz(model, ga)
            gb_pos = get_grid_xyz(model, gb)
            g0_pos = get_grid_xyz(model, g0)
            # Original orientation vector from element axis to G0
            axis = gb_pos - ga_pos
            axis = axis / np.linalg.norm(axis)
            to_g0 = g0_pos - ga_pos
            # Project out the axial component to get perpendicular direction
            x_vec = to_g0 - np.dot(to_g0, axis) * axis
            if np.linalg.norm(x_vec) > 1e-10:
                x = list(x_vec / np.linalg.norm(x_vec))
            else:
                x = [0.0, 0.0, 1.0]  # Default if G0 is on element axis
            g0 = None
        except Exception:
            x = [0.0, 0.0, 1.0]  # Fallback default orientation
            g0 = None
    
    # Mark original element for removal
    elements_to_remove.add(eid)
    stats.elements_split += 1
    
    # Log final orientation being used for children
    if g0 is not None:
        logger.info(f"CBAR {eid}: Children will use G0 orientation (g0={g0})")
    else:
        logger.info(f"CBAR {eid}: Children will use X vector orientation: x={x}")
    
    # Use [0,0,0] for no offset (not None) to ensure pyNastran writes the values correctly
    zero_offset = [0.0, 0.0, 0.0]
    wa_effective = wa if wa is not None else zero_offset
    wb_effective = wb if wb is not None else zero_offset
    
    # Compute the midpoint offset using global coordinate transformation
    # This matches FEMAP's behavior: interpolate the PHYSICAL offset position
    # in global coordinates, then transform to local coordinates for each child
    
    # Get node positions
    ga_pos = get_grid_xyz(model, ga)
    gb_pos = get_grid_xyz(model, gb)
    nm_pos = get_grid_xyz(model, nm)
    
    # Compute element local coordinate systems
    # For parent: axis from GA to GB
    parent_axis = gb_pos - ga_pos
    parent_axis = parent_axis / np.linalg.norm(parent_axis)
    
    # Get orientation vector (X vector or direction to G0)
    if g0 is not None:
        g0_pos = get_grid_xyz(model, g0)
        orient_vec = g0_pos - ga_pos
    elif x is not None:
        orient_vec = np.array(x)
    else:
        orient_vec = np.array([0.0, 0.0, 1.0])
    
    # Compute parent local coordinate system (y and z axes)
    # y-axis: perpendicular to beam axis, in plane with orientation vector
    parent_y = orient_vec - np.dot(orient_vec, parent_axis) * parent_axis
    if np.linalg.norm(parent_y) > 1e-10:
        parent_y = parent_y / np.linalg.norm(parent_y)
    else:
        # Orientation vector is parallel to beam - pick arbitrary perpendicular
        if abs(parent_axis[0]) < 0.9:
            parent_y = np.cross(parent_axis, np.array([1.0, 0.0, 0.0]))
        else:
            parent_y = np.cross(parent_axis, np.array([0.0, 1.0, 0.0]))
        parent_y = parent_y / np.linalg.norm(parent_y)
    
    # z-axis: perpendicular to both x and y
    parent_z = np.cross(parent_axis, parent_y)
    
    # Transform WA and WB to global coordinates (offset from node position)
    # W = [w1, w2, w3] where w1 is y-offset, w2 is z-offset, w3 is x-offset
    wa_global = wa_effective[0] * parent_y + wa_effective[1] * parent_z + wa_effective[2] * parent_axis
    wb_global = wb_effective[0] * parent_y + wb_effective[1] * parent_z + wb_effective[2] * parent_axis
    
    # The PHYSICAL offset positions in global coordinates
    pa_offset = ga_pos + wa_global
    pb_offset = gb_pos + wb_global
    
    # Interpolate physical offset position at midpoint
    pm_offset = (pa_offset + pb_offset) / 2.0
    
    # Now compute local coordinate systems for child elements
    # Child 1: GA to NM
    child1_axis = nm_pos - ga_pos
    child1_axis = child1_axis / np.linalg.norm(child1_axis)
    child1_y = orient_vec - np.dot(orient_vec, child1_axis) * child1_axis
    if np.linalg.norm(child1_y) > 1e-10:
        child1_y = child1_y / np.linalg.norm(child1_y)
    else:
        child1_y = parent_y
    child1_z = np.cross(child1_axis, child1_y)
    
    # Child 2: NM to GB  
    child2_axis = gb_pos - nm_pos
    child2_axis = child2_axis / np.linalg.norm(child2_axis)
    child2_y = orient_vec - np.dot(orient_vec, child2_axis) * child2_axis
    if np.linalg.norm(child2_y) > 1e-10:
        child2_y = child2_y / np.linalg.norm(child2_y)
    else:
        child2_y = parent_y
    child2_z = np.cross(child2_axis, child2_y)
    
    # Transform midpoint offset FROM global TO each child's local coordinates
    # For child 1's WB (at midpoint): offset vector in global = pm_offset - nm_pos
    wm_global = pm_offset - nm_pos
    
    # Transform to child 1's local system (at NM, end B of child 1)
    child1_wb = [
        float(np.dot(wm_global, child1_y)),  # w1 = y component
        float(np.dot(wm_global, child1_z)),  # w2 = z component  
        float(np.dot(wm_global, child1_axis)),  # w3 = x component
    ]
    
    # Transform to child 2's local system (at NM, end A of child 2)
    child2_wa = [
        float(np.dot(wm_global, child2_y)),  # w1 = y component
        float(np.dot(wm_global, child2_z)),  # w2 = z component
        float(np.dot(wm_global, child2_axis)),  # w3 = x component
    ]
    
    logger.info(f"CBAR {eid}: Midpoint offset in global coords: {wm_global}")
    logger.info(f"CBAR {eid}: Child 1 WB (local): {child1_wb}")
    logger.info(f"CBAR {eid}: Child 2 WA (local): {child2_wa}")
    
    # Create 2 child bars with geometrically transformed offsets
    # Child 1 [GA, midpoint]: WA = parent's WA, WB = transformed midpoint offset
    # Child 2 [midpoint, GB]: WA = transformed midpoint offset, WB = parent's WB
    for child_nodes, pa_child, pb_child, child_wa, child_wb in [
        ([ga, nm], pa, 0, wa_effective, child1_wb), 
        ([nm, gb], 0, pb, child2_wa, wb_effective)
    ]:
        new_eid = id_alloc.allocate_element_id()
        new_elements.append({
            'type': 'CBAR',
            'eid': new_eid,
            'pid': pid,
            'nodes': child_nodes,
            'g0': g0,
            'x': x,
            'offt': offt,
            'pa': pa_child,
            'pb': pb_child,
            'wa': child_wa,
            'wb': child_wb,
        })
        logger.info(f"  -> Child CBAR {new_eid}: nodes={child_nodes}, x={x}, wa={child_wa}, wb={child_wb}")
        stats.elements_added += 1


def split_cbeam(
    model: BDF,
    elem: CBEAM,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    stats: RefinementStats,
    elements_to_remove: Set[ElementId],
    new_elements: List[dict]
) -> None:
    """
    Split a CBEAM into 2 child CBEAM elements.
    
    Split pattern:
        GA -------- GB    becomes    GA ---- NM ---- GB
        
    Child beams:
        B1: [GA, NM]  (first half)
        B2: [NM, GB]  (second half)
    
    Args:
        model: pyNastran BDF model
        elem: CBEAM element to split
        edge_cache: Edge midpoint cache
        id_alloc: ID allocator
        stats: Statistics tracker
        elements_to_remove: Set to add original element ID to
        new_elements: List to append new element definitions to
    """
    eid = elem.eid
    pid = elem.pid
    ga, gb = elem.nodes[:2]  # CBEAM nodes are [GA, GB]
    
    # Get orientation - could be G0 (node ID) or X vector
    g0 = elem.g0
    # Ensure X vector is copied correctly as [x1, x2, x3]
    # pyNastran may return this as numpy array or in different formats
    x = None
    if elem.x is not None:
        try:
            x_raw = elem.x
            if hasattr(x_raw, 'tolist'):
                x = x_raw.tolist()  # numpy array -> list
            elif isinstance(x_raw, (list, tuple)):
                x = list(x_raw)
            else:
                x = [float(x_raw[0]), float(x_raw[1]), float(x_raw[2])]
        except (IndexError, TypeError):
            x = None
    
    # Log parent orientation for debugging
    import logging
    logger = logging.getLogger(__name__)
    if g0 is not None:
        logger.info(f"CBEAM {eid}: Splitting - parent uses G0 orientation (g0={g0}), x={x}")
    else:
        logger.info(f"CBEAM {eid}: Splitting - parent uses X vector orientation: x={x} (X1={x[0] if x else None}, X2={x[1] if x else None}, X3={x[2] if x else None})")
    
    # Get optional fields
    offt = getattr(elem, 'offt', 'GGG')
    bit = getattr(elem, 'bit', None)
    pa = getattr(elem, 'pa', 0)
    pb = getattr(elem, 'pb', 0)
    sa = getattr(elem, 'sa', 0)
    sb = getattr(elem, 'sb', 0)
    
    # Extract offset vectors WA and WB, converting to list format
    wa = None
    wb = None
    wa_raw = getattr(elem, 'wa', None)
    wb_raw = getattr(elem, 'wb', None)
    if wa_raw is not None:
        try:
            if hasattr(wa_raw, 'tolist'):
                wa = wa_raw.tolist()
            elif isinstance(wa_raw, (list, tuple)):
                wa = list(wa_raw)
            else:
                wa = [float(wa_raw[0]), float(wa_raw[1]), float(wa_raw[2])]
        except (IndexError, TypeError):
            wa = None
    if wb_raw is not None:
        try:
            if hasattr(wb_raw, 'tolist'):
                wb = wb_raw.tolist()
            elif isinstance(wb_raw, (list, tuple)):
                wb = list(wb_raw)
            else:
                wb = [float(wb_raw[0]), float(wb_raw[1]), float(wb_raw[2])]
        except (IndexError, TypeError):
            wb = None
    
    logger.info(f"CBEAM {eid}: Parent offsets wa={wa}, wb={wb}")
    
    # Get or create midpoint node (shared via edge cache)
    nm = get_or_create_midpoint_node(model, ga, gb, edge_cache, id_alloc, stats)
    
    # Validate orientation - G0 cannot be one of the child element's nodes
    # If G0 is the midpoint or one of the original nodes, we have a problem
    if g0 is not None and g0 in (ga, gb, nm):
        # G0 conflicts with element nodes - convert to X vector instead
        logger.warning(f"CBEAM {eid}: G0={g0} conflicts with element nodes. "
                      f"Converting to X vector orientation.")
        try:
            # Get positions of the nodes to compute a perpendicular vector
            ga_pos = get_grid_xyz(model, ga)
            gb_pos = get_grid_xyz(model, gb)
            g0_pos = get_grid_xyz(model, g0)
            # Original orientation vector from element axis to G0
            axis = gb_pos - ga_pos
            axis = axis / np.linalg.norm(axis)
            to_g0 = g0_pos - ga_pos
            # Project out the axial component to get perpendicular direction
            x_vec = to_g0 - np.dot(to_g0, axis) * axis
            if np.linalg.norm(x_vec) > 1e-10:
                x = list(x_vec / np.linalg.norm(x_vec))
            else:
                x = [0.0, 0.0, 1.0]  # Default if G0 is on element axis
            g0 = None
        except Exception:
            x = [0.0, 0.0, 1.0]  # Fallback default orientation
            g0 = None
    
    # Mark original element for removal
    elements_to_remove.add(eid)
    stats.elements_split += 1
    
    # Log final orientation being used for children
    if g0 is not None:
        logger.info(f"CBEAM {eid}: Children will use G0 orientation (g0={g0})")
    else:
        logger.info(f"CBEAM {eid}: Children will use X vector orientation: x={x}")
    
    # Use [0,0,0] for no offset (not None) to ensure pyNastran writes the values correctly
    zero_offset = [0.0, 0.0, 0.0]
    wa_effective = wa if wa is not None else zero_offset
    wb_effective = wb if wb is not None else zero_offset
    
    # Compute the midpoint offset using global coordinate transformation
    # This matches FEMAP's behavior: interpolate the PHYSICAL offset position
    # in global coordinates, then transform to local coordinates for each child
    
    # Get node positions
    ga_pos = get_grid_xyz(model, ga)
    gb_pos = get_grid_xyz(model, gb)
    nm_pos = get_grid_xyz(model, nm)
    
    # Compute element local coordinate systems
    # For parent: axis from GA to GB
    parent_axis = gb_pos - ga_pos
    parent_axis = parent_axis / np.linalg.norm(parent_axis)
    
    # Get orientation vector (X vector or direction to G0)
    if g0 is not None:
        g0_pos = get_grid_xyz(model, g0)
        orient_vec = g0_pos - ga_pos
    elif x is not None:
        orient_vec = np.array(x)
    else:
        orient_vec = np.array([0.0, 0.0, 1.0])
    
    # Compute parent local coordinate system (y and z axes)
    # y-axis: perpendicular to beam axis, in plane with orientation vector
    parent_y = orient_vec - np.dot(orient_vec, parent_axis) * parent_axis
    if np.linalg.norm(parent_y) > 1e-10:
        parent_y = parent_y / np.linalg.norm(parent_y)
    else:
        # Orientation vector is parallel to beam - pick arbitrary perpendicular
        if abs(parent_axis[0]) < 0.9:
            parent_y = np.cross(parent_axis, np.array([1.0, 0.0, 0.0]))
        else:
            parent_y = np.cross(parent_axis, np.array([0.0, 1.0, 0.0]))
        parent_y = parent_y / np.linalg.norm(parent_y)
    
    # z-axis: perpendicular to both x and y
    parent_z = np.cross(parent_axis, parent_y)
    
    # Transformation matrix from local to global (columns are local axes in global)
    # W = [w1, w2, w3] where w1 is y-offset, w2 is z-offset, w3 is x-offset
    # Local offset = w1*y + w2*z + w3*x
    
    # Transform WA and WB to global coordinates (offset from node position)
    wa_global = wa_effective[0] * parent_y + wa_effective[1] * parent_z + wa_effective[2] * parent_axis
    wb_global = wb_effective[0] * parent_y + wb_effective[1] * parent_z + wb_effective[2] * parent_axis
    
    # The PHYSICAL offset positions in global coordinates
    pa_offset = ga_pos + wa_global
    pb_offset = gb_pos + wb_global
    
    # Interpolate physical offset position at midpoint
    pm_offset = (pa_offset + pb_offset) / 2.0
    
    # Now compute local coordinate systems for child elements
    # Child 1: GA to NM
    child1_axis = nm_pos - ga_pos
    child1_axis = child1_axis / np.linalg.norm(child1_axis)
    child1_y = orient_vec - np.dot(orient_vec, child1_axis) * child1_axis
    if np.linalg.norm(child1_y) > 1e-10:
        child1_y = child1_y / np.linalg.norm(child1_y)
    else:
        child1_y = parent_y
    child1_z = np.cross(child1_axis, child1_y)
    
    # Child 2: NM to GB  
    child2_axis = gb_pos - nm_pos
    child2_axis = child2_axis / np.linalg.norm(child2_axis)
    child2_y = orient_vec - np.dot(orient_vec, child2_axis) * child2_axis
    if np.linalg.norm(child2_y) > 1e-10:
        child2_y = child2_y / np.linalg.norm(child2_y)
    else:
        child2_y = parent_y
    child2_z = np.cross(child2_axis, child2_y)
    
    # Transform midpoint offset FROM global TO each child's local coordinates
    # For child 1's WB (at midpoint): offset vector in global = pm_offset - nm_pos
    wm_global = pm_offset - nm_pos
    
    # Transform to child 1's local system (at NM, end B of child 1)
    child1_wb = [
        float(np.dot(wm_global, child1_y)),  # w1 = y component
        float(np.dot(wm_global, child1_z)),  # w2 = z component  
        float(np.dot(wm_global, child1_axis)),  # w3 = x component
    ]
    
    # Transform to child 2's local system (at NM, end A of child 2)
    child2_wa = [
        float(np.dot(wm_global, child2_y)),  # w1 = y component
        float(np.dot(wm_global, child2_z)),  # w2 = z component
        float(np.dot(wm_global, child2_axis)),  # w3 = x component
    ]
    
    logger.info(f"CBEAM {eid}: Midpoint offset in global coords: {wm_global}")
    logger.info(f"CBEAM {eid}: Child 1 WB (local): {child1_wb}")
    logger.info(f"CBEAM {eid}: Child 2 WA (local): {child2_wa}")
    
    # Create 2 child beams with geometrically transformed offsets
    # Child 1 [GA, midpoint]: WA = parent's WA, WB = transformed midpoint offset
    # Child 2 [midpoint, GB]: WA = transformed midpoint offset, WB = parent's WB
    for child_nodes, pa_child, pb_child, sa_child, sb_child, child_wa, child_wb in [
        ([ga, nm], pa, 0, sa, 0, wa_effective, child1_wb), 
        ([nm, gb], 0, pb, 0, sb, child2_wa, wb_effective)
    ]:
        new_eid = id_alloc.allocate_element_id()
        new_elements.append({
            'type': 'CBEAM',
            'eid': new_eid,
            'pid': pid,
            'nodes': child_nodes,
            'g0': g0,
            'x': x,
            'offt': offt,
            'bit': bit,
            'pa': pa_child,
            'pb': pb_child,
            'wa': child_wa,
            'wb': child_wb,
            'sa': sa_child,
            'sb': sb_child,
        })
        logger.info(f"  -> Child CBEAM {new_eid}: nodes={child_nodes}, x={x}, wa={child_wa}, wb={child_wb}")
        stats.elements_added += 1


def split_crod(
    model: BDF,
    elem: CROD,
    edge_cache: EdgeCache,
    id_alloc: IdAllocator,
    stats: RefinementStats,
    elements_to_remove: Set[ElementId],
    new_elements: List[dict]
) -> None:
    """
    Split a CROD into 2 child CROD elements.
    
    Split pattern:
        GA -------- GB    becomes    GA ---- NM ---- GB
        
    Child rods:
        R1: [GA, NM]  (first half)
        R2: [NM, GB]  (second half)
    
    Args:
        model: pyNastran BDF model
        elem: CROD element to split
        edge_cache: Edge midpoint cache
        id_alloc: ID allocator
        stats: Statistics tracker
        elements_to_remove: Set to add original element ID to
        new_elements: List to append new element definitions to
    """
    eid = elem.eid
    pid = elem.pid
    ga, gb = elem.nodes[:2]  # CROD nodes are [GA, GB]
    
    # Get or create midpoint node (shared via edge cache)
    nm = get_or_create_midpoint_node(model, ga, gb, edge_cache, id_alloc, stats)
    
    # Mark original element for removal
    elements_to_remove.add(eid)
    stats.elements_split += 1
    
    # Create 2 child rods
    for child_nodes in [[ga, nm], [nm, gb]]:
        new_eid = id_alloc.allocate_element_id()
        new_elements.append({
            'type': 'CROD',
            'eid': new_eid,
            'pid': pid,
            'nodes': child_nodes,
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
    
    # Track edge lengths for diagnostics
    all_edge_lengths = []
    
    # IMPORTANT: Process 2D elements FIRST, then 1D elements
    # This ensures that when 1D elements share edges with 2D elements,
    # the midpoint nodes from 2D refinement are already in the edge cache.
    # This maintains conformal mesh connectivity.
    
    # --- PASS 1a: Process 2D elements based on edge length ---
    for eid in current_eids:
        elem = model.elements[eid]
        
        # Handle CQUAD4
        if isinstance(elem, CQUAD4):
            if not should_refine_element(eid, elem.pid, pid_filter, eid_range):
                continue
            
            max_edge = compute_max_edge_length_quad(model, list(elem.nodes))
            all_edge_lengths.append(max_edge)
            if max_edge > target_edge_length:
                split_cquad4(model, elem, edge_cache, id_alloc, stats,
                           elements_to_remove, new_elements)
        
        # Handle CTRIA3
        elif isinstance(elem, CTRIA3):
            if not should_refine_element(eid, elem.pid, pid_filter, eid_range):
                continue
            
            max_edge = compute_max_edge_length_tri(model, list(elem.nodes))
            all_edge_lengths.append(max_edge)
            if max_edge > target_edge_length:
                split_ctria3(model, elem, edge_cache, id_alloc, stats,
                           elements_to_remove, new_elements)
    
    # --- PASS 1b: Closure pass - split 2D elements that share edges with midpoints ---
    # This ensures conformal mesh: if element A was refined and created midpoints,
    # adjacent element B must also be refined to use those midpoints (no free edges)
    # Keep iterating until no more elements need closure refinement
    closure_needed = True
    while closure_needed:
        closure_needed = False
        for eid in current_eids:
            if eid in elements_to_remove:
                continue  # Already marked for removal
            
            elem = model.elements[eid]
            
            if isinstance(elem, CQUAD4):
                # Check if any edge has a midpoint in cache
                nodes = list(elem.nodes)
                edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), 
                        (nodes[2], nodes[3]), (nodes[3], nodes[0])]
                has_midpoint = any(edge_cache.get_midpoint(n1, n2) is not None 
                                  for n1, n2 in edges)
                if has_midpoint:
                    split_cquad4(model, elem, edge_cache, id_alloc, stats,
                               elements_to_remove, new_elements)
                    closure_needed = True  # New midpoints may affect other elements
            
            elif isinstance(elem, CTRIA3):
                # Check if any edge has a midpoint in cache
                nodes = list(elem.nodes)
                edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), 
                        (nodes[2], nodes[0])]
                has_midpoint = any(edge_cache.get_midpoint(n1, n2) is not None 
                                  for n1, n2 in edges)
                if has_midpoint:
                    split_ctria3(model, elem, edge_cache, id_alloc, stats,
                               elements_to_remove, new_elements)
                    closure_needed = True  # New midpoints may affect other elements
    
    # --- PASS 2: Process 1D elements (CBAR, CBEAM, CROD) ---
    # Now the edge cache has all midpoints from 2D refinement
    # 
    # IMPORTANT: 1D elements MUST be split if their edge has a midpoint from 2D
    # refinement, regardless of PID filter. This maintains conformal mesh.
    # PID filter only applies to length-based refinement decisions.
    for eid in current_eids:
        elem = model.elements[eid]
        
        # Handle CBAR
        if isinstance(elem, CBAR):
            ga, gb = elem.nodes[:2]
            edge_has_midpoint = edge_cache.get_midpoint(ga, gb) is not None
            
            # ALWAYS split if edge has midpoint (conformal mesh requirement)
            if edge_has_midpoint:
                split_cbar(model, elem, edge_cache, id_alloc, stats,
                          elements_to_remove, new_elements)
            # Otherwise, check PID filter and length
            elif should_refine_element(eid, elem.pid, pid_filter, eid_range):
                bar_length = compute_bar_length(model, [ga, gb])
                all_edge_lengths.append(bar_length)
                if bar_length > target_edge_length:
                    split_cbar(model, elem, edge_cache, id_alloc, stats,
                              elements_to_remove, new_elements)
        
        # Handle CBEAM
        elif isinstance(elem, CBEAM):
            ga, gb = elem.nodes[:2]
            edge_has_midpoint = edge_cache.get_midpoint(ga, gb) is not None
            
            # ALWAYS split if edge has midpoint (conformal mesh requirement)
            if edge_has_midpoint:
                split_cbeam(model, elem, edge_cache, id_alloc, stats,
                           elements_to_remove, new_elements)
            # Otherwise, check PID filter and length
            elif should_refine_element(eid, elem.pid, pid_filter, eid_range):
                beam_length = compute_bar_length(model, [ga, gb])
                all_edge_lengths.append(beam_length)
                if beam_length > target_edge_length:
                    split_cbeam(model, elem, edge_cache, id_alloc, stats,
                               elements_to_remove, new_elements)
        
        # Handle CROD
        elif isinstance(elem, CROD):
            ga, gb = elem.nodes[:2]
            edge_has_midpoint = edge_cache.get_midpoint(ga, gb) is not None
            
            # ALWAYS split if edge has midpoint (conformal mesh requirement)
            if edge_has_midpoint:
                split_crod(model, elem, edge_cache, id_alloc, stats,
                          elements_to_remove, new_elements)
            # Otherwise, check PID filter and length
            elif should_refine_element(eid, elem.pid, pid_filter, eid_range):
                rod_length = compute_bar_length(model, [ga, gb])
                all_edge_lengths.append(rod_length)
                if rod_length > target_edge_length:
                    split_crod(model, elem, edge_cache, id_alloc, stats,
                              elements_to_remove, new_elements)
    
    # Log edge length diagnostics
    if logger and all_edge_lengths:
        min_len = min(all_edge_lengths)
        max_len = max(all_edge_lengths)
        avg_len = sum(all_edge_lengths) / len(all_edge_lengths)
        logger.info(f"  Edge lengths: min={min_len:.4f}, max={max_len:.4f}, avg={avg_len:.4f}")
        logger.info(f"  Target: {target_edge_length}, Elements exceeding: {stats.elements_split}")
    
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
        elif elem_def['type'] == 'CBAR':
            model.add_cbar(
                eid=elem_def['eid'],
                pid=elem_def['pid'],
                nids=elem_def['nodes'],
                g0=elem_def['g0'],
                x=elem_def['x'],
                offt=elem_def['offt'],
                pa=elem_def['pa'],
                pb=elem_def['pb'],
                wa=elem_def['wa'],
                wb=elem_def['wb'],
            )
        elif elem_def['type'] == 'CBEAM':
            model.add_cbeam(
                eid=elem_def['eid'],
                pid=elem_def['pid'],
                nids=elem_def['nodes'],
                g0=elem_def['g0'],
                x=elem_def['x'],
                offt=elem_def['offt'],
                bit=elem_def['bit'],
                pa=elem_def['pa'],
                pb=elem_def['pb'],
                wa=elem_def['wa'],
                wb=elem_def['wb'],
                sa=elem_def['sa'],
                sb=elem_def['sb'],
            )
        elif elem_def['type'] == 'CROD':
            model.add_crod(
                eid=elem_def['eid'],
                pid=elem_def['pid'],
                nids=elem_def['nodes'],
            )
    
    # Re-cross-reference the model so new nodes work with get_position()
    # This is needed for subsequent passes to correctly compute edge lengths
    if new_elements:
        try:
            model.cross_reference()
        except Exception:
            # If cross-referencing fails, continue anyway - new nodes have CP=0
            # so their xyz values are already in global coordinates
            pass
    
    return stats


def calculate_total_mass(model: BDF, mass_to_lbs_factor: float = 396.4) -> Tuple[float, float]:
    """
    Calculate total mass of the model.
    
    Args:
        model: pyNastran BDF model (must be cross-referenced)
        mass_to_lbs_factor: Conversion factor from model mass units to lbs
        
    Returns:
        Tuple of (mass_in_model_units, mass_in_lbs)
    """
    try:
        # Use get_mass_breakdown which returns a dictionary with mass by property
        mass_breakdown = model.get_mass_breakdown(stop_on_failure=False)
        # Sum up all masses
        total_mass = 0.0
        for pid, mass_data in mass_breakdown.items():
            if isinstance(mass_data, (int, float)):
                total_mass += mass_data
            elif isinstance(mass_data, dict):
                # Some versions return nested dicts
                for key, val in mass_data.items():
                    if isinstance(val, (int, float)):
                        total_mass += val
        mass_lbs = total_mass * mass_to_lbs_factor
        return (total_mass, mass_lbs)
    except Exception as e:
        # If mass calculation fails, try alternative method
        try:
            # Try summing element masses directly
            total_mass = 0.0
            for eid, elem in model.elements.items():
                try:
                    elem_mass = elem.Mass()
                    total_mass += elem_mass
                except:
                    pass
            mass_lbs = total_mass * mass_to_lbs_factor
            return (total_mass, mass_lbs)
        except:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not calculate mass: {e}")
            return (0.0, 0.0)


def refine_mesh(
    input_file: str,
    output_file: str,
    target_edge_length: float,
    max_passes: int = 10,
    pids: Optional[List[int]] = None,
    eid_range: Optional[Tuple[int, int]] = None,
    start_nid: Optional[int] = None,
    start_eid: Optional[int] = None,
    mass_to_lbs_factor: float = 396.4,
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
        start_nid: Optional starting node ID for new nodes (default: max+1)
        start_eid: Optional starting element ID for new elements (default: max+1)
        mass_to_lbs_factor: Conversion factor from model mass units to lbs (default: 396.4)
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
    
    # Read BDF with cross-referencing to handle coordinate systems properly
    # This enables get_position() to return global coordinates for nodes
    # defined in local coordinate systems (CP != 0)
    model = BDF(debug=False)
    model.read_bdf(input_file, xref=True)
    
    # Initial counts
    initial_nodes = len(model.nodes)
    initial_elements = len(model.elements)
    
    # Count element types
    initial_quads = sum(1 for e in model.elements.values() if isinstance(e, CQUAD4))
    initial_tris = sum(1 for e in model.elements.values() if isinstance(e, CTRIA3))
    initial_bars = sum(1 for e in model.elements.values() if isinstance(e, CBAR))
    initial_beams = sum(1 for e in model.elements.values() if isinstance(e, CBEAM))
    initial_rods = sum(1 for e in model.elements.values() if isinstance(e, CROD))
    
    logger.info(f"Initial mesh: {initial_nodes} nodes, {initial_elements} elements")
    logger.info(f"  2D: {initial_quads} CQUAD4, {initial_tris} CTRIA3")
    logger.info(f"  1D: {initial_bars} CBAR, {initial_beams} CBEAM, {initial_rods} CROD")
    
    # Calculate initial mass
    initial_mass, initial_mass_lbs = calculate_total_mass(model, mass_to_lbs_factor)
    logger.info(f"Initial mass: {initial_mass:.6f} (model units) = {initial_mass_lbs:.2f} lbs")
    
    logger.info(f"Target edge length: {target_edge_length}")
    logger.info(f"Max passes: {max_passes}")
    
    if pids:
        logger.info(f"Filtering PIDs: {pids}")
    if eid_range:
        logger.info(f"Filtering EID range: {eid_range[0]} to {eid_range[1]}")
    
    # Initialize ID allocators
    max_grid_id = max(model.nodes.keys()) if model.nodes else 0
    max_elem_id = max(model.elements.keys()) if model.elements else 0
    existing_nids = set(model.nodes.keys())
    existing_eids = set(model.elements.keys())
    
    # Use user-specified starting IDs if provided, otherwise start after max existing
    if start_nid is not None:
        actual_start_nid = start_nid
        # Only warn if the specific starting ID already exists (actual collision)
        if actual_start_nid in existing_nids:
            logger.warning(f"⚠ start_nid ({actual_start_nid}) already exists in model! "
                          f"May cause ID collisions.")
        else:
            logger.info(f"Using user-specified start_nid: {actual_start_nid}")
    else:
        actual_start_nid = max_grid_id + 1
    
    if start_eid is not None:
        actual_start_eid = start_eid
        # Only warn if the specific starting ID already exists (actual collision)
        if actual_start_eid in existing_eids:
            logger.warning(f"⚠ start_eid ({actual_start_eid}) already exists in model! "
                          f"May cause ID collisions.")
        else:
            logger.info(f"Using user-specified start_eid: {actual_start_eid}")
    else:
        actual_start_eid = max_elem_id + 1
    
    logger.info(f"New node IDs will start at: {actual_start_nid}")
    logger.info(f"New element IDs will start at: {actual_start_eid}")
    logger.info(f"(Existing model: max NID={max_grid_id}, max EID={max_elem_id})")
    
    id_alloc = IdAllocator(
        next_grid_id=actual_start_nid,
        next_element_id=actual_start_eid
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
    final_bars = sum(1 for e in model.elements.values() if isinstance(e, CBAR))
    final_beams = sum(1 for e in model.elements.values() if isinstance(e, CBEAM))
    final_rods = sum(1 for e in model.elements.values() if isinstance(e, CROD))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("REFINEMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Passes completed: {total_stats['passes']}")
    logger.info(f"Nodes: {initial_nodes} -> {final_nodes} (+{final_nodes - initial_nodes})")
    logger.info(f"Elements: {initial_elements} -> {final_elements}")
    logger.info(f"  2D: CQUAD4 {initial_quads} -> {final_quads}, CTRIA3 {initial_tris} -> {final_tris}")
    logger.info(f"  1D: CBAR {initial_bars} -> {final_bars}, CBEAM {initial_beams} -> {final_beams}, CROD {initial_rods} -> {final_rods}")
    
    # Sanity checks
    logger.info("\n--- Sanity Checks ---")
    
    # Check 1: Original nodes preserved
    # (We never delete nodes, only add, so this is guaranteed)
    logger.info("✓ Original GRID IDs preserved (by design)")
    
    # Check 2: Element count math
    # For quads: each split removes 1, adds 4 (net +3)
    # For tris: each split removes 1, adds 4 (net +3)
    # For bars/beams/rods: each split removes 1, adds 2 (net +1)
    # Since we don't track split counts by type, skip this check if we have 1D elements
    if initial_bars == 0 and initial_beams == 0 and initial_rods == 0:
        expected_element_change = total_stats['total_elements_split'] * 3
        actual_element_change = final_elements - initial_elements
        if expected_element_change == actual_element_change:
            logger.info(f"✓ Element count change matches expected ({actual_element_change})")
        else:
            logger.warning(f"✗ Element count mismatch: expected +{expected_element_change}, got +{actual_element_change}")
    else:
        logger.info(f"  Element count change: +{final_elements - initial_elements} (mixed 1D/2D)")
    
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
        elif isinstance(elem, (CBAR, CBEAM, CROD)):
            if should_refine_element(eid, elem.pid, pid_filter, eid_range):
                bar_length = compute_bar_length(model, list(elem.nodes[:2]))
                if bar_length > target_edge_length:
                    violations += 1
    
    if violations == 0:
        logger.info("✓ All elements meet target edge length")
    else:
        logger.warning(f"✗ {violations} elements still exceed target edge length")
    
    # Check 4: Mass conservation
    final_mass, final_mass_lbs = calculate_total_mass(model, mass_to_lbs_factor)
    mass_diff = abs(final_mass - initial_mass)
    mass_diff_pct = (mass_diff / initial_mass * 100) if initial_mass > 0 else 0
    
    logger.info(f"\n--- Mass Check ---")
    logger.info(f"Initial mass: {initial_mass:.6f} (model units) = {initial_mass_lbs:.2f} lbs")
    logger.info(f"Final mass:   {final_mass:.6f} (model units) = {final_mass_lbs:.2f} lbs")
    logger.info(f"Difference:   {mass_diff:.6f} ({mass_diff_pct:.4f}%)")
    
    if mass_diff_pct < 0.01:  # Less than 0.01% difference
        logger.info("✓ Mass preserved (< 0.01% change)")
    elif mass_diff_pct < 0.1:  # Less than 0.1% difference
        logger.warning(f"⚠ Small mass change detected ({mass_diff_pct:.4f}%)")
    else:
        logger.error(f"✗ Significant mass change detected ({mass_diff_pct:.4f}%)! Check for errors.")
    
    # Check for ID overflow (8-character field limit = 99999999)
    MAX_ID = 99999999
    max_nid = max(model.nodes.keys()) if model.nodes else 0
    max_eid = max(model.elements.keys()) if model.elements else 0
    
    if max_nid > MAX_ID:
        logger.error(f"✗ GRID IDs exceed 8-character limit! Max NID: {max_nid}")
        logger.error("  Consider starting with a model that has smaller node IDs.")
    elif max_nid > MAX_ID // 10:
        logger.warning(f"⚠ GRID IDs getting large ({max_nid}). May cause issues with more refinement.")
    else:
        logger.info(f"✓ GRID IDs within safe range (max: {max_nid})")
    
    if max_eid > MAX_ID:
        logger.error(f"✗ Element IDs exceed 8-character limit! Max EID: {max_eid}")
        logger.error("  Consider starting with a model that has smaller element IDs.")
    elif max_eid > MAX_ID // 10:
        logger.warning(f"⚠ Element IDs getting large ({max_eid}). May cause issues with more refinement.")
    else:
        logger.info(f"✓ Element IDs within safe range (max: {max_eid})")
    
    # Write output
    logger.info(f"\nWriting refined BDF: {output_file}")
    
    # NOTE: Element-based loads (PLOAD4, PLOAD2, etc.) are NOT remapped.
    # To add load remapping, one would need to:
    # 1. Before refinement, collect all element-based loads
    # 2. Track parent->child element mappings
    # 3. After refinement, create new load cards for child elements
    # This is left as a future enhancement.
    
    # Write with explicit small field format (8-character fields) for FEMAP compatibility
    model.write_bdf(output_file, size=8, is_double=False)
    logger.info("Done.")
    
    total_stats['initial_nodes'] = initial_nodes
    total_stats['final_nodes'] = final_nodes
    total_stats['initial_elements'] = initial_elements
    total_stats['final_elements'] = final_elements
    total_stats['violations'] = violations
    total_stats['initial_mass'] = initial_mass
    total_stats['final_mass'] = final_mass
    total_stats['initial_mass_lbs'] = initial_mass_lbs
    total_stats['final_mass_lbs'] = final_mass_lbs
    total_stats['mass_diff_pct'] = mass_diff_pct
    
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
  python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5 --start-nid 9800001 --start-eid 9800001
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
    parser.add_argument('--start-nid', type=int,
                        help='Starting node ID for new nodes (default: max_existing + 1)')
    parser.add_argument('--start-eid', type=int,
                        help='Starting element ID for new elements (default: max_existing + 1)')
    parser.add_argument('--mass-factor', type=float, default=396.4,
                        help='Mass conversion factor to lbs (default: 396.4)')
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
            start_nid=args.start_nid,
            start_eid=args.start_eid,
            mass_to_lbs_factor=args.mass_factor,
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
