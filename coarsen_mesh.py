#!/usr/bin/env python3
"""
coarsen_mesh.py - Nastran BDF Mesh Coarsening Tool

Coarsens (decimates) a Nastran BDF mesh by collapsing short edges.
Nodes connected to 1D elements, RBEs, masses, SPCs, MPCs, or property
boundaries are protected and will not be moved or removed.

USAGE:
    python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0
    python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0 --pids 100,101
    python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0 --protect-pids 200

OPTIONS:
    --in, -i          Input BDF file path (required)
    --out, -o         Output BDF file path (required)
    --target-min      Collapse edges shorter than this length (required)
    --pids, -p        Only coarsen elements with these PIDs (optional)
    --protect-pids    Additional PIDs whose nodes should never be moved (optional)
    --fill-holes      Detect and collapse interior holes, removing RBE wagon wheels
    --max-iterations  Max collapse iterations (default: unlimited)
    --min-quality     Minimum element quality to allow collapse (default: 0.1)
    --max-normal-dev  Max element normal deviation from original surface in degrees
                      (default: 15). Prevents OML shape distortion on curved regions.
    --max-chord-dev   Max centroid departure from original surface (default: auto,
                      same as target-min). Prevents chord shortcutting across curves.
    --feature-angle   Dihedral angle threshold for protecting sharp corners/creases
                      (default: 30 degrees). Nodes on edges where adjacent element
                      normals differ by more than this are locked in place.
    --max-warp        Maximum allowed quad warp angle in degrees (default: 10).
                      Rejects collapses that would create warped CQUAD4 elements.
    --punch           Read BDF in punch mode
    --verbose, -v     Enable verbose logging

ALGORITHM:
    1. Build set of protected nodes (1D, RBE, mass, SPC, MPC, PID boundary, mesh boundary)
    2. Build edge list sorted by length (shortest first)
    3. For each edge shorter than target:
       a. If both nodes protected -> skip
       b. If one node protected -> collapse to protected node
       c. If neither protected -> collapse to midpoint
       d. Check quality of resulting elements before committing
       e. Update element connectivity and remove degenerate elements
    4. Write output BDF

NOTES:
    - CQUAD4 that degenerates (2 nodes merge) is converted to CTRIA3
    - CTRIA3 that degenerates (2 nodes merge) is removed
    - Protected nodes are NEVER moved or removed
    - Property boundaries are automatically protected

Author: Mesh Coarsening Tool v1.0
"""

import argparse
import logging
import sys
import os
import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

try:
    from pyNastran.bdf.bdf import BDF
except ImportError:
    print("ERROR: pyNastran is required. Install with: pip install pyNastran")
    sys.exit(1)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protected node detection
# ---------------------------------------------------------------------------

def build_protected_nodes(
    model: BDF,
    eligible_pids: Optional[Set[int]] = None,
    extra_protect_pids: Optional[Set[int]] = None,
    node_positions: Optional[Dict[int, np.ndarray]] = None,
    feature_angle: float = 30.0,
) -> Tuple[Set[int], Set[int]]:
    """
    Build the set of nodes that must NOT be moved or removed.

    Protected categories:
      - Nodes on 1D elements (CBAR, CBEAM, CROD, CONROD, CTUBE)
      - Nodes on connector elements (CBUSH, CELAS, CDAMP, etc.)
      - Nodes on rigid elements (RBE2, RBE3, RBAR, RROD)
      - Nodes on mass elements (CONM1, CONM2, CMASS1-4)
      - Nodes referenced by SPCs or MPCs
      - Nodes on property boundaries (shared by elements with different PIDs)
      - Nodes on sharp boundary corners (detected by feature_angle)
      - Nodes on geometric feature edges (adjacent element normals differ by
        more than feature_angle degrees -- sharp corners/creases)
      - Nodes on elements with extra_protect_pids

    Note: smooth boundary-run nodes are NOT protected so that outer edges
    can be coarsened.  When collapsing a boundary edge the caller should
    move the surviving node to the midpoint to preserve geometry.

    Args:
        node_positions: {nid: xyz} needed for feature edge detection.
                        If None, feature edge detection is skipped.
        feature_angle: Dihedral angle threshold in degrees (default 30).
                       Edges where adjacent element normals differ by more
                       than this angle are treated as geometric features.

    Returns:
        (protected_nodes, boundary_nodes) - the set of locked nodes and the
        set of all mesh-boundary nodes (including non-protected ones).
    """
    protected: Set[int] = set()

    # --- 1D and connector elements ---
    connector_types = {'CBAR', 'CBEAM', 'CROD', 'CONROD', 'CTUBE',
                       'CBUSH', 'CBUSH1D', 'CBUSH2D',
                       'CELAS1', 'CELAS2', 'CELAS3', 'CELAS4',
                       'CDAMP1', 'CDAMP2', 'CDAMP3', 'CDAMP4',
                       'CVISC', 'CGAP'}
    for eid, elem in model.elements.items():
        if elem.type in connector_types:
            for n in elem.nodes:
                if isinstance(n, int) and n > 0:
                    protected.add(n)

    # --- Rigid elements ---
    for eid, elem in model.rigid_elements.items():
        # Collect all node references from rigid elements.
        # pyNastran RBE3.node_ids / .nodes can be empty, so we must
        # explicitly check gn, Gmi (RBE2) and refgrid, Gijs (RBE3).
        rigid_nodes: Set[int] = set()

        # Generic attributes
        for attr in ('node_ids', 'nodes'):
            val = getattr(elem, attr, [])
            if val:
                for n in val:
                    if isinstance(n, int):
                        rigid_nodes.add(n)
                    elif isinstance(n, list):
                        for nn in n:
                            if isinstance(nn, int):
                                rigid_nodes.add(nn)

        # RBE2: independent node (gn) and dependent nodes (Gmi)
        gn = getattr(elem, 'gn', None)
        if isinstance(gn, int):
            rigid_nodes.add(gn)
        gmi = getattr(elem, 'Gmi', [])
        if gmi:
            for n in gmi:
                if isinstance(n, int):
                    rigid_nodes.add(n)

        # RBE3: protect the reference grid only.  Gijs (weighted/dependent)
        # nodes are interpolation points — they can be collapsed and the
        # RBE3 updated to use the surviving node.  The safety-net in
        # collapse_edge handles the Gijs update via _replace_node_in_rigid.
        refgrid = getattr(elem, 'refgrid', None)
        if isinstance(refgrid, int):
            rigid_nodes.add(refgrid)

        # RBAR / RROD: nodes attribute
        for attr in ('ga', 'gb'):
            val = getattr(elem, attr, None)
            if isinstance(val, int):
                rigid_nodes.add(val)

        protected |= rigid_nodes

    # --- Mass elements ---
    for eid, elem in model.masses.items():
        node_ids = getattr(elem, 'node_ids', getattr(elem, 'nodes', []))
        for n in node_ids:
            if isinstance(n, int) and n > 0:
                protected.add(n)

    # --- SPCs ---
    for spc_id, spc_list in model.spcs.items():
        for spc in spc_list:
            node_ids = getattr(spc, 'node_ids', getattr(spc, 'nodes', []))
            for n in node_ids:
                if isinstance(n, int):
                    protected.add(n)

    # --- MPCs ---
    for mpc_id, mpc_list in model.mpcs.items():
        for mpc in mpc_list:
            node_ids = getattr(mpc, 'node_ids', getattr(mpc, 'nodes', []))
            for n in node_ids:
                if isinstance(n, int):
                    protected.add(n)

    # --- Property boundaries ---
    # A node shared by 2D elements with different PIDs is protected IF
    # it touches a PID that is NOT in the eligible set.
    # (Nodes shared between two eligible PIDs can still be collapsed.)
    node_pids: Dict[int, Set[int]] = {}
    shell_types = {'CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6'}
    for eid, elem in model.elements.items():
        if elem.type not in shell_types:
            continue
        pid = elem.pid if isinstance(elem.pid, int) else elem.pid
        for n in elem.nodes:
            nid = n if isinstance(n, int) else n
            if nid not in node_pids:
                node_pids[nid] = set()
            node_pids[nid].add(pid)

    for nid, pids_on_node in node_pids.items():
        if len(pids_on_node) > 1:
            if eligible_pids is not None:
                # Protect if node touches any PID NOT in the eligible set
                non_eligible = pids_on_node - eligible_pids
                if non_eligible:
                    protected.add(nid)
            else:
                # No PID filter: protect all PID boundary nodes
                protected.add(nid)

    # --- Extra protect PIDs ---
    if extra_protect_pids:
        for eid, elem in model.elements.items():
            pid = elem.pid if isinstance(elem.pid, int) else elem.pid
            if pid in extra_protect_pids:
                for n in elem.nodes:
                    protected.add(n if isinstance(n, int) else n)

    # --- Mesh boundaries (edges with only one adjacent 2D element) ---
    # Only count edges within the eligible PID set
    edge_count: Dict[Tuple[int, int], int] = {}
    for eid, elem in model.elements.items():
        if elem.type not in shell_types:
            continue
        nodes = [n if isinstance(n, int) else n for n in elem.nodes]
        n_nodes = len(nodes)
        for i in range(n_nodes):
            n1, n2 = nodes[i], nodes[(i + 1) % n_nodes]
            edge = (min(n1, n2), max(n1, n2))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Identify boundary nodes (on free edges) but do NOT blanket-protect them.
    # Only sharp corners will be protected (below).  Smooth boundary runs
    # can be coarsened — intermediate nodes are removed and the surviving
    # node stays on the original edge line.
    boundary_node_set_preliminary: Set[int] = set()
    for (n1, n2), count in edge_count.items():
        if count == 1:
            boundary_node_set_preliminary.add(n1)
            boundary_node_set_preliminary.add(n2)

    # --- Geometric feature edges (sharp corners/creases) ---
    # An interior edge shared by two 2D elements is a feature edge if
    # the dihedral angle between the element normals exceeds the threshold.
    # All nodes on feature edges are protected to preserve sharp geometry.
    if node_positions is not None and feature_angle > 0:
        feature_cos = np.cos(np.radians(feature_angle))
        feature_protected = 0

        # Build edge -> list of adjacent element IDs (only for interior edges)
        edge_elems: Dict[Tuple[int, int], List[int]] = {}
        for eid, elem in model.elements.items():
            if elem.type not in shell_types:
                continue
            nodes = [n if isinstance(n, int) else n for n in elem.nodes]
            n_nodes = len(nodes)
            for i in range(n_nodes):
                e_n1, e_n2 = nodes[i], nodes[(i + 1) % n_nodes]
                edge = (min(e_n1, e_n2), max(e_n1, e_n2))
                if edge not in edge_elems:
                    edge_elems[edge] = []
                edge_elems[edge].append(eid)

        # Compute element normals (reusing existing function)
        elem_normals: Dict[int, np.ndarray] = {}
        for eid, elem in model.elements.items():
            if elem.type not in shell_types:
                continue
            nodes = [n if isinstance(n, int) else n for n in elem.nodes]
            positions = [node_positions[n] for n in nodes
                        if n in node_positions]
            if len(positions) >= 3:
                v1 = positions[1] - positions[0]
                v2 = positions[2] - positions[0]
                cross = np.cross(v1, v2)
                mag = np.linalg.norm(cross)
                if mag > 1e-15:
                    elem_normals[eid] = cross / mag
                else:
                    elem_normals[eid] = np.array([0.0, 0.0, 0.0])

        # Check each interior edge for feature angle
        for (e_n1, e_n2), eids in edge_elems.items():
            if len(eids) != 2:
                continue
            eid_a, eid_b = eids[0], eids[1]
            if eid_a not in elem_normals or eid_b not in elem_normals:
                continue
            norm_a = elem_normals[eid_a]
            norm_b = elem_normals[eid_b]
            if np.linalg.norm(norm_a) < 1e-15 or np.linalg.norm(norm_b) < 1e-15:
                continue
            cos_angle = np.dot(norm_a, norm_b)
            if cos_angle < feature_cos:
                if e_n1 not in protected:
                    feature_protected += 1
                if e_n2 not in protected:
                    feature_protected += 1
                protected.add(e_n1)
                protected.add(e_n2)

        # --- Exterior (free-edge) feature corners ---
        # At boundary nodes where two free edges meet, check if the chain
        # makes a sharp turn.  A straight boundary has a turning angle of
        # ~180 degrees; anything less than (180 - feature_angle) means the
        # boundary direction changes sharply.  Protect the corner node AND
        # its interior (non-boundary) neighbors so that collapsing edges
        # one row inward cannot round the corner.
        boundary_turn_threshold = 180.0 - feature_angle  # e.g. 150 deg
        boundary_cos = np.cos(np.radians(boundary_turn_threshold))
        boundary_corners = 0
        boundary_interior = 0

        # Build boundary-node adjacency from free edges
        free_edges_set: Set[Tuple[int, int]] = set()
        boundary_adj: Dict[int, List[int]] = {}
        for (e_n1, e_n2), eids in edge_elems.items():
            if len(eids) != 1:
                continue
            free_edges_set.add((e_n1, e_n2))
            if e_n1 not in boundary_adj:
                boundary_adj[e_n1] = []
            boundary_adj[e_n1].append(e_n2)
            if e_n2 not in boundary_adj:
                boundary_adj[e_n2] = []
            boundary_adj[e_n2].append(e_n1)

        boundary_nodes_set = set(boundary_adj.keys())

        # Protect boundary endpoints and T-junctions (nodes with != 2
        # boundary neighbors are topological singularities that must stay)
        for nid, neighbors in boundary_adj.items():
            if len(neighbors) != 2:
                protected.add(nid)

        for nid, neighbors in boundary_adj.items():
            if len(neighbors) < 2 or nid not in node_positions:
                continue
            # Compute edge vectors from this node to its boundary neighbors
            vectors = []
            for nbr in neighbors:
                if nbr in node_positions:
                    vec = node_positions[nbr] - node_positions[nid]
                    mag = np.linalg.norm(vec)
                    if mag > 1e-15:
                        vectors.append(vec / mag)
            if len(vectors) < 2:
                continue

            # Check all pairs (usually just 2 neighbors at a boundary node)
            is_sharp = False
            for vi in range(len(vectors)):
                for vj in range(vi + 1, len(vectors)):
                    cos_a = np.dot(vectors[vi], vectors[vj])
                    # cos_a ~ -1 means straight (180 deg), higher means sharper turn
                    if cos_a > boundary_cos:
                        is_sharp = True
                        break
                if is_sharp:
                    break

            if is_sharp:
                if nid not in protected:
                    boundary_corners += 1
                protected.add(nid)

                # Protect interior neighbors (one hop inward) so edges
                # adjacent to this corner cannot be collapsed
                for eid2, elem2 in model.elements.items():
                    if elem2.type not in shell_types:
                        continue
                    enodes = [n if isinstance(n, int) else n
                              for n in elem2.nodes]
                    if nid in enodes:
                        for en in enodes:
                            if en != nid and en not in boundary_nodes_set:
                                if en not in protected:
                                    boundary_interior += 1
                                protected.add(en)

        if boundary_corners > 0 or boundary_interior > 0:
            logger.debug(f"Exterior feature detection: {boundary_corners} "
                        f"corner nodes + {boundary_interior} interior "
                        f"neighbors protected")

        if feature_protected > 0:
            logger.debug(f"Interior feature edges: {feature_protected} "
                        f"nodes protected (dihedral > {feature_angle} deg)")

    return protected, boundary_node_set_preliminary


# ---------------------------------------------------------------------------
# Edge list construction
# ---------------------------------------------------------------------------

def build_edge_list(
    model: BDF,
    eligible_pids: Optional[Set[int]] = None,
) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[int, Set[int]]]:
    """
    Build edge-to-elements and node-to-elements maps for 2D elements.

    Returns:
        edge_to_elems: {(n1, n2): {eid, ...}}
        node_to_elems: {nid: {eid, ...}}
    """
    shell_types = {'CQUAD4', 'CTRIA3'}
    edge_to_elems: Dict[Tuple[int, int], Set[int]] = {}
    node_to_elems: Dict[int, Set[int]] = {}

    for eid, elem in model.elements.items():
        if elem.type not in shell_types:
            continue
        if eligible_pids is not None:
            pid = elem.pid if isinstance(elem.pid, int) else elem.pid
            if pid not in eligible_pids:
                continue

        nodes = [n if isinstance(n, int) else n for n in elem.nodes]
        n_nodes = len(nodes)

        for nid in nodes:
            if nid not in node_to_elems:
                node_to_elems[nid] = set()
            node_to_elems[nid].add(eid)

        for i in range(n_nodes):
            n1, n2 = nodes[i], nodes[(i + 1) % n_nodes]
            edge = (min(n1, n2), max(n1, n2))
            if edge not in edge_to_elems:
                edge_to_elems[edge] = set()
            edge_to_elems[edge].add(eid)

    return edge_to_elems, node_to_elems


def compute_edge_length_cached(node_positions: Dict[int, np.ndarray], n1: int, n2: int) -> float:
    """Compute distance between two nodes using cached global positions."""
    p1 = node_positions[n1]
    p2 = node_positions[n2]
    return float(np.linalg.norm(p2 - p1))


def build_edge_heap(
    node_positions: Dict[int, np.ndarray],
    edge_to_elems: Dict[Tuple[int, int], Set[int]],
    target_min: float,
) -> List[Tuple[float, int, int]]:
    """Build a min-heap of (length, n1, n2) for edges shorter than target."""
    heap = []
    for (n1, n2) in edge_to_elems:
        if n1 not in node_positions or n2 not in node_positions:
            continue
        length = compute_edge_length_cached(node_positions, n1, n2)
        if length < target_min:
            heapq.heappush(heap, (length, n1, n2))
    return heap


# ---------------------------------------------------------------------------
# Original surface reference (for OML preservation)
# ---------------------------------------------------------------------------

def build_surface_reference(
    model: BDF,
    node_positions: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Optional[KDTree], np.ndarray]:
    """
    Build an original-surface reference for OML shape preservation.

    Computes centroid and unit normal of every 2D element in the current mesh
    and builds a KD-tree of all original node positions.

    Returns:
        original_normals:   {eid: unit_normal_vector}
        original_centroids: {eid: centroid_xyz}
        surface_kdtree:     KDTree of original node positions (or None)
        surface_points:     (N, 3) array of node positions used in the KDTree
    """
    shell_types = {'CQUAD4', 'CTRIA3'}
    original_normals: Dict[int, np.ndarray] = {}
    original_centroids: Dict[int, np.ndarray] = {}

    for eid, elem in model.elements.items():
        if elem.type not in shell_types:
            continue
        nodes = [n if isinstance(n, int) else n for n in elem.nodes]
        positions = [node_positions[n] for n in nodes if n in node_positions]
        if len(positions) < 3:
            continue

        centroid = np.mean(positions, axis=0)
        original_centroids[eid] = centroid

        # Normal from first three vertices (works for both tri and quad)
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        cross = np.cross(v1, v2)
        mag = np.linalg.norm(cross)
        if mag > 1e-15:
            original_normals[eid] = cross / mag
        else:
            original_normals[eid] = np.array([0.0, 0.0, 0.0])

    # Build KD-tree from original element centroids for nearest-surface queries.
    # Element centroids are a better surface representation than raw node positions
    # because they capture the actual surface location (nodes may sit at edges/corners).
    centroid_eids = list(original_centroids.keys())
    if centroid_eids:
        surface_points = np.array([original_centroids[eid] for eid in centroid_eids])
        surface_kdtree = KDTree(surface_points)
    else:
        surface_points = np.empty((0, 3))
        surface_kdtree = None

    return original_normals, original_centroids, surface_kdtree, surface_points


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def compute_tri_area_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """Compute area and unit normal of a triangle."""
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    area = np.linalg.norm(cross) / 2.0
    if area > 1e-15:
        normal = cross / (2.0 * area)
    else:
        normal = np.array([0.0, 0.0, 0.0])
    return area, normal


def compute_quad_warp_angle(p0: np.ndarray, p1: np.ndarray,
                            p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute the warp angle of a CQUAD4 in degrees.

    Warp is the angle between the normals of the two triangles formed by
    splitting the quad along each diagonal.  A perfectly planar quad has
    warp = 0.  Nastran typically fails elements with warp > 10-15 degrees.
    """
    # Split along diagonal 0-2
    n1_a = np.cross(p1 - p0, p2 - p0)
    n1_b = np.cross(p2 - p0, p3 - p0)
    mag_a = np.linalg.norm(n1_a)
    mag_b = np.linalg.norm(n1_b)
    if mag_a < 1e-15 or mag_b < 1e-15:
        return 180.0
    cos1 = np.clip(np.dot(n1_a / mag_a, n1_b / mag_b), -1.0, 1.0)

    # Split along diagonal 1-3
    n2_a = np.cross(p2 - p1, p3 - p1)
    n2_b = np.cross(p3 - p1, p0 - p1)
    mag_a = np.linalg.norm(n2_a)
    mag_b = np.linalg.norm(n2_b)
    if mag_a < 1e-15 or mag_b < 1e-15:
        return 180.0
    cos2 = np.clip(np.dot(n2_a / mag_a, n2_b / mag_b), -1.0, 1.0)

    # Warp angle is the worst of the two diagonals
    warp1 = np.degrees(np.arccos(cos1))
    warp2 = np.degrees(np.arccos(cos2))
    return max(warp1, warp2)


def compute_interior_angles(positions: List[np.ndarray]) -> List[float]:
    """Compute interior angles (degrees) at each vertex of a polygon."""
    n = len(positions)
    angles = []
    for i in range(n):
        v_prev = positions[(i - 1) % n] - positions[i]
        v_next = positions[(i + 1) % n] - positions[i]
        mag_prev = np.linalg.norm(v_prev)
        mag_next = np.linalg.norm(v_next)
        if mag_prev < 1e-15 or mag_next < 1e-15:
            angles.append(0.0)
            continue
        cos_a = np.clip(np.dot(v_prev, v_next) / (mag_prev * mag_next),
                        -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    return angles


def compute_element_quality(node_positions: Dict[int, np.ndarray],
                            node_ids: List[int],
                            max_warp: float = 10.0) -> float:
    """
    Compute a quality metric for a 2D element.
    Returns a value between 0 (degenerate/unacceptable) and 1 (perfect).

    Checks:
      - Aspect ratio: min_edge / max_edge
      - Interior angles: no angle < 5 deg or > 175 deg
      - Warp (CQUAD4 only): non-planarity must be < max_warp degrees
      - Taper (CQUAD4 only): ratio of areas of triangles formed by diagonals
    """
    if len(node_ids) < 3:
        return 0.0

    positions = [node_positions[n] for n in node_ids]
    n = len(positions)

    # Edge lengths
    edges = []
    for i in range(n):
        edge_len = np.linalg.norm(positions[(i + 1) % n] - positions[i])
        edges.append(edge_len)

    max_edge = max(edges)
    min_edge = min(edges)
    if max_edge < 1e-15:
        return 0.0

    aspect = min_edge / max_edge

    # Interior angle check
    angles = compute_interior_angles(positions)
    min_angle = min(angles) if angles else 0.0
    max_angle = max(angles) if angles else 180.0

    if min_angle < 1.0:
        return 0.0

    if n == 3:
        if max_angle > 175.0 or min_angle < 5.0:
            return 0.0
    elif n == 4:
        if max_angle > 175.0 or min_angle < 5.0:
            return 0.0

        # Warp check (quad only)
        warp = compute_quad_warp_angle(positions[0], positions[1],
                                       positions[2], positions[3])
        if warp > max_warp:
            return 0.0

        # Taper check: ratio of areas of two triangles from each diagonal
        area_a, _ = compute_tri_area_normal(positions[0], positions[1],
                                             positions[2])
        area_b, _ = compute_tri_area_normal(positions[0], positions[2],
                                             positions[3])
        if area_a > 1e-15 and area_b > 1e-15:
            taper = min(area_a, area_b) / max(area_a, area_b)
            if taper < 0.1:
                return 0.0
        else:
            return 0.0

    return aspect


def check_collapse_quality(
    model: BDF,
    node_positions: Dict[int, np.ndarray],
    n_remove: int,
    n_keep: int,
    node_to_elems: Dict[int, Set[int]],
    min_quality: float,
    original_normals: Optional[Dict[int, np.ndarray]] = None,
    original_centroids: Optional[Dict[int, np.ndarray]] = None,
    surface_kdtree: Optional[KDTree] = None,
    max_normal_dev_cos: float = -1.0,
    max_chord_dev: float = 0.0,
    max_warp: float = 10.0,
) -> str:
    """
    Check if collapsing n_remove into n_keep would produce acceptable elements.

    Returns:
        "ok" if the collapse is acceptable.
        "quality" if element quality or inversion check failed.
        "normal_dev" if normal deviation from original surface exceeded.
        "chord_dev" if centroid departure from original surface exceeded.
    """
    affected_eids = set()
    if n_remove in node_to_elems:
        affected_eids |= node_to_elems[n_remove]
    if n_keep in node_to_elems:
        affected_eids |= node_to_elems[n_keep]

    for eid in affected_eids:
        if eid not in model.elements:
            continue
        elem = model.elements[eid]
        if elem.type not in ('CQUAD4', 'CTRIA3'):
            continue

        old_nodes = [n if isinstance(n, int) else n for n in elem.nodes]

        # Simulate the collapse
        new_nodes = [n_keep if n == n_remove else n for n in old_nodes]

        # Remove duplicates to check for degenerate
        unique_nodes = []
        for n in new_nodes:
            if n not in unique_nodes:
                unique_nodes.append(n)

        # CTRIA3 with < 3 unique nodes -> will be removed (ok)
        if elem.type == 'CTRIA3' and len(unique_nodes) < 3:
            continue

        # CQUAD4 with < 3 unique nodes -> would be degenerate line (bad)
        if elem.type == 'CQUAD4' and len(unique_nodes) < 3:
            return "quality"

        # CQUAD4 with 3 unique -> will become CTRIA3
        if elem.type == 'CQUAD4' and len(unique_nodes) == 3:
            new_nodes = unique_nodes

        # Check quality of resulting element
        if len(new_nodes) >= 3:
            if not all(n in node_positions for n in new_nodes):
                return "quality"
            quality = compute_element_quality(node_positions, new_nodes,
                                                max_warp=max_warp)
            if quality < min_quality:
                return "quality"

            # Check for inverted element (normal direction should be consistent)
            positions = [node_positions[n] for n in new_nodes]
            if len(positions) >= 3:
                _, new_normal = compute_tri_area_normal(
                    positions[0], positions[1], positions[2]
                )
                old_positions = [node_positions[n] for n in old_nodes
                                if n in node_positions]
                if len(old_positions) >= 3:
                    _, old_normal = compute_tri_area_normal(
                        old_positions[0], old_positions[1], old_positions[2]
                    )
                    if np.dot(new_normal, old_normal) < 0:
                        return "quality"  # Inverted

            # --- OML preservation: normal deviation check ---
            if original_normals is not None and max_normal_dev_cos > -1.0:
                if eid in original_normals:
                    orig_normal = original_normals[eid]
                    if np.linalg.norm(orig_normal) > 1e-15 and len(positions) >= 3:
                        new_norm_mag = np.linalg.norm(new_normal)
                        if new_norm_mag > 1e-15:
                            cos_angle = np.dot(new_normal, orig_normal)
                            if cos_angle < max_normal_dev_cos:
                                return "normal_dev"

            # --- OML preservation: chord deviation check ---
            if surface_kdtree is not None and max_chord_dev > 0.0:
                new_centroid = np.mean(positions, axis=0)
                dist, _ = surface_kdtree.query(new_centroid)
                if dist > max_chord_dev:
                    return "chord_dev"

    return "ok"


# ---------------------------------------------------------------------------
# Edge collapse
# ---------------------------------------------------------------------------

def collapse_edge(
    model: BDF,
    node_positions: Dict[int, np.ndarray],
    n_remove: int,
    n_keep: int,
    node_to_elems: Dict[int, Set[int]],
    edge_to_elems: Dict[Tuple[int, int], Set[int]],
    all_node_to_elems: Dict[int, Set[int]],
    removed_nodes: Set[int],
    stats: Dict,
) -> None:
    """
    Collapse edge by merging n_remove into n_keep.

    Updates model elements in place, removes degenerate elements,
    converts CQUAD4 -> CTRIA3 when appropriate.

    all_node_to_elems tracks ALL elements (including non-eligible PIDs)
    to ensure node references are updated everywhere.
    """
    shell_types = {'CQUAD4', 'CTRIA3'}

    # Get ALL elements referencing n_remove (including non-eligible PIDs)
    affected_eids = set()
    if n_remove in all_node_to_elems:
        affected_eids = all_node_to_elems[n_remove].copy()

    elements_to_remove = set()
    elements_to_convert = {}  # eid -> new CTRIA3 node list

    for eid in affected_eids:
        if eid not in model.elements:
            continue
        elem = model.elements[eid]

        if elem.type not in shell_types:
            # Non-shell element (CBUSH, CBAR, etc.): just replace the node reference
            for i in range(len(elem.nodes)):
                if isinstance(elem.nodes[i], int) and elem.nodes[i] == n_remove:
                    elem.nodes[i] = n_keep
            continue

        old_nodes = [n if isinstance(n, int) else n for n in elem.nodes]

        # Replace n_remove with n_keep
        new_nodes = [n_keep if n == n_remove else n for n in old_nodes]

        # Check for degenerate
        unique_nodes = []
        for n in new_nodes:
            if n not in unique_nodes:
                unique_nodes.append(n)

        if elem.type == 'CTRIA3':
            if len(unique_nodes) < 3:
                elements_to_remove.add(eid)
                stats['elements_removed'] += 1
                continue
            # Update nodes in place
            for i in range(len(elem.nodes)):
                if elem.nodes[i] == n_remove:
                    elem.nodes[i] = n_keep

        elif elem.type == 'CQUAD4':
            if len(unique_nodes) < 3:
                elements_to_remove.add(eid)
                stats['elements_removed'] += 1
                continue
            elif len(unique_nodes) == 3:
                # Convert CQUAD4 -> CTRIA3
                elements_to_convert[eid] = unique_nodes
                stats['quads_to_tris'] += 1
                continue
            else:
                # Still a valid quad, just update nodes
                for i in range(len(elem.nodes)):
                    if elem.nodes[i] == n_remove:
                        elem.nodes[i] = n_keep

    # Remove degenerate elements
    for eid in elements_to_remove:
        if eid in model.elements:
            del model.elements[eid]
        # Clean up maps
        for nid in list(node_to_elems.keys()):
            if eid in node_to_elems.get(nid, set()):
                node_to_elems[nid].discard(eid)

    # Convert CQUAD4 -> CTRIA3
    for eid, tri_nodes in elements_to_convert.items():
        if eid not in model.elements:
            continue
        old_elem = model.elements[eid]
        pid = old_elem.pid if isinstance(old_elem.pid, int) else old_elem.pid
        theta_mcid = getattr(old_elem, 'theta_mcid', 0.0)
        zoffset = getattr(old_elem, 'zoffset', 0.0)
        tflag = getattr(old_elem, 'tflag', 0)
        T1 = getattr(old_elem, 'T1', None)
        T2 = getattr(old_elem, 'T2', None)
        T3 = getattr(old_elem, 'T3', None)

        # Remove old quad
        del model.elements[eid]

        # Add new tri with same EID and PID
        model.add_ctria3(
            eid=eid, pid=pid, nids=tri_nodes,
            theta_mcid=theta_mcid, zoffset=zoffset,
            tflag=tflag, T1=T1, T2=T2, T3=T3,
        )

        # Update node_to_elems for the removed node
        for nid in node_to_elems:
            if eid in node_to_elems[nid] and nid not in tri_nodes:
                node_to_elems[nid].discard(eid)

    # Transfer element references from n_remove to n_keep
    if n_remove in node_to_elems:
        remaining = node_to_elems[n_remove] - elements_to_remove
        if n_keep not in node_to_elems:
            node_to_elems[n_keep] = set()
        node_to_elems[n_keep] |= remaining
        del node_to_elems[n_remove]

    # Also update all_node_to_elems
    if n_remove in all_node_to_elems:
        remaining_all = all_node_to_elems[n_remove] - elements_to_remove
        if n_keep not in all_node_to_elems:
            all_node_to_elems[n_keep] = set()
        all_node_to_elems[n_keep] |= remaining_all
        del all_node_to_elems[n_remove]

    # Update edge_to_elems: remove edges involving n_remove, add for n_keep
    edges_to_remove = [e for e in edge_to_elems if n_remove in e]
    for edge in edges_to_remove:
        eids = edge_to_elems.pop(edge)
        # Create replacement edge with n_keep
        other = edge[0] if edge[1] == n_remove else edge[1]
        if other == n_keep:
            continue  # This was the collapsed edge itself
        new_edge = (min(other, n_keep), max(other, n_keep))
        if new_edge not in edge_to_elems:
            edge_to_elems[new_edge] = set()
        edge_to_elems[new_edge] |= (eids - elements_to_remove)

    # Remove the collapsed edge
    collapsed_edge = (min(n_remove, n_keep), max(n_remove, n_keep))
    edge_to_elems.pop(collapsed_edge, None)

    # Safety net: update rigid element references (RBE2, RBE3, RBAR, etc.)
    # so that no rigid element points to the removed node.
    for rbe_eid, rbe_elem in model.rigid_elements.items():
        _replace_node_in_rigid(rbe_elem, n_remove, n_keep)

    # Safety net: update mass element references
    for mass_eid, mass_elem in model.masses.items():
        mass_nodes = getattr(mass_elem, 'nodes',
                             getattr(mass_elem, 'node_ids', []))
        if mass_nodes:
            for i in range(len(mass_nodes)):
                if mass_nodes[i] == n_remove:
                    mass_nodes[i] = n_keep

    # Safety net: update SPC references
    for spc_id, spc_list in model.spcs.items():
        for spc in spc_list:
            spc_nodes = getattr(spc, 'node_ids',
                                getattr(spc, 'nodes', []))
            if spc_nodes:
                for i in range(len(spc_nodes)):
                    if spc_nodes[i] == n_remove:
                        spc_nodes[i] = n_keep

    # Safety net: update MPC references
    for mpc_id, mpc_list in model.mpcs.items():
        for mpc in mpc_list:
            mpc_nodes = getattr(mpc, 'node_ids',
                                getattr(mpc, 'nodes', []))
            if mpc_nodes:
                for i in range(len(mpc_nodes)):
                    if mpc_nodes[i] == n_remove:
                        mpc_nodes[i] = n_keep

    # Remove node from model and position cache
    if n_remove in model.nodes:
        del model.nodes[n_remove]
    removed_nodes.add(n_remove)

    stats['edges_collapsed'] += 1
    stats['nodes_removed'] += 1


# ---------------------------------------------------------------------------
# Main coarsening function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hole detection and removal
# ---------------------------------------------------------------------------

def find_free_edge_loops(model: BDF) -> List[List[int]]:
    """
    Find closed loops of free edges (edges with only 1 adjacent 2D element).
    Each loop is a list of node IDs forming the boundary of a hole.
    """
    shell_types = {'CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6'}
    from collections import defaultdict

    edge_count: Dict[Tuple[int, int], int] = {}
    for eid, elem in model.elements.items():
        if elem.type not in shell_types:
            continue
        nodes = list(elem.nodes)
        n = len(nodes)
        for i in range(n):
            n1, n2 = nodes[i], nodes[(i + 1) % n]
            edge = (min(n1, n2), max(n1, n2))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Build adjacency for free edges only
    free_adj: Dict[int, List[int]] = defaultdict(list)
    for (n1, n2), count in edge_count.items():
        if count == 1:
            free_adj[n1].append(n2)
            free_adj[n2].append(n1)

    if not free_adj:
        return []

    # Trace closed loops
    visited: Set[int] = set()
    loops: List[List[int]] = []

    for start in free_adj:
        if start in visited:
            continue
        loop = []
        current = start
        prev = None
        while True:
            visited.add(current)
            loop.append(current)
            neighbors = [n for n in free_adj[current] if n != prev]
            if not neighbors:
                break
            next_node = neighbors[0]
            if next_node == start and len(loop) > 2:
                loops.append(loop)
                break
            if next_node in visited:
                break
            prev = current
            current = next_node

    return loops


def classify_loops_as_holes(
    loops: List[List[int]],
    node_positions: Dict[int, np.ndarray],
    model: BDF,
) -> List[List[int]]:
    """
    Classify free edge loops as interior holes vs outer boundary.

    Strategy: a loop is an "outer boundary" only if it is significantly
    larger than all other loops (at least 3x the next largest). If all
    loops are similar size (like dome apex holes on a closed tank), they
    are all treated as interior holes.

    A single loop is also treated as a hole if it's small relative to
    the model (perimeter < 20% of model extent).
    """
    # Compute perimeter for each loop
    perimeters = []
    for loop in loops:
        perimeter = 0.0
        for i in range(len(loop)):
            n1 = loop[i]
            n2 = loop[(i + 1) % len(loop)]
            if n1 in node_positions and n2 in node_positions:
                perimeter += float(np.linalg.norm(
                    node_positions[n2] - node_positions[n1]
                ))
        perimeters.append(perimeter)

    if not perimeters:
        return []

    # Compute model extent for scale reference
    all_pos = np.array(list(node_positions.values()))
    model_extent = np.max([all_pos[:, i].ptp() for i in range(3)])

    max_perimeter = max(perimeters)
    sorted_perimeters = sorted(perimeters, reverse=True)

    holes = []
    for i, loop in enumerate(loops):
        is_outer = False

        if len(loops) == 1:
            # Single loop: outer boundary if perimeter is large relative to model
            if perimeters[i] > 0.2 * model_extent:
                is_outer = True
        else:
            # Multiple loops: outer boundary only if it's much larger than all others
            if perimeters[i] == max_perimeter:
                second_largest = sorted_perimeters[1] if len(sorted_perimeters) > 1 else 0
                if second_largest > 0 and max_perimeter / second_largest > 3.0:
                    is_outer = True

        if is_outer:
            logger.debug(f"  Outer boundary: {len(loop)} nodes, "
                        f"perimeter={perimeters[i]:.2f}")
        else:
            holes.append(loop)
            logger.debug(f"  Interior hole: {len(loop)} nodes, "
                        f"perimeter={perimeters[i]:.2f}")

    return holes


def find_wagon_wheel_rbe(
    model: BDF,
    hole_nodes: Set[int],
) -> Optional[Tuple[int, int, str]]:
    """
    Find an RBE wagon wheel whose spokes connect to hole boundary nodes.

    Looks for RBE2/RBE3 elements where dependent nodes overlap with the
    hole boundary. The independent node (center) is returned.

    Returns:
        (rbe_eid, center_nid, rbe_type) or None if no wagon wheel found
    """
    for eid, elem in model.rigid_elements.items():
        if elem.type == 'RBE2':
            # RBE2: independent node = gn, dependent nodes = Gmi
            ind_node = getattr(elem, 'gn', None)
            dep_nodes = set(getattr(elem, 'Gmi', []))
            # Check if dependent nodes overlap significantly with hole boundary
            overlap = dep_nodes & hole_nodes
            if len(overlap) >= 3:  # At least 3 spokes to the hole
                return (eid, ind_node, 'RBE2')

        elif elem.type == 'RBE3':
            # RBE3: reference node = refgrid, weighted nodes = Gijs
            ref_node = getattr(elem, 'refgrid', None)
            gijs = getattr(elem, 'Gijs', [])
            dep_nodes = set()
            for gij_list in gijs:
                if isinstance(gij_list, list):
                    dep_nodes.update(gij_list)
                elif isinstance(gij_list, int):
                    dep_nodes.add(gij_list)
            overlap = dep_nodes & hole_nodes
            if len(overlap) >= 3:
                return (eid, ref_node, 'RBE3')

    return None


def find_elements_on_center_node(
    model: BDF,
    center_nid: int,
    wagon_wheel_eid: int,
) -> Dict[str, List]:
    """
    Find all elements connected to the wagon wheel center node
    (excluding the wagon wheel itself).

    Returns dict with keys: 'rigid_elements', 'elements', 'masses'
    """
    connected = {
        'rigid_elements': [],  # (eid, type)
        'elements': [],        # (eid, type)
        'masses': [],          # (eid, type)
    }

    # Check rigid elements (other RBEs)
    for eid, elem in model.rigid_elements.items():
        if eid == wagon_wheel_eid:
            continue
        node_ids = getattr(elem, 'node_ids', [])
        if not node_ids:
            node_ids = []
            # Try common attributes
            for attr in ['gn', 'refgrid', 'Gmi', 'Gijs']:
                val = getattr(elem, attr, None)
                if isinstance(val, int):
                    node_ids.append(val)
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, int):
                            node_ids.append(v)
                        elif isinstance(v, list):
                            node_ids.extend(v2 for v2 in v if isinstance(v2, int))
        if center_nid in node_ids:
            connected['rigid_elements'].append((eid, elem.type))

    # Check regular elements (1D, 2D)
    for eid, elem in model.elements.items():
        if center_nid in list(elem.nodes):
            connected['elements'].append((eid, elem.type))

    # Check masses
    for eid, elem in model.masses.items():
        node_ids = getattr(elem, 'node_ids', getattr(elem, 'nodes', []))
        if center_nid in list(node_ids):
            connected['masses'].append((eid, elem.type))

    return connected


def fill_holes(
    model: BDF,
    node_positions: Dict[int, np.ndarray],
) -> Dict:
    """
    Detect and collapse interior holes, removing RBE wagon wheels.

    For each interior hole:
    1. Find and remove RBE wagon wheel (if present)
    2. Create a new node at the centroid of the hole boundary
    3. Reattach any outgoing connections from the RBE center to the new node
    4. Merge all hole boundary nodes into the new node
    5. Clean up degenerate elements

    Args:
        model: pyNastran BDF model (un-cross-referenced)
        node_positions: Cached global node positions

    Returns:
        Stats dictionary
    """
    stats = {
        'holes_found': 0,
        'holes_collapsed': 0,
        'rbes_removed': 0,
        'nodes_removed': 0,
        'elements_removed': 0,
        'quads_to_tris': 0,
        'connections_reattached': 0,
    }

    shell_types = {'CQUAD4', 'CTRIA3'}

    # Step 1: Find free edge loops
    loops = find_free_edge_loops(model)
    logger.info(f"Free edge loops found: {len(loops)}")

    if len(loops) <= 1:
        logger.info("No interior holes detected (0 or 1 boundary loop)")
        return stats

    # Step 2: Classify as holes vs outer boundary
    holes = classify_loops_as_holes(loops, node_positions, model)
    stats['holes_found'] = len(holes)
    logger.info(f"Interior holes: {len(holes)}")

    if not holes:
        return stats

    # Allocate new node IDs
    max_nid = max(model.nodes.keys())

    for hole_idx, hole_loop in enumerate(holes):
        hole_nodes = set(hole_loop)
        logger.info(f"\n--- Hole {hole_idx + 1}/{len(holes)}: "
                    f"{len(hole_loop)} boundary nodes ---")

        # Step 3: Find wagon wheel RBE
        rbe_info = find_wagon_wheel_rbe(model, hole_nodes)
        center_nid = None
        connected = {'rigid_elements': [], 'elements': [], 'masses': []}

        if rbe_info:
            rbe_eid, center_nid, rbe_type = rbe_info
            logger.info(f"  Wagon wheel: {rbe_type} {rbe_eid}, "
                       f"center node={center_nid}")

            # Find what else is connected to the center node
            connected = find_elements_on_center_node(
                model, center_nid, rbe_eid
            )
            logger.info(f"  Center node connections: "
                       f"{len(connected['rigid_elements'])} RBEs, "
                       f"{len(connected['elements'])} elements, "
                       f"{len(connected['masses'])} masses")

            # Remove the wagon wheel RBE
            if rbe_eid in model.rigid_elements:
                del model.rigid_elements[rbe_eid]
                stats['rbes_removed'] += 1
                logger.info(f"  Removed wagon wheel {rbe_type} {rbe_eid}")
        else:
            logger.info(f"  No wagon wheel RBE found for this hole")

        # Step 4: Compute centroid and create new node
        hole_positions = [node_positions[n] for n in hole_loop
                         if n in node_positions]
        if not hole_positions:
            logger.warning(f"  No positions for hole nodes, skipping")
            continue

        centroid = np.mean(hole_positions, axis=0)
        max_nid += 1
        new_nid = max_nid

        model.add_grid(new_nid, list(centroid), cp=0, cd=0, ps=0, seid=0)
        node_positions[new_nid] = centroid
        logger.info(f"  Created centroid node {new_nid} at "
                    f"[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

        # Step 5: Reattach outgoing connections from center to new node
        if center_nid is not None:
            # Reattach other RBEs
            for rbe_eid_other, rbe_type_other in connected['rigid_elements']:
                if rbe_eid_other in model.rigid_elements:
                    elem = model.rigid_elements[rbe_eid_other]
                    _replace_node_in_rigid(elem, center_nid, new_nid)
                    stats['connections_reattached'] += 1
                    logger.info(f"  Reattached {rbe_type_other} {rbe_eid_other} "
                               f"from node {center_nid} -> {new_nid}")

            # Reattach regular elements (1D, etc.)
            for elem_eid, elem_type in connected['elements']:
                if elem_eid in model.elements:
                    elem = model.elements[elem_eid]
                    for i in range(len(elem.nodes)):
                        if elem.nodes[i] == center_nid:
                            elem.nodes[i] = new_nid
                    stats['connections_reattached'] += 1
                    logger.debug(f"  Reattached {elem_type} {elem_eid} "
                                f"from node {center_nid} -> {new_nid}")

            # Reattach masses
            for mass_eid, mass_type in connected['masses']:
                if mass_eid in model.masses:
                    elem = model.masses[mass_eid]
                    nodes = getattr(elem, 'nodes', getattr(elem, 'node_ids', []))
                    for i in range(len(nodes)):
                        if nodes[i] == center_nid:
                            nodes[i] = new_nid
                    stats['connections_reattached'] += 1
                    logger.info(f"  Reattached {mass_type} {mass_eid} "
                               f"from node {center_nid} -> {new_nid}")

            # Remove old center node
            if center_nid in model.nodes:
                del model.nodes[center_nid]
                node_positions.pop(center_nid, None)
                stats['nodes_removed'] += 1

        # Step 6: Merge all hole boundary nodes into the new centroid node
        # Update all elements referencing any hole boundary node
        for eid, elem in list(model.elements.items()):
            if elem.type not in shell_types:
                continue
            old_nodes = list(elem.nodes)
            changed = False
            new_nodes = []
            for n in old_nodes:
                if n in hole_nodes:
                    new_nodes.append(new_nid)
                    changed = True
                else:
                    new_nodes.append(n)

            if not changed:
                continue

            # Check for degeneracy
            unique = []
            for n in new_nodes:
                if n not in unique:
                    unique.append(n)

            if elem.type == 'CTRIA3' and len(unique) < 3:
                # Degenerate tri -> remove
                del model.elements[eid]
                stats['elements_removed'] += 1
                continue

            if elem.type == 'CQUAD4':
                if len(unique) < 3:
                    # Degenerate to a line -> remove
                    del model.elements[eid]
                    stats['elements_removed'] += 1
                    continue
                elif len(unique) == 3:
                    # Convert to CTRIA3
                    pid = elem.pid
                    theta_mcid = getattr(elem, 'theta_mcid', 0.0)
                    zoffset = getattr(elem, 'zoffset', 0.0)
                    tflag = getattr(elem, 'tflag', 0)
                    T1 = getattr(elem, 'T1', None)
                    T2 = getattr(elem, 'T2', None)
                    T3 = getattr(elem, 'T3', None)
                    del model.elements[eid]
                    model.add_ctria3(
                        eid=eid, pid=pid, nids=unique,
                        theta_mcid=theta_mcid, zoffset=zoffset,
                        tflag=tflag, T1=T1, T2=T2, T3=T3,
                    )
                    stats['quads_to_tris'] += 1
                    continue

            # Valid element, update nodes
            for i in range(len(elem.nodes)):
                elem.nodes[i] = new_nodes[i]

        # Remove old hole boundary nodes
        for nid in hole_loop:
            if nid in model.nodes:
                del model.nodes[nid]
                node_positions.pop(nid, None)
                stats['nodes_removed'] += 1

        stats['holes_collapsed'] += 1
        logger.info(f"  Hole {hole_idx + 1} collapsed successfully")

    logger.info(f"\nHole fill summary: {stats['holes_collapsed']} holes collapsed, "
                f"{stats['nodes_removed']} nodes removed, "
                f"{stats['elements_removed']} elements removed")

    return stats


def _replace_node_in_rigid(elem, old_nid: int, new_nid: int) -> None:
    """Replace a node ID in a rigid element (RBE2, RBE3, etc.)."""
    # RBE2
    if hasattr(elem, 'gn') and elem.gn == old_nid:
        elem.gn = new_nid
    if hasattr(elem, 'Gmi'):
        elem.Gmi = [new_nid if n == old_nid else n for n in elem.Gmi]

    # RBE3
    if hasattr(elem, 'refgrid') and elem.refgrid == old_nid:
        elem.refgrid = new_nid
    if hasattr(elem, 'Gijs'):
        new_gijs = []
        for gij in elem.Gijs:
            if isinstance(gij, list):
                new_gijs.append([new_nid if n == old_nid else n for n in gij])
            elif isinstance(gij, int) and gij == old_nid:
                new_gijs.append(new_nid)
            else:
                new_gijs.append(gij)
        elem.Gijs = new_gijs

    # Generic node_ids fallback
    if hasattr(elem, 'nodes'):
        for i in range(len(elem.nodes)):
            if elem.nodes[i] == old_nid:
                elem.nodes[i] = new_nid


# ---------------------------------------------------------------------------
# Mass calculation
# ---------------------------------------------------------------------------

def calculate_total_mass(
    model: BDF,
    mass_to_lbs_factor: float = 386.4,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Calculate total mass of the model (must be cross-referenced).

    Returns:
        (mass_model_units, mass_lbs, per_pid_breakdown)
    """
    per_pid: Dict[int, float] = {}
    total_mass = 0.0

    try:
        mass_breakdown = model.get_mass_breakdown(stop_on_failure=False)
        for pid, mass_data in mass_breakdown.items():
            if isinstance(mass_data, (int, float)):
                per_pid[pid] = float(mass_data)
                total_mass += float(mass_data)
            elif isinstance(mass_data, dict):
                pid_total = 0.0
                for val in mass_data.values():
                    if isinstance(val, (int, float)):
                        pid_total += float(val)
                per_pid[pid] = pid_total
                total_mass += pid_total
    except Exception:
        # Fallback: sum element masses directly
        for eid, elem in model.elements.items():
            try:
                m = elem.Mass()
                pid = getattr(elem, 'pid', 0)
                pid = pid if isinstance(pid, int) else 0
                per_pid[pid] = per_pid.get(pid, 0.0) + m
                total_mass += m
            except Exception:
                pass

    # Add CONM2 / CMASS masses
    conm_mass = 0.0
    for mid, mass_elem in model.masses.items():
        try:
            m = mass_elem.Mass()
            conm_mass += m
        except Exception:
            pass
    if conm_mass > 0:
        per_pid[-1] = conm_mass  # Use PID=-1 for lumped masses
        total_mass += conm_mass

    mass_lbs = total_mass * mass_to_lbs_factor
    return total_mass, mass_lbs, per_pid


def coarsen_mesh(
    input_file: str,
    output_file: str,
    target_min: float,
    pids: Optional[List[int]] = None,
    protect_pids: Optional[List[int]] = None,
    max_iterations: int = 0,
    min_quality: float = 0.1,
    max_normal_dev: float = 15.0,
    max_chord_dev: float = 0.0,
    feature_angle: float = 30.0,
    max_warp: float = 10.0,
    do_fill_holes: bool = False,
    punch: bool = False,
    verbose: bool = False,
    mass_to_lbs_factor: float = 386.4,
) -> Dict:
    """
    Coarsen a BDF mesh by collapsing edges shorter than target_min.

    Args:
        input_file: Input BDF path
        output_file: Output BDF path
        target_min: Collapse edges shorter than this
        pids: Only coarsen elements with these PIDs
        protect_pids: Additional PIDs whose nodes are protected
        max_iterations: Max collapses (0 = unlimited)
        min_quality: Minimum element quality (0-1) to allow a collapse
        max_normal_dev: Maximum allowed element normal deviation in degrees
                        from original surface (default: 15.0). Set to 0 to disable.
        max_chord_dev: Maximum allowed centroid departure distance from original
                       surface (default: 0 = auto, same as target_min).
        feature_angle: Dihedral angle threshold in degrees for feature edge
                       detection (default: 30). Nodes on edges where adjacent
                       element normals differ by more than this are protected.
                       Set to 0 to disable.
        max_warp: Maximum allowed quad warp angle in degrees (default: 10).
                  Rejects collapses that would create warped quads.
        do_fill_holes: Detect and collapse interior holes, removing RBE wagon wheels
        punch: Read in punch mode
        verbose: Verbose logging
        mass_to_lbs_factor: Conversion factor from model mass units to lbs
    """
    # Setup logging
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    log_file = os.path.splitext(output_file)[0] + '_coarsen.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    )
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info(f"Reading BDF: {input_file}")
    logger.info(f"Detailed log: {log_file}")

    # Read with xref to get global positions, then uncross-reference
    # so we can manipulate node IDs directly as integers
    model = BDF()
    model.read_bdf(input_file, xref=True, punch=punch)

    # Cache all node positions in global coordinates BEFORE uncross-referencing
    node_positions: Dict[int, np.ndarray] = {}
    for nid, node in model.nodes.items():
        node_positions[nid] = node.get_position().copy()

    # Capture initial mass while model is still cross-referenced
    initial_mass, initial_mass_lbs, initial_mass_by_pid = \
        calculate_total_mass(model, mass_to_lbs_factor)
    logger.info(f"Initial mass: {initial_mass:.6f} (model units) = "
                f"{initial_mass_lbs:.2f} lbs")

    # Un-cross-reference so elem.nodes are integers, not objects
    model.uncross_reference()

    total_nodes = len(model.nodes)
    total_elements = len(model.elements)
    logger.info(f"Initial mesh: {total_nodes} nodes, {total_elements} elements")
    logger.info(f"Target minimum edge length: {target_min}")
    logger.info(f"Minimum element quality: {min_quality}")

    # --- OML preservation: build original surface reference ---
    original_normals, original_centroids, surface_kdtree, _ = \
        build_surface_reference(model, node_positions)

    # Compute cosine threshold from degree input (0 = disabled)
    if max_normal_dev > 0:
        max_normal_dev_cos = np.cos(np.radians(max_normal_dev))
    else:
        max_normal_dev_cos = -1.0  # disabled

    # Auto chord deviation: same as target_min if not explicitly set
    if max_chord_dev <= 0:
        max_chord_dev = target_min

    logger.info(f"Max normal deviation: {max_normal_dev:.1f} deg "
                f"(cos threshold={max_normal_dev_cos:.4f})")
    logger.info(f"Max chord deviation: {max_chord_dev:.4f}")
    logger.info(f"Max quad warp angle: {max_warp:.1f} deg")

    # Fill holes BEFORE coarsening (so collapsed hole nodes don't interfere)
    hole_stats = {}
    if do_fill_holes:
        logger.info("\n--- Hole Detection and Removal ---")
        hole_stats = fill_holes(model, node_positions)

    # Build eligible PID set
    eligible_pids = set(pids) if pids else None
    extra_protect = set(protect_pids) if protect_pids else None

    # Build protected node set
    protected, boundary_nodes = build_protected_nodes(
        model, eligible_pids, extra_protect,
        node_positions=node_positions,
        feature_angle=feature_angle,
    )
    logger.info(f"Protected nodes: {len(protected)}")
    logger.info(f"Boundary nodes: {len(boundary_nodes)} "
                f"({len(boundary_nodes & protected)} protected, "
                f"{len(boundary_nodes - protected)} coarsenable)")

    # Build connectivity maps
    edge_to_elems, node_to_elems = build_edge_list(model, eligible_pids)
    logger.info(f"Edges in eligible elements: {len(edge_to_elems)}")

    # Build FULL node-to-elements map (all elements, not just eligible)
    # This is needed so that when a node is removed, ALL elements referencing
    # it get updated - not just the eligible ones
    all_node_to_elems: Dict[int, Set[int]] = {}
    for eid, elem in model.elements.items():
        for n in elem.nodes:
            nid = n if isinstance(n, int) else n
            if isinstance(nid, int) and nid > 0:
                if nid not in all_node_to_elems:
                    all_node_to_elems[nid] = set()
                all_node_to_elems[nid].add(eid)

    # Build initial edge heap
    heap = build_edge_heap(node_positions, edge_to_elems, target_min)
    logger.info(f"Edges shorter than {target_min}: {len(heap)}")

    if not heap:
        logger.info("No edges to collapse. Writing output unchanged.")
        model.write_bdf(output_file, size=16, is_double=False)
        return {'nodes_removed': 0, 'edges_collapsed': 0}

    # Stats
    stats = {
        'edges_collapsed': 0,
        'nodes_removed': 0,
        'elements_removed': 0,
        'quads_to_tris': 0,
        'skipped_both_protected': 0,
        'skipped_quality': 0,
        'skipped_normal_dev': 0,
        'skipped_chord_dev': 0,
    }

    removed_nodes: Set[int] = set()
    iteration = 0

    # Per-PID area budget: compute initial area per PID and track removals.
    # Reject collapses that would push any PID past 1% total area loss.
    shell_types_area = {'CQUAD4', 'CTRIA3'}
    pid_total_area: Dict[int, float] = {}
    elem_pid_map: Dict[int, int] = {}
    for eid, elem in model.elements.items():
        if elem.type not in shell_types_area:
            continue
        pid = elem.pid if isinstance(elem.pid, int) else elem.pid
        elem_pid_map[eid] = pid
        nodes = [n if isinstance(n, int) else n for n in elem.nodes]
        pos = [node_positions[n] for n in nodes if n in node_positions]
        if len(pos) >= 3:
            a, _ = compute_tri_area_normal(pos[0], pos[1], pos[2])
            if len(pos) == 4:
                a2, _ = compute_tri_area_normal(pos[0], pos[2], pos[3])
                a += a2
            pid_total_area[pid] = pid_total_area.get(pid, 0.0) + a
    pid_area_removed: Dict[int, float] = {pid: 0.0 for pid in pid_total_area}
    max_area_loss_frac = 0.02  # 2% of total PID area

    # Process edges shortest first
    while heap:
        if max_iterations > 0 and iteration >= max_iterations:
            logger.info(f"Reached max iterations ({max_iterations})")
            break

        length, n1, n2 = heapq.heappop(heap)

        # Skip if either node was already removed
        if n1 in removed_nodes or n2 in removed_nodes:
            continue

        # Skip if nodes no longer exist
        if n1 not in node_positions or n2 not in node_positions:
            continue

        # Recompute length (may have changed due to prior collapses)
        current_length = compute_edge_length_cached(node_positions, n1, n2)
        if current_length >= target_min:
            continue

        # Check protection
        n1_protected = n1 in protected
        n2_protected = n2 in protected

        if n1_protected and n2_protected:
            stats['skipped_both_protected'] += 1
            logger.debug(f"  Edge ({n1},{n2}) L={current_length:.4f}: both protected, skip")
            continue

        # Determine which node to keep
        if n1_protected:
            n_keep, n_remove = n1, n2
        elif n2_protected:
            n_keep, n_remove = n2, n1
        else:
            # Neither protected: keep the one with more element connections
            c1 = len(node_to_elems.get(n1, set()))
            c2 = len(node_to_elems.get(n2, set()))
            if c1 >= c2:
                n_keep, n_remove = n1, n2
            else:
                n_keep, n_remove = n2, n1

        # Quality and OML preservation check before collapse
        collapse_result = check_collapse_quality(
            model, node_positions, n_remove, n_keep,
            all_node_to_elems, min_quality,
            original_normals=original_normals,
            original_centroids=original_centroids,
            surface_kdtree=surface_kdtree,
            max_normal_dev_cos=max_normal_dev_cos,
            max_chord_dev=max_chord_dev,
            max_warp=max_warp,
        )
        if collapse_result != "ok":
            stats[f'skipped_{collapse_result}'] += 1
            logger.debug(f"  Edge ({n1},{n2}) L={current_length:.4f}: "
                        f"{collapse_result} check failed, skip")
            continue

        # Per-PID area budget check: only for boundary collapses where
        # element removal causes actual area loss.  Interior collapses
        # redistribute area without losing it.
        involves_boundary = (n_keep in boundary_nodes or n_remove in boundary_nodes)
        area_loss_by_pid: Dict[int, float] = {}

        if involves_boundary:
            affected_for_area = set()
            if n_remove in all_node_to_elems:
                affected_for_area |= all_node_to_elems[n_remove]
            if n_keep in all_node_to_elems:
                affected_for_area |= all_node_to_elems[n_keep]

            for eid in affected_for_area:
                if eid not in model.elements:
                    continue
                elem = model.elements[eid]
                if elem.type not in shell_types_area:
                    continue
                pid = elem_pid_map.get(eid, -1)
                old_nodes_a = [n if isinstance(n, int) else n for n in elem.nodes]
                old_pos = [node_positions[n] for n in old_nodes_a
                           if n in node_positions]
                if len(old_pos) < 3:
                    continue
                a_old, _ = compute_tri_area_normal(
                    old_pos[0], old_pos[1], old_pos[2])
                if len(old_pos) == 4:
                    a2, _ = compute_tri_area_normal(
                        old_pos[0], old_pos[2], old_pos[3])
                    a_old += a2

                new_nds = [n_keep if n == n_remove else n for n in old_nodes_a]
                unique = []
                for n in new_nds:
                    if n not in unique:
                        unique.append(n)
                if len(unique) < 3:
                    area_loss_by_pid[pid] = \
                        area_loss_by_pid.get(pid, 0.0) + a_old
                    continue
                new_pos = [node_positions[n] for n in unique
                           if n in node_positions]
                if len(new_pos) >= 3:
                    a_new, _ = compute_tri_area_normal(
                        new_pos[0], new_pos[1], new_pos[2])
                    if len(new_pos) == 4:
                        a2, _ = compute_tri_area_normal(
                            new_pos[0], new_pos[2], new_pos[3])
                        a_new += a2
                    if a_old > a_new:
                        area_loss_by_pid[pid] = \
                            area_loss_by_pid.get(pid, 0.0) + (a_old - a_new)

            budget_exceeded = False
            for pid, loss in area_loss_by_pid.items():
                total = pid_total_area.get(pid, 0.0)
                if total > 1e-15:
                    cumulative = pid_area_removed.get(pid, 0.0) + loss
                    if cumulative / total > max_area_loss_frac:
                        budget_exceeded = True
                        break

            if budget_exceeded:
                stats['skipped_quality'] += 1
                logger.debug(f"  Edge ({n1},{n2}) L={current_length:.4f}: "
                            f"PID area budget exceeded, skip")
                continue

        # For boundary edges: move surviving node to midpoint to preserve
        # the boundary line geometry and minimize area loss.
        both_boundary = (n_keep in boundary_nodes and n_remove in boundary_nodes)
        if both_boundary and not n1_protected and not n2_protected:
            midpoint = (node_positions[n_keep] + node_positions[n_remove]) / 2.0
            node_positions[n_keep] = midpoint
            if n_keep in model.nodes:
                node = model.nodes[n_keep]
                node.xyz = midpoint.copy()

        # Update area budget after successful collapse
        for pid, loss in area_loss_by_pid.items():
            pid_area_removed[pid] = pid_area_removed.get(pid, 0.0) + loss

        # Perform the collapse
        logger.debug(f"  Collapse ({n_remove} -> {n_keep}) L={current_length:.4f}")
        collapse_edge(model, node_positions, n_remove, n_keep, node_to_elems,
                      edge_to_elems, all_node_to_elems, removed_nodes, stats)

        # Remove n_remove from node_positions
        node_positions.pop(n_remove, None)

        # Add new/updated edges from n_keep to the heap
        for (en1, en2), eids in edge_to_elems.items():
            if n_keep in (en1, en2) and eids:
                if en1 not in removed_nodes and en2 not in removed_nodes:
                    if en1 in node_positions and en2 in node_positions:
                        new_length = compute_edge_length_cached(node_positions, en1, en2)
                        if new_length < target_min:
                            heapq.heappush(heap, (new_length, en1, en2))

        iteration += 1

        # Progress logging every 1000 collapses
        if iteration % 1000 == 0:
            logger.info(f"  ... {iteration} edges collapsed so far")

    # Summary
    final_nodes = len(model.nodes)
    final_elements = len(model.elements)

    logger.info("")
    logger.info("=" * 50)
    logger.info("COARSENING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Nodes: {total_nodes} -> {final_nodes} "
                f"(removed {stats['nodes_removed']})")
    logger.info(f"Elements: {total_elements} -> {final_elements} "
                f"(removed {stats['elements_removed']})")
    logger.info(f"Edges collapsed: {stats['edges_collapsed']}")
    logger.info(f"CQUAD4 -> CTRIA3 conversions: {stats['quads_to_tris']}")
    logger.info(f"Skipped (both protected): {stats['skipped_both_protected']}")
    logger.info(f"Skipped (quality): {stats['skipped_quality']}")
    logger.info(f"Skipped (normal deviation): {stats['skipped_normal_dev']}")
    logger.info(f"Skipped (chord deviation): {stats['skipped_chord_dev']}")
    if hole_stats:
        logger.info(f"Holes collapsed: {hole_stats.get('holes_collapsed', 0)}")
        logger.info(f"RBEs removed: {hole_stats.get('rbes_removed', 0)}")
        logger.info(f"Connections reattached: {hole_stats.get('connections_reattached', 0)}")

    # Clean up RBE3 elements: remove duplicate Gijs entries that result
    # from multiple collapsed nodes mapping to the same survivor.
    for rbe_eid, rbe_elem in model.rigid_elements.items():
        if rbe_elem.type != 'RBE3':
            continue
        gijs = getattr(rbe_elem, 'Gijs', [])
        if not gijs:
            continue
        # Gijs is a list of lists; deduplicate within each weight group
        new_gijs = []
        for gij_group in gijs:
            if isinstance(gij_group, list):
                seen = set()
                deduped = []
                for n in gij_group:
                    if n not in seen:
                        seen.add(n)
                        deduped.append(n)
                new_gijs.append(deduped)
            else:
                new_gijs.append(gij_group)
        rbe_elem.Gijs = new_gijs
        # Update the corresponding weight counts (Gmi_wt maps to Gijs groups)
        # Each Gijs group corresponds to one (alpha, comps, Gij_list) triplet.
        # The pyNastran RBE3 stores: refgrid, refc, weights, comps, Gijs
        # where len(weights) == len(comps) == len(Gijs).

    # Write output in large-field format to preserve coordinate precision
    logger.info(f"\nWriting coarsened BDF: {output_file}")
    model.write_bdf(output_file, size=16, is_double=False)

    # --- Mass conservation check ---
    logger.info("\n--- Mass Check ---")
    try:
        check_model = BDF()
        check_model.read_bdf(output_file, xref=True, punch=punch)
        final_mass, final_mass_lbs, final_mass_by_pid = \
            calculate_total_mass(check_model, mass_to_lbs_factor)
    except Exception as e:
        logger.warning(f"Could not compute final mass: {e}")
        final_mass = 0.0
        final_mass_lbs = 0.0
        final_mass_by_pid = {}

    logger.info(f"Initial mass: {initial_mass:.6f} (model units) = "
                f"{initial_mass_lbs:.2f} lbs")
    logger.info(f"Final mass:   {final_mass:.6f} (model units) = "
                f"{final_mass_lbs:.2f} lbs")

    if initial_mass > 0:
        mass_diff = abs(final_mass - initial_mass)
        mass_diff_pct = mass_diff / initial_mass * 100
        logger.info(f"Difference:   {mass_diff:.6f} ({mass_diff_pct:.4f}%)")
        if mass_diff_pct < 0.01:
            logger.info("[OK] Mass conserved (< 0.01% difference)")
        elif mass_diff_pct < 1.0:
            logger.warning(f"[WARN] Mass changed by {mass_diff_pct:.4f}%")
        else:
            logger.error(f"[FAIL] Mass changed by {mass_diff_pct:.4f}% - "
                        f"check element removal")
    else:
        mass_diff_pct = 0.0
        logger.info("(No structural mass detected)")

    # Per-property mass breakdown
    all_pids = sorted(set(list(initial_mass_by_pid.keys()) +
                          list(final_mass_by_pid.keys())))
    if all_pids:
        logger.info("\nPer-property mass breakdown:")
        logger.info(f"  {'PID':>8s}  {'Initial':>12s}  {'Final':>12s}  "
                    f"{'Diff':>12s}  {'Diff%':>8s}")
        for pid in all_pids:
            m_init = initial_mass_by_pid.get(pid, 0.0)
            m_final = final_mass_by_pid.get(pid, 0.0)
            m_diff = m_final - m_init
            pct = (abs(m_diff) / m_init * 100) if m_init > 0 else 0.0
            label = "masses" if pid == -1 else str(pid)
            logger.info(f"  {label:>8s}  {m_init:12.6f}  {m_final:12.6f}  "
                       f"{m_diff:+12.6f}  {pct:7.3f}%")

    stats['initial_mass'] = initial_mass
    stats['final_mass'] = final_mass
    stats['initial_mass_lbs'] = initial_mass_lbs
    stats['final_mass_lbs'] = final_mass_lbs
    stats['mass_diff_pct'] = mass_diff_pct

    logger.info("\nDone.")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coarsen Nastran BDF mesh by collapsing short edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0
  python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0 --pids 100,101
  python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0 --protect-pids 200
  python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 1.5 --min-quality 0.2
  python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0 --max-normal-dev 10
  python coarsen_mesh.py --in model.bdf --out coarsened.bdf --target-min 2.0 --max-chord-dev 0.5
"""
    )

    parser.add_argument('--in', '-i', dest='input_file', required=True,
                        help='Input BDF file path')
    parser.add_argument('--out', '-o', dest='output_file', required=True,
                        help='Output BDF file path')
    parser.add_argument('--target-min', type=float, required=True,
                        help='Collapse edges shorter than this length')
    parser.add_argument('--pids', '-p', type=str, default=None,
                        help='Only coarsen elements with these PIDs (comma-separated)')
    parser.add_argument('--protect-pids', type=str, default=None,
                        help='Additional PIDs whose nodes are protected (comma-separated)')
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='Max collapse iterations (default: 0 = unlimited)')
    parser.add_argument('--min-quality', type=float, default=0.1,
                        help='Minimum element quality to allow collapse (default: 0.1)')
    parser.add_argument('--max-normal-dev', type=float, default=15.0,
                        help='Max element normal deviation from original surface in degrees '
                             '(default: 15). Set to 0 to disable.')
    parser.add_argument('--max-chord-dev', type=float, default=0.0,
                        help='Max centroid departure from original surface '
                             '(default: 0 = auto, same as target-min). '
                             'Prevents chord shortcutting across curves.')
    parser.add_argument('--feature-angle', type=float, default=30.0,
                        help='Dihedral angle threshold in degrees for protecting '
                             'sharp corners/creases (default: 30). Nodes on edges '
                             'where adjacent normals differ by more than this are '
                             'locked in place. Set to 0 to disable.')
    parser.add_argument('--max-warp', type=float, default=10.0,
                        help='Maximum allowed quad warp angle in degrees '
                             '(default: 10). Rejects collapses that would create '
                             'warped CQUAD4 elements.')
    parser.add_argument('--fill-holes', action='store_true',
                        help='Detect and collapse interior holes, removing RBE wagon wheels')
    parser.add_argument('--mass-factor', type=float, default=386.4,
                        help='Mass conversion factor to lbs (default: 386.4)')
    parser.add_argument('--punch', action='store_true',
                        help='Read BDF in punch mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Parse PID lists
    pids = None
    if args.pids:
        try:
            pids = [int(p.strip()) for p in args.pids.split(',')]
        except ValueError:
            print(f"ERROR: Invalid pids format: {args.pids}")
            sys.exit(1)

    protect_pids = None
    if args.protect_pids:
        try:
            protect_pids = [int(p.strip()) for p in args.protect_pids.split(',')]
        except ValueError:
            print(f"ERROR: Invalid protect-pids format: {args.protect_pids}")
            sys.exit(1)

    try:
        coarsen_mesh(
            input_file=args.input_file,
            output_file=args.output_file,
            target_min=args.target_min,
            pids=pids,
            protect_pids=protect_pids,
            max_iterations=args.max_iterations,
            min_quality=args.min_quality,
            max_normal_dev=args.max_normal_dev,
            max_chord_dev=args.max_chord_dev,
            feature_angle=args.feature_angle,
            max_warp=args.max_warp,
            do_fill_holes=args.fill_holes,
            punch=args.punch,
            verbose=args.verbose,
            mass_to_lbs_factor=args.mass_factor,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
