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
    --max-iterations  Max collapse iterations (default: unlimited)
    --min-quality     Minimum element quality to allow collapse (default: 0.1)
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
) -> Set[int]:
    """
    Build the set of nodes that must NOT be moved or removed.

    Protected categories:
      - Nodes on 1D elements (CBAR, CBEAM, CROD, CONROD, CTUBE)
      - Nodes on rigid elements (RBE2, RBE3, RBAR, RROD)
      - Nodes on mass elements (CONM1, CONM2, CMASS1-4)
      - Nodes referenced by SPCs or MPCs
      - Nodes on property boundaries (shared by elements with different PIDs)
      - Nodes on mesh boundaries (edge used by only one 2D element)
      - Nodes on elements with extra_protect_pids
    """
    protected: Set[int] = set()

    # --- 1D elements ---
    one_d_types = {'CBAR', 'CBEAM', 'CROD', 'CONROD', 'CTUBE'}
    for eid, elem in model.elements.items():
        if elem.type in one_d_types:
            for n in elem.nodes:
                protected.add(n if isinstance(n, int) else n)

    # --- Rigid elements ---
    for eid, elem in model.rigid_elements.items():
        node_ids = getattr(elem, 'node_ids', [])
        if not node_ids:
            node_ids = getattr(elem, 'nodes', [])
        for n in node_ids:
            if isinstance(n, int):
                protected.add(n)
            elif isinstance(n, list):
                for nn in n:
                    if isinstance(nn, int):
                        protected.add(nn)

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

    for (n1, n2), count in edge_count.items():
        if count == 1:
            protected.add(n1)
            protected.add(n2)

    return protected


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


def compute_element_quality(node_positions: Dict[int, np.ndarray], node_ids: List[int]) -> float:
    """
    Compute a simple quality metric for a 2D element.
    Returns a value between 0 (degenerate) and 1 (perfect).
    Uses aspect ratio: min_edge / max_edge.
    """
    if len(node_ids) < 3:
        return 0.0

    positions = [node_positions[n] for n in node_ids]
    n = len(positions)

    edges = []
    for i in range(n):
        edge_len = np.linalg.norm(positions[(i + 1) % n] - positions[i])
        edges.append(edge_len)

    max_edge = max(edges)
    min_edge = min(edges)

    if max_edge < 1e-15:
        return 0.0

    return min_edge / max_edge


def check_collapse_quality(
    model: BDF,
    node_positions: Dict[int, np.ndarray],
    n_remove: int,
    n_keep: int,
    node_to_elems: Dict[int, Set[int]],
    min_quality: float,
) -> bool:
    """
    Check if collapsing n_remove into n_keep would produce acceptable elements.
    Returns True if all resulting elements pass quality checks.
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
            return False

        # CQUAD4 with 3 unique -> will become CTRIA3
        if elem.type == 'CQUAD4' and len(unique_nodes) == 3:
            new_nodes = unique_nodes

        # Check quality of resulting element
        if len(new_nodes) >= 3:
            if not all(n in node_positions for n in new_nodes):
                return False
            quality = compute_element_quality(node_positions, new_nodes)
            if quality < min_quality:
                return False

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
                        return False  # Inverted

    return True


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

    # Remove node from model and position cache
    if n_remove in model.nodes:
        del model.nodes[n_remove]
    removed_nodes.add(n_remove)

    stats['edges_collapsed'] += 1
    stats['nodes_removed'] += 1


# ---------------------------------------------------------------------------
# Main coarsening function
# ---------------------------------------------------------------------------

def coarsen_mesh(
    input_file: str,
    output_file: str,
    target_min: float,
    pids: Optional[List[int]] = None,
    protect_pids: Optional[List[int]] = None,
    max_iterations: int = 0,
    min_quality: float = 0.1,
    punch: bool = False,
    verbose: bool = False,
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
        punch: Read in punch mode
        verbose: Verbose logging
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

    # Un-cross-reference so elem.nodes are integers, not objects
    model.uncross_reference()

    total_nodes = len(model.nodes)
    total_elements = len(model.elements)
    logger.info(f"Initial mesh: {total_nodes} nodes, {total_elements} elements")
    logger.info(f"Target minimum edge length: {target_min}")
    logger.info(f"Minimum element quality: {min_quality}")

    # Build eligible PID set
    eligible_pids = set(pids) if pids else None
    extra_protect = set(protect_pids) if protect_pids else None

    # Build protected node set
    protected = build_protected_nodes(model, eligible_pids, extra_protect)
    logger.info(f"Protected nodes: {len(protected)}")

    # Build connectivity maps
    edge_to_elems, node_to_elems = build_edge_list(model, eligible_pids)
    logger.info(f"Edges in eligible elements: {len(edge_to_elems)}")

    # Build FULL node-to-elements map (all elements, not just eligible)
    # This is needed so that when a node is removed, ALL elements referencing
    # it get updated - not just the eligible ones
    all_node_to_elems: Dict[int, Set[int]] = {}
    for eid, elem in model.elements.items():
        if elem.type in ('CQUAD4', 'CTRIA3', 'CBAR', 'CBEAM', 'CROD',
                          'CONROD', 'CTUBE', 'CQUAD8', 'CTRIA6'):
            for n in elem.nodes:
                nid = n if isinstance(n, int) else n
                if nid not in all_node_to_elems:
                    all_node_to_elems[nid] = set()
                all_node_to_elems[nid].add(eid)

    # Build initial edge heap
    heap = build_edge_heap(node_positions, edge_to_elems, target_min)
    logger.info(f"Edges shorter than {target_min}: {len(heap)}")

    if not heap:
        logger.info("No edges to collapse. Writing output unchanged.")
        model.write_bdf(output_file)
        return {'nodes_removed': 0, 'edges_collapsed': 0}

    # Stats
    stats = {
        'edges_collapsed': 0,
        'nodes_removed': 0,
        'elements_removed': 0,
        'quads_to_tris': 0,
        'skipped_both_protected': 0,
        'skipped_quality': 0,
    }

    removed_nodes: Set[int] = set()
    iteration = 0

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

        # Quality check before collapse
        if not check_collapse_quality(model, node_positions, n_remove, n_keep,
                                      all_node_to_elems, min_quality):
            stats['skipped_quality'] += 1
            logger.debug(f"  Edge ({n1},{n2}) L={current_length:.4f}: "
                        f"quality check failed, skip")
            continue

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

    # Write output (model is already un-cross-referenced)
    logger.info(f"\nWriting coarsened BDF: {output_file}")
    model.write_bdf(output_file)
    logger.info("Done.")

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
            punch=args.punch,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
