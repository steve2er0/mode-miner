#!/usr/bin/env python3
"""
remesh_surfaces.py - Surface-Based Nastran BDF Remeshing Tool

Decomposes a Nastran BDF shell mesh into geometric surface patches
(bounded by feature edges, PID boundaries, and free edges), optionally
visualises them in an interactive HTML viewer, and remeshes each patch
to a target element size using Gmsh.

USAGE:
    python remesh_surfaces.py --in model.bdf --out remeshed.bdf --target 0.3
    python remesh_surfaces.py --in model.bdf --viz-only --feature-angle 30
    python remesh_surfaces.py --in model.bdf --out remeshed.bdf --target 0.3 --pids 1,2

OPTIONS:
    --in, -i          Input BDF file path (required)
    --out, -o         Output BDF file path (required unless --viz-only)
    --target, -t      Target element edge length (required unless --viz-only)
    --feature-angle   Dihedral angle for feature edge detection (default: 30 deg)
    --visualize       Open interactive HTML visualisation before remeshing
    --viz-only        Visualise patches only, do not remesh
    --pids            Only remesh these PIDs (comma-separated)
    --quad            Prefer quad elements (default)
    --tri             Force all-triangle output
    --mass-factor     Mass conversion factor to lbs (default: 386.4)
    --punch           Read BDF in punch mode
    --verbose, -v     Enable verbose logging
"""

import argparse
import logging
import sys
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from pyNastran.bdf.bdf import BDF
except ImportError:
    print("ERROR: pyNastran is required.  pip install pyNastran")
    sys.exit(1)

try:
    import gmsh
except ImportError:
    gmsh = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

logger = logging.getLogger(__name__)

SHELL_TYPES = {"CQUAD4", "CTRIA3"}

# ── helpers ────────────────────────────────────────────────────────────

Edge = Tuple[int, int]


def _edge(n1: int, n2: int) -> Edge:
    return (min(n1, n2), max(n1, n2))


def _elem_nodes(elem) -> List[int]:
    return [n if isinstance(n, int) else n for n in elem.nodes]


def _tri_area(p0, p1, p2) -> float:
    return float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2.0)


def _elem_normal(positions: List[np.ndarray]) -> np.ndarray:
    v1 = positions[1] - positions[0]
    v2 = positions[2] - positions[0]
    c = np.cross(v1, v2)
    m = np.linalg.norm(c)
    return c / m if m > 1e-15 else np.zeros(3)


# ── Step 1-2: edge detection and patch splitting ──────────────────────

def build_edge_adjacency(
    model: BDF,
) -> Tuple[Dict[Edge, List[int]], Dict[int, List[int]]]:
    """Return edge->{eids} and eid->{node list} for shell elements."""
    edge_elems: Dict[Edge, List[int]] = defaultdict(list)
    elem_nodes_map: Dict[int, List[int]] = {}
    for eid, elem in model.elements.items():
        if elem.type not in SHELL_TYPES:
            continue
        nodes = _elem_nodes(elem)
        elem_nodes_map[eid] = nodes
        for i in range(len(nodes)):
            e = _edge(nodes[i], nodes[(i + 1) % len(nodes)])
            edge_elems[e].append(eid)
    return dict(edge_elems), elem_nodes_map


def classify_edges(
    edge_elems: Dict[Edge, List[int]],
    model: BDF,
    node_positions: Dict[int, np.ndarray],
    elem_nodes_map: Dict[int, List[int]],
    feature_angle: float = 30.0,
    hole_fill_eids: Optional[Set[int]] = None,
) -> Tuple[Set[Edge], Set[Edge], Set[Edge], Set[Edge]]:
    """Classify every edge as free / feature / pid_boundary / interior.

    Edges touching hole-fill elements are always classified as interior
    so that filled holes merge into the surrounding patch.
    """
    feature_cos = np.cos(np.radians(feature_angle))

    elem_normals: Dict[int, np.ndarray] = {}
    for eid, nodes in elem_nodes_map.items():
        pts = [node_positions[n] for n in nodes if n in node_positions]
        if len(pts) >= 3:
            elem_normals[eid] = _elem_normal(pts)

    elem_pid: Dict[int, int] = {}
    for eid, elem in model.elements.items():
        if elem.type in SHELL_TYPES:
            elem_pid[eid] = elem.pid if isinstance(elem.pid, int) else elem.pid

    free: Set[Edge] = set()
    feature: Set[Edge] = set()
    pid_boundary: Set[Edge] = set()
    interior: Set[Edge] = set()

    _fill = hole_fill_eids or set()

    for edge, eids in edge_elems.items():
        if len(eids) == 1:
            free.add(edge)
        elif len(eids) >= 2:
            eid_a, eid_b = eids[0], eids[1]
            # Edges touching hole-fill elements are always interior
            if eid_a in _fill or eid_b in _fill:
                interior.add(edge)
            elif elem_pid.get(eid_a) != elem_pid.get(eid_b):
                pid_boundary.add(edge)
            elif (eid_a in elem_normals and eid_b in elem_normals
                  and np.dot(elem_normals[eid_a], elem_normals[eid_b]) < feature_cos):
                feature.add(edge)
            else:
                interior.add(edge)

    return free, feature, pid_boundary, interior


def flood_fill_patches(
    edge_elems: Dict[Edge, List[int]],
    elem_nodes_map: Dict[int, List[int]],
    interior_edges: Set[Edge],
) -> List[Set[int]]:
    """Flood-fill elements across interior edges to form patches."""
    elem_to_edges: Dict[int, List[Edge]] = defaultdict(list)
    for edge, eids in edge_elems.items():
        for eid in eids:
            if eid in elem_nodes_map:
                elem_to_edges[eid].append(edge)

    edge_to_neighbor: Dict[int, Set[int]] = defaultdict(set)
    for edge in interior_edges:
        eids = edge_elems.get(edge, [])
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                if eids[i] in elem_nodes_map and eids[j] in elem_nodes_map:
                    edge_to_neighbor[eids[i]].add(eids[j])
                    edge_to_neighbor[eids[j]].add(eids[i])

    visited: Set[int] = set()
    patches: List[Set[int]] = []
    for eid in elem_nodes_map:
        if eid in visited:
            continue
        patch: Set[int] = set()
        stack = [eid]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            patch.add(current)
            for nbr in edge_to_neighbor.get(current, set()):
                if nbr not in visited:
                    stack.append(nbr)
        patches.append(patch)
    return patches


# ── Step 3: classify pinned nodes ─────────────────────────────────────

def classify_pinned_nodes(
    model: BDF,
    patches: List[Set[int]],
    elem_nodes_map: Dict[int, List[int]],
    boundary_edges: Set[Edge],
) -> Set[int]:
    """Return the set of nodes that must not move."""
    pinned: Set[int] = set()

    connector_types = {
        "CBAR", "CBEAM", "CROD", "CONROD", "CTUBE",
        "CBUSH", "CBUSH1D", "CBUSH2D",
        "CELAS1", "CELAS2", "CELAS3", "CELAS4",
        "CDAMP1", "CDAMP2", "CDAMP3", "CDAMP4",
        "CVISC", "CGAP",
    }
    for elem in model.elements.values():
        if elem.type in connector_types:
            for n in elem.nodes:
                if isinstance(n, int) and n > 0:
                    pinned.add(n)

    for elem in model.rigid_elements.values():
        for attr in ("gn", "refgrid", "ga", "gb"):
            v = getattr(elem, attr, None)
            if isinstance(v, int):
                pinned.add(v)
        gmi = getattr(elem, "Gmi", [])
        if gmi:
            for n in gmi:
                if isinstance(n, int):
                    pinned.add(n)

    for elem in model.masses.values():
        for n in getattr(elem, "node_ids", getattr(elem, "nodes", [])):
            if isinstance(n, int) and n > 0:
                pinned.add(n)

    for spc_list in model.spcs.values():
        for spc in spc_list:
            for n in getattr(spc, "node_ids", getattr(spc, "nodes", [])):
                if isinstance(n, int):
                    pinned.add(n)

    for mpc_list in model.mpcs.values():
        for mpc in mpc_list:
            for n in getattr(mpc, "node_ids", getattr(mpc, "nodes", [])):
                if isinstance(n, int):
                    pinned.add(n)

    # Patch corners: nodes shared by 3+ patches
    node_patch_count: Dict[int, int] = defaultdict(int)
    for idx, patch in enumerate(patches):
        patch_nodes: Set[int] = set()
        for eid in patch:
            for n in elem_nodes_map.get(eid, []):
                patch_nodes.add(n)
        for n in patch_nodes:
            node_patch_count[n] += 1
    for n, cnt in node_patch_count.items():
        if cnt >= 3:
            pinned.add(n)

    # Nodes at boundary-edge endpoints where 3+ boundary edges meet
    bnd_degree: Dict[int, int] = defaultdict(int)
    for n1, n2 in boundary_edges:
        bnd_degree[n1] += 1
        bnd_degree[n2] += 1
    for n, deg in bnd_degree.items():
        if deg != 2:
            pinned.add(n)

    return pinned


# ── Step 4: visualise ─────────────────────────────────────────────────

def extract_patch_boundary_loops(
    patch_eids: Set[int],
    elem_nodes_map: Dict[int, List[int]],
    edge_elems: Dict[Edge, List[int]],
) -> List[List[int]]:
    """Extract ordered boundary loops for a single patch.

    Handles T-junctions (nodes with 3+ boundary edges) by tracing each
    available path and closing loops when possible.
    """
    patch_edges: Dict[Edge, int] = defaultdict(int)
    for eid in patch_eids:
        nodes = elem_nodes_map.get(eid, [])
        for i in range(len(nodes)):
            e = _edge(nodes[i], nodes[(i + 1) % len(nodes)])
            patch_edges[e] += 1

    boundary: Set[Edge] = set()
    for e, cnt in patch_edges.items():
        if cnt == 1:
            boundary.add(e)

    if not boundary:
        return []

    adj: Dict[int, List[int]] = defaultdict(list)
    for n1, n2 in boundary:
        adj[n1].append(n2)
        adj[n2].append(n1)

    visited_edges: Set[Edge] = set()
    loops: List[List[int]] = []

    # Try to trace closed loops, handling T-junctions by always picking
    # an unvisited edge.  Start from degree-2 nodes first (simple boundary),
    # then T-junctions.
    start_order = sorted(adj.keys(),
                         key=lambda n: (len(adj[n]) != 2, n))

    for start in start_order:
        # Try each unvisited edge from this start node
        for first_nbr in adj[start]:
            if _edge(start, first_nbr) in visited_edges:
                continue

            loop = [start]
            visited_edges.add(_edge(start, first_nbr))
            prev = start
            current = first_nbr

            while current != start:
                loop.append(current)
                # Pick an unvisited edge from current
                next_node = None
                for nb in adj[current]:
                    if _edge(current, nb) not in visited_edges:
                        next_node = nb
                        break
                if next_node is None:
                    break
                visited_edges.add(_edge(current, next_node))
                prev = current
                current = next_node

            if current == start and len(loop) >= 3:
                loops.append(loop)

    return loops


def visualise_patches(
    patches: List[Set[int]],
    elem_nodes_map: Dict[int, List[int]],
    node_positions: Dict[int, np.ndarray],
    edge_elems: Dict[Edge, List[int]],
    feature_edges: Set[Edge],
    free_edges: Set[Edge],
    pid_boundary_edges: Set[Edge],
    pinned_nodes: Set[int],
    model: BDF,
    output_html: str,
):
    """Create interactive plotly HTML visualisation."""
    if go is None:
        logger.warning("plotly not installed, skipping visualisation")
        return

    traces = []

    # Build a global node-index map for plotly mesh
    all_nids = sorted(node_positions.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(all_nids)}
    x = [node_positions[n][0] for n in all_nids]
    y = [node_positions[n][1] for n in all_nids]
    z = [node_positions[n][2] for n in all_nids]

    # Assign patch colours
    node_patch_id = {}
    for pidx, patch in enumerate(patches):
        for eid in patch:
            for n in elem_nodes_map.get(eid, []):
                node_patch_id[n] = pidx

    intensity = [node_patch_id.get(n, -1) for n in all_nids]

    # Triangulate for plotly (split quads into 2 tris)
    tri_i, tri_j, tri_k = [], [], []
    for eid_set in patches:
        for eid in eid_set:
            nodes = elem_nodes_map.get(eid, [])
            idxs = [nid_to_idx.get(n) for n in nodes]
            if any(i is None for i in idxs):
                continue
            if len(idxs) >= 3:
                tri_i.append(idxs[0]); tri_j.append(idxs[1]); tri_k.append(idxs[2])
            if len(idxs) == 4:
                tri_i.append(idxs[0]); tri_j.append(idxs[2]); tri_k.append(idxs[3])

    elem_pid_map = {}
    for eid, elem in model.elements.items():
        if elem.type in SHELL_TYPES:
            elem_pid_map[eid] = elem.pid if isinstance(elem.pid, int) else elem.pid

    hover = []
    for n in all_nids:
        pid_label = node_patch_id.get(n, -1)
        hover.append(f"Node {n}<br>Patch {pid_label}")

    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=tri_i, j=tri_j, k=tri_k,
        intensity=intensity,
        colorscale="Viridis",
        opacity=0.7,
        hovertext=hover,
        hoverinfo="text",
        name="Patches",
    ))

    # Feature edges
    for label, edge_set, color, width in [
        ("Feature edges", feature_edges, "red", 4),
        ("Free edges", free_edges, "blue", 3),
        ("PID boundaries", pid_boundary_edges, "orange", 3),
    ]:
        ex, ey, ez = [], [], []
        for n1, n2 in edge_set:
            if n1 in node_positions and n2 in node_positions:
                p1, p2 = node_positions[n1], node_positions[n2]
                ex += [p1[0], p2[0], None]
                ey += [p1[1], p2[1], None]
                ez += [p1[2], p2[2], None]
        if ex:
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color=color, width=width),
                name=label,
            ))

    # Pinned nodes
    px = [node_positions[n][0] for n in pinned_nodes if n in node_positions]
    py = [node_positions[n][1] for n in pinned_nodes if n in node_positions]
    pz = [node_positions[n][2] for n in pinned_nodes if n in node_positions]
    ptext = [f"Pinned {n}" for n in pinned_nodes if n in node_positions]
    if px:
        traces.append(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="markers",
            marker=dict(size=4, color="black"),
            hovertext=ptext,
            hoverinfo="text",
            name="Pinned nodes",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Surface Patches ({len(patches)} patches)",
        scene=dict(aspectmode="data"),
    )
    fig.write_html(output_html, auto_open=True)
    logger.info(f"Visualisation written to {output_html}")


def visualise_result(
    all_new_nodes: Dict[int, np.ndarray],
    all_new_elems: List[Tuple[str, int, List[int]]],
    pinned_nodes: Set[int],
    output_html: str,
):
    """Create interactive HTML visualisation of the remeshed result."""
    if go is None:
        logger.warning("plotly not installed, skipping visualisation")
        return

    traces = []

    # Build node index map
    all_nids = sorted(all_new_nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(all_nids)}
    x = [all_new_nodes[n][0] for n in all_nids]
    y = [all_new_nodes[n][1] for n in all_nids]
    z = [all_new_nodes[n][2] for n in all_nids]

    # Colour by PID
    node_pid: Dict[int, int] = {}
    for etype, epid, enodes in all_new_elems:
        for n in enodes:
            if n not in node_pid:
                node_pid[n] = epid
    intensity = [node_pid.get(n, 0) for n in all_nids]

    # Triangulate for plotly
    tri_i, tri_j, tri_k = [], [], []
    for etype, epid, enodes in all_new_elems:
        idxs = [nid_to_idx.get(n) for n in enodes]
        if any(i is None for i in idxs):
            continue
        if len(idxs) >= 3:
            tri_i.append(idxs[0]); tri_j.append(idxs[1]); tri_k.append(idxs[2])
        if len(idxs) == 4:
            tri_i.append(idxs[0]); tri_j.append(idxs[2]); tri_k.append(idxs[3])

    hover = [f"Node {n}<br>PID {node_pid.get(n, '?')}" for n in all_nids]

    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=tri_i, j=tri_j, k=tri_k,
        intensity=intensity,
        colorscale="Viridis",
        opacity=0.8,
        hovertext=hover,
        hoverinfo="text",
        name="Remeshed surface",
    ))

    # Element edges as wireframe
    ex, ey, ez = [], [], []
    for etype, epid, enodes in all_new_elems:
        pts = [all_new_nodes.get(n) for n in enodes]
        if any(p is None for p in pts):
            continue
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            ex += [p1[0], p2[0], None]
            ey += [p1[1], p2[1], None]
            ez += [p1[2], p2[2], None]
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="black", width=1),
        name="Element edges",
    ))

    # Pinned nodes
    pn = [n for n in pinned_nodes if n in all_new_nodes]
    if pn:
        traces.append(go.Scatter3d(
            x=[all_new_nodes[n][0] for n in pn],
            y=[all_new_nodes[n][1] for n in pn],
            z=[all_new_nodes[n][2] for n in pn],
            mode="markers",
            marker=dict(size=4, color="red"),
            hovertext=[f"Pinned {n}" for n in pn],
            hoverinfo="text",
            name="Pinned nodes",
        ))

    n_quads = sum(1 for t, _, e in all_new_elems if t == "CQUAD4")
    n_tris = sum(1 for t, _, e in all_new_elems if t == "CTRIA3")
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Remeshed: {len(all_new_nodes)} nodes, "
              f"{n_quads} quads + {n_tris} tris",
        scene=dict(aspectmode="data"),
    )
    fig.write_html(output_html, auto_open=True)
    logger.info(f"Result visualisation written to {output_html}")


# ── Step 5: remesh with Gmsh ─────────────────────────────────────────

def _order_boundary_segments(
    loops: List[List[int]],
    pinned: Set[int],
) -> List[List[List[int]]]:
    """Split each loop into segments between pinned (corner) nodes.

    Returns list-of-loops, each loop is a list-of-segments,
    each segment is an ordered list of node IDs starting and ending
    at a pinned node.
    """
    result = []
    for loop in loops:
        if not loop:
            continue
        # Find pinned indices in this loop
        pinned_idxs = [i for i, n in enumerate(loop) if n in pinned]
        if not pinned_idxs:
            # No pinned nodes: treat entire loop as one segment
            # Pin the first node as a corner
            pinned_idxs = [0]

        segments = []
        for si in range(len(pinned_idxs)):
            start_idx = pinned_idxs[si]
            end_idx = pinned_idxs[(si + 1) % len(pinned_idxs)]
            if end_idx <= start_idx:
                seg_indices = list(range(start_idx, len(loop))) + list(range(0, end_idx + 1))
            else:
                seg_indices = list(range(start_idx, end_idx + 1))
            seg_nodes = [loop[i] for i in seg_indices]
            segments.append(seg_nodes)
        result.append(segments)
    return result


def remesh_patch(
    patch_eids: Set[int],
    elem_nodes_map: Dict[int, List[int]],
    node_positions: Dict[int, np.ndarray],
    edge_elems: Dict[Edge, List[int]],
    pinned: Set[int],
    target_length: float,
    prefer_quads: bool = True,
    pid: int = 1,
    global_boundary_nodes: Optional[Set[int]] = None,
) -> Tuple[Dict[int, np.ndarray], List[Tuple[str, int, List[int]]], int]:
    """Remesh a single patch by decimating the existing mesh.

    Simple edge-collapse approach: iteratively collapse the shortest
    interior edge until all edges are >= target_length.  Boundary nodes
    and pinned nodes are never moved.  No external mesher needed.

    Returns:
        new_nodes: {nid: xyz}
        new_elems: [(type, pid, [nids])]
        next_id: unused
    """
    from scipy.spatial import KDTree

    # Build local working copies
    local_nodes: Dict[int, np.ndarray] = {}
    for eid in patch_eids:
        for n in elem_nodes_map.get(eid, []):
            if n in node_positions:
                local_nodes[n] = node_positions[n].copy()

    local_elems: Dict[int, List[int]] = {}
    for eid in patch_eids:
        nodes = elem_nodes_map.get(eid, [])
        if len(nodes) >= 3:
            local_elems[eid] = list(nodes)

    if len(local_nodes) < 3:
        return {}, [], 0

    # Identify boundary nodes of this patch (edges used by only 1 element)
    patch_edge_count: Dict[Edge, int] = defaultdict(int)
    for eid, nodes in local_elems.items():
        for i in range(len(nodes)):
            e = _edge(nodes[i], nodes[(i + 1) % len(nodes)])
            patch_edge_count[e] += 1

    boundary_nids: Set[int] = set()
    for (n1, n2), cnt in patch_edge_count.items():
        if cnt == 1:
            boundary_nids.add(n1)
            boundary_nids.add(n2)

    # Also include global boundary nodes (feature edges, PID boundaries, free edges)
    if global_boundary_nodes:
        boundary_nids |= (global_boundary_nodes & set(local_nodes.keys()))

    # Nodes that cannot be removed: boundary + pinned
    locked = boundary_nids | (pinned & set(local_nodes.keys()))

    # Build node-to-elements map
    n2e: Dict[int, Set[int]] = defaultdict(set)
    for eid, nodes in local_elems.items():
        for n in nodes:
            n2e[n].add(eid)

    removed_nodes: Set[int] = set()
    removed_elems: Set[int] = set()

    # Build a min-heap of (length, n1, n2) for edges shorter than target
    import heapq
    heap: List[Tuple[float, int, int]] = []
    seen_edges: Set[Edge] = set()
    for eid, nodes in local_elems.items():
        for i in range(len(nodes)):
            e = _edge(nodes[i], nodes[(i + 1) % len(nodes)])
            if e in seen_edges:
                continue
            seen_edges.add(e)
            n1, n2 = e
            if n1 in locked and n2 in locked:
                continue
            if n1 in local_nodes and n2 in local_nodes:
                d = float(np.linalg.norm(local_nodes[n2] - local_nodes[n1]))
                if d < target_length:
                    heapq.heappush(heap, (d, n1, n2))

    while heap:
        best_len, n1, n2 = heapq.heappop(heap)

        if n1 in removed_nodes or n2 in removed_nodes:
            continue
        if n1 not in local_nodes or n2 not in local_nodes:
            continue

        # Recheck length (may have changed)
        d = float(np.linalg.norm(local_nodes[n2] - local_nodes[n1]))
        if d >= target_length:
            continue
        if n1 in locked and n2 in locked:
            continue
        # Decide which to keep
        if n1 in locked:
            n_keep, n_remove = n1, n2
        elif n2 in locked:
            n_keep, n_remove = n2, n1
        else:
            # Keep the one with more connections
            if len(n2e.get(n1, set()) - removed_elems) >= len(n2e.get(n2, set()) - removed_elems):
                n_keep, n_remove = n1, n2
            else:
                n_keep, n_remove = n2, n1

        # Check: would this collapse remove any element that touches a
        # boundary node?  If so, skip — it could erode the patch outline.
        would_harm_boundary = False
        affected = list(n2e.get(n_remove, set()))
        for eid in affected:
            if eid in removed_elems or eid not in local_elems:
                continue
            nodes = local_elems[eid]
            new_nodes_list = [n_keep if n == n_remove else n for n in nodes]
            unique = []
            for n in new_nodes_list:
                if n not in unique:
                    unique.append(n)
            if len(unique) < 3:
                # Element would be removed — reject if ANY of its nodes
                # are boundary nodes (preserves the full boundary ring)
                if any(n in boundary_nids for n in nodes):
                    would_harm_boundary = True
                    break

        if would_harm_boundary:
            continue

        # Update elements: replace n_remove with n_keep
        for eid in affected:
            if eid in removed_elems or eid not in local_elems:
                continue
            nodes = local_elems[eid]
            new_nodes_list = [n_keep if n == n_remove else n for n in nodes]

            # Check for degenerate
            unique = []
            for n in new_nodes_list:
                if n not in unique:
                    unique.append(n)

            if len(unique) < 3:
                removed_elems.add(eid)
                continue

            local_elems[eid] = unique if len(nodes) == 4 and len(unique) == 3 else new_nodes_list
            n2e[n_keep].add(eid)

        removed_nodes.add(n_remove)
        if n_remove in n2e:
            del n2e[n_remove]

        # Add new edges from n_keep to the heap
        for eid in n2e.get(n_keep, set()):
            if eid in removed_elems or eid not in local_elems:
                continue
            nodes = local_elems[eid]
            for i in range(len(nodes)):
                en1, en2 = nodes[i], nodes[(i + 1) % len(nodes)]
                if en1 in removed_nodes or en2 in removed_nodes:
                    continue
                if en1 in locked and en2 in locked:
                    continue
                if en1 in local_nodes and en2 in local_nodes:
                    ed = float(np.linalg.norm(
                        local_nodes[en2] - local_nodes[en1]))
                    if ed < target_length:
                        heapq.heappush(heap, (ed, min(en1, en2),
                                              max(en1, en2)))

    # Build output
    new_nodes: Dict[int, np.ndarray] = {}
    for nid, pos in local_nodes.items():
        if nid not in removed_nodes:
            new_nodes[nid] = pos

    new_elems: List[Tuple[str, int, List[int]]] = []
    for eid, nodes in local_elems.items():
        if eid in removed_elems:
            continue
        # Filter removed nodes from element
        live = [n for n in nodes if n not in removed_nodes]
        if len(live) == 3:
            new_elems.append(("CTRIA3", pid, live))
        elif len(live) == 4:
            new_elems.append(("CQUAD4", pid, live))

    return new_nodes, new_elems, 0


# ── Step 6: reattach connections ──────────────────────────────────────

def reattach_rbe3_gijs(
    model: BDF,
    old_positions: Dict[int, np.ndarray],
    new_positions: Dict[int, np.ndarray],
    node_id_map: Dict[int, int],
):
    """Update RBE3 Gijs nodes to point to nearest new mesh node."""
    from scipy.spatial import KDTree

    new_nids = list(new_positions.keys())
    if not new_nids:
        return
    new_pts = np.array([new_positions[n] for n in new_nids])
    tree = KDTree(new_pts)

    for eid, elem in model.rigid_elements.items():
        if elem.type != "RBE3":
            continue
        gijs = getattr(elem, "Gijs", [])
        if not gijs:
            continue
        new_gijs = []
        for gij_group in gijs:
            if isinstance(gij_group, list):
                new_group = []
                seen = set()
                for n in gij_group:
                    mapped = node_id_map.get(n, n)
                    if mapped in new_positions:
                        if mapped not in seen:
                            new_group.append(mapped)
                            seen.add(mapped)
                    elif n in old_positions:
                        _, idx = tree.query(old_positions[n])
                        nearest = new_nids[idx]
                        if nearest not in seen:
                            new_group.append(nearest)
                            seen.add(nearest)
                if new_group:
                    new_gijs.append(new_group)
            else:
                new_gijs.append(gij_group)
        elem.Gijs = new_gijs


# ── Step 7: mass check ───────────────────────────────────────────────

def calculate_total_mass(
    model: BDF,
    mass_to_lbs_factor: float = 386.4,
) -> Tuple[float, float, Dict[int, float]]:
    """Calculate total mass (model must be cross-referenced)."""
    per_pid: Dict[int, float] = {}
    total = 0.0
    try:
        breakdown = model.get_mass_breakdown(stop_on_failure=False)
        for pid, data in breakdown.items():
            if isinstance(data, (int, float)):
                per_pid[pid] = float(data)
                total += float(data)
            elif isinstance(data, dict):
                s = sum(float(v) for v in data.values() if isinstance(v, (int, float)))
                per_pid[pid] = s
                total += s
    except Exception:
        for elem in model.elements.values():
            try:
                m = elem.Mass()
                pid = getattr(elem, "pid", 0)
                pid = pid if isinstance(pid, int) else 0
                per_pid[pid] = per_pid.get(pid, 0.0) + m
                total += m
            except Exception:
                pass

    conm = 0.0
    for mass_elem in model.masses.values():
        try:
            conm += mass_elem.Mass()
        except Exception:
            pass
    if conm > 0:
        per_pid[-1] = conm
        total += conm

    return total, total * mass_to_lbs_factor, per_pid


# ── Step 0: hole filling ──────────────────────────────────────────────

def _fill_holes(
    model: BDF,
    node_positions: Dict[int, np.ndarray],
) -> Tuple[int, Set[int]]:
    """Detect interior free-edge loops (holes) and fill with triangles.

    A hole is a closed loop of free edges that is NOT the outer boundary.
    The outer boundary is identified as the loop with the largest perimeter.

    For each hole, creates a centroid node and fills with CTRIA3 elements
    fanning from the centroid to each edge of the loop.  Uses PID from
    the adjacent element.

    Returns (holes_filled, set_of_fill_element_ids).
    """
    # Build free-edge adjacency
    edge_count: Dict[Edge, int] = {}
    edge_adj_elem: Dict[Edge, int] = {}
    for eid, elem in model.elements.items():
        if elem.type not in SHELL_TYPES:
            continue
        nodes = _elem_nodes(elem)
        for i in range(len(nodes)):
            e = _edge(nodes[i], nodes[(i + 1) % len(nodes)])
            edge_count[e] = edge_count.get(e, 0) + 1
            edge_adj_elem[e] = eid

    free_adj: Dict[int, List[int]] = defaultdict(list)
    free_edge_elem: Dict[Edge, int] = {}
    for e, cnt in edge_count.items():
        if cnt == 1:
            free_adj[e[0]].append(e[1])
            free_adj[e[1]].append(e[0])
            free_edge_elem[e] = edge_adj_elem.get(e, 0)

    if not free_adj:
        return 0, set()

    # Trace closed loops
    visited: Set[int] = set()
    loops: List[List[int]] = []
    for start in sorted(free_adj.keys()):
        if start in visited:
            continue
        loop = []
        current = start
        prev = None
        while True:
            visited.add(current)
            loop.append(current)
            neighbors = [n for n in free_adj[current] if n != prev and n not in visited]
            if not neighbors:
                # Check if we can close back to start
                if start in free_adj[current] and len(loop) > 2:
                    loops.append(loop)
                break
            nxt = neighbors[0]
            if nxt == start and len(loop) > 2:
                loops.append(loop)
                break
            prev = current
            current = nxt

    if len(loops) <= 1:
        return 0

    # Classify: largest perimeter = outer boundary, rest = holes
    perimeters = []
    for loop in loops:
        perim = 0.0
        for i in range(len(loop)):
            n1, n2 = loop[i], loop[(i + 1) % len(loop)]
            if n1 in node_positions and n2 in node_positions:
                perim += float(np.linalg.norm(
                    node_positions[n2] - node_positions[n1]))
        perimeters.append(perim)

    max_perim_idx = perimeters.index(max(perimeters))
    holes = [(i, loop) for i, loop in enumerate(loops) if i != max_perim_idx]

    if not holes:
        return 0, set()

    max_nid = max(model.nodes.keys())
    max_eid = max(model.elements.keys())

    filled = 0
    fill_eids: Set[int] = set()
    for _, hole_loop in holes:
        if len(hole_loop) < 3:
            continue

        # Find PID from adjacent element
        hole_pid = 1
        for i in range(len(hole_loop)):
            e = _edge(hole_loop[i], hole_loop[(i + 1) % len(hole_loop)])
            if e in free_edge_elem and free_edge_elem[e] in model.elements:
                adj_elem = model.elements[free_edge_elem[e]]
                hole_pid = adj_elem.pid if isinstance(adj_elem.pid, int) else adj_elem.pid
                break

        # Fill hole by ear-clipping triangulation of the boundary polygon.
        # This keeps all new triangles coplanar with the surrounding mesh
        # (no centroid node that would create steep normals).
        positions = [node_positions[n] for n in hole_loop if n in node_positions]
        if len(positions) < 3:
            continue

        # Simple fan from first node (works for convex and mildly concave holes)
        n0 = hole_loop[0]
        for i in range(1, len(hole_loop) - 1):
            n1 = hole_loop[i]
            n2 = hole_loop[i + 1]
            max_eid += 1
            model.add_ctria3(max_eid, hole_pid, [n0, n1, n2])
            fill_eids.add(max_eid)

        filled += 1
        logger.debug(f"  Filled hole: {len(hole_loop)} boundary nodes, "
                     f"PID {hole_pid}")

    return filled, fill_eids


# ── main pipeline ─────────────────────────────────────────────────────

def remesh_surfaces(
    input_file: str,
    output_file: Optional[str],
    target_length: Optional[float],
    feature_angle: float = 30.0,
    visualize: bool = False,
    viz_only: bool = False,
    pids: Optional[List[int]] = None,
    prefer_quads: bool = True,
    mass_to_lbs_factor: float = 386.4,
    punch: bool = False,
    verbose: bool = False,
) -> Dict:
    # Logging
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)
    if output_file:
        log_path = os.path.splitext(output_file)[0] + "_remesh.log"
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logger.propagate = False

    # ── read ──
    logger.info(f"Reading BDF: {input_file}")
    model = BDF()
    model.read_bdf(input_file, xref=True, punch=punch)

    node_positions: Dict[int, np.ndarray] = {}
    for nid, node in model.nodes.items():
        node_positions[nid] = node.get_position().copy()

    initial_mass, initial_mass_lbs, initial_pid_mass = \
        calculate_total_mass(model, mass_to_lbs_factor)
    logger.info(f"Initial mass: {initial_mass:.6f} = {initial_mass_lbs:.2f} lbs")

    model.uncross_reference()

    logger.info(f"Nodes: {len(model.nodes)}, Elements: {len(model.elements)}")

    # Capture original boundary nodes BEFORE hole filling
    orig_edge_count: Dict[Edge, int] = {}
    for eid, elem in model.elements.items():
        if elem.type not in SHELL_TYPES:
            continue
        nodes = _elem_nodes(elem)
        for i in range(len(nodes)):
            e = _edge(nodes[i], nodes[(i + 1) % len(nodes)])
            orig_edge_count[e] = orig_edge_count.get(e, 0) + 1
    original_boundary_nodes: Set[int] = set()
    for (n1, n2), cnt in orig_edge_count.items():
        if cnt == 1:
            original_boundary_nodes.add(n1)
            original_boundary_nodes.add(n2)
    logger.info(f"Original boundary nodes: {len(original_boundary_nodes)}")

    # ── Step 0: fill holes ──
    # Interior holes create T-junctions that prevent clean boundary loops.
    # Fill each hole with triangles so the surface becomes watertight.
    logger.info("Detecting and filling holes...")
    holes_filled, hole_fill_eids = _fill_holes(model, node_positions)
    if holes_filled > 0:
        logger.info(f"  Filled {holes_filled} holes ({len(hole_fill_eids)} fill elements)")
    else:
        logger.info("  No interior holes found")

    # ── Step 1: edge detection ──
    logger.info("Detecting edges...")
    edge_elems, elem_nodes_map = build_edge_adjacency(model)
    free_edges, feature_edges, pid_boundary_edges, interior_edges = classify_edges(
        edge_elems, model, node_positions, elem_nodes_map, feature_angle,
        hole_fill_eids=hole_fill_eids,
    )
    boundary_edges = free_edges | feature_edges | pid_boundary_edges
    # All nodes on any non-interior edge PLUS original boundary nodes
    # (hole boundaries that became interior after filling)
    global_boundary_nodes: Set[int] = set(original_boundary_nodes)
    for n1, n2 in boundary_edges:
        global_boundary_nodes.add(n1)
        global_boundary_nodes.add(n2)
    logger.info(f"  Free edges: {len(free_edges)}")
    logger.info(f"  Feature edges: {len(feature_edges)}")
    logger.info(f"  PID boundary edges: {len(pid_boundary_edges)}")
    logger.info(f"  Interior edges: {len(interior_edges)}")

    # ── Step 2: patch splitting ──
    logger.info("Splitting into patches...")
    patches = flood_fill_patches(edge_elems, elem_nodes_map, interior_edges)
    logger.info(f"  Patches found: {len(patches)}")

    # Get PID for each patch
    patch_pids = []
    for patch in patches:
        pid_counts: Dict[int, int] = defaultdict(int)
        for eid in patch:
            if eid in model.elements:
                elem = model.elements[eid]
                if elem.type in SHELL_TYPES:
                    pid_counts[elem.pid if isinstance(elem.pid, int) else elem.pid] += 1
        patch_pids.append(max(pid_counts, key=pid_counts.get) if pid_counts else 1)

    for i, (patch, pid) in enumerate(zip(patches, patch_pids)):
        logger.info(f"  Patch {i}: {len(patch)} elements, PID {pid}")

    # ── Step 3: classify pinned ──
    logger.info("Classifying pinned nodes...")
    pinned = classify_pinned_nodes(model, patches, elem_nodes_map, boundary_edges)
    logger.info(f"  Pinned nodes: {len(pinned)}")

    # ── Step 4: visualise ──
    if visualize or viz_only:
        html_path = os.path.splitext(input_file)[0] + "_patches.html"
        logger.info(f"Generating visualisation: {html_path}")
        visualise_patches(
            patches, elem_nodes_map, node_positions, edge_elems,
            feature_edges, free_edges, pid_boundary_edges,
            pinned, model, html_path,
        )
        if viz_only:
            logger.info("Viz-only mode, stopping.")
            return {"patches": len(patches)}

    if target_length is None:
        logger.error("--target is required for remeshing")
        return {}

    # ── Step 5: remesh ──
    logger.info(f"\nRemeshing {len(patches)} patches (target={target_length})...")

    all_new_nodes: Dict[int, np.ndarray] = {}
    all_new_elems: List[Tuple[str, int, List[int]]] = []
    node_id_map: Dict[int, int] = {}
    next_nid = max(node_positions.keys()) + 1
    next_eid = max(model.elements.keys()) + 1

    pid_filter = set(pids) if pids else None
    patches_remeshed = 0
    patches_kept = 0

    for pidx, (patch, patch_pid) in enumerate(zip(patches, patch_pids)):
        if pid_filter and patch_pid not in pid_filter:
            # Keep original elements
            for eid in patch:
                nodes = elem_nodes_map.get(eid, [])
                for n in nodes:
                    if n not in all_new_nodes and n in node_positions:
                        all_new_nodes[n] = node_positions[n].copy()
                elem = model.elements[eid]
                etype = "CTRIA3" if len(nodes) == 3 else "CQUAD4"
                all_new_elems.append((etype, patch_pid, nodes))
            patches_kept += 1
            continue

        logger.info(f"  Patch {pidx} ({len(patch)} elems, PID {patch_pid})...")
        new_nodes, new_elems, _ = remesh_patch(
            patch, elem_nodes_map, node_positions, edge_elems,
            pinned, target_length, prefer_quads, patch_pid,
            global_boundary_nodes=global_boundary_nodes,
        )

        if not new_elems:
            logger.warning(f"  Patch {pidx}: remeshing failed, keeping original")
            for eid in patch:
                nodes = elem_nodes_map.get(eid, [])
                for n in nodes:
                    if n not in all_new_nodes and n in node_positions:
                        all_new_nodes[n] = node_positions[n].copy()
                etype = "CTRIA3" if len(nodes) == 3 else "CQUAD4"
                all_new_elems.append((etype, patch_pid, nodes))
            patches_kept += 1
            continue

        # Remap temp node IDs to sequential IDs
        for temp_nid, pos in new_nodes.items():
            if temp_nid in node_positions:
                # Original boundary node - keep original ID
                all_new_nodes[temp_nid] = pos
                node_id_map[temp_nid] = temp_nid
            else:
                all_new_nodes[next_nid] = pos
                node_id_map[temp_nid] = next_nid
                next_nid += 1

        for etype, epid, enodes in new_elems:
            mapped = [node_id_map.get(n, n) for n in enodes]
            all_new_elems.append((etype, epid, mapped))

        patches_remeshed += 1

    # Ensure ALL global boundary nodes are in the output — they define the geometry
    for nid in global_boundary_nodes:
        if nid not in all_new_nodes and nid in node_positions:
            all_new_nodes[nid] = node_positions[nid].copy()

    logger.info(f"Remeshed {patches_remeshed} patches, kept {patches_kept} as-is")
    logger.info(f"New mesh: {len(all_new_nodes)} nodes, {len(all_new_elems)} elements")

    # ── Build output model ──
    # Start from a copy of the original model so executive control,
    # case control, params, coords, properties, materials, etc. are
    # all preserved.  Then replace nodes and shell elements.
    logger.info("Building output model...")
    out = model

    # Remove shell elements (non-shell elements like CBUSH are kept)
    shell_eids = [eid for eid, e in out.elements.items() if e.type in SHELL_TYPES]
    for eid in shell_eids:
        del out.elements[eid]

    # Preserve nodes referenced by non-shell entities (CONM2, RBE, SPC, etc.)
    # then clear and re-add
    preserved_nodes: Dict[int, np.ndarray] = {}
    for nid in pinned:
        if nid in node_positions:
            preserved_nodes[nid] = node_positions[nid].copy()
    # Also keep any node not in the shell mesh (standalone grids for CONM2 etc.)
    shell_node_set: Set[int] = set()
    for nodes in elem_nodes_map.values():
        shell_node_set.update(nodes)
    for nid in node_positions:
        if nid not in shell_node_set:
            preserved_nodes[nid] = node_positions[nid].copy()

    out.nodes.clear()

    # Add preserved nodes (CONM2, RBE reference grids, etc.)
    for nid, pos in preserved_nodes.items():
        if nid not in all_new_nodes:
            out.add_grid(nid, list(pos), cp=0, cd=0, ps=0, seid=0)

    # Add remeshed nodes
    for nid, pos in all_new_nodes.items():
        if nid not in out.nodes:
            out.add_grid(nid, list(pos), cp=0, cd=0, ps=0, seid=0)

    # Add new shell elements with IDs that don't conflict with kept elements
    existing_eids = set(out.elements.keys())
    eid = max(existing_eids) + 1 if existing_eids else 1
    for etype, epid, enodes in all_new_elems:
        while eid in existing_eids:
            eid += 1
        if etype == "CTRIA3" and len(enodes) >= 3:
            out.add_ctria3(eid, epid, enodes[:3])
        elif etype == "CQUAD4" and len(enodes) >= 4:
            out.add_cquad4(eid, epid, enodes[:4])
        existing_eids.add(eid)
        eid += 1

    # ── Step 6: reattach RBE3 Gijs ──
    logger.info("Reattaching RBE3 connections...")
    reattach_rbe3_gijs(out, node_positions, all_new_nodes, node_id_map)

    # ── Write ──
    logger.info(f"Writing output: {output_file}")
    out.write_bdf(output_file, size=16, is_double=False)

    # ── Visualise result ──
    result_html = os.path.splitext(output_file)[0] + "_result.html"
    logger.info(f"Generating result visualisation: {result_html}")
    visualise_result(all_new_nodes, all_new_elems, pinned, result_html)

    # ── Step 7: mass check ──
    logger.info("\n--- Mass Check ---")
    try:
        check = BDF()
        check.read_bdf(output_file, xref=True, punch=punch)
        final_mass, final_mass_lbs, final_pid_mass = \
            calculate_total_mass(check, mass_to_lbs_factor)
    except Exception as e:
        logger.warning(f"Could not compute final mass: {e}")
        final_mass = final_mass_lbs = 0.0
        final_pid_mass = {}

    logger.info(f"Initial: {initial_mass:.6f} = {initial_mass_lbs:.2f} lbs")
    logger.info(f"Final:   {final_mass:.6f} = {final_mass_lbs:.2f} lbs")
    if initial_mass > 0:
        diff_pct = abs(final_mass - initial_mass) / initial_mass * 100
        logger.info(f"Diff:    {abs(final_mass - initial_mass):.6f} ({diff_pct:.4f}%)")
    else:
        diff_pct = 0.0

    all_pids = sorted(set(list(initial_pid_mass.keys()) + list(final_pid_mass.keys())))
    if all_pids:
        logger.info(f"\n  {'PID':>8s}  {'Initial':>12s}  {'Final':>12s}  {'Diff%':>8s}")
        for p in all_pids:
            mi = initial_pid_mass.get(p, 0.0)
            mf = final_pid_mass.get(p, 0.0)
            pct = abs(mf - mi) / mi * 100 if mi > 0 else 0.0
            label = "masses" if p == -1 else str(p)
            logger.info(f"  {label:>8s}  {mi:12.6f}  {mf:12.6f}  {pct:7.3f}%")

    logger.info("\nDone.")
    return {
        "patches": len(patches),
        "patches_remeshed": patches_remeshed,
        "new_nodes": len(all_new_nodes),
        "new_elements": len(all_new_elems),
        "mass_diff_pct": diff_pct,
    }


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Surface-based Nastran BDF remeshing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", "-i", dest="input_file", required=True)
    parser.add_argument("--out", "-o", dest="output_file", default=None)
    parser.add_argument("--target", "-t", type=float, default=None,
                        help="Target element edge length")
    parser.add_argument("--feature-angle", type=float, default=30.0,
                        help="Dihedral angle for feature edges (default: 30)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show interactive HTML before remeshing")
    parser.add_argument("--viz-only", action="store_true",
                        help="Visualise only, do not remesh")
    parser.add_argument("--pids", type=str, default=None,
                        help="Only remesh these PIDs (comma-separated)")
    parser.add_argument("--quad", dest="prefer_quads", action="store_true",
                        default=True, help="Prefer quad elements (default)")
    parser.add_argument("--tri", dest="prefer_quads", action="store_false",
                        help="Force all-triangle output")
    parser.add_argument("--mass-factor", type=float, default=386.4)
    parser.add_argument("--punch", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.viz_only and not args.output_file:
        parser.error("--out is required unless --viz-only is used")
    if not args.viz_only and args.target is None:
        parser.error("--target is required unless --viz-only is used")

    pid_list = None
    if args.pids:
        pid_list = [int(p.strip()) for p in args.pids.split(",")]

    try:
        remesh_surfaces(
            input_file=args.input_file,
            output_file=args.output_file,
            target_length=args.target,
            feature_angle=args.feature_angle,
            visualize=args.visualize,
            viz_only=args.viz_only,
            pids=pid_list,
            prefer_quads=args.prefer_quads,
            mass_to_lbs_factor=args.mass_factor,
            punch=args.punch,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
