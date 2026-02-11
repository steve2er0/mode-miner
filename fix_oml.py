#!/usr/bin/env python3
"""
fix_oml.py - Fix Outer Mold Line of Refined Cylindrical Mesh

After mesh refinement on a coarse cylindrical model, the new midpoint nodes
sit on the chord of the polygon (inside the true cylinder surface). This tool
detects the cylinder radius from the original nodes and projects all nodes
radially outward to the true circular cross-section.

USAGE:
    python fix_oml.py --in refined.bdf --out fixed.bdf
    python fix_oml.py --in refined.bdf --out fixed.bdf --radius 165.25
    python fix_oml.py --in refined.bdf --out fixed.bdf --axis X --center 0,0
    python fix_oml.py --in refined.bdf --out fixed.bdf --tolerance 0.5

OPTIONS:
    --in, -i          Input BDF file path (required)
    --out, -o         Output BDF file path (required)
    --radius, -r      Target cylinder radius (auto-detected if omitted)
    --axis, -a        Cylinder axis direction: X, Y, or Z (default: auto-detect)
    --center, -c      Cylinder center in the plane perpendicular to axis,
                      format: Y,Z or X,Z or X,Y (default: 0,0)
    --tolerance, -t   Max radial deviation from target to consider a node
                      "on the cylinder" (default: 5% of radius)
    --nids            Comma-separated list of node IDs to fix (optional,
                      default: fix all nodes within tolerance)
    --punch           Read BDF in punch mode (no executive/case control)
    --verbose, -v     Enable verbose logging

ALGORITHM:
    1. Read BDF and cross-reference to get global positions
    2. Detect cylinder axis (direction of least variation in node positions)
    3. Detect cylinder radius (maximum radial distance from axis)
    4. For each node within tolerance of the cylinder surface:
       a. Compute its current radial distance from the axis
       b. Scale its Y,Z (or appropriate) coordinates to match target radius
       c. Update the GRID card coordinates in the node's input coordinate system (CP)
    5. Write the updated BDF

NOTES:
    - Only modifies node positions; elements, properties, etc. are unchanged
    - Handles nodes in different coordinate systems (CP=0, cylindrical, etc.)
    - Nodes already at the correct radius are left unchanged
    - The cylinder axis position (X coordinate) is preserved for each node

Author: OML Fix Tool v1.0
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pyNastran.bdf.bdf import BDF
except ImportError:
    print("ERROR: pyNastran is required. Install with: pip install pyNastran")
    sys.exit(1)


logger = logging.getLogger(__name__)


def detect_cylinder_axis(model: BDF) -> str:
    """
    Auto-detect the cylinder axis by looking at the symmetry of the cross-section.
    
    For a cylinder, the two radial directions will have similar spreads (diameter)
    while the axial direction will be different. We identify the axis as the
    direction whose spread is most different from the other two.
    
    Returns:
        'X', 'Y', or 'Z'
    """
    positions = np.array([node.get_position() for node in model.nodes.values()])
    
    spreads = [
        positions[:, 0].ptp(),  # X spread
        positions[:, 1].ptp(),  # Y spread
        positions[:, 2].ptp(),  # Z spread
    ]
    
    logger.info(f"Position spreads: X={spreads[0]:.2f}, Y={spreads[1]:.2f}, Z={spreads[2]:.2f}")
    
    # For a cylinder, the two radial directions have similar spreads (both ~ diameter)
    # The axial direction has a different spread (the length).
    # Find the axis whose spread is most different from the other two.
    axis_names = ['X', 'Y', 'Z']
    
    # Compute how "different" each direction is from the other two
    diffs = []
    for i in range(3):
        others = [spreads[j] for j in range(3) if j != i]
        # How different is this spread from the average of the other two?
        diff = abs(spreads[i] - sum(others) / 2.0)
        diffs.append(diff)
    
    # Also check: the two radial directions should have similar spreads
    # The axial direction is the odd one out
    # Sort spreads and the two closest are the radial directions
    indexed_spreads = sorted(enumerate(spreads), key=lambda x: x[1])
    
    # If two spreads are very similar, they're the radial pair
    s0, s1, s2 = indexed_spreads[0][1], indexed_spreads[1][1], indexed_spreads[2][1]
    
    # Ratio between the two closest
    ratio_01 = abs(s1 - s0) / max(s0, 1e-10)
    ratio_12 = abs(s2 - s1) / max(s1, 1e-10)
    
    if ratio_01 < ratio_12:
        # s0 and s1 are similar -> they're radial, s2 is axial
        axis_idx = indexed_spreads[2][0]
    else:
        # s1 and s2 are similar -> they're radial, s0 is axial
        axis_idx = indexed_spreads[0][0]
    
    logger.info(f"Auto-detected cylinder axis: {axis_names[axis_idx]}")
    
    return axis_names[axis_idx]


def get_radial_components(axis: str) -> Tuple[int, int, int]:
    """
    Get the indices for (axial, radial1, radial2) components based on cylinder axis.
    
    Returns:
        Tuple of (axial_idx, r1_idx, r2_idx) into XYZ array
    """
    if axis == 'X':
        return (0, 1, 2)  # axial=X, radial=Y,Z
    elif axis == 'Y':
        return (1, 0, 2)  # axial=Y, radial=X,Z
    elif axis == 'Z':
        return (2, 0, 1)  # axial=Z, radial=X,Y
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be X, Y, or Z.")


def detect_cylinder_radius(model: BDF, axis: str, center: Tuple[float, float]) -> float:
    """
    Auto-detect the cylinder radius as the maximum radial distance from the axis.
    The original polygon vertices should be at the correct radius.
    
    Args:
        model: pyNastran BDF model
        axis: Cylinder axis ('X', 'Y', or 'Z')
        center: Center of cylinder in the radial plane (r1_center, r2_center)
        
    Returns:
        Detected cylinder radius
    """
    ax_idx, r1_idx, r2_idx = get_radial_components(axis)
    
    radii = []
    for node in model.nodes.values():
        pos = node.get_position()
        r1 = pos[r1_idx] - center[0]
        r2 = pos[r2_idx] - center[1]
        r = np.sqrt(r1**2 + r2**2)
        radii.append(r)
    
    radii = np.array(radii)
    max_r = np.max(radii)
    
    # The target radius is the maximum (original polygon vertices)
    # Verify by checking that a reasonable number of nodes are at this radius
    near_max = np.sum(np.abs(radii - max_r) < 0.01 * max_r)
    logger.info(f"Max radius: {max_r:.4f} ({near_max} nodes within 1%)")
    
    return max_r


def fix_oml(
    input_file: str,
    output_file: str,
    target_radius: Optional[float] = None,
    axis: Optional[str] = None,
    center: Tuple[float, float] = (0.0, 0.0),
    tolerance_pct: float = 5.0,
    nid_filter: Optional[List[int]] = None,
    punch: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Fix the outer mold line by projecting nodes to the true cylinder radius.
    
    Args:
        input_file: Input BDF file path
        output_file: Output BDF file path
        target_radius: Target cylinder radius (auto-detected if None)
        axis: Cylinder axis 'X', 'Y', or 'Z' (auto-detected if None)
        center: Cylinder center in the radial plane
        tolerance_pct: Percentage of radius used as tolerance band
        nid_filter: Optional list of specific node IDs to fix
        punch: Whether to read BDF in punch mode
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with statistics
    """
    # Setup logging
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    import os
    log_file = os.path.splitext(output_file)[0] + '_oml_fix.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    logger.propagate = False
    
    logger.info(f"Reading BDF: {input_file}")
    logger.info(f"Detailed log: {log_file}")
    
    # Read BDF
    model = BDF()
    model.read_bdf(input_file, xref=True, punch=punch)
    
    total_nodes = len(model.nodes)
    logger.info(f"Total nodes: {total_nodes}")
    
    # Detect or use specified axis
    if axis is None:
        axis = detect_cylinder_axis(model)
    else:
        axis = axis.upper()
    
    ax_idx, r1_idx, r2_idx = get_radial_components(axis)
    axis_names = ['X', 'Y', 'Z']
    logger.info(f"Cylinder axis: {axis} (axial={axis_names[ax_idx]}, "
                f"radial plane={axis_names[r1_idx]},{axis_names[r2_idx]})")
    logger.info(f"Cylinder center in radial plane: ({center[0]}, {center[1]})")
    
    # Detect or use specified radius
    if target_radius is None:
        target_radius = detect_cylinder_radius(model, axis, center)
        logger.info(f"Auto-detected target radius: {target_radius:.4f}")
    else:
        logger.info(f"Using specified target radius: {target_radius:.4f}")
    
    # Compute tolerance band
    tolerance = tolerance_pct / 100.0 * target_radius
    r_min = target_radius - tolerance
    r_max = target_radius + tolerance
    logger.info(f"Tolerance: {tolerance:.4f} ({tolerance_pct}% of radius)")
    logger.info(f"Fixing nodes with radius in [{r_min:.4f}, {r_max:.4f}]")
    
    # Process each node
    stats = {
        'total_nodes': total_nodes,
        'nodes_in_band': 0,
        'nodes_fixed': 0,
        'nodes_skipped_already_correct': 0,
        'nodes_skipped_outside_band': 0,
        'nodes_skipped_filter': 0,
        'max_radial_correction': 0.0,
        'avg_radial_correction': 0.0,
    }
    
    corrections = []
    
    for nid, node in model.nodes.items():
        # Get global position
        pos_global = node.get_position()
        
        # Compute current radial distance
        r1 = pos_global[r1_idx] - center[0]
        r2 = pos_global[r2_idx] - center[1]
        current_r = np.sqrt(r1**2 + r2**2)
        
        # Check if node is in the tolerance band
        if current_r < r_min or current_r > r_max:
            stats['nodes_skipped_outside_band'] += 1
            continue
        
        stats['nodes_in_band'] += 1
        
        # Check NID filter
        if nid_filter is not None and nid not in nid_filter:
            stats['nodes_skipped_filter'] += 1
            continue
        
        # Check if already at target radius
        if abs(current_r - target_radius) < 1e-6:
            stats['nodes_skipped_already_correct'] += 1
            logger.debug(f"Node {nid}: Already at target radius ({current_r:.6f})")
            continue
        
        # Compute the scale factor to push to target radius
        if current_r < 1e-10:
            logger.warning(f"Node {nid}: On the axis (R={current_r:.6f}), skipping")
            continue
        
        scale = target_radius / current_r
        radial_correction = target_radius - current_r
        
        # Compute new global position (only change radial components)
        new_pos_global = pos_global.copy()
        new_pos_global[r1_idx] = center[0] + r1 * scale
        new_pos_global[r2_idx] = center[1] + r2 * scale
        
        logger.debug(f"Node {nid}: R={current_r:.4f} -> {target_radius:.4f} "
                     f"(correction={radial_correction:+.4f})")
        
        # Now we need to write the new position back in the node's input
        # coordinate system (CP). For CP=0, it's just global XYZ.
        # For cylindrical CP, we need to convert back.
        cp_val = 0
        cp_raw = getattr(node, 'cp', 0)
        if hasattr(cp_raw, 'cid'):
            cp_val = cp_raw.cid
        elif isinstance(cp_raw, int):
            cp_val = cp_raw
        
        if cp_val == 0:
            # Global rectangular - directly set XYZ
            node.xyz[0] = new_pos_global[0]
            node.xyz[1] = new_pos_global[1]
            node.xyz[2] = new_pos_global[2]
        else:
            # Non-zero CP - transform global position back to local CP
            coord = model.coords[cp_val]
            if coord.type in ('CORD2C', 'CORD1C'):
                # Cylindrical: just update the R component
                # node.xyz = [R, theta, Z] in the cylindrical system
                node.xyz[0] = target_radius
                logger.debug(f"  Node {nid}: Cylindrical CP={cp_val}, setting R={target_radius:.4f}")
            elif coord.type in ('CORD2R', 'CORD1R'):
                # Rectangular local: transform back
                try:
                    new_local = coord.transform_node_to_local(new_pos_global)
                    node.xyz[0] = new_local[0]
                    node.xyz[1] = new_local[1]
                    node.xyz[2] = new_local[2]
                    logger.debug(f"  Node {nid}: Rectangular CP={cp_val}, "
                                f"new local={list(new_local)}")
                except Exception as e:
                    logger.warning(f"  Node {nid}: Failed to transform back to CP={cp_val}: {e}")
                    continue
            else:
                logger.warning(f"  Node {nid}: Unsupported CP type {coord.type}, skipping")
                continue
        
        stats['nodes_fixed'] += 1
        corrections.append(abs(radial_correction))
    
    # Compute correction statistics
    if corrections:
        stats['max_radial_correction'] = max(corrections)
        stats['avg_radial_correction'] = sum(corrections) / len(corrections)
    
    # Print summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("OML FIX SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total nodes: {stats['total_nodes']}")
    logger.info(f"Nodes in tolerance band: {stats['nodes_in_band']}")
    logger.info(f"Nodes fixed: {stats['nodes_fixed']}")
    logger.info(f"Nodes already correct: {stats['nodes_skipped_already_correct']}")
    logger.info(f"Nodes outside band: {stats['nodes_skipped_outside_band']}")
    if nid_filter:
        logger.info(f"Nodes filtered out: {stats['nodes_skipped_filter']}")
    if corrections:
        logger.info(f"Max radial correction: {stats['max_radial_correction']:.4f}")
        logger.info(f"Avg radial correction: {stats['avg_radial_correction']:.4f}")
    logger.info(f"Target radius: {target_radius:.4f}")
    
    # Write output
    logger.info(f"\nWriting fixed BDF: {output_file}")
    
    # Un-cross-reference before writing so pyNastran writes the raw node.xyz values
    model.uncross_reference()
    model.write_bdf(output_file)
    
    logger.info("Done.")
    
    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fix outer mold line of refined cylindrical mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_oml.py --in refined.bdf --out fixed.bdf
  python fix_oml.py --in refined.bdf --out fixed.bdf --radius 165.25
  python fix_oml.py --in refined.bdf --out fixed.bdf --axis X --center 0,0
  python fix_oml.py --in refined.bdf --out fixed.bdf --tolerance 3.0
  python fix_oml.py --in refined.bdf --out fixed.bdf --punch
"""
    )
    
    parser.add_argument('--in', '-i', dest='input_file', required=True,
                        help='Input BDF file path')
    parser.add_argument('--out', '-o', dest='output_file', required=True,
                        help='Output BDF file path')
    parser.add_argument('--radius', '-r', type=float, default=None,
                        help='Target cylinder radius (auto-detected if omitted)')
    parser.add_argument('--axis', '-a', type=str, default=None,
                        choices=['X', 'Y', 'Z', 'x', 'y', 'z'],
                        help='Cylinder axis direction (auto-detected if omitted)')
    parser.add_argument('--center', '-c', type=str, default='0,0',
                        help='Cylinder center in radial plane, format: val1,val2 (default: 0,0)')
    parser.add_argument('--tolerance', '-t', type=float, default=5.0,
                        help='Tolerance as %% of radius for node selection (default: 5.0)')
    parser.add_argument('--nids', type=str, default=None,
                        help='Comma-separated list of node IDs to fix (optional)')
    parser.add_argument('--punch', action='store_true',
                        help='Read BDF in punch mode (no executive/case control)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Parse center
    try:
        center_parts = args.center.split(',')
        center = (float(center_parts[0]), float(center_parts[1]))
    except (ValueError, IndexError):
        print(f"ERROR: Invalid center format '{args.center}'. Use: val1,val2")
        sys.exit(1)
    
    # Parse NID filter
    nid_filter = None
    if args.nids:
        try:
            nid_filter = [int(n.strip()) for n in args.nids.split(',')]
        except ValueError:
            print(f"ERROR: Invalid nids format '{args.nids}'. Use comma-separated integers.")
            sys.exit(1)
    
    # Run
    fix_oml(
        input_file=args.input_file,
        output_file=args.output_file,
        target_radius=args.radius,
        axis=args.axis,
        center=center,
        tolerance_pct=args.tolerance,
        nid_filter=nid_filter,
        punch=args.punch,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
