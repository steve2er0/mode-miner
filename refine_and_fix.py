#!/usr/bin/env python3
"""
refine_and_fix.py - Refine LOX tank mesh and fix OML in one shot.

1. Refine mesh to target edge length of 3, max 4 passes
2. Fix barrel OML (PIDs 621001-621020)
3. Fix dome OML (PIDs 620002-620015)

Usage:
    python refine_and_fix.py input.bdf output.bdf
"""

import subprocess
import sys
import os

# ---- Configuration ----
TARGET_EDGE_LENGTH = 3.0
MAX_PASSES = 4

BARREL_PIDS = [
    621001, 621002, 621003, 621004, 621005, 621006, 621007, 621008,
    621009, 621010, 621011, 621012, 621013, 621014, 621015, 621016,
    621017, 621018, 621019, 621020,
]

DOME_PIDS = [
    620002, 620003, 620004, 620005, 620008, 620009, 620010, 620011,
    620012, 620013, 620014, 620015,
]
# ------------------------

def run(cmd):
    """Run a command and exit on failure."""
    print(f"\n{'='*60}")
    print(f">> {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    if len(sys.argv) < 3:
        print("Usage: python refine_and_fix.py <input.bdf> <output.bdf>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isfile(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    refine_script = os.path.join(script_dir, "refine_shell_mesh.py")
    fix_oml_script = os.path.join(script_dir, "fix_oml.py")

    base, ext = os.path.splitext(output_file)
    refined_file = f"{base}_refined{ext}"
    barrel_fixed_file = f"{base}_barrel_fixed{ext}"

    # Step 1: Refine mesh
    print("\n*** STEP 1: Refine mesh ***")
    run([
        sys.executable, refine_script,
        "--in", input_file,
        "--out", refined_file,
        "--target", str(TARGET_EDGE_LENGTH),
        "--max-passes", str(MAX_PASSES),
    ])

    # Step 2: Fix barrel OML
    print("\n*** STEP 2: Fix barrel OML ***")
    barrel_pids_str = ",".join(str(p) for p in BARREL_PIDS)
    run([
        sys.executable, fix_oml_script,
        "--in", refined_file,
        "--out", barrel_fixed_file,
        "--pids", barrel_pids_str,
    ])

    # Step 3: Fix dome OML
    print("\n*** STEP 3: Fix dome OML ***")
    dome_pids_str = ",".join(str(p) for p in DOME_PIDS)
    run([
        sys.executable, fix_oml_script,
        "--in", barrel_fixed_file,
        "--out", output_file,
        "--pids", dome_pids_str,
    ])

    # Clean up intermediate files
    print(f"\nCleaning up intermediate files...")
    for f in [refined_file, barrel_fixed_file]:
        if os.path.isfile(f):
            os.remove(f)
            print(f"  Removed {f}")

    print(f"\n{'='*60}")
    print(f"DONE: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
