#!/usr/bin/env python3
"""
refine_and_fix.py - Fix OML, refine mesh, then fix OML again.

Pipeline:
1. Fix barrel OML on coarse mesh (so refinement midpoints land on the circle)
2. Fix dome OML on coarse mesh
3. Refine mesh to target edge length
4. Fix barrel OML again (clean up new midpoints)
5. Fix dome OML again

Usage:
    python refine_and_fix.py input.bdf output.bdf
"""

import subprocess
import sys
import os

# ---- Configuration ----
TARGET_EDGE_LENGTH = 3.0
MAX_PASSES = 4
MIN_EDGE_LENGTH = 0.5  # Don't split elements smaller than this

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


def fix_oml(script, input_file, output_file, pids, label):
    """Run fix_oml.py with given PIDs."""
    print(f"\n*** {label} ***")
    pids_str = ",".join(str(p) for p in pids)
    run([
        sys.executable, script,
        "--in", input_file,
        "--out", output_file,
        "--pids", pids_str,
    ])


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
    step1 = f"{base}_step1_barrel{ext}"
    step2 = f"{base}_step2_domes{ext}"
    step3 = f"{base}_step3_refined{ext}"
    step4 = f"{base}_step4_barrel{ext}"

    # Step 1: Fix barrel OML on coarse mesh
    fix_oml(fix_oml_script, input_file, step1, BARREL_PIDS,
            "STEP 1/5: Fix barrel OML (coarse)")

    # Step 2: Fix dome OML on coarse mesh
    fix_oml(fix_oml_script, step1, step2, DOME_PIDS,
            "STEP 2/5: Fix dome OML (coarse)")

    # Step 3: Refine mesh
    print(f"\n*** STEP 3/5: Refine mesh (target={TARGET_EDGE_LENGTH}, "
          f"min_edge={MIN_EDGE_LENGTH}, max_passes={MAX_PASSES}) ***")
    run([
        sys.executable, refine_script,
        "--in", step2,
        "--out", step3,
        "--target", str(TARGET_EDGE_LENGTH),
        "--min-edge", str(MIN_EDGE_LENGTH),
        "--max-passes", str(MAX_PASSES),
    ])

    # Step 4: Fix barrel OML on refined mesh
    fix_oml(fix_oml_script, step3, step4, BARREL_PIDS,
            "STEP 4/5: Fix barrel OML (refined)")

    # Step 5: Fix dome OML on refined mesh
    fix_oml(fix_oml_script, step4, output_file, DOME_PIDS,
            "STEP 5/5: Fix dome OML (refined)")

    # Clean up intermediate files
    print(f"\nCleaning up intermediate files...")
    for f in [step1, step2, step3, step4]:
        if os.path.isfile(f):
            os.remove(f)
            print(f"  Removed {f}")

    print(f"\n{'='*60}")
    print(f"DONE: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
