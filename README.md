# Mode Miner

Mode Miner is a free, local desktop tool for exploring and understanding
structural dynamics results from NASTRAN.

It helps engineers identify which mode shapes are actually driving
frequency response peaks — visually and intuitively.

Mode Miner is a post-processor, not a solver.

---

## What Mode Miner Does

- Loads NASTRAN SOL 103 OP2 and matching BDF files
- Renders the structure in 3D
- Animates mode shapes
- Computes SISO mobility (v/F) using modal superposition
- Detects peaks in the FRF
- Attributes each peak to the top contributing modes
- Lets you click a peak and immediately see the responsible mode shape

The goal is simple:
**see why the structure responds the way it does.**

---

## What Mode Miner Is Not

- Not a solver
- Not a pre-processor or mesher
- Not cloud-based
- Not AI-driven
- Not a replacement for NASTRAN

---

## Typical Use Cases

- Understanding unexpected FRF peaks
- Screening modes that actually matter
- Teaching modal dynamics visually
- Debugging boundary conditions and load paths
- Faster interpretation of large modal models

---

## Tech Stack

- Python
- pyNastran (OP2 / BDF parsing)
- NumPy / SciPy (modal math)
- Matplotlib (3D visualization)
- PySide6 (desktop UI)

---

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with test data
python run_test.py

# Run normal application
python run.py
```

1. Click **Load BDF** to load a NASTRAN bulk data file
2. Click **Load OP2** to load modal analysis results
3. Select a mode from the list to animate it
4. Use **Stop Animation** to return to the static mesh

---

## Mesh Refinement Tool (`refine_shell_mesh.py`)

A CLI tool to refine Nastran BDF meshes by splitting elements whose edge lengths exceed a target value. Supports 2D shell elements (CQUAD4, CTRIA3) and 1D elements (CBAR, CBEAM, CROD).

```bash
# Basic usage
python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5

# With max passes and PID filter
python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5 --max-passes 8 --pids 10,12,15

# With element ID range filter
python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5 --eid-range 1000:5000

# With custom starting IDs for new nodes/elements
python refine_shell_mesh.py --in model.bdf --out refined.bdf --target 2.5 --start-nid 9800001 --start-eid 9800001
```

Key features:
- Conformal mesh: shared edges always share the same midpoint node
- Preserves original GRID and element IDs
- Handles cylindrical coordinate systems (CORD2C) for offsets and orientation vectors
- Mass conservation check
- Detailed log file for 1D element diagnostics

---

## OML Fix Tool (`fix_oml.py`)

After mesh refinement on a coarse cylindrical model, new midpoint nodes sit on polygon chords inside the true cylinder surface. This tool detects the cylinder radius and projects nodes radially outward to the true circular/elliptical cross-section.

```bash
# Auto-detect everything (axis, radius per station)
python fix_oml.py --in refined.bdf --out fixed.bdf --pids 100,101,102

# Specify axis and radius explicitly
python fix_oml.py --in refined.bdf --out fixed.bdf --pids 100 --axis X --radius 165.25

# Punch-format BDF (no executive/case control deck)
python fix_oml.py --in refined.bdf --out fixed.bdf --pids 100 --punch
```

Key features:
- Per-station radius detection: handles tapered cylinders and domes
- PID filtering to target only the OML skin (avoids interior structure, feedlines, tunnels)
- Handles multiple coordinate systems (rectangular and cylindrical CP)
- Fixes 1D element orientations after node projection
- Preserves axial positions while fixing radial positions

### Cylindrical Tank with Domes

For structures like a LOX tank with a cylindrical barrel and elliptical domes, run the barrel and dome PIDs separately for cleanest results at the dome-barrel transition:

```bash
# Step 1: Fix barrel skin
python fix_oml.py --in model.bdf --out step1.bdf \
  --pids 621001,621002,621003,621004,621005,621006,621007,621008,621009,621010,621011,621012,621013,621014,621015,621016,621017,621018,621019,621020

# Step 2: Fix dome skin
python fix_oml.py --in step1.bdf --out fixed.bdf \
  --pids 620002,620003,620004,620005,620008,620009,620010,620011,620012,620013,620014,620015
```

Or run all skin PIDs at once (works well, with minor spread at dome-barrel boundary):

```bash
python fix_oml.py --in model.bdf --out fixed.bdf \
  --pids 620002,620003,620004,620005,620008,620009,620010,620011,620012,620013,620014,620015,621001,621002,621003,621004,621005,621006,621007,621008,621009,621010,621011,621012,621013,621014,621015,621016,621017,621018,621019,621020
```

---

## Status

Mode Miner is under active development.
Early versions focus on modal visualization and FRF peak attribution.
The mesh refinement and OML fix tools are production-ready utilities.

Contributions, feedback, and testing are welcome.

---

## Philosophy

If an engineer can click a peak and immediately see
which mode shapes are driving it,
the tool has done its job.
