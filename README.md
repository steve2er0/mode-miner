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

## Status

Mode Miner is under active development.
Early versions focus on modal visualization and FRF peak attribution.

Contributions, feedback, and testing are welcome.

---

## Philosophy

If an engineer can click a peak and immediately see
which mode shapes are driving it,
the tool has done its job.
