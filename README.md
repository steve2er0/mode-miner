# Wavemap

A local desktop tool for visualizing and understanding structural dynamics results from NASTRAN SOL 103 modal analysis.

## Features

- Load NASTRAN BDF and OP2 files
- Render 3D mesh with PyVista
- List modal frequencies
- Animate mode shapes

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

1. Click **Load BDF** to load a NASTRAN bulk data file
2. Click **Load OP2** to load modal analysis results
3. Select a mode from the list to animate it
4. Use **Stop Animation** to return to the static mesh

## Keyboard Shortcuts

- `Ctrl+B` - Load BDF file
- `Ctrl+O` - Load OP2 file
- `R` - Reset camera view
- `Ctrl+Q` - Exit

## Project Structure

```
wavemap/
├── run.py                    # Entry point
├── requirements.txt          # Dependencies
└── src/wavemap/
    ├── ingest/               # File readers
    │   ├── bdf_reader.py     # BDF mesh extraction
    │   └── op2_reader.py     # OP2 modal results
    ├── model/                # Data structures
    │   ├── modal_model.py    # Modal analysis container
    │   └── dof_map.py        # DOF indexing
    └── ui/                   # User interface
        ├── main_window.py    # Application window
        ├── mesh_view.py      # 3D visualization
        └── mode_list.py      # Mode selection
```

## Requirements

- Python 3.10+
- pyNastran
- NumPy
- SciPy
- PyVista
- pyvistaqt
- PySide6

