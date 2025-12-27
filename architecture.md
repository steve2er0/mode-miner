Mode Miner — Architecture (Local Desktop)

High-Level Flow
1. Load BDF
2. Load OP2
3. Build DOF and mode index
4. Visualize mesh
5. Compute FRF
6. Detect peaks
7. Attribute peaks to modes
8. Animate mode shapes

Core Data Model

NodeDOF
- (grid_id, component)
- component: 1..6

ModalModel
- frequencies
- eigenvectors
- DOF index
- normalization info
- optional modal mass

Module Layout

src/
- ingest/
  - bdf_reader.py
  - op2_reader.py
- model/
  - modal_model.py
  - dof_map.py
- compute/
  - damping.py
  - frf.py
  - peaks.py
  - attribution.py
- ui/
  - main_window.py
  - mesh_view.py
  - mode_list.py
  - frf_plot.py

Compute APIs (MVP)
- frf_vf(model, response_dof, input_dof, freq, damping)
- detect_peaks(|FRF|)
- attribute_peak(model, peak_freq, input_dof, response_dof)

Peak Attribution
- evaluate per-mode contribution at peak frequency
- rank by contribution
- expose top-N modes

Definition of Done
- Load mesh
- Animate modes
- Plot v/F
- Click peak → show driving mode