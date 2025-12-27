Mode Miner — Goals

Vision
Mode Miner is a free, local desktop tool that helps engineers see and understand
structural dynamics results from NASTRAN.

It focuses on modal behavior, FRFs, and explaining why response peaks exist.

Mode Miner is a post-processor, not a solver.

What Mode Miner Is
- Local PC application
- SOL 103 post-processing tool
- Visual, physics-first
- Free and shareable

What Mode Miner Is Not
- Not a solver
- Not a pre-processor or mesher
- Not cloud-based
- Not AI-driven

MVP Goals

1. Visualize the Structure
- Load BDF
- Render mesh in 3D
- Click nodes directly from the mesh

2. Understand the Modes
- Load SOL 103 OP2
- List mode frequencies
- Animate mode shapes

3. Explain FRF Peaks (Core Value)
- Compute v/F (mobility) via modal superposition
- Automatically detect peaks
- For each peak:
  - rank top-N contributing modes
- Click a peak to immediately view the driving mode shape(s)

Success Criteria
An engineer can:
- open an OP2 + BDF
- click nodes on the mesh
- plot a mobility FRF
- click a peak and see exactly which modes are driving it