Mode Miner — Requirements

Inputs
- SOL 103 OP2
- Matching BDF

Required OP2 Content
- Eigenvalues (frequencies)
- Eigenvectors at GRID DOFs
- Determinable normalization:
  - mass-normalized preferred
  - otherwise modal mass required

If normalization cannot be determined:
- FRF computation must be disabled

Required BDF Content
- GRID
- CORD2*
- SPC / SPC1
- MPC / RBE parsed for warnings only

Supported Analyses (MVP)
- Modal inspection
- Mode shape animation
- SISO mobility (v/F)
- FRF peak detection
- Peak-to-mode attribution (top-N modes)

Excitation
- Single-point force
- User selects:
  - input node + DOF
  - response node + DOF

Outputs
- FRF magnitude and phase
- Peak table:
  - peak frequency
  - dominant mode(s)
  - percent contribution
- Mode shape animation from peaks

Guardrails
- Warn on constrained DOFs
- Warn if frequency band exceeds modal coverage
- Refuse invalid requests clearly

Explicitly Excluded
- Running NASTRAN
- Stress or strain recovery
- Base excitation (future)
- MIMO / coherence