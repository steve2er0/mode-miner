Mode Miner — Validation

Reference
- NASTRAN SOL 111 (FRFs)

Validation Targets
- Peak frequency: ±0.5%
- FRF magnitude: ±1 dB (away from anti-resonances)
- Phase: ±10 degrees where response is significant

Golden Models
1. Simple beam
2. Plate / shell
3. Model with rigid elements (RBE2/RBE3)

Checks
- normalization consistency
- damping consistency
- frequency grid consistency
- DOF mapping correctness

Exit Criteria
- FRFs match reference within tolerance
- peak attribution identifies correct modes
- mode animations align with peak physics