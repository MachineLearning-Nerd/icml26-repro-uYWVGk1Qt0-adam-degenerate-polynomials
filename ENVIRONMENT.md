# Environment

- CPU-only, 4 logical cores; no GPU used
- Python 3.12
- Deterministic scalar arithmetic: `mpmath` at 80 decimal digits for Adam/RMSProp tails
- Float64 NumPy for the million-step GD and momentum controls
- Paper: arXiv:2603.09581; PDF SHA-256 `1a3f05ff0b6faec91a86c17d4dc8919b7ee89e1ee48a44dcaf6b1699b2abce17`
- No official implementation was located; this is an independent implementation from the equations in the paper.

