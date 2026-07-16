# Claim 1 - Linear versus sublinear


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_788b51d929bc", "created_at": "2026-07-16T16:30:16+00:00", "title": "VERIFIED across four degeneracy orders"}
-->
## Verdict: VERIFIED

| k | Adam measured | Adam theory | R² | GD exponent | Momentum exponent |
|---:|---:|---:|---:|---:|---:|
| 4 | 55.114 | 55.119 | 0.999999873 | -0.499991 | -0.499715 |
| 6 | 110.236 | 110.237 | 0.999999969 | -0.249997 | -0.249883 |
| 8 | 165.358 | 165.356 | 0.999999989 | -0.166665 | -0.166579 |
| 10 | 220.476 | 220.475 | 0.999999993 | -0.124999 | -0.124921 |

Adam threshold counts are fitted against `ln(1/epsilon)` over eleven thresholds
from 1e-4 through 1e-24. GD and momentum each run one million iterations and are
fitted in log-time over the final 500 recorded tail points. Their algebraic
exponents match `-1/(k-2)`, while Adam requires a constant number of iterations
per added log-precision unit.
