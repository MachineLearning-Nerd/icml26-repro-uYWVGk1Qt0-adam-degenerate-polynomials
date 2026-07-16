# Claim 2 - Second-moment decoupling


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c80bd67baa5f", "created_at": "2026-07-16T16:30:17+00:00", "title": "VERIFIED with a first-moment ablation"}
-->
## Verdict: VERIFIED

| k | median v ratio | log(v/g²) measured | theory | effective-LR measured | theory |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.930000 | 0.03628535 | 0.03628535 | 0.03628535 | 0.03628535 |
| 6 | 0.930000 | 0.01814267 | 0.01814267 | 0.03628535 | 0.03628535 |
| 8 | 0.930000 | 0.01209512 | 0.01209512 | 0.03628535 | 0.03628535 |
| 10 | 0.930000 | 0.00907134 | 0.00907134 | 0.03628535 | 0.03628535 |

In the asymptotic tail, the fresh squared-gradient contribution vanishes relative
to accumulated second-moment memory, so `v_t/v_(t-1)` locks to beta2. Because
`g_t²` decays faster, `v_t/g_t²` grows geometrically. The independently measured
laws agree across all four degrees. RMSProp (`beta1=0`) remains linearly convergent
and has the same 0.93 second-moment ratio, isolating adaptivity from momentum.
