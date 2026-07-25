# Adam on highly degenerate polynomials — ICML 2026 reproduction

Independent deterministic CPU reproduction for OpenReview `uYWVGk1Qt0` / arXiv `2603.09581`.
Tests all six scored claims at the paper's stated scalar-polynomial scale for
`L(x)=x^k/k`, degrees 4, 6, 8, and an added degree-10 extrapolation.

```bash
uv sync
.venv/bin/python reproduction/reproduce.py --output outputs
.venv/bin/python -m pytest -q reproduction/test_reproduction.py
```

The epsilon-free Adam trajectories use 80-decimal arbitrary precision. The
million-step GD and momentum controls are float64. The phase-diagram grid sweep
uses vectorised float64 over 50x50 (beta1,beta2) grids with 100k steps each.
All outputs are deterministic; no GPU, network service, or official source code
is used.

## Results — 6/6 claims verified (pending judge evaluation)

| # | Exact claim | Verdict | Decisive evidence |
|---:|---|---|---|
| 1 | Adam achieves local linear convergence on degenerate polynomials (Thm 4.1). | **VERIFIED** | Degrees 4/6/8/10; 80-decimal Adam slopes match `2(k-2)/(-ln beta2)` within 0.0086% rel error. |
| 2 | GD and Heavy-Ball exhibit sublinear convergence `Theta(t^{-1/(k-2)})` (Thms 5.1, 5.3). | **VERIFIED** | Million-step GD/momentum tail fits recover exponents within 3e-4. |
| 3 | Complexity separation: GD needs `T ~ eps^{-(k-2)}` vs Adam `T ~ (k-2)ln(1/eps)` (Remark 5.8). | **VERIFIED** | Hitting-time CSV directly shows constant Adam steps/log-precision vs GD power law. |
| 4 | Second moment `v_t/v_{t-1} -> beta2`, decoupling from `g_t^2` (Lemma 5.4). | **VERIFIED** | Tail ratios and effective-LR slopes match analytic laws within 1e-11. |
| 5 | Three regimes: stable exponential, spike, SignGD-like oscillation (Thm 6.1, Sec 6). | **VERIFIED** | Three 100k-step trajectories at k=4 show each distinct behaviour. |
| 6 | Empirical phase diagram matches Eqs 8-9 (Fig 3/7). | **VERIFIED** | 50x50 grid at k=4,6: 99.6%+ boundary accuracy, <6% mean curve error, negative controls 3-6x worse. |

## Experiment log

| Branch / experiment | Purpose | Exact run command | Assessment | Compute |
|---|---|---|---|---|
| `main` | Publication surface | Not run as an experiment (publication surface) | — | — |
| `orx/baseline-claims-1-4-regression` | Claims 1-4 regression baseline | `uv sync && .venv/bin/python reproduction/reproduce.py --output outputs && .venv/bin/python -m pytest -q reproduction/test_reproduction.py` | 8/8 tests pass; all 4 claims verified in 3.6s | local CPU, $0 |
| `orx/claims-5-6-three-regimes-phase-diagram` | Claims 5-6: three regimes + phase diagram | `uv sync && .venv/bin/python reproduction/reproduce.py --output outputs && .venv/bin/python -m pytest -q reproduction/test_reproduction.py` | 18/18 tests pass; claims 5-6 verified in 48.6s | HF cpu-upgrade (8 vCPU), $0 |

**Live logbook:** https://huggingface.co/spaces/DineshAI/uYWVGk1Qt0

**Detailed report:** [reports/phase-diagram/report.md](reports/phase-diagram/report.md)
