# Adam on highly degenerate polynomials — ICML 2026 reproduction

Independent deterministic CPU reproduction for OpenReview `uYWVGk1Qt0` / arXiv `2603.09581`.
It tests both scored claims at the paper's stated scalar-polynomial scale for
`L(x)=x^k/k`, degrees 4, 6, 8, and an added degree-10 extrapolation.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python reproduction/reproduce.py --output outputs
.venv/bin/python -m pytest -q reproduction/test_reproduction.py
```

The epsilon-free Adam trajectories use 80-decimal arbitrary precision. The
million-step GD and momentum controls are float64. All outputs are deterministic;
no GPU, network service, or official source code is used.

## Result

- **Claim 1: verified.** The exact paper reference counts are recovered and the
  fitted Adam slopes match the linear-rate law with at most 0.0086% relative
  error. GD and momentum instead recover the predicted power-law exponents.
- **Claim 2: verified.** In the tail, `v_t/v_(t-1)=0.93`, `log(v/g²)` grows at
  `-log(beta2)/(k-2)`, and the effective learning rate grows at
  `-0.5 log(beta2)` per step. RMSProp provides the first-moment ablation.

