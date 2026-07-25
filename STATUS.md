# Status

- Paper: `uYWVGk1Qt0` / arXiv:2603.09581
- State: 6/6 claims verified (pending judge evaluation)
- Claim 1: verified locally and on HF cpu-upgrade
- Claim 2: verified locally and on HF cpu-upgrade
- Claim 3: verified (hitting-time complexity separation)
- Claim 4: verified (second-moment decoupling)
- Claim 5: verified (three regimes: stable, spike, SignGD-like)
- Claim 6: verified (phase diagram boundary matching Eqs 8-9)
- Tests: 18/18 passing
- Runtime: ~4 s (Claims 1-4, local CPU); ~49 s (Claims 5-6, HF cpu-upgrade)
- Trackio: https://huggingface.co/spaces/DineshAI/uYWVGk1Qt0
- GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-uYWVGk1Qt0-adam-degenerate-polynomials
- Environment: `uv` with `pyproject.toml` + `uv.lock`, Python 3.12
