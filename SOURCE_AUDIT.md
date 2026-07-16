# Source and claim audit

The paper was read directly from arXiv:2603.09581. No official implementation
was found, so the reproduction is clean-room code from the stated equations.

## Mapping

- Section 4 / Eq. (1–2): Adam moments and parameter update.
- Section 4 simplifications: epsilon is zero and asymptotic bias correction is
  omitted. Initialization follows the paper: `x0=1`, `m0=g0`, `v0=g0²`.
- Theorem 4.1 / Eq. (10): `x_(t+1)/x_t -> beta2^(1/(2(k-2)))`.
- Theorem 5.1 and 5.3: GD/gradient flow and heavy-ball momentum have exponent
  `-1/(k-2)`.
- Lemma 5.4: `v_t/v_(t-1) -> beta2`; this implies exponential effective-learning-
  rate growth and the measured `v/g²` decoupling law.

## Independent checks

The code does not import or vendor author code. It independently compares fitted
rates to closed-form predictions, includes degree 10 beyond the principal paper
plots, uses RMSProp (`beta1=0`) as a mechanism ablation, and runs fail-closed tests.

## Scope

This is a full-scale reproduction of the paper's deterministic scalar-polynomial
claims, not a claim about arbitrary neural networks. Numerical agreement supports
the asymptotic results but is not represented as a substitute for their proofs.

