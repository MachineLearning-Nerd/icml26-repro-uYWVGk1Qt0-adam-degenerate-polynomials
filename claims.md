# Scored claims

1. Adam achieves local linear convergence on degenerate polynomials (Theorem 4.1), significantly outperforming the sub-linear convergence of gradient descent and momentum (Theorems 5.1, 5.3).
2. Adam's acceleration stems from decoupling the second moment `v_t` from the instantaneous squared gradient `g_t^2` (Lemma 5.4).
3. Complexity separation: GD requires `T_epsilon ~ epsilon^{-(k-2)}` whereas Adam achieves `T_epsilon ~ (k-2) ln(1/epsilon)` (Remark 5.8).
4. Second moment estimate satisfies `v_t/v_{t-1} -> beta_2`, decoupling `v_t` from `g_t^2` and producing an effective learning rate that grows as `beta_2^{-t/2}` (Lemma 5.4).
5. Three convergence regimes governed by `beta_1` relative to `beta_2`: stable exponential convergence, intermediate regime with spike from fixed-point instability, and SignGD-like oscillatory regime (Theorem 6.1, Section 6, Figure 5-7).
6. Empirical phase diagram of stability boundaries across different degeneracy orders k matches theoretical predictions of Equations 8-9 (Figure 3).
