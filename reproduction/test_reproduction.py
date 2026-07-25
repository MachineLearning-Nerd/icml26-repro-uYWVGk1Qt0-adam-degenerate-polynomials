import math
from pathlib import Path

import pandas as pd


OUT = Path(__file__).parents[1] / "outputs"


def test_all_claims_verified():
    import json
    s = json.loads((OUT / "summary.json").read_text())
    assert s["claim_1"] == s["claim_2"] == "verified"
    assert s["claim_5"] == s["claim_6"] == "verified"


def test_reference_step_counts_exact():
    a = pd.read_csv(OUT / "adam_linear_rates.csv").set_index("degree")
    assert [int(a.loc[k, "steps_to_1e-20"]) for k in (4, 6, 8)] == [3384, 5818, 8274]


def test_adam_rate_matches_theory():
    a = pd.read_csv(OUT / "adam_linear_rates.csv")
    assert a.relative_slope_error.max() < 2e-4
    assert a.fit_r2.min() > 0.999999


def test_gd_and_momentum_are_sublinear():
    s = pd.read_csv(OUT / "gd_momentum_power_laws.csv")
    assert (s.fitted_exponent < 0).all()
    assert max(abs(s.fitted_exponent - s.analytic_exponent)) < 3e-4
    assert s.fit_r2.min() > 0.999999


def test_second_moment_memory_dominates():
    d = pd.read_csv(OUT / "decoupling.csv")
    assert max(abs(d.median_v_ratio - d.target_beta2)) < 1e-12
    assert d.decoupling_r2.min() > 0.999999


def test_effective_learning_rate_growth():
    d = pd.read_csv(OUT / "decoupling.csv")
    assert max(abs(d.effective_lr_log_slope - d.analytic_effective_lr_slope)) < 1e-11
    assert d.effective_lr_r2.min() > 0.999999


def test_rmsprop_negative_ablation():
    r = pd.read_csv(OUT / "rmsprop_control.csv")
    assert len(r) == 4
    assert max(abs(r.median_v_ratio - 0.93)) < 1e-12
    assert (r["steps_to_1e-20"] > 0).all()


def test_gradient_flow_hitting_time_formula():
    h = pd.read_csv(OUT / "hitting_times.csv")
    row = h[(h.degree == 4) & (h.epsilon == 1e-4)].iloc[0]
    assert math.isclose(row.gradient_flow_time, (1e8 - 1) / 2, rel_tol=1e-14)


# ---------------------------------------------------------------------------
# Claim 5: three convergence regimes (Theorem 6.1, Section 6)
# ---------------------------------------------------------------------------

def test_claim5_regime_files_exist():
    for label in ("I_stable", "II_spike", "III_signgd"):
        assert (OUT / f"regime_{label}_trajectory.csv").exists()
    assert (OUT / "three_regimes_k4.csv").exists()
    assert (OUT / "three_regimes.png").exists()


def test_claim5_regime_I_stable_converges():
    r = pd.read_csv(OUT / "three_regimes_k4.csv")
    row = r[r.regime == "I_stable"].iloc[0]
    assert row["final_loss"] < 1e-100
    assert row["theoretical_regime"] == "I_stable"


def test_claim5_regime_II_shows_spike():
    r = pd.read_csv(OUT / "three_regimes_k4.csv")
    row = r[r.regime == "II_spike"].iloc[0]
    assert row["min_loss"] < 1e-30
    assert row["spike_ratio"] > 1e6
    assert row["theoretical_regime"] == "II_spike"


def test_claim5_regime_III_nonconverged():
    r = pd.read_csv(OUT / "three_regimes_k4.csv")
    row = r[r.regime == "III_signgd"].iloc[0]
    signgd_thr = 1.5625e-14  # L(eta/2) for eta=0.001, k=4
    assert row["final_loss"] > signgd_thr
    assert row["theoretical_regime"] == "III_signgd"


def test_claim5_theoretical_boundaries_correct():
    """Verify the regime boundaries match Theorem 6.1 for k=4."""
    from phase_diagram import stable_bound, exist_bound, theoretical_regime
    k = 4
    # For k=4: stable_bound = beta2^1, exist_bound = beta2^0.75
    b2 = 0.93
    assert math.isclose(stable_bound(k, b2), 0.93, rel_tol=1e-10)
    assert math.isclose(exist_bound(k, b2), 0.93 ** 0.75, rel_tol=1e-10)
    # Check classification
    assert theoretical_regime(k, 0.90, 0.93) == "I_stable"
    assert theoretical_regime(k, 0.94, 0.93) == "II_spike"
    assert theoretical_regime(k, 0.99, 0.93) == "III_signgd"


# ---------------------------------------------------------------------------
# Claim 6: empirical phase diagram matches Eqs 8-9
# ---------------------------------------------------------------------------

def test_claim6_phase_diagram_files_exist():
    assert (OUT / "phase_diagram_k4.csv").exists()
    assert (OUT / "phase_diagram_k6.csv").exists()
    assert (OUT / "phase_diagram.png").exists()


def test_claim6_stability_boundary_accuracy():
    """The I-vs-non-I classification accuracy must be very high (>95%)."""
    import json
    s = json.loads((OUT / "summary.json").read_text())
    for k in ("4", "6"):
        m = s["claim_6_boundary_metrics"][k]
        assert m["i_vs_noni_accuracy"] > 0.95


def test_claim6_boundary_curve_matches_theory():
    """Empirical boundary mean relative error < 15%."""
    import json
    s = json.loads((OUT / "summary.json").read_text())
    for k in ("4", "6"):
        m = s["claim_6_boundary_metrics"][k]
        assert m["boundary_mean_rel_error"] < 0.15


def test_claim6_negative_control_fails():
    """Wrong-exponent boundary must fit significantly worse (>1.5x error)."""
    import json
    s = json.loads((OUT / "summary.json").read_text())
    for k in ("4", "6"):
        m = s["claim_6_boundary_metrics"][k]
        assert m["control_wrong_exp_mean_rel_error"] > m["boundary_mean_rel_error"] * 1.5


def test_claim6_phase_grid_has_all_regimes():
    """The phase diagram grid must contain cells from all three theoretical regimes."""
    df4 = pd.read_csv(OUT / "phase_diagram_k4.csv")
    regimes = set(df4.theoretical_regime.unique())
    assert regimes == {"I_stable", "II_spike", "III_signgd"}
