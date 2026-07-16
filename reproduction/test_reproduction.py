import math
from pathlib import Path

import pandas as pd


OUT = Path(__file__).parents[1] / "outputs"


def test_all_claims_verified():
    import json
    s = json.loads((OUT / "summary.json").read_text())
    assert s["claim_1"] == s["claim_2"] == "verified"


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
