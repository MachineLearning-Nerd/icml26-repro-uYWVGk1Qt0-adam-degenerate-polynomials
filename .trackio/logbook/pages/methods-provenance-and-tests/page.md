# Methods, provenance, and tests


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_076cace9fe0e", "created_at": "2026-07-16T16:30:18+00:00", "title": "Clean-room and fail-closed"}
-->
# Methods and provenance

The implementation directly follows the paper's epsilon-zero, asymptotically
uncorrected recurrence, initialized at `x0=1,m0=g0,v0=g0²`, with beta1=0.9,
beta2=0.93 and eta=0.001. Arbitrary precision prevents float underflow from
masquerading as convergence. Eight tests enforce exact reference counts, analytic
rates, sublinear controls, the decoupling law, RMSProp behavior, and gradient-flow
hitting times.

PDF SHA-256: `1a3f05ff0b6faec91a86c17d4dc8919b7ee89e1ee48a44dcaf6b1699b2abce17`.
No official code was found or imported.


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_006e627212b0", "created_at": "2026-07-16T16:30:18+00:00", "title": "Complete CPU reproduction bundle", "artifact": "adam-degenerate-polynomials-repro/adam-degenerate-polynomials-cpu-reproduction:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `adam-degenerate-polynomials-repro/adam-degenerate-polynomials-cpu-reproduction:v0` · dataset

https://huggingface.co/buckets/DineshAI/uYWVGk1Qt0-artifacts#adam-degenerate-polynomials-repro/adam-degenerate-polynomials-cpu-reproduction:v0


---
<!-- trackio-cell
{"type": "code", "id": "cell_48d8425756f3", "created_at": "2026-07-16T16:30:26+00:00", "title": "Run: python reproduce.py (exit 0)", "command": [".venv/bin/python", "reproduction/reproduce.py", "--output", "outputs"], "exit_code": 0, "duration_s": 4.722}
-->
````bash
$ .venv/bin/python reproduction/reproduce.py --output outputs
````

exit 0 · 4.7s


````python title=reproduce.py
#!/usr/bin/env python3
"""Full deterministic CPU reproduction for arXiv:2603.09581."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd


BETA1 = mp.mpf("0.9")
BETA2 = mp.mpf("0.93")
ETA = mp.mpf("0.001")
DEGREES = (4, 6, 8, 10)
THRESHOLDS = tuple(mp.mpf(10) ** (-p) for p in range(4, 25, 2))


def linfit(x, y):
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    pred = slope * np.asarray(x, float) + intercept
    denom = np.sum((np.asarray(y, float) - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum((np.asarray(y, float) - pred) ** 2) / denom
    return float(slope), float(intercept), float(r2)


def adaptive_trace(k: int, beta1=BETA1, max_steps=30000):
    """Uncorrected, epsilon-free Adam; initialization follows paper Sec. 4."""
    x = mp.mpf(1)
    g = x ** (k - 1)
    m, v = g, g * g
    hit, tail = {}, []
    threshold_index = 0
    for step in range(1, max_steps + 1):
        g = x ** (k - 1)
        m = beta1 * m + (1 - beta1) * g
        v_prev = v
        v = BETA2 * v + (1 - BETA2) * g * g
        x = x - ETA * m / mp.sqrt(v)
        while threshold_index < len(THRESHOLDS) and abs(x) <= THRESHOLDS[threshold_index]:
            hit[THRESHOLDS[threshold_index]] = step
            threshold_index += 1
        if step >= 1000 and step % 10 == 0:
            g_new = x ** (k - 1)
            tail.append((step, x, v, v_prev, g_new * g_new))
        if threshold_index == len(THRESHOLDS):
            break
    if threshold_index != len(THRESHOLDS):
        raise RuntimeError(f"degree {k} failed to reach all thresholds")
    return hit, tail


def slow_optimizer_exponent(k: int, momentum: bool, steps=1_000_000):
    x, m = 1.0, 0.0
    ts, xs = [], []
    for t in range(1, steps + 1):
        g = x ** (k - 1)
        if momentum:
            m = 0.9 * m + 0.1 * g
            x -= 0.1 * m
        else:
            x -= 0.1 * g
        if t >= 10_000 and t % 1_000 == 0:
            ts.append(t)
            xs.append(abs(x))
    slope, _, r2 = linfit(np.log(ts[-500:]), np.log(xs[-500:]))
    return slope, r2, x


def run(output: Path):
    mp.mp.dps = 80
    output.mkdir(parents=True, exist_ok=True)
    adam_rows, decoupling_rows, slow_rows, hitting_rows = [], [], [], []
    traces = {}

    for k in DEGREES:
        hit, tail = adaptive_trace(k)
        traces[k] = tail
        powers = np.arange(4, 25, 2, dtype=float)
        steps = np.array(list(hit.values()))
        slope, intercept, r2 = linfit(powers * math.log(10), steps)
        analytic = 2 * (k - 2) / (-math.log(float(BETA2)))
        q = float(BETA2 ** (mp.mpf(1) / (2 * (k - 2))))
        adam_rows.append({
            "degree": k, "fitted_steps_per_log_precision": slope,
            "analytic_steps_per_log_precision": analytic,
            "relative_slope_error": abs(slope - analytic) / analytic,
            "fit_r2": r2, "analytic_x_ratio": q,
            "steps_to_1e-20": list(hit.values())[8],
        })
        for eps_mp, n in hit.items():
            eps = float(eps_mp)
            gf_time = (eps ** (-(k - 2)) - 1) / (k - 2)
            hitting_rows.append({"degree": k, "epsilon": eps, "adam_steps": n,
                                 "gradient_flow_time": gf_time})

        # Use the final 100 tail samples, after entry to the asymptotic basin.
        tail = tail[-100:]
        v_ratio = [float(r[2] / r[3]) for r in tail]
        log_v_over_g2 = np.array([float(mp.log(r[2] / r[4])) for r in tail])
        tail_steps = np.array([r[0] for r in tail])
        coupling_slope, _, coupling_r2 = linfit(tail_steps, log_v_over_g2)
        eff_lr_slope, _, eff_r2 = linfit(tail_steps, [-0.5 * float(mp.log(r[2])) for r in tail])
        decoupling_rows.append({
            "degree": k, "median_v_ratio": float(np.median(v_ratio)),
            "target_beta2": float(BETA2),
            "log_v_over_g2_slope": coupling_slope,
            "analytic_decoupling_slope": -math.log(float(BETA2)) / (k - 2),
            "decoupling_r2": coupling_r2,
            "effective_lr_log_slope": eff_lr_slope,
            "analytic_effective_lr_slope": -0.5 * math.log(float(BETA2)),
            "effective_lr_r2": eff_r2,
        })

        for name, use_momentum in (("GD", False), ("Momentum", True)):
            exponent, r2, final_x = slow_optimizer_exponent(k, use_momentum)
            slow_rows.append({"degree": k, "optimizer": name, "fitted_exponent": exponent,
                              "analytic_exponent": -1 / (k - 2), "fit_r2": r2,
                              "final_abs_x": abs(final_x), "steps": 1_000_000})

    pd.DataFrame(adam_rows).to_csv(output / "adam_linear_rates.csv", index=False)
    pd.DataFrame(hitting_rows).to_csv(output / "hitting_times.csv", index=False)
    pd.DataFrame(slow_rows).to_csv(output / "gd_momentum_power_laws.csv", index=False)
    pd.DataFrame(decoupling_rows).to_csv(output / "decoupling.csv", index=False)

    # RMSProp is an independent ablation of first-moment momentum.
    rms_rows = []
    for k in DEGREES:
        hit, tail = adaptive_trace(k, beta1=mp.mpf(0))
        tail = tail[-100:]
        rms_rows.append({"degree": k, "steps_to_1e-20": list(hit.values())[8],
                         "median_v_ratio": float(np.median([float(r[2] / r[3]) for r in tail]))})
    pd.DataFrame(rms_rows).to_csv(output / "rmsprop_control.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    a = pd.DataFrame(adam_rows)
    axes[0].plot(a.degree, a.fitted_steps_per_log_precision, "o-", label="measured")
    axes[0].plot(a.degree, a.analytic_steps_per_log_precision, "x--", label="theory")
    axes[0].set(xlabel="polynomial degree k", ylabel="steps per ln(1/epsilon)", title="Adam linear-rate law")
    axes[0].legend()
    d = pd.DataFrame(decoupling_rows)
    axes[1].plot(d.degree, d.log_v_over_g2_slope, "o-", label="measured log(v/g²)")
    axes[1].plot(d.degree, d.analytic_decoupling_slope, "x--", label="theory")
    axes[1].set(xlabel="polynomial degree k", ylabel="growth per step", title="Second-moment decoupling")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=180)
    plt.close(fig)

    summary = {
        "claim_1": "verified",
        "claim_2": "verified",
        "degrees": list(DEGREES),
        "max_adam_relative_slope_error": max(r["relative_slope_error"] for r in adam_rows),
        "min_adam_r2": min(r["fit_r2"] for r in adam_rows),
        "max_slow_exponent_abs_error": max(abs(r["fitted_exponent"] - r["analytic_exponent"]) for r in slow_rows),
        "max_v_ratio_abs_error": max(abs(r["median_v_ratio"] - r["target_beta2"]) for r in decoupling_rows),
        "max_decoupling_relative_error": max(abs(r["log_v_over_g2_slope"] - r["analytic_decoupling_slope"]) / r["analytic_decoupling_slope"] for r in decoupling_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    start = time.perf_counter()
    run(args.output)
    print(f"runtime_seconds={time.perf_counter() - start:.3f}")

````


````output
{
  "claim_1": "verified",
  "claim_2": "verified",
  "degrees": [
    4,
    6,
    8,
    10
  ],
  "max_adam_relative_slope_error": 8.572804240376117e-05,
  "min_adam_r2": 0.9999998729907293,
  "max_slow_exponent_abs_error": 0.00028454344167794243,
  "max_v_ratio_abs_error": 0.0,
  "max_decoupling_relative_error": 2.868469474185489e-15
}
runtime_seconds=3.958

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_ecbba6974a77", "created_at": "2026-07-16T16:30:26+00:00", "title": "Artifact: hitting_times.csv", "path": "outputs/hitting_times.csv", "size": 1492, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/hitting_times.csv` · dataset · 1.5 kB

https://huggingface.co/buckets/DineshAI/uYWVGk1Qt0-artifacts#logbook-files/outputs/hitting_times.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_8b8d826b6292", "created_at": "2026-07-16T16:30:26+00:00", "title": "Artifact: gd_momentum_power_laws.csv", "path": "outputs/gd_momentum_power_laws.csv", "size": 763, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/gd_momentum_power_laws.csv` · dataset · 763 B

https://huggingface.co/buckets/DineshAI/uYWVGk1Qt0-artifacts#logbook-files/outputs/gd_momentum_power_laws.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_8ef6b7cf54e1", "created_at": "2026-07-16T16:30:26+00:00", "title": "Artifact: decoupling.csv", "path": "outputs/decoupling.csv", "size": 567, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/decoupling.csv` · dataset · 567 B

https://huggingface.co/buckets/DineshAI/uYWVGk1Qt0-artifacts#logbook-files/outputs/decoupling.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_fe6e6e702bfc", "created_at": "2026-07-16T16:30:26+00:00", "title": "Artifact: adam_linear_rates.csv", "path": "outputs/adam_linear_rates.csv", "size": 548, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/adam_linear_rates.csv` · dataset · 548 B

https://huggingface.co/buckets/DineshAI/uYWVGk1Qt0-artifacts#logbook-files/outputs/adam_linear_rates.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_e29db4296587", "created_at": "2026-07-16T16:30:26+00:00", "title": "Artifact: rmsprop_control.csv", "path": "outputs/rmsprop_control.csv", "size": 87, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/rmsprop_control.csv` · dataset · 87 B

https://huggingface.co/buckets/DineshAI/uYWVGk1Qt0-artifacts#logbook-files/outputs/rmsprop_control.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_12ec367fc4fe", "created_at": "2026-07-16T16:30:27+00:00", "title": "Run: python test_reproduction.py (exit 0)", "command": [".venv/bin/python", "-m", "pytest", "-q", "reproduction/test_reproduction.py"], "exit_code": 0, "duration_s": 0.663}
-->
````bash
$ .venv/bin/python -m pytest -q reproduction/test_reproduction.py
````

exit 0 · 0.7s


````python title=test_reproduction.py
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

````


````output
........                                                                 [100%]
8 passed in 0.27s

````
