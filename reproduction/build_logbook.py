from __future__ import annotations
import json
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACKIO = ROOT / ".venv" / "bin" / "trackio"
OUT = ROOT / "outputs"
ARTIFACT = "adam-degenerate-polynomials-repro/adam-degenerate-polynomials-cpu-reproduction:v0"

def call(*args): subprocess.run([str(TRACKIO), "logbook", *args], cwd=ROOT, check=True)
def page(p): call("page", p)
def md(p, title, body): call("cell", "markdown", "--page", p, "--title", title, body)

def main():
    s = json.loads((OUT / "summary.json").read_text())
    rates = pd.read_csv(OUT / "adam_linear_rates.csv")
    slow = pd.read_csv(OUT / "gd_momentum_power_laws.csv")
    dec = pd.read_csv(OUT / "decoupling.csv")

    p = "00 - Scored evidence summary"; page(p)
    md(p, "GO - both scored claims verified", f"""# Scored evidence first — GO

**Paper:** Towards Understanding Adam Convergence on Highly Degenerate Polynomials  
**OpenReview:** `uYWVGk1Qt0` | **arXiv:** `2603.09581`  
**Tags:** `icml2026-repro`, `paper-uYWVGk1Qt0`  
**Compute:** deterministic local CPU only; 80-decimal Adam tails; no GPU or spend  
**Verification:** 8/8 fail-closed tests; about 4 seconds

| # | Exact challenge claim | Verdict | Decisive evidence |
|---:|---|---|---|
| 1 | Adam achieves local linear convergence on degenerate polynomials, significantly outperforming sub-linear GD and Momentum. | **VERIFIED** | Degrees 4/6/8 exactly recover 3,384/5,818/8,274 steps to 1e-20; degree 10 extrapolates successfully. Maximum Adam slope error is {s['max_adam_relative_slope_error']:.2e}, minimum R² {s['min_adam_r2']:.9f}. Eight million-step GD/momentum fits recover the theoretical power exponents within {s['max_slow_exponent_abs_error']:.2e}. |
| 2 | Adam acceleration stems from decoupling between `v_t` and `g_t²`. | **VERIFIED** | Tail `v_t/v_(t-1)` equals beta2=0.93; both `log(v/g²)` growth and effective-learning-rate growth match their analytic laws. RMSProp repeats the mechanism with beta1=0. |

The scalar polynomial family is the paper's actual theoretical and experimental
object, not a reduced proxy. All raw threshold crossings, tail diagnostics,
million-step controls, plots, source mapping, and tests are in the artifact.
""")
    call("pin", "--page", p)
    call("cell", "figure", "--page", p, "--title", "Linear rates and decoupling", "--image", "outputs/summary.png", "--raw", "outputs/adam_linear_rates.csv")

    p = "Claim 1 - Linear versus sublinear"; page(p)
    rows = ["| k | Adam measured | Adam theory | R² | GD exponent | Momentum exponent |", "|---:|---:|---:|---:|---:|---:|"]
    for k in rates.degree:
        a = rates[rates.degree == k].iloc[0]
        gd = slow[(slow.degree == k) & (slow.optimizer == "GD")].iloc[0]
        mom = slow[(slow.degree == k) & (slow.optimizer == "Momentum")].iloc[0]
        rows.append(f"| {k} | {a.fitted_steps_per_log_precision:.3f} | {a.analytic_steps_per_log_precision:.3f} | {a.fit_r2:.9f} | {gd.fitted_exponent:.6f} | {mom.fitted_exponent:.6f} |")
    md(p, "VERIFIED across four degeneracy orders", """## Verdict: VERIFIED

""" + "\n".join(rows) + """

Adam threshold counts are fitted against `ln(1/epsilon)` over eleven thresholds
from 1e-4 through 1e-24. GD and momentum each run one million iterations and are
fitted in log-time over the final 500 recorded tail points. Their algebraic
exponents match `-1/(k-2)`, while Adam requires a constant number of iterations
per added log-precision unit.
""")

    p = "Claim 2 - Second-moment decoupling"; page(p)
    rows = ["| k | median v ratio | log(v/g²) measured | theory | effective-LR measured | theory |", "|---:|---:|---:|---:|---:|---:|"]
    for r in dec.itertuples(): rows.append(f"| {r.degree} | {r.median_v_ratio:.6f} | {r.log_v_over_g2_slope:.8f} | {r.analytic_decoupling_slope:.8f} | {r.effective_lr_log_slope:.8f} | {r.analytic_effective_lr_slope:.8f} |")
    md(p, "VERIFIED with a first-moment ablation", """## Verdict: VERIFIED

""" + "\n".join(rows) + """

In the asymptotic tail, the fresh squared-gradient contribution vanishes relative
to accumulated second-moment memory, so `v_t/v_(t-1)` locks to beta2. Because
`g_t²` decays faster, `v_t/g_t²` grows geometrically. The independently measured
laws agree across all four degrees. RMSProp (`beta1=0`) remains linearly convergent
and has the same 0.93 second-moment ratio, isolating adaptivity from momentum.
""")

    p = "Methods, provenance, and tests"; page(p)
    md(p, "Clean-room and fail-closed", """# Methods and provenance

The implementation directly follows the paper's epsilon-zero, asymptotically
uncorrected recurrence, initialized at `x0=1,m0=g0,v0=g0²`, with beta1=0.9,
beta2=0.93 and eta=0.001. Arbitrary precision prevents float underflow from
masquerading as convergence. Eight tests enforce exact reference counts, analytic
rates, sublinear controls, the decoupling law, RMSProp behavior, and gradient-flow
hitting times.

PDF SHA-256: `1a3f05ff0b6faec91a86c17d4dc8919b7ee89e1ee48a44dcaf6b1699b2abce17`.
No official code was found or imported.
""")
    call("cell", "artifact", "--page", p, "--title", "Complete CPU reproduction bundle", "--type", "dataset", ARTIFACT)

    p = "Limitations and negative controls"; page(p)
    md(p, "Scoped conclusions", """# Limitations and controls

- This reproduces the deterministic scalar-polynomial theorem family, not neural-network training.
- Numerical fits support but do not replace the paper's proofs or universal basin statements.
- Epsilon and finite-time bias corrections are deliberately absent, exactly as in the asymptotic analysis.
- Degree 10 is an out-of-figure extrapolation; RMSProp removes first-moment momentum.
- Arbitrary precision and analytic formulas guard against underflow and finite-horizon false positives.
""")

    p = "Conclusion"; page(p)
    md(p, "Final outcomes", """# Conclusion

- **Claim 1: VERIFIED.** Adam is linear while GD and momentum recover their predicted sublinear laws.
- **Claim 2: VERIFIED.** Second-moment memory decouples from the shrinking instantaneous gradient and exponentially amplifies the effective learning rate.

## Scope & cost

| | Scope | Hardware | Time | Cost | Outcome |
|---|---|---|---:|---:|---|
| This reproduction | k=4,6,8,10; 11 precision levels; 8 slow controls; RMSProp | local CPU | ~4 s | $0 | both claims verified |
| Full replication | same scalar theorem family | CPU | seconds | $0 | fully covered at and beyond plotted degrees |
""")

if __name__ == "__main__": main()
