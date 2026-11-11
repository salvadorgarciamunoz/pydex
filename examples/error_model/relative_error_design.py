"""
relative_error_design.py
========================
How the measurement-error model changes WHEN you should measure.

First-order reaction  A -> B,   A(t) = A0*exp(-k*t),   parameters [k, A0].
One response, A, measured at a sampling time chosen from [0, 10]. Two
parameters need two support points, so the design is a choice of two times.

Why this matters
----------------
With a constant measurement variance (``error_cov``) a late, small reading of A
carries the same weight as an early, large one, so the design stays where A is
still large. For most analytics the error is substantially RELATIVE: a reading
of 0.05 is good to a few per cent of 0.05, not to the same absolute amount as a
reading of 1.0. The two-component model

        variance = (rel_sd * y)**2 + abs_sd**2

captures both, and pydex weights each (candidate, sampling time) by the
reciprocal of that variance at the NOMINAL predicted value y. Information from
the tail is then worth more than a constant-variance design thinks, and the
optimal second time moves later.

How far it moves is governed by ``abs_sd``, the floor under the relative term:
it is what stops the design chasing a reading predicted near zero. The runs
below show that at k = 0.5, where the constant-error design is {0, 2}:

    abs_sd = 0.02    second time 2.8
    abs_sd = 0.005   second time 4.4
    abs_sd = 0.001   second time 7.0

so ``abs_sd`` is not a detail to guess.

Choosing the numbers
--------------------
Both ``rel_sd`` and ``abs_sd`` are ONE STANDARD DEVIATION.

  * "+/-10%" quoted as a 95% interval is rel_sd = 0.05, NOT 0.10.
  * abs_sd is the instrument's absolute precision near zero (baseline, blank
    variability). It is NOT the limit of quantification: LOQ is conventionally
    about 10 sigma, so divide the LOQ by ten.
  * abs_sd must be > 0. Without it a response predicted near zero gets an
    unbounded weight; pydex refuses a zero, and warns when one response's
    weights span more than 1e6 across the grid.

Names, not positions
--------------------
The keys of ``error_variance`` are matched EXACTLY against ``response_names``
(one per measured response). A typo raises instead of silently assigning a
variance to the wrong response, and the order of the dict is irrelevant. The
last block below shows the typo being refused.
"""
import numpy as np
from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    t = ti_controls[0]
    k, A0 = model_parameters
    return np.array([A0 * np.exp(-k * t)])


T_CANDIDATES = np.linspace(0.0, 10.0, 51).reshape(-1, 1)
THETA = np.array([0.5, 1.0])      # [k, A0]


def make_designer(error_variance):
    d = Designer()
    d.simulate               = simulate
    d.model_parameters       = THETA
    d.model_parameter_names  = ["k", "A0"]
    d.response_names         = ["A"]
    d.ti_controls_candidates = T_CANDIDATES
    d.error_variance         = error_variance
    d.initialize(verbose=0)
    return d


def support_points(error_variance):
    d = make_designer(error_variance)
    d.design_experiment(
        d.d_opt_criterion,
        solver="ipopt",
        solver_options={"linear_solver": "ma57", "tol": 1e-10, "max_iter": 3000},
        write=False,
    )
    # read the efforts before any reporting call: printing a design trims it
    e = np.array(d.efforts, dtype=float).ravel()
    t = T_CANDIDATES.ravel()
    return [(float(t[i]), float(e[i])) for i in range(len(e)) if e[i] > 1e-4]


def line(label, pts):
    times = ",  ".join(f"t = {t:4.1f} ({p:.2f})" for t, p in sorted(pts))
    print(f"  {label:<58} {times}")


if __name__ == "__main__":
    print("\n  D-optimal sampling times for A(t) = A0*exp(-k t), k = 0.5, A0 = 1")
    print("  (effort in brackets)\n")

    line("constant error      (abs_sd = 0.02)",
         support_points({"A": {"abs_sd": 0.02}}))
    for abs_sd in (0.02, 0.005, 0.001):
        line(f"relative error      (rel_sd = 0.05, abs_sd = {abs_sd})",
             support_points({"A": {"rel_sd": 0.05, "abs_sd": abs_sd}}))

    print("\n  A misspelt response name is refused, not guessed:")
    try:
        make_designer({"a": {"rel_sd": 0.05, "abs_sd": 0.02}})
    except ValueError as exc:
        print(f"    ValueError: {exc}")
