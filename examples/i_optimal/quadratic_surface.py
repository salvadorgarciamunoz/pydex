"""
quadratic_surface.py
====================
I-optimal design: experiments that predict well ACROSS A REGION.

Three criteria, three different questions:

  D-optimal   which experiments pin down the PARAMETERS most tightly?
  V-optimal   which predict best AT ONE operating point you have chosen?
  I-optimal   which predict best AVERAGED OVER a whole operating region?

D is the criterion for structural decisions about a model. I is the criterion
for using one: it is the standard choice in response-surface work, because
what you want from a fitted surface is a trustworthy prediction anywhere
inside the envelope you operate in.

The model is a full quadratic in two factors on [-1, 1]^2 -- six parameters,
no dynamics, nothing to integrate. It is deliberately the simplest thing that
shows the difference, and the answer is known in advance from
response-surface theory, so you can check the output rather than trust it.

The syntax
----------
One designer, two designs. A D-optimal design is one call:

    designer.design_experiment("D-optimal", solver="ipopt")

An I-optimal design is the SAME call with a different criterion, plus one
line before it saying WHERE the model has to predict well:

    designer.set_prediction_region(bounds=[(-1, 1), (-1, 1)])
    designer.design_experiment("I-optimal", solver="ipopt")

The criterion name is normalised, so "D-optimal", "d_opt" and "D" all mean
the same thing. A criterion callable such as ``designer.d_opt_criterion``
works too, which is what you need if you write one of your own.

Note that the SAME designer does both. A designer holds a model, a candidate
grid and the nominal parameters; the criterion is a choice made at design
time, not a property of the designer. Running a third criterion on it would
be one more call.

``set_prediction_region()`` is the only new idea. You give it bounds; it
chooses how finely to sample inside them -- refining until the answer stops
moving -- and reports what it used. The bounds must sit INSIDE the candidate
grid, since the candidates are the conditions you have said you can run.

What you should see
-------------------
Both criteria pick the SAME nine support points -- the 3^2 factorial -- and
differ only in how they weight them:

    design        corners      edges        centre
    D-optimal     14.58% ea.   8.02% ea.    9.62%
    I-optimal      9.11% ea.   9.12% ea.   27.09%

Centre weight nearly triples while the corners lose a third. That is the
textbook signature: the centre of a region contributes most to the average
prediction variance, so a design that must predict well everywhere spends
more of its budget there. D-optimal, chasing parameter precision, pushes to
the extremes where the quadratic terms are largest.

Scored both ways, each design through BOTH criteria: I lowers the average
prediction variance by 21.4%, and D wins on log det FIM by 0.608. Neither is
better. They answer different questions, and the two numbers are on
different scales -- criterion values are not comparable across criteria, and
an I-optimal value is not comparable across regions either, since changing
the bounds changes what is being averaged. Compare DESIGNS.

Run
---
    python quadratic_surface.py

Needs an NLP solver. ``solver="pounce"`` installs from pip and needs no
downloaded binaries; ``solver="ipopt"`` works equally well.
"""

import numpy as np

from pydex.core.designer import Designer

SOLVER = "pounce"


def simulate(ti_controls, model_parameters):
    """y = b0 + b1 x1 + b2 x2 + b12 x1 x2 + b11 x1^2 + b22 x2^2."""
    x1, x2 = ti_controls
    basis = np.array([1.0, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    return np.array([basis @ model_parameters])


# ---------------------------------------------------------------------------
# The usual setup, once
# ---------------------------------------------------------------------------
designer = Designer()
designer.simulate = simulate

# This model is LINEAR in its parameters, so the sensitivities do not depend
# on them and neither does the design. The values still have to be set --
# pydex evaluates sensitivities at them either way.
designer.model_parameters = np.ones(6)

designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[[-1, 1], [-1, 1]],
    levels=[5, 5],
)
designer.model_parameter_names = ["b0", "b1", "b2", "b12", "b11", "b22"]
designer.ti_controls_names = ["x1", "x2"]
designer.response_names = ["y"]

designer.initialize(verbose=0)


# ---------------------------------------------------------------------------
# D-optimal: the experiments that pin down the PARAMETERS
# ---------------------------------------------------------------------------
print("\nD-OPTIMAL")

designer.design_experiment("D-optimal", solver=SOLVER)
designer.print_optimal_candidates_table()

efforts_d = designer.efforts.copy()


# ---------------------------------------------------------------------------
# I-optimal: the experiments that predict well OVER THE REGION
#
# The same designer, the same candidate grid. One line says where the model
# has to predict well; the next solves for it.
# ---------------------------------------------------------------------------
print("\nI-OPTIMAL")

region = designer.set_prediction_region(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
print(f"prediction region: {region['rule']} rule, "
      f"{region['nodes']} nodes per factor, {region['n_points']} points, "
      f"converged={region['converged']}")

designer.design_experiment("I-optimal", solver=SOLVER)
designer.print_optimal_candidates_table()

efforts_i = designer.efforts.copy()


# ---------------------------------------------------------------------------
# Score BOTH designs through BOTH criteria
#
# Scoring each design by the criterion it was optimised for would compare
# different quantities. The region is still set from above, so designer.W
# holds the prediction sensitivities over it; eval_fim() rebuilds the FIM for
# whichever effort vector it is given.
# ---------------------------------------------------------------------------
W = np.asarray(designer.W)

designer.eval_fim(efforts_d.flatten())
apv_d = float(np.trace(W @ np.linalg.inv(designer.fim) @ W.T))
logdet_d = float(np.linalg.slogdet(designer.fim)[1])

designer.eval_fim(efforts_i.flatten())
apv_i = float(np.trace(W @ np.linalg.inv(designer.fim) @ W.T))
logdet_i = float(np.linalg.slogdet(designer.fim)[1])

print(f"\n{'':12}{'avg prediction variance':>26}{'log det FIM':>16}")
print(f"{'D-optimal':12}{apv_d:26.6f}{logdet_d:16.6f}")
print(f"{'I-optimal':12}{apv_i:26.6f}{logdet_i:16.6f}")
print(f"\n  I lowers the average prediction variance by "
      f"{100 * (1 - apv_i / apv_d):.1f}%")
print(f"  D wins on log det FIM by {logdet_d - logdet_i:.4f}")
print("\n  Different questions, different answers. "
      "Compare designs, not criterion values.")
