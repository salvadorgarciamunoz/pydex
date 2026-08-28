"""
van_laar_design.py
==================
D-optimal design for fitting the two van Laar parameters (A12, A21) from
isothermal total-pressure and vapour-composition measurements.

A STATIC model that is nonlinear in its parameters. Contrast with a linear
response surface, whose optimal design sits on the corners of the control space
whatever the parameter values: here the design depends on the nominal
parameters, because the sensitivities do.

Sampling times do not appear anywhere in this script -- there is no time axis,
so `n_spt` does not apply. See examples/ode/ for the dynamic cases.

Naming
------
The four labelling attributes are set below. They are optional to the
mathematics and worth setting anyway: they put real names into the design
table and onto the plot axes. `model_parameter_names` is the one that is more
than cosmetic, since `interest_parameters` is matched against it by name.

Measured results (pydex 0.7.4, 21 candidates, D-optimal)
--------------------------------------------------------
    Criterion value            16.782682463020976
    Optimal candidates         2 of 21

    Experiment  Candidate    x1      T   Effort
             1          3  0.05   80.0   47.46%
             2          9  0.35   80.0   52.54%

    apportion(8)  ->  4 runs and 4 runs
    rounded design is 99.88% as informative as the continuous design

Two support points for two parameters, both at the highest temperature, and
at x1 = 0.05 and 0.35 -- NOT at the corners of the composition range. Nineteen
of the twenty-one candidates receive zero effort.

The design moves with the nominal parameters: at theta = [1.10, 1.45] the
support becomes candidates 6 and 12 and the criterion 16.081177748563384. That
is a property of nonlinear models, and the reason sequential design exists.
"""

import numpy as np

from pydex.core.designer import Designer

from van_laar_model import simulate

designer = Designer()
designer.simulate = simulate

# Nominal parameter values -- the current best guess for A12, A21.
designer.model_parameters = np.array([1.65, 0.95])

# The candidate grid: every experiment we could run.
tic = designer.enumerate_candidates(
    bounds=[
        [0.05, 0.95],    # x1, liquid mole fraction of component 1
        [40.0, 80.0],    # T, degC
    ],
    levels=[
        7,               # 7 compositions
        3,               # 3 temperatures
    ],
)
designer.ti_controls_candidates = tic

# Measurement error covariance, stated explicitly. Two responses on different
# scales, so the identity default would treat a 0.3 kPa pressure error and a
# 0.004 composition error as equally serious.
designer.error_cov = np.diag([0.3 ** 2, 0.004 ** 2])

# Labels. Optional, but they turn "Time-invariant Control 0" into "x1" in the
# design table and label every plot axis. Note the SINGULAR "parameter" in
# model_parameter_names.
designer.model_parameter_names = ["A12", "A21"]
designer.ti_controls_names = ["x1", "T"]
designer.response_names = ["P", "y1"]
designer.model_parameter_unit_names = ["-", "-"]
designer.response_unit_names = ["kPa", "-"]

designer.initialize(verbose=2)

designer.design_experiment(
    designer.d_opt_criterion,
    solver="ipopt",
    write=False,
)

designer.print_optimal_candidates()
designer.print_optimal_candidates_table()
designer.apportion(8)
