from pydex.core.designer import Designer
from case_5_model import simulate
import numpy as np

"""
case_5.py
=========
GENUINE pseudo-Bayesian Type-1 D-optimal design -- the Arrhenius version of
case_4's network, designed under UNCERTAINTY in all four kinetic parameters
rather than at a single nominal guess. `model_parameters` is a 2-D array of
shape (n_scr, 4): each row is one scenario drawn from a uniform prior, and
the type-1 criterion averages det(FIM) OVER SCENARIOS rather than averaging
the information matrices first (type 0) -- see the class docstring on
design_experiment() for the distinction and why type 1 always falls back to
scipy SLSQP rather than a native Pyomo solve.

WHY THIS IS THE "REAL" VERSION OF case_4's IDEA
-------------------------------------------------
case_4.py designs assuming k1, k2 are known exactly. That's fine when you
trust the nominal guess; it says nothing about how the design should change
if you don't. This script is that other case: instead of one point estimate
of [theta0_1, theta1_1, theta0_2, theta1_2], there's a range you're willing
to entertain, and the design should be good on average across that range,
not just at one guess that might be wrong.

A GOTCHA WORTH KNOWING
------------------------
`save_atomics` must be passed as a design_experiment() keyword, not set as
`designer._save_atomics` beforehand -- the keyword's own default silently
overwrites the attribute regardless of what it held going in. Same footgun
PROJECT_NOTES.md documents for `regularize_fim`.

SCENARIO COUNT
----------------
N_SCR defaults to 20 rather than a much larger ensemble: at roughly 13
seconds per scenario on this grid (11 candidates x 21 sampling times), each
additional 100 scenarios costs about 20 more minutes. Raise it for a
smoother pseudo-Bayesian estimate -- the code doesn't change, only the
runtime does.
"""

designer = Designer()
designer.simulate = simulate

tic = designer.enumerate_candidates(
    bounds=[
        [273.15, 323.15],   # reaction temperature (K)
    ],
    levels=[11],
)
designer.ti_controls_candidates = tic

spt = np.array([np.linspace(0, 10, 21) for _ in tic])
designer.sampling_times_candidates = spt

N_SCR = 20   # ~13 s/scenario on this grid -- see "SCENARIO COUNT" above
np.random.seed(0)
mp = np.random.uniform(
    low=[0, 2, -2, 4],
    high=[2, 4, 0, 6],
    size=[N_SCR, 4],
)
designer.model_parameters = mp

# Same measurement-noise assumption as case_4.py. Note: error_cov is a
# uniform diagonal here, so its absolute scale doesn't change WHICH design
# gets picked (every candidate's FIM is scaled by the same constant), but
# it does change the reported criterion value by a large, easy-to-miss
# constant (n_mp * ln(ratio) between two candidate variances) -- worth
# having a real noise assumption behind the number regardless.
designer.error_cov = np.diag([0.01, 0.01, 0.01])

designer.initialize(verbose=2)
designer.sens_report_freq = 2

""" Pseudo-Bayesian Type-1 D-optimal design """
designer.design_experiment(
    designer.d_opt_criterion,
    optimize_sampling_times=True,
    pseudo_bayesian_type=1,
    save_atomics=True,   # must be passed here -- see docstring above
)
designer.print_optimal_candidates()
designer.plot_optimal_predictions()
designer.plot_optimal_sensitivities(interactive=False)

designer.show_plots()
