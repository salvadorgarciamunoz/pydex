from pydex.core.designer import Designer
from case_1_model import simulate, build_pyomo_model
import numpy as np

"""
case_1.py
=========
D-optimal design for the first-order reaction dCA/dt = -k*CA, with exact
sensitivities from the Implicit Function Theorem (pyomo_model_fn assigned).
One parameter, one control, one response -- the smallest useful dynamic
example.

SAMPLING TIMES
--------------
n_spt is omitted, so sampling times are OPTIMIZED: effort is allocated per
(candidate, sampling time) cell and the optimiser chooses which of the 101
listed times to measure. The other two cases are n_spt=k (exactly k samples
per run, optimiser picks which k) and n_spt=<number of listed times> (fixed
grid, every listed time measured on every run, effort per experiment). n_spt
is the only control; there is no flag that switches optimisation on or off.

MEASURED RESULT
---------------
The design collapses onto a single point: candidate 5 (CA0 = 5) sampled at
t = 4.00, carrying 100% of the effort. D-optimal criterion values across the
three sensitivity paths of this family:

    case_1.py                        (colloc+IFT)   1.2188793   <- here
    case_1_no_ift.py                 (colloc+FD)    1.2188792
    case_1_no_ift_no_collocation.py  (scipy+FD)     1.2188777

Because the continuous design already sits on a single support point, every
apportionment is exact: apportion() reports the rounded design as 100.00% as
informative as the continuous one for each of 2, 3, 4, 5 and 6 runs.

This model has n_mp == 1, so the criterion is a 1x1 FIM case. d_opt returns
-log(det(FIM)) there, on the same formula the matrix case uses -- worth knowing
if you compare against a hand-computed value.
"""
designer_1 = Designer()
designer_1.simulate = simulate
designer_1.pyomo_model_fn = build_pyomo_model  # IFT sensitivities via Pyomo
# use_pyomo_ift and n_jobs are auto-set by initialize() when pyomo_model_fn is provided

print("IFT path: Collocation + IPOPT (PyomoNLP)")

theta_nom = np.array([0.25])  # value of k, a 1D np.array with size = 1
designer_1.model_parameters = theta_nom  # assigning it to the designer's theta
tic = designer_1.enumerate_candidates(
    bounds=[
        [0.1, 5],
    ],
    levels=[
        5,
    ],
)
designer_1.ti_controls_candidates = tic

# Labels for reports and plot axes. Optional, but they replace generated
# defaults like "Time-invariant Control 0" with the real quantity.
designer_1.model_parameter_names = ["k"]
designer_1.ti_controls_names     = ["CA0"]
designer_1.response_names        = ["CA"]
designer_1.model_parameter_unit_names = ["1/min"]
designer_1.response_unit_names        = ["mol/L"]
designer_1.time_unit_name             = "min"

designer_1.sampling_times_candidates = np.array([
    np.linspace(0, 50, 101)
    for _ in tic
])
designer_1._num_steps = 15
designer_1.initialize(verbose=2)

"""
===============================================================
[Optional]: check responses and sensitivities of all candidates
===============================================================
"""
if False:
    designer_1.simulate_candidates(plot_simulation_times=True)
    designer_1.plot_predictions()
    sens = designer_1.eval_sensitivities(save_sensitivities=False, store_predictions=True)
    designer_1.plot_sensitivities()

""" solve OED problem """
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
)

designer_1.print_optimal_candidates()
for n_exp in [2, 3, 4, 5, 6]:
    designer_1.apportion(n_exp)
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
