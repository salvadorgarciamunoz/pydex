from pydex.core.designer import Designer
from case_1_model import simulate, build_pyomo_model
import numpy as np

""" computing experiment with a scipy's integrator """
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
    optimize_sampling_times=True,
)

designer_1.print_optimal_candidates()
for n_exp in [2, 3, 4, 5, 6]:
    designer_1.apportion(n_exp)
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
