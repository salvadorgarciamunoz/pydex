from pydex.core.designer import Designer
from case_4_model import simulate
import numpy as np

"""
case_4.py
=========
LOCAL D-optimal design for a batch/semi-batch A -> B -> C network with two
rate constants [k1, k2] and one control (feed rate f_in). Finite-difference
sensitivities over forward integration (scipy `vode`) -- no pyomo_model_fn,
so this is the `_no_ift_no_collocation` sensitivity path in the naming
scheme used by the case_1/case_2/case_3 families, even though this file
doesn't carry that suffix (see examples/README.md).

Designs assuming k1 and k2 are known exactly at their nominal values. For
the same idea under parameter UNCERTAINTY -- a scenario array instead of a
single point -- see case_5.py's pseudo-Bayesian design on the Arrhenius
variant of this network. Note that `pseudo_bayesian_type` only takes effect
when `model_parameters` is 2-D; against a 1-D nominal vector like the one
here, it has no effect and the solve stays a local D-optimal design.

`f_in` feeds pure A with no outflow term, so this is a continuously-fed
semi-batch reactor rather than a closed system: total moles grow as
`1 + f_in * t` once `f_in > 0`.

The BAYESIAN INFERENCE section below is not a pydex capability -- it's
ordinary downstream analysis of the completed design, added because it's a
natural next step after designing an experiment: simulate what you'd
actually measure, then check whether you can recover the parameters from
it. See that section's docstring for the approach.
"""

designer = Designer()
designer.simulate = simulate

tic = designer.enumerate_candidates(
    bounds=[
        [0, 0.3],   # feed rate f_in
    ],
    levels=[11],
)
designer.ti_controls_candidates = tic

# Labels for reports and plot axes. Optional, but they replace generated
# defaults like "Time-invariant Control 0" with the real quantity.
designer.model_parameter_names = ["k1", "k2"]
designer.ti_controls_names     = ["f_in"]
designer.response_names        = ["cA", "cB", "cC"]
designer.time_unit_name        = "min"


spt = np.array([np.linspace(0, 10, 101) for _ in tic])
designer.sampling_times_candidates = spt

# nominal rate constants
theta_nom = np.array([1.0, 1.0])   # [k1, k2]
designer.model_parameters = theta_nom

designer.error_cov = np.diag([0.01, 0.01, 0.01])
designer._norm_sens_by_params = True   # already the default; kept for clarity
designer.initialize(verbose=2)
designer.sens_report_freq = 2

""" Local D-optimal design """
designer.design_experiment(
    designer.d_opt_criterion,
    solver="ipopt",
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_predictions()
designer.plot_optimal_sensitivities(interactive=False)
designer.apportion(4)

"""
BAYESIAN INFERENCE -- recovering [k1, k2] from data simulated at the
apportioned design, via PyMC. Written against the public Designer API
(`optimal_candidates`, `apportionments`, `error_cov`) plus the model's own
`simulate()`, so it keeps working if the optimizer picks a different design.

WHY A GRADIENT-FREE SAMPLER
-----------------------------
`simulate()` runs a Pyomo model through scipy's Simulator -- a black box as
far as PyMC is concerned, with no symbolic gradient. It's wrapped with
`pytensor.wrap_py` and sampled with `pm.Metropolis`, since PyMC's default
NUTS sampler requires gradients this Op cannot provide. Multiprocess
sampling (cores > 1) works fine with this wrapper -- PyMC pickles it to
worker processes without issue.

WHAT TO EXPECT
----------------
A 2-parameter local D-optimal design with an unconstrained budget typically
collapses to a single condition run several times, rather than spreading
effort across several -- replicating at the one most informative point.
4 chains x (500 tune + 800 draws) in parallel (N_CORES=4) takes a few
minutes and should recover k1 and k2 close to their true values, with
r_hat typically landing around 1.01-1.04 -- inside PyMC's own "probably
fine, more draws would help" range rather than fully converged. Raise
N_CHAINS/N_TUNE/N_DRAWS for tighter diagnostics, at roughly linear cost in
runtime. N_CORES can be lower than N_CHAINS; PyMC just queues the rest.
"""
if True:
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor import wrap_py
    import arviz as az

    np.random.seed(0)
    N_CHAINS, N_TUNE, N_DRAWS = 4, 500, 800   # see docstring
    N_CORES = 4                                # match to your machine's free cores

    # Pull the apportioned design off the designer rather than hardcoding it,
    # so this keeps working if the optimizer picks something else.
    design_points = []   # (ti_controls, sampling_time, n_replicates)
    for opt_cand, app in zip(designer.optimal_candidates, designer.apportionments):
        tic_i, spt_list = opt_cand[1], opt_cand[3]
        for spt_j, n_reps in zip(spt_list, app):
            n_reps = int(round(n_reps))
            if n_reps > 0:
                design_points.append((tuple(tic_i), float(spt_j), n_reps))

    unique_conditions = [(t, s) for t, s, _ in design_points]
    error_sd = np.sqrt(np.diag(designer.error_cov))
    k_true = theta_nom   # same values the design was built on

    # Synthetic "observed" data: true response + measurement noise, once per
    # replicate. condition_index maps each observation row back to its
    # (ti_controls, sampling_time) so replicates of the same condition don't
    # each trigger a separate simulate() call inside the sampler.
    obs_rows, condition_index = [], []
    for idx, (tic_i, spt_j, n_reps) in enumerate(design_points):
        c_true = simulate(ti_controls=list(tic_i), sampling_times=np.array([spt_j]),
                           model_parameters=k_true)[0]
        for _ in range(n_reps):
            obs_rows.append(c_true + np.random.normal(0, error_sd))
            condition_index.append(idx)
    obs_data = np.array(obs_rows)
    condition_index = np.array(condition_index)

    @wrap_py(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dmatrix])
    def simulate_op(k1, k2):
        return np.array([
            simulate(ti_controls=list(tic_i), sampling_times=np.array([spt_j]),
                      model_parameters=[float(k1), float(k2)])[0]
            for (tic_i, spt_j) in unique_conditions
        ])

    with pm.Model():
        k1 = pm.Uniform("k1", lower=0.01, upper=5.0)   # k >= 0 per the Pyomo model's own bounds
        k2 = pm.Uniform("k2", lower=0.01, upper=5.0)
        preds_unique = simulate_op(k1, k2)
        pm.Normal("obs", mu=preds_unique[condition_index], sigma=error_sd, observed=obs_data)

        idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                           step=pm.Metropolis(), random_seed=0, cores=N_CORES)

    print(az.summary(idata))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(idata.posterior["k1"].values.T, alpha=0.7)
    axes[0].axhline(k_true[0], color="red", ls="--", label="true k1")
    axes[0].set_xlabel("draw"); axes[0].set_ylabel("k1"); axes[0].legend()
    axes[0].set_title("k1 trace (one line per chain)")

    k1_samples = idata.posterior["k1"].values.flatten()
    k2_samples = idata.posterior["k2"].values.flatten()
    axes[1].scatter(k1_samples, k2_samples, s=4, alpha=0.25)
    axes[1].scatter([k_true[0]], [k_true[1]], color="red", marker="x", s=120,
                     label="true (k1, k2)")
    axes[1].set_xlabel("k1"); axes[1].set_ylabel("k2"); axes[1].legend()
    axes[1].set_title("Joint posterior")
    fig.tight_layout()

designer.show_plots()

