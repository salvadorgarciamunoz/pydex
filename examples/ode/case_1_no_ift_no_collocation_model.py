"""
case_1_no_ift_no_collocation_model.py
======================================
First-order reaction:  A -> B,   dCA/dt = -k * CA,   CA(0) = CA0

Variant of case_1_no_ift_model.py that integrates the Pyomo.DAE model
using the Pyomo Simulator (scipy/vode) instead of Lagrange-Radau
orthogonal collocation.

Two functions are provided:

  build_pyomo_model()
      Pyomo.DAE model integrated by the Pyomo Simulator.  The DAE is
      NOT discretised — DerivativeVar components remain active after
      integration.  Returns a tuple in the IFT contract format for
      structural consistency, but is intentionally NOT assigned to
      designer.pyomo_model_fn.

      NOTE: If this function were assigned to designer.pyomo_model_fn,
      the safety check in designer.py would fire and raise a RuntimeError,
      because active DerivativeVar components signal that the model has
      not been discretised into the algebraic form required by PyomoNLP
      for IFT sensitivity computation.

  simulate()
      Thin wrapper around build_pyomo_model() that extracts the response
      array from the integrated Pyomo model and returns it in the format
      designer.simulate expects.  Assigned to designer.simulate only.

      Sensitivities are computed by pydex via finite differences on top of
      simulate() — no IFT, no PyomoNLP Jacobian, no collocation.

This example demonstrates:
  1. The wrapper pattern with a Pyomo Simulator (integration) backend.
  2. Why assigning an integrated (non-discretised) Pyomo model to
     designer.pyomo_model_fn would correctly trigger the safety switch.

NOTE: The optimal design should match case_1_no_ift.py — both use FD
sensitivities and accurate CA values, just via different solvers.
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# Integration settings
NUMPOINTS = 200   # number of time points passed to the Simulator


# =============================================================================
# build_pyomo_model — Pyomo Simulator (scipy/vode)
# NOT assigned to designer.pyomo_model_fn — would trigger the safety switch
# =============================================================================

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None,
                      numpoints=NUMPOINTS):
    """
    Build and integrate a Pyomo.DAE model for dCA/dt = -k*CA using
    the Pyomo Simulator (scipy/vode).

    The DAE is NOT discretised — DerivativeVar components remain active.
    If this function were assigned to designer.pyomo_model_fn, the safety
    check in designer.py would detect active DerivativeVar components and
    raise a RuntimeError, since IFT requires a fully discretised algebraic
    system.

    IC is set by fixing ca[0] directly — the Pyomo Simulator does not
    support equality constraints for initial conditions.

    Parameters
    ----------
    ti_controls      : array-like  [CA0]
    model_parameters : array-like  [k]
    sampling_times   : array-like or None  — measurement times (absolute)
    numpoints        : int  — number of time points for the Simulator

    Returns  (simplified — IFT contract not used here)
    -------
    m           : integrated ConcreteModel
    tsim        : np.ndarray — Simulator's internal time grid (numpoints,)
    ca_profile  : np.ndarray — CA values at each tsim point (numpoints,)
    """
    k_val   = float(model_parameters[0])
    CA0_val = float(ti_controls[0])

    spt_abs = np.asarray(sampling_times, dtype=float)
    t_final = float(np.max(spt_abs))

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(0.0, t_final))

    m.k = pyo.Var(initialize=k_val);  m.k.fix(k_val)

    m.ca     = pyo.Var(m.t, initialize=CA0_val, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    # DerivativeVar remains active — no disc.apply_to() call.
    # This is what triggers the safety switch if pyomo_model_fn is assigned.

    def material_balance_rule(m, t):
        return m.dca_dt[t] == -m.k * m.ca[t]

    m.material_balance = pyo.Constraint(m.t, rule=material_balance_rule)

    # IC set by fixing ca[0] — the Simulator does not support equality
    # constraints for initial conditions (unlike the collocation path)
    m.ca[0].fix(CA0_val)

    # Integrate using scipy's VODE integrator via the Pyomo Simulator.
    # The Simulator builds its own internal time grid (tsim) — use that
    # directly as t_sorted_full so the interpolation grid is always dense.
    simulator = dae.Simulator(m, package='scipy')
    tsim, profiles = simulator.simulate(numpoints=numpoints, integrator='vode')
    simulator.initialize_model()

    # tsim is the Simulator's internal time grid — always numpoints long
    t_sorted_full = list(tsim.flatten())

    # profiles columns correspond to simulated state variables in order.
    # For this model: ca is the only state, so profiles[:,0] is CA(t).
    # Return tsim and ca_profile directly — the simulate wrapper uses these
    # for interpolation, avoiding any m.ca[t] key lookup on the sparse m.t.
    ca_profile = profiles[:, 0]

    return m, tsim, ca_profile


# =============================================================================
# simulate — wrapper around build_pyomo_model
# Assigned to designer.simulate  (pydex signature 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Thin wrapper around build_pyomo_model().

    Integrates the Pyomo.DAE model and extracts CA at each requested
    sampling time via cubic interpolation over the integration grid.

    pydex calls this function to:
      - evaluate responses for all candidates during initialize()
      - compute finite-difference sensitivities during design_experiment()
      - extract predictions for plotting

    Parameters
    ----------
    ti_controls      : array-like  [CA0]
    sampling_times   : array-like  measurement times
    model_parameters : array-like  [k]

    Returns
    -------
    ca : np.ndarray, shape (n_spt,)
    """
    # Flatten to 1D and strip non-finite values (pydex may pass NaN padding)
    spt = np.asarray(sampling_times, dtype=float).flatten()
    spt = spt[np.isfinite(spt)]

    m, tsim, ca_profile = build_pyomo_model(ti_controls, model_parameters, spt)

    # Interpolate CA from the Simulator's dense time grid to the
    # requested sampling times.
    ca_interp = interp1d(tsim.flatten(), ca_profile, kind='cubic',
                         assume_sorted=True)
    return ca_interp(spt)


# =============================================================================
# Main: quick sanity check
# =============================================================================

if __name__ == '__main__':
    spt = np.linspace(0.1, 10, 11)  # avoid t=0 for interpolation check

    y = simulate(
        ti_controls=[1],
        sampling_times=spt,
        model_parameters=[0.25],
    )

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(spt, y, label='Pyomo Simulator (vode)')
    axes.set_xlabel('Time')
    axes.set_ylabel('$C_A$')
    axes.set_title('First-order reaction  (k=0.25, CA0=1)  —  no IFT, no collocation')
    axes.legend()
    plt.show()
