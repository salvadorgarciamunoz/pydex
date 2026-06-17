"""
case_3_model.py
===============
Scipy ODE integration model for the Michaelis-Menten-style reaction network,
used by the finite-difference (FD) sensitivity path in case_3.py.

Reaction system
---------------
    A → B    (irreversible, inhibited power-law rate)

    dCA/dt = -τ * r
    dCB/dt =  τ * ν * r

    r(CA, T) = k1(T) * CA^α / (k2(T) + k3(T) * CA^β)

    ki(T) = exp(θ_i0 + θ_i1 * (T - 273.15) / T)    i = 1, 2, 3

State variables : CA(t), CB(t)   [mol/L]   (normalised time t ∈ [0, 1])
Parameters      : θ = [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]
Controls        : ti_controls = [CA0 (mol/L), T (K), τ]

Dependencies
------------
    numpy, scipy, matplotlib (for __main__ sanity check only)
"""

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from time import time

# ---------------------------------------------------------------------------
# Integration settings
# ---------------------------------------------------------------------------
_RTOL   = 1e-10
_ATOL   = 1e-12
_METHOD = 'Radau'   # implicit RK5 — handles stiff kinetics robustly


# =============================================================================
# simulate — pydex signature 2
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Evaluate the Michaelis-Menten reaction model by direct ODE integration
    and return concentrations at each requested sampling time.

    Pydex signature type 2: simulate(ti_controls, sampling_times, model_parameters)

    Parameters
    ----------
    ti_controls : array-like, length 3
        [CA0 (mol/L), T (K), τ]

    sampling_times : array-like
        Normalised measurement times in [0, 1].  pydex may pad shorter
        candidate rows with NaN — these are stripped before integration.

    model_parameters : array-like, length 9
        [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]

    Returns
    -------
    y : np.ndarray, shape (n_spt, 2)
        Columns: [CA (mol/L), CB (mol/L)] at each sampling time.
        Returns an array of NaN if integration fails, allowing
        numdifftools Richardson extrapolation to skip the evaluation.
    """
    # ── Unpack controls ───────────────────────────────────────────────────
    CA0 = float(ti_controls[0])
    T   = float(ti_controls[1])
    tau = float(ti_controls[2])

    # ── Unpack parameters ─────────────────────────────────────────────────
    theta_10, theta_11 = float(model_parameters[0]), float(model_parameters[1])
    theta_20, theta_21 = float(model_parameters[2]), float(model_parameters[3])
    theta_30, theta_31 = float(model_parameters[4]), float(model_parameters[5])
    nu                 = float(model_parameters[6])
    alpha              = float(model_parameters[7])
    beta               = float(model_parameters[8])

    # ── Pre-compute Arrhenius rate constants (isothermal) ─────────────────
    T_shift = (T - 273.15) / T
    k1 = np.exp(theta_10 + theta_11 * T_shift)
    k2 = np.exp(theta_20 + theta_21 * T_shift)
    k3 = np.exp(theta_30 + theta_31 * T_shift)

    # ── Sampling time processing ──────────────────────────────────────────
    spt = np.asarray(sampling_times, dtype=float).flatten()
    spt = spt[np.isfinite(spt) & (spt >= 0)]
    n_spt = len(spt)
    t_end = float(np.max(spt))

    # ── ODE right-hand side ───────────────────────────────────────────────
    def rhs(t, y):
        CA, CB = y
        CA_pos = max(CA, 0.0)
        if CA_pos == 0.0:
            r = 0.0
        else:
            r = k1 * (CA_pos ** alpha) / (k2 + k3 * (CA_pos ** beta))
        dCA = -tau * r
        dCB =  tau * nu * r
        return [dCA, dCB]

    # ── Integrate ─────────────────────────────────────────────────────────
    try:
        sol = solve_ivp(
            rhs,
            t_span       = (0.0, t_end),
            y0           = [CA0, 0.0],
            method       = _METHOD,
            t_eval       = spt,
            rtol         = _RTOL,
            atol         = _ATOL,
            dense_output = False,
        )
    except Exception:
        return np.full((n_spt, 2), np.nan)

    if not sol.success:
        return np.full((n_spt, 2), np.nan)

    # sol.y shape: (2, n_spt) → transpose to (n_spt, 2)
    return sol.y.T


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == '__main__':
    start = time()
    tic = [10, 303.15, 10]
    mp  = [5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5]
    spt = np.linspace(0.001, 1, 201)

    c  = simulate(tic, spt, mp)
    cA, cB = c[:, 0], c[:, 1]
    print(f"One simulation took {time() - start:.3f} CPU seconds.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spt, cA, label="$c_A$")
    ax.plot(spt, cB, label="$c_B$")
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("Concentration (mol/L)")
    ax.set_title("Michaelis-Menten reaction  (scipy Radau, FD sensitivity path)")
    ax.legend()
    fig.tight_layout()
    plt.show()
