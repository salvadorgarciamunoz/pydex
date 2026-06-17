"""
case_2_no_ift_no_collocation_model.py
======================================
Scipy ODE integration model for the A→B reaction with Arrhenius kinetics,
used by the finite-difference (FD) sensitivity path in
case_2_no_ift_no_collocation.py.

Reaction system
---------------
    A → B    (irreversible, power-law rate)

    dCA/dt = -k * CA^α
    dCB/dt =  ν * k * CA^α

    k(T) = exp(θ₀ + θ₁ * (T - 273.15) / T)      [Arrhenius, reparametrised]

State variables : CA(t), CB(t)   [mol/L]
Parameters      : θ = [θ₀, θ₁, α, ν]
Controls        : ti_controls = [CA0 (mol/L), T (K)]

Relationship to the other case_2 variants
------------------------------------------
Three variants demonstrate the full spectrum of sensitivity methods in pydex:

  case_2.py                       — Pyomo.DAE collocation + IFT (exact symbolic
                                    sensitivities via KKT implicit function
                                    theorem; fastest and most accurate)

  case_2_no_ift.py                — Pyomo.DAE collocation + FD (same NLP model,
                                    finite-difference Jacobian; slower and
                                    requires robustness precautions)

  case_2_no_ift_no_collocation.py — scipy.integrate + FD  ← THIS FILE
                                    (no Pyomo at all; integrate the IVP directly
                                    with an adaptive-step solver; simplest and
                                    most robust FD path)

Why scipy/VODE is more robust for FD than collocation
------------------------------------------------------
The collocation variant solves a large NLP (≈ 285 variables, IPOPT) for every
FD perturbation.  When a perturbed parameter pushes CA toward zero, IPOPT's
interior-point barrier hits the hard lower bound and declares infeasibility —
not because the physics is infeasible, but because the NLP formulation cannot
step past the constraint.  Three architectural workarounds are needed (parameter
clamping, relaxed CA bound, NaN fallback).

scipy.integrate.solve_ivp has none of these problems:

  • No hard bounds — the integrator evaluates the RHS numerically at each
    adaptive step.  If perturbed θ₁ makes k large, the step size shrinks
    automatically.  CA approaching zero does not trigger any constraint
    violation.

  • No symbolic expression tree — CA**α is evaluated as a Python float at
    each timestep.  A perturbed negative α with CA > 0 is still a valid float;
    there is no Pyomo expression construction that fails before integration.

  • Graceful degradation — if integration genuinely fails (e.g. unbounded stiff
    blow-up), solve_ivp returns success=False and a partial trajectory.  We
    detect this explicitly and return NaN, but it only triggers for truly
    pathological perturbations far from the nominal.

The only defensive line needed is a single `max(CA, 0.0)` guard in the RHS to
prevent complex-valued powers near CA = 0.  Everything else follows naturally
from the IVP formulation.

Sensitivity accuracy
--------------------
Both the collocation+FD and scipy+FD variants compute approximate Jacobians
via Richardson extrapolation (numdifftools).  The accuracy is similar for
well-conditioned problems.  The primary advantage of this variant is
*robustness*, not accuracy — the IFT path (case_2.py) provides exact symbolic
sensitivities and should be preferred when available.

Solver choice
-------------
'Radau' is the default solver here.  It is an implicit Runge-Kutta method of
order 5 that handles stiff systems efficiently.  For this A→B reaction the
stiffness arises at high temperatures where k is large and CA decays rapidly.
'LSODA' (Adams/BDF automatic switching) is a good alternative if you are
uncertain about stiffness.  'RK45' works well when the system is not stiff
(low temperature, small k).

Dependencies
------------
    numpy, scipy, matplotlib (for __main__ sanity check only)

Typical usage (from case_2_no_ift_no_collocation.py)
------------------------------------------------------
    from case_2_no_ift_no_collocation_model import simulate
    designer.simulate = simulate
    # designer.pyomo_model_fn is NOT set → FD sensitivity path is used
    # No Pyomo is imported or required.
"""

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Integration settings
# ---------------------------------------------------------------------------
# Relative and absolute tolerances for the IVP solver.  Tighter tolerances
# improve FD Jacobian accuracy by reducing integration noise, at the cost of
# slightly more function evaluations.  1e-10 / 1e-12 is appropriate for the
# Richardson extrapolation step sizes used by numdifftools.
_RTOL = 1e-10
_ATOL = 1e-12

# IVP solver method.  'Radau' (implicit RK5) handles the stiff regime
# (high temperature, large k) robustly.  'LSODA' switches automatically
# between Adams (non-stiff) and BDF (stiff) and is a good alternative.
_METHOD = 'Radau'


# =============================================================================
# simulate — pydex signature 2
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Evaluate the A→B model by direct ODE integration and return concentrations
    at each requested sampling time.

    This function is assigned to designer.simulate and is called by pydex
    during both prediction (plotting, print_optimal_candidates) and
    sensitivity analysis (finite-difference Jacobian via numdifftools).

    Pydex signature type 2: simulate(ti_controls, sampling_times, model_parameters)

    Parameters
    ----------
    ti_controls : array-like, length 2
        [CA0 (mol/L), T (K)]
        CA0 : initial concentration of A.
        T   : isothermal reaction temperature (constant throughout batch).

    sampling_times : array-like
        Absolute measurement times (min).  pydex may pad shorter candidate
        rows with NaN — these are stripped before integration.

    model_parameters : array-like, length 4
        [θ₀, θ₁, α, ν]
        θ₀  : Arrhenius pre-exponential offset  (ln(k_ref) - θ₁)
        θ₁  : Arrhenius activation-energy group  Ea / (R * T_ref)
        α   : reaction order in CA
        ν   : stoichiometric coefficient CB/CA
        During FD sensitivity analysis, pydex / numdifftools perturbs each
        element of this array by a finite step.  The perturbed vector is
        passed directly here, so model_parameters[j] may be outside its
        nominal physical range.  The `max(CA, 0)` guard in the RHS and the
        NaN fallback below handle these edge cases.

    Returns
    -------
    y : np.ndarray, shape (n_spt, 2)
        Columns: [CA (mol/L), CB (mol/L)] at each sampling time.
        Returns an array of NaN (same shape) if integration fails, allowing
        numdifftools Richardson extrapolation to skip the failed evaluation
        and continue with smaller FD steps.

    Notes on robustness
    --------------------
    Unlike the collocation variant, this function does not need parameter
    clamping or relaxed bounds because:

      1. The integrator evaluates the RHS as plain Python/numpy floats.
         A perturbed α slightly below zero with CA > 0 still gives a valid
         float from CA**α — it only becomes NaN if CA itself reaches zero,
         which the `max(CA, 0)` guard prevents.

      2. There are no NLP bounds or interior-point barriers to violate.
         The integrator adapts its step size instead of failing.

      3. solve_ivp returns a structured result object; we check `sol.success`
         explicitly rather than catching an exception.

    The NaN fallback is still included as a last-resort safety net for truly
    pathological perturbations (e.g. α = -10 with CA = 0.5, which would give
    a very large positive RHS and potential blow-up).
    """
    # ── Unpack controls and parameters ────────────────────────────────────
    CA0 = float(ti_controls[0])
    T   = float(ti_controls[1])

    theta_0, theta_1, alpha, nu = [float(p) for p in model_parameters]

    # ── Sampling time processing ──────────────────────────────────────────
    # Strip NaN padding that pydex inserts when candidates have different
    # numbers of sampling times.
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]
    n_spt   = len(spt_abs)
    t_end   = float(np.max(spt_abs))

    # ── Arrhenius rate constant ───────────────────────────────────────────
    # k(T) = exp(θ₀ + θ₁ * (T - 273.15) / T)
    # Computed once per call — k is isothermal (constant T throughout batch).
    ln_k = theta_0 + theta_1 * (T - 273.15) / T
    k    = np.exp(ln_k)

    # ── ODE right-hand side ───────────────────────────────────────────────
    def rhs(t, y):
        """
        RHS of the A→B ODE system.

        dCA/dt = -k * max(CA, 0)^α
        dCB/dt =  ν * k * max(CA, 0)^α

        max(CA, 0) prevents complex-valued powers when numdifftools perturbs
        the parameters and integration overshoots CA = 0 by a tiny amount.
        This is the only defensive line needed — equivalent to a physical
        lower bound of zero on concentration without any NLP formulation.
        """
        CA, CB = y
        # Guard against numerical CA < 0 (overshoot near depletion)
        CA_pos = max(CA, 0.0)
        # Guard against CA = 0 with non-integer α (0^α is 0 for α > 0, but
        # Python raises ZeroDivisionError for α < 0 — set rate to 0 instead)
        if CA_pos == 0.0:
            rate = 0.0
        else:
            rate = k * (CA_pos ** alpha)
        return [-rate, nu * rate]

    # ── Integrate ─────────────────────────────────────────────────────────
    try:
        sol = solve_ivp(
            rhs,
            t_span  = (0.0, t_end),
            y0      = [CA0, 0.0],
            method  = _METHOD,
            t_eval  = spt_abs,
            rtol    = _RTOL,
            atol    = _ATOL,
            dense_output = False,
        )
    except Exception:
        # Catch any unexpected solver exception (e.g. numpy overflow in RHS)
        # and return NaN so Richardson extrapolation can skip this evaluation.
        return np.full((n_spt, 2), np.nan)

    if not sol.success:
        # solve_ivp failed gracefully — return NaN so pydex / numdifftools
        # treats this as a failed FD evaluation and skips it.
        return np.full((n_spt, 2), np.nan)

    # sol.y has shape (2, n_spt); transpose to (n_spt, 2)
    return sol.y.T


# =============================================================================
# Sanity check — run directly to verify the model produces a sensible profile
# =============================================================================

if __name__ == '__main__':
    # Nominal parameters: k_ref = 0.1 L/(mol·min), Ea = 5000 J/mol
    pre_exp_constant = 0.1
    activ_energy     = 5000.0
    R                = 8.314159
    T_ref            = 273.15   # K  (reference temperature for Arrhenius)

    # Reparametrised form: k(T) = exp(θ₀ + θ₁*(T - T_ref)/T)
    # At T = T_ref: k(T_ref) = exp(θ₀) = k_ref  →  θ₀ = ln(k_ref)
    # θ₁ = Ea / (R * T_ref)  (dimensionless activation energy group)
    theta_0   = np.log(pre_exp_constant) - activ_energy / (R * T_ref)
    theta_1   = activ_energy / (R * T_ref)
    theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])

    print(f"Nominal parameters: θ = {theta_nom}")
    k_at_50C = np.exp(theta_0 + theta_1 * (323.15 - 273.15) / 323.15)
    print(f"  k at T=323.15 K : {k_at_50C:.4f} 1/min")

    tic = [1.0, 323.15]                  # CA0 = 1 mol/L, T = 50°C
    spt = np.linspace(0.001, 200, 11)    # 11 sampling times over 200 min

    y = simulate(
        ti_controls=tic,
        sampling_times=spt,
        model_parameters=theta_nom,
    )

    print(f"\nConcentrations at sampling times:")
    print(f"  {'t (min)':>10}  {'CA (mol/L)':>12}  {'CB (mol/L)':>12}")
    print(f"  {'-'*38}")
    for t, row in zip(spt, y):
        print(f"  {t:>10.2f}  {row[0]:>12.6f}  {row[1]:>12.6f}")

    # Compare with collocation variant if available
    try:
        from case_2_no_ift_model import simulate as simulate_coll
        y_coll = simulate_coll(
            ti_controls=tic,
            sampling_times=spt,
            model_parameters=theta_nom,
        )
        max_diff_CA = np.max(np.abs(y[:, 0] - y_coll[:, 0]))
        max_diff_CB = np.max(np.abs(y[:, 1] - y_coll[:, 1]))
        print(f"\nMax |scipy - collocation|: CA = {max_diff_CA:.2e}, CB = {max_diff_CB:.2e}")
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spt, y[:, 0], label='$c_A$', marker='o')
    ax.plot(spt, y[:, 1], label='$c_B$', marker='o')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mol/L)')
    ax.set_title('A→B reaction  (scipy Radau integration, FD sensitivity path)')
    ax.legend()
    fig.tight_layout()
    plt.show()
