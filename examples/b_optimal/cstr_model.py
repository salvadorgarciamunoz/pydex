"""
cstr_model.py
==============
Reproduction of the two-CSTR-in-series model from Chen, Paulavicius &
Adjiman (2018), AIChE J. 64:3944-3957, Section "CSTR case study",
transcribed from the GAMS source published with the paper (reactor_code/,
files reactor.dat.gms and reactor.eqn.gms), hosted per the paper's Data
Statement. Parameters and bounds below are the paper's own values.

    A + D --r1--> B        (desired reaction, exothermic)
    B --r2--> C             (degradation of B into waste product C)

Two well-insulated (adiabatic) CSTRs in series. Rate law and parameters
transcribed exactly from reactor.dat.gms / reactor.eqn.gms:

    k_j(T) = k0(j) * exp(-Ea(j) / (T + 273))      T in Celsius
    rate(r1) = k1 * C_A * C_D                       (2nd order)
    rate(r2) = k2 * C_B                             (1st order)

    mass balance:   Qin*(C_in,i - C_out,i) + V*sum_j(nu(i,j)*rate_j) = 0
    energy balance: Qin*dens*Cp*(T_in - T_out) + V*sum_j(rate_j*(-dHrxn_j)) = 0
    mole fraction:  Frac_i = C_i / sum_ii(C_ii)     (current-stage total)

Only UPPER bounds on the outputs are enforced as constraints (FracC <=
0.002, FracA <= 0.02, T <= 85) -- confirmed directly from
reactor.eqn.gms's CProdConstr/AProdConstr/TConstr equations. The
out_lo_lim/out_up_lim pair in reactor.dat.gms is used ONLY to scale
outputs for the covariance criterion, not as an additional hard
constraint on the lower end.
"""
import numpy as np

# ---- EXACT parameters, transcribed from reactor.dat.gms ----
K0 = {"r1": 2800.0, "r2": 12.0}          # pre-exponential factors
EA = {"r1": 2995.0, "r2": 4427.0}        # activation "temperature" (Ea/R), Kelvin
DHRXN = {"r1": -80.0, "r2": 0.0}         # heats of reaction (r2 is thermoneutral)
DENS = 0.8
CP = 1.7
RHO_CP = DENS * CP                        # = 1.36, exactly as in the GAMS source

# ---- EXACT input bounds, transcribed from reactor.dat.gms in_lo_lim/in_up_lim ----
BOUNDS = {
    "A0":    (0.8, 1.1),        # mol/L, inlet A concentration
    "ratio": (0.8, 0.835),      # [A]0 / [D]0  (named ratAD in the GAMS source)
    "q0":    (0.0083, 0.08),    # L/h, feed flow rate (named Qin)
    "V1":    (0.5, 2.0),        # L
    "V2":    (0.5, 2.0),        # L
    "T0":    (22.0, 35.0),      # C
}
BOUND_ORDER = ["A0", "ratio", "q0", "V1", "V2", "T0"]

# ---- EXACT output constraints (upper only), transcribed from CProdConstr/
# AProdConstr/TConstr in reactor.eqn.gms ----
XC2_MAX = 0.002
XA2_MAX = 0.02
T2_MAX = 85.0


def _rate_constants(T_C):
    T_K_offset = T_C + 273.0   # the GAMS source uses +273, not +273.15
    k1 = K0["r1"] * np.exp(-EA["r1"] / T_K_offset)
    k2 = K0["r2"] * np.exp(-EA["r2"] / T_K_offset)
    return k1, k2


def _solve_cstr(CA_in, CD_in, CB_in, CC_in, T_in, tau, n_iter=60):
    """Steady-state solve of one adiabatic CSTR stage, rates evaluated at
    OUTLET conditions (as in the GAMS steady-state formulation). Solved by
    fixed-point iteration on T: guess T, solve the algebraic extents in
    closed form (exact for this stoichiometry), update T via the energy
    balance, repeat to convergence."""
    T = T_in
    for _ in range(n_iter):
        k1, k2 = _rate_constants(T)
        # extent of reaction 1 (2nd order, A+D->B): quadratic in x
        #   k1*tau*x^2 - (1 + k1*tau*(CA_in+CD_in))*x + k1*tau*CA_in*CD_in = 0
        a = k1 * tau
        b = -(1.0 + k1 * tau * (CA_in + CD_in))
        c = k1 * tau * CA_in * CD_in
        if a < 1e-30:
            x = 0.0
        else:
            disc = max(b * b - 4 * a * c, 0.0)
            x1 = (-b - np.sqrt(disc)) / (2 * a)
            x2 = (-b + np.sqrt(disc)) / (2 * a)
            cand = [v for v in (x1, x2) if 0.0 <= v <= min(CA_in, CD_in) + 1e-9]
            x = min(cand) if cand else 0.0
        CA = CA_in - x
        CD = CD_in - x
        CB = (CB_in + x) / (1.0 + k2 * tau)
        extent2 = k2 * CB * tau
        CC = CC_in + extent2

        T_new = T_in + (x * (-DHRXN["r1"]) + extent2 * (-DHRXN["r2"])) / RHO_CP
        if abs(T_new - T) < 1e-9:
            T = T_new
            break
        T = 0.5 * T + 0.5 * T_new   # damped update for stability
    return CA, CD, CB, CC, T


def simulate_cstr(ti_controls, model_parameters=None):
    """designer.simulate() contract. model_parameters is unused -- this is
    the paper's own fixed kinetic model, not something being fitted."""
    A0, ratio, q0, V1, V2, T0 = ti_controls
    D0 = A0 / ratio          # matches D0Defn: Conc0('D')*ratAD = Conc0('A')
    tau1 = V1 / q0
    tau2 = V2 / q0

    CA1, CD1, CB1, CC1, T1 = _solve_cstr(A0, D0, 0.0, 0.0, T0, tau1)
    CA2, CD2, CB2, CC2, T2 = _solve_cstr(CA1, CD1, CB1, CC1, T1, tau2)

    # mole fraction over the CURRENT total concentration at stage 2
    # (matches FracDefn exactly: Frac(i) = Conc(i) / sum_ii(Conc(ii)))
    C_total2 = CA2 + CB2 + CC2 + CD2
    xA2 = CA2 / C_total2
    xC2 = CC2 / C_total2
    return np.array([xC2, xA2, T2])


def feasible(ti_controls):
    xC2, xA2, T2 = simulate_cstr(ti_controls)
    return (xC2 <= XC2_MAX) and (xA2 <= XA2_MAX) and (T2 <= T2_MAX)
