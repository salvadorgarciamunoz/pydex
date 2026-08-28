"""
van_laar_model.py
=================
Binary VLE: fitting van Laar activity-coefficient parameters from
isothermal total-pressure measurements.

A steady-state (non-dynamic) model that is genuinely NONLINEAR in its
parameters -- the teaching step between a linear response surface and a
dynamic model.

Model
-----
Two-parameter van Laar activity coefficients for a binary mixture:

    ln g1 = A12 * ( A21*x2 / (A12*x1 + A21*x2) )**2
    ln g2 = A21 * ( A12*x1 / (A12*x1 + A21*x2) )**2

Modified Raoult's law gives the two partial pressures, and the measured
response is the pair (P, y1) -- total pressure and vapour composition:

    p1 = x1 * g1 * P1sat(T)
    p2 = x2 * g2 * P2sat(T)
    P  = p1 + p2
    y1 = p1 / P

Pure-component saturation pressures come from Antoine constants that are
KNOWN, not fitted -- only A12 and A21 are estimated.

Experimental variables (the candidate grid)
-------------------------------------------
    x1 : liquid mole fraction of component 1, in [0.05, 0.95]
    T  : temperature in degC, in [40, 80]

Responses
---------
    P  : total pressure (kPa)
    y1 : vapour mole fraction of component 1

Antoine constants (log10 Psat[kPa] = A - B/(C + T[degC])) are representative
values for an ethanol(1)/water(2)-like pair; the point of the example is the
design workflow, not the specific chemistry.
"""

import numpy as np

# Antoine constants — treated as known constants, not estimated parameters
ANTOINE = {
    1: (7.16879, 1552.601, 222.419),   # component 1
    2: (7.19621, 1730.630, 233.426),   # component 2
}


def psat(component, T_degC):
    """Saturation pressure in kPa from the Antoine equation."""
    A, B, C = ANTOINE[component]
    return 10.0 ** (A - B / (C + T_degC))


def simulate(ti_controls, model_parameters):
    """
    Predict (P, y1) for one experimental candidate.

    pydex inspects these argument NAMES to pick the calling convention, so
    they must be exactly `ti_controls` and `model_parameters` for a static
    model. Order matters too.

    Parameters
    ----------
    ti_controls      : [x1, T_degC]
    model_parameters : [A12, A21]

    Returns
    -------
    np.array([P, y1])
    """
    x1 = float(ti_controls[0])
    T = float(ti_controls[1])
    A12, A21 = float(model_parameters[0]), float(model_parameters[1])

    x2 = 1.0 - x1

    # van Laar activity coefficients.
    denom = A12 * x1 + A21 * x2
    ln_g1 = A12 * (A21 * x2 / denom) ** 2
    ln_g2 = A21 * (A12 * x1 / denom) ** 2

    g1, g2 = np.exp(ln_g1), np.exp(ln_g2)

    p1 = x1 * g1 * psat(1, T)
    p2 = x2 * g2 * psat(2, T)

    P = p1 + p2
    y1 = p1 / P

    return np.array([P, y1])


if __name__ == "__main__":
    # Sanity check before designing anything: does the model behave?
    theta = [1.65, 0.95]
    for x1 in (0.1, 0.5, 0.9):
        P, y1 = simulate([x1, 60.0], theta)
        print(f"x1={x1:4.2f}  T=60 C   P={P:7.3f} kPa   y1={y1:6.4f}")
