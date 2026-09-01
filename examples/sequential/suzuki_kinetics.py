"""
suzuki_kinetics.py
==================
Suzuki-Miyaura cross-coupling kinetics, written so the kinetic parameters are
ESTIMATED rather than fixed. Used by the sequential-design example in this
folder.

This is a self-contained model. `examples/b_optimal/suzuki_model.py` is a
separate copy with the parameters hard-coded, used for bracketing-optimal
design; the two are deliberately independent so neither constrains the other.

Chemistry
---------
    Ar-X + Ar'-B(OH)2  --k1-->  Ar-Ar'    desired coupling, product P
    Ar'-B(OH)2         --k2-->  Ar'-H     protodeboronation, boronic acid loss
    2 Ar'-B(OH)2       --k3-->  Ar'-Ar'   homocoupling, impurity D

k1 and k3 carry a fractional/first order dependence on Pd loading; k1 and k3
are also gated by a saturating base term.

Parameterisation -- and why it is not (A, Ea)
--------------------------------------------
Writing an Arrhenius rate as `A*exp(-Ea/RT)` and estimating (A, Ea) is the
classic correlation trap: over any narrow temperature window the two are very
nearly collinear, because a change in Ea can be absorbed almost exactly by a
compensating change in ln A. `run_estimability()` reports them as one
correlation group, and the fit is ill-conditioned.

The standard fix, used here, is to centre the rate on a reference temperature:

    ln k_i(T) = lnk_i_ref  -  (Ea_i / R) * (1/T - 1/T_ref)

At T = T_ref the second term vanishes, so `lnk_i_ref` is the rate at the
reference condition and `Ea_i` is its temperature dependence. The two are far
less correlated, and each has a meaning a chemist can argue about.

    model_parameters = [lnk1_ref, Ea1_kJ, lnk2_ref, Ea2_kJ]

Ea is carried in kJ/mol so the parameters are of comparable magnitude -- pydex
normalises sensitivities by parameter value, but keeping the scales sane also
helps the regression.

The homocoupling pair (k3) is held at nominal values. Impurity D stays below
about 1% over most of the operating box, so a small screen carries almost no
information about it; the sequential example demonstrates that rather than
assuming it.

Inputs (5)
    T        reaction temperature        [C]
    t_rxn    reaction time               [h]
    cat_mol  Pd loading                  [mol %]
    boron_eq boronic acid equivalents    [-]
    base_eq  base equivalents            [-]

Responses (3)
    yield_P  yield of P                  [fraction of limiting ArX]
    imp_D    homocoupling impurity       [fraction]
    res_B    residual boronic acid       [fraction of charged]
"""

import numpy as np

R_GAS = 8.314e-3            # kJ/(mol*K), so Ea is in kJ/mol
T_REF_C = 72.5              # centre of the 50-95 C bracketing range
T_REF_K = T_REF_C + 273.15

# ── Nominal ("true") parameters, used to generate synthetic measurements ────
# Chosen so that the informative region and the saturated region are both
# reachable inside the bracketing box -- which is what makes the example
# realistic.
LNK1_REF_TRUE = 0.30        # ln of the coupling rate constant at T_REF
EA1_TRUE = 55.0             # kJ/mol
LNK2_REF_TRUE = -3.50       # protodeboronation is a real competing loss here
EA2_TRUE = 85.0             # but much more temperature-sensitive
LNK3_REF_TRUE = -6.60       # homocoupling is slow; impurity D stays under ~1%
EA3_TRUE = 60.0

THETA_TRUE = np.array([LNK1_REF_TRUE, EA1_TRUE,
                       LNK2_REF_TRUE, EA2_TRUE,
                       LNK3_REF_TRUE, EA3_TRUE])
THETA_NAMES = ["lnk1_ref", "Ea1", "lnk2_ref", "Ea2", "lnk3_ref", "Ea3"]

# Bounds for the regression, in the same order.
THETA_LB = np.array([-4.0, 5.0, -12.0, 5.0, -14.0, 5.0])
THETA_UB = np.array([4.0, 200.0, -1.0, 200.0, -1.0, 200.0])
# NOTE: no element may be 0.0. pydex normalises sensitivities by parameter
# value, so a nominal of exactly zero gives abs_info = 0 and the parameter is
# flagged unresolvable for arithmetic reasons rather than physical ones.
THETA_START = np.array([0.5, 40.0, -6.0, 60.0, -8.0, 40.0])

# The same information keyed by NAME. Reading `BOUNDS_LO["Ea2"]` at a call
# site is easier to check than remembering that Ea2 is position 3.
BOUNDS_LO = dict(zip(THETA_NAMES, THETA_LB))
BOUNDS_HI = dict(zip(THETA_NAMES, THETA_UB))
START = dict(zip(THETA_NAMES, THETA_START))
TRUE = dict(zip(THETA_NAMES, THETA_TRUE))

CAT_ORDER1 = 0.5            # fractional order in Pd for the coupling
CAT_ORDER3 = 1.0            # first order in Pd for homocoupling
BASE_KM = 1.2               # saturating base activation

# ── TOTAL measurement error, per response ──────────────────────────────────
# Not analytical precision alone: this is the spread the model is actually
# compared against, so it carries sampling, repeatability, operator and
# batch-to-batch variation as well as the assay itself. Quantifying it is a
# measurement-systems exercise, not a chromatography one.
#   yield and residual boronic acid to about 1.5% absolute,
#   impurity to about 0.1% absolute.
SIGMA = np.array([0.015, 0.001, 0.015])
ERROR_COV = np.diag(SIGMA ** 2)

# ── Bracketing ranges a process chemist would explore ──────────────────────
BOUNDS = {
    "T":        (50.0, 95.0),
    "t_rxn":    (0.5, 8.0),
    "cat_mol":  (0.25, 3.0),
    "boron_eq": (1.0, 1.6),
    "base_eq":  (1.5, 3.5),
}
BOUND_ORDER = ["T", "t_rxn", "cat_mol", "boron_eq", "base_eq"]

CONTROL_NAMES = ["T", "t_rxn", "cat_mol", "boron_eq", "base_eq"]
RESPONSE_NAMES = ["yield_P", "imp_D", "res_B"]


def _arrhenius(lnk_ref, Ea_kJ, T_C):
    """Rate constant from a reference-centred Arrhenius form."""
    T_K = T_C + 273.15
    return np.exp(lnk_ref - (Ea_kJ / R_GAS) * (1.0 / T_K - 1.0 / T_REF_K))


def simulate(ti_controls, model_parameters):
    """pydex simulate() contract for a STATIC model.

    The reaction is integrated in time internally and only the end-of-batch
    state is reported, so as far as pydex is concerned this is a steady-state
    model with no sampling times: `t_rxn` is a CONTROL, not a sampling time.

    Args:
        ti_controls: [T, t_rxn, cat_mol, boron_eq, base_eq]
        model_parameters: [lnk1_ref, Ea1, lnk2_ref, Ea2, lnk3_ref, Ea3]

    Returns:
        np.array([yield_P, imp_D, res_B])
    """
    T, t_rxn, cat_mol, boron_eq, base_eq = (float(v) for v in ti_controls)
    (lnk1_ref, Ea1, lnk2_ref, Ea2,
     lnk3_ref, Ea3) = (float(v) for v in model_parameters)

    k1 = _arrhenius(lnk1_ref, Ea1, T) * (cat_mol ** CAT_ORDER1)
    k2 = _arrhenius(lnk2_ref, Ea2, T)
    k3 = _arrhenius(lnk3_ref, Ea3, T) * (cat_mol ** CAT_ORDER3)

    base_factor = base_eq / (BASE_KM + base_eq)
    k1_eff = k1 * base_factor
    k3_eff = k3 * base_factor

    # state: [ArX, ArB, P, D]; basis 1.0 equivalent of ArX (limiting)
    y = np.array([1.0, boron_eq, 0.0, 0.0])

    def rhs(s):
        ArX, ArB = max(s[0], 0.0), max(s[1], 0.0)
        r1 = k1_eff * ArX * ArB
        r2 = k2 * ArB
        r3 = k3_eff * ArB * ArB
        return np.array([-r1, -r1 - r2 - 2.0 * r3, +r1, +r3])

    n_steps = 400
    h = t_rxn / n_steps
    for _ in range(n_steps):
        a = rhs(y)
        b = rhs(y + 0.5 * h * a)
        c = rhs(y + 0.5 * h * b)
        d = rhs(y + h * c)
        y = np.maximum(y + (h / 6.0) * (a + 2 * b + 2 * c + d), 0.0)

    ArX, ArB, P, D = y
    return np.array([P, D, ArB / max(boron_eq, 1e-12)])


if __name__ == "__main__":
    print(f"reference temperature: {T_REF_C} C")
    print(f"true parameters: {dict(zip(THETA_NAMES, THETA_TRUE))}\n")
    print(f"{'T':>5}{'t':>5}{'cat':>5} | {'yield':>7}{'imp':>8}{'resB':>7}")
    for tic in ([55, 1.0, 0.4, 1.2, 2.0], [72.5, 2.0, 1.0, 1.2, 2.0],
                [95, 8.0, 3.0, 1.6, 3.5], [80, 4.0, 2.0, 1.3, 2.5]):
        y = simulate(tic, THETA_TRUE)
        print(f"{tic[0]:5.1f}{tic[1]:5.1f}{tic[2]:5.2f} | "
              f"{y[0]:7.3f}{y[1]:8.4f}{y[2]:7.3f}")
