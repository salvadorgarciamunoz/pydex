"""
suzuki_model.py
=================
Suzuki-Miyaura cross-coupling in a batch (or plug-flow-equivalent) reactor
-- a NEW case study, not from Chen et al. (2018), included to show
b_opt_criterion applied to a fresh pharmaceutical problem where the
"bracketing study" framing is exactly the industrial norm.

Chemistry (a standard, widely-reported Suzuki reaction network):

    Ar-X  +  Ar'-B(OH)2  --k1-->  Ar-Ar'   (desired cross-coupling, product P)
    Ar'-B(OH)2           --k2-->  Ar'-H    (protodeboronation, boronic acid loss)
    2 Ar'-B(OH)2         --k3-->  Ar'-Ar'  (homocoupling, impurity D)

The catalyst (Pd) loading enters k1 with a fractional-order dependence
(a common empirical form for Pd-catalysed couplings, reflecting the
pre-catalyst activation equilibrium).

Inputs (5) -- the factors a process chemist would bracket:
    T        : reaction temperature                            [C]
    t_rxn    : reaction time                                   [h]
    cat_mol  : Pd catalyst loading                             [mol %]
    boron_eq : boronic acid equivalents (vs. aryl halide)       [-]
    base_eq  : base equivalents                                [-]

Outputs (3) -- the critical quality attributes:
    yield_P  : yield of desired product P                       [fraction]
    imp_D    : homocoupling impurity D level                    [fraction]
    res_B    : residual unreacted boronic acid                  [fraction]

Process constraints (typical of a late-stage pharma step):
    yield_P >= 0.70    (economically viable conversion)
    imp_D   <= 0.015   (impurity specification, 1.5%)
    res_B   <= 0.10    (residual reagent limit for downstream workup)

Kinetic parameters here are representative of a realistic Suzuki system
(rate ordering k1 >> k3 > k2 at typical operating temperature, with
protodeboronation more temperature-sensitive than the coupling itself, so
that pushing temperature trades yield against boronic-acid loss). They
are illustrative rather than fitted to a specific literature substrate.
"""
import numpy as np

R_GAS = 8.314  # J/(mol*K)

# --- representative kinetic parameters ---
# Calibrated so that (a) yields stay physical (max 1.0 -- no RK4 overshoot),
# and (b) roughly 8% of the bracketing box is feasible, i.e. the constraints
# genuinely bite without making the space impossibly tight.
# desired cross-coupling (2nd order in [ArX][ArB], fractional order in Pd)
A1, EA1 = 8.0e8, 55_000.0
CAT_ORDER = 0.5           # fractional order in catalyst loading
# protodeboronation (1st order in [ArB]) -- more T-sensitive than coupling
A2, EA2 = 1.0e9, 85_000.0
# homocoupling (2nd order in [ArB]), Pd-mediated
A3, EA3 = 8.0e5, 60_000.0
CAT_ORDER3 = 1.0

BASE_KM = 1.2             # base saturation constant (Michaelis-like activation)


def _rate_constants(T_C, cat_mol):
    T_K = T_C + 273.15
    k1 = A1 * np.exp(-EA1 / (R_GAS * T_K)) * (cat_mol ** CAT_ORDER)
    k2 = A2 * np.exp(-EA2 / (R_GAS * T_K))
    k3 = A3 * np.exp(-EA3 / (R_GAS * T_K)) * (cat_mol ** CAT_ORDER3)
    return k1, k2, k3


def simulate_suzuki(ti_controls, model_parameters=None):
    """designer.simulate() contract. model_parameters unused -- fixed
    representative kinetic model, not something being fitted.

    Integrates the batch ODE system with a fixed-step RK4. Basis: 1.0
    equivalent of aryl halide (limiting reagent).
    """
    T, t_rxn, cat_mol, boron_eq, base_eq = ti_controls
    k1, k2, k3 = _rate_constants(T, cat_mol)

    # base activation factor: saturating in base equivalents
    base_factor = base_eq / (BASE_KM + base_eq)
    k1_eff = k1 * base_factor
    k3_eff = k3 * base_factor

    # state: [ArX, ArB, P, D]  (ArH tracked implicitly as loss)
    y = np.array([1.0, float(boron_eq), 0.0, 0.0])

    def rhs(s):
        ArX, ArB, P, D = s
        ArX = max(ArX, 0.0)
        ArB = max(ArB, 0.0)
        r1 = k1_eff * ArX * ArB          # cross-coupling
        r2 = k2 * ArB                    # protodeboronation
        r3 = k3_eff * ArB * ArB          # homocoupling (consumes 2 ArB)
        return np.array([
            -r1,                # ArX
            -r1 - r2 - 2 * r3,  # ArB
            +r1,                # P
            +r3,                # D
        ])

    n_steps = 400
    h = float(t_rxn) / n_steps
    for _ in range(n_steps):
        k_1 = rhs(y)
        k_2 = rhs(y + 0.5 * h * k_1)
        k_3 = rhs(y + 0.5 * h * k_2)
        k_4 = rhs(y + h * k_3)
        y = y + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        y = np.maximum(y, 0.0)

    ArX, ArB, P, D = y
    yield_P = P / 1.0                     # relative to limiting aryl halide
    imp_D = D / 1.0
    res_B = ArB / max(float(boron_eq), 1e-12)   # fraction of charged boronic acid left
    return np.array([yield_P, imp_D, res_B])


# Input ranges -- the bracketing ranges a process chemist would explore
BOUNDS = {
    "T":        (50.0, 95.0),    # C
    "t_rxn":    (0.5, 8.0),      # h
    "cat_mol":  (0.25, 3.0),     # mol % Pd
    "boron_eq": (1.0, 1.6),      # equivalents
    "base_eq":  (1.5, 3.5),      # equivalents
}
BOUND_ORDER = ["T", "t_rxn", "cat_mol", "boron_eq", "base_eq"]

# Process constraints (critical quality attributes)
YIELD_MIN = 0.70
IMP_D_MAX = 0.015
RES_B_MAX = 0.10


def feasible(ti_controls):
    yield_P, imp_D, res_B = simulate_suzuki(ti_controls)
    return (yield_P >= YIELD_MIN) and (imp_D <= IMP_D_MAX) and (res_B <= RES_B_MAX)
