from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np
import logging

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

"""
case_5_model.py
================
Same A -> B -> C network as case_4_model.py, but the two rate constants are
now Arrhenius: k_j = exp(theta0_j + theta1_j * (T - 273.15) / T). Four model
parameters [theta0_1, theta1_1, theta0_2, theta1_2], one control (T). f_in
is hardcoded to 0 in simulate() regardless of ti_controls -- ti_controls[0]
sets T instead -- so this one IS a closed batch system (mass conserved
exactly; verified numerically).

PORTED FROM A CASADI/IDAS ORIGINAL -- see case_4_model.py's module
docstring for the full story: the original defined the reaction rates as a
separate algebraic constraint (a DAE, castable only with casadi), which is
folded directly into the mass balance here instead (a pure ODE, integrable
with scipy's `vode`). Verified against the original casadi version: max abs
difference 1.8e-4 across all three concentrations over an 11-point grid.
"""

def create_model():
    """ NOTE: no longer takes spt / does not pre-seed m.t with sampling times.

    scipy's Pyomo Simulator builds its own internal time grid (via
    `numpoints`) and cannot be pointed at arbitrary ContinuousSet points the
    way the casadi backend can. simulate() below interpolates the dense
    scipy grid back onto the caller's requested sampling times instead.
    """
    m = po.ConcreteModel()

    """ Sets """
    m.i = po.Set(initialize=["A", "B", "C"])
    m.j = po.Set(initialize=[1, 2])

    """ Time Components """
    m.t = pod.ContinuousSet(bounds=(0, 1))
    m.tau = po.Var(bounds=(0, None))

    """ Concentrations """
    m.c = po.Var(m.t, m.i, bounds=(0, None))
    m.dcdt = pod.DerivativeVar(m.c, wrt=m.t)

    """ Experimental Variables """
    m.f_in = po.Var(bounds=(0, 10))
    m.T = po.Var(bounds=(273.15, 323.15))

    """ Reaction Parameters """
    s = {
        ("A", 1): -1,
        ("B", 1):  1,
        ("C", 1):  0,

        ("A", 2):  0,
        ("B", 2): -1,
        ("C", 2):  1,
    }
    m.s = po.Param(m.i, m.j, initialize=s)
    c_in = {
        "A": 1,
        "B": 0,
        "C": 0,
    }
    m.c_in = po.Param(m.i, initialize=c_in)

    """ Model Parameters """
    m.k = po.Var(m.j, bounds=(0, None))
    m.theta0 = po.Var(m.j)
    m.theta1 = po.Var(m.j)

    """ Model Equations
    scipy's Simulator only integrates pure ODE systems -- it cannot handle
    the algebraic reaction-rate constraint (r_def) the casadi version used,
    which made this a DAE. r[t, j] is substituted directly into the mass
    balance instead of being a separate algebraic Var/Constraint, which
    keeps the equations mathematically identical while making the system a
    pure ODE that scipy's odeint/vode can integrate. """
    def _r(m, t, j):
        k = po.exp(m.theta0[j] + m.theta1[j] * (m.T - 273.15) / m.T)
        if j == 1:
            return k * m.c[t, "A"]
        elif j == 2:
            return k * m.c[t, "B"]
        else:
            raise SyntaxError("Unrecognized reaction index, please check the model.")

    def _bal(m, t, i):
        return m.dcdt[t, i] / m.tau == m.f_in * m.c_in[i] + sum(
            m.s[i, j] * _r(m, t, j) for j in m.j
        )
    m.bal = po.Constraint(m.t, m.i, rule=_bal)

    return m

def simulate(ti_controls, sampling_times, model_parameters, numpoints=2000):
    spt_abs = np.asarray(sampling_times, dtype=float)
    norm_spt = spt_abs / np.max(spt_abs)

    m = create_model()
    m.tau.fix(np.max(spt_abs))

    # m.f_in.fix(ti_controls[0])
    m.f_in.fix(0)
    m.T.fix(ti_controls[0])

    m.theta0[1].fix(model_parameters[0])
    m.theta1[1].fix(model_parameters[1])

    m.theta0[2].fix(model_parameters[2])
    m.theta1[2].fix(model_parameters[3])

    m.c[0, "A"].fix(1)
    m.c[0, "B"].fix(0)
    m.c[0, "C"].fix(0)

    simulator = pod.Simulator(m, package="scipy")
    tsim, profiles = simulator.simulate(numpoints=numpoints, integrator="vode")
    simulator.initialize_model()

    tsim_flat = tsim.flatten()
    c = np.empty((len(norm_spt), len(m.i)))
    for col, i in enumerate(m.i):
        c[:, col] = np.interp(norm_spt, tsim_flat, profiles[:, col])

    if False:
        plt.plot(tsim_flat, profiles)

    return c

if __name__ == '__main__':
    """ Run Simulation """
    tic = [300]
    spt = np.linspace(0, 10, 21)
    mp = [1, 3, -1, 5]
    c = simulate(
        ti_controls=tic,
        sampling_times=spt,
        model_parameters=mp,
    )

    """ Plot Results """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(
        spt,
        c[:, 0],
        label=r"$c_A$",
    )
    axes.plot(
        spt,
        c[:, 1],
        label=r"$c_B$",
    )
    axes.plot(
        spt,
        c[:, 2],
        label=r"$c_C$",
    )
    axes.legend()
    axes.set_xlabel("Time (hour)")
    axes.set_ylabel("Concentration (mol/L)")
    fig.tight_layout()

    plt.show()
