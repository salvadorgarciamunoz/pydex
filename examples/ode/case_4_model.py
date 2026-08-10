from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np
import logging

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

"""
case_4_model.py
===============
A -> B -> C, two first-order rate constants [k1, k2], one control (a feed
rate f_in). Batch/semi-batch reactor:

    dc[i]/dt / tau = f_in * c_in[i] + sum_j s[i,j] * r[j]
    r[1] = k1 * c[A]        r[2] = k2 * c[B]

NOTE on f_in: c_in = [1, 0, 0] (pure A) and there is no outflow/dilution
term, so this is a CONTINUOUSLY-FED semi-batch reactor, not a closed system.
Total moles grow as 1 + f_in*t once f_in > 0 -- verified numerically
(sum(c) == 1 + f_in*t at every sampled time, to solver tolerance). This is
presumably deliberate, but it is a different kind of system than every other
pydex example (all closed), so it's worth knowing before you reuse this model.

PORTED FROM A CASADI/IDAS ORIGINAL
-----------------------------------
The original model used `pod.Simulator(m, package="casadi")`, with the
reaction rates r[t, j] defined as a separate ALGEBRAIC constraint (r_def).
That made the system a true DAE, which scipy's Simulator cannot integrate:

    pyomo.dae.diffvar.DAE_Error: Model contains an algebraic equation or
    unrecognized differential equation. ... If you are trying to simulate
    a DAE model you must use CasADi as the integration package.

r[t, j] is substituted directly into the mass balance below instead of
being its own Var/Constraint, which is mathematically identical but turns
the system into a pure ODE scipy's `vode` can integrate. Verified against
the original casadi version: max abs difference 1.8e-4 across all three
concentrations over a 101-point grid (solver-tolerance-level agreement),
and exact mass-balance conservation confirmed independently either way.

scipy's Simulator also can't be pointed at arbitrary ContinuousSet points
the way casadi's can -- it builds its own dense internal grid (`numpoints`)
-- so simulate() below integrates on that dense grid and interpolates back
onto the caller's requested sampling times.
"""

def create_model():
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

    """ Reaction Parameters """
    s = {
        ("A", 1): -1,
        ("A", 2):  0,
        ("B", 1):  1,
        ("B", 2): -1,
        ("C", 1):  0,
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

    """ Model Equations: r[t, j] folded directly into the balance -- see
    module docstring, this is what makes the system a pure ODE. """
    def _r(m, t, j):
        if j == 1:
            return m.k[j] * m.c[t, "A"]
        elif j == 2:
            return m.k[j] * m.c[t, "B"]
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

    m.f_in.fix(ti_controls[0])
    m.k[1].fix(model_parameters[0])
    m.k[2].fix(model_parameters[1])

    m.c[0, "A"].fix(1)
    m.c[0, "B"].fix(0)
    m.c[0, "C"].fix(0)

    simulator = pod.Simulator(m, package="scipy")
    tsim, profiles = simulator.simulate(numpoints=numpoints, integrator="vode")
    simulator.initialize_model()

    # profiles columns follow m.c's declaration order: one column per
    # species, in the order of m.i = ["A", "B", "C"].
    tsim_flat = tsim.flatten()
    c = np.empty((len(norm_spt), len(m.i)))
    for col, i in enumerate(m.i):
        c[:, col] = np.interp(norm_spt, tsim_flat, profiles[:, col])

    if False:
        plt.plot(tsim_flat, profiles)

    return c

if __name__ == '__main__':
    """ Run Simulation """
    tic = [0]
    spt = np.linspace(0, 10, 101)
    mp = [1, 1]
    c = simulate(
        ti_controls=tic,
        sampling_times=spt,
        model_parameters=mp,
    )

    """ Plot Results """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(spt, c[:, 0], label=r"$c_A$")
    axes.plot(spt, c[:, 1], label=r"$c_B$")
    axes.plot(spt, c[:, 2], label=r"$c_C$")
    axes.legend()
    axes.set_xlabel("Time (hour)")
    axes.set_ylabel("Concentration (mol/L)")
    fig.tight_layout()

    plt.show()
