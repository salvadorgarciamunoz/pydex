"""
asl_elimination_demo.py
=======================================================================
A TEACHING DEMO: how to write a Pyomo model so that pydex's IFT
(Implicit Function Theorem) sensitivity path works correctly.

-----------------------------------------------------------------------
WHAT IS THIS ABOUT?
-----------------------------------------------------------------------
When you give pydex a `pyomo_model_fn`, pydex computes parameter
sensitivities (d output / d parameter) using the IFT method.  To do
that it:

  1. Builds your Pyomo model.
  2. Temporarily UNFIXES the parameter Vars (theta_0, theta_1, ...).
  3. Compiles the model with the ASL backend (via PyomoNLP) to get a
     list of variable names, called the "primal vector".
  4. Finds each parameter's column in that list BY NAME.
  5. Uses those columns to assemble the sensitivities.

The key words in step 4 are BY NAME.  pydex looks up each parameter Var
by its string name (e.g. "theta_0"), NOT by its position in the list.

-----------------------------------------------------------------------
THE TWO BIG LESSONS
-----------------------------------------------------------------------
LESSON 1 — POSITION DOES NOT MATTER.
    It does not matter WHERE in the ASL primal vector your parameters
    end up.  ASL is free to shuffle variables around.  Because pydex
    looks them up by name, it always finds them.  Part 1 below proves
    this with three models that place the parameters at very different
    positions — all three work.

LESSON 2 — THE ONLY REAL FAILURE IS "TRUE ELIMINATION".
    pydex breaks only if a parameter is MISSING from the primal vector
    entirely.  This happens when ASL can "solve away" (eliminate) the
    parameter because it never appears in a live equation.  Part 2 below
    builds such a model on purpose and shows the diagnostic catching it.

-----------------------------------------------------------------------
THE GOLDEN RULE (how to stay safe)
-----------------------------------------------------------------------
Make sure every parameter Var actually appears in at least one equation
that also contains a FREE (unfixed) variable — for example a state
variable like ca[t].  If a parameter only ever multiplies fixed
constants, ASL may eliminate it.

-----------------------------------------------------------------------
THE EXAMPLE REACTION (used everywhere in this file)
-----------------------------------------------------------------------
    A -> products
    dCA/dt = -k(T) * CA
    k(T)   = exp( theta_0 + theta_1 * (T - 273.15) / T )

Two parameters to estimate:  theta = [theta_0, theta_1]
-----------------------------------------------------------------------
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
import logging

# Keep Pyomo quiet so the demo output stays readable.
logging.getLogger('pyomo').setLevel(logging.ERROR)

# The model-vetting tool.  Run this on your model BEFORE handing it to
# the pydex Designer.  It tells you whether IFT will work.
from diagnose_asl_elimination import diagnose_asl_elimination


# =======================================================================
# CONSTANTS used by all the example models
# =======================================================================

T_VAL   = 303.15   # temperature (K) — a fixed control for these examples
TAU_VAL = 100.0    # a time-scaling constant in the ODE
CA0_VAL = 1.0      # initial concentration of A

THETA_0 = -8.0     # nominal value of parameter 0
THETA_1 = 6.0      # nominal value of parameter 1

# Sampling times we pretend to measure at (used by the full-grid check).
SPT = [0.001, 0.25, 0.5, 0.75, 1.0]

# Collocation discretisation settings.
NFE = 10   # number of finite elements
NCP = 3    # collocation points per element


# =======================================================================
# SMALL HELPER FUNCTIONS
# Written the long, explicit way on purpose — easy to read line by line.
# =======================================================================

def discretise_and_solve(model):
    """Apply orthogonal collocation to a Pyomo.DAE model and solve it
    with IPOPT.  Raises if IPOPT does not converge."""
    # 1. Turn the continuous-time DAE into algebraic equations.
    transform = pyo.TransformationFactory('dae.collocation')
    transform.apply_to(model, nfe=NFE, ncp=NCP, scheme='LAGRANGE-RADAU')

    # 2. Set up the IPOPT solver.
    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol'] = 1e-10

    # 3. Solve.
    results = solver.solve(model, tee=False)

    # 4. Make sure it actually worked.
    condition = results.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        raise RuntimeError("IPOPT did not converge")


def make_time_grid(sampling_times):
    """Build the list of time points for the ContinuousSet.

    We use an even grid from 0 to 1 with NFE elements, then add the
    requested sampling times, then sort and de-duplicate."""
    even_grid = np.linspace(0, 1, NFE + 1).tolist()
    all_points = even_grid + list(sampling_times)
    unique_sorted = sorted(set(all_points))
    return unique_sorted


def snap_sampling_times_to_grid(time_grid, sampling_times):
    """For each requested sampling time, find the nearest point that
    actually exists on the collocation grid.

    Returned sorted and de-duplicated."""
    snapped = []
    for s in sampling_times:
        s = float(s)
        # Find the grid point closest to this requested time.
        nearest = min(time_grid, key=lambda grid_t: abs(grid_t - s))
        snapped.append(nearest)
    return sorted(set(snapped))


def collect_equality_bodies(model):
    """Collect the 'body' expression of every equality constraint.

    pydex's IFT contract asks for these.  For a constraint written as
    `lhs == rhs`, the body-minus-upper form is `lhs - rhs`."""
    bodies = []
    for constraint in model.component_objects(pyo.Constraint, active=True):
        for index in constraint:
            single = constraint[index]
            if single.equality:
                bodies.append(single.body - single.upper)
    return bodies


# =======================================================================
# PART 1 — POSITION DOES NOT MATTER
#
# Below are THREE different ways to build the SAME reaction model.
# Each one causes ASL to place theta_0 and theta_1 at different
# positions in the primal vector.  All three are CORRECT for IFT,
# because in every one the parameters still appear in a live equation
# (so they survive into the primal vector) and pydex finds them by name.
# =======================================================================

def build_RIGHT_params_first(ti_controls, model_parameters, sampling_times=None):
    """
    RIGHT WAY #1 — the simplest, cleanest model.

    The two parameters are written directly inside the ODE.  No extra
    helper variables.  ASL places them near the FRONT of the primal
    vector (positions 0 and 1), but remember: position does not matter.
    """
    # --- unpack the inputs ---
    T_value   = float(ti_controls[0])
    theta0    = float(model_parameters[0])
    theta1    = float(model_parameters[1])
    spt_array = np.asarray(sampling_times, dtype=float)

    # --- build the model skeleton ---
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=make_time_grid(spt_array))

    # --- the parameters: declared as Vars, then FIXED to their values ---
    # (pydex needs them as Vars so it can temporarily unfix them later.)
    m.theta_0 = pyo.Var(initialize=theta0)
    m.theta_0.fix(theta0)
    m.theta_1 = pyo.Var(initialize=theta1)
    m.theta_1.fix(theta1)

    # --- temperature: a fixed control, also a Var ---
    m.T = pyo.Var(initialize=T_value)
    m.T.fix(T_value)

    # --- the state variable and its time derivative ---
    m.ca = pyo.Var(m.t, initialize=CA0_VAL, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    # --- the ODE:  dCA/dt = -tau * k(T) * CA ---
    # NOTE: theta_0 and theta_1 appear here, multiplied together with the
    # FREE state variable ca[t].  That keeps them "alive" for ASL.
    def ode_rule(m, t):
        k = pyo.exp(m.theta_0 + m.theta_1 * (m.T - 273.15) / m.T)
        return m.dca_dt[t] == -TAU_VAL * k * m.ca[t]
    m.ode = pyo.Constraint(m.t, rule=ode_rule)

    # --- initial condition ---
    m.ic = pyo.Constraint(expr=m.ca[0] == CA0_VAL)

    # --- a trivial objective (we only want to SOLVE, not optimise) ---
    m.obj = pyo.Objective(expr=0.0)

    # --- discretise and solve ---
    discretise_and_solve(m)

    # --- assemble the IFT contract tuple ---
    # IMPORTANT: the parameter Vars MUST be the first entries of all_vars.
    time_grid = sorted(m.t)
    all_vars = [m.theta_0, m.theta_1]
    all_vars += [m.ca[t] for t in time_grid]
    all_vars += [m.dca_dt[t] for t in time_grid]

    bodies   = collect_equality_bodies(m)
    spt_used = snap_sampling_times_to_grid(time_grid, spt_array)

    return m, all_vars, bodies, spt_used


def build_RIGHT_params_displaced(ti_controls, model_parameters, sampling_times=None):
    """
    RIGHT WAY #2 — same physics, but with an extra helper variable k[t].

    Here we introduce a FREE indexed variable `k[t]` to hold the rate
    constant.  Because k[t] is a free indexed Var, ASL lists all of its
    time points BEFORE the parameters.  So theta_0 and theta_1 get
    "displaced" to much later positions in the primal vector.

    THIS IS STILL CORRECT.  The parameters still appear in a live
    equation (the definition of k[t]), so they survive, and pydex finds
    them by name regardless of position.
    """
    # --- unpack the inputs ---
    T_value   = float(ti_controls[0])
    theta0    = float(model_parameters[0])
    theta1    = float(model_parameters[1])
    spt_array = np.asarray(sampling_times, dtype=float)

    # A reasonable starting guess for k, for the solver.
    k_guess = float(np.exp(theta0 + theta1 * (T_value - 273.15) / T_value))

    # --- model skeleton ---
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=make_time_grid(spt_array))

    # --- parameters (fixed Vars) ---
    m.theta_0 = pyo.Var(initialize=theta0)
    m.theta_0.fix(theta0)
    m.theta_1 = pyo.Var(initialize=theta1)
    m.theta_1.fix(theta1)

    # --- temperature (fixed Var) ---
    m.T = pyo.Var(initialize=T_value)
    m.T.fix(T_value)

    # --- state variable + derivative ---
    m.ca = pyo.Var(m.t, initialize=CA0_VAL, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    # --- the extra helper: a FREE indexed Var k[t] ---
    # This is what "displaces" the parameters in the ASL ordering.
    m.k = pyo.Var(m.t, initialize=k_guess, bounds=(0, None))

    # --- equation defining k[t] (this is where the parameters live) ---
    def k_definition_rule(m, t):
        return m.k[t] == pyo.exp(m.theta_0 + m.theta_1 * (m.T - 273.15) / m.T)
    m.k_def = pyo.Constraint(m.t, rule=k_definition_rule)

    # --- the ODE now uses k[t] instead of the inline expression ---
    def ode_rule(m, t):
        return m.dca_dt[t] == -TAU_VAL * m.k[t] * m.ca[t]
    m.ode = pyo.Constraint(m.t, rule=ode_rule)

    # --- initial condition + trivial objective ---
    m.ic = pyo.Constraint(expr=m.ca[0] == CA0_VAL)
    m.obj = pyo.Objective(expr=0.0)

    # --- solve ---
    discretise_and_solve(m)

    # --- IFT contract tuple (parameters first!) ---
    time_grid = sorted(m.t)
    all_vars = [m.theta_0, m.theta_1]
    all_vars += [m.ca[t] for t in time_grid]
    all_vars += [m.k[t] for t in time_grid]
    all_vars += [m.dca_dt[t] for t in time_grid]

    bodies   = collect_equality_bodies(m)
    spt_used = snap_sampling_times_to_grid(time_grid, spt_array)

    return m, all_vars, bodies, spt_used


def build_RIGHT_params_very_displaced(ti_controls, model_parameters, sampling_times=None):
    """
    RIGHT WAY #3 — same physics, but with EVEN MORE helper variables.

    Now we add a free scalar `temp` plus two free indexed Vars,
    `ln_k[t]` and `k[t]`.  This pushes theta_0 and theta_1 very far down
    the ASL primal vector (in the original study, around position 122).

    STILL CORRECT, for the same reason: the parameters appear in the
    equation that defines ln_k[t], so they survive, and pydex finds them
    by name.
    """
    # --- unpack the inputs ---
    T_value   = float(ti_controls[0])
    theta0    = float(model_parameters[0])
    theta1    = float(model_parameters[1])
    spt_array = np.asarray(sampling_times, dtype=float)

    # Starting guesses for the solver.
    ln_k_guess = theta0 + theta1 * (T_value - 273.15) / T_value
    k_guess    = float(np.exp(ln_k_guess))

    # --- model skeleton ---
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=make_time_grid(spt_array))

    # --- parameters (fixed Vars) ---
    m.theta_0 = pyo.Var(initialize=theta0)
    m.theta_0.fix(theta0)
    m.theta_1 = pyo.Var(initialize=theta1)
    m.theta_1.fix(theta1)

    # --- a FREE scalar variable for temperature ---
    # It is pinned to T_value by an equality constraint, but because it
    # is free (not .fix()'d) it sits at the very front of the primal
    # vector, displacing the parameters further.
    m.temp = pyo.Var(initialize=T_value)
    m.temp_fix = pyo.Constraint(expr=m.temp == T_value)

    # --- state variable + derivative ---
    m.ca = pyo.Var(m.t, initialize=CA0_VAL, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    # --- two free indexed helper Vars ---
    m.ln_k = pyo.Var(m.t, initialize=ln_k_guess)
    m.k    = pyo.Var(m.t, initialize=k_guess, bounds=(0, None))

    # --- equation defining ln_k[t] (this is where the parameters live) ---
    def ln_k_rule(m, t):
        return m.ln_k[t] == m.theta_0 + m.theta_1 * (m.temp - 273.15) / m.temp
    m.ln_k_def = pyo.Constraint(m.t, rule=ln_k_rule)

    # --- equation defining k[t] from ln_k[t] ---
    def k_rule(m, t):
        return m.k[t] == pyo.exp(m.ln_k[t])
    m.k_def = pyo.Constraint(m.t, rule=k_rule)

    # --- the ODE uses k[t] ---
    def ode_rule(m, t):
        return m.dca_dt[t] == -TAU_VAL * m.k[t] * m.ca[t]
    m.ode = pyo.Constraint(m.t, rule=ode_rule)

    # --- initial condition + trivial objective ---
    m.ic = pyo.Constraint(expr=m.ca[0] == CA0_VAL)
    m.obj = pyo.Objective(expr=0.0)

    # --- solve ---
    discretise_and_solve(m)

    # --- IFT contract tuple (parameters first!) ---
    time_grid = sorted(m.t)
    all_vars = [m.theta_0, m.theta_1]
    all_vars += [m.ca[t] for t in time_grid]
    all_vars += [m.ln_k[t] for t in time_grid]
    all_vars += [m.k[t] for t in time_grid]
    all_vars += [m.dca_dt[t] for t in time_grid]

    bodies   = collect_equality_bodies(m)
    spt_used = snap_sampling_times_to_grid(time_grid, spt_array)

    return m, all_vars, bodies, spt_used


# =======================================================================
# PART 2 — THE WRONG WAY: TRUE ELIMINATION
#
# This model breaks IFT.  We list theta_1 in all_vars (so pydex expects
# to find it), but we never actually USE theta_1 in any equation — we
# bake its numeric value into a Python float instead.  ASL therefore
# never sees theta_1, it is absent from the primal vector, and pydex
# cannot find its column.
# =======================================================================

def build_WRONG_true_elimination(ti_controls, model_parameters, sampling_times=None):
    """
    WRONG WAY — theta_1 is eliminated.

    The trap: theta_1's value is folded into a plain Python number and
    used as a CONSTANT in the ODE.  The Pyomo model never references a
    theta_1 Var, so ASL has nothing to keep.  We still put a theta_1 Var
    in all_vars (as pydex requires the parameters first), but it is a
    "ghost": present in the list, absent from the compiled model.

    Expected result: diagnose_asl_elimination() reports theta_1 as
    ELIMINATED, and pydex would raise a clear error rather than running.
    """
    # --- unpack the inputs ---
    T_value   = float(ti_controls[0])
    theta0    = float(model_parameters[0])
    theta1    = float(model_parameters[1])
    spt_array = np.asarray(sampling_times, dtype=float)

    # --- model skeleton ---
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=make_time_grid(spt_array))

    # --- only theta_0 is declared and used as a real Var ---
    m.theta_0 = pyo.Var(initialize=theta0)
    m.theta_0.fix(theta0)
    # NOTE: there is deliberately NO theta_1 Var used in any equation.

    # --- temperature (fixed Var) ---
    m.T = pyo.Var(initialize=T_value)
    m.T.fix(T_value)

    # --- state variable + derivative ---
    m.ca = pyo.Var(m.t, initialize=CA0_VAL, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    # --- the ODE: theta_1's effect is baked in as a CONSTANT ---
    # `theta1_as_constant` is a plain Python float, NOT a Pyomo Var.
    # This is the mistake: theta_1 never enters the model as a variable.
    theta1_as_constant = theta1 * (T_value - 273.15) / T_value
    def ode_rule(m, t):
        k = pyo.exp(m.theta_0 + theta1_as_constant)
        return m.dca_dt[t] == -TAU_VAL * k * m.ca[t]
    m.ode = pyo.Constraint(m.t, rule=ode_rule)

    # --- initial condition + trivial objective ---
    m.ic = pyo.Constraint(expr=m.ca[0] == CA0_VAL)
    m.obj = pyo.Objective(expr=0.0)

    # --- solve ---
    discretise_and_solve(m)

    # --- now create a GHOST theta_1 Var, AFTER solving ---
    # It has no constraints, so ASL will not keep it.  We add it to
    # all_vars only so pydex tries (and fails) to find it.
    m.theta_1_ghost = pyo.Var(initialize=theta1)
    m.theta_1_ghost.fix(theta1)

    # --- IFT contract tuple (parameters first!) ---
    time_grid = sorted(m.t)
    all_vars = [m.theta_0, m.theta_1_ghost]   # theta_1_ghost is absent in ASL
    all_vars += [m.ca[t] for t in time_grid]
    all_vars += [m.dca_dt[t] for t in time_grid]

    bodies   = collect_equality_bodies(m)
    spt_used = snap_sampling_times_to_grid(time_grid, spt_array)

    return m, all_vars, bodies, spt_used


# =======================================================================
# HELPER: print the diagnostic verdict in plain language
# =======================================================================

def report_verdict(label, result):
    """Print a one-line PASS / FAIL / UNVERIFIED summary for a result
    dict returned by diagnose_asl_elimination()."""
    if result['errored']:
        # The diagnostic could not even build/compile the model.
        print(f"  {label}")
        print(f"      -> UNVERIFIED (could not run: {result['error']})")
    elif result['ift_ready']:
        # All parameters present in the primal vector — IFT will work.
        print(f"  {label}")
        print(f"      -> PASS  (all parameters reachable; IFT will work)")
    else:
        # At least one parameter is missing from the primal vector.
        missing = [name for _, name in result['eliminated_full']]
        missing += [name for _, name in result['eliminated_single']]
        unique_missing = sorted(set(missing))
        print(f"  {label}")
        print(f"      -> FAIL  (eliminated: {unique_missing}; IFT would break)")


# =======================================================================
# MAIN: run each example through the diagnostic
# =======================================================================

def main():
    line = "=" * 70

    print(line)
    print("  ASL ELIMINATION DEMO")
    print("  Reaction:  dCA/dt = -tau * exp(theta_0 + theta_1*(T-273)/T) * CA")
    print(line)

    # The test point and parameter values used for every example.
    ti_controls      = [T_VAL]
    model_parameters = [THETA_0, THETA_1]
    param_names      = ["theta_0", "theta_1"]

    # -------------------------------------------------------------------
    # PART 1 — three RIGHT models; all should PASS
    # -------------------------------------------------------------------
    print()
    print("PART 1 - Position does not matter (all three should PASS)")
    print("-" * 70)

    # We pair each builder with a short description, then run them all.
    right_models = [
        ("RIGHT #1: params written inline (params near the front)",
         build_RIGHT_params_first),
        ("RIGHT #2: params displaced by a free k[t]",
         build_RIGHT_params_displaced),
        ("RIGHT #3: params displaced even more (temp + ln_k + k)",
         build_RIGHT_params_very_displaced),
    ]

    for label, builder in right_models:
        result = diagnose_asl_elimination(
            builder,
            ti_controls=ti_controls,
            model_parameters=model_parameters,
            sampling_times=SPT,
            param_names=param_names,
            verbose=False,
        )
        report_verdict(label, result)

    # -------------------------------------------------------------------
    # PART 2 — the WRONG model; should FAIL
    # -------------------------------------------------------------------
    print()
    print("PART 2 - True elimination (this one should FAIL)")
    print("-" * 70)

    result = diagnose_asl_elimination(
        build_WRONG_true_elimination,
        ti_controls=ti_controls,
        model_parameters=model_parameters,
        sampling_times=SPT,
        param_names=param_names,
        verbose=False,
    )
    report_verdict("WRONG: theta_1 baked into a Python constant", result)

    # -------------------------------------------------------------------
    # TAKEAWAYS
    # -------------------------------------------------------------------
    print()
    print(line)
    print("  TAKEAWAYS")
    print(line)
    print("  1. Position in the ASL primal vector is irrelevant - pydex")
    print("     finds parameters by NAME.  All three RIGHT models pass.")
    print()
    print("  2. The only real failure is TRUE ELIMINATION: a parameter")
    print("     that never appears in a live equation, so ASL drops it.")
    print()
    print("  3. GOLDEN RULE: make every parameter appear in at least one")
    print("     equation that also contains a free variable (e.g. ca[t]).")
    print()
    print("  4. Always run diagnose_asl_elimination() on your model BEFORE")
    print("     handing it to the pydex Designer.")
    print(line)


if __name__ == "__main__":
    main()
