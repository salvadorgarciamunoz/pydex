# pydex — Python Design of Experiments

An open-source Python package for optimal experiment design, essential to
a modeller's toolbelt. If you develop a model of any kind, you will relate
to the challenges of estimating its parameters. This tool helps design
maximally informative experiments for collecting data to calibrate your
model.

## Fork Notice

This is a fork of [pydex](https://github.com/KennedyPutraKusumo/pydex)
by Kennedy Putra Kusumo et al., originally described in:

> Kusumo, K.P., Kuriyan, K., Vaidyaraman, S., García-Muñoz, S., Shah, N.
> & Chachuat, B. (2022). Risk mitigation in model-based experiment
> design: a continuous-effort approach to optimal campaigns.
> *Computers & Chemical Engineering*, 159, 107680.
> https://doi.org/10.1016/j.compchemeng.2022.107680

### Changes in this fork

- **V-optimal MBDoE**: two-stage workflow targeting prediction accuracy
  at a user-specified operating condition (`find_optimal_operating_point()`
  \+ `design_v_optimal()`)
- **Process optimisation**: constrained nonlinear optimisation of the
  operating point (Stage 1) before designing experiments. Solved via
  Pyomo PyNumero + `cyipopt` when available, falling back automatically
  to scipy SLSQP otherwise — `cyipopt` is **not** required
- **Pyomo-centric solver**: cvxpy removed; the pydex OED problem is now
  formulated and solved entirely via Pyomo, giving access to any solver
  Pyomo knows about (IPOPT, GLPK, Gurobi, CPLEX, Bonmin, SHOT, GAMS/BARON,
  …) through `design_experiment(solver=..., solver_options={...})`.
  IPOPT is the default.
- **Parallel IFT sensitivity evaluation**: when a `pyomo_model_fn` is
  provided, sensitivities are computed via the Implicit Function Theorem
  (IFT) using the Pyomo NLP Jacobian; parallelised over candidates (local
  designs) or scenarios (pseudo-Bayesian designs) via `joblib` loky workers
- **Pyomo.DAE support**: full DAE models can serve as both the simulator
  and the IFT sensitivity source; signature-2 multi-output models
  supported
- **regularize_fim fix**: `regularize_fim=True` now correctly adds ε·I
  to the symbolic FIM expression in the Pyomo solve (previously the flag
  was stored but had no computational effect on the native solve path)
- **Comprehensive test suite**: 32 end-to-end tests covering all design
  criteria, both sensitivity paths (FD and IFT), parallel correctness,
  prior FIM, save/load, visualisation, and more
- **Improved documentation**: Sphinx/NumPy-compliant docstrings throughout
  `designer.py`

---

## Installation

Install directly from this fork:

```bash
pip install git+https://github.com/salvadorgarciamunoz/pydex.git
```

> **Note:** `pip install pydex` from PyPI installs the *original* upstream
> package, **not** this fork. The two have different architectures — use
> the `git+` command above to get this version.

---

## Quick Start

See the `examples/` folder for worked examples using the current API.
Good starting points:

- `examples/ode/case_1.py` — the simplest D-optimal design (first-order
  reaction, single parameter), with exact IFT sensitivities
- `examples/ode/case_3_ift.py` — a larger D-optimal design (nine-parameter
  reaction network) showcasing the IFT speed-up
- `examples/jupyter/pydex_quickstart.ipynb` — introductory notebook on
  basic features

---

## Features

1. Simple, intuitive syntax — easy to get started, powerful enough for
   complex problems.
2. Continuous and exact (discrete) experimental designs via Adams
   apportionment.
3. Design criteria: D-optimal, A-optimal, E-optimal, V-optimal, CVaR
   variants, pseudo-Bayesian (types 0 and 1), MINLP sparse designs.
4. OED problem formulated entirely in Pyomo — any solver accessible
   through Pyomo can be used, with IPOPT as the default.
5. Pyomo.DAE support: DAE models as simulator and IFT sensitivity source.
6. Parallel sensitivity evaluation via `joblib` loky workers.
7. Convenient built-in visualisation via matplotlib.
8. Supports virtually any model written as a Python function, including
   ODE models solved via scipy or Pyomo.DAE.

---

## Dependencies

### Core (installed automatically)

| Package        | Purpose                                              |
|----------------|------------------------------------------------------|
| numpy          | Array operations                                     |
| scipy          | ODE integration, optimisation fallback               |
| matplotlib     | Visualisation                                        |
| numdifftools   | Numerical finite-difference sensitivities            |
| pyomo          | OED problem formulation, DAE modelling, IFT Jacobian |
| joblib         | Parallel sensitivity evaluation                      |
| dill           | Saving objects with weak references                  |

### Solvers

Because the OED problem is formulated entirely in Pyomo, solver
requirements depend on what you are solving.

**Standard continuous designs** (D/A/E/V-optimal, pseudo-Bayesian, CVaR)
require an NLP solver accessible through Pyomo. IPOPT is the default and
is recommended; the solver executable must be on your `PATH`:

```python
designer.design_experiment(
    criterion      = designer.d_opt_criterion,
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57", "tol": 1e-8},
)
```

Any other NLP solver registered with Pyomo's `SolverFactory`
(e.g. `bonmin`, `glpk`, `cplex`) can be used by passing
`solver=<solver_name>`.

> For best IPOPT performance, configure it with HSL linear solvers
> (`MA27`, `MA57`) — see `docs/ipopt_setup_guide.docx`. The open-source
> `MUMPS` solver works as a fallback.

**V-optimal operating-point optimisation** (Stage 1 of the V-optimal
workflow, `find_optimal_operating_point()`) attempts a Pyomo PyNumero
solve and, if `cyipopt` is installed, uses it for that path. If `cyipopt`
(or PyNumero) is unavailable, it falls back automatically to scipy's
SLSQP optimiser. Installing `cyipopt` is therefore **optional** — it can
speed up the operating-point step but is never required.

**Sparsity-enforcing MINLP designs** (`min_effort > 0`) require a MINLP
solver. BARON via GAMS is recommended:

```python
designer.design_experiment(
    criterion  = designer.d_opt_criterion,
    solver     = "gams",
    min_effort = 0.05,
)
```

---

## API Overview

### Basic D-optimal design

```python
from pydex.core.designer import Designer
import numpy as np

designer = Designer()
designer.simulate               = my_simulate_fn   # callable(tic, mp) -> array
designer.model_parameters       = np.array([...])
designer.ti_controls_candidates = candidate_grid

designer.initialize(verbose=1)
designer.eval_sensitivities()

designer.design_experiment(
    criterion      = designer.d_opt_criterion,
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57", "tol": 1e-8},
)
designer.print_optimal_candidates()
designer.apportion(n_exp=10)
```

### Pyomo.DAE model with automatic IFT and parallelisation

When `pyomo_model_fn` is provided, `use_pyomo_ift` and `n_jobs` are
auto-set at `initialize()` — no manual configuration needed:

```python
designer.simulate         = my_simulate_fn      # for predictions
designer.pyomo_model_fn   = my_build_model_fn   # for IFT Jacobian
designer.model_parameters = np.array([...])
designer.initialize(verbose=1)
# use_pyomo_ift=True and n_jobs=-1 set automatically

designer.design_experiment(
    criterion            = designer.d_opt_criterion,
    solver               = "ipopt",
    solver_options       = {"linear_solver": "ma57"},
    pseudo_bayesian_type = 0,   # for pseudo-Bayesian designs
)
```

### Pseudo-Bayesian design

```python
scenarios = np.column_stack([
    np.random.uniform(lb, ub, N),  # one column per uncertain parameter
    ...
])
designer.model_parameters = scenarios   # shape (N_scenarios, n_mp)

designer.design_experiment(
    criterion            = designer.d_opt_criterion,
    solver               = "ipopt",
    solver_options       = {"linear_solver": "ma57"},
    pseudo_bayesian_type = 0,   # 0 = average FIM; 1 = average criterion
)
```

---

## V-optimal MBDoE

V-optimal design minimises model prediction variance at a specific
operating condition `dw` (e.g. the economically optimal process point),
rather than minimising global parameter uncertainty as D/A-optimal designs
do. It follows a two-stage workflow:

**Stage 1 — Process optimisation:** find `dw` by solving a constrained
nonlinear programme over the operating space. This uses Pyomo PyNumero +
`cyipopt` if available, and falls back to scipy SLSQP otherwise.

**Stage 2 — V-optimal MBDoE:** design experiments that minimise
`J_V = trace(W FIM⁻¹ Wᵀ)` where `W` is the scaled sensitivity matrix
evaluated at `dw`.

```python
# Stage 1
designer.process_objective   = my_objective    # callable(tic, tvc, mp) -> float
designer.process_constraints = my_constraints  # callable(tic, tvc, mp) -> list
designer.dw_sense            = "maximize"
designer.dw_bounds_tic       = [(lb, ub), ...]

designer.find_optimal_operating_point(
    init_guess     = np.array([[60.0, 70.0, 1.0]]),
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57"},
)

# Stage 2
designer.dw_spt = np.array([t_final])
designer.design_experiment(
    criterion               = designer.v_opt_criterion,
    solver                  = "ipopt",
    solver_options          = {"linear_solver": "ma57"},
    optimize_sampling_times = True,
)
```

See `testing_scripts/v_optimal_test_case.py` for a complete worked
example with a three-reaction batch reactor system.

> Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
> V-optimal design of experiments for building fundamental models of
> pharmaceutical production processes. *Computers & Chemical Engineering*,
> 129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

---

## Examples

The `examples/` folder is organised into three subfolders.

### `examples/ode/` — ODE/DAE design scripts

Three reaction systems of increasing complexity. For each case, a
companion `*_model.py` module defines the model (`simulate` +
`build_pyomo_model`) and the driver script runs the design. The filename
suffixes select the sensitivity / integration path:

- plain `case_N.py` — exact IFT sensitivities via Pyomo collocation +
  IPOPT (PyomoNLP)
- `*_no_ift.py` — finite-difference sensitivities over the same
  collocation model (`pyomo_model_fn` not assigned)
- `*_no_ift_no_collocation.py` — finite differences over the Pyomo
  Simulator (scipy/vode) forward integration; also demonstrates the
  safety check that blocks IFT on a non-discretised model

**Case 1 — first-order reaction** (`dCA/dt = −k·CA`, single parameter):
`case_1.py`, `case_1_no_ift.py`, `case_1_no_ift_no_collocation.py`.

**Case 2 — A→B with Arrhenius kinetics** (four parameters `[θ₀, θ₁, α, ν]`,
controls `[CA0, T]`, responses `[CA, CB]`): `case_2.py`,
`case_2_no_ift.py`, `case_2_no_ift_no_collocation.py`.

**Case 3 — Michaelis–Menten-style network** (nine parameters, controls
`[CA0, T, τ]`, responses `[CA, CB]`):

- `case_3.py` — scipy / finite-difference baseline (slow; ~350 s
  sensitivity analysis)
- `case_3_ift.py` — exact IFT sensitivities via collocation + IPOPT
  (~5–15 s; roughly 20–70× faster)

### `examples/jupyter/` — introductory notebooks

Both notebooks use the current API.

- `pydex_quickstart.ipynb` — introductory D-optimal design for a
  steady-state system, fitting an order-1 polynomial response-surface
  model in two control variables
- `pydex_ode_model.ipynb` — D-optimal design for an ODE model: a batch
  reactor with an `A→νB` reaction, integrated via scipy

### `examples/ASL Elimination/` — IFT ASL-elimination demos

Demonstration and diagnostic scripts for the AMPL Solver Library (ASL)
variable-elimination behaviour encountered in the IFT sensitivity path:

- `asl_elimination_demo.py`
- `diagnose_asl_elimination.py`
- `pydex_ift_asl_guide.docx` — accompanying guide

---

## Testing scripts

The `testing_scripts/` folder contains standalone scripts used to verify
and demonstrate the package end-to-end. They double as larger worked
examples.

- **`pydex_full_capability_test.py`** — the comprehensive capability
  suite: 37 end-to-end tests built on the three-reaction batch model
  (A→B desired, A→I impurity, A→D decomposition), run in sequence and
  gated by a single pass/fail check. Coverage includes: setup and
  initialisation; candidate-grid helpers; sensitivity analysis,
  visualisation, and diagnosis; D-, A-, E-, and V-optimal designs;
  sampling-time optimisation; pseudo-Bayesian designs (types 0 and 1);
  CVaR D-optimal and the CVaR bi-objective Pareto frontier;
  continuous→exact apportionment (Adams method); prior FIM (from external
  covariance and from prior experiments); save/load of OED results and of
  full designer state; the complete visualisation suite;
  sparsity-enforcing MINLP designs (BARON via GAMS); both sensitivity
  paths (finite-difference and Pyomo IFT) with sequential-vs-parallel
  correctness checks; FD-vs-IFT and DAE-vs-analytical agreement;
  signature-2 multi-output models; the `regularize_fim` path;
  normalisation toggling; discrete (`n_exp`) designs; and IFT
  sampling-time optimisation.

  ```bash
  python testing_scripts/pydex_full_capability_test.py
  ```

  > The MINLP section requires GAMS/BARON; if those are unavailable that
  > section is the only part that will not run.

- **`v_optimal_test_case.py`** — full two-stage V-optimal MBDoE on the
  three-reaction batch reactor: Stage 1 finds the operating point that
  maximises yield subject to quality/safety constraints; Stage 2 designs
  the V-optimal experiment and compares it against A- and D-optimal
  designs to quantify the prediction-accuracy benefit.

- **`v_optimal_test_case_pyomo.py`** — the same V-optimal workflow run
  through the Pyomo IFT sensitivity path.

- **`first_order_reaction.py`** — minimal D-optimal example on the
  first-order reaction `A→B` using the analytic solution and
  finite-difference sensitivities. Includes a local D-optimal run and two
  pseudo-Bayesian runs (uncertainty in `k` only, and in both `k` and
  `A0`).

- **`first_order_reaction_pyomo.py`** — the same first-order problem
  solved through the Pyomo IFT path, illustrating the minimal
  `build_pyomo_model()` a user must supply for IFT sensitivities.

---

Do you have a question, suggestion, or feature request? Feel free to open
an issue or contact the original author at
[kennedy.putra.kusumo@gmail.com](mailto:kennedy.putra.kusumo@gmail.com).
</file_text>