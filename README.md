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
- **Ds-optimal design**: design for a chosen SUBSET of the parameters
  (`designer.interest_parameters`, set BY NAME) while marginalising the
  rest, via the Schur complement of the nuisance block. Works on models
  whose FIM is singular in a direction you do not care about, where
  D-optimality cannot proceed
- **Estimability analysis**: `run_estimability()` ranks parameters from
  most to least estimable (Yao/McAuley orthogonalisation via pivoted QR)
  and reports which are mutually correlated, hence interchangeable
- **Structural-singularity gate**: `design_experiment()` refuses a
  rank-deficient FIM by default and names the parameters responsible,
  rather than returning a plausible number from a floored Cholesky
  factor. Override with `allow_singular_fim=True`; Ds-optimality is exempt
- **Comprehensive test suite**: 60 sections / 301 assertions covering all
  design criteria, both sensitivity paths (FD and IFT), parallel
  correctness, prior FIM, save/load, visualisation, and more
- **Per-parameter finite-difference step**: the FD step is sized from each
  parameter's own nominal magnitude (1% by default, floored at `1e-8`),
  not a flat constant shared by every parameter. Before 0.3.0 the default
  was a flat `base_step=2`, which silently produced badly wrong
  sensitivities for any parameter with nominal magnitude well below 1 — a
  rate constant of 0.02 was perturbed by 100x its own value, far outside
  the linear regime Richardson extrapolation assumes, with the error
  growing along the trajectory. Nothing warned; the numbers were simply
  wrong, and only for the small-magnitude parameters, which made the
  pattern look like a structural problem with the model rather than a
  step-size problem. The same scaling applies to `set_prior_experiments()`
  and the V-optimal `W` matrix. Passing an explicit `base_step` restores
  the old unconditional behaviour and puts the scaling in your hands; run
  with `verbose >= 2` to see the resolved per-parameter step
- **Improved documentation**: Google-style docstrings throughout
  `designer.py`, rendered to HTML with Sphinx (see `docs/`)

A full list of changes, including the twelve bugs fixed in this fork and
which test section guards each, is in [CHANGELOG.md](CHANGELOG.md).

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

[`examples/README.md`](examples/README.md) indexes all of them and
explains the `_no_ift` / `_no_ift_no_collocation` naming scheme, which
lets you compare sensitivity methods on an otherwise identical problem.

## Documentation

Built docs: **https://salvadorgarciamunoz.github.io/pydex/**

The API reference is built from the in-source docstrings with Sphinx. To
build it yourself instead of using the hosted copy above:

```bash
pip install sphinx sphinx-rtd-theme
python -m sphinx -b html docs/source docs/build/html
```

Then open `docs/build/html/index.html`.

---

## Features

1. Simple, intuitive syntax — easy to get started, powerful enough for
   complex problems.
2. Continuous and exact (discrete) experimental designs via Adams
   apportionment.
3. **Design criteria** — see [Design Criteria](#design-criteria) below for
   the complete list; this is not just D/A/E/V.
4. **Estimability analysis and FIM diagnostics** — `run_estimability()`
   and `diagnose_fim_structure()`, usually the first things worth running
   on a new model, before choosing a criterion at all.
5. **Sequential / prior-informed design** — `set_prior_fim()` /
   `set_prior_experiments()` register information from experiments already
   run (an external covariance matrix, or raw conditions at arbitrary
   points), so a new design accounts for what you already know rather than
   starting from zero.
6. OED problem formulated entirely in Pyomo — any solver accessible
   through Pyomo can be used, with IPOPT as the default.
7. Pyomo.DAE support: DAE models as simulator and IFT sensitivity source.
8. Parallel sensitivity evaluation via `joblib` loky workers.
9. Convenient built-in visualisation via matplotlib.
10. Supports virtually any model written as a Python function, including
    ODE models solved via scipy or Pyomo.DAE.

---

## Design Criteria

Most criteria are bound methods passed to `design_experiment()`:
`designer.design_experiment(designer.d_opt_criterion, solver="ipopt")`.
CVaR-D is the exception — see its row below. Full details, including
failure modes and when to prefer one criterion over another, are in the
`Designer` class docstring (`pydex/core/designer.py`) and rendered in the
[API reference](https://salvadorgarciamunoz.github.io/pydex/).

| Criterion | Method | What it optimises |
|---|---|---|
| D-optimal | `d_opt_criterion` | `det(FIM)` — the default; invariant to reparameterisation |
| A-optimal | `a_opt_criterion` | `trace(FIM⁻¹)` — total parameter variance |
| E-optimal | `e_opt_criterion` | smallest eigenvalue of the FIM — the worst-determined direction |
| **Ds-optimal** | `ds_opt_criterion` | D-optimality on a chosen SUBSET of parameters (`interest_parameters`, set by name), marginalising the rest via the Schur complement — works even when a nuisance parameter is unidentifiable |
| V-optimal | `v_opt_criterion` | prediction variance at a specific operating condition `dw`, via a two-stage workflow — see [V-optimal MBDoE](#v-optimal-mbdoe) below |
| vdi | `vdi_criterion` | prediction variance aggregated over the whole operating-point grid, rather than one `dw` — distinct from V-optimal only when there are fewer measured responses than parameters |
| CVaR-D (risk-averse) | `cvar_d_opt_criterion` | average D-criterion over the worst `(1-beta)` fraction of parameter scenarios, via `solve_cvar_problem()` rather than `design_experiment()` |
| **b-optimal (bracketing)** | `b_opt_criterion` | NOT parameter-precision at all: brackets the operating space. Weighted sum of input-space D-optimality on the scaled input factors and output-space coverage of the predicted responses, traded off by `output_weight`. Needs an MINLP solver, an exact `n_exp`, and `simulate_candidates()` first — see below |

`b_opt_criterion` is the odd one out and worth reading the row twice: it does
not involve the Fisher information matrix, does not use sensitivities, and does
not try to determine parameters. It answers the regulator's question in a
bracketing study — which experiments span the operating space — and is posed as
binary subset selection over a pre-evaluated candidate pool. Implements Chen,
Paulavičius, Adjiman & García-Muñoz (2018), *AIChE J.* 64(11):3944–3957. Worked
examples in `examples/b_optimal/`; the reachability limitation of the
weighted-sum formulation is recorded in `CHANGELOG.md` under 0.4.0.

**What this table deliberately omits.** `u_opt_criterion` is marked
`# experimental` in the source, has zero coverage in the capability suite and is
not a term from the standard DoE literature. The six other prediction-variance
criteria (`dg`, `di`, `ag`, `ai`, `eg`, `ei`) are real and correctly
implemented, but are only ever evaluated on a fixed effort vector — none has
completed an actual `design_experiment()` optimisation, unlike `vdi_criterion`,
which has. Both omissions are intentional and should stay until that changes:
this list has previously been wrong in both directions, and the check is
`grep "_opt_criterion" pydex/core/designer.py` against the capability suite,
not what a previous pass wrote here.

**Pseudo-Bayesian designs** are not a separate criterion but a mode:
supply `model_parameters` as a scenario array (shape `(n_scenarios, n_mp)`)
instead of a single vector, and pass `pseudo_bayesian_type=0` (average
information, cheaper, native Pyomo solve) or `=1` (average criterion, more
faithful for non-linear criteria, falls back to SLSQP) to
`design_experiment()`. Applies to D, Ds, A, and E.

---

## Sampling Times — `n_spt` is the only control

For a dynamic model, `n_spt` decides what one unit of the effort budget buys.
There is no flag that switches sampling-time optimisation on or off.

| What you want | How to ask | Effort allocated per |
|---|---|---|
| Choose conditions **and** which times to measure | omit `n_spt` (default) | measurement: one (candidate, time) cell |
| Exactly `k` samples per run, optimiser chooses which `k` | `n_spt=k` | run of `k` samples: one (candidate, schedule) |
| Measure every listed time on every run | `n_spt=<number of listed times>` | one whole experiment |

```python
designer.design_experiment(designer.d_opt_criterion, solver="ipopt")
# optimised: the design picks which of the listed times to measure

designer.design_experiment(designer.d_opt_criterion, n_spt=2, solver="ipopt")
# exactly two samples per run; the optimiser picks the best pair

designer.design_experiment(designer.d_opt_criterion, n_spt=designer.n_spt,
                           solver="ipopt")
# fixed grid: every listed time measured on every run
```

The first and third are different design problems. With sampling times
optimised an uninformative time costs nothing, since it receives zero effort;
on a fixed grid you pay for it regardless.

- **Criterion values are not comparable across `n_spt` settings.** A fixed
  grid rescales the FIM by `1/n_spt`, shifting a log-det criterion by exactly
  `n_mp*ln(n_spt)`. Compare designs, not criterion values.
- **Unknown keywords to `design_experiment()` raise.** Rejected by name:
  `optimize_sampling_times`, `fixed_sampling_grid`, `package`, `optimizer`.
  Note `design_experiment()` has never had a `verbose` parameter, though
  `initialize(verbose=...)` does.
- **Candidate sampling grids must be common to every candidate**; a ragged
  grid raises at `initialize()`. Prior experiments are exempt —
  `set_prior_experiments()` accepts ragged NaN-padded schedules.

The report states the case in force rather than printing a boolean, e.g.
`FIXED -- all 11 listed time(s) measured on every run; effort allocated per
experiment`.

---

## Dependencies

### Core (installed automatically)

| Package        | Purpose                                              |
|----------------|------------------------------------------------------|
| numpy          | Array operations                                     |
| scipy          | ODE integration, optimisation fallback               |
| pandas         | Estimability tables returned by `run_estimability()` |
| matplotlib     | Visualisation (`>=3.4`, including 3.11 — see CHANGELOG) |
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

[POUNCE](https://github.com/jkitchin/pounce) is a pure-Rust port of IPOPT
whose default build requires no Fortran, HSL or system BLAS. It speaks the
AMPL NL/SOL protocol, so Pyomo drives it the same way it drives IPOPT:

```bash
pip install pyomo-pounce
```

```python
import pyomo_pounce          # registers the 'pounce' solver
designer.design_experiment(designer.d_opt_criterion, solver="pounce")
```

POUNCE covers both solver call sites. `solver=` handles the design
formulation; for the IFT sensitivity path the collocation NLP is solved inside
your own `pyomo_model_fn`, so change the `SolverFactory` call there instead.
On `examples/ode/case_1.py` with both driven by POUNCE 0.9.0, the D-optimal
criterion matched IPOPT to 3.6e-15 relative on the same support. IPOPT
remains the reference configuration, being what the capability suite runs
against.

> For best IPOPT performance, configure it with the HSL linear solvers
> (`MA27`, `MA57`), which require a separate licence from
> [HSL](https://licences.stfc.ac.uk/product/coin-hsl). The open-source
> `MUMPS` solver works as a fallback and is what a stock IPOPT build
> uses. Scripts that request `linear_solver: ma57` will need that value
> changed to `mumps` if HSL is not installed.

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

### Sequential design (accounting for prior experiments)

Register what you already know before designing the next round — either a
FIM/covariance from an external estimation tool (Case A), or the raw
conditions of experiments already run, from which pydex computes the FIM
itself (Case B):

```python
# Case A: you already have a parameter covariance matrix
sigma_theta = np.diag([0.01, 500.0, 0.005, 300.0]) ** 2
fim_raw     = np.linalg.inv(sigma_theta)
designer.set_prior_fim(
    fim              = fim_raw * np.outer(theta_est, theta_est),  # pydex's normalisation
    model_parameters = theta_est,
)

# Case B: you have the raw conditions instead, pydex computes the FIM
designer.set_prior_experiments(
    ti_controls      = np.array([[55.0, 65.0, 1.0], [60.0, 70.0, 1.5]]),
    sampling_times   = np.array([[0.25, 0.5, 1.0], [0.25, 0.75, 1.0]]),
    model_parameters = theta_est,
    n_repeats        = np.array([2, 1]),   # optional; defaults to 1 each
)

designer.design_experiment(criterion=designer.d_opt_criterion, solver="ipopt")
# the prior FIM is added to the candidate FIM automatically, and is
# rescaled if designer.model_parameters is updated between rounds
```

`designer.clear_prior()` removes registered prior information.

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
    criterion      = designer.v_opt_criterion,
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57"},
)
```

See `examples/v_optimal/` for a complete worked example with a
three-reaction batch reactor system.

> Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
> V-optimal design of experiments for building fundamental models of
> pharmaceutical production processes. *Computers & Chemical Engineering*,
> 129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

---

## Examples

The `examples/` folder is organised by topic; `examples/README.md`
documents every subfolder, including `vle/`, `b_optimal/` and
`sequential/`, which are not repeated here.

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

### `examples/vle/` — steady-state VLE, and comparing criteria

A binary vapour-liquid equilibrium problem, static and nonlinear in its two
van Laar parameters.

- `van_laar_model.py` / `van_laar_design.py` — the model and a D-optimal
  design over 7 compositions x 3 temperatures
- `van_laar_criteria.py` — D- vs A- vs E-optimal on the same problem, then
  pseudo-Bayesian D-optimal in both aggregation types, with a figure of the
  resulting parameter confidence regions

`examples/README.md` documents both runners in full.

### `examples/first_order/` — the simplest possible design problem

First-order reaction `A -> B`, `dA/dt = -k*A`, so `A(t) = A0*exp(-k*t)`. The
decision variable is the sampling time and the parameters are `[k, A0]`. Each
script runs a local D-optimal design and two pseudo-Bayesian designs, the
first with uncertainty in `k` alone and the second in both parameters.

- `first_order_design.py` — exact IFT sensitivities via Pyomo collocation,
  and the minimal `build_pyomo_model()` a user has to supply for that path
- `first_order_design_no_ift.py` — finite-difference sensitivities over the
  analytic solution

This is the right place to start if you want to see the whole workflow with
nothing else going on: the analytic D-optimal answer for one parameter is
`t* = 1/k`, so you can check the design by hand. Note both scripts use 200
scenarios for their pseudo-Bayesian runs, which is generous for a
demonstration and slow on a single core; drop `N_scr` if you just want to see
the shape of the thing.

Both scripts are self-contained rather than split into a `*_model.py` and a
runner. They are also the model the capability suite exercises most heavily,
so if you change either, check sections 19-28, 30, 33, 41-44, 52 and 60.

### `examples/v_optimal/` — V-optimal MBDoE

Two-stage V-optimal design on the three-reaction batch reactor: Stage 1
finds the operating point that maximises yield subject to quality and
safety constraints; Stage 2 designs the V-optimal experiment and compares
it against A- and D-optimal designs to quantify the prediction-accuracy
benefit at that operating point. Unlike `examples/ode/`, each script is
self-contained rather than split into a `*_model.py` and a driver.

- `v_optimal_design.py` — exact IFT sensitivities via Pyomo collocation
- `v_optimal_design_no_ift.py` — finite-difference sensitivities

Guarded by capability-suite sections 59 and 60.

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

## Testing

There are two tiers.

**`tests/` — fast, solver-free regression tests.** These stub out pydex's
plotting and logging modules and import `designer.py` directly, so they
need no IPOPT and run in seconds. This is what CI runs on every push:

```bash
mkdir -p _flat && cp pydex/core/designer.py _flat/designer.py
PYTHONPATH="$PWD/_flat" pytest -q tests/
```

**`testing_scripts/` — end-to-end scripts that need solvers.** These
verify and demonstrate the package end-to-end and double as larger worked
examples. They are deliberately *not* run in CI: they need IPOPT, an MINLP
solver, and PyNumero's compiled ASL extension, and take far longer than a
CI budget allows. Run the capability suite locally before tagging a
release.

`testing_scripts/smoke_test_designer.py` sits between the two tiers: it
needs only IPOPT, runs in seconds, and checks Ds-optimality resolution by
name, the A-optimality singular-FIM fix, Ds succeeding where D-optimal
cannot, and the `regularize_fim` path.

- **`pydex_full_capability_test.py`** — the comprehensive capability
  suite: 60 sections and 301 assertions built on the three-reaction batch model
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

---

Do you have a question, suggestion, or feature request? Feel free to open
an issue on [GitHub](https://github.com/salvadorgarciamunoz/pydex/issues).
</file_text>