## Fork Notice

This is a fork of [pydex](https://github.com/KennedyPutraKusumo/pydex)
by Kennedy Putra Kusumo et al., originally described in:

> Kusumo, K.P., Kuriyan, K., Vaidyanathan, R., Shadbahr, T., Khan, F.I.,
> Harun, N., Sin, G. & Braatz, R.D. (2022). Explicit-Constraint-Based
> Optimal Experiment Design for Differential Equation Models.
> *Computers & Chemical Engineering*, 107940.

### Changes in this fork

- **V-optimal MBDoE**: two-stage workflow targeting prediction accuracy
  at a user-specified operating condition (`find_optimal_operating_point()`
  \+ `design_v_optimal()`)
- **Process optimisation**: constrained nonlinear optimisation via IPOPT
  to find the optimal operating point (Stage 1) before designing experiments
- **IPOPT solver support**: all design criteria (D/A/E-optimal, CVaR,
  V-optimal) can now use IPOPT via `cyipopt` (`package="ipopt"`)
- **Improved documentation**: Sphinx/NumPy-compliant docstrings throughout
  `designer.py`, including full usage guides for the simulate function,
  control variable setup, and the V-optimal two-stage workflow

---

# Python Design of Experiments

An open-source Python package for optimal experiment design, essential to
a modeller's toolbelt. If you develop a model of any kind, you will relate
to the challenges of estimating its parameters. This tool helps design
maximally informative experiments for collecting data to calibrate your
model.

## Installation

Install directly from this fork:

```bash
pip install git+https://github.com/salvadorgarciamunoz/pydex.git
```

Or install the original release from PyPI:

```bash
pip install pydex
```

## Quick Start

- [Demo](https://github.com/KennedyPutraKusumo/pydex/blob/master/examples/pydex_quickstart.ipynb)
  of basic features.
- [Example](https://github.com/KennedyPutraKusumo/pydex/blob/master/examples/pydex_ode_model.ipynb)
  of experimental design for ODE models.
- [V-optimal example](examples/v_optimal_test_case.py): two-stage
  prediction-oriented MBDoE for a batch reactor with competing reactions.

## Features

1. Simple, intuitive syntax — easy to get started, powerful enough for
   complex problems.
2. Continuous and exact (discrete) experimental designs.
3. Design criteria: D-optimal, A-optimal, E-optimal, V-optimal, CVaR
   variants.
4. Interfaces to scipy, cvxpy, and IPOPT (via cyipopt) for optimisation.
5. Convenient built-in visualisation via matplotlib.
6. Supports virtually any model written as a Python function, including
   ODE models solved via scipy or Pyomo.DAE.

## Dependencies

### Core (installed automatically)

| Package        | Purpose                                              |
|----------------|------------------------------------------------------|
| numpy          | Array operations                                     |
| scipy          | ODE integration, optimisation interface              |
| matplotlib     | Visualisation                                        |
| numdifftools   | Numerical finite-difference sensitivities            |
| cvxpy          | Convex optimisation interface (D/A/E-optimal)        |
| dill           | Saving objects with weak references                  |
| corner         | Corner plots for high-dimensional parameter spaces   |
| emcee          | Bayesian MCMC inference                              |

### Optional: IPOPT support

IPOPT enables faster, more robust optimisation for large or ill-conditioned
designs, and is required for V-optimal and process optimisation (Stage 1).

```bash
pip install cyipopt
```

> **Note:** `cyipopt` requires a working IPOPT installation with a
> compatible linear solver. The open-source solver `MUMPS` is bundled
> with most binary distributions. For best performance, HSL solvers
> (`MA27`, `MA57`) are recommended — see
> `docs/ipopt_setup_guide.docx` for installation instructions.

## V-optimal MBDoE

V-optimal design minimises model prediction variance at a specific
operating condition `dw` (e.g. the economically optimal process point),
rather than minimising global parameter uncertainty as D/A-optimal designs
do. It follows a two-stage workflow:

**Stage 1 — Process optimisation:** find `dw` by solving a constrained
nonlinear programme over the operating space via IPOPT.

**Stage 2 — V-optimal MBDoE:** design experiments that minimise
`J_V = trace(W FIM⁻¹ Wᵀ)` where `W` is the scaled sensitivity matrix
evaluated at `dw`.

```python
# Stage 1
designer.process_objective   = my_objective    # callable(tic, tvc, mp) -> float
designer.process_constraints = my_constraints  # callable(tic, tvc, mp) -> list
designer.dw_sense            = "maximize"
designer.dw_bounds_tic       = [(lb, ub), ...]

dw_tic, dw_tvc = designer.find_optimal_operating_point(
    init_guess = np.array([[60.0, 70.0, 1.0]]),
    optimizer  = "mumps",
)

# Stage 2
designer.dw_spt = np.array([t_final])
designer.design_v_optimal(package="ipopt", optimizer="mumps",
                           optimize_sampling_times=True)
```

See `examples/v_optimal_test_case.py` for a complete worked example with
a batch reactor system featuring three competing reactions.

> Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
> V-optimal design of experiments for building fundamental models of
> pharmaceutical production processes. *Computers & Chemical Engineering*,
> 129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

## Examples

The `examples/` folder contains worked examples covering a range of model
types and design criteria:

- `examples/v_optimal_test_case.py` — V-optimal MBDoE, batch reactor,
  three competing reactions, two-stage workflow
- `examples/ipopt/` — D/A/E-optimal designs using IPOPT
- Jupyter notebooks (original): scipy and Pyomo ODE model examples

Do you have a question, suggestion, or feature request? Feel free to open
an issue or contact the original author at
[kennedy.putra.kusumo@gmail.com](mailto:kennedy.putra.kusumo@gmail.com).
