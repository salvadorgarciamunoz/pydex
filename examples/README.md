# pydex examples

Each example is a pair of files: a `*_model.py` holding the model (a
`simulate()` function and, where relevant, a `build_pyomo_model()`), and a
runner script that builds the `Designer`, sets the candidate grid, and
designs the experiment. Run the runner, not the model:

```bash
cd examples/ode
python case_2.py
```

Most examples solve an NLP, so they need IPOPT on `PATH`. See
`docs/source/installation.rst`.

## Naming scheme

The `case_N` families vary one axis at a time, so the suffixes tell you
which sensitivity path a script exercises:

| suffix | sensitivities | model solved by |
|---|---|---|
| *(none)* | exact, via the Implicit Function Theorem (`pyomo_model_fn` assigned) | Pyomo.DAE orthogonal collocation + IPOPT |
| `_no_ift` | finite differences | Pyomo.DAE orthogonal collocation + IPOPT |
| `_no_ift_no_collocation` | finite differences | Pyomo `Simulator` (scipy/vode) forward integration |

Comparing `case_1.py` against `case_1_no_ift.py` therefore isolates the
effect of the sensitivity method alone — same model, same grid, same
criterion. The three paths should agree on the design; the capability suite
asserts this (see §24 and §28).

## ODE examples — `examples/ode/`

### Case 1 — first-order reaction, one parameter

The smallest useful dynamic example: `dCA/dt = -k·CA`, one parameter, one
control, one response. Start here.

- `case_1.py` — IFT sensitivities via PyomoNLP.
- `case_1_no_ift.py` — finite differences over the collocation solve.
- `case_1_no_ift_no_collocation.py` — finite differences over forward
  integration. Also shows, in a commented block, what happens if
  `pyomo_model_fn` is assigned to a model built for the `Simulator` path.

### Case 2 — A→B with Arrhenius kinetics, four parameters

`dCA/dt = -k·CA^α`, `dCB/dt = ν·k·CA^α`, `k = exp(θ₀ + θ₁·(T-273.15)/T)`.
Four parameters `[θ₀, θ₁, α, ν]`, two controls `[CA0, T]`, two responses
`[CA, CB]` — the first example with multiple responses.

- `case_2.py` — D-optimal, IFT path.
- `case_2_no_ift.py`, `case_2_no_ift_no_collocation.py` — as above.
- `case_2_ds.py` — **Ds-optimal**: same model, grid and parameters as
  `case_2.py`, only the criterion changes. Worth reading even though this
  model's FIM is healthy: it shows designing for a *subset* of parameters
  while marginalising the rest, which is the usual reason to reach for Ds.

> **Note.** The collocation grid in this family previously admitted a
> sampling time a hair off an existing collocation node, producing a
> machine-epsilon finite element. IPOPT reported "Optimal Solution Found"
> while returning `CA = 31 mol/L` from `CA0 = 5`. If you adapt this example
> and see a physically impossible result reported as optimal, check that
> every sampling time lands exactly on a collocation node — refining `nfe`
> will not help.

### Case 3 — Michaelis–Menten-style network, nine parameters

Nine parameters, three controls `[cA0, T, τ]`, two responses. The largest
example, and the one where the sensitivity path matters most.

- `case_3.py` — scipy/finite-difference path. Spends roughly 350 s in
  sensitivity analysis (~45 model evaluations per candidate × 121
  candidates).
- `case_3_ift.py` — the same design via exact IFT sensitivities from the
  KKT conditions of the collocation NLP. This is the fast version, and the
  best illustration of why the IFT path exists.

### Case 4 / Case 5 — local vs pseudo-Bayesian D-optimal, A→B→C network

A different reaction network from case_1/2/3: A→B→C with one control each,
introducing **pseudo-Bayesian design** — designing under uncertainty in the
model parameters rather than at a single nominal guess. Neither carries a
`_no_ift`/`_no_ift_no_collocation` suffix even though both use the
finite-difference-over-forward-integration path (no `pyomo_model_fn`); the
suffix scheme above is specific to the case_1/2/3 sensitivity-path
comparison, and there's no IFT/collocation counterpart for this family.

- `case_4.py` — **local** D-optimal design: two rate constants `[k1, k2]`
  at a single nominal guess, one control (feed rate `f_in`). Note: `f_in`
  feeds pure A with no outflow term, so this is a continuously-fed
  semi-batch reactor, not a closed system — total moles grow as `1 +
  f_in·t` once `f_in > 0` (verified numerically, not just asserted).
  `pseudo_bayesian_type` only takes effect when `model_parameters` is 2-D,
  so passing it here (a 1-D nominal vector) has no effect — the design
  stays local. Also demonstrates recovering `[k1, k2]` from data simulated
  at the apportioned design, via **PyMC** (`pip install pymc arviz`) — see
  "Bayesian inference" below.
- `case_5.py` — the same network with Arrhenius kinetics (`f_in` fixed to
  0, so this one IS closed) and **genuine pseudo-Bayesian Type-1**
  D-optimal design: `model_parameters` is a scenario array drawn from a
  uniform prior over all four kinetic parameters, and the design optimises
  the criterion averaged over scenarios. Ships with `N_SCR = 20` rather
  than a much larger ensemble — measured at ~13 s/scenario on this grid, so
  this is a runtime/precision trade you can dial by changing one constant.
  `save_atomics` must be passed to `design_experiment()` as a keyword, not
  set as `designer._save_atomics` beforehand — the keyword's own default
  silently overwrites the attribute either way, the same as
  `regularize_fim` (see PROJECT_NOTES.md).

Both models fold their reaction rates directly into the mass balance
rather than defining them as a separate algebraic constraint, which keeps
each system a pure ODE — required by scipy's `Simulator` backend, which
only integrates ODEs, not DAEs.

### Bayesian inference in `case_4.py` — via PyMC, not pydex

`case_4.py` ends by recovering `[k1, k2]` from data simulated at the
apportioned design, using **PyMC** (`pip install pymc arviz` — not a pydex
dependency, only needed for this section): synthetic "observed" data is
simulated at the apportioned design's condition(s) using the same nominal
parameters the design was built on, plus measurement noise drawn from
`error_cov`, and PyMC samples the posterior over `[k1, k2]` given that data.

This is ordinary downstream analysis written against the public API
(`designer.optimal_candidates`, `designer.apportionments`, `error_cov`,
`simulate()`) — pydex has no Bayesian-inference capability of its own.
`simulate()` runs through scipy's Simulator, a black box with no symbolic
gradient, so it's wrapped with `pytensor.wrap_py` and sampled with
`pm.Metropolis` rather than PyMC's default (gradient-based) NUTS sampler. A
2-parameter local D-optimal design with an unconstrained budget typically
collapses to a single condition run several times rather than spread
across several; 4 chains × (500 tune + 800 draws) in parallel takes a few
minutes and should recover `k1`/`k2` close to their true values, with
r_hat typically landing around 1.01–1.04 — inside PyMC's own "probably
fine, more draws would help" range rather than fully converged. The
draw/tune/chain/core counts are constants at the top of that section if you
want tighter diagnostics at the cost of runtime.

## ASL elimination — `examples/ASL Elimination/`

- `asl_elimination_demo.py` — demonstrates the diagnostic in
  `pydex.utils.diagnose_asl_elimination`, which checks that every parameter
  `Var` survives into the ASL primal vector by name. This is the
  precondition pydex's IFT column-matching relies on; `initialize()` runs
  the same check automatically when the utility is importable.
- `pydex_ift_asl_guide.docx` — background on the IFT/ASL interaction.

## Jupyter — `examples/jupyter/`

- `pydex_quickstart.ipynb` — narrated walkthrough of a first design.
- `pydex_ode_model.ipynb` — the same for a dynamic model.

## Publication code — `publications/`

Scripts reproducing figures and results from the papers behind pydex. These
are archival: they are kept as published rather than updated with the API,
so treat them as a record rather than as current usage examples.
