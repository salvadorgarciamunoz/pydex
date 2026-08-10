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
