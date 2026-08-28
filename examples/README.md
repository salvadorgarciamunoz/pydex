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

## Labelling — name your parameters, controls and responses

Every example sets these, and yours should too. They are optional to the
mathematics and they change what you can read:

| attribute | what it labels |
|---|---|
| `model_parameter_names` | reports, plot axes, the estimability ranking — and `interest_parameters` is matched against it BY NAME, so Ds-optimal needs it |
| `ti_controls_names` | design-table column headings and effort-plot axes |
| `response_names` | predicted-response and sensitivity plot axes |
| `candidate_names` | plot titles and the candidate column; useful when candidates are named lots or formulations rather than grid points |
| `tv_controls_names` | time-varying control plots (no example here uses time-varying controls) |
| `model_parameter_unit_names`, `response_unit_names`, `time_unit_name` | units appended to axis labels |

Anything you leave unset is filled with a generated default —
`Time-invariant Control 0`, `Model Parameter 0`, `Candidate 0` — which is why
an unlabelled run still prints something sensible.

Note the SINGULAR `parameter` in `model_parameter_names`. The plural
`model_parameters_names`, and `measurable_responses_names`, were earlier names;
assigning to either raises an error naming the attribute to use instead.

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

## Sampling times — `n_spt` is the only control

Every dynamic example makes a choice here, so it is worth reading once. A
design spends a fixed budget of effort; `n_spt` decides what one unit of that
effort buys.

| what you want | how to ask | effort allocated per |
|---|---|---|
| choose the conditions **and** which times to measure | omit `n_spt` (the default) | measurement: one (candidate, time) cell |
| exactly `k` samples per run, optimiser picks which `k` | `n_spt=k` | run of `k` samples: one (candidate, schedule) |
| measure every listed time on every run | `n_spt=<number of listed times>` | one whole experiment |

The first and third are genuinely different design problems. With sampling
times optimised, an uninformative time costs nothing because it receives zero
effort; on a fixed grid you pay for it regardless.

There is **no flag** that switches sampling-time optimisation on or off.
`optimize_sampling_times` and `fixed_sampling_grid` no longer exist and raise
if passed, as does any other unrecognised keyword to `design_experiment()` —
including `verbose`, which has never been a parameter of that method
(`initialize(verbose=...)` is a different method and is fine).

Two consequences worth knowing before comparing runs:

- **Criterion values are not comparable across `n_spt` settings.** A fixed
  grid rescales the FIM by `1/n_spt`, which shifts a log-det criterion by
  exactly `n_mp*ln(n_spt)`. Compare designs, not criterion values. The same
  trap applies to changing `error_cov` scaling.
- **Candidate sampling grids must be common to every candidate.** A ragged
  grid raises at `initialize()`. Use a union grid with sampling times
  optimised — an uninformative time receives zero effort. (Prior experiments
  are exempt: `set_prior_experiments()` accepts ragged NaN-padded schedules,
  since experiments already run have whatever schedules they had.)

The report states which case is in force rather than printing a boolean, e.g.
`FIXED -- all 11 listed time(s) measured on every run; effort allocated per
experiment`.

## Steady-state VLE — `examples/vle/`

A binary vapour–liquid equilibrium problem: estimating the two van Laar
activity-coefficient parameters from total-pressure and vapour-composition
measurements. **Static and nonlinear in the parameters** — the step between a
linear response surface and a dynamic model.

- `van_laar_model.py` — the model. Two-parameter van Laar activity
  coefficients, modified Raoult's law, and Antoine constants that are known
  rather than fitted. Run it directly for a sanity check of the predictions.
- `van_laar_design.py` — the runner. D-optimal design over 7 compositions × 3
  temperatures.

No sampling times: this model has no time axis, so `n_spt` does not apply.

The design is **two support points for two parameters**, both at the highest
temperature, at `x1 = 0.05` and `0.35` — *not* at the corners of the
composition range, which is where a linear model would put them. Criterion
`16.782682`; `apportion(8)` gives 4 runs and 4, and the rounded design is
99.88% as informative as the continuous one.

Because the model is nonlinear, **the design depends on the nominal
parameters**: at `theta = [1.10, 1.45]` the support moves to candidates 6 and
12. That is the motivation for sequential design, not a defect.

This example also demonstrates the labelling attributes
(`model_parameter_names`, `ti_controls_names`, `response_names` and the unit
names), which turn `Time-invariant Control 0` into `x1` in the design table and
label every plot axis. Note the singular `parameter` — the plural form is
refused with an error naming the correct attribute.

Needs an NLP solver. `solver="pounce"` works with nothing beyond `pip`; IPOPT
works equally well.

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

> **Note.** The collocation grid in this family is sensitive to sampling
> times that sit a hair off an existing collocation node: embedding one
> produces a machine-epsilon finite element, and IPOPT then reports "Optimal
> Solution Found" while returning `CA = 31 mol/L` from `CA0 = 5`. The model
> files snap such times to the nearest node and warn when they do. If you
> adapt this example and see a physically impossible result reported as
> optimal, check that every sampling time lands exactly on a collocation
> node — refining `nfe` will not help.

### Case 3 — Michaelis–Menten-style network, nine parameters

Nine parameters, three controls `[cA0, T, τ]`, two responses. The largest
example, and the one where the sensitivity path matters most.

- `case_3.py` — scipy/finite-difference path. Spends roughly 350 s in
  sensitivity analysis (~45 model evaluations per candidate × 121
  candidates).
- `case_3_ift.py` — the same design via exact IFT sensitivities from the
  KKT conditions of the collocation NLP. This is the fast version, and the
  best illustration of why the IFT path exists.

Both run `run_estimability()` before designing, and both act on the result:
the nine-parameter form is structurally singular — the rate law has an exact
invariance, since adding a constant to every `theta_i0` at once (or to every
`theta_i1`) leaves every prediction identical — so `design_experiment()`
refuses it. Each script fixes the two parameters its own analysis flags,
re-runs estimability on the reduced seven, and then designs.

`case_3.py` makes the complementary point: **estimability analysis needs no
tractable model.** `run_estimability()` reads the sensitivity matrix and does
not care that these sensitivities are finite differences over an opaque
integrator rather than exact derivatives — `simulate()` could be a legacy
binary or a commercial simulator. Two things follow from the lower accuracy.
The UNRESOLVABLE threshold is inferred from the sensitivity method (`1e-3` for
finite differences against `1e-7` for IFT), and the two parameters flagged are
*different*: `case_3_ift.py` fixes `theta_20` and `theta_21`, `case_3.py` fixes
`theta_21` and `theta_31`. Both are correct — which member of a triple you hold
still is a convention, and finite differences break the tie differently.

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

## Bracketing-optimal design — `examples/b_optimal/`

Worked scenarios for `b_opt_criterion`, which implements the
bracketing-optimal design of Chen, Paulavičius, Adjiman & García-Muñoz
(2018), *AIChE J.* 64(11):3944–3957, doi:10.1002/aic.16214. This criterion
answers a different question from the rest of pydex: **not** "which
experiments best determine my parameters" but "which experiments best
bracket my operating space" — the regulator's question in a pharmaceutical
bracketing study. It is not sensitivity-based and has nothing to do with
the FIM.

Two objectives are combined by weighted sum, controlled by `output_weight`:

- **input-space bracketing** (`output_weight=0`) — D-optimality applied to
  the scaled input-factor values themselves, giving an orthogonal,
  corner-seeking design in the process inputs;
- **output-space coverage** (`output_weight=1`) — maximise the volume
  spanned by the candidates' predicted responses, so the design maps the
  output space rather than a sliver of it.

The problem is posed as binary subset selection over a pre-evaluated
candidate pool, so it needs an MINLP solver (`solver="bonmin"`) and an exact
design size (`n_exp`).

**Two usage requirements, both of which raise rather than fail quietly:**

- **`simulate_candidates()` must be called first.** b_opt is the only
  criterion that reads `designer.response`, so forgetting this is a new
  mistake; it raises `RuntimeError`.
- **`n_exp >= max(phi, n_resp + 2)`**, where `phi` is the number of input
  factors. Both Cholesky lifts floor their diagonal at `1e-8`, so a
  rank-deficient matrix cannot be represented and the program becomes
  strictly infeasible — and an MINLP solver does not report infeasibility
  quickly, it appears to hang. Hence the up-front `ValueError`.

  The output term is the subtle one. Its covariance is *centered*, so its
  rank is at most `n_exp - 1`, which makes the algebraic bound
  `n_exp >= n_resp + 1`. That is not sufficient in practice: at exactly that
  value the covariance is full rank with no margin above the Cholesky floor,
  and the solver reports infeasible whenever the output term carries weight.
  The implemented bound is therefore `n_resp + 2`, which is why
  `scenario_1`'s design-size sweep starts at 4 rather than 3.

- **A failed solve raises rather than returning a design.** If the MINLP
  comes back `infeasible`, `unbounded` or `error`, `design_experiment()`
  raises `RuntimeError`; the returned selection is also validated against
  `n_exp` independently of what the solver reported. Time- or
  iteration-limited solves still return their incumbent, but warn that it is
  not a proven optimum, and record `designer._b_opt_termination` and
  `designer._b_opt_proven_optimal` so a caller can tell the two apart.

Scenarios:

- `scenario_1_film_coating.py` + `film_coating_model.py` — the paper's
  motivating example, a tablet film coater: 3 inputs (inlet air
  temperature, coating-solution flow, air flow) to 2 outputs (exhaust
  temperature, exhaust %RH). Reproduces the paper's Table 1 and Figures
  2–5, including the Pareto front swept over `output_weight` and the
  Pareto-front family as design size grows. The coater model is a
  physically grounded thermodynamic model, **not** the paper's own
  Supporting-Information equations, so trends and figure structure match
  while absolute numbers do not.
- `scenario_2_cstr.py` + `cstr_model.py` — the paper's second case study,
  two CSTRs in series: 6 inputs to 3 outputs, reproducing Figures 8 and 9.
  The kinetic and energy model is transcribed from the authors' own GAMS
  source, so this one does carry the paper's actual numbers.
- `scenario_3_suzuki.py` + `suzuki_model.py` — a Suzuki–Miyaura coupling,
  **not** from the paper: 5 inputs to 3 critical quality attributes, with
  real process constraints. This is the practitioner-oriented walkthrough
  and the one that verifies the machinery: Part A cross-checks the design
  against **exhaustive enumeration** of all 658,008 five-point subsets and
  reports the gap; Part B covers choosing a design size; Part C exercises
  the anti-clustering control; Part D exercises the guards. The kinetics
  are representative of a realistic system rather than fitted to a
  substrate.

### Running these

Each script can be run from any working directory — the model modules are
imported relative to the script's own location:

```
python examples/b_optimal/scenario_1_film_coating.py
```

**Figures are written to the working directory** (`OUT_DIR = "."`), so you
choose where they land by choosing where you run from. Deliberately not
pinned to the script directory, which would drop generated PNGs inside the
repository — the removed `publications/` folder carried 59 MB of exactly that,
and it is not worth repeating. Don't commit the output.

`scenario_1` runs in a couple of minutes and `scenario_2` in about three.
`scenario_3` is the slow one: its Part A exhaustive enumeration alone is
several minutes, and Parts B/C cap each solve with
`solver_options={"bonmin.time_limit": 90}`. **A capped solve returns
bonmin's best incumbent, not a proven optimum**, so treat individual rows in
those tables as indicative — the pure-input instances (`output_weight=0`)
are by far the hardest and are the ones that hit the cap. Part A runs
uncapped, which is why it is the one place global optimality is claimed, and
it is checked by enumeration rather than asserted.

Note also that bonmin gives no global-optimality guarantee for nonconvex
MINLP. Its record on these instances is perfect — four independent
brute-force cross-checks agreeing exactly — but that is empirical evidence
for these problems, not a proof.


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

## Publication code

The scripts reproducing figures from the two 2022 papers behind pydex
(*Comput. Chem. Eng.* 159:107680 and *React. Chem. Eng.* 7(11):2359–2374) are
**not** kept in this fork. They target the pre-0.2.0 cvxpy/MOSEK API, which no
longer exists here, so they cannot run against current pydex. The code archive
cited by those papers is <https://github.com/omega-icl/pydex>.
