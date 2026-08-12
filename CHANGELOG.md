# Changelog

All notable changes to this fork are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-11

### Fixed

- **The finite-difference step was a flat constant applied to every model
  parameter regardless of its magnitude, at three separate call sites.**
  numdifftools builds its step sequence as
  `step_nom * base_step * step_ratio ** i`, where the default `step_nom`
  heuristic is `max(log(e + |x|), 1)`. That heuristic floors at 1, so it
  scales the step *up* for parameters larger than O(1) and does nothing at
  all for parameters smaller than O(1). With a flat `base_step = 2`, a
  parameter of nominal value 0.02 therefore received an initial
  perturbation of ~2.0 — a hundred times its own value. Richardson
  extrapolation then worked downward from a starting point far outside the
  local linear regime it assumes, fitting curvature and saturation rather
  than a derivative.

  The failure was silent — no warning, no exception, no convergence
  complaint — and selective in a way that disguised it. Parameters with
  nominal magnitude near or above 1 (Michaelis constants, concentrations)
  were unaffected; small-magnitude kinetic rate constants were badly wrong,
  and the error grew with elapsed time along a trajectory. On a minimal
  two-ODE reproduction this produced a 65% disagreement between the
  finite-difference and IFT sensitivity paths at the longest sampling time.
  The IFT path was correct throughout; the finite-difference step was the
  defect. Confirmed against an independent `scipy.integrate` Radau solution
  at `rtol=1e-13`, converged across five step sizes: IFT matched ground
  truth to 4–5 significant figures, finite differences did not.

  Step resolution is now a single module-level helper,
  `_resolve_fd_base_step()`, returning a per-parameter array:
  `max(relative_base_step * abs(theta_i), absolute_step_floor)` with
  defaults `1e-2` and `1e-8`. The floor exists because a pure percentage
  gives a parameter of nominal value exactly 0 a step of exactly 0.
  Magnitude is used rather than signed value, so a negative parameter does
  not get a sign-flipped perturbation. For a pseudo-Bayesian scenario set
  (shape `(n_scr, n_mp)`) the magnitude is the per-parameter maximum across
  scenarios, so no scenario silently under-steps relative to the others.

  All three finite-difference sites now call it:

  1. **`eval_sensitivities()`** — `base_step` now defaults to `None`,
     meaning "resolve per parameter". Passing an explicit `base_step`
     (scalar or array) reproduces the previous unconditional behaviour
     exactly and bypasses the scaling; `relative_base_step` and
     `absolute_step_floor` are exposed as arguments.
  2. **`set_prior_experiments()`** — was flat `2`. This one compounds: the
     corrupted sensitivities propagate into the prior FIM that every
     subsequent sequential design builds on.
  3. **`_eval_W_matrix()`** (V-optimal weighting matrix) — was flat `2`,
     while its own docstring claimed "identical step generator settings" to
     `eval_sensitivities()`. That claim was false and is corrected; this
     path uses the defaults and does not expose the arguments.

  The clearest evidence is capability-suite §53, which compares the
  finite-difference sensitivities against a **closed-form analytic**
  derivative rather than against another numerical path:

  | | FD vs analytic |
  |---|---|
  | before | `7.343e-04` |
  | after  | `5.258e-13` |

  Nine orders of magnitude, on a case where the correct answer is known
  exactly. §52's IFT-vs-FD cross-check also tightened across all nine
  criteria plus both Ds subsets (`d_opt` `1e-4 -> 0`, `e_opt` `1e-4 -> 0`,
  `eg_opt` `2e-4 -> 0`, `ds_opt` one-parameter subset `3e-4 -> 0`): the two
  independent paths now agree more closely than before, with FD having
  moved toward IFT.

  Fixing sites 2 and 3 moved exactly three values in the suite and nothing
  else — sequential D-optimal `32.89070549 -> 32.89095883`, and the
  V-optimal criterion `9.993E-04 -> 9.981E-04` / `-0.00100701 ->
  -0.00100586`. §52 and §53 were byte-identical across that run, confirming
  the refactor of `eval_sensitivities()` onto the shared helper was
  behaviour-preserving. The shifts are small because the suite's models use
  well-scaled parameters near O(1), which is where a flat step of 2 happens
  to be roughly right; models with small rate constants are where these
  paths were badly wrong, and the suite contains none.

  **Sites 2 and 3 remain untested at their defaults.** They survived the
  first pass of this fix precisely because the capability suite passes
  explicit `base_step` overrides in the sections that reach them
  (`base_step=0.01` and `base_step=1e-4`), masking the default. Anyone
  adding coverage here should omit the override.

  A side effect worth recording: §07 (pseudo-Bayesian type 0) got roughly
  12–14% faster (sensitivity analysis `220.39 -> 189.60` CPU seconds
  sequential, `220.31 -> 194.66` parallel) with its criterion unchanged to
  four decimal places. Perturbed models are now a small displacement from
  the nominal solution rather than a hundredfold one, so IPOPT converges in
  fewer iterations. Not the goal of the change, but consistent with it.

  No assertion required re-baselining — every pinned tolerance in the suite
  was loose enough to absorb the improvement.

  Verbose output at level >= 2 now prints the resolved per-parameter step
  under `FD base_step (per parameter)`. The step being invisible is a large
  part of why this survived as long as it did.

- **`_PYNUMERO_ASL_AVAILABLE` was treated as a runtime guarantee when it is
  only an import-time one.** The flag is set by a `try: import ... except:`
  around `PyomoNLP`. That import tests whether the *Python* class is
  importable; PyNumero's Python interface ships with Pyomo and always is.
  The ASL Jacobian machinery it wraps depends on a separately-built
  compiled extension that can be missing or broken while the import above
  still succeeds — so the flag can read `True` on a machine where
  `PyomoNLP(m)` raises the moment it is called.

  Both call sites in `_eval_sensitivities_pyomo_ift()` — the main build and
  the causal per-sampling-time rebuild — branched on that flag and wrapped
  the call in `try/finally`, not `try/except`. The `finally` re-fixed the
  parameter Vars correctly, but any exception from `PyomoNLP` propagated
  uncaught. Because the branch had already been chosen by the frozen flag,
  the pure-Python `differentiate()` fallback sitting in the `else` arm was
  unreachable in exactly the situation it exists for: the run died instead
  of falling back.

  Both sites now catch the failure, emit a `RuntimeWarning` naming the
  original exception, downgrade `_PYNUMERO_ASL_AVAILABLE` so the remainder
  of the process goes straight to the fallback rather than re-attempting a
  backend already shown not to work, and continue on the pure-Python
  Jacobian. The parameter-refixing `finally` is unchanged. The fallback
  loop itself, previously duplicated verbatim at both sites, is now a
  single module-level `_pyomo_ift_fd_jacobian()` helper.

  **Verified by inspection only.** This branch does not execute on a
  machine with a working ASL extension, which includes every machine the
  capability suite has been run on, so the suite does not cover it. It is
  deliberately minimal for that reason.

## [0.2.1] - 2026-08-10

### Removed

- **`pydex/core/bnb/`** — the pre-Pyomo, cvxpy-era branch-and-bound
  implementation. It was unreachable (referenced by no module, its own
  `__init__.py` was empty, and `tree.py` did not even import `node.py`), but it
  was still shipping in the wheel, and `node.py` opened with `import cvxpy` —
  a dependency this fork removed. So `import pydex.core.bnb` on a fresh install
  raised `ModuleNotFoundError: No module named 'cvxpy'`: the same class of
  defect as the undeclared `pandas`, hiding in a module nothing imports. It also
  contained five `is` comparisons against string literals, which have raised
  `SyntaxWarning` since Python 3.8 — evidence it had not been executed in years.
  Sparse designs are served by the Pyomo MINLP path (`min_effort` with bonmin).

  Nothing could have depended on this: the module could not be imported at all
  on a clean install, which is why removing it is a patch release rather than a
  breaking change.

### Added

- **CI byte-compiles `examples/` and `testing_scripts/`.** Neither tree is
  installed (`packages.find` is scoped to `pydex*`), collected by pytest, or run
  by any CI job, so a plain syntax error in an example previously shipped
  unnoticed. The new `syntax` job runs `compileall` on py3.9 as well as py3.12:
  the floor is the version that matters, because compiling on 3.12 does **not**
  catch PEP 701 f-string quote reuse — 3.12 is where that became legal. It also
  uses `-W error`, promoting `SyntaxWarning` to a failure, which catches `is`
  against string literals and invalid escape sequences. The now-deleted `bnb`
  was the only thing blocking that stricter setting.

## [0.2.0] - 2026-08-10

First tagged release of this fork. Everything below is relative to upstream
[KennedyPutraKusumo/pydex](https://github.com/KennedyPutraKusumo/pydex), which
this diverged from substantially: the solver stack moved from cvxpy to Pyomo,
exact IFT sensitivities and V-optimal design were added, and twelve defects in
`designer.py` were fixed.

### Added

- **`Designer.run_estimability(tol=None, corr_tol=0.95, plot=True, report=True)`**
  — ranks model parameters from most to least estimable using the Yao et al.
  (2003) orthogonalisation as set out in Table 1 of Wu, McLean, Harris &
  McAuley (2011), implemented as pivoted QR. Verified against a literal
  Table-1 implementation: identical ordering, residual norms agreeing to
  machine precision. Returns `table` and `correlation` as pandas DataFrames
  plus correlation groups and raw arrays, and draws four figures. Reports
  three distinct quantities: `abs_info` (absolute, dimensionless — below 1
  the candidate grid cannot determine the parameter to within its own
  magnitude), `E`/`E_UD` (relative ranking for weighted/unweighted least
  squares respectively) and `group` (mutually correlated, hence
  interchangeable, parameters). Advisory only — no other code path reads it.
- **Structural-singularity gate on `design_experiment()`**, controlled by
  `allow_singular_fim` (default `False`). Refuses a design whose FIM is
  rank-deficient for *every* admissible design and names the parameters
  responsible, rather than returning a plausible-looking number from a
  Cholesky factor with a floored diagonal. Ds-optimality is exempt, since
  marginalising a singular nuisance block is precisely its purpose.
- **`Designer.diagnose_fim_structure()`** — reports rank, condition number
  and the eigenvector composition of null directions of the
  fully-supported FIM, and names the implicated parameters.
- **V-optimal MBDoE**: two-stage workflow targeting prediction accuracy at a
  user-specified operating condition (`find_optimal_operating_point()` then
  `design_v_optimal()`).
- **Pyomo.DAE support** — a DAE model can serve as both simulator and IFT
  sensitivity source; signature-2 multi-output models supported.
- **Parallel IFT sensitivity evaluation** via `joblib`, parallelised over
  candidates (local designs) or scenarios (pseudo-Bayesian designs).
- **Sphinx documentation** under `docs/`, built from the in-source
  docstrings via `autodoc` + `napoleon`.
- **`CHANGELOG.md`** (this file) and an examples index at
  `examples/README.md`.

### Changed

- **cvxpy removed.** The OED problem is now formulated and solved entirely
  through Pyomo, giving access to any solver Pyomo knows about (IPOPT, GLPK,
  Gurobi, CPLEX, Bonmin, SHOT, GAMS/BARON, …) via
  `design_experiment(solver=..., solver_options={...})`. IPOPT is the default.
- **`pandas` is now a hard, module-level dependency** of `designer.py`
  (`run_estimability` returns DataFrames) and is declared in the packaging
  metadata — see *Fixed* below.
- **Documentation restructured around what a user is choosing between.** The
  class docstring gave V-optimal and Ds-optimal a full prose section each while
  the other nine criteria appeared only as one-line entries in a monospaced
  block, which made the rendered page read as though those two were the
  supported options. There is now a `Design types` section with a subsection
  per family — D, A, E, Ds, V, the prediction-variance family, pseudo-Bayesian,
  CVaR-D and U — each explaining what it optimises, when to choose it, and what
  it fails at.
- **New `Tools and helpers` section** covering estimability analysis, the FIM
  and sensitivity diagnostics, ASL elimination, and apportionment. The
  estimability guidance in particular was reachable only by opening
  `run_estimability`'s own reference entry, despite being the first thing worth
  running on a new model; it is now a section of its own that explains how to
  read `abs_info` against `E` against the correlation groups.
- **Examples page added to the documentation** (`docs/source/examples.rst`) —
  the naming scheme as a table, a subsection per case family, the ASL demo, and
  the test scripts as worked examples. Notebooks are linked to GitHub rather
  than rendered inline, which avoids adding a notebook-execution extension to
  the docs toolchain.
- **POUNCE documented as an alternative solver** in both the README and the
  installation page: a pure-Rust port of IPOPT whose default build needs no
  Fortran, HSL or system BLAS, registered with Pyomo's `SolverFactory` by
  `pip install pyomo-pounce` plus `import pyomo_pounce`. Verified on
  `examples/ode/case_1.py` driving both the design formulation and the IFT
  collocation solve: D-optimal criterion agreeing with IPOPT to 3.6e-15
  relative on the same support, sensitivities still via PyomoNLP/ASL. IPOPT
  remains the reference configuration, being what the capability suite runs
  against.
- **Removed the class-level `References` section**, which cited a single paper
  and sat below the numerical-controls tables. Method-level citations, such as
  the Wu et al. (2011) reference in `run_estimability`, are kept where they
  apply.

- All 69 previously-undocumented public callables are now documented,
  Google-style. 14 pre-existing NumPy-style docstrings are left as-is and
  handled by `napoleon_numpy_docstring`.

### Fixed

Twelve defects in `designer.py`, each guarded by a section of
`testing_scripts/pydex_full_capability_test.py` (section numbers in
parentheses):

1. **A-optimality scored `0` — its *best* attainable value — for a singular
   or indefinite FIM**, so a structurally broken design ranked as perfect.
   Now returns `+inf`. (§38)
2. **`_pb_scenario_worker` passed the wrong sampling times** and hard-coded
   `_dynamic_system=True`. (§22, §27, §41)
3. **`dg`/`di` returned noise or `+inf`**; `0*exp(inf) = nan` silently
   dropped 78 of 135 blocks. (§29, §40)
4. **IFT derived response names positionally**, producing duplicated
   response rows. Names are now matched by name. (§42, §44)
5. **IFT snapped absolute sampling times against a normalised grid.** (§44)
6. **`apportion` branch inverted** — allocated 4 of 12 requested
   experiments. (§47)
7. **`_greatest_effort_apportionment` mutated the caller's effort array.**
   (§47)
8. **`_pb_d`/`_pb_e` fell through to a silent `None`.** (§38)
9. **`tight_layout` warnings on 3-D axes** across 16 call sites. (full-suite
   output scan)
10. **`_eval_W_matrix` crashed on static single-response models**
    (numdifftools cannot take a Jacobian of a length-1 output). (manual)
11. **`_swap_candidates`/`_revert_candidates` were never wired together**,
    making the `vdi` criterion unreachable. (§51)
12. **`_revert_candidates` restored `tv_controls` *and* `sampling_times`
    from `old_tic_cands`.** (§51)

Also fixed, outside `designer.py`:

- **`pandas` was undeclared in the packaging metadata**, so a fresh install
  failed at import with a bare `ModuleNotFoundError`. Now declared, along
  with version floors for the other runtime dependencies.
- **`designer.py` would not parse below Python 3.12.** Two `print` statements
  reused the double-quote delimiter inside an f-string expression
  (`f"...{getattr(self, "_cvar_mean_phi", float("nan"))}..."`). PEP 701 made
  that legal in 3.12; on 3.9 and 3.11 it is a hard `SyntaxError`, so the
  package could not even be imported. Both now use single quotes inside. Found
  by the CI matrix on its first run — the local environment is 3.12, where the
  syntax is valid, so nothing local could have caught it.

- **The class docstring advertised two criteria that do not exist.** It listed
  a `G-optimal` criterion — there is no `g_opt_criterion` anywhere in the
  codebase — and `CVaR-D/A/E`, when only `cvar_d_opt_criterion` exists and
  `solve_cvar_problem` rejects any criterion whose name lacks `cvar`. Both
  claims were rendered on the published API page. The G-optimal line is
  removed; the CVaR entry now names CVaR-D only. The nearest real equivalent of
  classical G-optimality is the `eg` criterion (worst-case largest eigenvalue
  of PVAR), which the prediction-variance section documents.

- **`diagnose_sensitivity()`'s report rules were hardcoded to 100 characters**
  while the table's width is computed from the parameter names
  (`2 + 20 + n_mp*(pcw+2) + 22`). The rule was therefore wrong in both
  directions — 158 characters of table under a 100-character rule for a
  6-parameter model, and a 100-character rule overhanging an 82-character
  table for a 2-parameter one. The header is now built first and the rules
  and centred title derive their width from it.

- **`run_estimability()`'s `abs info` figure carried the wrong axis label and
  no threshold.** The x-axis read "normalised to the largest", inherited from
  the two E panels, which is false for `abs_info` — it is an absolute,
  dimensionless quantity, and that is the whole reason it is reported. The
  figure also drew no threshold line, so the one place `abs_info < 1` was
  invisible was the plot, while the returned table's `underdetermined`
  column and the printed report both flag it. Each panel now carries its own
  threshold and axis label: `abs info` gets a dashed line at 1 labelled
  "under-determined below 1", the E indices keep the resolution tolerance.
  The `_plot_estimability` docstring's two stale references to the removed
  "step-1 norm" metric are corrected to `abs info`.

- **`plot_sensitivities()` crashed on any single-response or
  single-parameter model.** `plt.subplots(nrows=n_m_r, ncols=n_mp)`
  collapses its return to a 1-D array of axes when either dimension is 1
  (and to a bare `Axes` when both are), but the body indexes
  `axes[row, col]` unconditionally — so the method raised `IndexError`
  (or `TypeError` for a 1x1 grid) on three of the four possible grid
  shapes. Only a model with multiple responses *and* multiple parameters
  worked, which is why the capability suite never caught it. Fixed with
  `squeeze=False`; the sibling `_plot_optimal_sensitivities()` already
  handled this by reshaping after the call and was unaffected. Unrelated
  to matplotlib version — reproduced identically on 3.10.9 and 3.11.1.

- **matplotlib 3.11 compatibility.** `designer.py` called
  `matplotlib.cm.get_cmap()` at two plotting sites. That function was
  deprecated in matplotlib 3.7 and **removed in 3.11**, so those paths
  raised `AttributeError` on any current install. Both now use
  `matplotlib.pyplot.get_cmap(name, lut)`, matching the third call site
  which already did. `matplotlib.cm` is no longer imported. Verified by
  running the plotting paths under both matplotlib 3.10 and 3.11; no
  version cap is needed and the `>=3.4` floor is unchanged.
- **`examples/ode` case-2 collocation grid**: a sampling time sat a hair off
  an existing collocation node, creating a machine-epsilon finite element.
  IPOPT reported "Optimal Solution Found" while returning `CA = 31 mol/L`
  from `CA0 = 5`, and `case_2_no_ift.py` reported a D-optimal criterion of
  45.31 against the correct 10.72. Refining `nfe` did not help — the
  signature of a formulation problem rather than truncation error.

Fixed in `testing_scripts/pydex_full_capability_test.py`:

- **The CyIpoptNLP warning suppression never worked.** Pyomo emits through
  `logging`, not the `warnings` module, so `warnings.filterwarnings()` could
  not see it and the deprecation appeared in every run. Replaced with a
  `logging.Filter` attached to the handler pyomo installs on the `pyomo`
  logger — it must be the handler, not the logger, because records
  propagating up from `pyomo.dae` do not have the parent logger's filters
  applied. The same filter collapses the `More finite elements were found in
  ContinuousSet` message, which a full run emitted **871 times** for 2613
  lines — 58% of a 4492-line log. Both are counted rather than discarded and
  reported at the end of the run, because the case_2 collocation bug had this
  fingerprint and a change in the count is worth seeing.
- **The stale-module guard matched the test file's own name.** The loop
  clearing `sys.modules` used `startswith('pydex')`, which also matches
  `pydex_full_capability_test`, so importing the suite under its own name
  died with `KeyError`. Harmless under `__main__`, but it blocked importing a
  single section for debugging. Now matches `'pydex'` exactly or the `'pydex.'`
  prefix.

### Known issues

- `vdi_criterion` collapses to D-optimality when `n_m_r == n_mp`, because
  `W` is then square and `det(PVAR) = det(W)²/det(FIM)`. It is a distinct
  criterion only when there are fewer measured responses than parameters.
- `run_estimability` has no regression section in the capability suite yet —
  it is the only new capability without a guard.
- `design_experiment(regularize_fim=True)` overwrites the `_regularize_fim`
  attribute from its keyword argument, so setting the attribute directly on
  the instance is silently discarded.
