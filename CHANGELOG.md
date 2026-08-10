# Changelog

All notable changes to this fork are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
