# Changelog

All notable changes to this fork are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-08-26

### Fixed

- **The combination atomic FIM averaged sensitivities instead of summing
  information.** For a design with `n_spt=k`, a sampling-time combination's
  contribution was built as `(mean_t S_t)ᵀ W (mean_t S_t)` — the outer product
  of the MEAN sensitivity. Information from independent measurements ADDS, so
  a combination collecting k samples contributes `Σ_t S_tᵀ W S_t`: square,
  then add. Averaging first collapses k samples into one pseudo-sample at the
  average sensitivity and discards exactly the information that comes from the
  times being *different*; in the limit `S_t = −S_t'` it reports ZERO
  information from two highly informative measurements.

  **This changed designs, not just criterion values.** Measured on a
  4-candidate/4-time model, the averaged form deviated from the summed form by
  40% relative and REORDERED the candidates — it preferred candidate 2 where
  the correct form prefers candidate 3.

  A second, separate expression in the cached-atomics branch used
  `mean(Sᵀ W S)`, a third quantity (exactly `sum/k`, so design-preserving).
  That branch was verified UNREACHABLE by instrumentation — the atomics stored
  for `_specified_n_spt` number `n_c × n_spt_comb` while its reshape expects
  `n_c × n_spt`, so the staleness guard always forces a recompute — but it has
  been corrected too, so reviving it cannot silently reintroduce the defect.

  Every design using `n_spt=k` is affected. Superseded reference values are
  listed at the end of this entry.

### Removed

- **`optimize_sampling_times` removed.** It never entered the optimisation: it
  appears nowhere in `_solve_pyomo` or `_solve_scipy_slsqp`, and `True` vs
  `False` gave bit-identical efforts (`max |difference| = 0.0`, criterion
  identical to every digit). It selected only how results were REPORTED, while
  reading as though it controlled whether sampling times were optimised —
  which they always are, per (candidate, sampling-time) cell, unless `n_spt`
  says otherwise.

  That name caused real damage: five examples documented behaviour they never
  had, capability suite sections 03 and 06 were byte-identical duplicates that
  reported the same criterion indefinitely without anyone being able to
  explain it, and section 45's `assert v_opt <= v_fixed` compared a number
  with itself.

  Reporting granularity is a property of the problem rather than a user
  preference, so it is now derived from the model. Passing the argument raises
  with a message naming the replacement.

- **`fixed_sampling_grid`, added in 0.5.0, withdrawn.** With the atomic FIM
  corrected, `n_spt = <number of listed sampling times>` expresses exactly the
  same design problem through the single existing mechanism: `C(n, n) == 1`,
  so one schedule per candidate containing every time, hence one effort per
  experiment. Verified identical — design agreement `2.2e-16` and criterion
  agreement `3.6e-15` against a static multi-response reformulation, which
  reaches the FIM through entirely different machinery. Two ways to pose one
  problem, resting on two different pieces of arithmetic, is worse than one,
  so its constraint machinery was deleted from both solver paths rather than
  kept as a wrapper. Its `1/n_spt` criterion rescaling goes with it.

- **Unknown keywords to `design_experiment()` now raise.** `**kwargs` used to
  swallow anything unrecognised, so a typo (`optimise_sampling_times`,
  `n_spts`) or a withdrawn argument silently produced a different design with
  no error. `optimize_sampling_times`, `fixed_sampling_grid`, `package` and
  `optimizer` (the last two from the pre-0.2.0 cvxpy API, never parameters of
  this fork) each raise with a message naming the replacement.

  **This immediately found nine live instances of the bug it was written to
  prevent**, all in this repository and all silently discarded until now:
  `verbose=0` passed to `design_experiment()` eight times (capability suite
  section 55 twice, and all three `examples/b_optimal/` scenarios) — the
  method has no `verbose` parameter and never has, so none of those calls was
  ever quiet — and `package="pyomo"` once in section 39. All nine call sites
  were corrected rather than the check being relaxed. It also caught a
  `README.md` snippet and a `Designer` docstring example that passed
  non-existent arguments and would have raised if copy-pasted.

### Changed

- **`n_spt` is now the only control over sampling times, and documents three
  cases** — in the `Designer` class docstring, `design_experiment`'s parameter
  documentation, the `sampling_times_candidates` property, capability suite
  sections 06 and 56, and every affected example:

  | What you want | How to ask | Effort allocated per |
  |---|---|---|
  | Optimize sampling: choose conditions AND which times | omit `n_spt` (default) | measurement — (candidate, time) cell |
  | Exactly k samples per run; optimiser chooses which k | `n_spt=k` | run of k samples |
  | Measure every listed time, on every run | `n_spt=<number listed>` | run of the whole series |

  The first and third are genuinely different design problems and generally
  give different answers, because when sampling times are optimized an
  uninformative time costs nothing — it simply gets zero effort — whereas on a
  fixed grid you pay for it regardless. On the suite's own model, optimizing
  leaves **3,195 of 3,200** (candidate, time) cells at zero effort.

- **The design report no longer prints a boolean the reader has to decode.**
  `Sampling Times Optimized: True/False` is replaced at all four print sites
  by a derived line stating which of the three cases applies, e.g.
  `Sampling Times: FIXED -- all 11 listed time(s) measured on every run;
  effort allocated per experiment`. The old line was misleading in both
  directions: it reported `False` while sampling-time effort was in fact being
  optimised, and once the flag was removed it reported `True` even for a fixed
  grid — the one case where sampling times are *not* being chosen. The now
  redundant `Number of Samples Per Experiment` line was folded in.

- **Five examples claimed a fixed sampling grid while requesting the
  optimized case.** `case_2.py`, `case_2_no_ift.py` and
  `case_2_no_ift_no_collocation.py` documented round 1 as "Every selected
  experiment is measured at ALL ELEVEN sampling times"; `case_3.py` and
  `case_3_ift.py` printed "All 11 evenly-spaced time points used per run".
  Their fixed-grid designs now pass `n_spt` equal to the full grid, which is
  what those sentences describe. `case_2_ds.py` makes no such claim and is
  unchanged.

### Testing

- Capability suite: **301 assertions across 60 sections** (was 297 / 60).
  Absorbed Pyomo noise unchanged at **865 + 1** — the fingerprint has held
  across four releases, because the noise originates in `pyomo.dae`
  collocation which none of the new sections builds.
- **Section 06 repurposed.** It was a byte-identical duplicate of section 03.
  It now contrasts optimized sampling times with a fixed grid, asserting they
  give different designs, that the fixed grid allocates one effort per
  experiment, and that optimizing genuinely leaves listed times unused.
  Section 03's criterion is unchanged at `23.2105`.
- **Section 45's comparison de-vacuumed.** `assert v_opt <= v_fixed` compared
  identical calls; the two sides now differ (`8.81110749` optimized vs
  `11.23398687` fixed) and an added assertion requires them to differ, so it
  cannot silently become vacuous again.
- **Section 56 rewritten** to assert the fixed-grid route and the static
  multi-response reformulation agree on BOTH design and criterion, and that
  the optimized default is a different answer. Tightened from `1e-5` with an
  `n_mp·ln(n_spt)` offset to `2.2e-16` / `3.6e-15` with no offset.
- `tests/test_fixed_sampling_grid.py` removed with the feature (129 → 123
  solver-free tests).
- Drift references unchanged: section 17 `23.7240` (BARON/GAMS), section 53
  `5.258e-13`, section 54 `1.802e-12` / `3.526e-09`, section 03 `23.2105`.

### Superseded reference values

Recomputed after the atomic fix and the example corrections:

| what | was | now |
|---|---|---|
| `case_2.py` round 1 (fixed grid, all 11) | `10.657395` (mislabelled) | `19.489976` |
| `case_2.py` round 2 (optimized) | — | `10.657395` |
| `case_2.py` round 3 (`n_spt=2`) | `10.610118` | `13.429393` |
| `case_3.py` design 1 (fixed grid, all 11) | `69.59152865937244` | `75.68585637815247` |
| `case_3.py` design 2 (`n_spt=5`) | `55.96287449735705` | `74.93783375850808` |
| `case_3_ift.py` design 1 (fixed grid, all 11) | — | `33.532162683726085` |
| `case_3_ift.py` design 2 (`n_spt=5`) | — | `32.32621199537052` |

Note `case_2.py`'s old round-1 figure is exactly its new round-2 figure,
confirming those two rounds had been posing the same problem. `case_3_ift.py`
design 2 is bit-identical before and after, as expected — it already used
`n_spt=5` and the corrected atomic, so only design 1 could move.
`case_2_no_ift*.py`'s recorded round-1 values are marked superseded in their
docstrings and need re-measuring.

## [0.5.0] - 2026-08-25

### Added

- **`fixed_sampling_grid` argument to `design_experiment()`** — allocates one
  effort per EXPERIMENT rather than per (candidate, sampling-time) cell, so
  every listed sampling time is measured on every run and the only decision
  is WHICH experimental conditions to run. This is the common industrial case
  of a fixed analytical schedule you do not control; it previously could not
  be posed directly on a dynamic model, because per-sampling-time effort was
  always a free variable.

  Implemented as `n_spt - 1` linear equalities per candidate, applied on BOTH
  the native Pyomo path and the SLSQP fallback — omitting the latter would
  have let the flag be silently ignored for every non-native criterion
  (pseudo-Bayesian type 1, `vdi`, the six prediction-variance criteria, CVaR).

  The resulting FIM is `1/n_spt` times the per-experiment FIM. Because
  `n_spt` is constant this is a CONSTANT rescaling: the design is unaffected
  and log-det criteria shift by exactly `n_mp * ln(n_spt)`, the same
  structure as `error_cov` scaling. **Criterion values are therefore not
  comparable across this flag; designs are.**

  Requires a dynamic model, a grid common to every candidate, and is mutually
  exclusive with `n_spt` (which asks the optimiser to CHOOSE which times to
  sample). Ragged/NaN-padded grids raise `NotImplementedError` deliberately:
  there the rescaling is `1/n_spt_c`, different per candidate, so it would
  silently reweight candidates and change which design is optimal.

  Default `False`, reproducing all previous behaviour exactly. Verified
  against an INDEPENDENT reference — the same experiment posed as a static
  multi-response model, which reaches the FIM through entirely different
  machinery — agreeing on the design to `2.1e-08` with the criterion offset
  matching `n_mp * ln(n_spt)` to 8 significant figures. Guarded by capability
  suite section 56 and `tests/test_fixed_sampling_grid.py`.

### Fixed

- **A-optimality treated a numerically singular FIM as invertible, depending
  on arithmetic path.** `_a_opt_criterion`, `_pb_a_opt_criterion` and
  `_safe_fim_inverse` each tested positive-definiteness with a strict
  `eig > 0` and no tolerance. For a FIM that is exactly singular in theory,
  the residual eigenvalue in the null direction is floating-point roundoff
  whose SIGN depends on the order operations were performed in: measured on
  the same mathematical matrix, accumulate-by-candidate gave `+1.2e-19`, a
  single `S.T @ diag(e) @ S` matmul gave `-9.3e-20`, and longdouble gave
  `-2.4e-20`. On the positive side the FIM "inverted" and A-optimality
  returned a huge finite value (~`8.2e+18`) instead of `+inf`.

  All three now use a RELATIVE cutoff (`rtol=1e-12` against the largest
  eigenvalue), matching `diagnose_fim_structure`'s existing convention for
  the same question. **This is a user-visible behaviour change**: a design
  whose FIM is singular-but-tiny-positive now reports infeasible where it
  previously returned a finite criterion. That is the intended semantics —
  a huge finite value is a near-best score for a minimised criterion, so the
  old behaviour attracted the optimiser toward rank-deficient supports, the
  very bug `smoke_test_designer.py` CHECK 2 exists to catch. Guarded by
  `tests/test_a_opt_singular_fim_guard.py` (12 tests).

- **A static model's `sampling_times_candidates` had the wrong SHAPE.**
  `_get_component_sizes` allocated it following `ti_controls_candidates`,
  i.e. `(n_c, n_tic)`, while `n_spt` is always 1 for a static model, so it
  should be `(n_c, 1)`. 0.4.1 made the contents deterministic but left the
  shape. Latent (every read is behind an `if self._dynamic_system:` guard),
  so this is a correctness cleanup rather than a live bug fix; the array is
  pickled by `save_state`, so the save/load round-trip is now covered too
  (capability suite section 58).

### Changed

- **`optimize_sampling_times` documentation corrected — it does not control
  the formulation.** The flag appears nowhere in either solver path: per
  sampling-time effort is a free decision variable either way, and passing
  `False` vs `True` yields BIT-IDENTICAL efforts (verified: `max |diff| =
  0.0`, criterion identical to all digits, which is why suite sections 03
  and 06 have always reported the same value). What actually changes the
  formulation is `n_spt` — and because `n_spt` force-overrides this flag to
  `True`, the flag looked causal while being only a necessary companion.
  It controls reporting and candidate extraction only.

  In particular the `sampling_times_candidates` docstring claimed that with
  `optimize_sampling_times=False` "every listed time is measured". That was
  false — the optimiser routinely drives most grid times to zero effort
  regardless — and `fixed_sampling_grid` is now the way to require it.

- `.gitignore` now covers `_flat/`, the scratch directory the README's own
  solver-free test recipe tells you to create inside the repo.

### Testing

- Capability suite: **297 assertions across 60 sections** (was 287 / 57).
  New sections 56 (`fixed_sampling_grid` vs an independent reformulation),
  57 (`apportion()` on a NON-D-optimal design — nothing previously
  apportioned anything but D-optimal, which is exactly why 0.4.1's
  `UnboundLocalError` in the efficiency block survived), and 58 (static
  placeholder shape and save/load round-trip). Absorbed Pyomo noise
  unchanged at 865 + 1.
- Solver-free `tests/`: **129 passed** (was 62). Four new files.
- Every new assertion was confirmed to FAIL against the pre-fix code, and two
  first-draft tests were discarded for being vacuous — one whose fixture
  planted a `1e-19` eigenvalue that a `Q @ diag @ Q.T` reconstruction's own
  rounding swamped, and two that asserted on a locally built dict rather than
  on `designer.py`.

## [0.4.1] - 2026-08-25

### Added

- **`print_optimal_candidates_table()` and `get_optimal_candidates_table()`**
  on `Designer` — a tabular, one-row-per-suggested-experiment view of the
  optimal design. Previously the only output was `print_optimal_candidates()`,
  which shows time-invariant controls as raw, unlabelled vectors — readable
  only by decoding against `ti_controls_names` by hand.
  `get_optimal_candidates_table()` returns a `pandas.DataFrame` (numeric,
  unrounded, directly exportable to CSV); `print_optimal_candidates_table()`
  prints a formatted version and is now also called automatically at the end
  of `print_optimal_candidates()`, appended below its existing output, not
  replacing it.

  Columns: `Experiment` (sequential 1..N — the number to use when
  communicating the protocol; does not correspond to anything in the
  original candidate pool), `Candidate` (1-indexed position in the original
  candidate pool, kept for cross-reference with plot legends, solver
  progress logs, and `candidate_names` in exported results), one column per
  `ti_controls_names`, `Schedule` (present only when sampling times were
  optimised with a fixed `n_spt` — two schedules on one candidate are two
  *separate, mandatory* experiments, a required split of effort, not
  alternative/optional ways of running the same one), `Sampling Time`, and
  `Effort`.

  One behavioural fix that fell out of building this: the fixed-grid
  (sampling times not optimised) case now reports only the times carrying
  nonzero effort, rather than the whole predefined grid regardless of use —
  a grid point can end up at zero effort at the optimum, and the previous
  unfiltered dump implied it was part of the recommended protocol when it
  wasn't.

  Found, not fixed, while verifying this against real solves: for a
  **static** system, `get_optimal_candidates()`'s internal sampling-times
  field (`opt_cand[3]`) reads back as uninitialised memory (indexing into a
  `sampling_times_candidates` that is never meaningfully allocated when
  there is no time dependency). The new table never touches that field for
  static systems, so it's unaffected, but anything else reading it directly
  is not safe. Not fixed here — separate issue, separate fix.

  Verified before commit: capability suite 287/287 (absorbed-noise count
  unchanged, 865+1), `smoke_test_designer.py` 4/4, sections 54/55 (via
  direct function import — see Sandbox setup note),
  `PYTHONPATH=_flat pytest -q tests/` 51/51, `compileall` and
  `ast.parse(..., feature_version=(3,9))` both clean.

### Fixed

- **`print_optimal_candidates()` and `apportion()` printed the WRONG
  candidate's sampling grid.** Both reports' fixed-grid branches
  (`optimize_sampling_times=False`) did
  `print(self.sampling_times_candidates[i])` where `i` is the `enumerate`
  counter over `optimal_candidates` -- the position in the SUPPORTED list, not
  the candidate index. Whenever the supported candidates were not the first N
  of the pool, the report showed some other candidate's times under the right
  candidate's heading. Reproduced on pristine v0.4.0: a design supported on
  candidates 3 and 4 printed candidate 2's grid (`[0.44 0.55 0.66]`) beneath
  `[Candidate 4]`, whose real grid is `[1.11 1.22 1.33]`. Both branches now
  read `opt_cand[3]`, which `get_optimal_candidates()` populates from
  `opt_cand[0]`. Visible in capability-suite section 30, which has a distinct
  grid per candidate.

- **Both reports listed sampling times carrying no effort as though they were
  part of the design.** The fixed-grid branch printed the candidate's entire
  grid regardless of effort. Times at zero effort are genuinely NOT in the
  design: verified by construction that the FIM depends on how effort is
  distributed across sampling times even when `optimize_sampling_times` is
  `False` -- holding a candidate's total fixed and moving its effort onto a
  zero-effort time changed `log det(FIM)` from `3.509` to `-34.666`. Both
  reports now list only the effort-carrying times and state how many grid
  times were omitted.

  Note `apportion()`'s run count is per CANDIDATE, not per sampling time
  (`opt_eff` collapses to one value per candidate on this path), so its report
  states that each run samples at the listed times rather than splitting runs
  across them.

  Both defects were found by diffing the existing report against the new table
  on a real solve -- no assertion covered either, so the capability suite
  passed 287/287 while the output was wrong. `tests/test_optimal_candidates_report.py`
  now guards both, and both assertions were confirmed to FAIL against the
  pre-fix `designer.py`. The first version of that guard was VACUOUS and
  passed against the broken code: numpy renders `np.array([0.10, 0.20])` as
  `[0.1 0.2]`, so asserting on `"0.10"` could never fire. The fixture now uses
  values (`0.11`, `0.77`, ...) that render identically whether printed by
  numpy or by the formatted report.

- **`apportion()` crashed with `UnboundLocalError: efficiency` for 11 of the
  15 public criteria.** The efficiency block assigned `efficiency` inside a
  four-way `if`/`elif` chain (`d_opt`, `ds_opt`, `a_opt`, `e_opt`) with no
  `else`, then unconditionally called `np.squeeze(efficiency)`. Every other
  criterion -- `v_opt`, `vdi`, `cvar_d`, `b_opt`, `u_opt` and all six
  prediction-variance criteria (`dg`, `di`, `ag`, `ai`, `eg`, `ei`) -- fell
  through and raised, making apportionment unusable for most of the library.
  Confirmed present on pristine v0.4.0. A relative-efficiency RATIO genuinely
  is undefined for those criteria, so the fix does not invent one: the four
  supported criteria report exactly as before, and the rest now say the
  efficiency is not reported and why, while still printing the Kiefer bound
  (which does hold). Verified across all ten solver-reachable criteria.

  This is a *reporting* fix only -- the apportionment itself was always
  correct, which is why the capability suite never caught it: sections 10 and
  32 apportion a D-optimal design, the one case that worked.

- **A static system's `sampling_times_candidates` was uninitialised memory.**
  `_get_component_sizes` allocated it with `np.empty_like(ti_controls_candidates)`
  for signature-1 (static) models, and `get_optimal_candidates()` copies that
  straight into `opt_cand[3]` -- so a static design carried nondeterministic
  garbage (values like `4.4e-315`) in a field that looks like data. Now
  zero-filled. Latent rather than live: every read is behind an
  `if self._dynamic_system:` guard, so nothing displayed or computed from it,
  and the new table deliberately never reads it for static systems. The array's
  SHAPE is also wrong -- it follows `ti_controls_candidates` while `n_spt` is 1
  -- but correcting that reaches save/load and the sampling-time padding
  helpers, so it is left alone and documented in place.

  Neither of those two fixes is covered by `tests/` (both need a solved
  design); both were verified by execution across the criterion set.

- **`ag`, `ai`, `eg` and `ei` crashed mid-solve on an un-invertible FIM.**
  `eval_pim` sets `self.pvars = None` when `_safe_fim_inverse` cannot invert
  the FIM, and its own comment states the intent: this is so "the consuming
  criteria report an infeasible design (+inf) instead, which is what an
  optimiser can actually act on". `dg_opt`, `di_opt` and `vdi` honour that
  contract; `ag`, `ai`, `eg` and `ei` did not -- they iterated `None` and
  raised

      TypeError: 'NoneType' object is not iterable

  from inside the SLSQP objective, aborting the entire `design_experiment()`
  run instead of steering the optimiser away from an infeasible point. Found
  when a real `design_experiment(eg_opt_criterion)` died this way. The four
  unguarded criteria are exactly the four with no docstring, which is
  plausibly why they were missed when `dg`/`di` were hardened. All six now
  return `+inf`, the worst attainable value for a minimised criterion and
  therefore correct regardless of each one's internal sign convention.

  Guarded by six parametrised tests in
  `tests/test_optimal_candidates_report.py`; the four affected cases were
  confirmed to FAIL against the pre-guard `designer.py` while `dg`/`di` pass,
  so the test discriminates rather than merely passing.

  Note the capability suite exercises all six criteria (sections 29 and 40)
  and never hit this: those runs stay on a well-conditioned FIM, so `pvars`
  is never `None` there.

### Removed

- **`publications/`** (61 MB, 41 scripts, two paper subfolders). The folder
  claimed to hold "the original Python codes written to compute results that
  were previously published", for:

  - Kusumo, Kuriyan, Vaidyaraman, García-Muñoz, Shah & Chachuat, *Risk
    mitigation in model-based experiment design: a continuous-effort approach
    to optimal campaigns*, **Comput. Chem. Eng.** 159 (2022) 107680,
    doi:10.1016/j.compchemeng.2022.107680
  - Kusumo, Kuriyan, Vaidyaraman, García-Muñoz, Shah & Chachuat,
    *Probabilistic framework for optimal experimental campaigns in the
    presence of operational constraints*, **React. Chem. Eng.** 7(11) (2022)
    2359–2374, doi:10.1039/D1RE00465D

  That claim no longer holds for this fork. `risk_mitigation/` calls
  `design_experiment(package="cvxpy", ...)` against MOSEK — the pre-0.2.0 API
  removed in 0.2.0 — so the scripts are not merely numerically stale, they
  cannot execute at all: cvxpy is neither a dependency nor importable here,
  and there is no `package=` argument anywhere in `designer.py`. Retaining
  them under a README asserting they reproduce published figures
  misrepresents this fork.

  **This is not the papers' cited code archive.** The React. Chem. Eng.
  paper's ESI names `https://github.com/omega-icl/pydex` as the source for
  both the package and the case-study files; the same author group and
  lineage covers the Comput. Chem. Eng. paper. Reproducibility for both
  therefore rests with `omega-icl/pydex`, not with this fork, and removing
  the folder here breaks no code-availability citation.

  Removed from HEAD only — no history rewrite. Every file remains recoverable
  from this repository's history and from the `v0.2.0`–`v0.4.0` tags; a
  rewrite would have invalidated those tags and would not shrink existing
  clones anyway. Use `--depth 1` or a sparse checkout for a small clone.

  Stale references removed alongside it: the "Publication code" sections of
  `docs/source/examples.rst` and `examples/README.md`, and the
  `!publications/**/*.pkl` exception in `.gitignore`.

## [0.4.0] - 2026-08-19

### Added

- **`b_opt_criterion` — bracketing-optimal design**, implementing Chen,
  Paulavičius, Adjiman & García-Muñoz (2018), *AIChE J.* 64(11):3944–3957,
  doi:10.1002/aic.16214. This answers a different question from every other
  criterion in pydex: not "which experiments best determine my parameters"
  but "which experiments best bracket my operating space" — the regulator's
  question in a pharmaceutical bracketing study. It is not
  sensitivity-based and does not involve the Fisher information matrix at
  all. Two objectives are combined by weighted-sum scalarisation (the
  paper's Eq. 24), selected with `output_weight`: input-space bracketing
  (D-optimality applied to the scaled input-factor values, giving an
  orthogonal corner-seeking design) at `0`, and output-space coverage
  (maximising the volume spanned by the candidates' predicted responses) at
  `1`.

  Where the paper solves a continuous multiobjective/bilevel NLP, this is
  posed as **binary subset selection over a pre-evaluated candidate pool**:
  the process model is evaluated once per candidate offline, and the MINLP
  contains only a cardinality constraint, two Cholesky lifts and the
  objective — no process model at all. Both log-determinants are lifted
  through Cholesky factors (`M = LLᵀ`, `log det M = 2 Σ log L_jj`), reusing
  the existing `is_d` pattern, so every monomial is bilinear. An earlier
  LDL-with-division formulation produced trilinear monomials, which have
  weaker relaxation envelopes.

  Requires an MINLP solver (`solver="bonmin"`), an exact design size
  (`n_exp`), and `simulate_candidates()` beforehand — b_opt is the only
  criterion that reads `designer.response`.

  The criterion is isolated by construction: `is_b = 'b_opt' in crit_name`
  is collision-free against all 36 existing criterion names, and an early
  return in `_solve_pyomo` dispatches to a dedicated `_solve_pyomo_b_opt`
  ahead of the atomic-FIM computation, the structural-singularity gate and
  the Ds feasibility pre-check, none of which are meaningful here.
  `is_native` and `use_minlp` are unmodified and the `min_effort` sparsity
  constraints are unmoved; only two lines change outside the new code —
  `design_experiment`'s signature and its dispatch call.

- **Two guards on `b_opt_criterion`, both measured rather than derived.**

  `n_exp >= max(phi, n_resp + 2)`, where `phi` is the number of input
  factors. Both Cholesky lifts are built unconditionally, whatever
  `output_weight` is, and both floor their diagonal at `1e-8`; since
  `M == L Lᵀ` with every `L[j,j] >= 1e-8` forces
  `det(M) >= 1e-8**(2*dim)`, a rank-deficient `M` cannot be represented and
  the program is strictly infeasible. The output covariance is *centered*,
  so its rank is at most `n_exp - 1` and the algebraic bound is
  `n_resp + 1` — but that proves insufficient in practice: at exactly that
  value the covariance is full rank with no margin above the floor, and
  bonmin reports the problem infeasible whenever the output term carries
  weight. Measured on a 10-candidate `phi=2`/`n_resp=2` pool: `n_exp=3`
  infeasible at `output_weight >= 0.5`, `n_exp=4` solving at once.

  The check is up front rather than left to the solver because proving
  infeasibility of a nonconvex MINLP is the expensive direction. On a
  70-candidate `phi=6` pool, `n_exp=5` ran for over 17 minutes of bonmin CPU
  without terminating, while `n_exp=6` solved in seconds — so without the
  guard, a mistyped `n_exp` costs an unbounded hang and prints no diagnosis.

  A **failed solve no longer yields a design**. When bonmin returns
  `infeasible`, `pyo.value(m.b[i])` evaluates to 1.0 for every candidate, so
  the previous warn-and-continue produced a "design" selecting the entire
  pool at effort `1/n_exp` each — breaching the cardinality constraint, with
  efforts summing to `n_c/n_exp` rather than 1 — behind nothing but a
  printed warning. Hard terminations (`infeasible`, `unbounded`, `error`)
  now raise `RuntimeError` before any extraction, and the returned selection
  is validated against `n_exp` and for binariness independently of what the
  solver reported. Solves stopping at a time or iteration limit still return
  their incumbent, but say so, and record `_b_opt_termination` and
  `_b_opt_proven_optimal` so a caller can tell an incumbent from a proven
  optimum.

- **`tests/test_b_opt_guards.py`** — 23 solver-free argument-validation
  tests, so CI covers the b_opt guards on py3.9/3.11/3.12. The `tests/`
  total goes 26 → 51. The behaviour tests need an MINLP solver and live in
  the capability suite, which CI does not run; splitting them this way means
  a brand-new criterion is not left with documentation promising specific
  behaviour and no automatic guard.

- **Capability suite section 55** — `b_opt_criterion` verified against
  **exhaustive enumeration**. On a 10-candidate pool at `n_exp=4`, all
  C(10,4)=210 subsets are enumerated in-process and bonmin's design is
  asserted to be the true global optimum, not merely a good one: exact
  agreement at `output_weight` 0.0, 0.5 and 1.0 with objective gap
  `0.000e+00`. Also asserts that the two extremes select *different*
  designs (without which the section would pass on a criterion that ignored
  `output_weight`), that efforts are exactly `1/n_exp` and sum to 1, that
  the weight sweep is monotone in both objectives with no dominated point,
  and that the `n_exp` bound is enforced.

- **Capability suite section 54** — `run_estimability` regression on both
  sensitivity paths against an **analytic reference**, closing the gap
  recorded under Known issues in 0.2.0. The fixture is an over-parameterised
  first-order decay, `A(t) = exp(p1 + p2) * exp(-k*t)`, in which `p1` and
  `p2` enter only as a sum: the closed-form sensitivities `dA/dp1` and
  `dA/dp2` are therefore *identical*, so correlation `+1`, FIM rank exactly
  2 of 3 and one null direction are all known on paper rather than recorded
  from a previous run. Finite differences agree with the closed form to
  `1.8e-12` and exact IFT to `3.5e-09`; both find rank 2 of 3 with the
  redundant pair as culprits, and the documented E-index floors (`1e-3` for
  finite differences, `1e-7` for IFT) are pinned explicitly, since
  `examples/ode/case_3.py` and `case_3_ift.py` both now depend on them.

  Because the two columns are analytically equal, *which* member of the pair
  gets flagged is decided by floating-point noise — and the two paths do
  disagree, which is the `case_3` FD-versus-IFT disagreement in miniature
  and provably a convention rather than a contradiction. The section
  therefore asserts the set and the count, never the member.

  The section passes **no** `base_step` override, so the default
  per-parameter finite-difference step introduced in 0.3.0 is what is
  tested; run with the pre-0.3.0 flat `base_step=2` the same fixture returns
  `2.3e-03` and fails. Every other section reaching a finite-difference step
  passes an explicit one, which would have masked a broken default
  indefinitely.

  Suite totals: 261 → 287 assertions across 57 sections. The absorbed-noise
  fingerprint is unchanged at 865 + 1.

- **`examples/b_optimal/`** — three worked scenarios. `scenario_1` and
  `scenario_2` reproduce the paper's film-coating and two-CSTR case studies
  (Table 1 and Figures 2–5, and Figures 8–9 respectively); `scenario_3`
  applies the criterion to a Suzuki–Miyaura coupling not from the paper and
  cross-checks its design against exhaustive enumeration of all 658,008
  five-point subsets. Model fidelity differs and is stated in each file: the
  CSTR model is transcribed from the GAMS source published with the paper
  and carries its actual numbers, whereas the coater model is an independent
  thermodynamic model rather than the paper's Supporting-Information
  equations, and the Suzuki kinetics are representative rather than fitted.

- **`examples/ode/case_3.py` gained the estimability workflow** already
  present in `case_3_ift.py` (previously un-versioned, on `main` since
  `f68a635`): estimability on all nine parameters, fix the two flagged
  unresolvable, estimability again on the reduced seven, then design. With
  all nine free `design_experiment()` refuses the model, since the rate law
  has an exact invariance — adding a constant to every `theta_i0` at once,
  or to every `theta_i1`, leaves every prediction unchanged. The point of
  the finite-difference version is that estimability analysis needs no
  tractable model: `run_estimability()` reads the sensitivity matrix and is
  indifferent to its provenance. Reference criterion values for the
  seven-parameter form are `69.59152865937244` (fixed sampling times) and
  `55.96287449735705` (5 of 11 times optimised).

### Known issues

- **The weighted sum reaches only the convex-hull portion of the Pareto
  frontier.** This is a real limit on what a user can ask for, and it is
  measured rather than argued. Maximising
  `(1-w)*log f_in + w*log f_out` is geometrically a straight line of slope
  `-(1-w)/w` swept in from outside the achievable set, so the winner is
  always a point on the convex hull of that set. A design that is genuinely
  Pareto-efficient but sits slightly *inside* the hull — in a dent of the
  frontier — is therefore optimal for **no weight at all** and cannot be
  returned however finely `output_weight` is swept. Enumerated exhaustively
  on the `scenario_1` coater pool (36 candidates, `n_exp=4`, all 58,905
  subsets): 13 Pareto-efficient designs, of which **5 are reachable and 8 are
  not**. The Pareto figures in `scenario_1` and `scenario_2` consequently
  under-report the frontier, and a practitioner sweeping the weight is never
  shown those designs.

  The restriction is not purely a loss, which is why it is recorded here
  rather than treated as a defect. A hull design is optimal over a *range* of
  weights and is properly Pareto-optimal, with a bounded marginal trade-off
  between bracketing and coverage; dent designs are optimal at knife-edge
  preferences where that exchange rate can be arbitrarily extreme. The
  monotone weight sweep asserted in capability-suite section 55 — `f_in`
  non-increasing and `f_out` non-decreasing in `output_weight` — is a
  *consequence* of traversing a convex hull over a fixed finite candidate
  set, not an independent property, and it is what makes the sweep figures
  legible. And `output_weight` has a clean reading as a marginal rate of
  substitution, so "0.5 is the balanced compromise" is meaningful.

  Two candidate completions, neither implemented:

  The paper's own Tchebycheff / L∞ scalarisation (Eq. 27) minimises the worst
  weighted shortfall from a reference point. Its level sets are right-angled
  cones rather than straight lines, and a corner can reach into a dent that no
  line can. Cheap here: one extra continuous variable and two linear
  constraints over log-determinant expressions the model already builds, with
  the reference point available free from the two single-objective solves any
  sweep already performs. But the plain weighted form admits **weakly**
  Pareto-optimal points, i.e. designs that are dominated — for a regulatory
  bracketing study that is a worse failure than an unreachable design, so it
  would need the augmented form (a small `rho * sum` term) to restore a
  proper-efficiency guarantee. Note also that a Tchebycheff sweep carries no
  monotonicity guarantee, so section 55's assertion would not transfer.

  For this *discrete* formulation the better fit is probably the
  epsilon-constraint method: maximise `log f_out` subject to
  `log f_in >= epsilon`. It provably reaches every Pareto-efficient design
  including those in dents, needs one extra linear constraint on an existing
  expression, has no weak-Pareto pathology, and its parameter states a
  requirement directly — "at least this much input bracketing" is much closer
  to how a bracketing requirement is actually specified than a weight is.
  Bounds on `epsilon` come from the same two single-objective solves.

  The bilevel formulation of the paper's Eqs. 28-31 is likewise not
  implemented.

- **Bonmin offers no global-optimality guarantee** for nonconvex MINLP. Its
  record on the instances tried here is perfect — five independent
  brute-force cross-checks agreeing exactly, including section 55 — but that
  is empirical evidence about those problems, not a proof.
- **Cost grows sharply with the number of input factors.** The coater
  (`phi=3`) solves in seconds; a `phi=6` reactor pool of 246 candidates took
  minutes per solve, and the shipped scenario is scaled down accordingly.
  This does not repeal the NP-hardness of cardinality-constrained subset
  selection.
- **The anti-clustering mechanism (`_b_opt_min_sep_frac`) is untested in
  anger.** It generates precomputed linear mutual-exclusion constraints on
  near-duplicate response candidates, the discrete analogue of the paper's
  tuned log-barrier. In every case tried it generated zero constraints,
  because the output term already separated the points. Selecting from a
  fixed pre-filtered pool is plausibly immune to the pathology it exists to
  fix rather than merely controlling it, but that is unverified.

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
