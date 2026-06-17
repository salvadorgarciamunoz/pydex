"""
pydex.utils.diagnose_asl_elimination
=====================================
Diagnostic utility for Pyomo IFT models used with pydex.

The failure mode this tool detects is TRUE ELIMINATION:
a parameter Var that is absent from the ASL NLP primal vector entirely.

Two checks are performed
-------------------------
1. FULL-GRID CHECK (user-supplied sampling_times)
   Mirrors the initial model build pydex performs before the causal loop.

2. SINGLE-POINT CHECK (one sampling time — the midpoint of the supplied grid)
   Mirrors pydex's causal rebuild: for each output time pydex calls
   build_pyomo_model_fn with sampling_times=[t_val_f] — a single float.
   The reduced collocation grid can expose parameter chains that are
   eliminable at single-point scale but not at full-grid scale.
   A model that passes the full-grid check but fails the single-point
   check will crash during eval_sensitivities() with RuntimeError.

Background
----------
When pydex evaluates IFT sensitivities it:
  1. Temporarily unfixes the parameter Vars
  2. Instantiates PyomoNLP(model) — ASL compiles the NLP
  3. Builds col_order by name-matching each entry in all_vars against
     nlp.primals_names() — the column ordering is NAME-BASED, not
     position-based
  4. Reorders J_full columns as J = J_full[:, col_order]
  5. Splits:  J_p = J[:, :n_mp]  and  J_z = J[:, n_mp:]

Because step 3 performs name-based lookup, ASL's internal variable
ordering (free Vars before fixed Vars, etc.) is IRRELEVANT — pydex
always finds the right columns regardless of where ASL placed them.

The ONLY way pydex IFT can fail is if a parameter Var is completely
absent from nlp.primals_names() — step 3 raises RuntimeError when
the name-match returns None.  This is true ASL symbolic elimination:
the variable was substituted away before the NL file was written.

What triggers true elimination
-------------------------------
ASL eliminates a Var when its defining equality constraint has a
right-hand side that is fully determined by fixed quantities — i.e.
there is no path through the constraint graph to any free variable.

After step 1 (unfix parameter Vars), the parameters themselves are
free, so any constraint that directly involves them is live.  True
elimination is therefore rare with full grids but more common with
single-point grids where the reduced collocation structure can make
previously live chains appear collapsible.

Usage
-----
    from pydex.utils.diagnose_asl_elimination import diagnose_asl_elimination
    from my_model import build_pyomo_model

    diagnose_asl_elimination(
        build_pyomo_model,
        ti_controls      = [10.0, 303.15, 10.0],
        model_parameters = [5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5],
        sampling_times   = [0.001, 0.5, 1.0],
        param_names      = ["θ_10","θ_11","θ_20","θ_21","θ_30","θ_31","ν","α","β"],
    )

Requirements
------------
    pip install pyomo pyomo[pynumero]
    (PyomoNLP / pynumero_ASL must be available — same requirement as IFT)
"""

import numpy as np


class _DegenerateModel(Exception):
    """Raised internally when a probe point builds an all-fixed model.

    Signals that the supplied ti_controls hit a trivial/boundary branch of the
    model builder, so the ASL reachability check should be retried at a
    different (non-degenerate) point.  Not part of the public API.
    """
    pass


def _match_param_name(pname, nlp_primal_names):
    """
    Index of a parameter Var name within an ASL primals_names() list, or None
    if absent entirely (true ASL elimination — Failure Mode B).

    This MUST stay byte-for-byte equivalent to pydex.core.designer._match_nlp_var,
    including the exact-first ordering.  It is the gate the user runs before
    handing a model to the Designer; if it matched differently from the
    Designer's run-time matcher, a model could pass this check and then bind to
    a different Jacobian column (or fail to match) during the real design.

    EXACT MATCH WINS: an exact-equality pass runs before the qualified-name
    fallback pass, so that a model carrying both a top-level Var and a
    block-nested Var with the same leaf name (e.g. 'theta' and 'b.theta') never
    aliases one onto the other based on ASL's primal ordering.  (Kept as a
    standalone copy rather than importing designer, so this diagnostic stays a
    lightweight, dependency-minimal tool the user can run on its own.)
    """
    # Pass 1 — exact equality wins, independent of position.
    for i, n in enumerate(nlp_primal_names):
        if n == pname:
            return i
    # Pass 2 — qualified-name suffix / final-segment fallbacks.
    leaf = pname.split(".")[-1]
    for i, n in enumerate(nlp_primal_names):
        if (n.endswith("." + pname)
                or pname.endswith("." + n)
                or n == leaf):
            return i
    return None



def _check_survival(build_pyomo_model_fn, ti_controls, model_parameters,
                    sampling_times, PyomoNLP, pyo):
    """
    Internal helper: build the model, unfix params, instantiate PyomoNLP,
    and return (survived, eliminated, nlp_primal_names, param_var_names, n_all_vars).

    survived  : list of (i, label_placeholder, pyomo_name, nlp_name, asl_pos)
    eliminated: list of (i, label_placeholder, pyomo_name)

    Raises _DegenerateModel if the built model has no free Vars (an all-fixed
    "degenerate" model, e.g. a boundary point such as t=0).  Such a model is
    not a valid probe for the ASL reachability check: PyomoNLP cannot compile
    it, AND pydex's IFT path routes all-fixed models to its pure-Python
    differentiate() fallback (never touching ASL).  The structural check is
    candidate-independent, so the caller should simply retry at a non-degenerate
    point rather than treat this as a pass or a failure.
    """
    m, all_vars, all_bodies, t_sorted = build_pyomo_model_fn(
        ti_controls,
        model_parameters,
        sampling_times=list(sampling_times),
    )
    n_mp = len(model_parameters)
    param_vars      = all_vars[:n_mp]
    param_var_names = [str(v) for v in param_vars]

    # Degenerate-model guard: if every active Var is fixed, this point trips a
    # trivial/boundary branch in the builder.  The reachability verdict does
    # not depend on the numeric point, so signal degeneracy and let the caller
    # retry elsewhere instead of crashing inside PyomoNLP.
    if not any(not v.is_fixed()
               for v in m.component_data_objects(pyo.Var, active=True)):
        raise _DegenerateModel(
            f"all-fixed (degenerate) model at ti_controls={list(ti_controls)}"
        )

    for pv in param_vars:
        pv.unfix()
    try:
        nlp              = PyomoNLP(m)
        nlp_primal_names = nlp.primals_names()
    finally:
        for pv in param_vars:
            pv.fix()

    survived  = []
    eliminated = []
    for i, (pvar, pname) in enumerate(zip(param_vars, param_var_names)):
        matched_idx = _match_param_name(pname, nlp_primal_names)
        if matched_idx is not None:
            matched_name = nlp_primal_names[matched_idx]
            survived.append((i, "", pname, matched_name, matched_idx))
        else:
            eliminated.append((i, "", pname))

    return survived, eliminated, nlp_primal_names, param_var_names, len(all_vars)


def _nudge_ti_controls(ti_controls):
    """Yield perturbed copies of ti_controls that move time-like entries off a
    boundary (e.g. t=0), to escape a model builder's degenerate branch.

    The ASL reachability verdict is candidate-independent, so the specific
    nudged values do not matter — they only need to avoid the trivial branch.
    We try a few increasing offsets applied to any near-zero entry; if none
    of those help, we also try scaling every entry up by small factors.
    """
    base = np.asarray(ti_controls, dtype=float).flatten()
    seen = set()

    def _emit(arr):
        key = tuple(np.round(arr, 12))
        if key in seen:
            return None
        seen.add(key)
        return arr.tolist()

    for offset in (1.0, 0.5, 0.1, 5.0, 10.0):
        cand = base.copy()
        # bump any entry that is at/near zero (typical degenerate boundary)
        near_zero = np.isclose(cand, 0.0)
        if near_zero.any():
            cand[near_zero] = offset
            out = _emit(cand)
            if out is not None:
                yield out
    for scale in (2.0, 10.0, 0.5):
        out = _emit(base * scale)
        if out is not None:
            yield out


def _check_survival_robust(build_pyomo_model_fn, ti_controls, model_parameters,
                           sampling_times, PyomoNLP, pyo):
    """Run _check_survival, automatically retrying at a non-degenerate point if
    the supplied ti_controls build an all-fixed (degenerate) model.

    Because the structural reachability check does not depend on the numeric
    operating point, this lets the diagnostic own degeneracy handling entirely:
    callers (designer.py or a user at the REPL) need not know which points trip
    a builder's trivial/boundary branch.
    """
    try:
        return _check_survival(build_pyomo_model_fn, ti_controls,
                               model_parameters, sampling_times, PyomoNLP, pyo)
    except _DegenerateModel:
        pass  # supplied point was degenerate — try nudged points below

    last_degenerate = None
    for alt_tic in _nudge_ti_controls(ti_controls):
        try:
            return _check_survival(build_pyomo_model_fn, alt_tic,
                                   model_parameters, sampling_times,
                                   PyomoNLP, pyo)
        except _DegenerateModel as d:
            last_degenerate = d
            continue
    # Every probed point was degenerate — surface it so the caller can report
    # "unverified" rather than a spurious pass/fail.
    raise _DegenerateModel(
        "could not find a non-degenerate operating point to probe "
        f"(base ti_controls={list(ti_controls)})"
    )


def diagnose_asl_elimination(
    build_pyomo_model_fn,
    ti_controls,
    model_parameters,
    sampling_times,
    param_names=None,
    verbose=True,
):
    """
    Diagnose whether any model parameters are eliminated by ASL for a
    given Pyomo model, matching the exact procedure pydex uses for IFT.

    Runs TWO checks:
      1. Full-grid check  — with the supplied sampling_times
      2. Single-point check — with a single time from the grid midpoint,
         replicating pydex's causal per-spt rebuild

    A model must pass BOTH checks to be IFT-ready.

    Note on position checking
    -------------------------
    pydex builds col_order by NAME-BASED lookup (matching all_vars strings
    against nlp.primals_names()), so it correctly locates parameter columns
    regardless of their position in the ASL primal list.  Position is
    irrelevant and is shown for information only.

    Parameters
    ----------
    build_pyomo_model_fn : callable
        The user-supplied model builder.  Must follow the pydex IFT contract:
            (model, all_vars, all_bodies, t_sorted) =
                build_pyomo_model_fn(ti_controls, model_parameters,
                                     sampling_times=sampling_times)
        Parameter Vars must be the first n_mp entries in all_vars.

    ti_controls : array-like
        Time-invariant controls for the test candidate.

    model_parameters : array-like, length n_mp
        Nominal parameter values.

    sampling_times : array-like
        Sampling times for the full-grid check.  The single-point check
        uses the midpoint element of this array.

    param_names : list of str, optional
        Human-readable parameter names.  Defaults to ["p0","p1",...].

    verbose : bool
        Print the full report.  Default True.

    Returns
    -------
    result : dict with keys:
        "survived_full"       : list of (index, name) surviving full-grid check
        "eliminated_full"     : list of (index, name) eliminated in full-grid check
        "survived_single"     : list of (index, name) surviving single-point check
        "eliminated_single"   : list of (index, name) eliminated in single-point check
        "nlp_primal_names"    : ASL primal names from full-grid check
        "param_var_names"     : Pyomo string names of parameter Vars
        "all_survived"        : bool — True if all params survived BOTH checks
        "ift_ready"           : bool — True only if both checks RAN cleanly and
                                nothing was eliminated (False if errored)
        "errored"             : bool — True if either check failed to build/compile
        "errored_full"        : bool — full-grid check raised
        "errored_single"      : bool — single-point check raised
        "error"               : str | None — first error message, if any
    """
    # ── Import PyomoNLP ───────────────────────────────────────────────────────
    try:
        from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    except ImportError:
        raise ImportError(
            "PyomoNLP (pynumero_ASL) is required for this diagnostic.\n"
            "Install with:  pip install pyomo[pynumero]\n"
            "and ensure the ASL shared library is available."
        )
    import pyomo.environ as pyo

    n_mp = len(model_parameters)

    if param_names is None:
        param_names = [f"p{i}" for i in range(n_mp)]

    if len(param_names) != n_mp:
        raise ValueError(
            f"param_names has {len(param_names)} entries but "
            f"model_parameters has {n_mp}."
        )

    spt_full   = np.asarray(sampling_times, dtype=float).flatten()
    spt_full   = spt_full[np.isfinite(spt_full)]
    # Single-point: use the midpoint of the supplied grid
    mid_idx    = len(spt_full) // 2
    spt_single = np.array([float(spt_full[mid_idx])])

    if verbose:
        _banner("ASL Elimination Diagnostic")
        print(f"  Model builder   : {build_pyomo_model_fn.__name__}")
        print(f"  ti_controls     : {list(ti_controls)}")
        print(f"  model_parameters: {list(model_parameters)}")
        print(f"  n_mp            : {n_mp}")
        print(f"  Full-grid spt   : {list(spt_full)}  ({len(spt_full)} points)")
        print(f"  Single-point spt: {list(spt_single)}  (causal rebuild proxy)")
        print()

    # ── CHECK 1: full-grid ────────────────────────────────────────────────────
    if verbose:
        print("Check 1: Full-grid model (user-supplied sampling_times)...")
    try:
        (surv_full, elim_full,
         nlp_names_full, pvar_names, n_all_vars) = _check_survival_robust(
            build_pyomo_model_fn, ti_controls, model_parameters,
            spt_full, PyomoNLP, pyo
        )
        full_ok    = len(elim_full) == 0
        full_error = None
    except Exception as e:
        # Model build / PyomoNLP compile failed.  This is NOT elimination — we
        # simply could not verify.  Do NOT fabricate eliminated entries (the
        # old behaviour reported every parameter as eliminated with fake 'pN'
        # names, masking the real exception).  Leave eliminated empty and record
        # the error so the caller can distinguish "unverified" from "bad".
        surv_full = []; elim_full = []
        nlp_names_full = []; pvar_names = [f"p{i}" for i in range(n_mp)]
        n_all_vars = 0; full_ok = False; full_error = e

    if verbose:
        if full_error:
            print(f"  ✗  Full-grid check FAILED (model build error): {full_error}")
        elif full_ok:
            print(f"  ✓  All {n_mp} parameters survived  "
                  f"(all_vars={n_all_vars}, ASL NLP={len(nlp_names_full)} vars)")
            for i, _, pname, nlp_name, asl_pos in surv_full:
                print(f"     [{i}] {param_names[i]:>12s}  "
                      f"Pyomo: {pname:<28s}  ASL pos {asl_pos} (irrelevant)")
        else:
            print(f"  ✗  {len(elim_full)}/{n_mp} parameters eliminated:")
            for i, _, pname in elim_full:
                print(f"     [{i}] {param_names[i]:>12s}  Pyomo: {pname}")
        print()

    # ── CHECK 2: single-point (causal rebuild proxy) ──────────────────────────
    if verbose:
        print(f"Check 2: Single-point model (spt={spt_single[0]}) "
              f"— mimicking pydex causal rebuild...")
    try:
        (surv_single, elim_single,
         nlp_names_single, _, n_all_vars_s) = _check_survival_robust(
            build_pyomo_model_fn, ti_controls, model_parameters,
            spt_single, PyomoNLP, pyo
        )
        single_ok    = len(elim_single) == 0
        single_error = None
    except Exception as e:
        # As above: a build failure here is "unverified", not elimination.
        surv_single = []; elim_single = []
        nlp_names_single = []; single_ok = False; single_error = e

    if verbose:
        if single_error:
            print(f"  ✗  Single-point check FAILED (model build error): {single_error}")
        elif single_ok:
            print(f"  ✓  All {n_mp} parameters survived  "
                  f"(all_vars={n_all_vars_s}, ASL NLP={len(nlp_names_single)} vars)")
            for i, _, pname, nlp_name, asl_pos in surv_single:
                print(f"     [{i}] {param_names[i]:>12s}  "
                      f"Pyomo: {pname:<28s}  ASL pos {asl_pos} (irrelevant)")
        else:
            print(f"  ✗  {len(elim_single)}/{n_mp} parameters eliminated:")
            for i, _, pname in elim_single:
                print(f"     [{i}] {param_names[i]:>12s}  Pyomo: {pname}")
        print()

    # ── Overall result ────────────────────────────────────────────────────────
    # Three distinct states, kept separate so a build/compile failure is never
    # silently reported as parameter elimination:
    #   • errored  : a check could not run (model build / PyomoNLP failure)
    #   • eliminated: a check ran and a parameter was genuinely absent from the
    #                 ASL primal vector (true elimination)
    #   • ift_ready: both checks ran cleanly AND nothing was eliminated
    errored          = (full_error is not None) or (single_error is not None)
    truly_eliminated = bool(elim_full) or bool(elim_single)
    all_ok           = (not errored) and (not truly_eliminated)

    if verbose:
        _banner("Diagnosis")
        if errored:
            print(
                "  ⚠  Could not verify — a check failed to build/compile.\n"
                "  This is NOT an elimination verdict; the model could not be\n"
                "  evaluated at the supplied test point (e.g. a degenerate\n"
                "  boundary condition such as t=0 producing an all-fixed model).\n"
            )
            if full_error is not None:
                print(f"  Full-grid check error   : "
                      f"{type(full_error).__name__}: {full_error}")
            if single_error is not None:
                print(f"  Single-point check error: "
                      f"{type(single_error).__name__}: {single_error}")
            print()
        elif all_ok:
            print(
                "  ✓  Both checks passed — model is correctly structured for\n"
                "  pydex IFT sensitivities.\n"
                "\n"
                "  Full-grid check   : all parameters survive the initial model\n"
                "  build pydex uses before the causal loop.\n"
                "\n"
                "  Single-point check: all parameters survive the per-spt causal\n"
                "  rebuild pydex uses inside eval_sensitivities().\n"
            )
        else:
            failed_full   = [param_names[i] for i, *_ in elim_full]
            failed_single = [param_names[i] for i, *_ in elim_single]
            if failed_full:
                print(
                    f"  ✗  Full-grid failure: {failed_full} eliminated.\n"
                    "\n"
                    "  These parameters appear only in constraints whose RHS is\n"
                    "  fully determined by fixed quantities — no path to any free\n"
                    "  variable.  Fix: ensure each parameter appears directly in at\n"
                    "  least one constraint involving a collocation state variable\n"
                    "  (ca[t], cb[t], etc.).\n"
                )
            if failed_single and not failed_full:
                print(
                    f"  ✗  Single-point failure: {failed_single} eliminated in\n"
                    "  the causal rebuild but survived the full-grid check.\n"
                    "\n"
                    "  The reduced single-point collocation grid exposes a chain\n"
                    "  that ASL can collapse at single-point scale.  This will\n"
                    "  cause RuntimeError inside pydex's causal per-spt loop.\n"
                    "\n"
                    "  Fix: introduce a free (unfixed) scalar Var for temperature\n"
                    "  or another control that appears multiplicatively with each\n"
                    "  affected parameter, e.g.:\n"
                    "      m.temp     = pyo.Var(initialize=T_val)  # free\n"
                    "      m.temp_fix = pyo.Constraint(expr=m.temp == T_val)\n"
                    "  This breaks the singleton chain and keeps the parameter\n"
                    "  live in both full and single-point grids.\n"
                )
            elif failed_single and failed_full:
                print(
                    f"  ✗  Both checks failed: {failed_full} (full), "
                    f"{failed_single} (single-point).\n"
                    "  Fix the full-grid failure first (see above).\n"
                )

        # Full ASL primal list from full-grid check
        if nlp_names_full:
            _banner("Full ASL primal variable list — full-grid check (first 40)")
            n_show = min(40, len(nlp_names_full))
            for j, name in enumerate(nlp_names_full[:n_show]):
                is_param = any(
                    name == pn or name.endswith("." + pn) or pn.endswith("." + name)
                    for _, _, pn, _, _ in surv_full
                )
                tag = "  ← parameter" if is_param else ""
                print(f"  [{j:4d}] {name}{tag}")
            if len(nlp_names_full) > n_show:
                print(f"  ... ({len(nlp_names_full) - n_show} more) ...")
            print()

    # Clean up names for return dict
    surv_full_out   = [(i, param_names[i]) for i, *_ in surv_full]
    elim_full_out   = [(i, param_names[i]) for i, *_ in elim_full]
    surv_single_out = [(i, param_names[i]) for i, *_ in surv_single]
    elim_single_out = [(i, param_names[i]) for i, *_ in elim_single]

    # Surface the first error message (if any) for the caller.
    _err = full_error if full_error is not None else single_error
    _err_msg = (f"{type(_err).__name__}: {_err}") if _err is not None else None

    return {
        "survived_full"     : surv_full_out,
        "eliminated_full"   : elim_full_out,
        "survived_single"   : surv_single_out,
        "eliminated_single" : elim_single_out,
        "nlp_primal_names"  : nlp_names_full,
        "param_var_names"   : pvar_names,
        "all_survived"      : all_ok,
        "ift_ready"         : all_ok,
        # ── Error reporting (NEW) — distinguishes "unverified" from "bad" ──
        "errored"           : errored,
        "errored_full"      : full_error is not None,
        "errored_single"    : single_error is not None,
        "error"             : _err_msg,
    }


def _banner(title, width=80):
    print()
    print(f" {title} ".center(width, "─"))


# =============================================================================
# Quick self-test when run directly
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("Running diagnostic on case_3_ift_model.build_pyomo_model...")
    print()

    try:
        from case_3_ift_model import build_pyomo_model
    except ImportError:
        print("case_3_ift_model not found — place this file in the same "
              "directory as case_3_ift_model.py and re-run.")
        sys.exit(1)

    result = diagnose_asl_elimination(
        build_pyomo_model,
        ti_controls      = [10.0, 303.15, 10.0],
        model_parameters = [5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5],
        sampling_times   = [0.001, 0.5, 1.0],
        param_names      = [
            "θ_10", "θ_11",
            "θ_20", "θ_21",
            "θ_30", "θ_31",
            "ν", "α", "β",
        ],
    )

    if result["ift_ready"]:
        print("✓  Model is IFT-ready: all parameters survived both ASL checks.")
        sys.exit(0)
    else:
        n_full   = len(result["eliminated_full"])
        n_single = len(result["eliminated_single"])
        print(f"✗  Failures: {n_full} full-grid, {n_single} single-point.")
        sys.exit(1)
