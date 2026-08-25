"""
Regression tests for the optimal-design REPORT: print_optimal_candidates()'s
fixed-grid branch, and the tabular get_optimal_candidates_table().

Deliberately SOLVER-FREE. optimal_candidates is populated by hand in exactly
the shape get_optimal_candidates() produces, so no design is solved and the
whole file runs in milliseconds.

Two defects are guarded here. Both were live in the fixed-grid branch
(optimize_sampling_times=False) and both were found by diffing the report
against the new table on a real solve, NOT by any assertion:

  1. The branch printed ``self.sampling_times_candidates[i]`` where ``i`` is
     the enumerate counter over optimal_candidates -- i.e. the POSITION in the
     supported list, not the candidate index. Whenever the supported
     candidates were not the first N of the pool it printed a DIFFERENT
     candidate's sampling grid under the right candidate's heading.

  2. The branch printed the candidate's entire grid regardless of effort,
     implying every grid time was part of the protocol. It is not: the FIM
     depends on how effort is distributed across sampling times even when
     optimize_sampling_times is False (verified by construction -- moving a
     candidate's effort onto a zero-effort time changes log-det(FIM)
     drastically), so a time at zero effort is genuinely not in the design.

Both assertions were confirmed to FAIL against the pre-fix designer.py.
"""
import sys
import types

import numpy as np
import pytest


def _stub_pydex_modules():
    """
    designer.py imports pydex.utils.trellis_plotter and pydex.core.logger at
    module level purely for plotting/logging helpers these tests never
    exercise. Stub them so the tests don't require the full pydex package
    (and its plotting stack) to be installed.
    """
    if "pydex" in sys.modules:
        return
    pydex_pkg = types.ModuleType("pydex")
    pydex_utils = types.ModuleType("pydex.utils")
    pydex_core = types.ModuleType("pydex.core")
    trellis_mod = types.ModuleType("pydex.utils.trellis_plotter")
    logger_mod = types.ModuleType("pydex.core.logger")

    class TrellisPlotter:
        pass

    class Logger:
        pass

    trellis_mod.TrellisPlotter = TrellisPlotter
    logger_mod.Logger = Logger

    sys.modules["pydex"] = pydex_pkg
    sys.modules["pydex.utils"] = pydex_utils
    sys.modules["pydex.core"] = pydex_core
    sys.modules["pydex.utils.trellis_plotter"] = trellis_mod
    sys.modules["pydex.core.logger"] = logger_mod


_stub_pydex_modules()
from designer import Designer  # noqa: E402  (import after stubbing)


# Per-candidate DISTINCT grids, and the supported candidates are the LAST two
# of four -- so indexing the pool by list position (0, 1) instead of by
# candidate index (2, 3) yields a grid that is not merely mis-ordered but
# entirely disjoint from the correct one. That disjointness is what makes the
# assertions below unambiguous.
# NOTE the two-significant-decimal values are deliberate. numpy renders
# np.array([0.10, 0.20]) as "[0.1 0.2]", so asserting on "0.10" would never
# fire against the broken code and the test would be VACUOUS -- confirmed by
# running it against the pre-fix designer.py, where it passed. Values like
# 0.11 render identically ("0.11") whether printed by numpy or by the
# formatted report, so the assertions can actually fail.
_POOL_GRID = np.array([
    [0.11, 0.22, 0.33],   # candidate 0  <- what the buggy line printed
    [0.44, 0.55, 0.66],   # candidate 1  <- what the buggy line printed
    [0.77, 0.88, 0.99],   # candidate 2  <- correct for the 1st supported
    [1.11, 1.22, 1.33],   # candidate 3  <- correct for the 2nd supported
])


def _designer_fixed_grid():
    """A designer mid-report: dynamic, ti-controls, sampling times NOT optimised."""
    d = Designer()
    d._dynamic_system = True
    d._invariant_controls = True
    d._dynamic_controls = False
    d._opt_sampling_times = False
    d._specified_n_spt = False
    d._pseudo_bayesian = False
    d._regularize_fim = False
    d._prior_fim = None
    d._cvar_problem = False
    d.n_spt = 3
    d.n_c = 4
    d.sampling_times_candidates = _POOL_GRID
    d.ti_controls_names = np.array(["CA0"])
    d.ti_controls_candidates = np.array([[1.0], [2.0], [5.0], [10.0]])
    d.n_opt_c = 2
    # Shape matches get_optimal_candidates(): [idx, tic, tvc, spt, eff, ...].
    # Candidate 2 uses only its FIRST grid time; candidate 3 only its LAST.
    d.optimal_candidates = [
        [2, np.array([5.0]), np.array([]), _POOL_GRID[2],
         np.array([0.5, 0.0, 0.0]), [[0, 1, 2]], []],
        [3, np.array([10.0]), np.array([]), _POOL_GRID[3],
         np.array([0.0, 0.0, 0.5]), [[0, 1, 2]], []],
    ]
    return d


def test_fixed_grid_prints_the_candidates_own_grid(capsys):
    """Defect 1: the grid printed must belong to the candidate in the heading."""
    d = _designer_fixed_grid()
    d.print_optimal_candidates()
    out = capsys.readouterr().out

    # The two supported candidates' OWN times, which carry effort.
    assert "0.77" in out, "candidate 2's own grid time is missing"
    assert "1.33" in out, "candidate 3's own grid time is missing"

    # Times belonging to the UNSUPPORTED candidates 0 and 1. The pre-fix code
    # printed candidate 0's and candidate 1's grids here, so these strings
    # appearing is precisely the bug.
    for wrong in ("0.11", "0.22", "0.33", "0.44", "0.55", "0.66"):
        assert wrong not in out, (
            f"printed {wrong}, which belongs to an unsupported candidate -- "
            f"sampling_times_candidates is being indexed by list position "
            f"instead of by candidate index"
        )


def test_fixed_grid_omits_zero_effort_times(capsys):
    """Defect 2: grid times carrying no effort are not part of the protocol."""
    d = _designer_fixed_grid()
    d.print_optimal_candidates()
    out = capsys.readouterr().out

    # Zero-effort times from the SUPPORTED candidates' own grids.
    for unused in ("0.88", "0.99", "1.11", "1.22"):
        assert unused not in out, (
            f"printed {unused}, a grid time carrying zero effort, as though "
            f"it were part of the recommended design"
        )
    # And it says so rather than silently dropping them.
    assert "carry no effort" in out


def test_table_matches_the_printed_report():
    """The table and the report must not disagree about the same design."""
    d = _designer_fixed_grid()
    df = d.get_optimal_candidates_table()

    assert list(df["Experiment"]) == [1, 2]
    assert list(df["Candidate"]) == [3, 4]          # 1-indexed pool position
    assert list(df["CA0"]) == [5.0, 10.0]           # ti_controls BY NAME
    assert df["Sampling Time"].iloc[0] == [0.77]    # only the effort-carrying
    assert df["Sampling Time"].iloc[1] == [1.33]    # time, per candidate
    assert np.isclose(df["Effort"].sum(), 1.0)
    # Schedule is meaningless when n_spt is not fixed, so it must be absent
    # entirely rather than present-and-blank.
    assert "Schedule" not in df.columns


def test_table_static_system_has_no_sampling_time_column():
    """
    A static system has no time axis, so the column must be absent. This also
    pins that the table never reads opt_cand[3] for a static system -- that
    field reads back as uninitialised memory there (a separate, pre-existing
    defect in get_optimal_candidates), so touching it would be unsafe.
    """
    d = Designer()
    d._dynamic_system = False
    d._invariant_controls = True
    d._dynamic_controls = False
    d._opt_sampling_times = False
    d._specified_n_spt = False
    d.ti_controls_names = np.array(["Flow", "Pressure"])
    d.n_opt_c = 2
    d.optimal_candidates = [
        [0, np.array([10.0, 2.0]), np.array([]), None, np.array([0.4]), [], []],
        [1, np.array([12.0, 3.0]), np.array([]), None, np.array([0.6]), [], []],
    ]
    df = d.get_optimal_candidates_table()

    assert "Sampling Time" not in df.columns
    assert list(df["Experiment"]) == [1, 2]
    assert list(df["Flow"]) == [10.0, 12.0]
    assert list(df["Pressure"]) == [2.0, 3.0]
    assert np.isclose(df["Effort"].sum(), 1.0)


def test_table_schedules_are_separate_mandatory_experiments():
    """
    With a fixed n_spt, each schedule is its OWN row: two schedules on one
    candidate are two separate, mandatory experiments (a required split of
    effort), not alternative ways of running the same one.
    """
    d = Designer()
    d._dynamic_system = True
    d._invariant_controls = True
    d._dynamic_controls = False
    d._opt_sampling_times = True
    d._specified_n_spt = True
    d.ti_controls_names = np.array(["CA0"])
    d.n_opt_c = 1
    d.optimal_candidates = [
        [2, np.array([5.0]), np.array([]),
         [np.array([0.2, 0.5]), np.array([0.3, 0.6])],
         [np.array([0.15, 0.15]), np.array([0.35, 0.35])], [], []],
    ]
    df = d.get_optimal_candidates_table()

    assert len(df) == 2, "each schedule must be its own experiment row"
    assert list(df["Experiment"]) == [1, 2]
    assert list(df["Schedule"]) == [1, 2]
    assert list(df["Candidate"]) == [3, 3], "both rows are the same candidate"
    assert df["Sampling Time"].iloc[0] == [0.2, 0.5]
    assert df["Sampling Time"].iloc[1] == [0.3, 0.6]
    assert np.isclose(df["Effort"].iloc[0], 0.30)
    assert np.isclose(df["Effort"].iloc[1], 0.70)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------
# eval_pim's None contract, honoured by the prediction-variance criteria.
#
# eval_pim sets self.pvars = None when the FIM is not safely invertible, and
# its own comment states the intent: "Setting pvars to None lets the consuming
# criteria report an infeasible design (+inf) instead, which is what an
# optimiser can actually act on."
#
# dg_opt, di_opt and vdi honoured that. ag/ai/eg/ei did NOT -- they iterated
# None and raised TypeError mid-solve, aborting the whole design run instead of
# steering the optimiser away from an infeasible point. The four unguarded ones
# are exactly the four with no docstring, which is plausibly why they were
# missed when dg/di were hardened.
#
# Reproduced from a real design_experiment(eg_opt_criterion) run that died with
#   TypeError: 'NoneType' object is not iterable
# at _eg_opt_criterion. Confirmed to FAIL for ag/ai/eg/ei before the guard.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("criterion", [
    "_dg_opt_criterion",
    "_di_opt_criterion",
    "_ag_opt_criterion",
    "_ai_opt_criterion",
    "_eg_opt_criterion",
    "_ei_opt_criterion",
])
def test_pvar_criteria_return_inf_when_pvars_is_none(criterion, monkeypatch):
    """A None pvars must yield +inf (worst for a minimised criterion), not raise."""
    d = Designer()
    d.n_c = 2
    d.n_spt = 2
    d.n_r = 2
    d._fd_jac = True
    # Stand in for eval_pim: report an un-invertible FIM, exactly as the real
    # one does via _safe_fim_inverse returning None.
    monkeypatch.setattr(d, "eval_pim", lambda efforts, **kw: setattr(d, "pvars", None))
    d.pvars = None

    value = getattr(d, criterion)(np.full((2, 2), 0.25))

    assert value == np.inf, (
        f"{criterion} returned {value!r} for an infeasible design; eval_pim's "
        f"contract is that a None pvars is reported as +inf"
    )
