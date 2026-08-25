"""
Regression tests for apportion()'s rounding-EFFICIENCY reporting.

THE DEFECT THIS GUARDS (0.4.1 defect 3)
---------------------------------------
The efficiency block assigned `efficiency` inside a four-way if/elif chain
(d_opt, ds_opt, a_opt, e_opt) with NO else, then called
`np.squeeze(efficiency)` unconditionally. For the other 11 public criteria
nothing ever assigned the name, so apportion() raised

    UnboundLocalError: cannot access local variable 'efficiency'

AFTER printing the whole per-candidate protocol and the Kiefer bound -- i.e.
it did all the work, then threw the result away. A relative-efficiency RATIO
is genuinely undefined for those criteria, so the fix does not invent one:
the four report as before, the rest state that efficiency is not reported and
why, and all of them still print the Kiefer bound.

WHY THIS FILE EXISTS / A CORRECTED PREMISE
------------------------------------------
PROJECT_NOTES.md Open Item 18 records this defect as "verified by execution
only -- it needs a solved design, so it cannot go in the solver-free tests/
tier", and proposes adding a capability-suite section instead. That premise
is WRONG, and this file is the demonstration: apportion() reads
`self.efforts`, the candidate arrays and `self._current_criterion`, and calls
the criterion by name via getattr. Every one of those can be populated by
hand exactly as tests/test_optimal_candidates_report.py already does for
defects 1/2/4, and the criterion itself can be stubbed with an instance
attribute (which shadows the class method). No solver, no simulate(), no FIM.

Confirmed by reconstructing the pre-fix designer.py: the
`test_*_reports_unavailable_*` cases below raise UnboundLocalError against it
at designer.py's `efficiency = np.squeeze(efficiency)`, and pass against the
fix. So this defect CAN be guarded in the fast tier, which is strictly better
than a capability-suite section -- the suite is not run by CI.

Note the entire efficiency block sits inside `if self._verbose >= 1:`, so
these tests must set _verbose >= 1 or they would exercise nothing at all.

SIGN CONVENTIONS
----------------
`self._criterion_value` is the OPTIMISER's objective value while
`criterion(efforts)` is the minimised criterion, and the two differ in sign
per criterion. Verified against real IPOPT solves of the capability suite's
batch-reactor model: section 04 reports an A-optimal value of -1.0622
(negative) and section 05 an E-optimal value of 1.1130 (positive), and a
static D-optimal solve reports +13.8155 (a positive log-det). The fixtures
below use values consistent with those observations, and each assertion
recomputes the documented formula rather than pinning a hard-coded
percentage.

Run with:  python -m pytest test_apportion_efficiency_report.py -v
"""
import sys
import types

import numpy as np
import pytest


def _stub_pydex():
    if "pydex" in sys.modules:
        return
    for name in ("pydex", "pydex.utils", "pydex.core"):
        sys.modules[name] = types.ModuleType(name)
    tp = types.ModuleType("pydex.utils.trellis_plotter")
    lg = types.ModuleType("pydex.core.logger")
    tp.TrellisPlotter = type("TrellisPlotter", (), {})
    lg.Logger = type("Logger", (), {})
    sys.modules["pydex.utils.trellis_plotter"] = tp
    sys.modules["pydex.core.logger"] = lg


_stub_pydex()
from designer import Designer  # noqa: E402

# Criteria with NO standard relative-efficiency definition -- exactly the ones
# that raised UnboundLocalError pre-fix. The public criterion list minus the
# four the if/elif chain handles. Enumerated against
# `grep "_opt_criterion\|_criterion" designer.py` rather than trusted from
# prose, per PROJECT_NOTES.md's warning about criteria lists.
#
# The 15 public criteria, enumerated from source:
#   a_opt, ag_opt, ai_opt, b_opt, cvar_d_opt, d_opt, dg_opt, di_opt, ds_opt,
#   e_opt, eg_opt, ei_opt, u_opt, v_opt, vdi_criterion
# Minus the four handled by the if/elif chain (d_opt, ds_opt, a_opt, e_opt)
# leaves the 11 below -- matching 0.4.1's "raised UnboundLocalError for 11 of
# the 15 public criteria" exactly. A first draft of this list had only 9,
# omitting b_opt and cvar_d_opt; the count is the check that caught it.
_NO_RATIO = [
    "ag_opt_criterion",
    "ai_opt_criterion",
    "b_opt_criterion",
    "cvar_d_opt_criterion",
    "dg_opt_criterion",
    "di_opt_criterion",
    "eg_opt_criterion",
    "ei_opt_criterion",
    "u_opt_criterion",
    "v_opt_criterion",
    "vdi_criterion",
]

_RATIO = ["d_opt_criterion", "ds_opt_criterion", "a_opt_criterion", "e_opt_criterion"]


def _designer(criterion, criterion_value, rounded_value, n_mp=2, interest=None):
    """
    A designer positioned exactly where apportion() picks up: a solved
    STATIC design supported on candidates 1 and 4 of 4, with the criterion
    stubbed to return `rounded_value` for the rounded effort vector.

    Static (_dynamic_system=False) deliberately: it takes the shortest path
    through apportion()'s reporting so the test targets the efficiency block
    rather than the sampling-time report branches, which
    test_optimal_candidates_report.py already covers.
    """
    d = Designer()
    d._dynamic_system = False
    d._invariant_controls = True
    d._dynamic_controls = False
    d._opt_sampling_times = False
    d._specified_n_spt = False
    d._pseudo_bayesian = False
    d._regularize_fim = False
    d._prior_fim = None
    d._cvar_problem = False
    d._verbose = 1          # REQUIRED: the efficiency block is inside this gate
    d._save_atomics = False
    d.n_c = 4
    d.n_spt = 1
    d.n_mp = n_mp
    d.n_opt_c = 2
    d.ti_controls_names = np.array(["x"])
    d.ti_controls_candidates = np.array([[0.0], [1.0], [2.0], [3.0]])
    d.tv_controls_candidates = np.empty((4, 1))
    d.sampling_times_candidates = np.zeros((4, 1))
    d.efforts = np.array([[0.5], [0.0], [0.0], [0.5]])
    d._current_criterion = criterion
    d._criterion_value = criterion_value
    if interest is not None:
        d.model_parameter_names = np.array(["a", "b"])
        d.interest_parameters = interest
    # instance attribute shadows the class method -> criterion is stubbed
    setattr(d, criterion, lambda efforts: rounded_value)
    return d


# ==================================================== the 11 without a ratio
@pytest.mark.parametrize("criterion", _NO_RATIO)
def test_no_ratio_criterion_does_not_raise(criterion, capsys):
    """
    THE regression case. Pre-fix every one of these raised
    UnboundLocalError from inside apportion(), after having already printed
    the protocol.
    """
    d = _designer(criterion, criterion_value=2.0, rounded_value=2.5)
    app = d.apportion(n_exp=10)
    assert app.sum() == 10


@pytest.mark.parametrize("criterion", _NO_RATIO)
def test_no_ratio_criterion_says_so_explicitly(criterion, capsys):
    """The fix must EXPLAIN the absence, not silently omit the line -- a
    missing line is indistinguishable from a reporting bug."""
    d = _designer(criterion, criterion_value=2.0, rounded_value=2.5)
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    assert "not" in out and "relative-efficiency definition" in out
    assert criterion in out


@pytest.mark.parametrize("criterion", _NO_RATIO)
def test_kiefer_bound_still_reported_without_a_ratio(criterion, capsys):
    """The Kiefer/rounding bound is well defined for EVERY criterion and must
    survive the absence of a ratio. Pre-fix it was printed but the crash
    immediately followed, so a caller never got to use it."""
    d = _designer(criterion, criterion_value=2.0, rounded_value=2.5)
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    assert "guaranteed to be at least" in out


# ======================================================= the four with a ratio
def test_d_opt_efficiency_matches_documented_formula(capsys):
    """exp((1/n_mp) * (-rounded - continuous)); n_mp = 2 here."""
    cv, rv, n_mp = 13.8155, -13.5, 2
    d = _designer("d_opt_criterion", cv, rv, n_mp=n_mp)
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    expected = np.exp(1 / n_mp * (-rv - cv)) * 100
    assert f"{expected:.2f}%" in out
    assert expected < 100.0, "fixture sanity: rounded design should be worse"


def test_ds_opt_efficiency_uses_n_interest_not_n_mp(capsys):
    """
    Ds-optimality is D-optimality on the interest subspace, so the exponent
    is 1/n_s (ONE interest parameter here), not 1/n_mp (two). Using n_mp
    would give a visibly different percentage, so this pins the distinction.
    """
    cv, rv = 13.8155, -13.5
    d = _designer("ds_opt_criterion", cv, rv, n_mp=2, interest=["a"])
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    expected_ns = np.exp(1 / 1 * (-rv - cv)) * 100      # correct: n_s = 1
    expected_nmp = np.exp(1 / 2 * (-rv - cv)) * 100     # wrong: n_mp = 2
    assert f"{expected_ns:.2f}%" in out
    assert f"{expected_nmp:.2f}%" not in out


def test_a_opt_efficiency_matches_documented_formula(capsys):
    """-continuous / rounded. Section 04 of the capability suite reports a
    NEGATIVE A-optimal criterion value, hence cv < 0 here."""
    cv, rv = -1.0622, 1.30
    d = _designer("a_opt_criterion", cv, rv)
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    expected = (-cv / rv) * 100
    assert f"{expected:.2f}%" in out
    assert expected < 100.0


def test_e_opt_efficiency_matches_documented_formula(capsys):
    """-rounded / continuous. Section 05 reports a POSITIVE E-optimal value."""
    cv, rv = 1.1130, -1.00
    d = _designer("e_opt_criterion", cv, rv)
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    expected = (-rv / cv) * 100
    assert f"{expected:.2f}%" in out
    assert expected < 100.0


@pytest.mark.parametrize("criterion", _RATIO)
def test_ratio_criteria_do_not_print_the_unavailable_message(criterion, capsys):
    """The two branches must be mutually exclusive: a criterion WITH a ratio
    must never also claim it has none."""
    interest = ["a"] if criterion == "ds_opt_criterion" else None
    cv = -1.0622 if criterion == "a_opt_criterion" else 1.1130
    rv = 1.30 if criterion == "a_opt_criterion" else -1.00
    d = _designer(criterion, cv, rv, interest=interest)
    d.apportion(n_exp=10)
    out = capsys.readouterr().out
    assert "relative-efficiency definition" not in out


# =========================================================== behaviour guards
def test_apportionment_itself_is_unaffected_by_reporting(capsys):
    """
    0.4.1 recorded the defect as "reporting only -- the apportionment itself
    was always correct". Pin that: the returned run counts must be the same
    whether or not the criterion has a ratio.
    """
    d_ratio = _designer("a_opt_criterion", -1.0622, 1.30)
    d_no_ratio = _designer("ai_opt_criterion", 2.0, 2.5)
    assert np.array_equal(d_ratio.apportion(n_exp=10),
                          d_no_ratio.apportion(n_exp=10))


def test_compute_actual_efficiency_false_skips_the_block(capsys):
    """With the ratio switched off, neither message should appear, and no
    criterion call should be needed at all."""
    def _boom(efforts):
        raise AssertionError("criterion must not be called")

    d = _designer("ai_opt_criterion", 2.0, 2.5)
    d.ai_opt_criterion = _boom
    d.apportion(n_exp=10, compute_actual_efficiency=False)
    out = capsys.readouterr().out
    assert "relative-efficiency definition" not in out
    assert "as informative as" not in out
    assert "guaranteed to be at least" in out


# ============================================================ the verbose gate
# Worth pinning explicitly, because it is part of WHY defect 3 survived so
# long and it is easy to break by accident. The ENTIRE reporting block --
# protocol, Kiefer bound, efficiency, and the criterion call that feeds it --
# lives inside `if self._verbose >= 1:`. Confirmed against real IPOPT solves:
# at the default verbose=0, apportion() across nine criteria printed nothing
# and never evaluated the rounded criterion at all, so the pre-fix
# UnboundLocalError could not fire either. Anyone reproducing defect 3 (or
# writing the capability-suite section Open Item 18 proposes) must raise
# verbose or the test silently exercises nothing -- the same "a test that can
# silently skip is worse than no test" trap as sections 54/55.
def test_verbose_zero_prints_nothing_and_skips_criterion_call(capsys):
    def _boom(efforts):
        raise AssertionError("criterion must not be called at verbose=0")

    d = _designer("ai_opt_criterion", 2.0, 2.5)
    d._verbose = 0
    d.ai_opt_criterion = _boom
    app = d.apportion(n_exp=10)
    out = capsys.readouterr().out
    assert out == "", "no reporting is expected at verbose=0"
    assert app.sum() == 10, "apportionment must still be computed and returned"


def test_verbose_zero_still_apportions_identically(capsys):
    """The gate must affect REPORTING only, never the numbers."""
    quiet = _designer("ai_opt_criterion", 2.0, 2.5)
    quiet._verbose = 0
    loud = _designer("ai_opt_criterion", 2.0, 2.5)
    assert np.array_equal(quiet.apportion(n_exp=10), loud.apportion(n_exp=10))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
