"""
Tests for ``designer.i_opt_criterion``.

It exists so that every criterion is passed to ``design_experiment()`` the
same way. It forwards to ``v_opt_criterion`` -- I-optimality is not a
different formula, it is V-optimality with the prediction locations weighted
by a quadrature rule.

The part that needs guarding is the DISPATCH. ``_solve_pyomo`` routes by
criterion name, and the substring ``"i_opt"`` appears inside ten existing
criterion names: ai_opt, di_opt, ei_opt, their private forms, and their
pseudo-Bayesian variants. A substring test would misroute every one of them
into the I-optimal branch, so the match is on the exact name. That
enumeration is asserted here rather than left as a claim in a comment.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_flat"))


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


def _simulate(ti_controls, model_parameters):
    x1, x2 = ti_controls
    basis = np.array([1.0, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    return np.array([basis @ model_parameters])


def _designer():
    d = Designer()
    d.simulate = _simulate
    d.model_parameters = np.ones(6)
    d.ti_controls_candidates = d.enumerate_candidates(
        bounds=[[-1, 1], [-1, 1]], levels=[5, 5])
    d.response_names = ["y"]
    d.ti_controls_names = ["x1", "x2"]
    d.initialize(verbose=0)
    return d


BOUNDS = [(-1.0, 1.0), (-1.0, 1.0)]


# ---------------------------------------------------------------------------
# the alias
# ---------------------------------------------------------------------------
def test_the_criterion_exists_and_is_public():
    assert callable(getattr(Designer, "i_opt_criterion", None))


def test_it_returns_exactly_what_v_opt_returns():
    """It is the same formula, so the values must agree bit for bit."""
    d = _designer()
    d.set_prediction_region(bounds=BOUNDS, nodes=3)
    efforts = np.full(25, 1.0 / 25)
    assert d.i_opt_criterion(efforts) == d.v_opt_criterion(efforts)


# ---------------------------------------------------------------------------
# dispatch: the collision this guard exists for
# ---------------------------------------------------------------------------
def test_ten_existing_criteria_contain_the_i_opt_substring():
    """Pinned by enumeration, not assumed.

    If this list ever shrinks to nothing the exact-match dispatch could be
    relaxed; while it does not, a substring test is unsafe.
    """
    names = [n for n in dir(Designer)
             if n.endswith("criterion") and "i_opt" in n]
    private = [n for n in vars(Designer)
               if n.endswith("criterion") and "i_opt" in n
               and n.startswith("_")]
    colliding = set(names + private) - {"i_opt_criterion"}
    assert len(colliding) >= 3, sorted(colliding)
    for expected in ("ai_opt_criterion", "di_opt_criterion",
                     "ei_opt_criterion"):
        assert expected in colliding


@pytest.mark.parametrize("name", ["ai_opt_criterion", "di_opt_criterion",
                                  "ei_opt_criterion"])
def test_colliding_criteria_are_not_treated_as_i_optimal(name):
    """The telling case: these must NOT demand a prediction region.

    The routing happens inside _solve_pyomo, so this has to go through
    design_experiment() -- calling the criterion function directly bypasses
    the dispatch entirely and proves nothing. Measured: a test that called
    the criterion directly passed against a substring-dispatch mutant.

    No region is set, so with a substring dispatch these would hit the
    I-optimal guard and raise for want of dw_weights. Any OTHER failure is
    fine and expected: these criteria may well fail further down, and
    without a solver installed the call cannot complete at all. What is
    asserted is only that the failure is not the I-optimal one.
    """
    d = _designer()
    try:
        d.design_experiment(criterion=getattr(d, name), solver="pounce")
    except Exception as err:
        assert "prediction region" not in str(err), (
            f"{name} was routed into the I-optimal branch: {err}")


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------
def test_design_experiment_refuses_i_opt_without_a_region():
    d = _designer()
    with pytest.raises(SyntaxError, match="set_prediction_region"):
        d.design_experiment(criterion=d.i_opt_criterion, solver="pounce")


def test_the_refusal_names_v_opt_as_the_alternative():
    d = _designer()
    with pytest.raises(SyntaxError, match="v_opt_criterion"):
        d.design_experiment(criterion=d.i_opt_criterion, solver="pounce")


def test_v_opt_itself_is_not_subject_to_the_region_guard():
    """v_opt with points but no weights is legitimate: it is a total."""
    d = _designer()
    d.dw_tic = np.array([[0.0, 0.0], [0.5, 0.5]])
    d.dw_spt = np.array([0.0])
    d._eval_W_matrix()
    efforts = np.full(25, 1.0 / 25)
    d.eval_fim(efforts)
    assert np.isfinite(d.v_opt_criterion(efforts))
