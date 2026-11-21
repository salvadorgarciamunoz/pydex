"""
Tests for the two public routes to a V-optimal design agreeing.

``design_v_optimal()`` builds the W matrix before solving.
``design_experiment(criterion=v_opt_criterion)`` did not, and the guard that
noticed sits in ``_solve_pyomo`` BEFORE the criterion function is ever
called -- so the lazy build inside ``_v_opt_criterion`` was unreachable on
that path. The result was a refusal telling the user to assign ``dw_tic``
when ``dw_tic`` was already assigned, on the exact form the README
documented.

Nobody got a wrong answer from it: it refused rather than computing
something subtly off. But a documented call that cannot work is a defect,
and the misdirecting message made it worse.
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


REGION = np.array([[a, b] for a in np.linspace(-1, 1, 3)
                   for b in np.linspace(-1, 1, 3)])


def _criterion_at(d, efforts):
    d.eval_fim(np.asarray(efforts).flatten())
    W = np.asarray(d.W)
    return float(np.trace(W @ np.linalg.inv(d.fim) @ W.T))


# The POSITIVE case -- that design_experiment(criterion=v_opt_criterion) now
# runs to a solved design -- needs an NLP solver and so cannot live here. It
# is checked in section 62 of the capability suite, where it is also asserted
# to give the same design as design_v_optimal(). What is solver-free, and
# what this file covers, is that the refusals survive for the cases that
# genuinely cannot proceed, and that the message no longer misdirects.


def test_missing_dw_tic_still_raises_and_names_it():
    d = _designer()
    d.dw_spt = np.array([0.0])
    with pytest.raises((RuntimeError, SyntaxError), match="dw_tic"):
        d.design_experiment(criterion=d.v_opt_criterion, solver="pounce")


def test_missing_dw_spt_still_raises_and_names_it():
    d = _designer()
    d.dw_tic = REGION
    with pytest.raises((RuntimeError, SyntaxError), match="dw_spt"):
        d.design_experiment(criterion=d.v_opt_criterion, solver="pounce")


def test_the_message_points_at_both_ways_of_setting_a_region():
    """A user who hits this should be told which method to reach for."""
    d = _designer()
    with pytest.raises((RuntimeError, SyntaxError)) as err:
        d.design_experiment(criterion=d.v_opt_criterion, solver="pounce")
    msg = str(err.value)
    assert "find_optimal_operating_point" in msg
    assert "set_prediction_region" in msg


def test_the_old_message_is_gone():
    """The previous text told the user to do what they had already done."""
    d = _designer()
    with pytest.raises((RuntimeError, SyntaxError)) as err:
        d.design_experiment(criterion=d.v_opt_criterion, solver="pounce")
    assert "requires W matrix" not in str(err.value)
