"""
Tests for ``designer.design_i_optimal()``.

Only the guards and the reporting contract are here: the design itself needs
an NLP solver, so it is covered by section 62 of the capability suite, where
it is checked against the response-surface result that an I-optimal design
moves effort to the centre of the region while D-optimal pushes to the
extremes.

What is pinned here is the refusal. I-optimality is V-optimality with the
prediction locations weighted, so without ``dw_weights`` the criterion would
be the TOTAL prediction variance over whatever happens to be in ``dw_tic``,
reported as though it were an average. Returning that silently is the
failure this guard exists to prevent.
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


def test_refuses_without_a_prediction_region():
    with pytest.raises(SyntaxError, match="set_prediction_region"):
        _designer().design_i_optimal()


def test_refuses_when_dw_is_set_but_unweighted():
    """dw_tic alone is a V-optimal setup, not an I-optimal one.

    This is the case the guard is really for: everything looks ready, and
    without the check the criterion would be a sum reported as an average.
    """
    d = _designer()
    d.dw_tic = np.array([[0.0, 0.0], [0.5, 0.5]])
    d.dw_spt = np.array([0.0])
    with pytest.raises(SyntaxError, match="set_prediction_region"):
        d.design_i_optimal()


def test_the_refusal_names_the_alternative():
    """Someone who wanted the total should be told which method gives it."""
    with pytest.raises(SyntaxError, match="design_v_optimal"):
        _designer().design_i_optimal()


def test_i_optimal_result_starts_as_none():
    assert _designer().i_optimal_result is None


def test_a_region_alone_does_not_populate_the_result():
    """Only a solve fills it, so a stale region cannot look like a design."""
    d = _designer()
    d.set_prediction_region(bounds=[(-1, 1), (-1, 1)], nodes=3)
    assert d.i_optimal_result is None
    assert d.dw_weights is not None          # the region IS ready


def test_the_method_exists_and_is_public():
    assert callable(getattr(Designer, "design_i_optimal", None))


def test_there_is_no_i_opt_criterion():
    """I-optimality has no formula of its own, and the API should not imply one.

    It is V-optimality with weighted locations. A separate criterion would be
    a second copy of one formula, which the project's one-formula-per-criterion
    contract exists to prevent.
    """
    assert not hasattr(Designer, "i_opt_criterion")
