"""
Tests for passing a criterion to ``design_experiment()`` BY NAME, and for the
refusal of a prediction criterion under a scenario set.

Two things are pinned here.

**Strings resolve, callables still work.** ``design_experiment("D-optimal")``
is the ordinary route; a bound method is still accepted, because the suite,
the examples and user-written criteria all rely on it. The dispatch inside
``_solve_pyomo`` routes by ``criterion.__name__``, so the string has to be
resolved to the bound method before anything downstream sees it.

**A prediction criterion under a scenario set raises clearly.** V- and
I-optimality have no pseudo-Bayesian form: they depend on the nominal
parameters through the prediction sensitivities as well as through the FIM,
and averaging information has no clean meaning when the prediction
sensitivities move with the parameters too. Before this, the failure was a
numpy matmul shape mismatch raised from deep inside ``_eval_W_matrix``,
which told the user nothing.
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


def _designer(model_parameters=None):
    d = Designer()
    d.simulate = _simulate
    d.model_parameters = (np.ones(6) if model_parameters is None
                          else model_parameters)
    d.ti_controls_candidates = d.enumerate_candidates(
        bounds=[[-1, 1], [-1, 1]], levels=[5, 5])
    d.response_names = ["y"]
    d.ti_controls_names = ["x1", "x2"]
    d.initialize(verbose=0)
    return d


SCENARIOS = np.random.default_rng(0).normal(1.0, 0.2, size=(8, 6))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spelling", [
    "D-optimal", "d-optimal", "D_optimal", "d_opt", "D opt", "d",
    "D-OPTIMAL", "  d-optimal  ", "d_opt_criterion",
])
def test_spellings_of_d_all_resolve_to_the_same_method(spelling):
    d = _designer()
    assert d._resolve_criterion(spelling) == d.d_opt_criterion


@pytest.mark.parametrize("name, attr", [
    ("D-optimal", "d_opt_criterion"),
    ("Ds-optimal", "ds_opt_criterion"),
    ("A-optimal", "a_opt_criterion"),
    ("E-optimal", "e_opt_criterion"),
    ("V-optimal", "v_opt_criterion"),
    ("I-optimal", "i_opt_criterion"),
    ("B-optimal", "b_opt_criterion"),
])
def test_every_advertised_name_resolves(name, attr):
    d = _designer()
    assert d._resolve_criterion(name) == getattr(d, attr)


def test_ds_is_not_swallowed_by_d():
    """The one pair that could collide under a loose match."""
    d = _designer()
    assert d._resolve_criterion("Ds-optimal") == d.ds_opt_criterion
    assert d._resolve_criterion("D-optimal") == d.d_opt_criterion
    assert d.ds_opt_criterion != d.d_opt_criterion


def test_a_callable_passes_straight_through():
    """Required: the suite, the examples and user criteria all pass these."""
    d = _designer()
    assert d._resolve_criterion(d.a_opt_criterion) == d.a_opt_criterion

    def my_own_criterion(efforts):        # a user-written criterion
        return 0.0
    assert d._resolve_criterion(my_own_criterion) is my_own_criterion


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------
def test_an_unknown_name_raises_and_lists_the_valid_ones():
    d = _designer()
    with pytest.raises(SyntaxError) as err:
        d._resolve_criterion("Q-optimal")
    msg = str(err.value)
    for expected in ("D-optimal", "Ds-optimal", "A-optimal", "E-optimal",
                     "V-optimal", "I-optimal", "B-optimal"):
        assert expected in msg


def test_cvar_is_pointed_at_its_own_entry_point():
    """CVaR is not a design_experiment() criterion; saying so beats a bare refusal."""
    d = _designer()
    with pytest.raises(SyntaxError, match="solve_cvar_problem"):
        d._resolve_criterion("CVaR-D")


def test_cvar_is_not_in_the_name_table():
    assert not any("cvar" in k for k in Designer._CRITERION_NAMES)


@pytest.mark.parametrize("bad", [3, None, ["D-optimal"], 2.5])
def test_a_non_string_non_callable_raises(bad):
    d = _designer()
    with pytest.raises(SyntaxError, match="must be a string"):
        d._resolve_criterion(bad)


# ---------------------------------------------------------------------------
# prediction criteria under a scenario set
# ---------------------------------------------------------------------------
def test_set_prediction_region_refuses_under_scenarios():
    d = _designer(SCENARIOS)
    assert d._pseudo_bayesian is True
    with pytest.raises(NotImplementedError, match="pseudo-Bayesian"):
        d.set_prediction_region(bounds=[(-1, 1), (-1, 1)])


def test_the_refusal_explains_what_to_do_instead():
    d = _designer(SCENARIOS)
    with pytest.raises(NotImplementedError) as err:
        d.set_prediction_region(bounds=[(-1, 1), (-1, 1)])
    msg = str(err.value)
    assert "1-D" in msg                  # design at a single nominal
    assert "D, Ds, A or E" in msg        # or use a calibration criterion


def test_v_optimal_refuses_under_scenarios_without_a_region():
    """V-optimal can be reached without set_prediction_region(), so the
    dispatch branch needs its own guard."""
    d = _designer(SCENARIOS)
    d.dw_tic = np.array([[0.0, 0.0]])
    d.dw_spt = np.array([0.0])
    with pytest.raises(NotImplementedError, match="pseudo-Bayesian"):
        d.design_experiment("V-optimal", solver="pounce")


def test_calibration_criteria_are_unaffected_by_the_guard():
    """D, Ds, A and E all have pseudo-Bayesian forms and must still resolve."""
    d = _designer(SCENARIOS)
    for name in ("D-optimal", "Ds-optimal", "A-optimal", "E-optimal"):
        assert callable(d._resolve_criterion(name))


def test_a_one_dimensional_parameter_vector_is_not_refused():
    d = _designer()
    assert d._pseudo_bayesian is False
    info = d.set_prediction_region(bounds=[(-1, 1), (-1, 1)], nodes=3)
    assert info["n_points"] == 9
