"""
Tests for ``designer.set_prediction_region()`` — resolving a region into the
``dw_tic`` / ``dw_spt`` / ``dw_weights`` triple an I-optimal design needs.

Solver-free: nothing here calls an optimiser. The region machinery is under
test, not the design.

The contract being pinned:

* ``bounds`` is required and is NOT defaulted to the candidate grid -- the
  region you want to predict over is a modelling decision;
* a region wider than the candidate grid is REFUSED, because the candidates
  are the conditions the user has declared they can run. The check is
  inclusive at the ends and marginal (one axis at a time), and
  ``allow_extrapolation`` downgrades it to a warning;
* ``times`` has no default on a dynamic model;
* with ``nodes`` left None the rule refines until the moment matrix stops
  changing, so the node count is measured rather than guessed.
"""
import os
import sys
import types
import warnings

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


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _simulate_quadratic(ti_controls, model_parameters):
    x1, x2 = ti_controls
    basis = np.array([1.0, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    return np.array([basis @ model_parameters])


def _static(levels=(5, 5)):
    d = Designer()
    d.simulate = _simulate_quadratic
    d.model_parameters = np.ones(6)
    d.ti_controls_candidates = d.enumerate_candidates(
        bounds=[[-1, 1], [-1, 1]], levels=list(levels))
    d.response_names = ["y"]
    d.ti_controls_names = ["x1", "x2"]
    d.initialize(verbose=0)
    return d


def _simulate_decay(ti_controls, sampling_times, model_parameters):
    a0 = ti_controls[0]
    t = np.asarray(sampling_times, dtype=float)
    return (a0 * np.exp(-model_parameters[0] * t)).reshape(-1, 1)


def _dynamic():
    d = Designer()
    d.simulate = _simulate_decay
    d.model_parameters = np.array([0.5])
    d.ti_controls_candidates = d.enumerate_candidates(bounds=[[1.0, 5.0]],
                                                      levels=[5])
    d.sampling_times_candidates = np.linspace(0.5, 4.0, 5)
    d.response_names = ["A"]
    d.ti_controls_names = ["A0"]
    d.initialize(verbose=0)
    return d


FULL = [(-1.0, 1.0), (-1.0, 1.0)]


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------
def test_bounds_length_must_match_the_controls():
    with pytest.raises(SyntaxError, match="time-invariant control"):
        _static().set_prediction_region(bounds=[(-1, 1)])


@pytest.mark.parametrize("bad", [(1.0, 1.0), (1.0, -1.0), (np.nan, 1.0)])
def test_invalid_interval_raises(bad):
    with pytest.raises(SyntaxError, match="interval"):
        _static().set_prediction_region(bounds=[bad, (-1, 1)])


# ---------------------------------------------------------------------------
# coverage: the region must sit inside the candidate grid
# ---------------------------------------------------------------------------
def test_region_wider_than_the_candidates_is_refused():
    with pytest.raises(SyntaxError, match="wider than the candidate grid"):
        _static().set_prediction_region(bounds=[(-2, 2), (-1, 1)])


def test_the_refusal_names_the_offending_control_and_both_ranges():
    with pytest.raises(SyntaxError) as err:
        _static().set_prediction_region(bounds=[(-1, 1), (-3, 0.5)])
    msg = str(err.value)
    assert "x2" in msg and "x1" not in msg.split("wider")[1].split("The")[0]
    assert "[-3, 0.5]" in msg and "[-1, 1]" in msg


def test_coverage_is_inclusive_at_the_ends():
    """Bounds equal to the candidate extent must PASS.

    A strict comparison would refuse the most ordinary request there is --
    predict over exactly the region you can run in.
    """
    info = _static().set_prediction_region(bounds=FULL, nodes=3)
    assert info["n_points"] == 9


def test_a_subregion_is_always_fine():
    info = _static().set_prediction_region(bounds=[(-0.5, 0.5), (0.0, 1.0)],
                                           nodes=3)
    assert info["n_points"] == 9


def test_allow_extrapolation_downgrades_to_a_warning():
    d = _static()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d.set_prediction_region(bounds=[(-2, 2), (-1, 1)], nodes=3,
                                allow_extrapolation=True)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert d.dw_tic is not None


# ---------------------------------------------------------------------------
# times
# ---------------------------------------------------------------------------
def test_dynamic_model_requires_times():
    with pytest.raises(SyntaxError, match="times is required"):
        _dynamic().set_prediction_region(bounds=[(1.0, 5.0)])


def test_times_multiply_the_locations():
    d = _dynamic()
    d.set_prediction_region(bounds=[(1.0, 5.0)], nodes=4,
                            times=[1.0, 2.0, 3.0])
    assert d.dw_weights.size == 4 * 3          # points x times
    assert np.asarray(d.W).shape[0] == 4 * 3 * 1   # x responses
    assert float(np.sum(d.dw_weights)) == pytest.approx(3.0)


def test_static_model_ignores_times():
    d = _static()
    d.set_prediction_region(bounds=FULL, nodes=3, times=[1.0, 2.0])
    assert d.dw_weights.size == 9


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def test_weights_are_normalised():
    d = _static()
    d.set_prediction_region(bounds=FULL, nodes=4)
    assert float(np.sum(d.dw_weights)) == pytest.approx(1.0, rel=1e-12)


def test_explicit_points_bypass_generation():
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.25]])
    d = _static()
    info = d.set_prediction_region(bounds=FULL, points=pts)
    assert info["rule"] == "explicit"
    assert info["n_points"] == 3
    assert np.allclose(d.dw_tic, pts)


def test_explicit_weights_are_honoured():
    pts = np.array([[0.0, 0.0], [0.5, 0.5]])
    d = _static()
    d.set_prediction_region(bounds=FULL, points=pts, weights=[3.0, 1.0])
    assert d.dw_weights[0] / d.dw_weights[1] == pytest.approx(3.0)


def test_unknown_rule_raises():
    with pytest.raises(SyntaxError, match="rule must be"):
        _static().set_prediction_region(bounds=FULL, rule="simpson")


def test_sobol_is_reproducible_with_a_fixed_seed():
    a = _static(); a.set_prediction_region(bounds=FULL, rule="sobol", nodes=64)
    b = _static(); b.set_prediction_region(bounds=FULL, rule="sobol", nodes=64)
    assert np.array_equal(a.dw_tic, b.dw_tic)


def test_sobol_points_lie_inside_the_bounds():
    d = _static()
    d.set_prediction_region(bounds=[(-0.5, 0.5), (0.0, 1.0)],
                            rule="sobol", nodes=64)
    assert d.dw_tic[:, 0].min() >= -0.5 and d.dw_tic[:, 0].max() <= 0.5
    assert d.dw_tic[:, 1].min() >= 0.0 and d.dw_tic[:, 1].max() <= 1.0


# ---------------------------------------------------------------------------
# adaptive refinement
# ---------------------------------------------------------------------------
def test_adaptive_converges_and_reports_what_it_used():
    d = _static()
    info = d.set_prediction_region(bounds=FULL)
    assert info["converged"] is True
    assert info["rule"] == "gauss"
    assert info["relative_change"] < 1e-8
    assert info["n_points"] == info["nodes"] ** 2


def test_two_successive_levels_are_required_before_stopping():
    """The accepted rule is the SECOND level under tolerance, not the first.

    On this quadratic the moment matrix is exact from four nodes on, so the
    relative change collapses to ~1e-13 at n=4 and stays there. A rule that
    stopped at the first level under tolerance would accept n=4; requiring
    two successive levels accepts n=5. Asserting the exact value is what
    makes this test discriminate -- a bound like "<= 6" passes either way,
    and a mutant that stops on one level survives it.
    """
    info = _static().set_prediction_region(bounds=FULL)
    assert info["nodes"] == 5, info
    assert info["converged"] is True


def test_fixed_nodes_skip_refinement():
    info = _static().set_prediction_region(bounds=FULL, nodes=3)
    assert info["nodes"] == 3
    assert info["n_points"] == 9
    assert info["relative_change"] is None
    assert info["converged"] is None


def test_an_unreachable_tolerance_warns_but_still_produces_a_region():
    """The convergence target is internal, so this forces it via the class.

    Not a user-facing path: tol and max_nodes are deliberately NOT arguments.
    The behaviour still needs guarding, because a region that failed to
    converge must warn and must still be usable.
    """
    d = _static()
    old_tol, old_max = Designer._REGION_TOL, Designer._REGION_MAX_NODES
    try:
        Designer._REGION_TOL = 1e-30
        Designer._REGION_MAX_NODES = 4
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = d.set_prediction_region(bounds=FULL)
    finally:
        Designer._REGION_TOL, Designer._REGION_MAX_NODES = old_tol, old_max
    assert any("did not converge" in str(w.message) for w in caught)
    assert info["converged"] is False
    assert d.dw_tic is not None and d.dw_weights is not None


def test_tol_and_max_nodes_are_not_arguments():
    """Pinned deliberately: the decision is the library's, not the user's."""
    import inspect
    params = inspect.signature(Designer.set_prediction_region).parameters
    assert "tol" not in params and "max_nodes" not in params


def test_the_region_is_left_ready_for_a_design():
    """dw_tic, dw_spt, dw_weights and W must all be consistent afterwards."""
    d = _static()
    d.set_prediction_region(bounds=FULL, nodes=4)
    n_loc = len(d.dw_tic) * len(np.atleast_1d(d.dw_spt))
    assert d.dw_weights.size == n_loc
    assert np.asarray(d.W).shape == (n_loc * d.n_m_r, d.n_mp)
