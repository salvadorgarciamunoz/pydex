"""
Tests for ``designer.dw_weights`` — quadrature weights over the prediction
locations used by the V-optimal W matrix.

Solver-free: every assertion is made at a FIXED effort vector, so the whole
file runs in CI without ipopt, bonmin or ASL. The optimiser is not under test
here; the weighting of W is.

The contract being pinned:

* ``dw_weights`` unset leaves W exactly as it was, so V-optimality is
  unchanged (this is the back-compat gate);
* one weight per (dw point, time) PAIR, not per row — responses are the
  components of one observation and stay summed;
* weights are normalised internally, so the criterion becomes a weighted
  AVERAGE and the user supplies a quadrature rule rather than remembering to
  divide;
* a Gauss-Legendre rule integrates the region EXACTLY for a polynomial model,
  which is the property that makes a cheap rule preferable to a fine grid.
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


# ---------------------------------------------------------------------------
# fixtures: a full quadratic in two factors, and a two-response variant
# ---------------------------------------------------------------------------
def _simulate_quadratic(ti_controls, model_parameters):
    x1, x2 = ti_controls
    basis = np.array([1.0, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    return np.array([basis @ model_parameters])


def _simulate_two_response(ti_controls, model_parameters):
    x1, x2 = ti_controls
    basis = np.array([1.0, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    y = basis @ model_parameters
    return np.array([y, 2.0 * y + x1])


def _simulate_decay(ti_controls, sampling_times, model_parameters):
    """A dynamic two-response model, for the (point, time) weighting."""
    k1, k2 = model_parameters[0], model_parameters[1]
    a0 = ti_controls[0]
    t = np.asarray(sampling_times, dtype=float)
    a = a0 * np.exp(-k1 * t)
    b = a0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    return np.column_stack([a, b])


def _dynamic_designer():
    d = Designer()
    d.simulate = _simulate_decay
    d.model_parameters = np.array([0.5, 0.2])
    d.ti_controls_candidates = d.enumerate_candidates(bounds=[[1.0, 5.0]],
                                                      levels=[5])
    d.sampling_times_candidates = np.linspace(0.5, 4.0, 5)
    d.response_names = ["A", "B"]
    d.initialize(verbose=0)
    return d


def _designer(simulate=_simulate_quadratic, response_names=("y",)):
    d = Designer()
    d.simulate = simulate
    d.model_parameters = np.ones(6)
    d.ti_controls_candidates = d.enumerate_candidates(
        bounds=[[-1, 1], [-1, 1]], levels=[5, 5])
    d.response_names = list(response_names)
    d.ti_controls_names = ["x1", "x2"]
    d.initialize(verbose=0)
    return d


def _uniform_region(n):
    g = np.linspace(-1, 1, n)
    return np.array([[a, b] for a in g for b in g])


def _gauss_region(n):
    nodes, w = np.polynomial.legendre.leggauss(n)
    pts = np.array([[a, b] for a in nodes for b in nodes])
    wts = np.array([wa * wb for wa in w for wb in w])
    return pts, wts / wts.sum()


def _W_for(points, weights=None, **kwargs):
    d = _designer(**kwargs)
    d.dw_tic = np.asarray(points)
    d.dw_spt = np.array([0.0])
    if weights is not None:
        d.dw_weights = weights
    d._eval_W_matrix()
    return d


def _criterion(d, efforts):
    d.eval_fim(np.asarray(efforts).flatten())
    return float(np.trace(d.W @ np.linalg.inv(d.fim) @ d.W.T))


UNIFORM_EFFORT = np.full(25, 1.0 / 25)


# ---------------------------------------------------------------------------
# back-compat: unset must change nothing
# ---------------------------------------------------------------------------
def test_unset_dw_weights_leaves_W_bit_identical():
    region = _uniform_region(5)
    untouched = _W_for(region)
    explicit_none = _W_for(region, weights=None)
    assert np.array_equal(untouched.W, explicit_none.W)


def test_default_is_none():
    assert _designer().dw_weights is None


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def test_uniform_weights_divide_the_criterion_by_the_location_count():
    """Equal weights turn the V-optimal SUM into the plain average."""
    region = _uniform_region(5)
    unweighted = _W_for(region)
    weighted = _W_for(region, weights=np.ones(len(region)))
    j_sum = _criterion(unweighted, UNIFORM_EFFORT)
    j_avg = _criterion(weighted, UNIFORM_EFFORT)
    assert j_avg == pytest.approx(j_sum / len(region), rel=1e-12)


def test_weights_are_normalised_so_overall_scale_is_irrelevant():
    region = _uniform_region(5)
    base = np.linspace(0.5, 2.0, len(region))
    a = _W_for(region, weights=base)
    b = _W_for(region, weights=1000.0 * base)
    assert np.allclose(a.W, b.W, rtol=0, atol=0)
    # ... and the weights must actually have been applied, or this passes
    # against an implementation that ignores dw_weights entirely
    plain = _W_for(region)
    assert not np.allclose(a.W, plain.W)


def test_a_zero_weight_removes_that_location_entirely():
    region = _uniform_region(3)
    w = np.ones(len(region))
    w[4] = 0.0                                    # the centre point
    d = _W_for(region, weights=w)
    assert np.all(d.W[4] == 0.0)
    assert not np.all(d.W[0] == 0.0)


# ---------------------------------------------------------------------------
# one weight per LOCATION, not per row
# ---------------------------------------------------------------------------
def test_weights_apply_to_every_response_of_a_location():
    region = _uniform_region(3)
    w = np.arange(1.0, len(region) + 1)
    d = _W_for(region, weights=w,
               simulate=_simulate_two_response, response_names=("y1", "y2"))
    plain = _W_for(region,
                   simulate=_simulate_two_response, response_names=("y1", "y2"))
    scale = np.sqrt(w / w.sum())
    # rows are (location, response) with response fastest
    for loc in range(len(region)):
        for resp in range(2):
            row = loc * 2 + resp
            assert np.allclose(d.W[row], scale[loc] * plain.W[row], rtol=1e-12)


def test_weights_span_times_as_well_as_points():
    """A weight is per (point, time) pair, so the length is r_w * n_spt_dw.

    Uses a DYNAMIC model: a static one has no time axis, and _eval_W_matrix
    collapses dw_spt to a single placeholder for it, so the point could not
    be made there. Asserted on VALUES, not just on W's shape -- a shape
    assertion is true whether or not the weighting exists.
    """
    def build(weights):
        d = _dynamic_designer()
        d.dw_tic = np.array([[2.0], [4.0]])
        d.dw_spt = np.array([1.0, 3.0])          # 2 points x 2 times
        if weights is not None:
            d.dw_weights = weights
        d._eval_W_matrix()
        return d

    plain = build(None)
    assert plain.W.shape[0] == 2 * 2 * 2         # points x times x responses

    w = np.array([1.0, 2.0, 3.0, 4.0])           # one per (point, time)
    weighted = build(w)
    scale = np.sqrt(w / w.sum())
    expected = np.repeat(scale, 2)[:, None] * plain.W   # 2 responses per row
    assert np.allclose(weighted.W, expected, rtol=1e-12, atol=0)


def test_weight_length_must_count_times_not_just_points():
    d = _dynamic_designer()
    d.dw_tic = np.array([[2.0], [4.0]])
    d.dw_spt = np.array([1.0, 3.0])
    d.dw_weights = np.ones(2)                    # should be 2 x 2 = 4
    with pytest.raises(SyntaxError, match="prediction locations"):
        d._eval_W_matrix()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad, reason", [
    (np.ones(3), "wrong length"),
    (np.ones(100), "wrong length the other way"),
])
def test_wrong_length_raises(bad, reason):
    with pytest.raises(SyntaxError, match="dw_weights"):
        _W_for(_uniform_region(5), weights=bad)


def test_negative_weight_raises():
    w = np.ones(25)
    w[0] = -1.0
    with pytest.raises(SyntaxError, match="non-negative"):
        _W_for(_uniform_region(5), weights=w)


def test_all_zero_weights_raise():
    with pytest.raises(SyntaxError, match="sum to"):
        _W_for(_uniform_region(5), weights=np.zeros(25))


def test_non_finite_weights_raise():
    w = np.ones(25)
    w[3] = np.inf
    with pytest.raises(SyntaxError):
        _W_for(_uniform_region(5), weights=w)


# ---------------------------------------------------------------------------
# the property that makes quadrature worth having
# ---------------------------------------------------------------------------
def test_gauss_legendre_integrates_the_region_exactly():
    """For a quadratic model f f^T reaches degree 4, so n >= 3 nodes is exact.

    Three nodes and twelve must therefore agree to numerical precision at a
    fixed design. The residual difference is finite-difference noise in W, not
    quadrature error, which is why the tolerance is 1e-9 and not tighter.
    """
    values = []
    for n in (3, 5, 8, 12):
        pts, wts = _gauss_region(n)
        values.append(_criterion(_W_for(pts, weights=wts), UNIFORM_EFFORT))
    spread = max(values) - min(values)
    assert spread / abs(values[0]) < 1e-9, values


def test_a_uniform_grid_does_not_integrate_exactly():
    """The discriminating counterpart: without it the test above proves nothing.

    If a uniform grid agreed to the same tolerance, the Gauss test would be
    passing on a property every rule has rather than on exactness.
    """
    pts_g, wts_g = _gauss_region(5)
    j_gauss = _criterion(_W_for(pts_g, weights=wts_g), UNIFORM_EFFORT)

    values = []
    for n in (5, 11, 21):
        pts = _uniform_region(n)
        values.append(_criterion(
            _W_for(pts, weights=np.ones(len(pts))), UNIFORM_EFFORT))

    # every value must be an AVERAGE, i.e. the same order as the Gauss value.
    # Without working weights these are SUMS over 25, 121 and 441 locations
    # and this bound fails -- which is what makes the test discriminate.
    for j, n in zip(values, (5, 11, 21)):
        assert 0.5 * j_gauss < j < 2.0 * j_gauss, (n, j, j_gauss)

    spread = max(values) - min(values)
    assert spread / abs(values[0]) > 1e-3, values


def test_a_uniform_grid_converges_toward_the_gauss_value():
    """Both rules estimate the same integral, so refining must close the gap.

    It closes slowly. Equal weights on a CLOSED uniform grid over-count the
    boundary, giving O(1/n) convergence from above: at 41 x 41 = 1681 points
    the error is still a few percent, where three Gauss nodes are exact. That
    is the measured case for preferring a quadrature rule over a fine grid,
    and it is asserted here rather than merely stated.
    """
    pts_g, wts_g = _gauss_region(5)
    j_gauss = _criterion(_W_for(pts_g, weights=wts_g), UNIFORM_EFFORT)

    errors = []
    for n in (11, 21, 41):
        pts = _uniform_region(n)
        j = _criterion(_W_for(pts, weights=np.ones(len(pts))), UNIFORM_EFFORT)
        errors.append(abs(j - j_gauss) / abs(j_gauss))

    assert errors[0] > errors[1] > errors[2], errors   # monotone improvement
    assert errors[-1] < 0.05, errors                   # and getting close
    assert errors[0] > 1e-2, errors                    # but nowhere near exact
