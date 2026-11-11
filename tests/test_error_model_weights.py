"""
`error_variance` with `rel_sd > 0`: prediction-dependent measurement weights
(commit 2 of 2).

The variance of a measurement predicted at y is

    variance = (rel_sd * y)**2 + abs_sd**2          weight W = 1 / variance

evaluated ONCE per (candidate, sampling time) at the NOMINAL prediction and held
fixed through the finite-difference loop. Every consumer of the grid weights
(the FIM, estimability, the sensitivity diagnostics) must use the same array,
and the prior experiments and the V-optimal dw must use weights from THEIR OWN
predictions. Getting the wrong prediction set is silent, which is why most tests
here are checked against arithmetic written out by hand, not against another
code path.

THE CLOSED-FORM FIXTURE
-----------------------
A static model, linear in its parameters so the finite differences are exact:

    y1 = a*u + b            y2 = a*u**2 + 2*b            theta = (a, b)

    dy1/da = u    dy1/db = 1        dy2/da = u**2    dy2/db = 2

pydex NORMALISES sensitivities by parameter magnitude (`_norm_sens_by_params`),
so a hand reference must scale column j by theta_j or it is off by a factor
that depends on the parameter -- the mistake recorded under 0.7.2. The weights
differ strongly across the three candidates (the predictions are 2.5..6.5 and
3..19) and the efforts are deliberately non-uniform, so a mix-up of candidate
index, response index or prediction set cannot cancel.

Solver-free throughout.

Each test was run against the pre-change `designer.py` (the commit that
introduced `error_variance` with constant error only) and fails there; see the
commit message for the split.
"""
import sys
import types
import warnings

import dill
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

THETA = np.array([2.0, 0.5])
U = np.array([1.0, 2.0, 3.0])
EFF = np.array([0.5, 0.3, 0.2])
NAMES = ["conc", "temp"]
SPEC = {"conc": {"rel_sd": 0.10, "abs_sd": 0.5},
        "temp": {"rel_sd": 0.05, "abs_sd": 0.2}}


def _sim(ti_controls, model_parameters):
    a, b = model_parameters
    u = ti_controls[0]
    return np.array([a * u + b, a * u * u + 2.0 * b])


def _designer(spec=SPEC, theta=THETA, u=U, **attrs):
    d = Designer()
    d.simulate = _sim
    d.model_parameters = theta
    d.ti_controls_candidates = np.asarray(u, dtype=float).reshape(-1, 1)
    d.model_parameter_names = ["a", "b"]
    d.response_names = NAMES
    if spec is not None:
        d.error_variance = spec
    for k, v in attrs.items():
        setattr(d, k, v)
    d.initialize(verbose=0)
    return d


# ---- the hand reference ----------------------------------------------------
def _y(theta, u):
    a, b = theta
    return np.array([a * u + b, a * u * u + 2.0 * b])


def _s_scaled(theta, u):
    """(n_r, n_mp): dy/dtheta, columns scaled by theta, as pydex stores it."""
    return np.array([[u, 1.0], [u * u, 2.0]]) * np.asarray(theta)[None, :]


def _weights(y, spec=SPEC):
    rel = np.array([spec[n].get("rel_sd", 0.0) for n in NAMES])
    sd = np.array([spec[n]["abs_sd"] for n in NAMES])
    return 1.0 / ((rel * y) ** 2 + sd ** 2)


def _hand_atom(theta, u, spec=SPEC):
    s = _s_scaled(theta, u)
    return s.T @ np.diag(_weights(_y(theta, u), spec)) @ s


def _hand_fim(theta=THETA, u=U, eff=EFF, spec=SPEC):
    return sum(e * _hand_atom(theta, uu, spec) for e, uu in zip(eff, u))


def _e(d, eff=EFF):
    return np.asarray(eff, dtype=float).reshape(d.n_c, d.n_spt)


# --------------------------------------------------------------------------
# What the API now means
# --------------------------------------------------------------------------
def test_rel_sd_is_accepted_and_leaves_no_single_covariance():
    d = _designer()
    assert d._error_het is True
    # None ON PURPOSE: a consumer nobody taught the weights fails loudly
    # instead of silently using constant ones
    assert d.error_cov is None and d.error_fim is None


# --------------------------------------------------------------------------
# The FIM against arithmetic
# --------------------------------------------------------------------------
def test_fim_matches_closed_form():
    d = _designer()
    fim = np.asarray(d.eval_fim(_e(d)))
    np.testing.assert_allclose(fim, _hand_fim(), rtol=1e-6)


def test_fim_is_not_the_constant_weight_fim():
    """Guards the guard: weighting by the abs_sd floor alone, or by anything
    constant, gives a different matrix."""
    d = _designer()
    fim = np.asarray(d.eval_fim(_e(d)))
    const = _designer(spec={n: {"abs_sd": SPEC[n]["abs_sd"]} for n in NAMES})
    fim_c = np.asarray(const.eval_fim(_e(const)))
    assert not np.allclose(fim, fim_c, rtol=1e-3)


def test_weights_follow_the_candidate_not_a_neighbour():
    """One candidate carries all the effort: its own weights, exactly."""
    d = _designer()
    fim = np.asarray(d.eval_fim(_e(d, [0.0, 1.0, 0.0])))
    np.testing.assert_allclose(fim, _hand_atom(THETA, 2.0), rtol=1e-6)


def test_repeated_eval_sensitivities_does_not_shift_the_weights():
    """designer.response ACCUMULATES across calls. After theta changes, the
    pile holds theta_1's blocks followed by theta_2's; the weights must come
    from THIS call's blocks, not from the first n_c of the pile. (Both calls at
    the SAME theta would give identical blocks and prove nothing.)"""
    d = _designer()
    d.eval_sensitivities()                                # at THETA
    th2 = np.array([2.6, 0.3])
    d.model_parameters = th2
    d.initialize(verbose=0)                               # re-points at th2
    d.eval_sensitivities()
    assert len(d.response) == 2 * d.n_c                   # the accumulation is real
    d.atomic_fims = None
    fim = np.asarray(d.eval_fim(_e(d)))
    np.testing.assert_allclose(fim, _hand_fim(theta=th2), rtol=1e-6)
    assert not np.allclose(_hand_fim(theta=th2), _hand_fim(), rtol=1e-3)


def test_predictions_are_recorded_at_nominal_theta():
    d = _designer()
    d.eval_sensitivities()
    expected = np.array([_y(THETA, u) for u in U]).reshape(3, 1, 2)
    np.testing.assert_allclose(d._grid_pred, expected, rtol=0, atol=1e-12)


def test_weights_follow_the_pseudo_bayesian_scenario():
    """Each scenario's FIM uses ITS OWN nominal predictions."""
    scr = np.array([[2.0, 0.5], [2.6, 0.3]])
    d = _designer(theta=scr, _pseudo_bayesian_type=0)
    d.eval_fim(_e(d))
    assert len(d.scr_fims) == 2
    for s, th in enumerate(scr):
        np.testing.assert_allclose(np.asarray(d.scr_fims[s]),
                                   _hand_fim(theta=th), rtol=1e-6)
    # and the two scenarios genuinely differ, so this is not vacuous
    assert not np.allclose(d.scr_fims[0], d.scr_fims[1], rtol=1e-3)


# --------------------------------------------------------------------------
# A DYNAMIC model: the weights are per (candidate, sampling TIME)
# --------------------------------------------------------------------------
# With n_spt == 1 every time index is 0, so a per-time indexing slip is
# invisible to the static fixture above. This one has four times.
_TH_D = np.array([0.5, 1.5])
_C_D = np.array([1.0, 2.0])
_TS_D = np.array([0.5, 1.0, 2.0, 4.0])
_REL_D, _ABS_D = 0.10, 0.05


def _sim_dyn(ti_controls, sampling_times, model_parameters):
    k, a = model_parameters
    t = np.asarray(sampling_times, dtype=float)
    return (a * ti_controls[0] * np.exp(-k * t)).reshape(-1, 1)


def _atom_dyn(theta, c, t):
    k, a = theta
    y = a * c * np.exp(-k * t)
    s = np.array([-k * t * y, y])                  # d y/d theta, times theta
    return np.outer(s, s) / ((_REL_D * y) ** 2 + _ABS_D ** 2)


def _dyn_designer():
    d = Designer()
    d.simulate = _sim_dyn
    d.model_parameters = _TH_D
    d.model_parameter_names = ["k", "a"]
    d.response_names = ["A"]
    d.ti_controls_candidates = _C_D.reshape(-1, 1)
    d.sampling_times_candidates = np.tile(_TS_D, (2, 1))
    d.error_variance = {"A": {"rel_sd": _REL_D, "abs_sd": _ABS_D}}
    d.initialize(verbose=0)
    return d


def test_dynamic_fim_uses_the_weight_of_each_sampling_time():
    d = _dyn_designer()
    eff = np.array([[0.05, 0.10, 0.15, 0.20], [0.10, 0.10, 0.15, 0.15]])
    fim = np.asarray(d.eval_fim(eff))
    hand = sum(eff[i, j] * _atom_dyn(_TH_D, _C_D[i], _TS_D[j])
               for i in range(2) for j in range(4))
    np.testing.assert_allclose(fim, hand, rtol=1e-8)


def test_dynamic_prior_uses_its_own_times_and_predictions():
    d = _dyn_designer()
    times = (0.7, 3.0, 6.0)                        # not the candidate times
    d.set_prior_experiments(ti_controls=np.array([[1.5]]),
                            model_parameters=_TH_D,
                            sampling_times=np.array([list(times)]),
                            n_repeats=[2])
    hand = 2.0 * sum(_atom_dyn(_TH_D, 1.5, t) for t in times)
    np.testing.assert_allclose(np.asarray(d._prior_fim), hand, rtol=1e-8)


# --------------------------------------------------------------------------
# The three prediction sets: grid, prior conditions, dw
# --------------------------------------------------------------------------
def test_prior_fim_uses_weights_at_the_prior_conditions_not_the_grid():
    d = _designer()
    u_prior = 4.0                                  # NOT in the candidate grid
    d.set_prior_experiments(ti_controls=np.array([[u_prior]]),
                            model_parameters=THETA, n_repeats=[2])
    np.testing.assert_allclose(np.asarray(d._prior_fim),
                               2.0 * _hand_atom(THETA, u_prior), rtol=1e-6)
    # the grid's weights would give a different number
    wrong = 2.0 * sum(_hand_atom(THETA, uu) for uu in U) / len(U)
    assert not np.allclose(d._prior_fim, wrong, rtol=1e-3)


def test_v_optimal_W_uses_the_error_at_the_dw_prediction():
    d = _designer()
    u_dw = 2.5
    d.dw_tic = np.array([[u_dw]])
    d.dw_spt = np.array([1.0])
    W = np.asarray(d._eval_W_matrix())
    y = _y(THETA, u_dw)
    rel = np.array([SPEC[n]["rel_sd"] for n in NAMES])
    sd = np.array([SPEC[n]["abs_sd"] for n in NAMES])
    s_y = np.sqrt((rel * y) ** 2 + sd ** 2)        # std dev AT the prediction
    raw = np.array([[u_dw, 1.0], [u_dw ** 2, 2.0]])  # W is NOT theta-normalised
    expected = raw * (s_y[:, None] / np.abs(THETA)[None, :])
    np.testing.assert_allclose(W, expected, rtol=1e-6)


def test_off_diagonal_error_cov_in_W_is_reported_not_silent():
    """The V-optimal W scales by sqrt(diag(error_cov)) only. That is not
    changed here, but it must no longer be silent."""
    d = _designer(spec=None, error_cov=np.array([[0.04, 0.01], [0.01, 0.09]]))
    d.dw_tic = np.array([[2.5]])
    d.dw_spt = np.array([1.0])
    with pytest.warns(RuntimeWarning, match="off-diagonal"):
        d._eval_W_matrix()


# --------------------------------------------------------------------------
# Every consumer uses the same weights
# --------------------------------------------------------------------------
def _spy(d):
    rec = []
    orig = d._grid_error_weights

    def spy():
        w = orig()
        rec.append(w.copy())
        return w

    d._grid_error_weights = spy
    return rec


def test_fim_estimability_and_diagnostics_use_the_same_weight_array():
    d = _designer()
    rec = _spy(d)
    d.eval_fim(_e(d))
    d.run_estimability(plot=False, report=False)
    d.diagnose_sensitivity(report=False)
    assert len(rec) >= 3, "a consumer did not ask for the grid weights"
    hand = np.array([_weights(_y(THETA, u)) for u in U]).reshape(3, 1, 2)
    for w in rec:
        assert np.array_equal(w, rec[0])           # identical, not just close
    np.testing.assert_allclose(rec[0], hand, rtol=1e-12)


def test_estimability_absolute_information_matches_closed_form():
    d = _designer()
    out = d.run_estimability(plot=False, report=False)
    # abs info_j = sum over candidates and responses of w * (theta-scaled s)^2
    expected = sum(
        (_weights(_y(THETA, u))[:, None] * _s_scaled(THETA, u) ** 2).sum(axis=0)
        for u in U)
    got = np.array([out["abs_info"]["a"], out["abs_info"]["b"]])
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_estimability_ranking_uses_the_row_weights():
    """The weighted E index comes from QR with pivoting on Z_w, whose rows are
    sqrt(w) * dy/dtheta (theta divided back OUT -- estimability works on the
    raw derivative). Built here from the arithmetic, not from pydex."""
    from scipy.linalg import qr

    def _e_index(rows):
        _, R, piv = qr(np.vstack(rows), mode="economic", pivoting=True)
        nrm = np.abs(np.diag(R))
        return {["a", "b"][j]: float(v) for j, v in zip(piv, nrm / nrm[0])}

    raw = [np.array([[u, 1.0], [u * u, 2.0]]) for u in U]
    weighted = _e_index([np.sqrt(_weights(_y(THETA, u)))[:, None] * r
                         for u, r in zip(U, raw)])
    unweighted = _e_index(raw)
    # the fixture must be able to tell the two apart, or this proves nothing
    assert not np.isclose(min(weighted.values()), min(unweighted.values()),
                          rtol=1e-3)

    out = _designer().run_estimability(plot=False, report=False)
    assert out["weighted"] is True
    for name, v in weighted.items():
        assert out["e_index"][name] == pytest.approx(v, rel=1e-6)
    for name, v in unweighted.items():
        assert out["e_index_ud"][name] == pytest.approx(v, rel=1e-6)


def test_diagnose_sensitivity_diagonal_matches_closed_form():
    d = _designer()
    d.eval_sensitivities()
    out = d.diagnose_sensitivity(report=False)
    expected = np.array([np.diag(_hand_atom(THETA, u)) for u in U])
    np.testing.assert_allclose(np.asarray(out["diag_A"]), expected, rtol=1e-6)


# --------------------------------------------------------------------------
# Guards: stale caches, missing predictions
# --------------------------------------------------------------------------
def test_changing_error_variance_invalidates_cached_atomics():
    """A changed spec must reach the FIM: the cached atomic FIMs carry their
    weights, and initialize() is what invalidates them."""
    d = _designer()
    f1 = np.asarray(d.eval_fim(_e(d))).copy()
    assert d.atomic_fims is not None               # cached
    new = {"conc": {"rel_sd": 0.30, "abs_sd": 0.5},
           "temp": {"rel_sd": 0.05, "abs_sd": 0.2}}
    d.error_variance = new
    d.initialize(verbose=0)
    f2 = np.asarray(d.eval_fim(_e(d)))
    np.testing.assert_allclose(f2, _hand_fim(spec=new), rtol=1e-6)
    assert not np.allclose(f1, f2, rtol=1e-3)


def test_store_predictions_false_raises_rather_than_unweight():
    d = _designer()
    with pytest.raises(ValueError, match="store_predictions"):
        d.eval_sensitivities(store_predictions=False)


def test_store_predictions_false_is_fine_for_constant_error():
    """A NEGATIVE CONTROL: it cannot fail against a designer that never raises
    (the pre-change file), so it is not proved that way. It is proved against a
    MUTANT whose guard raises regardless of the error model; see the commit
    message. It exists so the guard above cannot quietly become too broad."""
    d = _designer(spec={"conc": {"abs_sd": 0.5}, "temp": {"abs_sd": 0.2}})
    d.eval_sensitivities(store_predictions=False)       # must not raise
    assert d.sensitivities is not None


def test_loaded_sensitivities_carry_no_predictions_and_say_so(tmp_path,
                                                              monkeypatch):
    d = _designer()
    d.eval_sensitivities()
    monkeypatch.chdir(tmp_path)
    with open("sens.pkl", "wb") as fh:
        dill.dump(d.sensitivities, fh)
    d.load_sensitivity("sens.pkl")
    d.atomic_fims = None                                # force a rebuild
    with pytest.raises(RuntimeError, match="no.*predictions|none are recorded"):
        d.eval_fim(_e(d))


def test_strict_measured_subset_width_mismatch_is_explained():
    y = np.zeros((3, 1, 3))                             # 3 columns, 2 measured
    d = _designer()
    with pytest.raises(ValueError, match="exactly the measured"):
        d._error_weights(y)


def test_warns_when_one_responses_weights_span_more_than_1e6():
    d = _designer(spec={"conc": {"rel_sd": 0.1, "abs_sd": 0.01},
                        "temp": {"abs_sd": 0.2}},
                  u=[0.0, 1000.0])
    with pytest.warns(RuntimeWarning, match="conc.*span"):
        d.eval_fim(_e(d, [0.5, 0.5]))


def test_warns_on_a_negative_prediction_but_stays_finite():
    d = _designer(theta=np.array([2.0, -5.0]))          # y1 = 2u - 5 < 0 at u=1
    with pytest.warns(RuntimeWarning, match="negative"):
        fim = np.asarray(d.eval_fim(_e(d)))
    assert np.all(np.isfinite(fim))


# --------------------------------------------------------------------------
# The goal-oriented grid: its own predictions, hence its own weights
# --------------------------------------------------------------------------
_TH3 = np.array([2.0, 0.6, 1.5])


def _sim_go(ti_controls, sampling_times, model_parameters):
    x = float(ti_controls[0])
    t0, t1, t2 = (float(model_parameters[0]), float(model_parameters[1]),
                  float(model_parameters[2]))
    t = np.asarray(sampling_times, dtype=float)
    return np.column_stack([t0 * np.exp(-t1 * x * t),
                            t2 / (1.0 + t1 * x * t)])


def _go_designer(spec):
    d = Designer()
    d.simulate = _sim_go
    d.model_parameters = _TH3
    d.ti_controls_candidates = np.array([[0.5], [1.0], [2.0]])
    d.sampling_times_candidates = np.array([[0.5, 1.0, 2.0]] * 3)
    d.measurable_responses = [0, 1]
    d.model_parameter_names = ["theta_0", "theta_1", "theta_2"]
    d.response_names = ["y1", "y2"]
    d.error_variance = spec
    d._verbose = 0
    d.initialize(verbose=0)
    d._fd_jac = True
    d.eval_sensitivities(save_sensitivities=False)
    d.go_simulate = _sim_go
    d.go_tic = np.array([[1.2], [1.6]])
    d.go_spt = np.array([[1.0, 2.0]] * 2)
    d.n_c_go, d.n_spt_go, d.n_tic_go, d.n_r_go = 2, 2, 1, 2
    return d


@pytest.mark.parametrize("spec", [
    {"y1": 0.01, "y2": 0.01},                                   # constant
    {"y1": {"rel_sd": 0.1, "abs_sd": 0.05},
     "y2": {"rel_sd": 0.1, "abs_sd": 0.05}},                    # prediction-dependent
], ids=["constant", "relative"])
def test_goal_oriented_swap_works_with_error_variance(spec):
    """_swap_candidates() exchanges error_cov with go_error_cov and
    re-initializes. The first version of error_variance tracked the matrix it
    derived by identity, so the second swap read its own earlier result as a
    user-supplied error_cov and raised."""
    d = _go_designer(spec)
    e = np.ones(d.n_c * d.n_spt) / (d.n_c * d.n_spt)
    d.reset_pvar_logdet_mode()
    val = float(d.vdi_criterion(e))
    assert np.isfinite(val)
    assert d._candidates_swapped is False
    assert d.n_c == 3 and d.n_spt == 3


def test_goal_oriented_grid_keeps_its_own_predictions():
    d = _go_designer({"y1": {"rel_sd": 0.1, "abs_sd": 0.05},
                      "y2": {"rel_sd": 0.1, "abs_sd": 0.05}})
    e = np.ones(d.n_c * d.n_spt) / (d.n_c * d.n_spt)
    d.reset_pvar_logdet_mode()
    d.vdi_criterion(e)
    main = np.array([_sim_go(np.array([x]), np.array([0.5, 1.0, 2.0]), _TH3)
                     for x in (0.5, 1.0, 2.0)])
    go = np.array([_sim_go(np.array([x]), np.array([1.0, 2.0]), _TH3)
                   for x in (1.2, 1.6)])
    np.testing.assert_allclose(d._grid_pred, main, atol=1e-12)
    np.testing.assert_allclose(d._go_grid_pred, go, atol=1e-12)


def test_error_variance_with_go_error_cov_raises():
    d = Designer()
    d.simulate = _sim_go
    d.model_parameters = _TH3
    d.ti_controls_candidates = np.array([[0.5], [1.0]])
    d.sampling_times_candidates = np.array([[0.5, 1.0]] * 2)
    d.model_parameter_names = ["t0", "t1", "t2"]
    d.response_names = ["y1", "y2"]
    d.error_variance = {"y1": 0.01, "y2": 0.01}
    d.go_error_cov = np.diag([0.01, 0.01])
    with pytest.raises(ValueError, match="go_error_cov"):
        d.initialize(verbose=0)


# --------------------------------------------------------------------------
# Unsetting, and the saved state
# --------------------------------------------------------------------------
def test_unsetting_error_variance_drops_the_derived_error_cov():
    d = _designer(spec={"conc": 0.04, "temp": 0.09})
    assert d.error_cov is not None
    d.error_variance = None
    d.initialize(verbose=0)
    assert d._error_cov_defaulted                       # back to the default
    assert np.array_equal(d.error_cov, np.eye(2))


def test_save_and_load_state_carry_the_error_specification(tmp_path,
                                                           monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = _designer()
    d._generate_result_path = lambda fn, ext: str(tmp_path / f"{fn}.{ext}")
    d.save_state()

    fresh = Designer()
    fresh.load_state("/state.pkl")
    assert fresh.error_variance == SPEC
    # the derived error_cov is NOT restored beside it: both set would raise
    assert fresh.error_cov is None


def test_a_state_written_by_the_pre_change_code_still_loads(tmp_path,
                                                            monkeypatch):
    """A REAL pre-change state file, not a list built by hand to look like one.

    tests/data/state_15f5276.dill was written by `save_state()` of the
    `designer.py` at git 15f5276 (the commit before `error_variance`
    existed), on the model in this file with an explicit `error_cov`:

        git show 15f5276:pydex/core/designer.py > old/designer.py
        # stub pydex.utils.trellis_plotter / pydex.core.logger, import it,
        # build the designer, then:  d.save_state()

    It is committed rather than generated at test time because CI checks out
    one commit and cannot `git show` an old one. The length check below is what
    stops it being regenerated from the current code, which would turn this
    into a round trip within one version: a different question, and the one
    the capability suite's sections 15 and 58 already ask.
    """
    import pathlib
    import shutil
    fixture = pathlib.Path(__file__).parent / "data" / "state_15f5276.dill"
    with open(fixture, "rb") as fh:
        raw = dill.load(fh)
    assert len(raw) == 10, (
        f"the fixture has {len(raw)} items; it must be the 10-item format the "
        f"code at 15f5276 wrote, not a state written by the current code")

    monkeypatch.chdir(tmp_path)
    shutil.copy(fixture, tmp_path / "state.pkl")
    fresh = Designer()
    fresh.load_state("/state.pkl")
    assert fresh.n_c == 3 and fresh.n_mp == 2 and fresh.n_m_r == 2
    np.testing.assert_array_equal(fresh.model_parameters, THETA)
    np.testing.assert_array_equal(fresh.ti_controls_candidates,
                                  U.reshape(-1, 1))
    # an old state recorded no error specification: nothing is restored, and
    # in particular nothing is invented
    assert fresh.error_variance is None and fresh.error_cov is None


def test_error_cov_round_trips_through_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = _designer(spec=None, error_cov=np.diag([0.04, 0.09]))
    d._generate_result_path = lambda fn, ext: str(tmp_path / f"{fn}.{ext}")
    d.save_state()
    fresh = Designer()
    fresh.load_state("/state.pkl")
    assert np.array_equal(fresh.error_cov, np.diag([0.04, 0.09]))
    assert fresh.error_variance is None
