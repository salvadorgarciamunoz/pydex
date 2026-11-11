"""
`error_variance`: the name-keyed measurement-error API (commit 1 of 2).

WHY THIS EXISTS
---------------
`error_cov` is an n_m_r x n_m_r matrix indexed by POSITION. Nothing checks it
against `response_names`, so a transposed or mis-ordered covariance on a
multi-response model silently changes every design. `error_variance` is keyed
by response NAME, matched exactly (the way `interest_parameters` is matched
against `model_parameter_names`), so order cannot be wrong by construction.

SCOPE OF THIS COMMIT
--------------------
The API only, and NO behaviour change: a dict resolves to the constant diagonal
`error_cov` it is equivalent to, and every downstream consumer sees that. The
prediction-dependent part (`rel_sd > 0`) is tested in test_error_model_weights.py.
In the first commit it raised rather than being accepted and ignored, because
ignoring it would have handed back a homoscedastic design under a label that
claims otherwise.

Solver-free: the equivalence tests go through `eval_fim` (finite-difference
sensitivities over `simulate()`), not through an optimiser.

THE DISCRIMINATING FIXTURE
--------------------------
Two responses with DIFFERENT variances (0.01 and 0.09). With equal variances a
swapped assignment would be invisible, so every ordering assertion would pass
on a broken implementation.

Each test below was run against the pre-change `designer.py` (where
`error_variance` is a plain, ignored attribute) and fails there.
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

NOM = np.array([0.5, 2.0])
SPT = np.linspace(0.5, 4.0, 4)
NAMES = ["conc", "temp"]
VAR = {"conc": 0.01, "temp": 0.09}          # different on purpose


def _simulate(ti_controls, sampling_times, model_parameters):
    k, a = model_parameters
    c = ti_controls[0]
    t = np.asarray(sampling_times, dtype=float)
    return np.column_stack([a * c * np.exp(-k * t), a * (1 - np.exp(-k * t))])


def _designer(names=NAMES, **attrs):
    """An UNinitialised two-response designer; attrs set before initialize()."""
    d = Designer()
    d.simulate = _simulate
    d.model_parameters = NOM
    d.ti_controls_candidates = np.array([[1.0], [2.0], [3.0]])
    d.sampling_times_candidates = np.tile(SPT, (3, 1))
    d.model_parameter_names = ["k", "a"]
    if names is not None:
        d.response_names = names
    for k, v in attrs.items():
        setattr(d, k, v)
    return d


def _init(d):
    d.initialize(verbose=0)
    return d


def _efforts(d):
    return np.full((d.n_c, d.n_spt), 1.0 / (d.n_c * d.n_spt))


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------
def test_resolves_to_diagonal_in_response_names_order():
    """The variance lands on the response it was NAMED for, not by position."""
    d = _init(_designer(error_variance=dict(VAR)))
    assert np.array_equal(d.error_cov, np.diag([0.01, 0.09]))


def test_dict_order_is_irrelevant():
    a = _init(_designer(error_variance={"conc": 0.01, "temp": 0.09}))
    b = _init(_designer(error_variance={"temp": 0.09, "conc": 0.01}))
    assert np.array_equal(a.error_cov, b.error_cov)
    assert np.array_equal(a.error_fim, b.error_fim)
    # and the permuted dict did not silently swap them
    assert b.error_cov[0, 0] == 0.01 and b.error_cov[1, 1] == 0.09


def test_unknown_name_raises():
    d = _designer(error_variance={"conc": 0.01, "temp": 0.09, "pressure": 1.0})
    with pytest.raises(ValueError, match="pressure"):
        d.initialize(verbose=0)


def test_missing_measured_response_raises():
    d = _designer(error_variance={"conc": 0.01})
    with pytest.raises(ValueError, match="temp"):
        d.initialize(verbose=0)


def test_names_are_exact_not_case_insensitive():
    d = _designer(error_variance={"Conc": 0.01, "temp": 0.09})
    with pytest.raises(ValueError, match="Conc"):
        d.initialize(verbose=0)


def test_setting_both_error_variance_and_error_cov_raises():
    d = _designer(error_variance=dict(VAR), error_cov=np.diag([0.01, 0.09]))
    with pytest.raises(ValueError, match="Both error_variance and error_cov"):
        d.initialize(verbose=0)


def test_reinitialize_is_not_mistaken_for_setting_both():
    """initialize() writes error_cov itself; calling it again must not trip
    the 'both set' check on the matrix it derived."""
    d = _init(_designer(error_variance=dict(VAR)))
    # anchor to the expected value: comparing before/after alone is vacuous,
    # since an implementation that ignored error_variance (identity default
    # both times) would also satisfy it
    assert np.array_equal(d.error_cov, np.diag([0.01, 0.09]))
    before = d.error_cov.copy()
    d.initialize(verbose=0)
    assert np.array_equal(d.error_cov, before)


def test_variance_after_a_defaulted_initialize_is_accepted():
    """An identity error_cov that initialize() DEFAULTED is not a user
    statement, so adding error_variance afterwards must not raise."""
    d = _init(_designer())                       # no error info: defaults
    assert d._error_cov_defaulted
    d.error_variance = dict(VAR)
    d.initialize(verbose=0)
    assert np.array_equal(d.error_cov, np.diag([0.01, 0.09]))
    assert not d._error_cov_defaulted           # now a real statement


def test_user_error_cov_after_error_variance_raises():
    d = _init(_designer(error_variance=dict(VAR)))
    d.error_cov = np.diag([1.0, 1.0])            # the user's own, not derived
    with pytest.raises(ValueError, match="Both error_variance and error_cov"):
        d.initialize(verbose=0)


def test_not_a_dict_raises():
    d = _designer(error_variance=[0.01, 0.09])   # positional list: the bug
    with pytest.raises(ValueError, match="dict keyed by response name"):
        d.initialize(verbose=0)


def test_measured_subset_convention():
    """One name per MEASURED response (the convention error_cov's rows use)."""
    d = _designer(names=["conc"], measurable_responses=np.array([0]),
                  error_variance={"conc": 0.04})
    d.initialize(verbose=0)
    assert d.n_m_r == 1
    assert np.array_equal(d.error_cov, np.diag([0.04]))


def test_names_for_all_responses_when_only_a_subset_is_measured_raises():
    d = _designer(names=NAMES, measurable_responses=np.array([0]),
                  error_variance={"conc": 0.04, "temp": 0.09})
    with pytest.raises(ValueError, match="MEASURED"):
        d.initialize(verbose=0)


def test_default_auto_names_are_usable_as_keys():
    d = _designer(names=None,
                  error_variance={"Response 0": 0.01, "Response 1": 0.09})
    d.initialize(verbose=0)
    assert np.array_equal(d.error_cov, np.diag([0.01, 0.09]))


# --------------------------------------------------------------------------
# Value validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_abs_sd_not_positive_raises(bad):
    d = _designer(error_variance={"conc": {"abs_sd": bad}, "temp": 0.09})
    with pytest.raises(ValueError, match="abs_sd must be > 0"):
        d.initialize(verbose=0)


def test_abs_sd_is_required_not_defaulted():
    d = _designer(error_variance={"conc": {"rel_sd": 0.1}, "temp": 0.09})
    with pytest.raises(ValueError, match="'abs_sd' is required"):
        d.initialize(verbose=0)


def test_negative_rel_sd_raises():
    d = _designer(error_variance={"conc": {"abs_sd": 0.1, "rel_sd": -0.1},
                                  "temp": 0.09})
    with pytest.raises(ValueError, match="rel_sd must be >= 0"):
        d.initialize(verbose=0)


def test_explicit_zero_rel_sd_is_the_constant_case():
    d = _init(_designer(error_variance={
        "conc": {"abs_sd": 0.1, "rel_sd": 0.0}, "temp": 0.09}))
    assert d.error_cov[0, 0] == 0.1 ** 2


def test_unknown_key_in_entry_raises():
    d = _designer(error_variance={"conc": {"abs_sd": 0.1, "abs_sigma": 1},
                                  "temp": 0.09})
    with pytest.raises(ValueError, match="abs_sigma"):
        d.initialize(verbose=0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"),
                                 True, "0.1", None])
def test_bad_bare_variance_raises(bad):
    d = _designer(error_variance={"conc": bad, "temp": 0.09})
    with pytest.raises(ValueError, match="conc"):
        d.initialize(verbose=0)


def test_abs_sd_is_a_standard_deviation_and_is_squared():
    d = _init(_designer(error_variance={"conc": {"abs_sd": 0.3},
                                        "temp": {"abs_sd": 0.5}}))
    assert d.error_cov[0, 0] == 0.3 ** 2
    assert d.error_cov[1, 1] == 0.5 ** 2


# --------------------------------------------------------------------------
# Homoscedastic equivalence to error_cov: bit-identical, end to end
# --------------------------------------------------------------------------
def test_homoscedastic_variance_is_bit_identical_to_error_cov():
    a = _init(_designer(error_variance=dict(VAR)))
    b = _init(_designer(error_cov=np.diag([0.01, 0.09])))
    assert np.array_equal(a.error_cov, b.error_cov)
    assert np.array_equal(a.error_fim, b.error_fim)
    assert a._error_cov_defaulted == b._error_cov_defaulted is False


def test_homoscedastic_variance_gives_bit_identical_fim():
    """Through the real eval_fim path, not just the matrices."""
    a = _init(_designer(error_variance=dict(VAR)))
    b = _init(_designer(error_cov=np.diag([0.01, 0.09])))
    fa = np.asarray(a.eval_fim(_efforts(a)))
    fb = np.asarray(b.eval_fim(_efforts(b)))
    assert np.array_equal(fa, fb)
    # and the weighting is not a no-op: unit error gives a different FIM
    c = _init(_designer(error_cov=np.eye(2)))
    fc = np.asarray(c.eval_fim(_efforts(c)))
    assert not np.allclose(fa, fc)


def test_permuted_dict_gives_bit_identical_fim():
    a = _init(_designer(error_variance={"conc": 0.01, "temp": 0.09}))
    b = _init(_designer(error_variance={"temp": 0.09, "conc": 0.01}))
    ref = _init(_designer(error_cov=np.diag([0.01, 0.09])))
    fb = np.asarray(b.eval_fim(_efforts(b)))
    assert np.array_equal(np.asarray(a.eval_fim(_efforts(a))), fb)
    # anchored to an independent error_cov reference: a == b alone is vacuous
    # if both silently fell back to the same identity default
    assert np.array_equal(fb, np.asarray(ref.eval_fim(_efforts(ref))))


def test_swapped_names_give_a_different_fim():
    """Guards the guard: if names were ignored, swapping them would change
    nothing. They are not ignored."""
    a = _init(_designer(error_variance={"conc": 0.01, "temp": 0.09}))
    b = _init(_designer(error_variance={"conc": 0.09, "temp": 0.01}))
    assert not np.allclose(np.asarray(a.eval_fim(_efforts(a))),
                           np.asarray(b.eval_fim(_efforts(b))))
