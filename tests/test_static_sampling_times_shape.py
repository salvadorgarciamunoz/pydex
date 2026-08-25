"""
Regression test for Open Item 17 (PROJECT_NOTES.md): the static-model
placeholder `sampling_times_candidates` must be shaped (n_c, 1), not
(n_c, n_tic).

BACKGROUND
----------
A static model (simulate signature 1, `(ti_controls, model_parameters)`) has
no time axis, so pydex fabricates a placeholder `sampling_times_candidates`
purely so downstream shape arithmetic has something to index -- every
CONSUMING read of its content is behind an `if self._dynamic_system:` guard,
so the VALUES never mattered (0.4.1 already fixed those from uninitialised
memory to deterministic zeros). But the shape was wrong: it followed
`ti_controls_candidates`, i.e. (n_c, n_tic), while n_spt is always 1 for a
static model, so the correct shape is (n_c, 1). This was left deliberately
uncorrected in 0.4.1 because fixing it reaches `save_state`/`load_state`
(the array is pickled) and `_pad_sampling_times`.

This test is solver-free: initialising a static Designer only calls
`simulate()` once as a plain Python function (to infer n_r), no IPOPT
involved, so the whole file belongs in the fast CI tier.

Run with:  python -m pytest test_static_sampling_times_shape.py -v
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


def _static_simulate(ti_controls, model_parameters):
    """Signature-1 (static) model: two ti_controls, so a shape bug here
    would show up as sampling_times_candidates having width 2, not 1."""
    x, z = ti_controls
    a, b = model_parameters
    return np.array([a + b * x + 0.05 * z])


def _make_static_designer(n_candidates=6):
    d = Designer()
    d.simulate = _static_simulate
    d.model_parameters = np.array([1.0, 2.0])
    d.ti_controls_candidates = np.array([
        [x, z] for x, z in zip(
            np.linspace(0, 10, n_candidates),
            np.linspace(-5, 5, n_candidates),
        )
    ])
    d.error_cov = np.array([[0.01]])
    d.initialize(verbose=0)
    return d


class TestStaticSamplingTimesShape:

    def test_shape_is_n_c_by_1_not_n_c_by_n_tic(self):
        """
        THE regression case. Pre-fix, this placeholder followed
        ti_controls_candidates's shape: with 2 time-invariant controls
        (as used here specifically to make the bug visible), the wrong
        shape would be (n_c, 2), not (n_c, 1).
        """
        d = _make_static_designer(n_candidates=6)
        assert d.n_tic == 2, "fixture sanity check: need >1 tic to expose the bug"
        assert d.sampling_times_candidates.shape == (6, 1)

    def test_n_spt_is_1(self):
        d = _make_static_designer()
        assert d.n_spt == 1

    def test_values_are_zero_and_deterministic(self):
        """0.4.1 already fixed the VALUES (uninitialised memory -> zeros);
        this just confirms that guard still holds under the new shape."""
        d = _make_static_designer()
        assert np.all(d.sampling_times_candidates == 0.0)
        assert d.sampling_times_candidates.dtype == np.float64

    def test_single_tic_case_also_correct(self):
        """With n_tic == 1, the old buggy shape (n_c, n_tic) and the correct
        shape (n_c, 1) happen to coincide -- this is the case that would NOT
        have caught the bug, included so the two-tic case above isn't the
        only one on record."""
        def simulate_1tic(ti_controls, model_parameters):
            x = ti_controls[0]
            a, b = model_parameters
            return np.array([a + b * x])

        d = Designer()
        d.simulate = simulate_1tic
        d.model_parameters = np.array([1.0, 2.0])
        d.ti_controls_candidates = np.array([[x] for x in np.linspace(0, 10, 5)])
        d.error_cov = np.array([[0.01]])
        d.initialize(verbose=0)
        assert d.n_tic == 1
        assert d.sampling_times_candidates.shape == (5, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
