"""
Regression test for Open Item 35 (PROJECT_NOTES.md): `run_estimability()` must
hand back the figures its own plotting helper builds.

BACKGROUND
----------
`_plot_estimability()` builds one bar chart per estimability index plus a
correlation heat map, and ends with `return figs`. The caller discarded that
return value:

    if plot:
        self._plot_estimability(out, tol)      # return value thrown away
    return out

The figures ARE created and left open, so under an INTERACTIVE backend
`designer.show_plots()` (i.e. `plt.show()`) displays them and nothing is
visibly lost -- which is why this survived so long. Under a NON-INTERACTIVE
backend, though, a caller who wants to SAVE them had no handle at all and had
to diff `plt.get_fignums()` around the call to recover them. Every other
plotter on `Designer` returns its figures (the capability suite relies on it,
e.g. `figs = d.plot_predictions()`), so this one method broke an existing
convention.

The count is 3 or 4, not always 4: one bar chart per index, and the E-index
panel only exists when `error_cov` was supplied. Both cases are asserted here,
because "it returned a list" is a much weaker claim than "it returned exactly
the figures that were drawn".

This test is solver-free: `run_estimability()` computes sensitivities and does
linear algebra on them, with no `design_experiment()` call and no IPOPT, so
the whole file belongs in the fast CI tier. matplotlib is forced to Agg so
nothing tries to open a window.

Run with:  python -m pytest test_estimability_returns_figures.py -v
"""
import sys
import types

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


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
    """Static two-response model in three parameters.

    `c` enters only through a 1e-6 coefficient, so it is genuinely the least
    estimable of the three. That is not what is under test, but it keeps the
    estimability ranking non-degenerate so the plotting path has real content
    to draw rather than a pathological all-equal case.
    """
    a, b, c = model_parameters
    x = ti_controls[0]
    return np.array([a * x + b * x ** 2 + 1e-6 * c * x,
                     a + b * x])


def _make_designer(with_error_cov):
    d = Designer()
    d.simulate = _simulate
    d.model_parameters = np.array([1.0, 0.5, 2.0])
    d.model_parameter_names = ["a", "b", "c"]
    d.ti_controls_names = ["x"]
    d.response_names = ["y1", "y2"]
    d.ti_controls_candidates = np.linspace(0.5, 2.0, 7).reshape(-1, 1)
    if with_error_cov:
        d.error_cov = np.eye(2)
    d.initialize(verbose=0)
    return d


@pytest.fixture(autouse=True)
def _close_figures():
    """Leave no figures behind -- this file deliberately creates several."""
    plt.close("all")
    yield
    plt.close("all")


class TestEstimabilityReturnsFigures:

    def test_figures_key_exists(self):
        """
        THE regression case. Pre-fix the key was absent entirely, so
        `out["figures"]` raised KeyError and `out.get("figures")` was None
        even though four figures had just been drawn.
        """
        d = _make_designer(with_error_cov=True)
        out = d.run_estimability(plot=True, report=False)
        assert "figures" in out, (
            "run_estimability() drew figures but did not return their handles"
        )

    @pytest.mark.parametrize("with_error_cov, expected", [(True, 4), (False, 3)])
    def test_returns_every_figure_it_drew(self, with_error_cov, expected):
        """
        The returned list must be exactly the figures created by the call --
        not a subset, and not a stale registry read.

        Four with `error_cov` (abs info, E, E-UD, correlation) and three
        without, since the E-index panel needs the weighting. Asserting the
        count against a `plt.get_fignums()` delta measured around the SAME
        call is what makes this stronger than `isinstance(x, list)`: a fix
        that returned an empty list, or the wrong list, would still fail.
        """
        d = _make_designer(with_error_cov=with_error_cov)
        before = set(plt.get_fignums())
        out = d.run_estimability(plot=True, report=False)
        created = set(plt.get_fignums()) - before

        assert len(created) == expected, (
            f"expected {expected} figures to be drawn, got {len(created)}"
        )
        figs = out["figures"]
        assert isinstance(figs, list)
        assert len(figs) == expected
        assert {f.number for f in figs} == created, (
            "the returned handles are not the figures this call created"
        )

    def test_returned_handles_are_usable_headless(self):
        """
        The point of the fix: under a non-interactive backend a caller must be
        able to save the figures without touching pyplot's global registry.
        """
        d = _make_designer(with_error_cov=True)
        out = d.run_estimability(plot=True, report=False)
        for fig in out["figures"]:
            assert hasattr(fig, "savefig")
            assert fig.get_axes(), "a returned figure has no axes drawn on it"

    def test_plot_false_returns_none_and_draws_nothing(self):
        """
        `plot=False` must still populate the key -- a caller should not have to
        branch on its presence -- but with None rather than an empty list, so
        "plotting was not attempted" is distinguishable from "plotting produced
        nothing", and so iterating it fails loudly rather than silently doing
        nothing.
        """
        d = _make_designer(with_error_cov=True)
        before = set(plt.get_fignums())
        out = d.run_estimability(plot=False, report=False)
        assert set(plt.get_fignums()) - before == set()
        assert "figures" in out
        assert out["figures"] is None

    def test_the_rest_of_the_payload_is_untouched(self):
        """
        Adding a key must not disturb the existing contract. Guards against a
        fix that rebuilt the dict rather than adding to it.
        """
        d = _make_designer(with_error_cov=True)
        out = d.run_estimability(plot=True, report=False)
        for key in ("table", "correlation", "ranking", "groups", "corr_tol",
                    "tol", "corr_matrix", "corr_names", "abs_info",
                    "e_index_ud", "flagged", "order", "n_rows", "weighted"):
            assert key in out, f"pre-existing key {key!r} disappeared"
        assert list(out["table"]["parameter"]) == out["ranking"]
