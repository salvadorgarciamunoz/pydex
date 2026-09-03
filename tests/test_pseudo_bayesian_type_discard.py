"""
Regression test for the ``pseudo_bayesian_type`` silent-discard defect.

BACKGROUND
----------
`design_experiment(pseudo_bayesian_type=...)` is the only supported way to
choose between the two pseudo-Bayesian aggregations:

    0  average the INFORMATION -- criterion of the mean FIM
    1  average the CRITERION   -- mean of the per-scenario criterion values

When the keyword is omitted the code did this, unconditionally:

    if pseudo_bayesian_type is None:
        self._pseudo_bayesian_type = 0

which is correct for a fresh designer and wrong after `load_oed_result()`.
That method RESTORES `_pseudo_bayesian_type` from the saved file, so:

    solve as type 1  ->  save  ->  fresh designer  ->  load_oed_result()
    ->  design_experiment() without repeating the keyword

silently switched the aggregation from average-criterion to
average-information. Measured on a 40-scenario van Laar problem: the criterion
went from 16.397751 to 16.541379, with no warning, no error, and a report that
then truthfully said type 0. Nothing looked wrong. The user never touched a
private attribute -- the library set it and the library discarded it.

THE FIX, AND WHY IT IS A WARNING RATHER THAN STICKY STATE
---------------------------------------------------------
Omitting an optional argument should mean "use the default", not "reuse
whatever this object happens to be carrying". Making the value sticky would
have fixed the load case at the cost of a call whose behaviour depends on
history. So the default is unchanged -- omitting the keyword still gives 0 --
and the discard now warns instead of happening in silence.

That makes the warning itself the whole fix, which is why the negative cases
below matter as much as the positive one: a warning that fires on ordinary use
would be trained away within a day.

This file is solver-free. It drives `_pseudo_bayesian_type` resolution through
a stubbed `design_experiment` path rather than a real solve, so it belongs in
the fast CI tier.

Run with:  python -m pytest test_pseudo_bayesian_type_discard.py -v
"""
import sys
import types
import warnings

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


def _simulate(ti_controls, model_parameters):
    a, b = model_parameters
    x = ti_controls[0]
    return np.array([a * x + b * x ** 2])


def _designer(model_parameters):
    d = Designer()
    d.simulate = _simulate
    d.model_parameters = np.asarray(model_parameters, dtype=float)
    d.model_parameter_names = ["a", "b"]
    d.ti_controls_names = ["x"]
    d.response_names = ["y"]
    d.ti_controls_candidates = np.linspace(0.5, 2.0, 5).reshape(-1, 1)
    d.error_cov = np.eye(1)
    d.initialize(verbose=0)
    return d


SCENARIOS = np.column_stack([
    np.linspace(0.8, 1.2, 6),
    np.linspace(0.4, 0.6, 6),
])
NOMINAL = np.array([1.0, 0.5])

_MESSAGE_MARKER = "pseudo_bayesian_type was not passed"


def _resolve(designer, **kwargs):
    """Run only the pseudo_bayesian_type resolution, capturing warnings.

    `design_experiment()` would need a solver; the block under test runs long
    before that, so it is exercised directly. If the private helper is ever
    renamed this raises rather than passing vacuously.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            designer.design_experiment(
                designer.d_opt_criterion, solver="__no_such_solver__",
                write=False, **kwargs,
            )
        except Exception:
            # The solve is expected to fail; the resolution happens first.
            pass
    hits = [w for w in caught if _MESSAGE_MARKER in str(w.message)]
    return designer._pseudo_bayesian_type, hits


class TestSilentDiscardWarns:

    def test_warns_when_a_restored_type_is_discarded(self):
        """
        THE regression case: the designer is carrying type 1 (as
        load_oed_result would leave it) and the keyword is omitted.
        """
        d = _designer(SCENARIOS)
        d._pseudo_bayesian_type = 1          # as load_oed_result() leaves it
        resolved, hits = _resolve(d)

        assert hits, (
            "omitting pseudo_bayesian_type discarded a restored value of 1 "
            "without warning"
        )
        assert "1" in str(hits[0].message), \
            "the warning should name the value being discarded"
        assert resolved == 0, (
            "the default must still be 0 -- this fix warns, it does not make "
            "the setting sticky"
        )

    def test_the_default_is_still_zero(self):
        """The contract is unchanged: omitting the keyword means 0."""
        d = _designer(SCENARIOS)
        resolved, hits = _resolve(d)
        assert resolved == 0
        assert not hits, "a fresh designer has nothing to discard"


class TestStaysQuietOtherwise:
    """
    A warning that fires on ordinary use is worse than none, so these are as
    important as the case above. Each of them warned zero times when measured.
    """

    def test_quiet_for_a_local_design(self):
        d = _designer(NOMINAL)
        resolved, hits = _resolve(d)
        assert not hits, "a local design has no pseudo-Bayesian type at all"
        assert resolved is None

    @pytest.mark.parametrize("requested", [0, 1, "avg_inf", "avg_crit",
                                           "average_information",
                                           "average_criterion"])
    def test_quiet_when_the_keyword_is_passed(self, requested):
        d = _designer(SCENARIOS)
        d._pseudo_bayesian_type = 1          # something to discard, if it did
        resolved, hits = _resolve(d, pseudo_bayesian_type=requested)
        assert not hits, f"passing {requested!r} explicitly must not warn"
        assert resolved == requested

    def test_quiet_when_the_existing_value_is_already_the_default(self):
        """
        Carrying 0 and defaulting to 0 discards nothing, so there is nothing
        to say.
        """
        d = _designer(SCENARIOS)
        d._pseudo_bayesian_type = 0
        resolved, hits = _resolve(d)
        assert not hits
        assert resolved == 0


class TestRejectionUnchanged:

    def test_an_invalid_type_still_raises(self):
        d = _designer(SCENARIOS)
        with pytest.raises(SyntaxError):
            d.design_experiment(d.d_opt_criterion, solver="ipopt",
                                write=False, pseudo_bayesian_type=2)
