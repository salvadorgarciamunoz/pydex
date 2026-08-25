"""
Regression tests for `fixed_sampling_grid` -- one effort per EXPERIMENT rather
than per (candidate, sampling-time) cell.

WHAT THE FLAG IS FOR
--------------------
A fixed analytical schedule you do not control: every listed sampling time is
measured on every run, and the only decision is WHICH experimental conditions
to run. Before this flag, per-sampling-time effort was always a free variable
(see the optimize_sampling_times investigation -- that flag turned out to
control reporting only, never the formulation), so this design problem could
not be posed directly on a dynamic model.

HOW IT IS IMPLEMENTED, AND WHY THE CRITERION SHIFTS
---------------------------------------------------
n_spt-1 linear equalities per candidate tie that candidate's cells together.
With e[c,k] = w_c/n_spt the FIM becomes

    FIM = (1/n_spt) * sum_c w_c * (sum_k A[c,k])

i.e. exactly 1/n_spt times the per-experiment FIM. Because n_spt is constant
(ragged grids are refused -- see below) this is a CONSTANT rescaling: the
argmax is unchanged and log-det criteria shift by exactly n_mp*ln(n_spt), the
same structure PROJECT_NOTES records for error_cov scaling. So the DESIGN is
comparable across the flag; the criterion VALUE is not.

VALIDATED AGAINST AN INDEPENDENT REFERENCE
------------------------------------------
Not against another pydex path doing the same thing. The reference is the same
experiment posed as a STATIC multi-response model (the fixed grid folded into
the response vector), which reaches the FIM through entirely different
machinery: n_spt=1, no sampling-time axis, no per-time atomics. Measured on a
3-parameter/2-time model whose optimal design is genuinely spread (support 2
of 5, weights 0.333985/0.666015 -- deliberately not a degenerate
all-on-one-candidate answer, which would have proved nothing):

    design agreement        2.1e-08
    criterion offset        2.079441514  vs  n_mp*ln(n_spt) = 2.079441542
    within-candidate spread 0.000e+00
    control (flag off)      spread 0.333, different design

The SLSQP path was checked separately (ai_opt_criterion, spread 3.5e-20):
the constraint is applied in BOTH solvers, because omitting the SLSQP one
would let the flag be silently ignored for every non-native criterion --
pseudo-Bayesian type 1, vdi, the six prediction-variance criteria and CVaR.
"Fix every call site, not the one you found."

The tests below are solver-free and cover the GUARDS plus the constraint
construction. The numerical agreement above needs IPOPT and belongs in the
capability suite, not here.
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

TIMES = np.array([1.0, 3.0])
TIC = np.array([[0.5], [1.0], [2.0]])
THETA = np.array([2.0, 0.5, 1.5])


def _sim_dyn(ti_controls, sampling_times, model_parameters):
    u = ti_controls[0]
    a, b, c = model_parameters
    t = np.asarray(sampling_times, dtype=float)
    return a * u + b * t + c * u * t


def _sim_static(ti_controls, model_parameters):
    return np.array([ti_controls[0] * model_parameters[0]])


def _dyn_designer(spt=None):
    d = Designer()
    d.simulate = _sim_dyn
    d.model_parameters = THETA.copy()
    d.ti_controls_candidates = TIC.copy()
    d.sampling_times_candidates = (
        np.tile(TIMES, (len(TIC), 1)) if spt is None else spt
    )
    d.error_cov = np.array([[0.01]])
    d.initialize(verbose=0)
    return d


# ===================================================================== default
def test_default_is_off():
    """The historical behaviour must be untouched unless asked for. Confirmed
    end-to-end too: the section-03 design still returns 23.210474126177722."""
    assert Designer()._fixed_sampling_grid is False


def test_default_off_after_initialize():
    assert _dyn_designer()._fixed_sampling_grid is False


# ====================================================================== guards
def test_static_model_rejected():
    """A static model has no sampling-time axis to hold fixed; effort is
    already per experiment, so the flag would be a silent no-op."""
    d = Designer()
    d.simulate = _sim_static
    d.model_parameters = np.array([1.0])
    d.ti_controls_candidates = np.array([[0.0], [1.0], [2.0]])
    d.error_cov = np.array([[0.01]])
    d.initialize(verbose=0)
    with pytest.raises(SyntaxError, match="static model"):
        d.design_experiment(criterion=d.d_opt_criterion,
                            fixed_sampling_grid=True)


def test_n_spt_is_mutually_exclusive():
    """n_spt asks the optimiser to CHOOSE which times to sample; this flag
    says all listed times are measured. Both at once is incoherent, and
    n_spt already force-overrides optimize_sampling_times with only a
    warning -- a pattern not worth repeating."""
    d = _dyn_designer(np.tile(np.array([1.0, 2.0, 3.0]), (len(TIC), 1)))
    with pytest.raises(SyntaxError, match="mutually"):
        d.design_experiment(criterion=d.d_opt_criterion,
                            fixed_sampling_grid=True, n_spt=2)


def test_ragged_grid_refused():
    """
    Deliberate refusal, not an oversight. On a NaN-padded grid the constraint
    rescales candidate c by 1/n_spt_c -- DIFFERENT per candidate, so no longer
    a constant offset, so it would silently reweight candidates and change
    which design is optimal. That needs a modelling decision, not a code
    change. (Ragged grids also do not survive eval_sensitivities today, so
    this refuses an unreachable case -- but the guard must outlive that bug.)
    """
    spt = np.full((len(TIC), 3), np.nan)
    spt[0, :3] = [1.0, 2.0, 3.0]
    spt[1, :2] = [1.0, 2.0]
    spt[2, :2] = [1.0, 3.0]
    d = _dyn_designer(spt)
    assert d._var_n_sampling_time is True, "fixture must be ragged"
    with pytest.raises(NotImplementedError, match="DIFFERENT numbers"):
        d.design_experiment(criterion=d.d_opt_criterion,
                            fixed_sampling_grid=True)


def test_guards_do_not_fire_when_flag_is_off():
    """Every guard above must be reachable ONLY via the flag -- otherwise
    adding it would break existing ragged/n_spt users."""
    spt = np.full((len(TIC), 3), np.nan)
    spt[0, :3] = [1.0, 2.0, 3.0]
    spt[1, :2] = [1.0, 2.0]
    spt[2, :2] = [1.0, 3.0]
    d = _dyn_designer(spt)
    # must raise nothing from the fixed_sampling_grid guards; any failure
    # here comes from the (separate, pre-existing) ragged sensitivity path
    try:
        d.design_experiment(criterion=d.d_opt_criterion,
                            fixed_sampling_grid=False)
    except (SyntaxError, NotImplementedError) as ex:
        pytest.fail(f"a fixed_sampling_grid guard fired with the flag off: {ex}")
    except Exception:
        pass  # pre-existing ragged failure, not this flag's concern


# ================================================================ persistence
# NOT tested here, deliberately. Writing oed_result and reading it back needs
# a solved design and a real pickle, so it cannot live in the solver-free
# tier. A first draft of this file "covered" it with assertions on a locally
# built dict -- which asserted nothing about designer.py and would have passed
# against any implementation, vacuous in exactly the way 0.4.1's defect-1
# guard was. Removed rather than shipped.
#
# Verified by EXECUTION instead: a real solve writes the key
# ("fixed_sampling_grid" present in oed_result), load_oed_result round-trips
# it, and a copy of that pickle with the key deleted still loads with the flag
# defaulting to False -- the .get() rather than [...] in load_oed_result. If
# that path ever needs a regression guard it belongs in the capability suite.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
