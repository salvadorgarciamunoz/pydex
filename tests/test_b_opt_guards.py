"""
Argument-validation regression tests for b_opt_criterion (Bracketing-optimal
design; Chen, Paulavicius & Adjiman 2018, AIChE J. 64:3944-3957).

These are deliberately SOLVER-FREE. Every guard under test raises before any
Pyomo model is built, so the whole file runs in milliseconds and belongs in
CI -- unlike the b_opt behaviour tests (brute-force agreement, weight
extremes, sweep behaviour), which need an MINLP solver and live in
testing_scripts/pydex_full_capability_test.py.

That split is deliberate. The capability suite is NOT run by CI (it needs
IPOPT, an MINLP solver and PyNumero's compiled ASL extension), so a criterion
whose only coverage lived there would have documentation promising specific
behaviour and no automatic guard at all.

The n_exp lower bound is the one worth understanding, because it is TWO
conditions and getting it wrong does not produce an error -- it produces a
hang. Both Cholesky lifts in _solve_pyomo_b_opt are built unconditionally,
whatever output_weight is, and both floor their diagonal at 1e-8:

    input  M_in  is phi x phi,          rank <= n_exp      -> n_exp >= phi
    output M_out is n_resp x n_resp and CENTERED, so
                                        rank <= n_exp - 1

Since M == L L^T with every L[j,j] >= 1e-8 forces det(M) >= 1e-8**(2*dim), a
rank-deficient M cannot be represented and the program is STRICTLY
infeasible.

For M_out the ALGEBRAIC bound is n_exp >= n_resp + 1, but the implemented
bound is n_exp >= n_resp + 2, and the difference was MEASURED rather than
derived. At n_exp == n_resp + 1 the centered covariance is rank-exactly-
n_resp with no margin, and bonmin reports the problem infeasible whenever the
output term carries weight: on a 10-candidate phi=2/n_resp=2 pool, n_exp=3 was
infeasible at output_weight >= 0.5 while n_exp=4 solved at once. The tests
below therefore assert + 2. Do not relax them back to + 1 on the strength of
the algebra. Proving infeasibility of a nonconvex MINLP is the expensive
direction: measured on a 70-candidate phi=6 pool, n_exp=5 ran for over 17
minutes of bonmin CPU without terminating, while n_exp=6 solved in seconds.
Hence the up-front check, and hence these tests -- the boundary cases below
(n_exp == phi and n_exp == n_resp + 1) are exactly what a future refactor of
either lift could break silently.

Run with:  python -m pytest test_b_opt_guards.py -v
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

RNG = np.random.default_rng(5)


def _designer(n_c=30, phi=3, n_resp=2, with_response=True):
    """Minimal designer carrying only what the b_opt guards read.

    No simulate, no sensitivities, no solver: the guards under test all fire
    before the Pyomo model is constructed. Candidates are random rather than
    gridded because none of these tests depend on the geometry -- only on the
    SHAPES (n_c, phi, n_resp).
    """
    d = Designer()
    d.ti_controls_candidates = RNG.uniform(0.0, 1.0, size=(n_c, phi))
    d.n_c = n_c
    if with_response:
        d.response = RNG.uniform(0.0, 1.0, size=(n_c, n_resp))
    return d


def _call(d, n_exp=None, output_weight=0.5):
    """Invoke the guarded entry point directly.

    _solve_pyomo_b_opt is private, but calling it is what keeps this file
    solver-free -- going through design_experiment() would drag in effort
    initialisation and the full dispatch path. The sibling Ds tests call
    _resolve_ds_idx() the same way.
    """
    return d._solve_pyomo_b_opt(
        e0=np.ones((d.n_c, 1)) / d.n_c,
        fix_effort=None,
        solver_options=None,
        n_exp=n_exp,
        output_weight=output_weight,
    )


# --------------------------------------------------------------------------
# n_exp: presence, type, and the trivial floor
# --------------------------------------------------------------------------

def test_missing_n_exp_rejected():
    # b_opt has no meaningful continuous relaxation: the cardinality
    # constraint IS the problem, so n_exp is mandatory rather than optional
    # as it is for every other criterion.
    with pytest.raises(ValueError, match="requires an exact number of experiments"):
        _call(_designer(), n_exp=None)


def test_non_integer_n_exp_rejected():
    with pytest.raises(ValueError, match="must be an integer"):
        _call(_designer(), n_exp=4.0)


def test_n_exp_below_two_rejected():
    # a covariance needs at least two points, whatever phi and n_resp are
    with pytest.raises(ValueError, match="must be an integer"):
        _call(_designer(), n_exp=1)


def test_n_exp_exceeding_candidate_count_rejected():
    with pytest.raises(ValueError, match="cannot exceed the number of candidates"):
        _call(_designer(n_c=10), n_exp=11)


# --------------------------------------------------------------------------
# the rank bound: n_exp >= max(phi, n_resp + 1)
# --------------------------------------------------------------------------

def test_n_exp_below_phi_rejected():
    # phi=6 dominates: n_resp+1 = 4, so 5 clears the output bound but not
    # the input one. This is the case measured at 17+ minutes of bonmin CPU
    # before the guard existed.
    with pytest.raises(ValueError, match="too small"):
        _call(_designer(n_c=40, phi=6, n_resp=3), n_exp=5)


def test_n_exp_below_output_bound_rejected():
    # phi=2 but n_resp=5, so the CENTERED output covariance dominates:
    # n_resp+1 = 6 is binding while the input bound is only 2. Guarding
    # phi alone would let this through and hang.
    with pytest.raises(ValueError, match="too small"):
        _call(_designer(n_c=40, phi=2, n_resp=5), n_exp=4)


def test_output_bound_applies_even_at_output_weight_zero():
    # Both lifts are constructed unconditionally; output_weight only changes
    # the OBJECTIVE. So a design that cannot represent the output covariance
    # is infeasible even when that term carries no weight. Easy to get wrong
    # by making the guard conditional on wout > 0.
    with pytest.raises(ValueError, match="too small"):
        _call(_designer(n_c=40, phi=2, n_resp=5), n_exp=4, output_weight=0.0)


def test_error_message_names_the_binding_condition():
    # The message must say WHICH bound bit, otherwise the user cannot tell
    # whether to raise n_exp or reduce the model.
    with pytest.raises(ValueError) as exc:
        _call(_designer(n_c=40, phi=6, n_resp=3), n_exp=5)
    msg = str(exc.value)
    assert "input-space matrix is 6x6" in msg
    assert "n_exp >= 6" in msg
    # the output bound (n_resp+1 = 4) is satisfied at n_exp=5, so it must
    # NOT be listed as a reason
    assert "output covariance" not in msg


def test_error_message_lists_both_conditions_when_both_bind():
    with pytest.raises(ValueError) as exc:
        _call(_designer(n_c=40, phi=6, n_resp=3), n_exp=2)
    msg = str(exc.value)
    assert "input-space matrix is 6x6" in msg
    assert "output covariance is 3x3" in msg


def test_error_message_warns_about_the_hang():
    # The failure mode this guard replaces is an unbounded solve, not a
    # crash. Someone who hits the guard while raising n_exp needs to know
    # that, or they will assume the solver is merely slow.
    with pytest.raises(ValueError, match="hang"):
        _call(_designer(n_c=40, phi=6, n_resp=3), n_exp=5)


# --------------------------------------------------------------------------
# the boundaries must NOT be rejected
# --------------------------------------------------------------------------
# These are the tests that would catch an off-by-one introduced by a later
# refactor of either lift. They must get PAST the guard; they are not
# expected to solve here, since no solver is configured. So the contract is
# narrow and deliberate: whatever happens next, it must not be the guard's
# ValueError.

def _passes_guard(d, n_exp, output_weight=0.5):
    try:
        _call(d, n_exp=n_exp, output_weight=output_weight)
    except ValueError as exc:
        if "too small" in str(exc):
            return False
        return True          # some other ValueError, i.e. past the guard
    except Exception:
        return True          # solver/environment failure, i.e. past the guard
    return True


def test_n_exp_equal_to_phi_accepted():
    # phi=6 binding, n_resp+1 = 4: n_exp=6 sits exactly on the input bound
    assert _passes_guard(_designer(n_c=40, phi=6, n_resp=3), n_exp=6)


def test_n_exp_equal_to_output_bound_accepted():
    # phi=2, n_resp=5 -> bound is n_resp+2 = 7. n_exp=7 sits exactly on it.
    assert _passes_guard(_designer(n_c=40, phi=2, n_resp=5), n_exp=7)


def test_n_exp_one_below_output_bound_rejected():
    # n_exp=6 satisfies the ALGEBRAIC bound (n_resp+1) but not the implemented
    # one. This is the case that reported `infeasible` from the solver before
    # the bound was raised, so it must be rejected up front.
    with pytest.raises(ValueError, match="too small"):
        _call(_designer(n_c=40, phi=2, n_resp=5), n_exp=6)


def test_both_bounds_simultaneously_tight_accepted():
    # phi=4, n_resp=2 -> max(4, 4) = 4: both bounds bind at once.
    assert _passes_guard(_designer(n_c=30, phi=4, n_resp=2), n_exp=4)


def test_film_coating_smallest_design_now_rejected():
    # phi=3, n_resp=2 is the film-coating scenario. Its Figure 4 sweep starts
    # at n_exp=3, which satisfies the old bound max(3, n_resp+1)=3 and FAILS
    # the corrected one, max(3, n_resp+2)=4. Recorded deliberately: that
    # scenario's smallest design size was relying on a bound that does not
    # hold, and the sweep should start at 4.
    with pytest.raises(ValueError, match="too small"):
        _call(_designer(n_c=30, phi=3, n_resp=2), n_exp=3)
    assert _passes_guard(_designer(n_c=30, phi=3, n_resp=2), n_exp=4)


# --------------------------------------------------------------------------
# the other guards
# --------------------------------------------------------------------------

def test_missing_response_rejected():
    # b_opt is the only criterion that reads self.response, so forgetting
    # simulate_candidates() is a new and easy mistake. RuntimeError rather
    # than ValueError: the argument is fine, the designer state is not.
    with pytest.raises(RuntimeError, match="simulate_candidates"):
        _call(_designer(with_response=False), n_exp=4)


def test_missing_response_checked_before_the_rank_bound():
    # Ordering matters for the diagnostic: n_resp is unknowable without
    # responses, so an n_exp complaint here would quote a meaningless bound.
    with pytest.raises(RuntimeError, match="simulate_candidates"):
        _call(_designer(phi=6, with_response=False), n_exp=2)


@pytest.mark.parametrize("w", [-0.1, 1.1, 2.0, -1.0])
def test_output_weight_outside_unit_interval_rejected(w):
    with pytest.raises(ValueError, match="output_weight"):
        _call(_designer(), n_exp=4, output_weight=w)


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
def test_output_weight_endpoints_accepted(w):
    # 0 and 1 are the two meaningful extremes (pure input bracketing and
    # pure output coverage), so the interval must be CLOSED.
    assert _passes_guard(_designer(), n_exp=4, output_weight=w)


def test_candidate_count_inconsistency_detected():
    # A stale n_c against a freshly assigned candidate array is the
    # signature of re-initialising without clearing derived state; it must
    # not be silently reconciled.
    d = _designer(n_c=30)
    d.n_c = 29
    with pytest.raises(RuntimeError, match="Internal inconsistency"):
        _call(d, n_exp=4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
