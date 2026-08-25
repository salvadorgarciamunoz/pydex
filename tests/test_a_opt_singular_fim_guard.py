"""
Regression tests for the relative-eigenvalue tolerance on A-optimality's
singular-FIM guard, and on the shared `_safe_fim_inverse` helper.

THE BUG THIS GUARDS
--------------------
`_a_opt_criterion`, `_pb_a_opt_criterion` and `_safe_fim_inverse` each tested
positive-definiteness with a STRICT `eigvals > 0`, no tolerance. That is the
wrong test for a FIM that is exactly singular in theory (e.g. two parameters
enter the model only as an additive sum, so the data can never separate
them): the eigenvalue in the null direction is not a clean zero once you
actually compute it in floating point, it is a residual of order 1e-19 to
1e-16 relative to the FIM's scale, and WHICH SIDE OF ZERO IT LANDS ON is
decided by summation order, not by anything about the model or design.

This was reproduced directly on the c1/c2 additive-constant model in
smoke_test_designer.py's CHECK 2 (`y = A0*exp(-k*t) + c1 + c2`, so
dy/dc1 = dy/dc2 = 1 exactly at every candidate -- no FD noise involved at
all, since FD is exact for linear parameters regardless of step size).
Three mathematically-equivalent ways of assembling the SAME FIM produced
three different signs for the same near-zero eigenvalue:

    pydex's own loop-accumulated FIM (matches d.fim exactly) : +1.2151e-19
    the same sensitivities via one S.T @ diag(efforts) @ S matmul : -9.294e-20
    the same computation carried in longdouble, cast back to float64 :
                                                                  -2.449e-20

Pre-fix, the tiny-POSITIVE case passed `eigvals > 0` for every eigenvalue, so
the FIM inverted "successfully" and A-optimality returned a huge but finite
number (~8.2e+18) instead of +inf -- exactly CHECK 2's failure mode. The
tiny-NEGATIVE cases already happened to return +inf pre-fix, purely by
accident of which side of zero the noise fell on.

The fix replaces the strict test with a RELATIVE cutoff (rtol=1e-12 against
the largest eigenvalue), matching the convention `diagnose_fim_structure`
already uses for the identical question ("is this FIM structurally
singular") on the identical matrix.

Tests below construct FIMs directly (no solver, no simulate() call) so this
whole file is solver-free and belongs in the fast CI tier, mirroring
test_ds_opt_numerics.py's approach for the equivalent Ds-optimal guards.

Run with:  python -m pytest test_a_opt_singular_fim_guard.py -v
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

NAMES = ["k", "A0", "c1", "c2"]
RNG = np.random.default_rng(2026_08_25)

# Eigenvalue scale taken directly from smoke_test_designer.py's CHECK 2, so
# the "tiny residual" cases below are not an arbitrary choice of magnitude --
# they are the actual numbers that were observed to flip sign across three
# equivalent arithmetic paths on the real model.
_OTHER_EIGS = np.array([8.340322e-04, 3.051250e-02, 1.605654e-01])


def _atoms(n_mp=4, n_e=4):
    return np.array([
        (lambda X: X @ X.T)(RNG.standard_normal((n_mp, n_mp + 3)))
        for _ in range(n_e)
    ])


def _designer(fim, atomics=None, fd_jac=True, verbose=0):
    d = Designer()
    d.n_mp = len(NAMES)
    d.model_parameter_names = NAMES
    d._fd_jac = fd_jac
    d._pseudo_bayesian = False
    d._large_memory_requirement = False
    d._verbose = verbose
    d.atomic_fims = atomics
    d.eval_fim = lambda e, store_predictions=True: (
        setattr(d, "fim", fim) or fim
    )
    return d


def _rank_deficient_fim(residual):
    """
    A 4x4 PSD-by-construction matrix with three well-separated eigenvalues
    (matching the smoke-test model's scale) and a fourth that is exactly
    `residual` -- used to plant the near-zero eigenvalue on either side of
    zero, deterministically, without depending on any actual FD computation.

    DELIBERATELY DIAGONAL, not a random-rotation Q @ diag(eigs) @ Q.T. The
    first draft of this fixture used a random orthogonal Q, and the
    reconstruction matmul alone introduces rounding of order
    n * eps * max(eigs) ~= 4 * 2.2e-16 * 0.16 ~= 1.4e-16 -- three orders of
    magnitude LARGER than the 1e-19 residual being injected, so the intended
    sign was completely swamped before eigvalsh ever saw it (verified: the
    injected +1.2151e-19 came back as -3.5e-18, and the test passed against
    BOTH the pre-fix and post-fix code -- a vacuous test that would have
    shipped if it hadn't been checked against the pre-fix file first, exactly
    the "test the check before trusting the check" trap in PROJECT_NOTES.md).
    A diagonal matrix has no such reconstruction step: its eigenvalues ARE
    its diagonal entries, verified here to survive eigvalsh exactly
    (1.2151e-19 in, 1.2151e-19 out, bit for bit).
    """
    eigs = np.concatenate([[residual], _OTHER_EIGS])
    return np.diag(eigs)


# ============================================================= _a_opt_criterion
class TestAOptSingularGuard:

    def test_tiny_positive_residual_is_infeasible(self):
        """
        THE regression case. Pre-fix this returned a huge finite number
        (`eigvals > 0` is True for +1.2e-19), not +inf.
        """
        fim = _rank_deficient_fim(residual=1.2151e-19)
        d = _designer(fim, _atoms())
        assert d._a_opt_criterion(np.ones(4) / 4) == np.inf

    def test_tiny_negative_residual_is_infeasible(self):
        """The mirror case (matches the -9.294e-20 sign observed from the
        single-matmul FIM assembly). Already correct pre-fix; must not
        regress."""
        fim = _rank_deficient_fim(residual=-9.294e-20)
        d = _designer(fim, _atoms())
        assert d._a_opt_criterion(np.ones(4) / 4) == np.inf

    def test_exactly_zero_residual_is_infeasible(self):
        fim = _rank_deficient_fim(residual=0.0)
        d = _designer(fim, _atoms())
        assert d._a_opt_criterion(np.ones(4) / 4) == np.inf

    def test_analytic_branch_returns_inf_with_correctly_shaped_zero_jac(self):
        fim = _rank_deficient_fim(residual=1.2151e-19)
        d = _designer(fim, _atoms(n_e=6), fd_jac=False)
        val, jac = d._a_opt_criterion(np.ones(6) / 6)
        assert val == np.inf
        assert jac.shape == (6,)
        assert np.all(jac == 0.0)

    def test_well_conditioned_fim_unaffected(self):
        """The fix must not become MORE conservative than needed: an
        ordinary well-conditioned FIM must still return the exact
        trace(FIM^-1), matching direct computation."""
        A = _atoms()
        e = RNG.random(4) + 0.5
        fim = sum(ei * a for ei, a in zip(e, A))
        d = _designer(fim, A)
        val = d._a_opt_criterion(e)
        assert np.isfinite(val)
        assert val == pytest.approx(np.trace(np.linalg.inv(fim)), rel=1e-10)

    def test_near_singular_but_above_tolerance_stays_feasible(self):
        """
        A genuinely small but non-noise eigenvalue (here ~1e-6 relative to
        the largest, six orders of magnitude above the 1e-12 cutoff) must
        stay invertible. The guard is meant to catch floating-point noise on
        an exactly-singular matrix, not to reject legitimately
        ill-conditioned but real designs.
        """
        residual = 1e-6 * _OTHER_EIGS.max()
        fim = _rank_deficient_fim(residual=residual)
        d = _designer(fim, _atoms())
        val = d._a_opt_criterion(np.ones(4) / 4)
        assert np.isfinite(val)


# ========================================================== _pb_a_opt_criterion
class TestPbAOptSingularGuard:

    def test_avg_crit_any_singular_scenario_is_infeasible(self):
        """avg_crit (type 1) averages per-scenario A-opt values; ANY
        infeasible scenario must poison the whole thing, exactly like the
        non-pseudo-Bayesian case."""
        bad = _rank_deficient_fim(residual=1.2151e-19)
        good = sum(
            ei * a for ei, a in zip(RNG.random(4) + 0.5, _atoms())
        )
        d = _designer(bad, _atoms())
        d._pseudo_bayesian = True
        d._pseudo_bayesian_type = 1
        d.scr_fims = [bad, good]
        d.eval_fim = lambda e, store_predictions=True: d.scr_fims
        assert d._pb_a_opt_criterion(np.ones(4) / 4) == np.inf

    def test_avg_crit_all_well_conditioned_is_finite(self):
        A = _atoms()
        e = RNG.random(4) + 0.5
        good1 = sum(ei * a for ei, a in zip(e, A))
        good2 = sum(ei * a for ei, a in zip(e + 0.1, A))
        d = _designer(good1, A)
        d._pseudo_bayesian = True
        d._pseudo_bayesian_type = 1
        d.scr_fims = [good1, good2]
        d.eval_fim = lambda ef, store_predictions=True: d.scr_fims
        val = d._pb_a_opt_criterion(e)
        assert np.isfinite(val)


# =============================================================== _safe_fim_inverse
class TestSafeFimInverseGuard:
    """`_safe_fim_inverse` feeds eval_pim / eval_pim_for_v_opt, which in turn
    feed vdi/dg/di/ag/ai/eg/ei_opt_criterion and V-optimal. Same fragility,
    same fix, tested directly against the helper rather than through a
    consuming criterion."""

    def test_tiny_positive_residual_returns_none(self):
        fim = _rank_deficient_fim(residual=1.2151e-19)
        d = Designer()
        d.fim = fim
        assert d._safe_fim_inverse() is None

    def test_tiny_negative_residual_returns_none(self):
        fim = _rank_deficient_fim(residual=-9.294e-20)
        d = Designer()
        d.fim = fim
        assert d._safe_fim_inverse() is None

    def test_well_conditioned_fim_inverts_normally(self):
        A = _atoms()
        e = RNG.random(4) + 0.5
        fim = sum(ei * a for ei, a in zip(e, A))
        d = Designer()
        d.fim = fim
        inv = d._safe_fim_inverse()
        assert inv is not None
        assert np.allclose(inv, np.linalg.inv(fim), rtol=1e-10)

    def test_near_singular_but_above_tolerance_still_inverts(self):
        residual = 1e-6 * _OTHER_EIGS.max()
        fim = _rank_deficient_fim(residual=residual)
        d = Designer()
        d.fim = fim
        assert d._safe_fim_inverse() is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
