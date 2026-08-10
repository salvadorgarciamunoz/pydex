"""
Numerical-robustness regression tests for Ds-optimality.

These lock in behaviour on degenerate / ill-conditioned FIMs. Each test
corresponds to a scenario that crashed or silently returned a wrong answer in
an earlier revision, so they are guards against regression rather than
hypotheticals.

Run with:  python -m pytest test_ds_opt_numerics.py -v
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

NAMES = ["Ka", "A0", "k1", "k2"]
INTEREST = ["Ka", "A0"]
IDX_N = np.array([2, 3])
RNG = np.random.default_rng(11)


def _atoms(n_mp=4, n_e=4):
    return np.array([
        (lambda X: X @ X.T)(RNG.standard_normal((n_mp, n_mp + 3)))
        for _ in range(n_e)
    ])


def _designer(fim, atomics=None, fd_jac=True, interest=INTEREST, verbose=0):
    d = Designer()
    d.n_mp = len(NAMES)
    d.model_parameter_names = NAMES
    d.interest_parameters = interest
    d._fd_jac = fd_jac
    d._pseudo_bayesian = False
    d._large_memory_requirement = False
    d._pseudo_bayesian_type = 0
    d._verbose = verbose
    d.atomic_fims = atomics
    d.eval_fim = lambda e, store_predictions=True: (
        setattr(d, "fim", fim) or fim
    )
    return d


def _val(d, n_e=4):
    return d._ds_opt_criterion(np.ones(n_e) / n_e)


# ---------------------------------------------------------------- degenerate
@pytest.mark.parametrize("bad_fim, label", [
    (0, "python int 0 (all-NaN sensitivities never increment self.fim)"),
    (np.array([0]), "1-D sentinel returned by _eval_fim on an all-zero FIM"),
    (np.zeros((4, 4)), "all-zero FIM (every effort driven to zero)"),
    (np.full((4, 4), np.nan), "non-finite FIM from a diverged simulation"),
])
def test_degenerate_fim_returns_inf_without_crashing(bad_fim, label):
    d = _designer(bad_fim, _atoms())
    assert _val(d) == np.inf, label


def test_degenerate_fim_analytic_jac_has_correct_length():
    # infeasible + analytic gradient must still return a correctly-shaped jac
    d = _designer(0, _atoms(n_e=6), fd_jac=False)
    val, jac = d._ds_opt_criterion(np.ones(6) / 6)
    assert val == np.inf
    assert jac.shape == (6,)


def test_degenerate_fim_jac_length_without_atomics():
    # atomic_fims unavailable -> fall back to the effort-vector size rather
    # than raising TypeError on len(None)
    d = _designer(0, None, fd_jac=False)
    val, jac = d._ds_opt_criterion(np.ones(5) / 5)
    assert val == np.inf and jac.shape == (5,)


# ------------------------------------------------- singular nuisance block
def test_unidentified_nuisance_param_is_NOT_infeasible():
    """
    The headline case. A nuisance parameter carrying zero information makes
    det(FIM) = det(M_nn) = 0, so the Schur determinant identity is 0/0 and an
    identity-based implementation reports +inf. But S itself is perfectly
    well defined, and this is the primary motivation for Ds-optimal design.
    """
    F = np.eye(4)
    F[2, 2] = 0.0            # k1 wholly unidentified
    d = _designer(F, _atoms())
    val = _val(d)
    assert np.isfinite(val), "unidentified nuisance param must remain feasible"
    # S == I_2 here, so log-det(S) == 0 and the criterion (negated) is 0
    assert val == pytest.approx(0.0, abs=1e-12)


def test_generalised_schur_matches_regularised_limit():
    """The pinv/lstsq Schur complement must equal the delta->0 limit."""
    n_mp, n_e = 4, 5
    A = _atoms(n_mp, n_e)
    A[:, 3, :] = 0.0
    A[:, :, 3] = 0.0          # k2 unidentified -> singular nuisance block
    e = RNG.random(n_e) + 0.5
    F = sum(ei * a for ei, a in zip(e, A))
    d = _designer(F, A)
    got = _val(d, n_e)

    idx_s, idx_n = np.array([0, 1]), IDX_N
    Mss = F[np.ix_(idx_s, idx_s)]
    Msn = F[np.ix_(idx_s, idx_n)]
    Mns = F[np.ix_(idx_n, idx_s)]
    Mnn = F[np.ix_(idx_n, idx_n)]
    S = Mss - Msn @ np.linalg.solve(Mnn + 1e-10 * np.eye(len(idx_n)), Mns)
    assert got == pytest.approx(-np.linalg.slogdet(S)[1], rel=1e-7)


def test_analytic_gradient_exact_with_singular_nuisance_block():
    """
    The P-based gradient tr(S^-1 P' A_i P) needs only S^-1, so it stays valid
    when M_nn is singular -- unlike the identity gradient which needs M_nn^-1.
    """
    n_mp, n_e = 4, 5
    A = _atoms(n_mp, n_e)
    A[:, 3, :] = 0.0
    A[:, :, 3] = 0.0
    e = RNG.random(n_e) + 0.5

    def fim_of(x):
        return sum(xi * a for xi, a in zip(np.asarray(x).ravel(), A))

    d = Designer()
    d.n_mp = n_mp
    d.model_parameter_names = NAMES
    d.interest_parameters = INTEREST
    d._pseudo_bayesian = False
    d._large_memory_requirement = False
    d._verbose = 0
    d.atomic_fims = A
    d.eval_fim = lambda x, store_predictions=True: (
        setattr(d, "fim", fim_of(x)) or d.fim
    )

    d._fd_jac = False
    _v, jac = d._ds_opt_criterion(e)
    d._fd_jac = True
    eps, fd = 1e-6, np.zeros(n_e)
    for i in range(n_e):
        ep, em = e.copy(), e.copy()
        ep[i] += eps
        em[i] -= eps
        fd[i] = (d._ds_opt_criterion(ep) - d._ds_opt_criterion(em)) / (2 * eps)
    assert np.max(np.abs(jac - fd)) < 1e-6


# ----------------------------------------- singular in the INTEREST subspace
def test_singular_interest_subspace_is_infeasible():
    """
    Conversely, an unidentified INTEREST parameter genuinely makes Ds
    infeasible -- S is singular -- and must be reported as such.
    """
    F = np.eye(4)
    F[0, 0] = 0.0            # Ka unidentified, and Ka IS of interest
    d = _designer(F, _atoms())
    assert _val(d) == np.inf


def test_collinear_interest_params_infeasible():
    F = np.eye(4)
    F[0, 1] = F[1, 0] = 1.0  # Ka, A0 perfectly collinear -> S singular
    d = _designer(F, _atoms())
    assert _val(d) == np.inf


# --------------------------------------------------- positive-definiteness
def test_indefinite_fim_is_flagged_not_silently_accepted(capsys):
    """
    det > 0 is NOT a PD test: diag(1,1,-1,-1) has det(FIM) > 0 AND
    det(M_nn) > 0, so a determinant-sign feasibility check passes it. The
    implementation must detect and report the non-PSD FIM.
    """
    F = np.diag([1.0, 1.0, -1.0, -1.0])
    d = _designer(F, _atoms(), verbose=1)
    _val(d)
    out = capsys.readouterr().out
    assert "not positive semi-definite" in out


def test_nuisance_block_never_worse_conditioned_than_fim():
    """
    Cauchy interlacing: a principal submatrix of a PSD matrix is PSD and no
    worse conditioned than its parent, so cond(M_nn) <= cond(FIM). This is why
    the nuisance block is never the conditioning bottleneck -- S is.
    """
    for _ in range(2000):
        X = RNG.standard_normal((4, 6))
        F = X @ X.T
        assert (np.linalg.cond(F[np.ix_(IDX_N, IDX_N)])
                <= np.linalg.cond(F) * (1 + 1e-9))


def test_ill_conditioned_schur_warns(capsys):
    # interest params nearly collinear after marginalisation
    F = np.eye(4)
    F[0, 1] = F[1, 0] = 1.0 - 1e-13
    d = _designer(F, _atoms(), verbose=1)
    d._ds_cond_warn = 1e6
    d._ds_opt_criterion(np.ones(4) / 4)
    assert "ill-conditioned" in capsys.readouterr().out


# ------------------------------------------------------------- consistency
def test_all_params_of_interest_equals_d_optimal():
    A = _atoms()
    e = RNG.random(4) + 0.5
    F = sum(ei * a for ei, a in zip(e, A))
    d_ds = _designer(F, A, interest=NAMES)
    d_d = _designer(F, A)
    assert _val(d_ds) == pytest.approx(d_d._d_opt_criterion(e), rel=1e-12)


def test_matches_determinant_identity_where_both_valid():
    A = _atoms()
    e = RNG.random(4) + 0.5
    F = sum(ei * a for ei, a in zip(e, A))   # well-conditioned PD FIM
    _, ldf = np.linalg.slogdet(F)
    _, ldn = np.linalg.slogdet(F[np.ix_(IDX_N, IDX_N)])
    d = _designer(F, A)
    assert _val(d) == pytest.approx(-(ldf - ldn), rel=1e-10)


# -------------------------------------------------------- pseudo-Bayesian
def test_pb_degenerate_scenario_fim_returns_inf():
    d = _designer(0, _atoms())
    d._pseudo_bayesian = True
    d.scr_fims = [0, 0]
    d.eval_fim = lambda e, store_predictions=True: d.scr_fims
    assert d._pb_ds_opt_criterion(np.ones(4) / 4) == np.inf


def test_pb_unidentified_nuisance_stays_feasible():
    F = np.eye(4)
    F[2, 2] = 0.0
    d = _designer(F, _atoms())
    d._pseudo_bayesian = True
    d._pseudo_bayesian_type = 1
    d.scr_fims = [F, np.eye(4)]
    d.eval_fim = lambda e, store_predictions=True: d.scr_fims
    assert np.isfinite(d._pb_ds_opt_criterion(np.ones(4) / 4))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
