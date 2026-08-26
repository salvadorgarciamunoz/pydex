"""
Two INVARIANTS plus the regressions they would have caught.

WHY THIS FILE EXISTS
--------------------
Three separate defects turned out to be instances of one missing contract:
nothing asserted that a criterion's SCALAR-FIM branch computed the same
function as its own MATRIX branch. Each criterion carried an
`if fim.size == 1:` short-circuit, and they had drifted:

    criterion  matrix branch        correct at 1x1   scalar branch returned
    d_opt      -log det(FIM)        -log(FIM)        -FIM      (LOG vs LINEAR)
    a_opt      trace(FIM^-1)        +1/FIM           -FIM      (sign AND form)
    e_opt      -min(eig)            -FIM             -FIM      correct
    v_opt      trace(W FIM^-1 W^T)  ~1/FIM           +FIM      DIRECTION REVERSED
    cvar_d     -log det(FIM)        -log(FIM)        -FIM      (LOG vs LINEAR)

E-optimality was the one that agreed, which is what identified the rule the
others were breaking. Observable consequences on real solves:

  * apportion() reported a rounding efficiency of 871% for a single-parameter
    D-optimal design whose rounded efforts were BIT-IDENTICAL to the
    continuous ones (true answer: exactly 100%), and ~0% for the same design
    under A-optimality. An efficiency above 100% is impossible for a rounding
    of the optimum.
  * compute_criterion_value() returned -FIM (-5297.49) where the documented
    log-det answer was -8.575.
  * v_opt, being minimised, preferred the LEAST informative experiment when
    scored through the numpy evaluator.

The fix is not to correct five copies but to DELETE them: slogdet, inv and
eigvalsh are all exact on a (1, 1) array, so the matrix branch already IS the
single-parameter answer. The invariant below then holds by construction rather
than by five code paths happening to agree.

The second invariant guards apportion()'s SHAPE handling. See the docstring on
test_ragged_support_never_yields_a_nonfinite_run_count.

All solver-free.
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


# =====================================================================
# INVARIANT 1 -- a criterion's value at a 1x1 FIM must equal its own
# matrix formula evaluated at that same 1x1 matrix.
# =====================================================================
def _designer_with_fim(fim, n_mp, n_e=3):
    """A designer whose eval_fim() simply installs a prescribed FIM."""
    d = Designer()
    d.n_mp = n_mp
    d._fd_jac = True
    d._pseudo_bayesian = False
    d._pseudo_bayesian_type = 0
    d._large_memory_requirement = False
    d._regularize_fim = False
    d._verbose = 0
    fim = np.atleast_2d(np.asarray(fim, dtype=float))
    d.atomic_fims = np.array([np.eye(n_mp) for _ in range(n_e)])
    d.fim = fim
    d.scr_fims = [fim]
    d.eval_fim = lambda x, store_predictions=True: fim
    return d


# The FIM value is deliberately far from 1.0: at FIM == 1 every candidate
# formula collapses to the same number (log 1 == 0, 1/1 == 1), so a fixture
# using 1.0 would pass against the broken code. 5297.49 is the value from the
# real single-parameter solve that exposed the 871% report.
_F = 5297.4935762011755


@pytest.mark.parametrize("crit_name,expected", [
    # -log det, which for a 1x1 matrix is -log(FIM)
    ("_d_opt_criterion", -np.log(_F)),
    # trace(FIM^-1), positive, which for a 1x1 matrix is 1/FIM
    ("_a_opt_criterion", 1.0 / _F),
    # -min(eig), which for a 1x1 matrix is -FIM
    ("_e_opt_criterion", -_F),
])
def test_scalar_fim_matches_the_matrix_formula(crit_name, expected):
    """
    The regression case. Pre-fix, d_opt returned -5297.49 instead of -8.575
    and a_opt returned -5297.49 instead of +1.888e-4, so both of these
    assertions fail against the removed scalar branches while e_opt (which
    already agreed) passes either way -- i.e. the test discriminates.
    """
    d = _designer_with_fim(_F, n_mp=1)
    got = getattr(d, crit_name)(np.ones(3) / 3)
    got = float(np.squeeze(got[0] if isinstance(got, tuple) else got))
    assert np.isclose(got, expected, rtol=1e-12), (crit_name, got, expected)


def test_a_opt_at_one_parameter_is_positive_like_the_matrix_branch():
    """
    Section 38 of the capability suite asserts that a well-conditioned FIM
    gives "a finite POSITIVE score" for A-optimality. It uses n_mp=4, so it
    never reached the scalar branch -- which returned -FIM, i.e. NEGATIVE,
    violating the very convention the suite pins.
    """
    d = _designer_with_fim(_F, n_mp=1)
    val = float(np.squeeze(d._a_opt_criterion(np.ones(3) / 3)))
    assert np.isfinite(val) and val > 0.0, val


def test_d_opt_scalar_and_matrix_agree_on_a_block_diagonal_FIM():
    """
    Independent cross-check of the same rule, not another restatement of it:
    log det of diag(a, b) equals log(a) + log(b), so scoring a 2x2 must equal
    the sum of scoring each 1x1 block. This holds only if both sizes use the
    same formula.
    """
    a, b = 3.0, 11.0
    two = float(np.squeeze(
        _designer_with_fim(np.diag([a, b]), n_mp=2)._d_opt_criterion(np.ones(3) / 3)
    ))
    one_a = float(np.squeeze(
        _designer_with_fim(a, n_mp=1)._d_opt_criterion(np.ones(3) / 3)
    ))
    one_b = float(np.squeeze(
        _designer_with_fim(b, n_mp=1)._d_opt_criterion(np.ones(3) / 3)
    ))
    assert np.isclose(two, one_a + one_b, rtol=1e-12), (two, one_a, one_b)


def test_cvar_d_opt_scalar_matches_its_matrix_formula():
    d = _designer_with_fim(_F, n_mp=1)
    d._pseudo_bayesian = True
    got = float(d.cvar_d_opt_criterion(np.atleast_2d(_F)))
    assert np.isclose(got, -np.log(_F), rtol=1e-12), got


def test_d_opt_degenerate_fim_is_infeasible_not_zero():
    """
    The removed scalar branch caught the np.array([0]) sentinel _eval_fim can
    return and produced -0.0 -- the BEST value for a minimised criterion, i.e.
    a design that cannot be evaluated scoring better than every real one. Same
    trap a_opt's guard was added for in 0.5.0; d_opt never got it.
    """
    d = _designer_with_fim(_F, n_mp=1)
    d.fim = np.array([0])
    d.eval_fim = lambda x, store_predictions=True: d.fim
    assert d._d_opt_criterion(np.ones(3) / 3) == np.inf


# =====================================================================
# INVARIANT 2 -- apportion() allocates exactly n_exp runs and never
# returns a non-finite or negative count, whatever the support shape.
# =====================================================================
def _dynamic_designer(efforts, n_spt=3, verbose=0):
    """A solved dynamic design with sampling times OPTIMISED (no n_spt)."""
    efforts = np.asarray(efforts, dtype=float)
    n_c = efforts.shape[0]
    d = Designer()
    d._dynamic_system = True
    d._invariant_controls = True
    d._dynamic_controls = False
    d._opt_sampling_times = True
    d._specified_n_spt = False
    d._pseudo_bayesian = False
    d._regularize_fim = False
    d._prior_fim = None
    d._cvar_problem = False
    d._verbose = verbose
    d._save_atomics = False
    d._b_opt_apportion_redundant = False
    d.n_c = n_c
    d.n_spt = n_spt
    d.n_mp = 2
    d.ti_controls_names = np.array(["u"])
    d.ti_controls_candidates = np.arange(n_c, dtype=float).reshape(-1, 1)
    d.tv_controls_candidates = np.empty((n_c, 1))
    d.sampling_times_candidates = np.tile(
        np.linspace(0.1, 0.1 * n_spt, n_spt), (n_c, 1)
    )
    d.efforts = efforts
    return d


def _flat(app):
    return np.concatenate(
        [np.asarray(x, dtype=float).ravel() for x in np.atleast_1d(app)]
    )


# RAGGED support: candidate 0 carries effort at 3 times, candidate 1 at 2.
# This is the NORMAL outcome of optimising sampling times -- the optimiser has
# no reason to use the same NUMBER of times on every candidate -- and it was
# reproduced on the first attempt by three different real IPOPT solves.
_RAGGED = np.array([[0.3, 0.3, 0.2],
                    [0.1, 0.1, 0.0]])

# NON-ragged but still 2-D: both candidates carry effort at 2 times. This is
# the fixture for the greatest-effort defect, which did NOT require raggedness.
_WIDE = np.array([[0.3, 0.2, 0.0],
                  [0.3, 0.2, 0.0]])


@pytest.mark.parametrize("n_exp", [4, 5, 6, 8, 10, 12])
def test_ragged_support_never_yields_a_nonfinite_run_count(n_exp):
    """
    THE Item 24 regression. opt_eff used to be a RECTANGULAR
    (n_opt_c, max_n_opt_spt) array NaN-padded to a common width; ceil(NaN) is
    NaN, so padding survived _adams_apportionment and the final
    `astype(int)` turned it into INT64_MIN (-9223372036854775808) behind a
    single RuntimeWarning. Pre-fix these cases return negative run counts.
    """
    d = _dynamic_designer(_RAGGED)
    ret = d.apportion(n_exp=n_exp)
    flat_ret = _flat(ret)
    assert np.all(np.isfinite(_flat(d.apportionments)))
    assert np.all(flat_ret >= 0), flat_ret
    assert int(_flat(d.apportionments).sum()) == n_exp


@pytest.mark.parametrize("n_exp", [1, 2, 3])
def test_greatest_effort_branch_allocates_exactly_n_exp(n_exp):
    """
    The second, MORE severe defect, and it needed no raggedness at all --
    only two or more support times on some candidate.
    _greatest_effort_apportionment selects with np.where(work == nanmax)[0],
    which on a 2-D array returns ROW indices, so it assigned a whole
    candidate's row of sampling times at once: apportion(2) allocated SIX
    runs, printed "Run 3/2 Experiments", and reported a Kiefer bound of
    150.82% -- impossible for a bound that is at most 100% -- with no warning
    of any kind. Pre-fix the total here is a multiple of the budget.
    """
    d = _dynamic_designer(_WIDE)
    d.apportion(n_exp=n_exp)
    assert int(_flat(d.apportionments).sum()) == n_exp


def test_greatest_effort_apportionment_refuses_2d_input():
    """
    The helper is only correct on a 1-D vector of real supports. Refusing the
    shape is what stops a future caller from silently re-introducing the
    row-indexing bug; the only pre-existing direct test of this helper passed
    it a 1-D array, which is why the defect survived.
    """
    d = Designer()
    d._verbose = 0
    with pytest.raises(ValueError, match="1-D"):
        d._greatest_effort_apportionment(np.array([[0.3, 0.2], [0.3, 0.2]]), 2)
    # and the 1-D case still works
    out = d._greatest_effort_apportionment(np.array([0.5, 0.3, 0.2]), 2)
    assert out.sum() == 2


def test_kiefer_bound_never_exceeds_one():
    """
    A bound that says the rounded design is "at least 150.82% as good as the
    continuous design" is reporting an impossibility, and that is what the
    over-allocating branch produced. Cheap, and it fails loudly on the class
    of bug rather than on one instance of it.
    """
    for efforts in (_RAGGED, _WIDE):
        for n_exp in (2, 3, 5, 10):
            d = _dynamic_designer(efforts)
            d.apportion(n_exp=n_exp)
            assert d.epsilon <= 1.0 + 1e-9, (efforts.tolist(), n_exp, d.epsilon)


def test_epsilon_and_trimmed_are_not_gated_on_verbosity():
    """
    `trimmed` changes the RETURNED value, so gating it on verbosity made a
    documented parameter a silent no-op at the default verbose=0 -- along with
    self.epsilon and non_trimmed_apportionments, which stayed None. Only the
    printing is gated now.
    """
    quiet = _dynamic_designer(_RAGGED, verbose=0)
    quiet.apportion(n_exp=10, trimmed=False)
    assert quiet.epsilon is not None
    assert quiet.non_trimmed_apportionments is not None
    # trimmed=False means the full (n_c, n_spt) view, not the per-support one
    assert quiet.apportionments.shape == (2, 3)
    assert int(np.nansum(quiet.apportionments)) == 10


def test_verbose_zero_still_skips_the_criterion_evaluation_by_default():
    """
    The efficiency ratio needs two criterion evaluations, and nothing reads
    them when silent. compute_actual_efficiency is tri-state: None (default)
    means "compute it if it will be reported", so the default cost profile at
    verbose=0 is unchanged; passing True forces it.
    """
    d = _dynamic_designer(_RAGGED, verbose=0)
    d._current_criterion = "d_opt_criterion"

    def _boom(efforts):
        raise AssertionError("criterion must not be evaluated at verbose=0")

    d.d_opt_criterion = _boom
    d.apportion(n_exp=10)                      # default: must not evaluate
    assert d.rounding_efficiency is None

    calls = []
    d.d_opt_criterion = lambda efforts: (calls.append(1), -1.0)[1]
    d.apportion(n_exp=10, compute_actual_efficiency=True)
    assert len(calls) == 2, "both endpoints must be scored, not one"


def test_apportion_returns_counts_on_the_n_spt_branch():
    """
    The _specified_n_spt branch used to `return` bare, so
    `app = d.apportion(n)` handed back None for every n_spt design while the
    other branches returned an array: one function, two return contracts.
    """
    d = _dynamic_designer(np.array([[0.4, 0.3, 0.0], [0.3, 0.0, 0.0]]))
    d._specified_n_spt = True
    d._n_spt_spec = 2
    d.spt_candidates_combs = np.array([[[0, 1], [0, 2], [1, 2]],
                                       [[0, 1], [0, 2], [1, 2]]])
    ret = d.apportion(n_exp=10)
    assert ret is not None, "the n_spt branch must return the run counts"
    assert int(_flat(ret).sum()) == 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
