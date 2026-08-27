"""
`set_prior_experiments()` must work in the DOCUMENTED order:
`initialize()`, then `set_prior_experiments()`.

It did not. `self.feval_sensitivity` was initialised to None and only set to 0
inside `eval_sensitivities()`' per-candidate loop, while
`set_prior_experiments()` computes its own sensitivities through a path that
reaches `_sensitivity_sim_wrapper`'s unconditional
`self.feval_sensitivity += 1` without ever entering that loop. The result was

    TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'

re-raised as "Sensitivity computation failed for prior experiment 1/1", which
names neither the cause nor anything the user did.

Capability suite section 12 exercises `set_prior_experiments()` and passes,
because it receives a designer fixture that earlier sections have already run
`design_experiment()` on -- leaving the counter an int. The defect is invisible
to any ordering that designs first, which is why a green suite did not catch
it. Same shape as the other blind spots in this project: "not exercised"
reported as "correct".

Solver-free: `set_prior_experiments()` uses finite differences over
`simulate()` and needs no optimiser.
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

# k is deliberately SMALL (0.02). That is the magnitude the pre-0.3.0 flat
# base_step=2 perturbed by ~2.0 -- a hundred times the parameter's own value --
# and set_prior_experiments() is one of the two sites whose DEFAULT step no
# existing test covers, because the suite passes explicit base_step overrides
# in the sections that reach them.
K_TRUE, A_TRUE = 0.02, 1.5
SIGMA2 = 0.01 ** 2
U = 1.0
# long sampling times, where finite-difference error grows
SPT = np.array([1.0, 10.0, 40.0, 80.0])


def _simulate(ti_controls, sampling_times, model_parameters):
    u = ti_controls[0]
    k, a = model_parameters
    t = np.asarray(sampling_times, dtype=float)
    return (a * u * np.exp(-k * t)).reshape(-1, 1)


def _designer():
    d = Designer()
    d.simulate = _simulate
    d.model_parameters = np.array([K_TRUE, A_TRUE])
    d.model_parameter_names = ["k", "A"]
    d.error_cov = np.array([[SIGMA2]])
    d.response_names = ["y"]
    d.ti_controls_names = ["u"]
    d.ti_controls_candidates = np.array([[U], [2.0]])
    d.sampling_times_candidates = np.tile(SPT, (2, 1))
    d.initialize(verbose=0)
    return d


def _analytic_prior_fim():
    """
    Independent reference: sensitivities differentiated BY HAND, not by a
    second numerical path. Agreement between two numerical methods is weak
    evidence; agreement with a closed form is strong.

    pydex normalises sensitivities by parameter magnitude
    (`_norm_sens_by_params` is True by default), i.e. it works with
    d y / d ln(theta) = theta * d y / d theta, so the reference FIM must be
    scaled the same way: diag(theta) @ FIM @ diag(theta). Comparing against
    the UNSCALED form shows a 125% deviation and looks exactly like a broken
    FD step -- a false positive worth not repeating.
    """
    dy_dk = -A_TRUE * U * SPT * np.exp(-K_TRUE * SPT)
    dy_dA = U * np.exp(-K_TRUE * SPT)
    S = np.stack([dy_dk, dy_dA], axis=-1)          # (n_spt, n_mp)
    fim = np.zeros((2, 2))
    for i in range(len(SPT)):
        s = S[i].reshape(1, 2)
        fim += s.T @ (np.eye(1) / SIGMA2) @ s
    theta = np.array([K_TRUE, A_TRUE])
    return np.diag(theta) @ fim @ np.diag(theta)


def test_feval_sensitivity_is_an_int_from_construction():
    """The narrow regression: the accumulator must never be None."""
    d = Designer()
    assert isinstance(d.feval_sensitivity, int)
    assert d.feval_sensitivity == 0


def test_set_prior_experiments_works_directly_after_initialize():
    """
    The documented order, and the one that raised. No design_experiment() or
    eval_sensitivities() call first -- that is the whole point.
    """
    d = _designer()
    d.set_prior_experiments(
        ti_controls=np.array([[U]]),
        model_parameters=np.array([K_TRUE, A_TRUE]),
        sampling_times=np.array([SPT]),
    )
    assert d._prior_fim is not None
    assert np.linalg.matrix_rank(np.asarray(d._prior_fim, dtype=float)) == 2


def test_prior_fim_is_correct_at_the_default_fd_step():
    """
    Open Item 3: `set_prior_experiments()` is untested at its DEFAULT finite-
    difference step, because the suite passes explicit base_step overrides in
    the sections reaching it. This deliberately passes no override, on a model
    whose small parameter (0.02) is the case the pre-0.3.0 flat step got badly
    wrong, and checks against a closed form.
    """
    d = _designer()
    d.set_prior_experiments(
        ti_controls=np.array([[U]]),
        model_parameters=np.array([K_TRUE, A_TRUE]),
        sampling_times=np.array([SPT]),
    )
    got = np.asarray(d._prior_fim, dtype=float)
    ref = _analytic_prior_fim()
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
    assert rel.max() < 1e-8, (
        f"prior FIM deviates from the closed form by {rel.max():.3e}\n"
        f"got:\n{got}\nexpected:\n{ref}"
    )


def test_prior_fim_is_the_same_whether_or_not_sensitivities_ran_first():
    """
    The ordering that used to fail and the ordering the suite happens to use
    must agree. If they ever diverge, one of the two paths is carrying state
    it should not.
    """
    a = _designer()
    a.set_prior_experiments(
        ti_controls=np.array([[U]]),
        model_parameters=np.array([K_TRUE, A_TRUE]),
        sampling_times=np.array([SPT]),
    )

    b = _designer()
    b.eval_sensitivities(save_sensitivities=False)
    b.set_prior_experiments(
        ti_controls=np.array([[U]]),
        model_parameters=np.array([K_TRUE, A_TRUE]),
        sampling_times=np.array([SPT]),
    )

    np.testing.assert_allclose(
        np.asarray(a._prior_fim, dtype=float),
        np.asarray(b._prior_fim, dtype=float),
        rtol=1e-12, atol=0.0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
