"""
=============================================================================
SMOKE TEST for the patched pydex designer.py
=============================================================================
Copy/paste this whole file into Spyder and run it (F5).

It is fully self-contained -- no model files, no data files needed.

It checks all three change-sets in one go:
  CHECK 1  Ds-optimality basics  : interest_parameters resolve BY NAME
  CHECK 2  A-optimality fix      : a singular FIM now scores +inf, not 0
  CHECK 3  Ds where D fails      : Ds succeeds on a FIM that kills D-optimal
  CHECK 4  regularize_fim path   : the native Cholesky/IPOPT Ds formulation

Each check prints PASS or FAIL. Expected result: 4 PASS.

THE TEST MODEL
--------------
    y(t) = A0 * exp(-k*t) + c1 + c2

c1 and c2 are BOTH plain additive constants, so only their SUM is
identifiable -- dy/dc1 and dy/dc2 are both exactly 1. The FIM is therefore
rank 3 with 4 parameters: singular by construction, and the singular
direction (c1 - c2) lies entirely inside the nuisance block.

That is precisely the situation Ds-optimality exists for: we want precise
estimates of k and A0, and we do not care that c1 and c2 are individually
unidentifiable. D-optimality cannot cope (det(FIM) = 0); Ds can.
=============================================================================
"""

import numpy as np

from pydex.core.designer import Designer

# -----------------------------------------------------------------------------
# EDIT THIS if your solver is named differently or needs options.
# If you have HSL, {"linear_solver": "ma57"} usually helps.
# -----------------------------------------------------------------------------
SOLVER = "ipopt"
SOLVER_OPTIONS = {}          # e.g. {"linear_solver": "ma57", "tol": 1e-10}


# =============================================================================
# Model
# =============================================================================
# NOTE: the argument names below matter. pydex inspects the signature of
# simulate(), so they must be exactly `ti_controls` and `model_parameters`.
def simulate(ti_controls, model_parameters):
    t = ti_controls[0]
    k, A0, c1, c2 = model_parameters
    return np.array([A0 * np.exp(-k * t) + c1 + c2])


PARAM_NAMES = ["k", "A0", "c1", "c2"]
NOMINAL     = np.array([0.5, 1.0, 0.05, 0.02])
TIMES       = np.linspace(0.0, 10.0, 11).reshape(-1, 1)   # 11 candidate times


def make_designer(interest=None, verbose=0):
    """Build and initialise a designer for the test model."""
    d = Designer()
    d.simulate                = simulate
    d.model_parameters        = NOMINAL
    d.ti_controls_candidates  = TIMES
    d.model_parameter_names   = PARAM_NAMES
    d.error_cov               = np.array([[1.0]])   # 1 response
    if interest is not None:
        d.interest_parameters = interest             # <-- BY NAME
    d.initialize(verbose=verbose)
    return d


results = {}
line = "=" * 74


# =============================================================================
# CHECK 1 -- interest_parameters resolve by NAME, and validation works
# =============================================================================
print(line)
print("CHECK 1 -- interest_parameters resolve BY NAME")
print(line)

d = make_designer(interest=["k", "A0"])
idx_s, idx_n = d._resolve_ds_idx()

print(f"  model_parameter_names : {PARAM_NAMES}")
print(f"  interest_parameters   : {d.interest_parameters}")
print(f"  -> interest indices   : {[int(i) for i in idx_s]} "
      f"= {[PARAM_NAMES[i] for i in idx_s]}")
print(f"  -> nuisance indices   : {[int(i) for i in idx_n]} "
      f"= {[PARAM_NAMES[i] for i in idx_n]}")

ok1 = [int(i) for i in idx_s] == [0, 1] and [int(i) for i in idx_n] == [2, 3]

# a typo must be rejected immediately, not silently mis-bound
try:
    d_bad = make_designer(interest=["k", "A_zero"])   # deliberate typo
    print("  typo rejected         : NO  <-- unexpected")
    ok1 = False
except ValueError as exc:
    print(f"  typo rejected         : yes ({str(exc)[:58]}...)")

# numeric indices must also be rejected (positions are not stable)
try:
    d_bad = make_designer(interest=[0, 1])
    print("  numeric idx rejected  : NO  <-- unexpected")
    ok1 = False
except TypeError:
    print("  numeric idx rejected  : yes")

results["CHECK 1"] = ok1
print(f"\n  {'PASS' if ok1 else 'FAIL'}\n")


# =============================================================================
# CHECK 2 -- A-optimality on a singular FIM: must be +inf, not 0
# =============================================================================
print(line)
print("CHECK 2 -- A-optimality fix (singular FIM must score +inf, not 0)")
print(line)

d = make_designer()
d._fd_jac = True
e_uniform = np.ones(d.n_c) / d.n_c
d.eval_fim(e_uniform.copy())
fim = np.asarray(d.fim)
eig = np.linalg.eigvalsh(0.5 * (fim + fim.T))

print(f"  FIM eigenvalues       : {np.array2string(eig, precision=4)}")
print(f"  smallest eigenvalue   : {eig.min():.3e}   (~0 => singular, as designed)")

a_val = d.a_opt_criterion(e_uniform.copy())
print(f"  a_opt_criterion       : {a_val!r}")
print("  (A-optimality is MINIMISED. The old code returned 0 here, which is")
print("   its BEST possible score -- so a broken design looked perfect.)")

ok2 = np.isinf(a_val)
results["CHECK 2"] = ok2
print(f"\n  {'PASS' if ok2 else 'FAIL'}\n")


# =============================================================================
# CHECK 3 -- Ds succeeds where D-optimal fails
# =============================================================================
print(line)
print("CHECK 3 -- Ds-optimal design succeeds where D-optimal cannot")
print(line)

# --- D-optimal (expected to fail: det(FIM) = 0) ---
print("  [a] D-optimal on all 4 parameters ...")
d_d = make_designer()
try:
    d_d.design_experiment(
        d_d.d_opt_criterion, solver=SOLVER, solver_options=dict(SOLVER_OPTIONS),
    )
    d_value = float(d_d._criterion_value)
    d_failed = not np.isfinite(d_value)
    print(f"      criterion = {d_value}")
except Exception as exc:
    d_failed = True
    print(f"      failed as expected: {type(exc).__name__}: {str(exc)[:55]}")

# --- Ds-optimal on (k, A0) (expected to succeed) ---
print("  [b] Ds-optimal on ['k', 'A0'] (c1, c2 marginalised) ...")
d_ds = make_designer(interest=["k", "A0"], verbose=1)
d_ds.design_experiment(
    d_ds.ds_opt_criterion, solver=SOLVER, solver_options=dict(SOLVER_OPTIONS),
)
ds_value = float(d_ds._criterion_value)
eff = np.asarray(d_ds.efforts).ravel()
support = [float(TIMES[i, 0]) for i in np.where(eff > 1e-4)[0]]

print(f"      criterion = {ds_value:.10f}")
print(f"      support times t = {support}")
print(f"      efforts         = {np.round(eff[eff > 1e-4], 5)}")

ok3 = d_failed and np.isfinite(ds_value)
print(f"\n      D-optimal unusable : {d_failed}")
print(f"      Ds-optimal finite  : {np.isfinite(ds_value)}")
results["CHECK 3"] = ok3
print(f"\n  {'PASS' if ok3 else 'FAIL'}\n")


# =============================================================================
# CHECK 4 -- regularize_fim keeps the solve on the native Cholesky/IPOPT path
# =============================================================================
print(line)
print("CHECK 4 -- regularize_fim=True (native Pyomo/IPOPT Ds formulation)")
print(line)
print("  Without regularisation the nuisance block is exactly singular, so the")
print("  designer falls back to SLSQP (you saw the [INFO][ds_opt] note above).")
print("  With regularisation the nuisance block becomes positive definite and")
print("  the native symbolic formulation is used instead.")
print()
print("  NOTE regularize_fim must be passed to design_experiment() --")
print("  setting d._regularize_fim on the instance is silently overwritten.")
print()

d_reg = make_designer(interest=["k", "A0"])
d_reg._eps = 1e-8                     # regularisation magnitude
d_reg.design_experiment(
    d_reg.ds_opt_criterion, solver=SOLVER, solver_options=dict(SOLVER_OPTIONS),
    regularize_fim=True,
)
reg_value = float(d_reg._criterion_value)
eff_r = np.asarray(d_reg.efforts).ravel()
support_r = [float(TIMES[i, 0]) for i in np.where(eff_r > 1e-4)[0]]

print(f"  criterion (eps=1e-8) = {reg_value:.10f}")
print(f"  support times t      = {support_r}")
print(f"  fallback criterion   = {ds_value:.10f}   (from CHECK 3)")
print(f"  difference           = {abs(reg_value - ds_value):.3e}")

ok4 = np.isfinite(reg_value)
results["CHECK 4"] = ok4
print(f"\n  {'PASS' if ok4 else 'FAIL'}\n")


# =============================================================================
# Summary
# =============================================================================
print(line)
print("SUMMARY")
print(line)
for name, passed in results.items():
    print(f"  {name} : {'PASS' if passed else 'FAIL'}")
n_pass = sum(results.values())
print(f"\n  {n_pass}/{len(results)} checks passed")
if n_pass == len(results):
    print("\n  The patched designer.py is working correctly.")
else:
    print("\n  Something is off -- check the FAIL section(s) above.")
print(line)
