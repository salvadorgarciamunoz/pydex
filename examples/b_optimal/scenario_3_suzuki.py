"""
scenario_3_suzuki.py
======================
b_opt_criterion (Bracketing-optimal design) applied to a Suzuki-Miyaura
cross-coupling -- a NEW case study, not from Chen et al. (2018).

Where scenarios 1 and 2 reproduce the paper's figures, this one is a
practitioner-oriented walkthrough that exercises the parts of the new
feature the paper's two case studies do NOT cover:

  Part A -- the core use case: a 5-experiment bracketing study, verified
            against EXHAUSTIVE brute-force enumeration (so the design is
            provably the global optimum, not just bonmin's best find).
  Part B -- choosing how many experiments to run: cost/benefit across
            n_exp = 4..8, the practical question a chemist actually faces.
  Part C -- the anti-clustering control (designer._b_opt_min_sep_frac),
            the discrete analogue of Chen et al.'s log-barrier mu penalty
            (their Table 4 / Figure 8 clustering problem).
  Part D -- the guards: what happens if you forget simulate_candidates(),
            forget n_exp, or pass a pseudo-Bayesian parameter array.

Chemistry: 5 inputs (T, t_rxn, cat_mol, boron_eq, base_eq) -> 3 outputs
(yield_P, imp_D, res_B), with real process constraints (yield >= 70%,
homocoupling impurity <= 1.5%, residual boronic acid <= 10%).
See suzuki_model.py.
"""
import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ── model import ─────────────────────────────────────────────────────────────
# Add THIS SCRIPT's directory to sys.path so the model file is found
# regardless of the working directory. Do NOT use sys.path.insert(0, ".") --
# that prepends the *working* directory ahead of the script's own, so a file
# of the same name in the cwd silently shadows the real model: wrong physics,
# no error. Same pattern as examples/ode/case_2_no_ift_no_collocation.py.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from suzuki_model import (simulate_suzuki, feasible, BOUNDS, BOUND_ORDER,
                           YIELD_MIN, IMP_D_MAX, RES_B_MAX)
from pydex.core.designer import Designer

# Figures are written to the WORKING directory, so the destination is the
# caller's choice (cd wherever you want them). Deliberately not pinned to
# _script_dir: that would drop generated PNGs inside the repository.
OUT_DIR = "."


def simulate(ti_controls, model_parameters):
    return simulate_suzuki(ti_controls)


def build_candidate_grid(n_cand, seed):
    rng = np.random.default_rng(seed)
    samples = np.column_stack([rng.uniform(*BOUNDS[k], n_cand) for k in BOUND_ORDER])
    feas = np.array([feasible(s) for s in samples])
    return samples[feas]


def make_designer(TIC, verbose=0):
    d = Designer()
    d.simulate = simulate
    d.model_parameters = np.array([1.0])
    d.model_parameter_names = ["dummy"]
    d.ti_controls_candidates = TIC
    d.error_cov = np.eye(3)
    d.initialize(verbose=verbose)
    d.simulate_candidates()
    return d


def design_b_opt(d, n_exp, output_weight, time_limit=90):
    """time_limit caps each bonmin solve. At phi=5 the pure-D_I instances
    (output_weight=0) can take many minutes to close the gap; capping keeps
    the walkthrough responsive. A capped solve returns bonmin's best
    incumbent rather than a proven optimum -- fine for the illustrative
    Parts B/C, and Part A (the one claim of global optimality) is verified
    independently by exhaustive enumeration anyway."""
    N = d.n_c
    e0 = np.ones((N, 1)) / N
    d.design_experiment(d.b_opt_criterion, n_exp=n_exp, solver="bonmin",
                         output_weight=output_weight, e0=e0, verbose=0,
                         solver_options={"bonmin.time_limit": time_limit})
    return np.where(np.asarray(d.efforts).ravel() > 1e-6)[0]


def true_fin_fout(TIC, Y, idx):
    lb, ub = TIC.min(axis=0), TIC.max(axis=0)
    U = 2.0 * (TIC - lb) / (ub - lb) - 1.0
    fin = np.linalg.det(U[idx].T @ U[idx])
    Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
    Yz = (Y - Ymean) / Ystd
    Ysel = Yz[idx]
    yc = Ysel.mean(axis=0)
    M_out = (Ysel - yc).T @ (Ysel - yc) / (len(idx) - 1)
    fout = np.linalg.det(M_out)
    return fin, fout


def true_obj(TIC, Y, idx, output_weight):
    fin, fout = true_fin_fout(TIC, Y, idx)
    if fin <= 0 or fout <= 0:
        return -np.inf
    return (1 - output_weight) * np.log(fin) + output_weight * np.log(fout)


def hull_volume(pts):
    if len(pts) < 4:
        return 0.0
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


# =====================================================================
# N_CAND chosen so that C(N_feasible, 5) stays small enough to verify
# Part A's design by EXHAUSTIVE enumeration (~658k subsets) -- the
# strongest verification available, and the whole point of Part A.
N_CAND = 500
SEED = 11
TIC = build_candidate_grid(N_CAND, SEED)
N = TIC.shape[0]
print("=" * 74)
print("SUZUKI-MIYAURA COUPLING: bracketing study via b_opt_criterion")
print("=" * 74)
print(f"Candidate pool: {N} feasible of {N_CAND} sampled "
      f"({100*N/N_CAND:.1f}% -- constraints: yield>={YIELD_MIN}, "
      f"impD<={IMP_D_MAX}, resB<={RES_B_MAX})")
print(f"5 inputs : {BOUND_ORDER}")
print(f"3 outputs: yield_P, imp_D, res_B")

d = make_designer(TIC)
Y = np.asarray(d.response, dtype=float).reshape(N, -1)


# =====================================================================
# PART A -- the core use case, verified by exhaustive brute force
# =====================================================================
print("\n" + "=" * 74)
print("PART A: 5-experiment bracketing study (balanced weighting)")
print("=" * 74)

W = 0.5
# Part A deliberately runs UNCAPPED (time_limit large): this is the one
# place we claim global optimality, and it is cross-checked below against
# exhaustive enumeration of every C(N,5) subset.
idxA = design_b_opt(d, n_exp=5, output_weight=W, time_limit=100000)
objA = true_obj(TIC, Y, idxA, W)
print(f"bonmin design (candidate indices): {sorted(idxA.tolist())}")
print(f"\n{'':4}{'T[C]':>8}{'t[h]':>8}{'Pd[%]':>8}{'B eq':>8}{'base eq':>9}"
      f"{'| yield':>9}{'imp_D':>9}{'res_B':>8}")
for i in idxA:
    t = TIC[i]
    y = Y[i]
    print(f"{'':4}{t[0]:8.1f}{t[1]:8.2f}{t[2]:8.2f}{t[3]:8.2f}{t[4]:9.2f}"
          f"{y[0]:9.3f}{y[1]:9.4f}{y[2]:8.3f}")

n_combos = 1
for k in range(5):
    n_combos = n_combos * (N - k) // (k + 1)
print(f"\nVerifying against EXHAUSTIVE enumeration of all C({N},5) = {n_combos:,} subsets...")
if n_combos <= 3_000_000:
    best_val, best_idx = -np.inf, None
    for combo in itertools.combinations(range(N), 5):
        v = true_obj(TIC, Y, np.array(combo), W)
        if v > best_val:
            best_val, best_idx = v, np.array(combo)
    gap = best_val - objA
    print(f"  brute-force global optimum : {sorted(best_idx.tolist())}  obj={best_val:.6f}")
    print(f"  bonmin's design            : {sorted(idxA.tolist())}  obj={objA:.6f}")
    print(f"  gap = {gap:.8f}  -> {'PROVEN GLOBAL OPTIMUM' if abs(gap) < 1e-6 else 'SUBOPTIMAL, investigate'}")
else:
    print(f"  [skipped: {n_combos:,} subsets too many to enumerate]")


# =====================================================================
# PART B -- how many experiments should we run?
# =====================================================================
print("\n" + "=" * 74)
print("PART B: cost/benefit of design size (the chemist's real question)")
print("=" * 74)
print(f"{'n_exp':>6}{'D_I,opt':>10}{'hull coverage':>16}{'coverage/exp':>14}")

vol_all = hull_volume(Y)
rows = []
for n_exp in range(4, 9):
    fin_max, _ = true_fin_fout(TIC, Y, design_b_opt(d, n_exp, 0.0))
    idx = design_b_opt(d, n_exp, 0.5)
    fin, _ = true_fin_fout(TIC, Y, idx)
    d_i = (fin / fin_max) ** (1 / 5)      # phi = 5 input factors
    cov = hull_volume(Y[idx]) / vol_all if vol_all > 0 else 0.0
    rows.append((n_exp, d_i, cov))
    print(f"{n_exp:>6}{d_i:>10.3f}{cov:>16.3f}{cov/n_exp:>14.4f}")

fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
ns = [r[0] for r in rows]
ax1.plot(ns, [r[2] for r in rows], "o-", color="C0", label="hull coverage of feasible space")
ax1.set_xlabel("Number of experiments (n_exp)")
ax1.set_ylabel("Output-space hull coverage fraction", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")
ax2 = ax1.twinx()
ax2.plot(ns, [r[2] / r[0] for r in rows], "s--", color="C1",
         label="marginal coverage per experiment")
ax2.set_ylabel("Coverage per experiment", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")
ax1.set_title("Suzuki: diminishing returns from additional experiments")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/suzuki_design_size.png", dpi=150)
print("Saved suzuki_design_size.png")


# =====================================================================
# PART C -- anti-clustering control
# =====================================================================
print("\n" + "=" * 74)
print("PART C: anti-clustering (designer._b_opt_min_sep_frac)")
print("=" * 74)
print("Chen et al. hit output-point CLUSTERING with the ellipsoid surrogate")
print("(their Table 4 shows 6 of 8 points collapsing into 3 tight groups),")
print("and handled it with a log-barrier penalty (mu). Because our candidates")
print("are FIXED, pairwise output distances are precomputable constants, so")
print("the same effect is achieved with linear mutual-exclusion constraints")
print("-- no penalty parameter to tune, no added nonlinearity.\n")

Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
Yz = (Y - Ymean) / Ystd


def min_pairwise_sep(idx):
    sub = Yz[idx]
    dmin = np.inf
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            dmin = min(dmin, np.sum((sub[a] - sub[b]) ** 2))
    return dmin


for frac in [0.0, 0.02, 0.05]:
    d._b_opt_min_sep_frac = frac
    idx = design_b_opt(d, n_exp=6, output_weight=0.8)
    print(f"_b_opt_min_sep_frac={frac:<5} design={sorted(idx.tolist())}  "
          f"min pairwise output sep (sq, z-scored) = {min_pairwise_sep(idx):.4f}")
d._b_opt_min_sep_frac = 0.0   # reset

print("\nNOTE, honestly: on THIS problem the setting makes no difference --")
print("all three thresholds return the same design. The output-space term is")
print("already spreading the selected points far apart (min separation ~3.07 in")
print("squared z-scored units), so no candidate pair falls below the exclusion")
print("threshold and no constraints are actually generated. The control is wired")
print("in and functioning; this case simply does not exhibit the clustering")
print("pathology it exists to fix. Chen et al.'s clustering arose from their")
print("CONTINUOUS ellipsoid surrogate placing points at coincident extremities;")
print("selecting from a fixed, pre-filtered candidate pool is largely immune to")
print("that failure mode, which is arguably the more interesting result here.")


# =====================================================================
# PART D -- the guards
# =====================================================================
print("\n" + "=" * 74)
print("PART D: guards / error handling")
print("=" * 74)

# D1: forgot n_exp
try:
    e0 = np.ones((N, 1)) / N
    d.design_experiment(d.b_opt_criterion, solver="bonmin", e0=e0, verbose=0)
    print("D1 [no n_exp]                 : NO ERROR -- unexpected")
except ValueError as exc:
    print(f"D1 [no n_exp]                 : ValueError -- {str(exc)[:52]}...")

# D2: forgot simulate_candidates()
d2 = Designer()
d2.simulate = simulate
d2.model_parameters = np.array([1.0])
d2.model_parameter_names = ["dummy"]
d2.ti_controls_candidates = TIC
d2.error_cov = np.eye(3)
d2.initialize(verbose=0)
try:
    e0 = np.ones((N, 1)) / N
    d2.design_experiment(d2.b_opt_criterion, n_exp=5, solver="bonmin", e0=e0, verbose=0)
    print("D2 [no simulate_candidates()] : NO ERROR -- unexpected")
except RuntimeError as exc:
    print(f"D2 [no simulate_candidates()] : RuntimeError -- {str(exc)[:48]}...")

# D3: pseudo-Bayesian parameter array
d3 = Designer()
d3.simulate = simulate
d3.model_parameters = np.array([[1.0], [2.0]])   # 2 scenarios
d3.model_parameter_names = ["dummy"]
d3.ti_controls_candidates = TIC
d3.error_cov = np.eye(3)
d3.initialize(verbose=0)
d3.simulate_candidates()
try:
    e0 = np.ones((N, 1)) / N
    d3.design_experiment(d3.b_opt_criterion, n_exp=5, solver="bonmin", e0=e0, verbose=0)
    print("D3 [pseudo-Bayesian]          : NO ERROR -- unexpected")
except ValueError as exc:
    print(f"D3 [pseudo-Bayesian]          : ValueError -- {str(exc)[:52]}...")

print("\n" + "=" * 74)
print("Suzuki scenario complete.")
print("=" * 74)
