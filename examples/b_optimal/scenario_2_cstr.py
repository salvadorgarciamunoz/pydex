"""
scenario_2_cstr.py
====================
Demonstrates b_opt_criterion (Bracketing-optimal design) on the second
case study from Chen, Paulavicius & Adjiman (2018), AIChE J. 64:3944-3957:
two continuously stirred tank reactors (CSTRs) in series.

Model: cstr_model.py -- a reproduction of the paper's two-CSTR-in-series
kinetic/energy model, transcribed from the GAMS source published with the
paper (reactor_code/reactor.dat.gms, reactor.eqn.gms; see the paper's Data
Statement). Same 6 inputs, same 3 outputs, same bounds, same constraints
([A]0, [A]0/[D]0, q0, V1, V2, T0 -> xC2 <= 0.002, xA2 <= 0.02, T2 <= 85 C)
as the paper -- this time with the paper's actual numbers, not
representative placeholders.

Figures reproduced (paper numbering):
  Figure 8 -- convex hull formed by an 8-experiment design's output points,
              compared to the total feasible output region
  Figure 9 -- Pareto-front approximation: output-space (convex hull)
              coverage vs. number of equivalent D_I-optimal experiments,
              for n_exp = 6, 7, 8

All designs are produced with:  designer.design_experiment(designer.b_opt_criterion,
n_exp=..., solver="bonmin", output_weight=...)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

from cstr_model import simulate_cstr, feasible, BOUNDS, BOUND_ORDER
from pydex.core.designer import Designer

# Figures are written to the WORKING directory, so the destination is the
# caller's choice (cd wherever you want them). Deliberately not pinned to
# _script_dir: that would drop generated PNGs inside the repository.
OUT_DIR = "."


def simulate(ti_controls, model_parameters):
    return simulate_cstr(ti_controls)


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
    d.ti_controls_names = ["A0", "ratio", "q0", "V1", "V2", "T0"]
    d.response_names = ["xC2", "xA2", "T2"]
    d.ti_controls_candidates = TIC
    d.error_cov = np.eye(3)
    d.initialize(verbose=verbose)
    d.simulate_candidates()
    return d


def design_b_opt(d, n_exp, output_weight, verbose=0):
    N = d.n_c
    e0 = np.ones((N, 1)) / N
    d.design_experiment(d.b_opt_criterion, n_exp=n_exp, solver="bonmin",
                         output_weight=output_weight, e0=e0)
    return np.where(np.asarray(d.efforts).ravel() > 1e-6)[0]


def true_fin_fout(TIC, Y, idx):
    lb, ub = TIC.min(axis=0), TIC.max(axis=0)
    U = 2.0 * (TIC - lb) / (ub - lb) - 1.0
    fin = np.linalg.det(U[idx].T @ U[idx])
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    Yz = (Y - Y_mean) / Y_std
    Ysel = Yz[idx]
    yc = Ysel.mean(axis=0)
    M_out = (Ysel - yc).T @ (Ysel - yc) / (len(idx) - 1)
    fout = np.linalg.det(M_out)
    return fin, fout


def convex_hull_volume_3d(pts):
    if len(pts) < 4:
        return 0.0
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# NOTE: with phi=6 input factors (vs. phi=3 for the coater), the binary
# subset-selection MINLP is substantially harder combinatorially --
# individual bonmin solves on a 246-candidate pool took minutes rather
# than seconds. Scaled down accordingly for a tractable demonstration.
N_CAND = 15000
SEED = 3
TIC = build_candidate_grid(N_CAND, SEED)
N = TIC.shape[0]
print(f"CSTR candidate pool: {N} feasible candidates (of {N_CAND} sampled)")
d = make_designer(TIC)
Y = np.asarray(d.response, dtype=float).reshape(N, -1)   # (N, 3): [xC2, xA2, T2]
print("6 inputs:", BOUND_ORDER)
print("3 outputs: xC2, xA2, T2")

# =========================================================================
# FIGURE 8 -- convex hull formed by an 8-experiment design's output points
# vs the total feasible output region
# =========================================================================
print("\n" + "=" * 70)
print("FIGURE 8: convex hull, 8-experiment design vs. total feasible region")
print("=" * 70)

idx8 = design_b_opt(d, n_exp=8, output_weight=0.7)
print(f"8-experiment design (output_weight=0.7): {sorted(idx8.tolist())}")
for i in idx8:
    print(f"  xC2={Y[i,0]:.5f}  xA2={Y[i,1]:.5f}  T2={Y[i,2]:.2f}")

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Y[:, 1], Y[:, 0], Y[:, 2], color="gray", alpha=0.12, s=8,
           label="all feasible candidates")
pts8 = Y[idx8][:, [1, 0, 2]]   # reorder to (xA2, xC2, T2) matching paper's axis convention
ax.scatter(pts8[:, 0], pts8[:, 1], pts8[:, 2], color="C0", s=60, label="8-experiment design")
try:
    hull = ConvexHull(pts8)
    for simplex in hull.simplices:
        tri = Poly3DCollection([pts8[simplex]], alpha=0.15, facecolor="C0", edgecolor="C0")
        ax.add_collection3d(tri)
except Exception as e:
    print(f"  (hull rendering skipped: {e})")
ax.set_xlabel(r"$x_{A2}$")
ax.set_ylabel(r"$x_{C2}$")
ax.set_zlabel(r"$T_2$ [°C]")
ax.set_title("Figure 8 analogue: convex hull (8-experiment design) vs.\ntotal feasible output region")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/cstr_figure8_convex_hull.png", dpi=150)
print("Saved cstr_figure8_convex_hull.png")


# =========================================================================
# FIGURE 9 -- Pareto-front approximation: convex hull coverage fraction vs.
# number of equivalent D_I-optimal experiments, n_exp = 6, 7, 8
# =========================================================================
print("\n" + "=" * 70)
print("FIGURE 9: Pareto-front approximation, n_exp = 6, 7, 8")
print("=" * 70)

weights = np.round(np.arange(0.0, 1.0001, 0.2), 2)   # coarser: 6 points instead of 11
fig, ax = plt.subplots(figsize=(7, 5.5))
for n_exp in [6, 7, 8]:
    fin_max_n, _ = true_fin_fout(TIC, Y, design_b_opt(d, n_exp, 0.0))
    idx_out_only_n = design_b_opt(d, n_exp, 1.0)
    vol_max_n = convex_hull_volume_3d(Y[idx_out_only_n])

    pts = []
    for w in weights:
        idx = design_b_opt(d, n_exp=n_exp, output_weight=float(w))
        fin, _ = true_fin_fout(TIC, Y, idx)
        vol = convex_hull_volume_3d(Y[idx])
        d_i_opt = (fin / fin_max_n) ** (1 / 6)   # phi=6 input factors
        n_equiv = d_i_opt * n_exp
        vol_scaled = vol / vol_max_n if vol_max_n > 0 else 0.0
        pts.append((n_equiv, vol_scaled))
        print(f"  n_exp={n_exp} w={w:4.1f}  n_equiv={n_equiv:5.2f}  "
              f"hull_coverage={vol_scaled:6.3f}  design={sorted(idx.tolist())}")
    pts = sorted(pts)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "o-", label=f"{n_exp}pt", markersize=5)

ax.set_xlabel("Number of equivalent D_I-optimal experiments")
ax.set_ylabel("Convex hull coverage fraction")
ax.set_title("Figure 9 analogue: Pareto-front approximation, reactor case study")
ax.legend(title="Design size")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/cstr_figure9_pareto.png", dpi=150)
print("Saved cstr_figure9_pareto.png")

print("\nDone. CSTR scenario complete.")
