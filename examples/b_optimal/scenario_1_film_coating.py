"""
scenario_1_film_coating.py
============================
Demonstrates b_opt_criterion (Bracketing-optimal design) on the tablet
film-coating case study from Chen, Paulavicius & Adjiman (2018), AIChE J.
64:3944-3957 -- the paper's MOTIVATING EXAMPLE and first case study.

Model: film_coating_model.py, a thermodynamic model of the coater that
predicts (T_air,out, %RH_out) from (T_air,in, M_coat, Q_air).

IMPORTANT on fidelity: this is an independent physically-grounded model, not
the equations given in the paper's Supporting Information. It uses the same
three input factors and two outputs over the same ranges, so the structure,
trends and figure types below correspond to the paper's -- but the absolute
numbers do not, and should not be compared against it.

Figures reproduced (paper numbering):
  Table 1  -- two similarly input-orthogonal 4-point designs, showing very
              different output-space coverage
  Figure 2 -- output-space coverage comparison for those two designs
  Figure 3 -- Pareto front (D_I-optimality vs scaled output-space area) for
              a 4-experiment design, swept over output_weight
  Figure 4 -- Pareto-front family as the number of experiments increases
              (4 to 8 points; 3 is below the n_exp rank bound)
  Figure 5 -- output space covered by a "compromise" 5-experiment design

All designs are produced with:  designer.design_experiment(designer.b_opt_criterion,
n_exp=..., solver="bonmin", output_weight=...)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

from film_coating_model import pred_exhaust
from pydex.core.designer import Designer

# Figures are written to the WORKING directory, so the destination is the
# caller's choice (cd wherever you want them). Deliberately not pinned to
# _script_dir: that would drop generated PNGs inside the repository.
OUT_DIR = "."


def simulate(ti_controls, model_parameters):
    """designer.simulate() contract. model_parameters is unused -- the
    coater is a fixed deterministic model, not something being fitted."""
    T_in, M_coat, Q_air = ti_controls
    T_exh, RH_exh, _, _ = pred_exhaust(T_in_C=T_in, Fair_in_CFM=Q_air, SolnFR_in_gpm=M_coat)
    return np.array([T_exh, RH_exh])


def build_candidate_grid(n_cand, seed):
    """Sample candidates over the paper's own ranges (motivating example,
    p.3945): T_air,in in [20,85] C, M_coat in [10,80] g/min, Q_air in
    [150,450] ft^3/min. Filtered to physically feasible (RH_exh <= 100)."""
    rng = np.random.default_rng(seed)
    T_in = rng.uniform(20, 85, n_cand)
    M_coat = rng.uniform(10, 80, n_cand)
    Q_air = rng.uniform(150, 450, n_cand)
    TIC = np.column_stack([T_in, M_coat, Q_air])
    feas = np.array([pred_exhaust(T_in_C=r[0], Fair_in_CFM=r[2], SolnFR_in_gpm=r[1])[1] <= 100.0
                      for r in TIC])
    return TIC[feas]


def make_designer(TIC, verbose=0):
    d = Designer()
    d.simulate = simulate
    d.model_parameters = np.array([1.0])   # dummy -- no fitted parameters
    d.model_parameter_names = ["dummy"]
    d.ti_controls_candidates = TIC
    d.error_cov = np.eye(2)
    d.initialize(verbose=verbose)
    d.simulate_candidates()                # populates d.response for b_opt
    return d


def design_b_opt(d, n_exp, output_weight, verbose=0):
    N = d.n_c
    e0 = np.ones((N, 1)) / N
    d.design_experiment(d.b_opt_criterion, n_exp=n_exp, solver="bonmin",
                         output_weight=output_weight, e0=e0)
    idx = np.where(np.asarray(d.efforts).ravel() > 1e-6)[0]
    return idx


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


def convex_hull_area(pts):
    if len(pts) < 3:
        return 0.0
    return ConvexHull(pts).volume  # 2D "volume" is the enclosed area


# ---------------------------------------------------------------------
# Setup: a moderately sized fixed candidate pool used for every figure
# ---------------------------------------------------------------------
N_CAND = 40
SEED = 7
TIC = build_candidate_grid(N_CAND, SEED)
N = TIC.shape[0]
print(f"Film-coating candidate pool: {N} feasible candidates (of {N_CAND} sampled)")
d = make_designer(TIC)
Y = np.asarray(d.response, dtype=float).reshape(N, -1)   # (N, 2): [T_exh, RH_exh]

# =========================================================================
# TABLE 1 / FIGURE 2 -- two similarly input-orthogonal designs, very
# different output coverage. Paper's Table 1: "Design 1" (D_I,opt ~100%,
# tiny output coverage) vs "Design 2" (D_I,opt ~99%, large output coverage)
# =========================================================================
print("\n" + "=" * 70)
print("TABLE 1 / FIGURE 2: input-orthogonal designs, output coverage compared")
print("=" * 70)

idx_input_only = design_b_opt(d, n_exp=4, output_weight=0.0)   # "Design 1"-like: pure D_I
fin_1, fout_1 = true_fin_fout(TIC, Y, idx_input_only)

idx_tradeoff = design_b_opt(d, n_exp=4, output_weight=0.3)     # "Design 2"-like: slight output weight
fin_2, fout_2 = true_fin_fout(TIC, Y, idx_tradeoff)

# scale D_I,opt relative to the pure-input optimum (Eq. 10 in the paper)
fin_max, _ = true_fin_fout(TIC, Y, idx_input_only)
d_i_opt_1 = (fin_1 / fin_max) ** (1 / 3)
d_i_opt_2 = (fin_2 / fin_max) ** (1 / 3)

print(f"{'Design':<25}{'D_I,opt (scaled)':<20}{'TIC (T_in, M_coat, Q_air)'}")
print(f"{'1 (input-only)':<25}{d_i_opt_1:<20.3f}")
for i in idx_input_only:
    print(f"  {TIC[i]}")
print(f"{'2 (slight output wt.)':<25}{d_i_opt_2:<20.3f}")
for i in idx_tradeoff:
    print(f"  {TIC[i]}")

# Figure 2: output-space coverage comparison, side-by-side convex hulls
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, idx, title in zip(
    axes, [idx_input_only, idx_tradeoff],
    [f"Design 1 (input-only)\ncoverage-blind by construction", "Design 2 (slight output weight)"]
):
    pts = Y[idx]
    ax.scatter(pts[:, 0], pts[:, 1], color="C0", zorder=3, s=60)
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        poly = Polygon(pts[hull.vertices], closed=True, facecolor="C0", alpha=0.25,
                        edgecolor="C0")
        ax.add_patch(poly)
    ax.scatter(Y[:, 0], Y[:, 1], color="gray", alpha=0.15, s=10, zorder=1,
               label="all candidates")
    ax.set_xlabel(r"$T_{air,out}$ [°C]")
    ax.set_ylabel("%RH_out [%]")
    ax.set_title(title, fontsize=10)
fig.suptitle("Figure 2 analogue: output-space coverage, two input-orthogonal designs")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/coater_figure2_output_coverage.png", dpi=150)
print("Saved coater_figure2_output_coverage.png")


# =========================================================================
# FIGURE 3 -- Pareto front for a 4-experiment design: scaled output-space
# area (convex hull, relative to the max achievable) vs D_I-optimality
# (relative to the pure-input optimum), swept over output_weight
# =========================================================================
print("\n" + "=" * 70)
print("FIGURE 3: Pareto front, 4-experiment design")
print("=" * 70)

fin_max4, _ = true_fin_fout(TIC, Y, design_b_opt(d, 4, 0.0))
idx_out_only4 = design_b_opt(d, 4, 1.0)
area_max4 = convex_hull_area(Y[idx_out_only4])

weights = np.round(np.arange(0.0, 1.0001, 0.1), 2)
fig3_points = []
for w in weights:
    idx = design_b_opt(d, n_exp=4, output_weight=float(w))
    fin, _ = true_fin_fout(TIC, Y, idx)
    area = convex_hull_area(Y[idx])
    d_i_opt = (fin / fin_max4) ** (1 / 3)
    area_scaled = area / area_max4 if area_max4 > 0 else 0.0
    fig3_points.append((w, d_i_opt, area_scaled, tuple(sorted(idx.tolist()))))
    print(f"output_weight={w:4.1f}  D_I,opt={d_i_opt:6.3f}  "
          f"scaled_hull_area={area_scaled:6.3f}  design={sorted(idx.tolist())}")

fig, ax = plt.subplots(figsize=(6, 5))
d_i_vals = [p[1] for p in fig3_points]
area_vals = [p[2] for p in fig3_points]
order = np.argsort(d_i_vals)
ax.plot(np.array(d_i_vals)[order], np.array(area_vals)[order], "o-", color="C0")
ax.set_xlabel(r"$D_I$-optimality (scaled)")
ax.set_ylabel("Scaled output-space area (convex hull)")
ax.set_title("Figure 3 analogue: Pareto front, 4-experiment coater design")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/coater_figure3_pareto_4exp.png", dpi=150)
print("Saved coater_figure3_pareto_4exp.png")


# =========================================================================
# FIGURE 4 -- Pareto-front family as the number of experiments grows
# (3 to 8), x-axis = "equivalent number of D_I-optimal experiments"
# =========================================================================
print("\n" + "=" * 70)
print("FIGURE 4: Pareto-front family, n_exp = 3..8")
print("=" * 70)

fig, ax = plt.subplots(figsize=(7, 5.5))
# Sweep starts at 4, not 3. With phi=3 input factors and n_resp=2 responses the
# implemented bound is n_exp >= max(phi, n_resp + 2) = 4: at n_exp=3 the centered
# output covariance has no rank margin above the Cholesky floor and the MINLP is
# strictly infeasible. design_experiment() rejects it up front.
for n_exp in range(4, 9):
    fin_max_n, _ = true_fin_fout(TIC, Y, design_b_opt(d, n_exp, 0.0))
    idx_out_only_n = design_b_opt(d, n_exp, 1.0)
    area_max_n = convex_hull_area(Y[idx_out_only_n])

    pts = []
    for w in weights:
        idx = design_b_opt(d, n_exp=n_exp, output_weight=float(w))
        fin, _ = true_fin_fout(TIC, Y, idx)
        area = convex_hull_area(Y[idx])
        # "equivalent number of D_I-optimal experiments": D_I,opt^(1/phi) * n_exp
        # matches the paper's x-axis convention (Fig. 4 caption)
        d_i_opt = (fin / fin_max_n) ** (1 / 3)
        n_equiv = d_i_opt * n_exp
        area_scaled = area / area_max_n if area_max_n > 0 else 0.0
        pts.append((n_equiv, area_scaled))
    pts = sorted(pts)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "o-", label=f"{n_exp}pt", markersize=5)

ax.set_xlabel("Number of equivalent D_I-optimal experiments")
ax.set_ylabel("Convex hull coverage fraction")
ax.set_title("Figure 4 analogue: Pareto-front family vs. number of experiments")
ax.legend(title="Design size")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/coater_figure4_pareto_family.png", dpi=150)
print("Saved coater_figure4_pareto_family.png")


# =========================================================================
# FIGURE 5 -- output space covered by a "compromise" 5-experiment design
# =========================================================================
print("\n" + "=" * 70)
print("FIGURE 5: output space covered by a 5-experiment compromise design")
print("=" * 70)

idx5 = design_b_opt(d, n_exp=5, output_weight=0.5)
fin5, _ = true_fin_fout(TIC, Y, idx5)
fin_max5, _ = true_fin_fout(TIC, Y, design_b_opt(d, 5, 0.0))
idx_out_only5 = design_b_opt(d, 5, 1.0)
area_max5 = convex_hull_area(Y[idx_out_only5])
area5 = convex_hull_area(Y[idx5])
print(f"5-experiment compromise design: {sorted(idx5.tolist())}")
print(f"  D_I,opt (scaled)        = {(fin5/fin_max5)**(1/3):.3f}")
print(f"  output area (scaled)   = {area5/area_max5:.3f}")

fig, ax = plt.subplots(figsize=(6, 5))
pts = Y[idx5]
ax.scatter(pts[:, 0], pts[:, 1], color="C0", s=70, zorder=3, label="selected design")
if len(pts) >= 3:
    hull = ConvexHull(pts)
    ax.add_patch(Polygon(pts[hull.vertices], closed=True, facecolor="C0",
                          alpha=0.25, edgecolor="C0"))
# outer boundary: convex hull of ALL feasible candidates (approx. of the
# "largest possible output space", paper's Figure 5 outer line)
hull_all = ConvexHull(Y)
ax.plot(np.append(Y[hull_all.vertices, 0], Y[hull_all.vertices[0], 0]),
        np.append(Y[hull_all.vertices, 1], Y[hull_all.vertices[0], 1]),
        "r-", alpha=0.6, label="largest possible output space")
ax.set_xlabel(r"$T_{air,out}$ [°C]")
ax.set_ylabel("%RH_out [%]")
ax.set_title("Figure 5 analogue: 5-experiment compromise design", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/coater_figure5_5exp_compromise.png", dpi=150)
print("Saved coater_figure5_5exp_compromise.png")

print("\nDone. Film-coating scenario complete.")
