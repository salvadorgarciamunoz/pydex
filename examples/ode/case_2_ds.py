from pydex.core.designer import Designer
from case_2_model import simulate, build_pyomo_model
import numpy as np

"""
case_2_ds.py
============
Ds-OPTIMAL design: designing for a SUBSET of the model parameters.

Same model, candidate grid and nominal parameters as case_2.py -- only the
criterion changes.  Read case_2.py first if you have not.

    dCA/dt = -k * CA^alpha
    dCB/dt =  nu * k * CA^alpha
    k      = exp(theta_0 + theta_1 * (T - 273.15) / T)

    parameters : [theta_0, theta_1, alpha, nu]
    factors    : [CA0, T]
    responses  : [CA, CB]


WHY YOU WOULD USE Ds HERE
-------------------------
Not because anything is broken.  This model's FIM is perfectly healthy:

    rank 4/4,  cond = 1.46e+03,  no unidentifiable direction

and D-optimality works fine on it.  Ds is the right tool for a different
reason: you usually do not care equally about all four parameters.

    theta_0, theta_1  -- the Arrhenius group.  These are the kinetics, they
                         transfer to other reactors and other conditions, and
                         they are what you are actually trying to learn.
    alpha, nu         -- reaction order and stoichiometric coefficient.  They
                         must be estimated because the model needs them, but
                         their precision is of no interest: nu in particular is
                         usually fixed by the reaction chemistry.

D-optimality maximises det(FIM), which weights all four equally.  Ds-optimality
maximises the determinant of the Schur complement over the INTEREST parameters
only, marginalising the nuisance parameters out.  You are spending your
experimental budget on the question you care about instead of spreading it
across four questions, one of which you did not ask.


WHAT THIS SCRIPT SHOWS
----------------------
  1. The FIM is healthy -- Ds is a CHOICE here, not a rescue
  2. Ds selects a materially different design from D
  3. The trade, quantified: joint interest precision vs total information
  4. A TRAP: per-parameter standard deviations will mislead you here
  5. Verification that the criterion is what it claims to be
  6. When Ds IS a rescue, and how to tell


A NOTE ON THE SAMPLING GRID
---------------------------
This model normalises time by tau = max(sampling_times), which makes the
sampling grid a trap on collocation models -- np.linspace(0.001, 200, 11) once
silently corrupted the solve.  See the PITFALL section of case_2_model.py.  The
model guards against it now and warns when the guard engages.
"""

PARAM_NAMES = [r"$\theta_0$", r"$\theta_1$", r"$\alpha$", r"$\nu$"]
PLAIN       = ["theta_0", "theta_1", "alpha", "nu"]
INTEREST    = PARAM_NAMES[:2]        # theta_0, theta_1
I_IDX       = [0, 1]                 # interest positions
N_IDX_      = [2, 3]                 # nuisance = the complement

pre_exp_constant = 0.1
activ_energy     = 5000.0
R, T_ref         = 8.314159, 273.15
theta_0 = np.log(pre_exp_constant) - activ_energy / (R * T_ref)
theta_1 = activ_energy / (R * T_ref)
theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])

SOLVER         = "ipopt"
SOLVER_OPTIONS = {}   # add {"linear_solver": "ma57"} if you have HSL


def build_designer(interest_parameters=None):
    """A designer configured exactly as in case_2.py, optionally with an
    interest-parameter subset declared for Ds-optimal design."""
    d = Designer()
    d.simulate       = simulate
    d.pyomo_model_fn = build_pyomo_model
    d.model_parameters = theta_nom

    d.ti_controls_candidates = d.enumerate_candidates(
        bounds=[[1, 5], [273.15, 323.15]],
        levels=[5, 5],
    )
    d.sampling_times_candidates = np.array([
        np.linspace(0.001, 200, 11)
        for _ in d.ti_controls_candidates
    ])
    d.measurable_responses  = [0, 1]
    d.response_names        = ["$c_A$", "$c_B$"]
    d.model_parameter_names = PARAM_NAMES
    d.error_cov             = np.diag([0.1, 0.1])
    d.ti_controls_names     = ["CA0", "T"]

    # ── the only line that differs from a D-optimal script ────────────────
    # Parameters are selected BY NAME.  A parameter's position in the FIM
    # follows the order of model_parameters, which need not match the order a
    # Pyomo model declares its variables -- so position is not a stable
    # identifier.  An unknown name raises immediately rather than silently
    # binding to the wrong parameter.
    if interest_parameters is not None:
        d.interest_parameters = interest_parameters

    d.initialize(verbose=0)
    return d


def banner(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


def summarise(d, efforts, tol=1e-4):
    """Support candidates and their total effort."""
    n_spt = d.n_spt
    out = []
    for c in range(d.n_c):
        tot = efforts[c * n_spt:(c + 1) * n_spt].sum()
        if tot > tol:
            out.append((c + 1, np.asarray(d.ti_controls_candidates)[c], tot))
    return out


def covariance(d, efforts):
    """Parameter covariance = FIM^-1 (pseudo-inverse for safety)."""
    d._fd_jac = True
    d.eval_fim(np.asarray(efforts).copy())
    F = np.asarray(d.fim)
    return F, np.linalg.pinv(F, rcond=1e-12)


# =============================================================================
banner("1. The FIM is healthy — Ds here is a choice, not a rescue")
# =============================================================================
d0 = build_designer()
d0.eval_sensitivities(save_sensitivities=False)
n_eff = d0.n_c * d0.n_spt
uniform = np.ones(n_eff) / n_eff
F0, _ = covariance(d0, uniform)

w, V = np.linalg.eigh(0.5 * (F0 + F0.T))
print(f"  rank {np.linalg.matrix_rank(F0)}/{d0.n_mp}   "
      f"cond {np.linalg.cond(F0):.3e}")
print()
print(f"    {'eigenvalue':>12}  " + "  ".join(f"{p:>9}" for p in PLAIN))
for j in range(len(w)):
    print(f"    {w[j]:>12.4e}  "
          + "  ".join(f"{abs(V[i, j]):>9.4f}" for i in range(len(w))))

diag = d0.diagnose_fim_structure(report=False)
print()
print(f"  diagnose_fim_structure: singular = {diag['singular']}, "
      f"rank {diag['rank']}/{diag['n_mp']}")
print()
print("  Every parameter carries real weight in a well-separated eigenvalue --")
print("  nu dominates the 1.09e+01 direction, so it is comfortably identified.")
print("  There is no unidentifiable direction to marginalise away, and")
print("  design_experiment() will NOT refuse the D-optimal problem.")
print()
print("  This matters because Ds is often introduced as a rescue for singular")
print("  FIMs.  That is one use.  The use demonstrated here is the ordinary")
print("  one: you have four estimable parameters and you care about two.")


# =============================================================================
banner("2. D-optimal vs Ds-optimal: two different designs")
# =============================================================================
runs = {}
for label, interest, criterion in (
        ("D",  None,     "d_opt_criterion"),
        ("Ds", INTEREST, "ds_opt_criterion")):
    d = build_designer(interest)
        # Sampling times are OPTIMIZED here (no n_spt): effort is spent per
        # (condition, time) cell, so the optimiser chooses which listed times
        # to measure and may leave most at zero effort. Pass n_spt=k for
        # exactly k samples per run, or n_spt=<number of listed times> to FIX
        # the grid so every one is measured (see case_2.py round 1).
    d.design_experiment(
        getattr(d, criterion),
        solver=SOLVER,
        solver_options=dict(SOLVER_OPTIONS),
        write=False,
    )
    runs[label] = (d, np.asarray(d.efforts).ravel().copy(),
                   float(d._criterion_value))

for label in ("D", "Ds"):
    d, eff, val = runs[label]
    tag = "all four parameters" if label == "D" else f"interest {PLAIN[:2]}"
    print(f"\n  {label}-optimal  ({tag})   criterion = {val:.8f}")
    for cand, tic, tot in summarise(d, eff):
        print(f"     candidate {cand:>2}   CA0 = {tic[0]:>4.1f}, "
              f"T = {tic[1]:>7.2f}   effort {tot * 100:>6.2f}%")

print()
print("  D concentrates on two candidates, both at the highest CA0.  Ds spreads")
print("  across four, adding the CA0 = 4 pair.  That is a materially different")
print("  experiment, not a re-weighting of the same one.")


# =============================================================================
banner("3. The trade, quantified")
# =============================================================================
# Ds maximises det of the Schur complement over the interest block, which is
# equivalent to MINIMISING det of the interest block of the covariance matrix.
# Geometrically that is the squared area of the joint confidence ellipse for
# (theta_0, theta_1). It is a JOINT measure, not a per-parameter one.
stats = {}
for label in ("D", "Ds"):
    d, eff, _ = runs[label]
    F, C = covariance(d, eff)
    det_interest = float(np.linalg.det(C[np.ix_(I_IDX, I_IDX)]))
    logdet_fim = float(np.linalg.slogdet(F)[1])
    stats[label] = (det_interest, logdet_fim, np.sqrt(np.abs(np.diag(C))))

dC_D, ld_D, sd_D = stats["D"]
dC_S, ld_S, sd_S = stats["Ds"]

print(f"  {'design':<8} {'det(cov of interest)':>22} {'log det(FIM)':>14}")
print("  " + "-" * 46)
print(f"  {'D':<8} {dC_D:>22.6e} {ld_D:>14.6f}")
print(f"  {'Ds':<8} {dC_S:>22.6e} {ld_S:>14.6f}")
print()
print(f"  Ds shrinks the joint confidence region for {PLAIN[:2]} by "
      f"{(1 - dC_S / dC_D) * 100:.2f}%")
print(f"     (det ratio {dC_S / dC_D:.4f}; the ellipse AREA shrinks by "
      f"{(1 - np.sqrt(dC_S / dC_D)) * 100:.2f}%)")
print(f"  and pays for it with {(1 - np.exp(ld_S - ld_D)) * 100:.2f}% of the "
      f"total det(FIM)")
print(f"     (ratio {np.exp(ld_S - ld_D):.4f})")
print()
print("  That is the whole point of Ds in one line: give up information you did")
print("  not want in order to buy information you did.  The exchange rate here")
print("  is unusually favourable -- 13% better on the question asked for 43% of")
print("  the total -- because the nuisance directions were consuming effort that")
print("  contributed nothing to the kinetics.")



# =============================================================================
banner("4. Why does Ds use MORE support points than D?")
# =============================================================================
# Section 2 shows D settling on 2 candidates and Ds on 4.  That looks backwards
# -- Ds targets FEWER parameters, so surely it needs a smaller design?  Two
# separate things are going on, and neither is a defect.

print("  (a) The classical 'at least p support points' rule does not apply here.")
print()
# That bound assumes each support point contributes RANK-1 information: one
# scalar observation. Here a single candidate is an entire experiment with
# n_m_r responses measured at n_spt sampling times, so its information matrix
# has rank well above 1 and the bound never binds.
d_at = build_designer()
d_at._fd_jac = True
d_at._compute_atomics = True
d_at.eval_sensitivities(save_sensitivities=False)
d_at.eval_fim(np.ones(d_at.n_c * d_at.n_spt) / (d_at.n_c * d_at.n_spt))
A_at = np.asarray(d_at.atomic_fims)
per_cand = np.array([A_at[c * d_at.n_spt:(c + 1) * d_at.n_spt].sum(axis=0)
                     for c in range(d_at.n_c)])
ranks = [np.linalg.matrix_rank(per_cand[c], tol=1e-10) for c in range(d_at.n_c)]
print(f"      observations per candidate = {d_at.n_m_r} responses x "
      f"{d_at.n_spt} sampling times = {d_at.n_m_r * d_at.n_spt}")
print(f"      rank of ONE candidate's information matrix: "
      f"min {min(ranks)}, max {max(ranks)}   (n_mp = {d_at.n_mp})")
print(f"      -> rank {max(ranks)} from a single candidate, so TWO candidates")
print(f"         already reach the full rank {d_at.n_mp}.  The support-count")
print("         floor is 2 here, not 4, and D is free to sit on it.")
print()
print("  (b) Ds is a TWO-SIDED objective, which is a harder problem, not an")
print("      easier one.  When the FIM is invertible,")
print()
print("          det(S) = det(FIM) / det(M_nn)")
print()
print("      so maximising det(S) means maximising det(FIM) AND MINIMISING")
print("      det(M_nn).  Ds actively avoids learning about the nuisance")
print("      parameters, because that information sits in the denominator.")
print()

blocks = {}
for label in ("D", "Ds"):
    dd, eff, _ = runs[label]
    F, _ = covariance(dd, eff)
    Mss = F[np.ix_(I_IDX, I_IDX)]
    Msn = F[np.ix_(I_IDX, N_IDX_)]
    Mnn = F[np.ix_(N_IDX_, N_IDX_)]
    Sch = Mss - Msn @ np.linalg.solve(Mnn, Msn.T)
    blocks[label] = (F, Mss, Msn, Mnn, Sch)

print(f"      {'design':<7} {'det(FIM)':>13} {'det(M_nn)':>13} "
      f"{'ratio':>13} {'det(S)':>13} {'support':>9}")
print("      " + "-" * 76)
for label in ("D", "Ds"):
    dd, eff, _ = runs[label]
    F, Mss, Msn, Mnn, Sch = blocks[label]
    dF, dN, dS = (np.linalg.det(F), np.linalg.det(Mnn), np.linalg.det(Sch))
    print(f"      {label:<7} {dF:>13.5e} {dN:>13.5e} {dF / dN:>13.5e} "
          f"{dS:>13.5e} {len(summarise(dd, eff)):>9}")

FD_, MssD_, MsnD_, MnnD_, SchD_ = blocks["D"]
FS_, MssS_, MsnS_, MnnS_, SchS_ = blocks["Ds"]
print()
print(f"      det(FIM)   Ds/D = {np.linalg.det(FS_) / np.linalg.det(FD_):.4f}   "
      f"Ds gives up total information")
print(f"      det(M_nn)  Ds/D = {np.linalg.det(MnnS_) / np.linalg.det(MnnD_):.4f}   "
      f"Ds learns LESS about {PLAIN[2]}, {PLAIN[3]}")
print(f"      det(S)     Ds/D = {np.linalg.det(SchS_) / np.linalg.det(SchD_):.4f}   "
      f"net gain on the interest block")
print()
print("      The ratio column equals det(S) exactly, which is the identity")
print("      above holding numerically.")
print()
print("  (c) What Ds is really fighting: the correction term.")
print()
print("      S = M_ss - M_sn M_nn^-1 M_ns.  The subtracted part is the")
print("      information about the interest parameters that gets CONSUMED by")
print("      having to estimate the nuisance parameters alongside them.")
print()
print(f"      {'design':<7} {'trace(M_ss)':>13} {'trace(correction)':>19} "
      f"{'fraction lost':>15}")
print("      " + "-" * 58)
for label in ("D", "Ds"):
    _, Mss, Msn, Mnn, _ = blocks[label]
    corr = Msn @ np.linalg.solve(Mnn, Msn.T)
    print(f"      {label:<7} {np.trace(Mss):>13.4f} {np.trace(corr):>19.4f} "
          f"{np.trace(corr) / np.trace(Mss):>15.4f}")
print()
print("      Ds accepts LESS raw information about theta (trace(M_ss) falls)")
print("      in exchange for losing a smaller FRACTION of it to nuisance")
print("      entanglement.  Spreading effort over four conditions instead of")
print("      two is how it buys that decoupling: two support points give the")
print("      design very little freedom to shape the correlation structure")
print("      between the interest and nuisance blocks, four give it more.")
print()
print("      Note how punishing that fraction is even at the Ds design: most")
print("      of the apparent information about theta is consumed by having to")
print("      estimate alpha and nu at the same time.  That is the same effect")
print("      as the nuisance-vs-fixed inflation in section 8, seen from the")
print("      other side.")
print()
print("  THE GENERAL LESSON: 'fewer parameters of interest' does not mean")
print("  'smaller problem'.  Ds replaces a single determinant with a RATIO of")
print("  determinants, and ratio objectives generally want richer designs than")
print("  their numerators alone.  Expect Ds to spread support relative to D,")
print("  not concentrate it.")


# =============================================================================
banner("5. A TRAP: per-parameter standard deviations will mislead you")
# =============================================================================
print(f"  {'parameter':<9} {'sd @ D':>11} {'sd @ Ds':>11} {'ratio':>8}  role")
print("  " + "-" * 50)
for i, p in enumerate(PLAIN):
    role = "interest" if i in I_IDX else "nuisance"
    print(f"  {p:<9} {sd_D[i]:>11.5g} {sd_S[i]:>11.5g} "
          f"{sd_S[i] / sd_D[i]:>8.4f}  {role}")
print()
print("  Read that table naively and Ds looks broken: theta_1 got WORSE, and")
print("  the nuisance parameter alpha got BETTER.  Neither is a bug.")
print()
print("  Ds optimises the JOINT determinant over the interest block, not the")
print("  individual marginal variances.  det(cov_interest) can fall while one")
print("  diagonal entry rises, because the OFF-diagonal term -- the correlation")
print("  between theta_0 and theta_1 -- also changes.  A long thin confidence")
print("  ellipse and a rounder smaller one can have the same theta_1 extent.")
print()
corr = {}
for label in ("D", "Ds"):
    d, eff, _ = runs[label]
    _, C = covariance(d, eff)
    Ci = C[np.ix_(I_IDX, I_IDX)]
    corr[label] = Ci[0, 1] / np.sqrt(Ci[0, 0] * Ci[1, 1])
print(f"  corr(theta_0, theta_1) @ D  = {corr['D']:+.4f}")
print(f"  corr(theta_0, theta_1) @ Ds = {corr['Ds']:+.4f}")
print()
print("  If you need a per-parameter guarantee, Ds is the wrong criterion --")
print("  use A-optimality on the interest set, or add explicit constraints.")
print("  Ds answers 'how small is the joint uncertainty region', nothing else.")


# =============================================================================
banner("6. Verification — the criterion is what it claims to be")
# =============================================================================
# Ds criterion = -log det(Schur complement) = +log det(cov of interest block),
# so -log det(cov_interest) must reproduce the reported value.
_, _, ds_reported = runs["Ds"]
print(f"  -log det(cov of interest) @ Ds design = {-np.log(dC_S):.8f}")
print(f"  criterion reported by designer        = {ds_reported:.8f}")
print(f"  difference                            = "
      f"{abs(-np.log(dC_S) - ds_reported):.2e}")
print()
print("  The residual is the difference between the pseudo-inverse used here")
print("  and the Cholesky-based Schur path used internally; it is not error in")
print("  either.  If this check ever disagreed by more than ~1e-6 the")
print("  interest/nuisance split would be resolving to the wrong parameters.")


# =============================================================================
banner("7. When Ds IS a rescue, and how to tell")
# =============================================================================
print("""  Ds also handles the case this example does NOT show: a FIM that is
  singular because a NUISANCE direction carries no information.  Then
  det(FIM) = 0 for every design, D-optimality is infeasible everywhere, and
  Ds is still well posed because the Schur complement over the interest block
  remains positive definite.

  To tell which situation you are in, ask diagnose_fim_structure():

      diag = designer.diagnose_fim_structure()      # prints the table
      diag["singular"]      -> True if rank-deficient at full support
      diag["culprits"]      -> parameter NAMES in the unidentifiable direction

  The rule is exact: the Schur complement is singular if and only if the FIM's
  null space contains a direction with a non-zero component on an INTEREST
  parameter.  So any parameter named in diag["culprits"] must go to the
  nuisance set -- and if you need its precision, no design over these
  candidates can give it to you.  That is a modelling result, not a numerical
  one: reparameterise, add measurements, or fix the parameter.

  Ds returning +inf is that diagnosis, not a failure.

  A worked singular example lives in the test suite as section 37
  (y = A0*exp(-k t) + c1 + c2, where only c1 + c2 is identifiable).""")


# =============================================================================
banner("8. Interest vs nuisance vs FIXED — what the API does and does not do")
# =============================================================================
# designer.interest_parameters declares the INTEREST set. Everything else
# becomes nuisance: the partition is the strict complement,
#     idx_nuisance = [j for j in range(n_mp) if j not in idx_interest]
# There are exactly TWO categories. pydex has no mechanism for declaring a
# parameter FIXED/known, and that is a real distinction, not a formality.
S_IDX, N_IDX = I_IDX, N_IDX_
F_u, _ = covariance(d0, uniform)
Mss = F_u[np.ix_(S_IDX, S_IDX)]
Msn = F_u[np.ix_(S_IDX, N_IDX)]
Mns = F_u[np.ix_(N_IDX, S_IDX)]
Mnn = F_u[np.ix_(N_IDX, N_IDX)]

cov_nuisance = np.linalg.inv(Mss - Msn @ np.linalg.solve(Mnn, Mns))  # marginalised
cov_fixed    = np.linalg.inv(Mss)                                    # removed

print("  Same two parameters, two different statistical questions:\n")
print("  (a) NUISANCE — unknown, estimated, precision not of interest.")
print("      The Schur complement marginalises them, so the interest")
print("      parameters carry the cost of NOT KNOWING them.")
print(f"        sd(theta_0) = {np.sqrt(cov_nuisance[0,0]):.6f}   "
      f"sd(theta_1) = {np.sqrt(cov_nuisance[1,1]):.6f}   "
      f"det = {np.linalg.det(cov_nuisance):.6e}")
print()
print("  (b) FIXED — known exactly, not estimated at all.  The correct")
print("      treatment is to REMOVE them from the FIM, then invert the")
print("      interest sub-block directly.  No Schur correction.")
print(f"        sd(theta_0) = {np.sqrt(cov_fixed[0,0]):.6f}   "
      f"sd(theta_1) = {np.sqrt(cov_fixed[1,1]):.6f}   "
      f"det = {np.linalg.det(cov_fixed):.6e}")
print()
_r = np.linalg.det(cov_nuisance) / np.linalg.det(cov_fixed)
print(f"  Declaring them nuisance inflates the joint interest variance "
      f"{_r:.3f}x")
print(f"  ({(_r - 1) * 100:.1f}% larger determinant), and sd(theta_0) alone by "
      f"{np.sqrt(cov_nuisance[0,0] / cov_fixed[0,0]):.4f}x.")
print()
print("  Nuisance is the CONSERVATIVE choice -- it never understates")
print("  uncertainty -- but it is not free.  If a parameter is genuinely known,")
print("  treating it as nuisance designs for a harder problem than you have.")
print()
print("  THERE IS NO designer.fixed_parameters.  To fix a parameter today:")
print("    * remove it from designer.model_parameters, and")
print("    * hard-code its value inside your model / simulate function.")
print("  The FIM then has no row or column for it, and D-optimality over the")
print("  remaining parameters is what you want -- Ds is not needed at all.")
print()
print("  Rule of thumb:")
print("    parameter is unknown and you want it precise   -> interest")
print("    parameter is unknown and you do not care       -> nuisance (Ds)")
print("    parameter is known                             -> remove it")


# =============================================================================
banner("9. Numbers: the full parameter covariance at each design")
# =============================================================================
print("  Full 4x4 covariance (FIM^-1). Diagonal = variances, off-diagonal =")
print("  covariances. Ds promises nothing outside the upper-left INTEREST block.\n")
covs = {}
for label in ("D", "Ds"):
    dd, eff, _ = runs[label]
    _, C = covariance(dd, eff)
    covs[label] = C
    print(f"  {label}-optimal:")
    print("            " + "".join(f"{p:>12}" for p in PLAIN))
    for i, p in enumerate(PLAIN):
        row = "".join(f"{C[i, j]:>12.5g}" for j in range(len(PLAIN)))
        print(f"  {p:>10}{row}{'   <- interest' if i in I_IDX else ''}")
    C2 = C[np.ix_(I_IDX, I_IDX)]
    print(f"      interest block: det = {np.linalg.det(C2):.6e}, "
          f"corr = {C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1]):+.4f}\n")


# =============================================================================
banner("10. Figure 1 — candidate grid, D and Ds side by side")
# =============================================================================
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

grid = np.asarray(d0.ti_controls_candidates, dtype=float)
x_all, y_all = grid[:, 0], grid[:, 1]
# shared padding so both panels have identical limits and markers never clip
xpad = 0.18 * (x_all.max() - x_all.min())
ypad = 0.22 * (y_all.max() - y_all.min())

fig1, axes1 = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)

for ax, (label, colour) in zip(axes1, (("D", "tab:blue"), ("Ds", "tab:red"))):
    dd, eff, val = runs[label]
    sel = summarise(dd, eff)

    # every candidate, as a faint reference lattice
    ax.scatter(x_all, y_all, s=26, facecolors="none", edgecolors="0.72",
               linewidths=0.9, zorder=2)

    # selected candidates: marker AREA proportional to effort, on a scale
    # shared between panels so the two are directly comparable
    pts = np.asarray([t for _, t, _ in sel], dtype=float)
    fr = np.asarray([f for _, _, f in sel], dtype=float)
    ax.scatter(pts[:, 0], pts[:, 1], s=200 + 3200 * fr, c=colour, alpha=0.40,
               edgecolors=colour, linewidths=1.8, zorder=3)

    # Label each with its effort. Offset direction flips with position: a
    # fixed downward offset pushed the bottom-row labels outside the axes and
    # into the tick labels. Points in the lower half get the label above,
    # upper half below, so everything stays inside the frame.
    y_mid = 0.5 * (y_all.min() + y_all.max())
    for (xx, yy), f in zip(pts, fr):
        below = yy > y_mid
        ax.annotate(f"{f * 100:.1f}%", (xx, yy),
                    textcoords="offset points",
                    xytext=(0, -28 if below else 28),
                    ha="center", va="top" if below else "bottom",
                    fontsize=9, fontweight="bold",
                    color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=colour, lw=0.7, alpha=0.92))

    tag = "all 4 parameters" if label == "D" else f"interest {PLAIN[:2]}"
    ax.set_title(f"{label}-optimal — {tag}\n"
                 f"criterion {val:.5f},  {len(sel)} support candidates",
                 fontsize=10.5)
    ax.set_xlabel("$C_{A0}$  (mol/L)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_xlim(x_all.min() - xpad, x_all.max() + xpad)
    ax.set_ylim(y_all.min() - ypad, y_all.max() + ypad)

axes1[0].set_ylabel("$T$  (K)")
# one legend for the figure, placed outside the data so nothing is obscured
h = [plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="0.72",
                ms=6, label=f"candidate grid ({len(grid)} points)"),
     plt.Line2D([], [], marker="o", ls="none", mfc="tab:blue", mec="tab:blue",
                alpha=0.5, ms=11, label="selected (area $\\propto$ effort)")]
fig1.legend(handles=h, loc="lower center", ncol=2, frameon=False,
            fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig1.suptitle("Where the experimental effort goes", fontsize=12.5, y=0.99)
fig1.tight_layout(rect=(0, 0.04, 1, 0.96))

print("  Left: D concentrates everything at CA0 = 5, one low and one high T.")
print("  Right: Ds keeps both temperature extremes but moves the bulk of the")
print("  effort to CA0 = 4, and splits across four conditions instead of two.")
print("  Separate panels because at CA0 = 5 the two designs overlap exactly --")
print("  drawn on one axis the markers and their labels collide.")


# =============================================================================
banner("11. Figure 2 — joint covariance of the interest parameters")
# =============================================================================
# constrained_layout instead of tight_layout: the colorbar axes is not
# tight_layout-compatible and emits a UserWarning otherwise.
fig2 = plt.figure(figsize=(13.5, 4.8), constrained_layout=True)
gs = fig2.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.34)


def ellipse_pts(cov2, nsig=1.0, n=500):
    vals, vecs = np.linalg.eigh(0.5 * (cov2 + cov2.T))
    vals = np.maximum(vals, 0.0)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack([np.cos(th), np.sin(th)]) @ (vecs * np.sqrt(vals)).T * nsig


# ── (a) confidence ellipses, SCALED BY THE D-OPTIMAL STANDARD DEVIATIONS ───
# In physical units sd(theta_1) is ~9x sd(theta_0), so an equal-aspect plot is
# a nearly vertical sliver and the two designs are indistinguishable. Dividing
# each axis by the D-optimal sd puts both ellipses at O(1): D becomes the
# reference, and Ds's shape change is legible. Areas remain comparable because
# both are divided by the same numbers.
axA = fig2.add_subplot(gs[0, 0])
ref = np.sqrt(np.diag(covs["D"]))[I_IDX]
for label, colour, ls in (("D", "tab:blue", "-"), ("Ds", "tab:red", "--")):
    C2 = covs[label][np.ix_(I_IDX, I_IDX)]
    xy = ellipse_pts(C2) / ref
    a = np.pi * np.sqrt(max(np.linalg.det(C2), 0.0))
    r = C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1])
    axA.plot(xy[:, 0], xy[:, 1], ls, color=colour, lw=2.2,
             label=f"{label}  (area {a:.3f}, $\\rho$ {r:+.2f})")
    axA.fill(xy[:, 0], xy[:, 1], color=colour, alpha=0.10)
axA.axhline(0, color="0.75", lw=0.8); axA.axvline(0, color="0.75", lw=0.8)
axA.set_aspect("equal", adjustable="box")
axA.set_xlabel(r"$\theta_0$ deviation  [$\sigma$ at D-optimal]")
axA.set_ylabel(r"$\theta_1$ deviation  [$\sigma$ at D-optimal]")
axA.set_title("Joint 1-$\\sigma$ region\n(axes scaled by D-optimal $\\sigma$)",
              fontsize=10.5)
axA.legend(loc="upper left", fontsize=8, framealpha=0.93,
           borderpad=0.4, handlelength=1.8)
axA.grid(alpha=0.25, linestyle=":")

# ── (b, c) the covariance matrices as heatmaps, shared colour scale ────────
vmax = max(np.abs(covs["D"]).max(), np.abs(covs["Ds"]).max())
for k, label in enumerate(("D", "Ds")):
    ax = fig2.add_subplot(gs[0, 1 + k])
    C = covs[label]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(len(PLAIN)):
        for j in range(len(PLAIN)):
            ax.text(j, i, f"{C[i, j]:.3g}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(C[i, j]) > 0.55 * vmax else "0.15")
    # outline the interest block -- the only part Ds makes a promise about
    ax.add_patch(Rectangle((min(I_IDX) - 0.5, min(I_IDX) - 0.5),
                           len(I_IDX), len(I_IDX), fill=False,
                           ec="black", lw=2.2, zorder=5))
    ax.set_xticks(range(len(PLAIN)))
    ax.set_yticks(range(len(PLAIN)))
    ax.set_xticklabels(PLAIN, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(PLAIN, fontsize=8)
    C2 = C[np.ix_(I_IDX, I_IDX)]
    ax.set_title(f"cov @ {label}-optimal\ndet(interest) = "
                 f"{np.linalg.det(C2):.4e}", fontsize=10.5)
    if k == 1:
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      label="covariance")

fig2.suptitle("Parameter covariance: the joint region Ds minimises, and the "
              "full matrices behind it", fontsize=12)

print("  Left panel: the object Ds minimises. Axes are divided by the")
print("  D-optimal standard deviations because sd(theta_1) is ~9x")
print("  sd(theta_0) -- in physical units with equal aspect the ellipse is a")
print("  vertical sliver and the two designs cannot be told apart.")
print()
print("  The Ds ellipse encloses LESS AREA while extending FURTHER in")
print("  theta_1. That is section 5's trap, visible: a per-parameter check on")
print("  theta_1 sees a regression exactly where the joint measure improves.")
print()
print("  Right panels: the full 4x4 covariance, shared colour scale, with the")
print("  interest block boxed. NOTE the colour scale is dominated by")
print(f"  var(theta_1) = {covs['D'][1,1]:.3g}, which is ~35x the next largest entry, so")
print("  the smaller covariances all render near-white -- read the printed")
print("  numbers in each cell, the colour is only a visual aid.")
print()
print("  Everything outside that box is what Ds traded away -- note nu's")
print("  variance rising from "
      f"{covs['D'][3, 3]:.4g} to {covs['Ds'][3, 3]:.4g}.")


banner("Summary")
print(f"""
  Ds-optimal design for {PLAIN[:2]}, nuisance {PLAIN[2:]}:

    D-optimal  criterion {runs['D'][2]:.6f}   {len(summarise(*runs['D'][:2]))} support candidates
    Ds-optimal criterion {runs['Ds'][2]:.6f}   {len(summarise(*runs['Ds'][:2]))} support candidates

    joint interest precision  {(1 - dC_S / dC_D) * 100:>6.2f}% better under Ds
    total det(FIM)            {(1 - np.exp(ld_S - ld_D)) * 100:>6.2f}% given up

  Workflow:
    1. designer.interest_parameters = [...]     # BY NAME
    2. designer.design_experiment(designer.ds_opt_criterion, ...)
    3. judge the result on det(cov of the interest block), NOT on
       per-parameter standard deviations (see section 5)

  See also
    case_2.py      D-optimal on the same model
    case_2_model.py   PITFALL section — the sampling-grid trap
""")


runs["Ds"][0].print_optimal_candidates()
plt.show()
