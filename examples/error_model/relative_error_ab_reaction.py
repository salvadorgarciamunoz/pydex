"""
relative_error_ab_reaction.py
=============================
What a relative measurement error does to a DYNAMIC design: the same A→B
reaction, the same candidate grid, four error models.

This is ``examples/ode/case_2_no_ift_no_collocation.py`` with one thing
changed.  That example sets ::

    designer.error_cov = np.diag([0.1, 0.1])

one constant variance for c_A and c_B over the whole batch.  Here the same
design is solved again with ``designer.error_variance``, a per-response error
model ::

    variance = (rel_sd * y_hat) ** 2 + abs_sd ** 2

evaluated at the nominal predicted response.  This is the two-component model
of Rocke & Lorenzato (1995), *Technometrics* 37(2):176-184, also called
constant-plus-proportional; R's ``nlme`` parameterises it identically as
``varConstProp``.

Why this model and not a static one
-----------------------------------
Over a 200 min batch at the most informative candidate, c_A falls from 5.00 to
0.22 mol/L — a **22x spread** — while c_B rises from 0 to 2.39.  A single
constant variance says a measurement of c_A = 0.22 is known exactly as well as
one of c_A = 5.00.  No analytical method behaves that way.  Since pydex's
distinguishing capability is choosing *when* to measure, that assumption is
not a detail at the margin: it decides the answer.

What changes, measured
----------------------
All four designs below use the same 25-candidate grid (5 c_A0 levels x 5
temperatures), the same 11-point sampling grid, ``n_spt=1`` (one measurement
per run, the optimiser chooses when), D-optimality and POUNCE.

======================================  =====================  ==============
error model                             conditions chosen      times chosen
======================================  =====================  ==============
``error_cov = diag([0.1, 0.1])``        c_A0 = 5 only          60 to 200 min
relative, 1.5% RSD, floor 0.01          c_A0 = 3, 5            100 to 200 min
c_A relative, c_B constant              c_A0 = 2, 5            100 to 200 min
relative, 1.5% RSD, floor 0.002         c_A0 = 1, 2, 5         180 to 200 min
======================================  =====================  ==============

Two trends, and the second is the one worth taking away.

**Measurements move later.**  Under a relative model the absolute standard
deviation of c_A *shrinks* as c_A decays, so a late measurement is more
precise in absolute terms and carries more weight.  What stops the design
going later still is ``abs_sd``: the additive floor, which dominates once the
signal is small.  The two relative cases differ only in that floor, and
tightening it five-fold pulls almost every measurement to the last available
time.

**Dilute experiments stop being wasteful.**  Under constant variance the
entire design sits at c_A0 = 5 mol/L, the most concentrated candidate — a
dilute run produces a small absolute signal against the same fixed noise, so
it looks uninformative.  Under a relative model a dilute run is measured
proportionally just as well: at a 0.01 floor the design reaches down to
c_A0 = 3, and at 0.002 it gives c_A0 = 1 nearly a third of the budget.  **A
constant-variance assumption biases a design toward high concentration.**
That is not a property of the chemistry; it is an artefact of the weighting.

The relative term matters as much as the floor.  At 5% RSD rather than 1.5%
every measurement collapses onto the last sampling time, which demonstrates
the mechanism but leaves nothing to look at.  1.5% is used here because it
keeps the design informative about *when* as well as *where*.

Criterion values are NOT comparable across these runs
------------------------------------------------------
10.657 for the constant case against 25.775 for the relative one does not mean
the second design is better.  Scaling the measurement-error covariance
rescales the FIM, so a log-det criterion moves without the underlying
experiment changing.  Compare the DESIGNS, and record the error model next to
any criterion value you quote.

A near-zero prediction is a hazard, and it is why ``abs_sd`` exists
-------------------------------------------------------------------
c_B is exactly 0 at t = 0.  With a purely proportional model its variance
would be zero there and its weight infinite, and the design would chase a
measurement that carries no information.  ``abs_sd > 0`` is enforced for
exactly this reason, and it is a physical quantity, not a numerical guard:
the real noise near zero — baseline, blank variability, integration error on a
small peak.  Elicit it from the standard deviation of low-standard replicates,
**not** from the LOQ (LOQ ~ 10*sigma by convention, so divide).

Here the design does not place effort at t = 0 even so, because the
sensitivity of c_B to every parameter also vanishes there: information is
S^T W S, and a large W cannot rescue a zero S.  On a model whose response is
small but still sensitive, it would.

Quoting a relative error
-------------------------
``rel_sd`` is one standard deviation, so ``rel_sd = CV``.  "10% relative
error" means ``rel_sd = 0.10`` if it is an RSD, and ``rel_sd = 0.05`` if it is
a 95% interval — a factor of 4 in variance.  Ask which before you believe it.

Run
---
    python relative_error_ab_reaction.py

Needs scipy and an NLP solver.  ``solver="pounce"`` installs from pip with no
downloaded binaries; ``solver="ipopt"`` works equally well.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from pydex.core.designer import Designer

# The forward model is shared with examples/ode/.  Resolve it from THIS file's
# location, never from the working directory: a bare sys.path.insert(0, ".")
# lets a same-named file in the cwd shadow the real model, silently.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ode"))
from case_2_no_ift_no_collocation_model import simulate   # noqa: E402

SOLVER = "pounce"

# ---------------------------------------------------------------------------
# Nominal parameters -- identical to case_2_no_ift_no_collocation.py
# ---------------------------------------------------------------------------
R = 8.314159
T_REF = 273.15
_theta_0 = np.log(0.1) - 5000.0 / (R * T_REF)
_theta_1 = 5000.0 / (R * T_REF)
THETA_NOM = np.array([_theta_0, _theta_1, 1.0, 0.5])

# 5 initial concentrations x 5 temperatures = 25 candidates
TIC = np.array([
    [ca0, T]
    for ca0 in [1.0, 2.0, 3.0, 4.0, 5.0]
    for T in [273.15, 285.65, 298.15, 310.65, 323.15]
])

# One sampling grid, shared by every candidate.  A 1-D vector is broadcast
# across candidates at initialize(); the (n_c, n_spt) form also works.
SPT = np.linspace(0.001, 200, 11)


def build_designer():
    """A designer with everything set except the measurement error."""
    designer = Designer()
    designer.simulate = simulate
    designer.model_parameters = THETA_NOM
    designer.ti_controls_candidates = TIC
    designer.sampling_times_candidates = SPT
    designer.model_parameter_names = ["theta_0", "theta_1", "alpha", "nu"]
    designer.ti_controls_names = ["CA0", "T"]
    designer.response_names = ["CA", "CB"]
    return designer


def solve_and_report(label, designer):
    """Solve one already-configured design, print it, and return the efforts."""
    designer.initialize(verbose=0)
    designer.design_experiment(
        designer.d_opt_criterion,
        n_spt=1,              # one measurement per run; the design picks when
        solver=SOLVER,
    )

    print(f"\n{label}")
    print(f"  criterion {designer.oed_result['criterion_value']:.8f}"
          "   (not comparable across error models -- see the docstring)")
    print(f"  {'CA0':>5}  {'T (K)':>7}  {'t (min)':>8}  {'effort':>7}")
    for cand, slot in np.argwhere(designer.efforts > 1e-3):
        print(f"  {TIC[cand, 0]:5.0f}  {TIC[cand, 1]:7.2f}  "
              f"{SPT[slot]:8.1f}  {designer.efforts[cand, slot] * 100:6.2f}%")
    return np.asarray(designer.efforts)


# ---------------------------------------------------------------------------
# The error models, written once so the bands and the designs cannot drift
# ---------------------------------------------------------------------------
# Scenario 1 is set on the designer as error_cov = np.diag([0.1, 0.1]); the
# dict below is the same thing in error_variance form, which is what lets one
# helper draw all four bands.
ERROR_MODELS = [
    ("1. constant variance, diag([0.1, 0.1])",
     {"CA": 0.1, "CB": 0.1}),
    ("2. relative, 1.5% RSD, floor 0.01",
     {"CA": {"rel_sd": 0.015, "abs_sd": 0.01},
      "CB": {"rel_sd": 0.015, "abs_sd": 0.01}}),
    ("3. mixed: CA relative, CB constant",
     {"CA": {"rel_sd": 0.015, "abs_sd": 0.01},
      "CB": 0.1}),
    ("4. relative, 1.5% RSD, floor 0.002",
     {"CA": {"rel_sd": 0.015, "abs_sd": 0.002},
      "CB": {"rel_sd": 0.015, "abs_sd": 0.002}}),
]


def sigma_of(spec, y_hat):
    """Standard deviation of one measurement, from its error specification.

    This is the expression pydex evaluates internally to build the weights:
    a bare number is a variance, and a dict is the two-component model

        variance = (rel_sd * y_hat) ** 2 + abs_sd ** 2

    with rel_sd and abs_sd given as STANDARD DEVIATIONS.
    """
    if isinstance(spec, dict):
        rel_sd = spec.get("rel_sd", 0.0)
        abs_sd = spec["abs_sd"]
        return np.sqrt((rel_sd * np.asarray(y_hat)) ** 2 + abs_sd ** 2)
    return np.full(np.shape(y_hat), np.sqrt(spec))


def plot_error_bands(condition=(5.0, 323.15), k_sigma=1.96):
    """What each error model CLAIMS about the measurements, before designing.

    The same predicted trajectory under each error specification, with a
    +/- 1.96 sigma band -- roughly a 95% interval for a single measurement.
    No design is involved: this is purely what you are telling pydex about
    your analytics.

    The constant band is the one to look at. It has the same width at
    c_A = 0.22 as at c_A = 5.00, so at the end of the batch the interval is
    wider than the signal and runs below zero -- an assertion about the
    instrument that no analytical method satisfies. The relative bands
    narrow with the signal and flatten onto the additive floor, which is why
    the designs built on them are willing to measure late.
    """
    t_dense = np.linspace(0.001, SPT.max(), 200)
    traj = simulate(list(condition), t_dense, THETA_NOM)

    fig = plt.figure(figsize=(11, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.32)
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    for ax in axes.ravel()[1:]:
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])

    for ax, (label, spec) in zip(axes.ravel(), ERROR_MODELS):
        for col, (name, colour) in enumerate(
                zip(("CA", "CB"), ("tab:blue", "tab:orange"))):
            y = traj[:, col]
            sd = sigma_of(spec[name], y)
            ax.plot(t_dense, y, color=colour, lw=1.6, label=f"$c_{name[-1]}$")
            ax.fill_between(t_dense, y - k_sigma * sd, y + k_sigma * sd,
                            color=colour, alpha=0.25, linewidth=0)
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[:, 0]:
        ax.set_ylabel("concentration (mol/L)")

    # On a shared concentration axis the relative bands are too narrow to read,
    # which is the honest comparison but hides the mechanism. Plot the standard
    # deviation of a c_A measurement directly, on a log scale: the proportional
    # term falls with the signal and then flattens onto the additive floor.
    ax_sd = fig.add_subplot(gs[2, :])
    for (label, spec), style in zip(ERROR_MODELS, ["-", "-", "--", "-"]):
        ax_sd.semilogy(t_dense, sigma_of(spec["CA"], traj[:, 0]),
                       style, lw=1.6, label=label)
    ax_sd.set_xlabel("time (min)")
    ax_sd.set_ylabel("$\\sigma$ of a $c_A$ measurement (mol/L)")
    ax_sd.set_title("The same four models, as the standard deviation they "
                    "imply for $c_A$", fontsize=9, loc="left")
    ax_sd.grid(alpha=0.25, linewidth=0.5, which="both")
    ax_sd.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        f"What each error model claims, at $c_{{A0}}$={condition[0]:.0f} "
        f"mol/L, T={condition[1]:.0f} K\n"
        f"bands are +/- {k_sigma:.2f} sigma for a single measurement; "
        "the dotted line is zero",
        fontsize=11)
    fig.subplots_adjust(top=0.90, left=0.09, right=0.97, bottom=0.07)
    return fig


def plot_designs(results):
    """One panel per error model: where the design puts its effort.

    Position is the experimental condition, marker area is the effort, and
    colour is the sampling time the design chose -- so the two trends the
    docstring describes are both visible at a glance: colour moving toward
    the late end of the scale, and markers spreading down the c_A0 axis.
    Grey dots are candidates that received no effort. Where one candidate
    carries effort at several times the markers are fanned horizontally, so
    their T positions are indicative rather than exact.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    norm = plt.Normalize(SPT.min(), SPT.max())

    for ax, (label, efforts) in zip(axes.ravel(), results):
        ax.scatter(TIC[:, 1], TIC[:, 0], s=18, c="0.8",
                   edgecolors="none", zorder=1, label="candidate")
        sel = np.argwhere(efforts > 1e-3)

        # One candidate can carry effort at several sampling times, which would
        # stack markers on the same point. Fan those out along T so each cell
        # stays visible; the printed table has the exact values.
        per_cand = {}
        for c, t in sel:
            per_cand.setdefault(c, []).append(t)
        xs, ys, areas, times = [], [], [], []
        for c, slots in per_cand.items():
            for i, t in enumerate(sorted(slots)):
                xs.append(TIC[c, 1] + (i - (len(slots) - 1) / 2) * 2.4)
                ys.append(TIC[c, 0])
                areas.append(efforts[c, t] * 2600)
                times.append(SPT[t])
        sc = ax.scatter(xs, ys, s=areas, c=times, cmap="viridis", norm=norm,
                        edgecolors="k", linewidths=0.6, zorder=2)
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("T (K)")
    for ax in axes[:, 0]:
        ax.set_ylabel("$c_{A0}$ (mol/L)")
    axes[0, 0].set_ylim(0.4, 5.6)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("sampling time chosen (min)")
    fig.suptitle("Where the design spends its effort, under four measurement "
                 "error models\n"
                 "marker area = effort, colour = when to measure",
                 fontsize=11)
    return fig


def plot_sampling_on_trajectories(results):
    """One panel per error model: the trajectories actually being measured.

    For every supported condition the predicted c_A (solid) and c_B (dashed)
    are drawn over the batch, with a marker at each sampling time the design
    chose. A measurement is of both responses at once, so the markers appear
    on both curves. Marker area is the effort on that cell.

    This is the same information as the condition map, read along the time
    axis instead: under a relative error the markers sit where c_A has
    decayed, because that is where its ABSOLUTE standard deviation -- and so
    its contribution to the information matrix -- is most favourable.
    """
    t_dense = np.linspace(0.001, SPT.max(), 200)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for ax, (label, efforts) in zip(axes.ravel(), results):
        per_cand = {}
        for c, t in np.argwhere(efforts > 1e-3):
            per_cand.setdefault(c, []).append(t)

        for colour, (cand, slots) in zip(plt.cm.tab10.colors, per_cand.items()):
            traj = simulate(TIC[cand], t_dense, THETA_NOM)
            ax.plot(t_dense, traj[:, 0], color=colour, lw=1.5,
                    label=f"$c_{{A0}}$={TIC[cand, 0]:.0f}, T={TIC[cand, 1]:.0f} K")
            ax.plot(t_dense, traj[:, 1], color=colour, lw=1.2, ls="--",
                    alpha=0.7)
            marked = simulate(TIC[cand], SPT[sorted(slots)], THETA_NOM)
            areas = [efforts[cand, t] * 1200 for t in sorted(slots)]
            ax.scatter(SPT[sorted(slots)], marked[:, 0], s=areas,
                       color=colour, edgecolors="k", linewidths=0.6, zorder=3)
            ax.scatter(SPT[sorted(slots)], marked[:, 1], s=areas,
                       color=colour, edgecolors="k", linewidths=0.6,
                       alpha=0.7, zorder=3)

        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7, loc="center right", framealpha=0.9)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("concentration (mol/L)")
    # markers sit ON the last sampling time, so leave room at both ends
    axes[0, 0].set_xlim(-12, SPT.max() + 14)

    fig.suptitle("What gets measured, and when\n"
                 "solid $c_A$, dashed $c_B$; marker area = effort",
                 fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # How far the responses actually move over the batch -- the reason a
    # single constant variance is the wrong model here.
    traj = simulate([5.0, 323.15], SPT, THETA_NOM)
    print("Predicted responses at CA0 = 5, T = 323.15 K:")
    print(f"  {'t (min)':>8}  {'CA':>8}  {'CB':>8}")
    for t, row in zip(SPT, traj):
        print(f"  {t:8.1f}  {row[0]:8.4f}  {row[1]:8.4f}")
    print(f"\n  CA spans {traj[0, 0] / traj[-1, 0]:.0f}x over the batch; "
          "CB starts at exactly zero.")

    # ---- 1. constant variance ------------------------------------------
    # One sigma for the whole batch, as case_2_no_ift_no_collocation.py sets
    # it. error_cov is a covariance MATRIX, indexed by position: its rows must
    # line up with the measured responses in order.
    designer = build_designer()
    designer.error_cov = np.diag([0.1, 0.1])          # variances, not sigmas
    eff_const = solve_and_report(
        "1. CONSTANT variance -- one sigma for the whole batch "
        "(error_cov, as case_2 sets it)", designer)

    # ---- 2. relative error ----------------------------------------------
    # error_variance is keyed by RESPONSE NAME, so the order cannot be wrong.
    # rel_sd and abs_sd are STANDARD DEVIATIONS -- pydex squares them -- and
    # abs_sd is in the response's own units (mol/L here).
    #
    #     variance = (rel_sd * y_hat) ** 2 + abs_sd ** 2
    #
    # rel_sd = 0.015 is a 1.5% RSD. If someone quotes "+/- 1.5%" meaning a 95%
    # interval, that is rel_sd = 0.0075, not 0.015 -- ask which they mean.
    designer = build_designer()
    designer.error_variance = {
        "CA": {"rel_sd": 0.015, "abs_sd": 0.01},
        "CB": {"rel_sd": 0.015, "abs_sd": 0.01},
    }
    eff_rel = solve_and_report(
        "2. RELATIVE error -- 1.5% RSD on both responses, floor 0.01 mol/L",
        designer)

    # ---- 3. one model per response ---------------------------------------
    # Each response is specified independently. A bare number is a variance,
    # exactly as a diagonal entry of error_cov would be, so responses can be
    # mixed freely: c_A relative, c_B constant.
    designer = build_designer()
    designer.error_variance = {
        "CA": {"rel_sd": 0.015, "abs_sd": 0.01},
        "CB": 0.1,
    }
    eff_mixed = solve_and_report(
        "3. MIXED -- CA relative, CB constant "
        "(a bare number is a variance, as in error_cov)", designer)

    # ---- 4. the floor matters --------------------------------------------
    # Same relative term, a five-fold tighter additive floor. abs_sd is the
    # real noise near zero -- baseline, blank variability, integration error
    # on a small peak -- so take it from the SD of low-standard replicates,
    # NOT from the LOQ (LOQ ~ 10*sigma by convention, so divide).
    designer = build_designer()
    designer.error_variance = {
        "CA": {"rel_sd": 0.015, "abs_sd": 0.002},
        "CB": {"rel_sd": 0.015, "abs_sd": 0.002},
    }
    eff_tight = solve_and_report(
        "4. RELATIVE with a TIGHTER floor -- 0.002 mol/L instead of 0.01",
        designer)

    results = [
        ("1. constant variance, diag([0.1, 0.1])", eff_const),
        ("2. relative, 1.5% RSD, floor 0.01", eff_rel),
        ("3. mixed: CA relative, CB constant", eff_mixed),
        ("4. relative, 1.5% RSD, floor 0.002", eff_tight),
    ]
    fig_map = plot_designs(results)
    fig_map.savefig("relative_error_conditions.png", dpi=150)

    fig_traj = plot_sampling_on_trajectories(results)
    fig_traj.savefig("relative_error_trajectories.png", dpi=150)

    fig_bands = plot_error_bands()
    fig_bands.savefig("relative_error_bands.png", dpi=150)

    print("\nWrote relative_error_bands.png, relative_error_conditions.png "
          "and relative_error_trajectories.png to the working directory.")
    plt.show()
