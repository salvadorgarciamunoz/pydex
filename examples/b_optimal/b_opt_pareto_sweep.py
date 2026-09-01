"""
b_opt_pareto_sweep.py
=====================
Generates the two figures the training deck needs for the bracketing-optimal
(b-optimal) module. Run it from inside examples/b_optimal/ so the model files
import cleanly.

    cd examples/b_optimal
    python b_opt_pareto_sweep.py                  # film coater, default
    python b_opt_pareto_sweep.py --model cstr     # the two-CSTR case
    python b_opt_pareto_sweep.py --model both

WHAT IT PRODUCES
----------------
    b_opt_fig1_<model>.png   sweeping output_weight at fixed n_exp.
                             Top row: chosen points in INPUT space.
                             Bottom row: the same points in OUTPUT space.
                             One column per weight, 5 weights:
                             0.0, 0.25, 0.5, 0.75, 1.0.

    b_opt_fig2_<model>.png   the Pareto front (f_in vs f_out), one curve per
                             n_exp, so you can see the whole frontier move
                             outward as the budget grows.

WHAT IT NEEDS
-------------
An MINLP solver. bonmin, from `idaes get-extensions`, with
`export PATH="$HOME/.idaes/bin:$PATH"`. b_opt is a binary subset-selection
problem, so an NLP solver such as ipopt or pounce will NOT do.

RUNTIME
-------
Each (n_exp, weight) pair is one bonmin solve. Figure 1 is 5 solves; figure 2
is 5 weights x len(N_EXP_LIST) solves. On the film coater with a 36-candidate
pool that is a few minutes; the CSTR pool is larger, so budget more. Reduce
N_CANDIDATES first if you want a quick look.

Values printed to stdout are the ones to quote on a slide -- the script writes
a small summary table alongside each figure.
"""

import argparse
import os
import sys
import traceback
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ── Lilly Soft Pink palette, so the figures drop straight into the deck ────
RED = "#E1251B"
BLACK = "#212121"
BROWN = "#521207"
BLUE = "#0F3A85"
GREEN = "#144B2D"
GOLD = "#B8860B"
STONE = "#B9C4CE"
PINK = "#FBCFC8"

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]     # 2 extremes + 3 intermediate
N_EXP_LIST = [4, 6, 8]                     # for figure 2
N_EXP_FIG1 = 6                             # fixed budget for figure 1
# Per-model sampling budgets. These are SAMPLED counts, not feasible ones:
# the coater keeps most of what it samples, but the CSTR's three quality
# constraints (xC2, xA2, T2) admit only about 0.5% of the box, which is why
# scenario_2_cstr.py itself samples 15000.
N_CANDIDATES = {"film": 36, "cstr": 15000}
SEED = 0


# ══════════════════════════════════════════════════════════════════════════
# Model plumbing — reuses the helpers already in the scenario scripts
# ══════════════════════════════════════════════════════════════════════════
def _import_quietly(module_name):
    """Import a scenario module without running its own demo.

    The scenario scripts have no `if __name__ == "__main__":` guard, so a
    plain import executes the whole demo (several solves and a pile of PNGs)
    before we get control. Redirecting stdout keeps the console readable; the
    solves still happen, which costs a minute or so on first import.
    """
    import contextlib, io, importlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = importlib.import_module(module_name)
    return mod


def load_model(name):
    """Return (build_candidate_grid, make_designer, design_b_opt,
    true_fin_fout, input_labels, output_labels)."""
    if name == "film":
        sc = _import_quietly("scenario_1_film_coating")
        return (sc.build_candidate_grid, sc.make_designer, sc.design_b_opt,
                sc.true_fin_fout,
                ["T_in", "M_coat", "Q_air"], ["T_exh", "RH_exh"])
    if name == "cstr":
        sc = _import_quietly("scenario_2_cstr")
        return (sc.build_candidate_grid, sc.make_designer, sc.design_b_opt,
                sc.true_fin_fout,
                ["A0", "ratio", "q0", "V1", "V2", "T0"], ["xC2", "xA2", "T2"])
    raise SystemExit(f"unknown model '{name}' — use film or cstr")


def run_case(sc, TIC, n_exp, weight):
    """One bonmin solve. Returns (indices, f_in, f_out, seconds)."""
    build_grid, make_designer, design_b_opt, true_fin_fout, _, _ = sc
    t0 = time.time()
    d = make_designer(TIC, verbose=0)
    idx = design_b_opt(d, n_exp=n_exp, output_weight=weight, verbose=0)
    Y = np.asarray(d.response).reshape(len(TIC), -1)
    f_in, f_out = true_fin_fout(TIC, Y, idx)
    return idx, f_in, f_out, time.time() - t0


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — sweeping output_weight at a fixed n_exp
# ══════════════════════════════════════════════════════════════════════════
def figure_1(sc, model, TIC):
    build_grid, make_designer, design_b_opt, true_fin_fout, in_lab, out_lab = sc

    d0 = make_designer(TIC, verbose=0)
    Y = np.asarray(d0.response).reshape(len(TIC), -1)

    results = []
    print(f"\n--- figure 1: n_exp = {N_EXP_FIG1}, sweeping output_weight ---")
    for w in WEIGHTS:
        idx, f_in, f_out, secs = run_case(sc, TIC, N_EXP_FIG1, w)
        results.append({"w": w, "idx": idx, "f_in": f_in, "f_out": f_out})
        print(f"  weight {w:4.2f}   f_in {f_in:11.4e}   f_out {f_out:11.4e}"
              f"   chosen {sorted(idx.tolist())}   [{secs:.1f}s]")

    fig, axes = plt.subplots(2, len(WEIGHTS), figsize=(3.1 * len(WEIGHTS), 6.4))
    fig.patch.set_facecolor("white")

    # normalise inputs to [-1, 1] for a fair 2-D view; plot the first two axes
    lb, ub = TIC.min(axis=0), TIC.max(axis=0)
    U = 2.0 * (TIC - lb) / (ub - lb) - 1.0

    for col, res in enumerate(results):
        idx = res["idx"]

        ax = axes[0, col]
        ax.scatter(U[:, 0], U[:, 1], s=26, facecolor="white",
                   edgecolor=STONE, linewidth=1.1, zorder=2)
        ax.scatter(U[idx, 0], U[idx, 1], s=115, facecolor=RED,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_title(f"output_weight = {res['w']:.2f}", fontsize=12,
                     fontweight="bold", color=BLACK, pad=9)
        if col == 0:
            ax.set_ylabel(f"INPUT space\n{in_lab[1]} (scaled)",
                          fontsize=11, color=BLACK)
        ax.set_xlabel(f"{in_lab[0]} (scaled)", fontsize=10, color=BLACK)

        ax = axes[1, col]
        ax.scatter(Y[:, 0], Y[:, 1], s=26, facecolor="white",
                   edgecolor=STONE, linewidth=1.1, zorder=2)
        ax.scatter(Y[idx, 0], Y[idx, 1], s=115, facecolor=BLUE,
                   edgecolor="white", linewidth=1.5, zorder=3)
        if col == 0:
            ax.set_ylabel(f"OUTPUT space\n{out_lab[1]}", fontsize=11, color=BLACK)
        ax.set_xlabel(out_lab[0], fontsize=10, color=BLACK)

    for ax in axes.ravel():
        ax.tick_params(labelsize=9, colors=BLACK, length=0)
        for sp in ax.spines.values():
            sp.set_color(STONE)

    fig.suptitle(
        f"b-optimal: input-space coverage (0) to output-space coverage (1)"
        f"   —   {model}, n_exp = {N_EXP_FIG1}",
        fontsize=14, fontweight="bold", color=BLACK, y=0.99)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    out = os.path.abspath(f"b_opt_fig1_{model}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  WROTE {out}  ({os.path.getsize(out)} bytes)")
    return results


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — the Pareto front, one curve per n_exp
# ══════════════════════════════════════════════════════════════════════════
def figure_2(sc, model, TIC):
    colours = [RED, BLUE, GREEN, GOLD, BROWN]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    summary = []
    print("\n--- figure 2: Pareto front per n_exp ---")
    for k, n_exp in enumerate(N_EXP_LIST):
        pts = []
        for w in WEIGHTS:
            idx, f_in, f_out, secs = run_case(sc, TIC, n_exp, w)
            pts.append((f_in, f_out))
            summary.append((n_exp, w, f_in, f_out))
            print(f"  n_exp {n_exp}  weight {w:4.2f}"
                  f"   f_in {f_in:11.4e}   f_out {f_out:11.4e}   [{secs:.1f}s]")
        pts = np.array(pts)
        order = np.argsort(pts[:, 0])
        ax.plot(pts[order, 0], pts[order, 1], "-o",
                color=colours[k % len(colours)], linewidth=2.0,
                markersize=9, markeredgecolor="white", markeredgewidth=1.4,
                label=f"n_exp = {n_exp}")
        for (fi, fo), w in zip(pts, WEIGHTS):
            ax.annotate(f"{w:.2f}", xy=(fi, fo), xytext=(5, 5),
                        textcoords="offset points", fontsize=9, color=BROWN)

    ax.set_xlabel("f_in   input-space coverage  (det of the scaled input moment matrix)",
                  fontsize=12, color=BLACK)
    ax.set_ylabel("f_out   output-space spread", fontsize=12, color=BLACK)
    ax.set_title(f"The frontier moves out as the budget grows — {model}",
                 fontsize=14, fontweight="bold", color=BLACK, pad=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelsize=10, colors=BLACK)
    for sp in ax.spines.values():
        sp.set_color(STONE)
    ax.grid(True, which="both", color=STONE, alpha=0.35, linewidth=0.6)
    ax.legend(frameon=False, fontsize=11.5, title="experiments",
              title_fontsize=11.5)

    plt.tight_layout()
    out = os.path.abspath(f"b_opt_fig2_{model}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  WROTE {out}  ({os.path.getsize(out)} bytes)")

    print("\n  summary table (quote these on the slide)")
    print("  n_exp  weight        f_in       f_out")
    for n_exp, w, fi, fo in summary:
        print(f"  {n_exp:5d}  {w:6.2f}  {fi:10.4e}  {fo:10.4e}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="film", choices=["film", "cstr", "both"])
    ap.add_argument("--n-candidates", type=int, default=None,
                    help="sampled candidates; defaults per model "
                         "(film 36, cstr 15000 -- the CSTR keeps ~0.5%)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fig", default="both", choices=["1", "2", "both"])
    args, _unknown = ap.parse_known_args()

    models = ["film", "cstr"] if args.model == "both" else [args.model]

    for model in models:
        sc = load_model(model)
        build_grid = sc[0]
        n_samp = args.n_candidates or N_CANDIDATES[model]
        if model == "cstr" and n_samp >= 5000:
            print(f"  sampling {n_samp} candidates and rejecting on the three"
                  f" quality constraints -- this takes a minute")
        TIC = build_grid(n_samp, args.seed)
        print(f"\n{'=' * 70}\n  {model}: {len(TIC)} feasible candidates "
              f"(sampled {n_samp}, seed {args.seed})\n{'=' * 70}")
        needed = max(max(N_EXP_LIST), N_EXP_FIG1) + 2
        if len(TIC) < needed:
            print(f"  ONLY {len(TIC)} feasible candidates but {needed} are needed.")
            print(f"  Nothing will be drawn. Re-run with a larger pool, e.g.")
            print(f"      python b_opt_pareto_sweep.py --model {model} "
                  f"--n-candidates {n_samp * 4}")
            continue
        try:
            if args.fig in ("1", "both"):
                figure_1(sc, model, TIC)
            if args.fig in ("2", "both"):
                figure_2(sc, model, TIC)
        except Exception:
            print(f"\n  FAILED while drawing {model} — traceback follows.")
            print("  Most likely: bonmin is not on PATH. Check with")
            print("      python -c \"from pyomo.environ import SolverFactory; "
                  "print(SolverFactory('bonmin').available())\"")
            traceback.print_exc()


# Runs both as a script and under Spyder/IPython %runfile, where argv may carry
# extras and __name__ is not always "__main__".
if __name__ == "__main__" or "get_ipython" in dir():
    print(f"working directory: {os.getcwd()}")
    main()
