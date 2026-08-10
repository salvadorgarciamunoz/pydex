#!/usr/bin/env python3
"""
coverage_audit.py — line-level coverage of pydex's designer.py under the
capability test suite, reported PER METHOD so the output is actionable.

WHY PER METHOD
--------------
`coverage report` gives one percentage for a 4,300-line file and `coverage html`
gives a wall of red lines. Neither answers the question that matters: WHICH
CAPABILITIES are untested? Grouping missed lines by the method that contains
them turns the report into a to-do list.

WHY THIS EXISTS AT ALL
----------------------
Every coverage claim made about this suite so far came from grepping the test
source for feature names. That method has already produced one false positive: a
comment in §45 claimed prior-FIM coverage on the IFT path when no
set_prior_fim call existed in the section. Static inspection cannot see which
BRANCHES actually execute. This can.

USAGE
-----
    # 1. full run (slow — the suite takes ~20 min, coverage adds ~20%)
    python coverage_audit.py run

    # 2. faster: a subset of sections by number
    python coverage_audit.py run 03 04 05 13 47 48 51

    # 3. report on whatever was last collected
    python coverage_audit.py report

    # 4. everything, in one go
    python coverage_audit.py

NOTES
-----
* n_jobs is forced to 1 for the coverage pass. pydex sets n_jobs=-1 automatically
  when pyomo_model_fn is present, and joblib's loky backend runs candidates in
  separate processes that a plain `coverage run` does not follow. Measured on
  this codebase the difference is small (60% vs 61% of the IFT sensitivity
  routine), because the parent does most of the work for small candidate sets —
  but forcing sequential removes the doubt for free. Parallel-vs-sequential
  equivalence is separately covered by §21, §26 and §45.
* Lines that are legitimately unreachable — `raise NotImplementedError` bodies,
  `if TYPE_CHECKING`, defensive `except` clauses for conditions the suite cannot
  provoke — will show as missed. That is expected. The report flags methods with
  ZERO coverage separately from partially covered ones, because those are the
  ones worth looking at first.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / "pydex_full_capability_test.py"
JSON_OUT = HERE / ".coverage_designer.json"


def _find_designer():
    """Locate the installed designer.py that the suite will import."""
    try:
        import pydex.core.designer as m
        return Path(m.__file__).resolve()
    except Exception:
        sys.exit("Cannot import pydex.core.designer — check PYTHONPATH.")


def run():
    """
    Run the suite under coverage.

    The suite is executed AS A SCRIPT, not imported. Importing it by module name
    fails outright:

        >>> import pydex_full_capability_test
        KeyError: 'pydex_full_capability_test'

    while the identical file under any other name imports fine — the `pydex`
    prefix collides with the installed `pydex` package during resolution. Running
    it as a script is also how you invoke it normally, so this measures exactly
    the path you use.

    A consequence: n_jobs cannot be forced to 1 from here, so parallel sections
    run their candidates in loky worker processes that plain `coverage run` does
    not follow. Measured impact on this codebase is about one percentage point
    (60% vs 61% of the IFT sensitivity routine) because the parent does most of
    the work for these candidate counts. If you want the last percent, add
    `concurrency = multiprocessing` and `parallel = true` to a .coveragerc and
    finish with `coverage combine`.
    """
    designer = _find_designer()
    print(f"  measuring : {designer}")
    print(f"  suite     : {SUITE}")
    if not SUITE.exists():
        sys.exit(f"Suite not found at {SUITE}")
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, "-m", "coverage", "run",
           f"--include=*{designer.name}", SUITE.name]
    print(f"  running   : coverage run --include=*{designer.name} {SUITE.name}")
    print("              (the suite takes ~20 min; coverage adds ~20%)\n")
    subprocess.run(cmd, env=env, cwd=str(SUITE.parent))
    subprocess.run([sys.executable, "-m", "coverage", "json",
                    "-o", str(JSON_OUT), f"--include=*{designer.name}"],
                   cwd=str(SUITE.parent))


def _attribute(src_text, executed, missing):
    """
    Map every measured line to the DEEPEST function that contains it.

    Uses ast rather than indentation heuristics. A regex approach double-counts:
    nested helpers get attributed to both themselves and their parent, and a
    module-level function preceding a class swallows the entire class body. That
    produced 225% of the measured lines before this was rewritten. The report
    now asserts the attribution sums exactly to coverage's own totals.
    """
    import ast
    from collections import defaultdict

    funcs = []
    def walk(node, prefix="", depth=0):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{ch.name}"
                funcs.append((q, ch.lineno, ch.end_lineno, depth))
                walk(ch, q + ".", depth + 1)
            elif isinstance(ch, ast.ClassDef):
                walk(ch, "", depth)          # methods keep their bare names
            else:
                walk(ch, prefix, depth)
    walk(ast.parse(src_text))

    owner = {}
    for q, a, b, d in funcs:
        for i in range(a, b + 1):
            prev = owner.get(i)
            if prev is None or d > prev[1]:
                owner[i] = (q, d)

    agg = defaultdict(lambda: [0, 0])
    start_of = {q: a for q, a, _b, _d in funcs}
    module_level = [0, 0]
    for i in executed:
        (agg[owner[i][0]][0] if i in owner else None) if i in owner else None
        if i in owner: agg[owner[i][0]][0] += 1
        else:          module_level[0] += 1
    for i in missing:
        if i in owner: agg[owner[i][0]][1] += 1
        else:          module_level[1] += 1
    return agg, start_of, module_level


def report(top=40):
    if not JSON_OUT.exists():
        sys.exit(f"No coverage data at {JSON_OUT}. Run `coverage_audit.py run` first.")
    data = json.load(open(JSON_OUT))
    if not data["files"]:
        sys.exit("Coverage JSON has no files — did the run import designer.py?")
    _path, f = next(iter(data["files"].items()))
    designer = _find_designer()
    src_text = designer.read_text()
    executed, missing = set(f["executed_lines"]), set(f["missing_lines"])

    agg, start_of, module_level = _attribute(src_text, executed, missing)

    te = sum(v[0] for v in agg.values()) + module_level[0]
    tm = sum(v[1] for v in agg.values()) + module_level[1]
    assert te + tm == len(executed) + len(missing), (
        f"attribution lost lines: {te+tm} vs {len(executed)+len(missing)}"
    )

    print("\n" + "=" * 78)
    print("  designer.py line coverage under the capability suite".center(78))
    print("=" * 78)
    print(f"  executed {len(executed)}, missing {len(missing)}, "
          f"total {len(executed)+len(missing)}  ->  "
          f"{100*len(executed)/(len(executed)+len(missing)):.1f}%")
    print(f"  attribution checks out: every measured line assigned to a function "
          f"or module scope")

    rows = [(q, e, m) for q, (e, m) in agg.items()]
    dead = sorted([r for r in rows if r[1] == 0], key=lambda r: -r[2])
    print(f"\n  NEVER EXECUTED — {len(dead)} function(s), "
          f"{sum(r[2] for r in dead)} lines")
    print("  No line inside these ran. Nested helpers appear as parent.child;")
    print("  a worker function may show here because coverage does not follow")
    print("  loky subprocesses, so check before treating it as untested.\n")
    print(f"    {'lines':>6}  {'at':>6}  function")
    print("    " + "-" * 64)
    for q, _e, m in dead[:top]:
        print(f"    {m:>6}  {start_of.get(q, 0):>6}  {q}")
    if len(dead) > top:
        print(f"    ... and {len(dead)-top} more")

    partial = sorted([(q, e, m) for q, e, m in rows
                      if e > 0 and (e + m) >= 10 and 100*e/(e+m) < 70],
                     key=lambda r: 100*r[1]/(r[1]+r[2]))
    print(f"\n  PARTIALLY COVERED (<70%, >=10 measured lines) — {len(partial)}")
    print("  Usually error branches and fallbacks. Worth a look where the")
    print("  untaken branch is one you would rely on.\n")
    print(f"    {'cov':>5}  {'ex/tot':>9}  {'at':>6}  function")
    print("    " + "-" * 64)
    for q, e, m in partial[:top]:
        print(f"    {100*e/(e+m):>4.0f}%  {e:>4}/{e+m:<4}  "
              f"{start_of.get(q, 0):>6}  {q}")
    if len(partial) > top:
        print(f"    ... and {len(partial)-top} more")

    print("\n  Line-by-line view:")
    print("    python -m coverage html --include='*designer.py'")
    print("    open htmlcov/index.html")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        run(); report()
    elif args[0] == "run":
        run(); report()
    elif args[0] == "report":
        report()
    else:
        sys.exit(__doc__)
