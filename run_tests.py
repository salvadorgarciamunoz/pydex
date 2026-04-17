"""
run_tests.py
------------
Smoke-test runner for pydex examples.

Runs each example module as an import-level check to verify that the
core functionality and any new additions are working correctly.
Each test module should be self-contained and raise an exception if
anything fails.

Usage
-----
    python run_tests.py

from the repository root.
"""

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(ROOT, "examples")

def run_test(subdir, module_path, label):
    """Change into subdir, import module, change back, report result."""
    target = os.path.join(EXAMPLES, subdir)
    original = os.getcwd()
    try:
        os.chdir(target)
        sys.path.insert(0, target)
        __import__(module_path)
        print(f"  PASS  {label}")
    except Exception:
        print(f"  FAIL  {label}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        os.chdir(original)
        if target in sys.path:
            sys.path.remove(target)


print("=" * 55)
print("pydex test suite")
print("=" * 55)

run_test("non_linear",            "examples.non_linear.test_non_linear",               "Non-linear examples")
run_test("linear",                "examples.linear.test_linear_examples",               "Linear examples")
run_test("ode",                   "examples.ode.test_ode",                              "ODE examples")
run_test("non_cubic_spaces",      "examples.non_cubic_spaces.test_non_cubic_spaces",    "Non-cubic spaces")
run_test("time_varying_controls", "examples.time_varying_controls.test_tvc",            "Time-varying controls")
run_test("ipopt",                 "examples.ipopt.test_ipopt",                          "IPOPT examples")

# V-optimal: import-level smoke test only (full run requires IPOPT + cyipopt)
print("\nV-optimal smoke test (import + model check, no IPOPT required)...")
try:
    v_example = os.path.join(EXAMPLES, "v_optimal_test_case.py")
    spec = __import__("importlib.util").util.spec_from_file_location(
        "v_optimal_test_case", v_example
    )
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  PASS  V-optimal example (import + model definitions)")
except Exception:
    print("  FAIL  V-optimal example")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 55)
print("All tests passed.")
print("=" * 55)
