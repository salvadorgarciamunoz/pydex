"""
Small regression tests for Designer.interest_parameters (Ds-optimality).

Covers the property that matters most for correctness: interest_parameters
must always be resolved to FIM positions BY NAME against
model_parameter_names, and must be validated as a genuine SUBSET of it —
never by numeric index/position, since position is not a stable identifier
across differently-ordered Pyomo model declarations.

Run with:  python -m pytest test_ds_opt_interest_parameters.py -v
"""
import sys
import types

import numpy as np
import pytest


def _stub_pydex_modules():
    """
    designer.py imports pydex.utils.trellis_plotter and pydex.core.logger at
    module level purely for plotting/logging helpers that this test never
    exercises. Stub them so the test doesn't require the full pydex package
    (and its plotting stack) to be installed.
    """
    if "pydex" in sys.modules:
        return
    pydex_pkg = types.ModuleType("pydex")
    pydex_utils = types.ModuleType("pydex.utils")
    pydex_core = types.ModuleType("pydex.core")
    trellis_mod = types.ModuleType("pydex.utils.trellis_plotter")
    logger_mod = types.ModuleType("pydex.core.logger")

    class TrellisPlotter:
        pass

    class Logger:
        pass

    trellis_mod.TrellisPlotter = TrellisPlotter
    logger_mod.Logger = Logger

    sys.modules["pydex"] = pydex_pkg
    sys.modules["pydex.utils"] = pydex_utils
    sys.modules["pydex.core"] = pydex_core
    sys.modules["pydex.utils.trellis_plotter"] = trellis_mod
    sys.modules["pydex.core.logger"] = logger_mod


_stub_pydex_modules()
from designer import Designer  # noqa: E402  (import after stubbing)


@pytest.fixture
def d():
    designer = Designer()
    designer.n_mp = 4
    designer.model_parameter_names = ["Ka", "A0", "k1", "k2"]
    return designer


def test_valid_subset_resolves_by_name(d):
    d.interest_parameters = ["Ka", "A0"]
    idx_s, idx_n = d._resolve_ds_idx()
    assert list(idx_s) == [0, 1]
    assert list(idx_n) == [2, 3]


def test_resolution_is_order_independent(d):
    # Same names, but model_parameter_names declared in a totally different
    # order (simulating a Pyomo model whose equations were declared in a
    # different sequence) — must still resolve to the correct positions.
    d.model_parameter_names = ["k2", "Ka", "k1", "A0"]
    d.interest_parameters = ["Ka", "A0"]
    idx_s, idx_n = d._resolve_ds_idx()
    assert list(idx_s) == [1, 3]
    assert list(idx_n) == [0, 2]


def test_unknown_name_rejected_eagerly(d):
    # model_parameter_names is already known, so this must fail immediately
    # at assignment time, not later inside a criterion evaluation.
    with pytest.raises(ValueError, match="not found in model_parameter_names"):
        d.interest_parameters = ["Ka", "not_a_real_param"]
    # and the bad assignment must not have partially taken effect
    assert d.interest_parameters is None


def test_unknown_name_rejected_when_set_before_names_known():
    # interest_parameters set before model_parameter_names exists must defer
    # validation, then still raise once model_parameter_names is known.
    designer = Designer()
    designer.n_mp = 2
    designer.interest_parameters = ["Ka", "ghost_param"]
    designer.model_parameter_names = ["Ka", "k1"]
    with pytest.raises(ValueError, match="not found in model_parameter_names"):
        designer._resolve_ds_idx()


def test_numeric_index_rejected(d):
    with pytest.raises(TypeError, match="NAMES"):
        d.interest_parameters = [0, 2]


def test_duplicate_names_deduped(d):
    d.interest_parameters = ["Ka", "Ka", "A0"]
    assert d.interest_parameters == ["Ka", "A0"]
    idx_s, idx_n = d._resolve_ds_idx()
    assert list(idx_s) == [0, 1]
    assert list(idx_n) == [2, 3]


def test_full_parameter_set_leaves_no_nuisance_params(d):
    # a degenerate but valid subset: interest == all parameters
    d.interest_parameters = ["Ka", "A0", "k1", "k2"]
    idx_s, idx_n = d._resolve_ds_idx()
    assert list(idx_s) == [0, 1, 2, 3]
    assert list(idx_n) == []


def test_reset_to_none():
    designer = Designer()
    designer.n_mp = 2
    designer.model_parameter_names = ["Ka", "k1"]
    designer.interest_parameters = ["Ka"]
    designer.interest_parameters = None
    assert designer.interest_parameters is None
    assert designer.ds_interest_idx is None
    assert designer.ds_nuisance_idx is None
    with pytest.raises(SyntaxError, match="interest_parameters to be"):
        designer._resolve_ds_idx()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
