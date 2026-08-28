"""Guards against silent assignment to attribute names the refactor renamed.

`model_parameters_names` and `measurable_responses_names` were upstream names
superseded when the fork's refactor introduced `model_parameter_names` and
`response_names`. Their declarations survived in `__init__` while nothing read
them, so assigning to one was accepted and discarded: a user's labels never
appeared and nothing said why. Designer now refuses those names.

Solver-free: constructs a bare Designer and only touches attributes.
"""

import pytest

from designer import Designer


RENAMED = {
    "model_parameters_names": "model_parameter_names",
    "measurable_responses_names": "response_names",
    "responses_names": "response_names",
}


@pytest.mark.parametrize("old_name,new_name", sorted(RENAMED.items()))
def test_assigning_a_renamed_attribute_raises(old_name, new_name):
    """Each superseded name must raise rather than be silently discarded."""
    designer = Designer()
    with pytest.raises(AttributeError) as excinfo:
        setattr(designer, old_name, ["a", "b"])
    message = str(excinfo.value)
    assert old_name in message, "the message must name the offending attribute"
    assert new_name in message, "the message must name the replacement"


@pytest.mark.parametrize("old_name", sorted(RENAMED))
def test_a_renamed_attribute_is_not_created_by_a_failed_assignment(old_name):
    """A refused assignment must not leave the attribute behind."""
    designer = Designer()
    with pytest.raises(AttributeError):
        setattr(designer, old_name, ["a", "b"])
    assert not hasattr(designer, old_name)


@pytest.mark.parametrize(
    "name,value",
    [
        ("model_parameter_names", ["A12", "A21"]),
        ("response_names", ["P", "y1"]),
        ("ti_controls_names", ["x1", "T"]),
        ("tv_controls_names", ["u"]),
        ("candidate_names", ["lot A", "lot B"]),
        ("model_parameter_unit_names", ["-", "-"]),
        ("response_unit_names", ["kPa", "-"]),
    ],
)
def test_the_supported_name_attributes_are_settable(name, value):
    """The guard must not interfere with the attributes that are real."""
    designer = Designer()
    setattr(designer, name, value)
    assert getattr(designer, name) == value


def test_the_guard_does_not_block_unrelated_attributes():
    """Users may still attach their own attributes to a Designer."""
    designer = Designer()
    designer.my_own_bookkeeping = {"run": 3}
    assert designer.my_own_bookkeeping == {"run": 3}


def test_the_dead_declarations_are_gone_from_a_fresh_designer():
    """A fresh Designer must not carry the superseded names at all.

    Before the fix these were initialised to None in __init__, which is what
    made a later assignment look plausible.
    """
    designer = Designer()
    for old_name in RENAMED:
        assert not hasattr(designer, old_name), (
            f"{old_name} should not exist on a fresh Designer"
        )


def test_the_rename_map_points_only_at_real_attributes():
    """Every replacement named in the map must itself be a real attribute.

    Protects against a future entry whose advice cannot be followed.
    """
    designer = Designer()
    for old_name, new_name in Designer._RENAMED_ATTRIBUTES.items():
        assert hasattr(designer, new_name), (
            f"{old_name} points at {new_name}, which is not a Designer attribute"
        )
        assert old_name not in vars(designer), (
            f"{old_name} is both refused and declared"
        )
