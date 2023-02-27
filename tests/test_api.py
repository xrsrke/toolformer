import pytest

from toolformer.api import execute_calculator


# generate test for execute_calculator
@pytest.mark.parametrize(
    "input, expected",
    (
        ["1 + 2", 3],
        ["2 / 0", ""],
        ["(2 * 3) + 4", 10],
    )
)
def test_execute_calculator(input, expected):
    assert execute_calculator(input) == expected