import pytest

from toolformer.api import BaseAPI, execute_calculator


# generate test for execute_calculator
@pytest.mark.parametrize(
    "input, expected",
    (
        ["1 + 2", 3],
        ["2 / 0", ""],
        ["(2 * 3) + 4", 10],
    )
)
def test_execute_calculator_api(input, expected):
    api = BaseAPI(func=execute_calculator)
    assert api(input) == expected