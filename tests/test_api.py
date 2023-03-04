import pytest

from toolformer.api import CalculatorAPI


# generate test for execute_calculator
@pytest.mark.parametrize(
    "input, expected",
    (
        ["1 + 2", str(3)],
        ["2 / 0", ""],
        ["(2 * 3) + 4", str(10)],
    )
)
def test_execute_calculator_api(input, expected):
    calculator_api = CalculatorAPI()

    output = calculator_api(input)

    assert output == expected
    assert isinstance(output, str)