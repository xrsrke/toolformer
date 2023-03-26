import os

import pytest

from toolformer.api import CalculatorAPI, WolframeAPI
from toolformer.prompt import calculator_prompt, wolframe_prompt


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
    calculator_api = CalculatorAPI("Calculator", calculator_prompt)

    output = calculator_api(input)

    assert output == expected
    assert isinstance(output, str)

def test_execute_wolframe_api():
    WOLFRAME_API_KEY = os.environ.get('WOLFRAME_API_KEY')
    wolframe_api = WolframeAPI("Wolframe", wolframe_prompt, api_key=WOLFRAME_API_KEY)

    input = "integrate x^2 sin^3 x dx"
    output = wolframe_api(input)

    assert isinstance(output, str)
    assert len(output) > 0