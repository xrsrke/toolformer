import pytest

from toolformer.utils import extract_api_content, extract_api_name

def test_extract_api_content():
    text = "From this, we have 10 - 5 minutes = [Calculator(10 - 5)] 5 minutes."
    # text = "From this, we have 10 - 5 minutes = [Calculator((2+3) - 1)] 5 minutes." # TODO: add test case for this
    target = "10 - 5"

    output = extract_api_content(text, api_name="Calculator")

    assert isinstance(output, str)
    assert output == target

@pytest.mark.parametrize(
    "text, is_end_token, target",
    [
        ("From this, we have 10 - 5 minutes = [Calculator(10 - 5)] 5 minutes.", True, "Calculator"),
        ("[Calculator(10 - 5)", False, "Calculator"),
    ],
)
def test_extract_api_name(text, is_end_token, target):
    output = extract_api_name(text, is_end_token=is_end_token)

    assert isinstance(output, str)
    assert output == target
