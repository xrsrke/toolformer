from toolformer.utils import extract_api_request_content

def test_extract_api_request_content():
    text = "From this, we have 10 - 5 minutes = [Calculator(10 - 5)] 5 minutes."
    # text = "From this, we have 10 - 5 minutes = [Calculator((2+3) - 1)] 5 minutes." # TODO: add test case for this
    target = "10 - 5"

    output = extract_api_request_content(text, api_name = "Calculator")

    assert isinstance(output, str)
    assert output == target