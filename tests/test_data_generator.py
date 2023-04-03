import os

import pytest

import torch
import torch.nn.functional as F
from langchain import PromptTemplate

from toolformer.data_generator import DataGenerator
from toolformer.api import CalculatorAPI, WolframeAPI
from toolformer.prompt import calculator_prompt, wolframe_prompt

WOLFRAME_API_KEY = os.environ.get('WOLFRAME_API_KEY')

def test_sampling_apis_call(
    data_generator, prompt_tempalte,
    tokenizer
):
    text = "From this, we have 10 - 5 minutes = 5 minutes."
    prompt = prompt_tempalte.format(input=text)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]

    api_start_idxs, generated_ids = data_generator.sample_api_position(prompt_ids)

    assert isinstance(api_start_idxs, torch.Tensor)
    assert isinstance(generated_ids, torch.Tensor)

def test_obtain_api_calls(data_generator, tokenizer):
    text = "From this, we have 10 - 5 minutes = 5 minutes."
    prompt_tempalte = PromptTemplate(template=calculator_prompt, input_variables=["input"])
    prompt_ids = tokenizer(prompt_tempalte.format(input=text), return_tensors="pt")["input_ids"][0]

    generated_text = "From this, we have 10 - 5 minutes = [Calculator(10 - 5)] 5 minutes."
    api_start_idxs = torch.tensor([
        # 1, 3, 5, 7,
        9, 10
    ])
    generated_ids = tokenizer(generated_text, return_tensors="pt")["input_ids"][0]

    candidate_ids = data_generator.obtain_api_response(prompt_ids, api_start_idxs, generated_ids)

    assert isinstance(candidate_ids, torch.Tensor)
    assert candidate_ids.shape[0] == len(api_start_idxs)

def test_filtering_api_call(default_config, model, tokenizer):
    text = "From this, we have 10 - 5 minutes = 5 minutes."
    text_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    api_start_ids = torch.tensor([
        # 1, 3, 5, 7,
        9, 10
    ])

    api = CalculatorAPI("Calculator", calculator_prompt)
    candidates = [
        # "From [Calculator(10 - 5)] 5 minutes.",
        # "From this, [Calculator(10 - 5)] 5 minutes.",
        # "From this, we have [Calculator(10 - 5)] 5 minutes.",
        # "From this, we have 10 - [Calculator(5)] 5 minutes.",
        "From this, we have 10 - 5 minutes [Calculator(10 - 5)] 5 minutes.",
        "From this, we have 10 - 5 minutes = [Calculator(10 - 5)] 5 minutes."
    ]
    MAX_PAD = 30
    PAD_TOKEN_ID = tokenizer.pad_token_id
    candidate_ids = torch.tensor([])

    # PADDING
    for x in candidates:
        text_id = tokenizer(x, return_tensors="pt")["input_ids"]
        candidate_ids = torch.cat([
            candidate_ids,
            F.pad(text_id, pad=(MAX_PAD-text_id.shape[-1], 0), value=PAD_TOKEN_ID)
        ], dim=0)
    candidate_ids = candidate_ids.long()

    generator = DataGenerator(default_config, model, tokenizer, apis=[])

    filtered_candidate_ids = generator.filter_api(api, text_ids, api_start_ids, candidate_ids)

    assert isinstance(filtered_candidate_ids, torch.Tensor)

calculator_api = CalculatorAPI("Calculator", calculator_prompt)
wolframe_api = WolframeAPI("Wolframe", wolframe_prompt, api_key=WOLFRAME_API_KEY)

@pytest.mark.parametrize("apis", [
    [calculator_api],
    # [wolframe_api], # TODO: fix it
    # [calculator_api, wolframe_api], # TODO: fix it
])
def test_generate_data_generator(default_config, model, tokenizer, apis):
    text = "From this, we have 10 - 5 minutes = 5 minutes."

    generator = DataGenerator(default_config, model, tokenizer, apis=apis)

    augumented_text_ids = generator.generate(text)

    assert augumented_text_ids.shape[0] == len(apis)
    assert augumented_text_ids.ndim == 3
    assert isinstance(augumented_text_ids, torch.Tensor)