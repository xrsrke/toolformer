import torch
import pytest
from langchain import PromptTemplate

from toolformer.model import ToolFormer
from toolformer.api import BaseAPI
from toolformer.prompt import calculator_prompt


class CalculatorAPI(BaseAPI):
    def __call__(self, text):
        return str(4269)


calculator_api = CalculatorAPI(
    name="Calculator",
    prompt_template=calculator_prompt
)


# @pytest.mark.skip(reason="haven't implemented yet")
def test_inference(model, tokenizer, default_config):
    text = "From this, we have 10 - 5 minutes = 5 minutes."

    # After fine-tune a model with augmented data,
    # the model should be able to call the API without few-shot learning
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template=calculator_prompt
    )
    input = prompt_template.format(input=text)
    target_output = str(4269)  # from the calculator API

    encoded_text = tokenizer(input, return_tensors="pt")
    toolformer = ToolFormer(
        model,
        apis=[calculator_api],
        config=default_config
    )

    output_ids = toolformer(
        input_ids=encoded_text["input_ids"],
        attention_mask=encoded_text["attention_mask"],
        max_new_tokens=30,
    )

    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.ndim == 2
    assert output_ids[0].shape[-1] > len(encoded_text["input_ids"][0])
    assert target_output in tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
