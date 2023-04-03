import torch
import pytest

from toolformer.model import ToolFormer

@pytest.mark.skip(reason="haven't implemented yet")
def test_inference(model, tokenizer, default_config):
    text = "What is the sum of 42 and 69?"
    target_output = 111

    encoded_text = tokenizer(text, return_tensors="pt")
    toolformer = ToolFormer(model, apis=[], config=default_config)

    output_ids = toolformer(
        input_ids=encoded_text["input_ids"],
        attention_mask=encoded_text["attention_mask"]
    )

    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.ndim == 2
    assert output_ids[0].shape[-1] > len(encoded_text["input_ids"][0])
    assert target_output in tokenizer.decode(output_ids[0], skip_special_tokens=True)