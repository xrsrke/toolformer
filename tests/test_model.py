import torch
from toolformer.model import ToolFormer

def test_inference(model, tokenizer, default_config):
    text = "Upon the death of the king, the crown passed to his son, "
    encoded_text = tokenizer(text, return_tensors="pt")
    toolformer = ToolFormer(model, apis=[], config=default_config)

    output = toolformer(
        input_ids=encoded_text["input_ids"],
        attention_mask=encoded_text["attention_mask"]
    )

    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2
    assert output[0].shape[-1] > len(encoded_text["input_ids"][0])