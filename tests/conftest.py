import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from toolformer.utils import yaml2dict

@pytest.fixture
def default_config():
    return yaml2dict('./configs/default.yaml')

@pytest.fixture
def model(default_config):
    return AutoModelForCausalLM.from_pretrained(default_config['model']['path'])

@pytest.fixture
def tokenizer(default_config):
    return AutoTokenizer.from_pretrained(default_config['tokenizer']['path'])