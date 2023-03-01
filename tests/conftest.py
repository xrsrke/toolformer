import pytest
from toolformer.utils import yaml2dict

@pytest.fixture
def default_configs():
    return yaml2dict('./configs/default.yaml')