from transformers import AutoModelForCausalLM, AutoTokenizer

from toolformer.data_generator import DataGenerator
from toolformer.prompt import calculator_prompt

def test_create_data_generator(default_config):
    model = AutoModelForCausalLM.from_pretrained(default_config['model']['path'])
    tokenizer = AutoTokenizer.from_pretrained(default_config['tokenizer']['path'])

    generator = DataGenerator(default_config, model, tokenizer, apis=[])

    pass

def test_generate_data_generator(default_config):
    data_generator = DataGenerator(default_config)

    model = AutoModelForCausalLM.from_pretrained(default_config['model']['path'])
    tokenizer = AutoTokenizer.from_pretrained(default_config['tokenizer']['path'])
    sentence = "Input: A runner completed a 10 kilometer race in 2 hours. Their average speed was 5 kilometers per hour."
    prompt = calculator_prompt.format(input=sentence)

    augmented_prompt = data_generator.generate(prompt, model, tokenizer)

    pass