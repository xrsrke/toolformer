from transformers import AutoModelForCausalLM, AutoTokenizer

from toolformer.data_generator import DataGenerator
from toolformer.prompt import calculator_prompt

def test_data_generator(default_configs):
    data_generator = DataGenerator(default_configs)

    model = AutoModelForCausalLM.from_pretrained(default_configs['model']['path'])
    tokenizer = AutoTokenizer.from_pretrained(default_configs['tokenizer']['path'])
    sentence = "Input: A runner completed a 10 kilometer race in 2 hours. Their average speed was 5 kilometers per hour."
    prompt = calculator_prompt.format(input=sentence)

    augmented_prompt = data_generator.generate(prompt, model, tokenizer)

    pass