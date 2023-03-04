from langchain import PromptTemplate

from toolformer.data_generator import DataGenerator
from toolformer.prompt import calculator_prompt

def test_create_data_generator(default_config):
    # model = AutoModelForCausalLM.from_pretrained(default_config['model']['path'])
    # tokenizer = AutoTokenizer.from_pretrained(default_config['tokenizer']['path'])

    # generator = DataGenerator(default_config, model, tokenizer, apis=[])
    pass

def test_sampling_apis_call(default_config, model, tokenizer):
    pass

def test_execute_apis_call(default_config):
    pass

def test_filtering_apis_call(default_config):
    pass

def test_generate_data_generator(default_config, model, tokenizer):
    text = "From this, we have 10 - 5 minutes = 5 minutes."
    prompt_tempalte = PromptTemplate(template=calculator_prompt, input_variables=["input"])

    generator = DataGenerator(default_config, model, tokenizer, apis=[])

    target_ids, conditioning_prompts = generator.generate(prompt_tempalte, text)