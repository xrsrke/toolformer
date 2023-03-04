# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_data_generator.ipynb.

# %% auto 0
__all__ = ['DataGenerator']

# %% ../nbs/04_data_generator.ipynb 4
import re
from typing import List, Callable, Tuple

import torch
import torch.nn.functional as F

from torchtyping import TensorType
from einops import rearrange
from langchain import PromptTemplate

from .api import BaseAPI
from .api import CalculatorAPI

# %% ../nbs/04_data_generator.ipynb 5
class DataGenerator:
    def __init__(self, config: dict, model: Callable, tokenizer: Callable, apis: List[BaseAPI],):
        start_character = config["data_generator"]["api_start_character"]
        end_character = config["data_generator"]["api_end_character"]
        output_character = config["data_generator"]["api_output_character"]
        
        # add a space, because when the model generate a token, it's also include a "space"
        self.api_start_token = tokenizer(f' {start_character}', return_tensors="pt")["input_ids"][0]
        self.api_end_token = tokenizer(end_character, return_tensors="pt")["input_ids"][0]
        self.api_output_token = tokenizer(f'{output_character}', return_tensors="pt")["input_ids"][0]
        
        self.top_k = config["data_generator"]["top_k"]
        self.sampling_threshold = config["data_generator"]["sampling_threshold"]
        self.filtering_threshold = config["data_generator"]["filtering_threshold"]
        
        self.apis = apis
        self.model = model
        self.tokenizer = tokenizer
        # TODO: handle for cases that the sentence contains ".\n\n"
        self.eos_token_id = tokenizer(".\n\n")["input_ids"][0]
    
    def extract_api_request_content(self, text: str, api_name: str) -> str:
        start_tag = f"{api_name}("
        end_tag = ")"
        start_idx = text.find(start_tag)
        if start_idx == -1:
            return None
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            return None
        return text[start_idx:end_idx]
    
    def extract_api_syntax(self, sentence: str, api_name: str) -> str:
        pattern = r"\[{}\(.*?\)\]".format(api_name)
        matches = re.findall(pattern, sentence)
        return matches
    
    def sample_api_position(
        self,
        prompt_ids: TensorType["batch_size", "seq_len"], # the ids of the prompt
    ) -> Tuple[
        TensorType["batch_size", "n_positions"], # The positions of api call
        TensorType["batch_size", "seq_len"] # The generated text
    ]:
        # TODO: add support batch
        
        # the ids of the prompt and generated_ids
        prompt_and_generated_ids = prompt_ids
        # only the ids of the generated_ids
        generated_ids = torch.tensor([])
        api_positions = torch.tensor([])
        i = torch.tensor([0])
        
        with torch.no_grad():    
            while True:
                logits = self.model(
                    input_ids=prompt_and_generated_ids.unsqueeze(0),
                ).logits

                last_logit = logits[0, -1, :]
                probs = torch.softmax(last_logit, dim=-1)
                
                # find the top k tokens for api call
                # TODO: add filter by larger than sampling_threshold
                top_k_tokens = torch.topk(probs, k=5, dim=-1)
                
                if self.api_start_token in top_k_tokens.indices:
                    # api_position = torch.tensor([len(generated_ids)]) # the current idx
                    api_positions = torch.cat((api_positions, i), dim=0)
                
                # sampling a token
                # next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.argmax(probs, dim=-1)
                next_token = next_token.unsqueeze(0)
                
                prompt_and_generated_ids = torch.cat([prompt_and_generated_ids, next_token], dim=0)
                generated_ids = torch.cat([generated_ids, next_token], dim=0)
                
                if next_token == self.eos_token_id:
                    break
                else:
                    i += 1
        
        return api_positions.long(), generated_ids.long()

    def obtain_api_response(
        self,
        prompt_ids: TensorType["batch_size", "seq_len"],
        positions: TensorType["batch_size", "n_positions"],
        generated_ids: TensorType["batch_size", "seq_len"]
    ) -> TensorType["batch_size", "n_positions", "seq_len"]:
        
        MAX_PAD = 50
        PAD_TOKEN = self.tokenizer.pad_token_id
        
        # the ids before the start of an api call
        pre_api_ids = torch.tensor([])

        for position in positions:
            text_ids = torch.cat([generated_ids[:position], self.api_start_token], dim=0)
            padded_text_ids = F.pad(text_ids, pad=(MAX_PAD - text_ids.shape[-1], 0), value=PAD_TOKEN)
            
            pre_api_ids = torch.cat([
                pre_api_ids,
                rearrange(padded_text_ids, "... -> 1 ...")
            ])
        
        PROMPT_LENGTH = len(prompt_ids)
        
        # TODO: optimzie this
        prompt_and_pre_api_ids = torch.tensor([])
        for x in pre_api_ids:
            prompt_and_pre_api_ids = torch.cat([
                prompt_and_pre_api_ids,
                torch.cat([prompt_ids, x]).unsqueeze(0)
            ], dim=0)
                     
        with torch.no_grad():
            candidates = self.model.generate(
                input_ids=prompt_and_pre_api_ids.long(),
                eos_token_id=self.eos_token_id,
                max_new_tokens=50,
            )
        
        # filter out the prompt template
        # only keep the generated ids
        candidates = candidates[:, PROMPT_LENGTH:]
        
        return candidates

    # def extract_pred_from_candidate(
    #     candidate_ids: TensorType["seq_len"] # the initial prompting to guide the model + generated_ids
    # ) -> TensorType["pred_len"]:
    #     """Extract the generated ids from the [prompt + generated_ids].
        
    #     Example: Your task is to add two numbers. [PREDICTION].
    #     Extracted ids: [PREDICTION].

    #     Returns:
    #         torch.Tensor: The ids of the prediction
    #     """
    #     pred_ids = candidate_ids[PROMPT_LENGTH:]
    #     return pred_ids
    
    def _generate_conditioning_prompts(
        self,
        prompt_ids: TensorType["batch_size", "seq_len"],
        candidate_ids: TensorType["batch_size", "n_candidates", "seq_len"],
    ):
        calculator_api = CalculatorAPI()
        # conditioning_prompts = torch.tensor([])
        conditioning_api_ids = torch.tensor([])
        target_ids = torch.tensor([])

        SPACE_TOKEN = self.tokenizer(" .", return_tensors="pt")["input_ids"][0]
        API_NAME = "Calculator"
        MAX_PAD = 100
        PAD_TOKEN = self.tokenizer.pad_token_id

        for text_ids in candidate_ids:
            # the ids of the prediction
            text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
            
            api_request_content = self.extract_api_request_content(text, api_name=API_NAME)
            api_response = calculator_api(api_request_content)
            api_response_ids = self.tokenizer(api_response, return_tensors="pt")["input_ids"][0]
            # Format: -> [api_response]
            api_response_with_arrow_ids = torch.cat([self.api_output_token, api_response_ids], dim=0)
            
            api_syntax = self.extract_api_syntax(text, api_name=API_NAME)
            api_syntax_ids = self.tokenizer(api_syntax, return_tensors="pt")["input_ids"][0]
            api_syntax_with_response_ids = torch.cat([api_syntax_ids[:-1], api_response_with_arrow_ids, api_syntax_ids[-1:]])
            api_syntax_without_response_ids = torch.cat([api_syntax_ids[:-1], self.api_output_token, api_syntax_ids[-1:]])
            
            api_start_idx = torch.where(text_ids == self.api_start_token)[0]
            pred_exclude_api_ids = text_ids[:api_start_idx]
            next_token_ids = text_ids[api_start_idx + 1]
                        
            promt_without_api = pred_exclude_api_ids
            # prompt_with_api_and_response = torch.cat([api_syntax_with_response_ids, SPACE_TOKEN, pred_exclude_api_ids], dim=0)
            # prompt_with_api_with_empty_response = torch.cat([api_syntax_without_response_ids, SPACE_TOKEN, pred_exclude_api_ids], dim=0)
            
            # padded_prompt_without_api = rearrange(
            #     F.pad(promt_without_api, pad=(0, (MAX_PAD - promt_without_api.shape[-1])), value=PAD_TOKEN),
            #     "... -> 1 ..."
            # )
            # padded_prompt_with_api_with_empty_response = rearrange(
            #     F.pad(prompt_with_api_with_empty_response, pad=(0, (MAX_PAD - prompt_with_api_with_empty_response.shape[-1])), value=PAD_TOKEN),
            #     "... -> 1 ..."
            # )
            # padded_prompt_with_api_and_response = rearrange(
            #     F.pad(prompt_with_api_and_response, pad=(0, (MAX_PAD - prompt_with_api_and_response.shape[-1])), value=PAD_TOKEN),
            #     "... -> 1 ..."
            # )
            
            padded_api_without_response = rearrange(
                F.pad(api_syntax_without_response_ids, pad=(0, (MAX_PAD - api_syntax_without_response_ids.shape[-1])), value=PAD_TOKEN),
                "... -> 1 ..."
            )
            padded_api_with_response = rearrange(
                F.pad(api_syntax_with_response_ids, pad=(0, (MAX_PAD - api_syntax_with_response_ids.shape[-1])), value=PAD_TOKEN),
                "... -> 1 ..."
            )
            
            # padded_prompt = torch.cat([
            #     padded_prompt_without_api,
            #     padded_prompt_with_api_with_empty_response,
            #     padded_prompt_with_api_and_response,
            # ], dim=0)
            
            padded_api_call = torch.cat([
                padded_api_without_response,
                padded_api_with_response
            ], dim=0)
            padded_api_call = rearrange(padded_api_call, "... -> 1 ...")
            
            # padded_prompt = rearrange(padded_prompt, "... -> 1 ...")

            # conditioning_prompts = torch.cat([conditioning_prompts, padded_prompt], dim=0).long()
            conditioning_api_ids = torch.cat([conditioning_api_ids, padded_api_call], dim=0).long()
            target_ids = torch.cat([target_ids, torch.tensor(next_token_ids)], dim=0).long()
                    
        return target_ids, conditioning_api_ids

    def filter_api( 
        self,
        prompt_ids: TensorType["batch_size", "seq_len"],
        candiates: TensorType["batch_size", "n_positions", "seq_len"]
    ):
        target_ids, conditioning_prompts = self._generate_conditioning_prompts(prompt_ids, candiates)
        return target_ids, conditioning_prompts
    
    def generate(
        self,
        prompt_tempalte: PromptTemplate,
        text: str,
    ) -> List[str]:
        # TODO: add support batch
        prompt = prompt_tempalte.format(input=text)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]  
    
        # sampling positions
        api_start_idx, generated_ids = self.sample_api_position(prompt_ids)
        
        # obtaining api responses
        candidates = self.obtain_api_response(prompt_ids, api_start_idx, generated_ids)

        # filtering
        target_ids, conditioning_prompts = self.filter_api(prompt_ids, candidates)
        
        return api_start_idx, generated_ids, target_ids, conditioning_prompts
