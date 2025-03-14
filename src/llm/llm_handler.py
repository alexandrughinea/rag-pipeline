import os
from pathlib import Path
from typing import AsyncIterator

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .llm_config import LLMConfig
from .llm_history import LLMHistory


class LLMHandler:

    def __init__(self):
        self.config = LLMConfig()

        if not self.config.model_pretrained_model_name_or_path:
            raise Exception(f"Model not found: {self.config.model_pretrained_model_name_or_path}")

        cache_dir_dir_path = Path(self.config.model_cache_dir)
        cache_dir_dir_path.mkdir(exist_ok=True)

        config = AutoConfig.from_pretrained(
            self.config.model_pretrained_model_name_or_path,
            trust_remote_code=True,
            cache_dir=self.config.model_cache_dir,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_pretrained_model_name_or_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_pretrained_model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_TOKEN", None)
        )

        self.history = LLMHistory()

    async def generate_stream(self, query: str, context: list[str]) -> AsyncIterator[str]:
        """Stream tokens from the model."""
        if context and isinstance(context[0], list):
            context = context[0]

        context_text = "\n".join(context)

        prompt = (
            f"### System:\n{self.config.model_behaviour_context}\n\n"
            f"### Context:\n{context_text}\n\n"
            f"### User:\n{query}\n\n"
            f"### Assistant:\n"
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.model_max_new_tokens,
                temperature=self.config.model_temperature,
                top_p=self.config.model_top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )

            generated_sequence = outputs.sequences[0]

            input_length = len(input_ids[0])

            generated_tokens = generated_sequence[input_length:]

            for i in range(len(generated_tokens)):
                token = generated_tokens[i:i+1]
                token_text = self.tokenizer.decode(token, skip_special_tokens=True)
                if token_text:
                    yield token_text

        except Exception as e:
            yield f"\nError during generation: {str(e)}"

    async def generate_conversation_stream(self, query: str, context: list, conversation_id: int = None) -> AsyncIterator[str]:
        """Stream tokens from the model in a conversation context."""
        if conversation_id is None:
            conversation_id = self.history.create_conversation()
            print(f"Conversation ID: {conversation_id}")

        if context and isinstance(context[0], list):
            context = context[0]

        context_text = "\n".join(context)

        conversation = self.history.get_conversation(conversation_id)
        conversation_context = "\n".join([
            f"{message['role']}: {message['content']}" for message in conversation
        ])

        prompt = (
            f"### Previous Conversation:\n{conversation_context}\n\n"
            f"### System:\n{self.config.model_behaviour_context}\n\n"
            f"### Context:\n{context_text}\n\n"
            f"### User:\n{query}\n\n"
            f"### Assistant:\n"
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        self.history.add_message(conversation_id, "user", query)
        response = ""

        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.model_max_new_tokens,
                temperature=self.config.model_temperature,
                top_p=self.config.model_top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )

            generated_sequence = outputs.sequences[0]

            input_length = len(input_ids[0])

            generated_tokens = generated_sequence[input_length:]

            for i in range(len(generated_tokens)):
                token = generated_tokens[i:i+1]
                token_text = self.tokenizer.decode(token, skip_special_tokens=True)
                if token_text:
                    yield token_text
                    response += token_text
                    
            self.history.add_message(conversation_id, "assistant", response.strip())

        except Exception as e:
            yield f"\nError during generation: {str(e)}"