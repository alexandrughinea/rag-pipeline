import os
from pathlib import Path
from typing import AsyncIterator

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .llm_config import LLMConfig
from .llm_history import LLMHistory


class LLMHandler:

    def __init__(self):
        self.config = LLMConfig()

        # Ensure model exists
        if not self.config.model_pretrained_model_name_or_path:
            raise Exception(f"Model not found: {self.config.model_pretrained_model_name_or_path}")

        # Create cache dir
        cache_dir_dir_path = Path(self.config.model_cache_dir)
        cache_dir_dir_path.mkdir(exist_ok=True)

        # Create config first
        config = AutoConfig.from_pretrained(
            self.config.model_pretrained_model_name_or_path,
            threads=self.config.model_cpu_threads,
            cache_dir=self.config.model_cache_dir,
            context_length=self.config.model_context_length,
        )

        # Config not needed here
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_pretrained_model_name_or_path,
        )

        # Config is needed here
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_pretrained_model_name_or_path,
            config=config,
            token=os.getenv("HF_TOKEN", None)
        )

        self.history = LLMHistory()

    async def generate_stream(self, query: str, context: list[str]) -> AsyncIterator[str]:
        """Stream tokens from the model."""

        if context and isinstance(context[0], list):
            context = context[0]

        context_text = "\n".join(context)

        prompt = self.tokenizer(
            f"Behaviour: {self.config.model_behaviour_context}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            for token in self.model(
                    prompt,
                    max_new_tokens=self.config.model_max_new_tokens,
                    temperature=self.config.model_temperature,
                    top_p=self.config.model_top_p,
                    stream=True
            ):
                yield token
        except Exception as e:
            yield f"\nError during generation: {str(e)}"

   
    async def generate_conversation_stream(self, query: str, context: list, conversation_id: int = None) -> AsyncIterator[str]:
        """Stream tokens from the model in a conversation context."""

        if conversation_id is None:
            conversation_id = self.history.create_conversation()

        print(f"Conversation ID: {conversation_id}")

        if context and isinstance(context[0], list):
            context = context[0]

        conversation = self.history.get_conversation(conversation_id)

        print(f"Conversation text: {conversation}")

        conversation_context = "\n".join([
            f"{message['role']}: {message['content']}" for message in conversation
        ])

        prompt = self.tokenizer(f"Previous conversation:\n{conversation_context}\n\n"
                  f"Behaviour: {self.config.model_behaviour_context}\n\n"
                  f"Context:\n{context}\n\n"
                  f"User: {query}\n"
                  f"Assistant:")

        self.history.add_message(conversation_id, "user", query)

        response = ""

        try:
            for token in self.model(
                    prompt,
                    max_new_tokens=self.config.model_max_new_tokens,
                    temperature=self.config.model_temperature,
                    top_p=self.config.model_top_p,
                    stream=True
            ):
                response += token
                yield token

            self.history.add_message(conversation_id, "assistant", response)
        except Exception as e:
            yield f"\nError during generation: {str(e)}"

