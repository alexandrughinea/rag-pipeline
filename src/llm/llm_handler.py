from typing import AsyncIterator

from ctransformers import AutoConfig, AutoModelForCausalLM

from .llm_config import LLMConfig
from .llm_history import LLMHistory


class LLMHandler:

    def __init__(self):
        self.config = LLMConfig()

        # Ensure model exists
        if not self.config.model_path_or_repo_id:
            raise Exception(f"Model not found: {self.config.model_path_or_repo_id}")

        # Create config first
        config = AutoConfig.from_pretrained(
            self.config.model_path_or_repo_id,
            #gpu_layers=self.config.model_gpu_layers,
            threads=self.config.model_cpu_threads,
            context_length=self.config.model_context_length,
        )

        # Use config when loading model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path_or_repo_id,
            model_type=self.config.model_type,
            model_file=self.config.model_file,
            config=config,
        )

        self.history = LLMHistory()
        self.behaviour_context = "Based on the following context, answer the question."
        "If the context doesn't contain relevant information,\n"
        "just respond that you could not find anything related.\n\n"

    async def generate_stream(self, query: str, context: list[str]) -> AsyncIterator[str]:
        """Stream tokens from the model."""

        if context and isinstance(context[0], list):
            context = context[0]

        context_text = "\n".join(context)

        prompt = (
            f"Behaviour: {self.behaviour_context}\n\n"
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

        prompt = (f"Previous conversation:\n{conversation_context}\n\n"
                  f"Behaviour: {self.behaviour_context}\n\n"
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

