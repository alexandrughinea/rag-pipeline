import os
from typing import Iterator

from ctransformers import AutoConfig, AutoModelForCausalLM


class LLMHandler:

    def __init__(self):
        default_model = "TheBloke/Llama-2-7B-Chat-GGUF"
        model_path_or_repo_id = os.getenv("LLM_MODEL_PATH_OR_REPO_ID", default_model)
        model_type = os.getenv("LLM_MODEL_TYPE", "llama")
        model_file = os.getenv("LLM_MODEL_FILE", "llama-2-7b-chat.Q4_K_M.gguf")
        model_context_length = os.getenv("LLM_MODEL_CONTEXT_LENGTH", "8192")
        #model_gpu_layers = os.getenv("LLM_MODEL_GPU_LAYERS", "1")
        model_cpu_threads = os.getenv("LLM_MODEL_CPU_THREADS", "8")

        # Ensure model exists
        if not model_path_or_repo_id:
            raise Exception(f"Model not found: {model_path_or_repo_id}")

        # Create config first
        config = AutoConfig.from_pretrained(
            model_path_or_repo_id,
            #gpu_layers=int(model_gpu_layers),
            threads=int(model_cpu_threads),
            context_length=int(model_context_length)
        )

        # Use config when loading model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id,
            model_type=model_type,
            model_file=model_file,
            config=config,
        )

    async def generate_stream(self, query: str, context: list[str]) -> Iterator[str]:
        """Stream tokens from the model."""
        # Flatten context if it's nested
        if context and isinstance(context[0], list):
            context = context[0]  # Take first list if nested
            context_text = "\n".join(context)

        prompt = (
            "Based on the following context, answer the question. "
            "If the context doesn't contain relevant information,\n"
            "just say you could not find anything related.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:

            for token in self.model(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    stream=True
            ):
                yield token
        except Exception as e:
            yield f"\nError during generation: {str(e)}"