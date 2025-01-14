import os


class LLMConfig:
    def __init__(self):
        self.model_pretrained_model_name_or_path = os.getenv("LLM_PRETRAINED_MODEL_NAME_OR_PATH", "TheBloke/Llama-2-7B-Chat-GGUF")
        self.model_type = os.getenv("LLM_MODEL_TYPE", "llama")
        self.model_file = os.getenv("LLM_MODEL_FILE", "llama-2-7b-chat.Q4_K_M.gguf")
        self.model_context_length = int(os.getenv("LLM_MODEL_CONTEXT_LENGTH", "8192"))
        self.model_cpu_threads = int(os.getenv("LLM_MODEL_CPU_THREADS", "8"))
        self.model_gpu_layers = os.getenv("LLM_MODEL_GPU_LAYERS", None)

        self.model_max_new_tokens = int(os.getenv("LLM_MODEL_MAX_NEW_TOKENS", "512"))
        self.model_temperature = float(os.getenv("LLM_MODEL_TEMPERATURE", "0.7"))
        self.model_top_p = float(os.getenv("LLM_MODEL_TOP_P", "0.95"))

        self.model_behaviour_context = os.getenv("LLM_MODEL_BEHAVIOUR_CONTEXT", None)

        self.model_cache_dir = os.getenv("LLM_CACHE_DIR", './cache')