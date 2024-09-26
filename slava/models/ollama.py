from langchain_ollama.llms import OllamaLLM

from slava.config import OLLAMA_MODEL_MAX_TOKENS, OLLAMA_MODEL_NUM_CTX, OLLAMA_MODEL_TEMPERATURE, OLLAMA_MODEL_TOP_K


class OllamaModel:
    def __init__(
        self,
        model_name,
        num_ctx: int = OLLAMA_MODEL_NUM_CTX,
        temperature: float = OLLAMA_MODEL_TEMPERATURE,
        top_k: int = OLLAMA_MODEL_TOP_K,
        max_tokens: int = OLLAMA_MODEL_MAX_TOKENS,
    ):
        self.model = OllamaLLM(
            model=model_name,
            num_ctx=num_ctx,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
        )
