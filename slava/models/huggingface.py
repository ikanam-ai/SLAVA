from langchain_huggingface import HuggingFacePipeline

from slava.config import HUGGINGFACE_MODEL_MAX_TOKENS, HUGGINGFACE_MODEL_TEMPERATURE, HUGGINGFACE_MODEL_TOP_K


class HuggingFaceModel:
    def __init__(
        self,
        model_name,
        max_tokens: int = HUGGINGFACE_MODEL_MAX_TOKENS,
        top_k: int = HUGGINGFACE_MODEL_TOP_K,
        temperature: float = HUGGINGFACE_MODEL_TEMPERATURE,
    ):
        self.model = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": max_tokens,
                "top_k": top_k,
                "temperature": temperature,
            },
        )
