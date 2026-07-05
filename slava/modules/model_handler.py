import re
from typing import Any

from langchain.prompts import PromptTemplate


class ModelHandler:
    def __init__(self, model_class=None):
        """
        model_class is an INSTANCE of a model wrapper.
        Examples:
            ModelHandler(GigaChatModel(...))
            ModelHandler(YandexGPTModel(...))
            ModelHandler(OllamaModel(...))
        """
        self.model_class = model_class
        self.template = PromptTemplate.from_template("{prompt}")

    @staticmethod
    def _to_text(response: Any) -> str:
        if response is None:
            return ""

        # LangChain AIMessage / BaseMessage.
        if hasattr(response, "content"):
            response = response.content

        # Some SDKs return list of blocks.
        if isinstance(response, list):
            chunks = []
            for item in response:
                if hasattr(item, "text"):
                    chunks.append(str(item.text))
                elif isinstance(item, dict):
                    chunks.append(str(item.get("text") or item.get("content") or ""))
                else:
                    chunks.append(str(item))
            response = "".join(chunks)

        text = str(response).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(
            r"^\s*Thinking\.\.\..*?\.\.\.done thinking\.\s*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        text = re.sub(r"^\s*(ответ|final answer|answer)\s*[:：\-–—]\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.replace("**", "").replace("__", "").strip()
        return text

    def generate_response(self, prompt: str) -> str:
        """Universal model call with explicit empty-response errors.

        Priority:
        1. Custom wrappers with get_response(), especially OllamaModel and YandexGPTModel.
        2. LangChain-compatible wrappers with .model.invoke().
        3. Callable objects.
        """
        if self.model_class is None:
            raise AttributeError("ModelHandler got model_class=None")

        formatted_prompt = self.template.format(prompt=prompt)

        # Important: check get_response BEFORE .model.
        # OllamaModel used to expose .model via LangChain, and that path could silently produce None/NaN.
        if hasattr(self.model_class, "get_response") and callable(self.model_class.get_response):
            response = self.model_class.get_response(formatted_prompt)
            text = self._to_text(response)
            if not text:
                raise ValueError(f"Empty response from {type(self.model_class)}.get_response")
            return text

        if hasattr(self.model_class, "model"):
            model = self.model_class.model
            if hasattr(model, "invoke") and callable(model.invoke):
                response = model.invoke(formatted_prompt)
                text = self._to_text(response)
                if not text:
                    raise ValueError(f"Empty response from {type(model)}.invoke")
                return text
            if callable(model):
                response = model(formatted_prompt)
                text = self._to_text(response)
                if not text:
                    raise ValueError(f"Empty response from callable {type(model)}")
                return text

        if callable(self.model_class):
            response = self.model_class(formatted_prompt)
            text = self._to_text(response)
            if not text:
                raise ValueError(f"Empty response from callable {type(self.model_class)}")
            return text

        raise AttributeError(
            f"ModelHandler does not know how to call object of type {type(self.model_class)}: "
            "no get_response(), no .model.invoke(), and not callable."
        )
