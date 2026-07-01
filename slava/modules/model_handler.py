from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage  # на всякий случай, если будешь что-то сам разруливать
from langchain_core.output_parsers import StrOutputParser


class ModelHandler:

    def __init__(self, model_class=None):
        """
        model_class — это ИНСТАНС модели, а не класс.
        Например:
            ModelHandler(GigaChatModel(...))
            ModelHandler(YandexGPTModel(...))
        """
        self.model_class = model_class
        self.template = PromptTemplate.from_template("{prompt}")

    def generate_response(self, prompt: str) -> str:
        """
        Универсальный вызов модели:
        - LangChain-обёртки с .model (GigaChatModel и др.)
        - кастомные клиенты типа YandexGPTModel с .get_response
        - просто callable-модели
        """
        formatted_prompt = self.template.format(prompt=prompt)

        if hasattr(self.model_class, "model"):
            chain = self.template | self.model_class.model | StrOutputParser()
            response = chain.invoke({"prompt": prompt})
            return response

        if hasattr(self.model_class, "get_response") and callable(self.model_class.get_response):
            return self.model_class.get_response(formatted_prompt)

        raise AttributeError(
            f"ModelHandler не знает, как вызывать объект типа {type(self.model_class)}: "
            f"нет ни .model, ни .get_response, ни __call__."
        )
