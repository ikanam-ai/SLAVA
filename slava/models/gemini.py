import google.generativeai as genai
from langchain_core.messages import HumanMessage

class GeminiModel:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = self.Model(genai.GenerativeModel("gemini-1.5-flash"))

    class Model:
        def __init__(self, model):
            self.model = model

        def invoke(self, messages: list[HumanMessage]) -> str:
            try:
                prompt = messages[0].content if messages else ""

                response = self.model.generate_content(prompt)

                if hasattr(response, 'text'):
                    return response.text.strip()
                else:
                    return "Модель вернула неожиданный формат ответа."

            except Exception:
                return "Ошибка при генерации ответа."