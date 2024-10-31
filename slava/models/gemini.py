import google.generativeai as genai
from langchain_core.messages import HumanMessage

from slava.config import GEMINI_MODEL_NAME, TEXT_COLUMN


class GeminiModel:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = self.Model(genai.GenerativeModel(GEMINI_MODEL_NAME))

    class Model:
        def __init__(self, model):
            self.model = model

        def invoke(self, messages: list[HumanMessage]) -> str:
            try:
                prompt = messages[0].content if messages else ""

                response = self.model.generate_content(prompt)

                if hasattr(response, TEXT_COLUMN):
                    return response.text.strip()
                else:
                    return "The model returned an unexpected response format."

            except Exception:
                return "Error when generating the response by the model."
