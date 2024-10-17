from langchain_core.messages import HumanMessage


class ModelHandler:

    def __init__(self, model_class=None):
        self.model_class = model_class

    def generate_response(self, prompt: str) -> str:
        response = self.model_class.model.invoke([HumanMessage(content=prompt)])
        return response
