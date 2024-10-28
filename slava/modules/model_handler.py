from langchain_core.prompts import PromptTemplate


class ModelHandler:

    def __init__(self, model_class=None):
        self.model_class = model_class

    def generate_response(self, prompt: str) -> str:
        template = PromptTemplate.from_template("{prompt}")
        chain = template | self.model_class.model.bind(skip_prompt=True)
        response = chain.invoke({"prompt": prompt})
        return response


