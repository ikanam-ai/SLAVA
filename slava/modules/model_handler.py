from langchain_core.messages import HumanMessage


class ModelHandler:
    """A class to handle model loading and response generation."""

    def __init__(self, model_class=None):
        self.model_class = model_class

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the model based on the given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The model's response.

        Raises:
            ValueError: If the model type is unknown.
        """
        response = self.model_class.model.invoke([HumanMessage(content=prompt)])
        return response
