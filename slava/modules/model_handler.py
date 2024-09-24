from langchain.chat_models import GigaChat
from langchain.llms import OpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFacePipeline
from langchain_ollama.llms import OllamaLLM


class ModelHandler:
    """A class to handle model loading and response generation."""

    def __init__(
        self, model_name: str, model_type: str = "huggingface", api_key: str = None
    ):
        """
        Initializes the ModelHandler.

        Args:
            model_name (str): The name of the model to load.
            model_type (str): The type of model (e.g., "huggingface", "ollama", "giga", "openai").
            api_key (str): The API key for accessing certain models, if required.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.model = None

    def load_model(self):
        """Loads the specified model based on the model type."""
        if self.model_type == "huggingface":
            self.model = HuggingFacePipeline.from_model_id(
                model_id=self.model_name,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 100,
                    "top_k": 50,
                    "temperature": 0.1,
                },
            )
        elif self.model_type == "ollama":
            self.model = OllamaLLM(
                model=self.model_name,
                num_ctx=1000,
                temperature=1.0,
                top_k=1,
                max_tokens=25,
            )
        elif self.model_type == "giga":
            self.model = GigaChat(
                credentials=self.api_key,
                scope="GIGACHAT_API_PERS",
                model=self.model_name,
            )
        elif self.model_type == "openai":
            self.model = OpenAI(
                api_key=self.api_key,
                temperature=0.7,
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the model based on the given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The model's response.

        Raises:
            ValueError: If the model type is unknown.
        """
        if self.model_type in ["huggingface", "ollama", "giga", "openai"]:
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            return response
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")


# Usage example (uncomment below to use)
# if __name__ == "__main__":
#     # Load model
#     model_handler = ModelHandler(
#         model_name="gpt-3.5-turbo",
#         model_type="openai",
#         api_key=os.getenv("OPENAI_API_KEY"),
#     )
#     model_handler.load_model()

#     # Generate response
#     prompt = "Расскажи, как работает квантовая физика простыми словами."
#     response = model_handler.generate_response(prompt)
#     print(response)
