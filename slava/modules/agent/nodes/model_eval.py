from typing import Dict, Optional, TypedDict, cast

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from slava.models.ollama import OllamaModel
from slava.modules.data_loader import DataLoader
from slava.modules.model_eval import ModelEval
from slava.modules.model_handler import ModelHandler


class ModelEvalInput(TypedDict):
    dataset_link: str


class ModelEvalOutput(BaseModel):
    success: bool
    data_path: str
    message: Optional[str] = None


def run_model_dataset_evaluation(headers: Optional[Dict[str, str]] = None) -> Runnable[ModelEvalInput, ModelEvalOutput]:
    """
    Создаёт цепочку для выполнения оценки модели на датасете.

    Args:
        headers (Optional[Dict[str, str]]): Дополнительные HTTP-заголовки (если нужны).

    Returns:
        Runnable[ModelEvalInput, ModelEvalOutput]: Цепочка для оценки модели.
    """

    class ModelEvaluationRunnable(Runnable[ModelEvalInput, ModelEvalOutput]):
        def invoke(self, input_data: ModelEvalInput) -> ModelEvalOutput:
            """
            Выполняет логику оценки модели.

            Args:
                input_data (ModelEvalInput): Данные с ссылкой на датасет.

            Returns:
                ModelEvalOutput: Результат оценки.
            """
            # Извлечение параметров
            dataset_link = input_data["dataset_link"]
            topic = input_data["topic"] + ".jsonl"
            model_name = input_data["model_name"]

            # Инициализация компонентов
            data_loader = DataLoader(repo_id=dataset_link, filename=topic)
            model_eval = ModelEval()

            # Загрузка данных с использованием DataLoader
            dataset = data_loader.load_data()

            print(dataset.head())

            # Инициализация модели и обработчика
            model_class = OllamaModel(model_name=model_name)
            model_handler = ModelHandler(model_class)

            # Запуск оценки
            results_filepath = model_eval.run_evaluation(model_name, dataset, model_handler)

            # Возвращаем успешный результат
            return ModelEvalOutput(
                success=True,
                data_path=results_filepath,
                message=f"Оценка модели на датасете '{dataset_link}' успешно завершена.",
            )

    # Оборачиваем в Runnable и приводим к правильному типу
    model_eval_runnable = cast(Runnable[ModelEvalInput, ModelEvalOutput], ModelEvaluationRunnable())
    return model_eval_runnable
