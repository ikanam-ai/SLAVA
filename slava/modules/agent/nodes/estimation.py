from typing import Dict, Optional, TypedDict, cast

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class DatasetEstimationInput(TypedDict):
    output_path: str


class DatasetEstimationOutput(BaseModel):
    success: bool
    data_path: str
    message: Optional[str] = None


class EstimationResult(BaseModel):
    digit: str = Field(description="0 или 1 — результат сравнения строк")
    comment: str = Field(description="Комментарий к результату сравнения")


def create_dataset_estimation_chain_vllm(
    headers: Optional[Dict[str, str]] = None, llm_name: str = "qwen2.5:72b-instruct-q4_0"
) -> Runnable[DatasetEstimationInput, DatasetEstimationOutput]:
    """
        Создаёт цепочку для оценки строк в колонках Response и Outputs датасета с использованием LLM.

        Args:
            headers (Optional[Dict[str, str]]): Дополнительные заголовки для LLM (необязательно).
    x
        Returns:
            Runnable[DatasetEstimationInput, DatasetEstimationOutput]: Цепочка обработки датасета.
    """

    class DatasetEstimationRunnable(Runnable[DatasetEstimationInput, DatasetEstimationOutput]):
        def invoke(self, input_data: DatasetEstimationInput) -> DatasetEstimationOutput:
            """
            Выполняет оценку строк в датасете и сохраняет результат.

            Args:
                input_data (DatasetEstimationInput): Входной датасет и путь для сохранения результата.

            Returns:
                DatasetEstimationOutput: Результат выполнения с путём до сохранённого файла.
            """
            # Проверка и подготовка датасета
            output_path = input_data["output_path"]
            dataset = pd.read_csv(output_path)

            if "response" not in dataset.columns or "outputs" not in dataset.columns:
                raise ValueError("В датасете отсутствуют колонки 'Response' и/или 'Outputs'.")

            # Настройка LLM
            llm = ChatOllama(
                model=llm_name, temperature=0, model_kwargs={"response_format": {"type": "json_object"}}
            ).with_structured_output(EstimationResult)

            parser = JsonOutputParser(pydantic_object=EstimationResult)

            prompt_template = PromptTemplate(
                template="""
                Ваша задача — проанализировать эквивалентность двух строк:
                1. Строки считаются эквивалентными, если они содержат одинаковые цифры в одинаковом порядке или выражают одинаковый смысл.
                2. Если строки различаются по смыслу или цифрам, они не эквивалентны.
                
                ### Формат вывода ###
                {format_instructions}
                
                ### Строки для анализа ###
                - Первая строка: {response}
                - Вторая строка: {output}
                """,
                input_variables=["response", "output"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            # Функция для оценки пары строк
            def evaluate_pair(response: str, output: str) -> EstimationResult:
                query = prompt_template.format(response=response, output=output)
                result = llm.invoke(query)  # parser
                return result

            # Оценка строк в датасете
            results = []
            for _, row in dataset.iterrows():
                response = row["response"]
                output = row["outputs"]
                result = evaluate_pair(response, output)
                results.append({"digit": result.digit, "comment": result.comment})

            # Добавление новой колонки с результатами
            results_df = pd.DataFrame(results)
            dataset["comparison"] = results_df["digit"]
            dataset["Comment"] = results_df["comment"]

            # Сохранение обновлённого датасета
            dataset.to_csv(output_path, index=False)

            return DatasetEstimationOutput(
                success=True,
                data_path=output_path,
                message=f"Датасет успешно обработан и сохранён в '{output_path}'.",
            )

    # Создаём runnable и возвращаем
    dataset_estimation_runnable = cast(
        Runnable[DatasetEstimationInput, DatasetEstimationOutput], DatasetEstimationRunnable()
    )
    return dataset_estimation_runnable
