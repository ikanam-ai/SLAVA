from typing import Dict, Optional, TypedDict, cast

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class SummaryInput(TypedDict):
    file_path: str


class SummaryOutput(BaseModel):
    success: bool
    summary: AIMessage
    message: Optional[str] = None


def create_agora_summary_chain_vllm(
    headers: Optional[Dict[str, str]] = None, llm_name: str = "qwen2.5:72b-instruct-q4_0"
) -> Runnable[SummaryInput, SummaryOutput]:
    """
    Создаёт цепочку для генерации аналитического отчёта на основе CSV-файла.

    Args:
        headers (Optional[Dict[str, str]]): Дополнительные HTTP-заголовки (необязательно).

    Returns:
        Runnable[SummaryInput, SummaryOutput]: Цепочка генерации отчёта.
    """

    class SummaryRunnable(Runnable[SummaryInput, SummaryOutput]):
        def invoke(self, input_data: SummaryInput) -> SummaryOutput:
            try:
                df = pd.read_csv(input_data["file_path"])
            except Exception as e:
                return SummaryOutput(success=False, summary_text="", message=f"Ошибка чтения файла: {e}")

            # Вычисление метрик
            total_count = len(df)
            correct_count = df["comparison"].astype(int).sum()
            errors_df = df[df["comparison"] == "0"]
            accuracy_percentage = round((correct_count / total_count) * 100, 2)

            errors_summary = "\n".join(
                [
                    f"- {row['comment']} (response: {row['response']}, outputs: {row['outputs']})"
                    for _, row in errors_df.iterrows()
                ]
            )

            llm = ChatOllama(
                model=llm_name,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

            # Настройка шаблона запроса
            prompt_template = PromptTemplate(
                template="""
                На основе данных сформируй аналитический отчёт:
                1. Укажи общее количество строк, прошедших оценку.
                2. Укажи количество правильных ответов и их процент от общего числа.
                3. Опиши основные причины ошибок, включая комментарии.

                ### Данные ###
                - Всего строк: {total_count}
                - Правильные ответы: {correct_count} ({accuracy_percentage}%)
                - Ошибки:
                {errors_summary}

                Сформируй отчёт в удобочитаемом текстовом формате на русском языке.
                """,
                input_variables=["total_count", "correct_count", "accuracy_percentage", "errors_summary"],
            )

            # Формирование запроса
            query = prompt_template.format(
                total_count=total_count,
                correct_count=correct_count,
                accuracy_percentage=accuracy_percentage,
                errors_summary=errors_summary,
            )

            # Вызов LLM
            try:
                response = llm.invoke(query)
            except Exception as e:
                return SummaryOutput(success=False, summary_text="", message=f"Ошибка генерации отчёта: {e}")

            # Возврат результата
            return SummaryOutput(
                success=True,
                summary=response,
                message="Аналитический отчёт успешно сгенерирован.",
            )

    summary_chain = cast(Runnable[SummaryInput, SummaryOutput], SummaryRunnable())
    return summary_chain
