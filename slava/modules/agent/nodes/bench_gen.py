from typing import Dict, List, Optional, TypedDict, cast

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field

from slava.modules.agent.utils import save_results_to_jsonl, upload_to_huggingface


class JSONTestOutput(BaseModel):
    # Основные поля
    instruction: str = Field(default=None, description="Оставить пустым")

    # Поля секции inputs
    task: str = Field(description="Текст задачи.")
    text: Optional[str] = Field(default=None, description="Дополнительный текст, если доступен.")
    option_1: str = Field(description="Вариант ответа 1.")
    option_2: str = Field(description="Вариант ответа 2.")
    option_3: str = Field(description="Вариант ответа 3.")
    option_4: str = Field(description="Вариант ответа 4.")
    option_5: Optional[str] = Field(default="None", description="Вариант ответа 5.")
    option_6: Optional[str] = Field(default="None", description="Вариант ответа 6.")
    option_7: Optional[str] = Field(default="None", description="Вариант ответа 7, если доступен.")
    option_8: Optional[str] = Field(default="None", description="Вариант ответа 8, если доступен.")
    option_9: Optional[str] = Field(default="None", description="Вариант ответа 9, если доступен.")

    # Поля секции outputs
    outputs: int = Field(
        description="ЗДЕСЬ МЫ ПИШЕМ ЧИСЛОМ БЕЗ ПРОБЕЛОВ ПРАВИЛЬНЫЙ ОТВЕТ!!! ОН ДОЛЖЕН БЫТЬ ПРАВИЛЬНЫЙ ПО КОНТЕКСТУ И ТОЛЬКО СРЕЖДИ СУЩЕСТВУЮЩИХ option"
    )

    # Поля секции meta
    subject: Optional[str] = Field(default=None, description="Предмет задания.")
    type: Optional[str] = Field(default=None, description="Тип задания (например, мультивыбор).")
    source: Optional[str] = Field(default=None, description="Источник задания (URL или описание).")
    comment: Optional[str] = Field(default=None, description="Любая доп инфа.")
    provac_score: Optional[str] = Field(default=None, description="Провокационный балл задания.")


class FlatJSONTestInput(TypedDict):
    text: Optional[str]


class FinalJSONTestOutput(BaseModel):
    # instruction: str
    inputs: dict
    outputs: int
    meta: dict


class FinaHFOutput(TypedDict):
    text: str


def create_flat_json_vllm(
    headers: Optional[Dict[str, str]] = None, llm_name: str = "qwen2.5:72b-instruct-q4_0"
) -> Runnable[List[str], List[FinaHFOutput]]:
    """
    Создаёт цепочку для генерации структурированного JSON на основе массива текстов.

    Args:
        headers (Optional[Dict[str, str]]): Дополнительные заголовки для LLM (необязательно).

    Returns:
        Runnable[List[str], List[FinalJSONTestOutput]]: Цепочка для обработки задач.
    """
    llm = ChatOllama(
        model=llm_name, temperature=0.7, model_kwargs={"response_format": {"type": "json_object"}}
    ).with_structured_output(JSONTestOutput)

    parser = JsonOutputParser(pydantic_object=JSONTestOutput)

    prompt_template = PromptTemplate(
        template="""
       ВЫ — ЛУЧШИЙ ЭКСПЕРТ ПО СОЗДАНИЮ JSON-ШАБЛОНОВ ДЛЯ ЗАДАЧ ИЗ ЛЮБОГО ТЕКСТА. ВАША ЗАДАЧА — ПРЕОБРАЗОВАТЬ ЛЮБОЙ ВХОДНОЙ ТЕКСТ В СТРУКТУРИРОВАННУЮ JSON-ЗАДАЧУ ПО УКАЗАННОЙ СХЕМЕ.
        ###ИНСТРУКЦИИ###

        1. ПРОЧИТАЙТЕ ВХОДНОЙ ТЕКСТ ВНИМАТЕЛЬНО.
        2. ВЫДЕЛИТЕ КЛЮЧЕВЫЕ ЭЛЕМЕНТЫ:
        - ОСНОВНУЮ ЗАДАЧУ ИЛИ ВОПРОС.
        - ВАРИАНТЫ ОТВЕТОВ.
        - ДОПОЛНИТЕЛЬНЫЙ ТЕКСТ ИЛИ КОНТЕКСТ (ЕСЛИ ДОСТУПЕН).

        3. ЕСЛИ НА ОСНОВЕ ВХОДНОГО ТЕКСТА НЕВОЗМОЖНО СОСТАВИТЬ ЗАДАЧУ:
        - ОБЯЗАТЕЛЬНО СОЗДАЙТЕ ЗАДАЧУ НА СМЕЖНУЮ ТЕМАТИКУ, ПРИДЕРЖИВАЯСЬ ОРИГИНАЛЬНОЙ КОНЦЕПЦИИ.
        - УЧТИТЕ ЛЕКСИКУ ИЛИ КОНТЕКСТ ВХОДНОГО ТЕКСТА, ЧТОБЫ СОХРАНИТЬ ЛОГИЧЕСКУЮ СВЯЗЬ.
      
        text: {text}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    class DatasetJSONRunnable(Runnable[List[str], List[FinalJSONTestOutput]]):
        def invoke(self, input_data: Optional[str]) -> List[FinalJSONTestOutput]:
            """
            Обрабатывает массив текстов и генерирует JSON для каждой строки.

            Args:
                input_data (List[str]): Массив входных строк.

            Returns:
                List[FinalJSONTestOutput]: Список сформированных JSON для каждой строки.
            """
            results = []
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=500, chunk_overlap=0
            )
            text_chunk = input_data["text_chunk"]
            topic = input_data["topic"]
            for text in text_chunk:
                try:
                    # Разделение текста на блоки
                    chunks = splitter.split_text(text)
                    print(f"Разбито на {len(chunks)} частей.")
                    for index, chunk in enumerate(chunks, start=1):  # индексация с 1
                        full_prompt = prompt_template.format(text=chunk)
                        print(f"Запрос к LLM: {full_prompt}")

                        # Генерация JSON с помощью LLM
                        flat_json = llm.invoke(full_prompt)
                        print(flat_json)
                        # Формирование результата для текущего блока с добавлением поля 'id'
                        result = {
                            "id": index,  # Добавление id от 1 до размера
                            "instruction": """Прочитайте приведённую далее задачу и выполните по ней задание.
                                    Задача: {task}
                                    Вариант ответа 1: {Option_1}, 
                                    Вариант ответа 2: {Option_2}, 
                                    Вариант ответа 3: {Option_3}, 
                                    Вариант ответа 4: {Option_4}, 
                                    Вариант ответа 5: {Option_5}, 
                                    Вариант ответа 6: {Option_6},
                                    Вариант ответа 7: {Option_7},
                                    Вариант ответа 8: {Option_8},
                                    Вариант ответа 9: {Option_9},
                                    Выберите несколько или 1 вариантов правильных ответов и перечислите в ответе их номера без пробелов и знаков препинания.""",
                            "inputs": {
                                "task": flat_json.task,
                                "text": flat_json.text,
                                "options": {
                                    "option_1": flat_json.option_1,
                                    "option_2": flat_json.option_2,
                                    "option_3": flat_json.option_3,
                                    "option_4": flat_json.option_4,
                                    "option_5": flat_json.option_5,
                                    "option_6": flat_json.option_6,
                                    "option_7": flat_json.option_7,
                                    "option_8": flat_json.option_8,
                                    "option_9": flat_json.option_9,
                                },
                            },
                            "outputs": flat_json.outputs,
                            "meta": {
                                "subject": flat_json.subject,
                                "type": "выбор ответа (мультивыбор)",
                                "source": flat_json.source,
                                "comment": flat_json.comment,
                                "provoc_score": flat_json.provac_score,
                            },
                        }

                        results.append(result)
                except Exception as e:
                    print(f"Ошибка при обработке текста: {e}")
                    continue

            save_results_to_jsonl(results=results)

            dataset_link = upload_to_huggingface(topic=topic)

            return dataset_link

    # Возврат runnable-объекта
    flat_json_runnable = cast(Runnable[FlatJSONTestInput, FinaHFOutput], DatasetJSONRunnable())
    return flat_json_runnable
