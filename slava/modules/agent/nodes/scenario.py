from typing import Dict, List, Optional, TypedDict, cast

from huggingface_hub import HfApi
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from slava.modules.agent.utils import get_jsonl_files


# Определение входных и выходных типов
class ScenarioInput(TypedDict):
    query: str
    messages: List[BaseMessage]


class ScenarioOutput(BaseModel):
    f"""
    ВЫ — АВТОНОМНЫЙ АГЕНТ AGORA, ОБЛАДАЮЩИЙ ГЛУБОКИМИ ЗНАНИЯМИ О ТЕСТИРОВАНИИ И ОЦЕНКЕ LLM. ВАША ЗАДАЧА — ОТВЕЧАТЬ НА ВОПРОСЫ О ПРИНЦИПАХ РАБОТЫ AGORA, ПРЕДОСТАВЛЯТЬ ПОЛЕЗНЫЕ СОВЕТЫ ПО ИСПОЛЬЗОВАНИЮ ФУНКЦИЙ И УКАЗЫВАТЬ, ЧТО ДОЛЖНО БЫТЬ В СООБЩЕНИИ, ЧТОБЫ ПОЛЬЗОВАТЕЛЬ ПОЛУЧИЛ ТОЧНЫЙ РЕЗУЛЬТАТ.

    ### ИНСТРУКЦИИ ###

    1. ОТВЕЧАЙТЕ ЧЕТКО И ЯСНО НА ВСЕ ВОПРОСЫ О ВОЗМОЖНОСТЯХ AGORA, УКАЗЫВАЯ НА ЕЁ ОСНОВНЫЕ ФУНКЦИИ:
    - ОЦЕНКА С ПОМОЩЬЮ БЕНЧМАРКА.
    - СОЗДАНИЕ КАСТОМНЫХ БЕНЧМАРКОВ.
    - ПОИСК ИНФОРМАЦИИ В ИНТЕРНЕТЕ ДЛЯ АНАЛИЗА.

    2. ЕСЛИ ВО ВХОДНОМ СООБЩЕНИИ ПОЛЬЗОВАТЕЛЯ ЧЕГО-ТО НЕ ХВАТАЕТ ДЛЯ ТОЧНОГО РЕЗУЛЬТАТА:
    - УКАЗЫВАЙТЕ, КАКИЕ ДАННЫЕ НЕОБХОДИМО ДОБАВИТЬ.
    - ОБЪЯСНЯЙТЕ, ПОЧЕМУ ЭТИ ДАННЫЕ НУЖНЫ.

    3. СОБЛЮДАЙТЕ ЛОГИКУ И ПОСЛЕДОВАТЕЛЬНОСТЬ В РАССУЖДЕНИЯХ:
    1. ПРОЧИТАЙТЕ ВХОДНОЕ СООБЩЕНИЕ И ВЫЯСНИТЕ, ЧТО УЖЕ УКАЗАНО.
    2. АНАЛИЗИРУЙТЕ, ЧЕГО НЕ ХВАТАЕТ ДЛЯ ДОСТИЖЕНИЯ ПОСТАВЛЕННОЙ ЦЕЛИ.
    3. ОБЪЯСНИТЕ, КАК ПОЛЬЗОВАТЕЛЬ МОЖЕТ ПРЕДОСТАВИТЬ НЕДОСТАЮЩУЮ ИНФОРМАЦИЮ.

    4. ВО ВСЕХ ВЫХОДНЫХ СООБЩЕНИЯХ ИСПОЛЬЗУЙТЕ ФОРМАТ JSON ДЛЯ ОТЧЁТА О НЕОБХОДИМОЙ ИЛИ ПОЛНОЙ ИНФОРМАЦИИ.

    ### Примеры ###
    Входной текст: "Протестируйте модель qwen2.5:7b-instruct-q4_0" на бенчмарке slava."
    Результат: {{ "model_name": "qwen2.5:7b-instruct-q4_0", "dataset_name": "slava" }}
    САМОЕ ГЛАВНОЕ ВЫБИРАТЬ БЕНЧМАРКИ С ОДИНАКОВЫМ ИМЕНЕМ ИЗ СПИСКА: benches_list
    ЕСЛИ НЕТУ ТО ПИШИ ДРУГОЕ НАЗВАНИЕ И ПРИМЕНЯЙ ТУЛЗУ!!!
    если инфы нет то не используйте тулзу и напишите причину
    ЕСЛИ НЕ НАЗВАНА МОДЕЛЬ ТО НЕ ИСПОЛЬЗУЙ ТУЛЗУ
    """
    model_name: Optional[str] = Field(description="Название модели, если обнаружено с маленой буквы 1 в 1 как написано")
    dataset_name: Optional[str] = Field(description="Название бенчмарка, если обнаружено или то как задал автор")


def create_agora_scenario_chain_vllm(
    headers: Optional[Dict[str, str]] = None, llm_name: str = "qwen2.5:72b-instruct-q4_0"
) -> Runnable[ScenarioInput, ScenarioOutput]:
    """
    Создаёт цепочку для обработки сценариев с использованием LLM.

    Args:
        headers (Optional[Dict[str, str]]): Заголовки для LLM (необязательно).

    Returns:
        Runnable[ScenarioInput, ScenarioOutput]: Обработчик сценариев.
    """
    # Инициализация LLM
    llm = ChatOllama(model=llm_name, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    llm_with_tools = llm.bind_tools([ScenarioOutput])

    class ScenarioRunnable(Runnable[ScenarioInput, ScenarioOutput]):
        def invoke(self, input_data: ScenarioInput) -> ScenarioOutput:
            """
            Обрабатывает входные данные с добавлением истории сообщений.

            Args:
                input_data (ScenarioInput): Входные данные с запросом и историей сообщений.

            Returns:
                ScenarioOutput: Результат обработки.
            """
            benches_list = get_jsonl_files()
            messages_text = "\n".join([msg.content for msg in input_data["messages"]])
            full_text = f"История сообщений:\n{messages_text}\n\nБенчмарки в доступе (ГОВОРИ ПРО НИХ ТОЛЬКО ЕСЛИ СПРОСЯТ про бенчмарки или про данные для проверки!) benches_list:{benches_list} \n\n ЕСЛИ ПРОСЯТ ПРОГНАТЬ МОДЕЛЬ НА БЕНЧМАРКЕ ТО ТОЛЬКО ПОСЛЕ ТОГО КАК В ЗАПРОСЕ И БОЛЕЕ РАННИХ ИСТОРИИ СООБЩЕНИЙ НАЗВАНА МОДЕЛЬ ЯВНО И ТЕМА/БЕНЧМАРК ИНАЧЕ УТОЧНЯЙ\n\nЗапрос:\n{input_data['query']}"

            # Выполнение вызова LLM
            result = llm_with_tools.invoke(full_text)

            # Парсинг результата
            return result

    # Создание экземпляра обработчика
    scenario_chain = ScenarioRunnable()
    return scenario_chain
