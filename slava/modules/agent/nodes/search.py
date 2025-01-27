from typing import Dict, List, Optional, TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class SearchInput(TypedDict):
    query: str


class SearchOutput(BaseModel):
    """
    Результат цепочки:
    - Возвращает тему, наиболее подходящую под запросы
    !!!ВСЕГДА ИСПОЛЬЗУЙ ЭТУ ТУЛЗУ!!!
    """

    topic: Optional[str] = Field(None, description="Тема запроса. ОДНО СЛОВА ЛАТИНИЦЕЙ ИЛИ НА АНГЛИЙСКОМ!!")
    retrieved_texts: Optional[List[str]] = Field(None, description="Список извлечённых текстов.")


def create_search_chain_vllm(
    headers: Optional[Dict[str, str]] = None, llm_name: str = "qwen2.5:72b-instruct-q4_0"
) -> Runnable[SearchInput, SearchOutput]:
    """
    Создаёт цепочку поиска, состоящую из:
    1. Анализа запроса с использованием LLM.
    2. Поиска в интернете по теме (если найдена).

    Args:
        headers (Optional[Dict[str, str]]): Заголовки для LLM (необязательно).

    Returns:
        Runnable[SearchInput, SearchOutput]: Цепочка поиска.
    """
    llm = ChatOllama(
        model=llm_name,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    llm_with_tools = llm.bind_tools([SearchOutput])
    parser = JsonOutputParser(pydantic_object=SearchOutput)

    class SearchRunnable(Runnable[SearchInput, SearchOutput]):
        def invoke(self, input_data: SearchInput) -> SearchOutput:
            """
            Выполняет анализ запроса, поиск темы и сбор информации из интернета.

            Args:
                input_data (SearchInput): Входные данные с запросом.

            Returns:
                SearchOutput: Результат цепочки.
            """
            prompt = PromptTemplate(
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
                template=(
                    "Вы — автономный агент поиска. Ваша задача: найти точную тему для запроса или "
                    "указать, что следует уточнить в запросе, чтобы найти нужную информацию.\n"
                    "НЕ ТРЕБУЙ УТОЧНЕНИЙ И НЕ ВЕДИ НИКАКОЙ ДИАЛОГ ТОЛЬКО ТЕМА."
                    "Ответ должен быть в формате JSON с полями 'topic'."
                    "Запрос: {query}"
                ),
            )
            full_prompt = prompt.format(query=input_data["query"])
            llm_result = llm_with_tools.invoke(full_prompt)

            llm_result = llm_result.tool_calls[0]["args"]
            search = DuckDuckGoSearchResults(output_format="list")

            search_results = search.invoke(llm_result["topic"])
            extracted_texts = []
            for result in search_results[:5]:
                extracted_texts.append(result["snippet"])
            return SearchOutput(
                topic=llm_result["topic"].strip(),
                retrieved_texts=extracted_texts,
            )

    return SearchRunnable()
