from typing import Dict, List, Optional, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

contextualize_system_text = """\
Учитывая историю чата и последние вопросы пользователей, которые могут ссылаться на контекст в истории чата, \
сформулируй отдельный запрос, который можно понять без истории чата. \
Не отвечай на вопрос, просто переформулируй его. Не меняй запрос, если он не ссылается на историю чата. Пиши на русском языке.\
"""
human_contextualize_prompt_template = PromptTemplate.from_template("Запрос: {input}\nПереформулированный вопрос:")

contextualize_chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=contextualize_system_text),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate(prompt=human_contextualize_prompt_template),
    ]
)


class ContextualizeInput(TypedDict):
    input: str
    chat_history: List[BaseMessage]


def create_agora_contextualize_chain_ollama(
    headers: Optional[Dict[str, str]] = None, llm_name: str = "qwen2.5:72b-instruct-q4_0"
) -> Runnable[ContextualizeInput, AIMessage]:
    llm = ChatOllama(model=llm_name, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})

    contextualize_chain = contextualize_chat_prompt_template | llm
    contextualize_chain = cast(Runnable[ContextualizeInput, AIMessage], contextualize_chain)
    return contextualize_chain
