from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel
from pymongo import MongoClient
from requests.auth import _basic_auth_str

from slava.config import MAX_LEN_USER_PROMPT
from slava.modules.agent.assistant_graph import AGORAAssistant
from slava.modules.agent.base import BaseHandler
from slava.modules.agent.runnables import create_agora_runnables_ollama
from slava.modules.agent.safe import LengthLimitProtector, ProtectionResult, ProtectionStatus, ProtectorAccumulator


class AGORAOptions(BaseModel):
    model_name: str
    psycopg_checkpointer: str


class AGORAHandler(BaseHandler):

    def __init__(self, options: AGORAOptions) -> None:
        self._agora_runnables = create_agora_runnables_ollama(
            llm_name=options.model_name,
            headers={"Authorization": _basic_auth_str("admin", "password")},
        )
        self._checkpointer_db_uri = options.psycopg_checkpointer
        self._protector = ProtectorAccumulator(protectors=[LengthLimitProtector(max_len=MAX_LEN_USER_PROMPT)])

    async def ahandle_prompt(self, prompt: str, chat_id: str) -> str:
        protector_res = self._protector.validate(prompt)
        if protector_res.status is not ProtectionStatus.ok:
            return protector_res.message

        input = {"query": prompt}
        config = {"configurable": {"thread_id": chat_id}}
        # async with await AsyncConnection.connect(self._checkpointer_db_uri, **POSTGRES_CONNECTION_KWARGS) as conn:
        mongodb_client = MongoClient(self._checkpointer_db_uri)
        checkpointer = MongoDBSaver(mongodb_client)
        assistant = AGORAAssistant(
            agora_runnables=self._agora_runnables,
            checkpointer=checkpointer,
        )
        # output = assistant.graph.invoke(input=input, config=config)

        output = assistant.graph.invoke(input, config=config)
        # except UndefinedTable:
        #     await checkpointer.setup()
        #     output = await assistant.graph.ainvoke(input=input, config=config)
        answer = output["final_output"]
        value = output["final_output"]
        return answer, value
