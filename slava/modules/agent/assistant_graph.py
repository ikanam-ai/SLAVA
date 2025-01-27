import logging
import os
import sys
from typing import Annotated, List, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import Messages, add_messages

from slava.config import repo_id
from slava.modules.agent.runnables import AGORAARunnablesOllama
from slava.modules.agent.utils import get_jsonl_files

logger = logging.getLogger("AGORA Assistant")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_sh)


def concat_up_to_10(left: Messages, right: Messages) -> Messages:
    return add_messages(left=left, right=right)[-10:]


class State(TypedDict):
    messages: Annotated[List[BaseMessage], concat_up_to_10]
    search_result: List[str]
    query: str
    estimation_result: str
    scenario_result: str
    output_path: str
    evaluation_result: str
    summary_result: str
    final_output: str
    topic: str
    bench_result: str


class AGORAAssistant:
    def __init__(
        self,
        agora_runnables: AGORAARunnablesOllama,
        checkpointer: MongoDBSaver,
    ) -> None:
        self.contextualize_chain = agora_runnables.contextualize_chain
        self.estimation_chain = agora_runnables.estimation_chain
        self.scenario_chain = agora_runnables.scenario_chain
        self.model_eval = agora_runnables.model_eval
        self.summary = agora_runnables.summary
        self.search = agora_runnables.search
        self.bench_gen = agora_runnables.bench_gen
        self.pars = agora_runnables.pars

        graph_builder = StateGraph(State)
        graph_builder.add_node("start_new_dialog", self._start_new_dialog)
        graph_builder.add_node("scenario", self._scenario)
        graph_builder.add_node("estimation", self._estimation)
        graph_builder.add_node("model_eval", self._model_eval)
        graph_builder.add_node("model_summary", self._model_summary)
        graph_builder.add_node("search", self._search)
        graph_builder.add_node("bench_gen", self._bench_gen)
        graph_builder.add_node("pars_pdf", self._parse_pdf)

        graph_builder.add_conditional_edges(
            START,
            self._should_continue_dialog,
            {"start_new_dialog": "start_new_dialog", "continue": "scenario", "bench": "pars_pdf"},
        )
        graph_builder.add_conditional_edges(
            "scenario",
            self._should_continue_tools,
            {"end": END, "eval": "model_eval", "gen": "search"},
        )

        graph_builder.add_edge("pars_pdf", "bench_gen")
        graph_builder.add_edge("search", "bench_gen")
        graph_builder.add_edge("bench_gen", "model_eval")
        graph_builder.add_edge("model_eval", "estimation")
        graph_builder.add_edge("estimation", "model_summary")
        graph_builder.add_edge("model_summary", END)
        graph_builder.add_edge("start_new_dialog", END)

        self.graph = graph_builder.compile(checkpointer=checkpointer)

    def _start_new_dialog(self, state: State) -> State:
        messages = state["messages"]
        return cast(
            State,
            {"messages": [RemoveMessage(id=m.id) for m in messages], "final_output": "Начинаем новый диалог!"},
        )

    def _should_continue_dialog(self, state: State) -> Literal["start_new_dialog", "continue"]:
        query = state["query"]
        if "начать новый диалог" == query:
            return "start_new_dialog"
        else:
            if query.endswith(".pdf") and os.path.exists(query):
                logger.info(f"Запрос распознан как путь к локальному PDF-файлу: {query}")
                return "bench"
            else:
                return "continue"

    def _should_continue_tools(self, state):
        messages_scenario = state["scenario_result"]
        logger.info(f"last_message: {messages_scenario}")
        if not messages_scenario.tool_calls:
            return "end"
        else:
            benches_list = get_jsonl_files()
            if messages_scenario.tool_calls[0]["args"]["dataset_name"] in benches_list:
                return "eval"
            else:
                return "gen"

    def _contextualize(self, state: State, config: RunnableConfig) -> State:
        thread_id = config["metadata"]["thread_id"]
        query = state["query"]
        messages = state["messages"]
        if not messages:
            return cast(State, {"standalone_question": query})
        else:
            logger.info(f"Thread ID: {thread_id}. Start contextualize LLM call.")
            contextualize_message = self.contextualize_chain.invoke(
                {
                    "chat_history": messages,
                    "input": query,
                }
            )
            logger.info(
                f"Thread ID: {thread_id}. End contextualize LLM call. "
                f"Response metadata: {contextualize_message.response_metadata} "
                f"Contextualized query: {contextualize_message.content}"
            )
            return cast(State, {"standalone_question": contextualize_message.content})

    def _parse_pdf(self, state: State, config: RunnableConfig) -> State:
        thread_id = config["metadata"]["thread_id"]
        pdf_link = state["query"]
        logger.info(f"Thread ID: {thread_id}: {pdf_link}. Start PDF parsing chain.")
        pdf_parsing_result = self.pars.invoke({"pdf_link": pdf_link})
        logger.info(f"Thread ID: {thread_id}. End PDF parsing chain. " f"PDF parsing result: {pdf_parsing_result}")

        if pdf_parsing_result.success:
            final_output = pdf_parsing_result.parsed_text
            message = "PDF parsing completed successfully."
        else:
            final_output = []
            message = f"PDF parsing failed: {pdf_parsing_result.message}"

        return cast(
            State,
            {
                "topic": pdf_link.replace(" ", "_").replace(".pdf", ""),
                "search_result": final_output,
            },
        )

    def _scenario(self, state: State, config: RunnableConfig) -> State:
        thread_id = config["metadata"]["thread_id"]
        query = state["query"]
        messages = state["messages"]
        logger.info(f"Thread ID: {thread_id}: {query}. Start scenario analysis LLM call. messages: {messages}")
        scenario_result = self.scenario_chain.invoke({"query": query, "messages": messages})
        logger.info(f"Thread ID: {thread_id}. End scenario analysis LLM call. " f"Scenario result: {scenario_result}")

        return cast(
            State,
            {
                "final_output": scenario_result.content,
                "scenario_result": scenario_result,
                "messages": [HumanMessage(content=query), scenario_result.content],
            },
        )

    def _search(self, state: State, config: RunnableConfig) -> State:
        """
        Выполняет поиск с использованием цепочки поиска, созданной в `create_search_chain_vllm`.

        Args:
            state (State): Состояние, содержащее запрос и другие данные.
            config (RunnableConfig): Конфигурация выполнения, содержащая метаданные.

        Returns:
            State: Обновлённое состояние с результатами поиска.
        """
        thread_id = config["metadata"]["thread_id"]
        query = state["query"]
        logger.info(f"Thread ID: {thread_id}: {query}. Start search chain call.")

        search_result = self.search.invoke({"query": query})

        return cast(
            State,
            {
                "topic": search_result.topic,
                "search_result": search_result.retrieved_texts,
            },
        )

    def _bench_gen(self, state: State, config: RunnableConfig) -> State:
        thread_id = config["metadata"]["thread_id"]
        text_chunk = state["search_result"]
        topic = state["topic"]
        bench_result = self.bench_gen.invoke({"text_chunk": text_chunk, "topic": topic})
        logger.info(f"Thread ID: {thread_id}. End benchmark generation LLM call. " f"Benchmark result: {bench_result}")

        return cast(
            State,
            {"final_output": bench_result, "bench_result": bench_result},
        )

    def _model_eval(self, state: State, config: RunnableConfig) -> State:
        """
        Выполняет оценку модели на заданном наборе данных.

        Args:
            state (State): Состояние, содержащее имя модели и датасета.
            config (RunnableConfig): Конфигурация выполнения.

        Returns:
            State: Состояние с результатами оценки модели.
        """
        thread_id = config["metadata"]["thread_id"]
        logger.info(f"State: scenario_result {state['scenario_result']} ")
        dataset_link = repo_id
        model_name = state["scenario_result"].tool_calls[0]["args"]["model_name"]
        topic = state["scenario_result"].tool_calls[0]["args"]["dataset_name"]

        logger.info(f"Thread ID: {thread_id}. Start model evaluation. Model: {model_name}")
        evaluation_result = self.model_eval.invoke(
            {"dataset_link": dataset_link, "topic": topic, "model_name": model_name}
        )
        logger.info(f"Thread ID: {thread_id}. Model evaluation completed. Result: {evaluation_result}")

        return cast(
            State,
            {"evaluation_result": evaluation_result, "output_path": evaluation_result.data_path},
        )

    def _estimation(self, state: State, config: RunnableConfig) -> State:
        """
        Выполняет оценку данных с использованием LLM.

        Args:
            state (State): Текущее состояние с датасетом.
            config (RunnableConfig): Конфигурация выполнения.

        Returns:
            State: Обновлённое состояние с результатами оценки.
        """
        thread_id = config["metadata"]["thread_id"]
        output_path = state["output_path"]
        logger.info(f"Thread ID: {thread_id}. output_path: {output_path} Start dataset estimation process.")
        estimation_result = self.estimation_chain.invoke({"output_path": output_path})
        logger.info(f"Thread ID: {thread_id}. Estimation result: {estimation_result}")

        return cast(
            State,
            {"estimation_result": f"Dataset processed and saved to {output_path}"},
        )

    def _model_summary(self, state: State, config: RunnableConfig) -> State:
        """
        Генерирует аналитический отчёт на основе результатов оценки модели.

        Args:
            state (State): Состояние, содержащее путь до CSV-файла с результатами.
            config (RunnableConfig): Конфигурация выполнения.

        Returns:
            State: Обновлённое состояние с аналитическим отчётом.
        """
        thread_id = config["metadata"]["thread_id"]
        output_path = state["evaluation_result"].data_path
        query = state["query"]
        logger.info(f"Thread ID: {thread_id}. Start generating summary for file: {output_path}")

        try:
            summary_result = self.summary.invoke({"file_path": output_path})
        except Exception as e:
            logger.error(f"Thread ID: {thread_id}. Error during summary generation: {e}")
            raise e

        if summary_result.success:
            logger.info(f"Thread ID: {thread_id}. Summary generated successfully.")
        else:
            logger.error(f"Thread ID: {thread_id}. Summary generation failed: {summary_result.message}")

        # Возвращаем обновлённое состояние
        return cast(
            State,
            {
                "final_output": summary_result.summary.content,
                "output_path": output_path,
                "messages": [HumanMessage(content=query), summary_result.summary],
            },
        )
