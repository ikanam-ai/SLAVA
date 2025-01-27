from dataclasses import dataclass
from typing import Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from slava.modules.agent.nodes.bench_gen import FlatJSONTestInput, create_flat_json_vllm
from slava.modules.agent.nodes.contextualize_chain import ContextualizeInput, create_agora_contextualize_chain_ollama
from slava.modules.agent.nodes.estimation import DatasetEstimationInput, create_dataset_estimation_chain_vllm
from slava.modules.agent.nodes.model_eval import ModelEvalInput, run_model_dataset_evaluation
from slava.modules.agent.nodes.parse_pdf import PdfParseInput, run_pdf_parsing_chain
from slava.modules.agent.nodes.scenario import ScenarioInput, create_agora_scenario_chain_vllm
from slava.modules.agent.nodes.search import SearchInput, create_search_chain_vllm
from slava.modules.agent.nodes.summary import SummaryInput, create_agora_summary_chain_vllm


@dataclass
class AGORAARunnablesOllama:
    contextualize_chain: Runnable[ContextualizeInput, AIMessage]
    estimation_chain: Runnable[DatasetEstimationInput, AIMessage]
    scenario_chain: Runnable[ScenarioInput, AIMessage]
    model_eval: Runnable[ModelEvalInput, AIMessage]
    summary: Runnable[SummaryInput, AIMessage]
    search: Runnable[SearchInput, AIMessage]
    bench_gen: Runnable[FlatJSONTestInput, AIMessage]
    pars: Runnable[PdfParseInput, AIMessage]


def create_agora_runnables_ollama(
    llm_name: str,
    headers: Optional[Dict[str, str]] = None,
) -> AGORAARunnablesOllama:
    contextualize_chain = create_agora_contextualize_chain_ollama(llm_name=llm_name, headers=headers)
    estimation_chain = create_dataset_estimation_chain_vllm(llm_name=llm_name, headers=headers)
    scenario_chain = create_agora_scenario_chain_vllm(llm_name=llm_name, headers=headers)
    summary = create_agora_summary_chain_vllm(llm_name=llm_name, headers=headers)
    search = create_search_chain_vllm(llm_name=llm_name, headers=headers)
    bench_gen = create_flat_json_vllm(llm_name=llm_name, headers=headers)
    model_eval = run_model_dataset_evaluation(headers=headers)
    pars = run_pdf_parsing_chain(headers=headers)

    return AGORAARunnablesOllama(
        contextualize_chain=contextualize_chain,
        estimation_chain=estimation_chain,
        scenario_chain=scenario_chain,
        model_eval=model_eval,
        summary=summary,
        search=search,
        bench_gen=bench_gen,
        pars=pars,
    )
