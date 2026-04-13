"""
Routes confirmed intents to the appropriate SubAgent instance.
"""
from __future__ import annotations
from backend.agents.base_agent import BaseAgent
from backend.agents.country_ranking_agent import CountryRankingAgent
from backend.agents.drug_reimbursement_agent import DrugReimbursementAgent
from backend.agents.enrollment_forecasting_agent import EnrollmentForecastingAgent
from backend.agents.protocol_analysis_agent import ProtocolAnalysisAgent
from backend.agents.site_list_merger_agent import SiteListMatchingAgent
from backend.agents.trial_benchmarking_agent import TrialBenchmarkingAgent, DEFAULT_DATASET
from backend.llm.llm_client import LLMClient
from backend.llm.web_search import WebSearchClient


class Router:
    def __init__(self, llm_client: LLMClient, config: dict = None,
                 web_search: WebSearchClient | None = None):
        citeline_dataset = (
            (config or {}).get("data_sources", {}).get("citeline_dataset", DEFAULT_DATASET)
        )
        self._registry: dict[str, BaseAgent] = {
            "site_list_matching":   SiteListMatchingAgent(llm_client),
            "trial_benchmarking":   TrialBenchmarkingAgent(llm_client, dataset_name=citeline_dataset,
                                                           web_search=web_search),
            "drug_reimbursement":   DrugReimbursementAgent(llm_client, web_search=web_search),
            "enrollment_forecasting": EnrollmentForecastingAgent(llm_client, web_search=web_search),
            "protocol_analysis":    ProtocolAnalysisAgent(llm_client, web_search=web_search),
            "country_ranking":     CountryRankingAgent(llm_client, web_search=web_search),
        }

    def get_agent(self, skill_id: str) -> BaseAgent | None:
        return self._registry.get(skill_id)

    def all_skills(self) -> dict[str, BaseAgent]:
        return dict(self._registry)
