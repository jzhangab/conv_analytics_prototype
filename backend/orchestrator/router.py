"""
Routes confirmed intents to the appropriate SubAgent instance.
"""
from __future__ import annotations
from backend.agents.base_agent import BaseAgent
from backend.agents.drug_reimbursement_agent import DrugReimbursementAgent
from backend.agents.enrollment_forecasting_agent import EnrollmentForecastingAgent
from backend.agents.site_list_merger_agent import SiteListMergerAgent
from backend.agents.trial_benchmarking_agent import TrialBenchmarkingAgent
from backend.llm.llm_client import LLMClient


class Router:
    def __init__(self, llm_client: LLMClient):
        self._registry: dict[str, BaseAgent] = {
            "site_list_merger": SiteListMergerAgent(llm_client),
            "trial_benchmarking": TrialBenchmarkingAgent(llm_client),
            "drug_reimbursement": DrugReimbursementAgent(llm_client),
            "enrollment_forecasting": EnrollmentForecastingAgent(llm_client),
        }

    def get_agent(self, skill_id: str) -> BaseAgent | None:
        return self._registry.get(skill_id)

    def all_skills(self) -> dict[str, BaseAgent]:
        return dict(self._registry)
