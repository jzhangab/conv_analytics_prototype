"""
Trial Benchmarking SubAgent.
Provides enrollment, dropout, duration, and site count benchmarks for
a given indication, age group, and trial phase.
"""
import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (TRIAL_BENCHMARKING_SYSTEM,
                                           TRIAL_BENCHMARKING_USER)
from backend.llm.response_parser import parse_benchmarking_response
from backend.state.conversation_state import ConversationState
from backend.utils.formatters import format_key_metrics_table, dict_list_to_table

logger = logging.getLogger(__name__)


class TrialBenchmarkingAgent(BaseAgent):
    skill_id = "trial_benchmarking"
    display_name = "Clinical Trial Benchmarking"
    description = "Benchmarks clinical trials by indication, age group, and phase."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group = params["age_group"]
        phase = params["phase"]

        messages = [
            {"role": "system", "content": TRIAL_BENCHMARKING_SYSTEM},
            {"role": "user", "content": TRIAL_BENCHMARKING_USER.format(
                indication=indication,
                age_group=age_group,
                phase=phase,
            )},
        ]

        try:
            raw = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
            data = parse_benchmarking_response(raw)
        except Exception as e:
            logger.error("Trial benchmarking LLM call failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Error during trial benchmarking: {e}"
            )

        # Build table rows: key metrics + notable patterns + challenges
        metrics_table = format_key_metrics_table(data.get("key_metrics", {}))

        # Notable patterns as a simple table
        patterns = data.get("notable_patterns", [])
        challenges = data.get("key_challenges", [])
        bullets_rows = (
            [{"Category": "Notable Pattern", "Detail": p} for p in patterns]
            + [{"Category": "Key Challenge", "Detail": c} for c in challenges]
        )
        bullets_table = dict_list_to_table(bullets_rows, columns=["Category", "Detail"])

        # Combine tables (frontend will render them sequentially)
        combined_table_data = metrics_table["rows"] + [["---", "---"]] + bullets_table["rows"]

        response_text = (
            f"**Trial Benchmarking: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + data.get("benchmark_summary", "")
            + "\n\n"
            + f"*{data.get('caveats', 'Note: These figures are based on general training data patterns and should be validated against live trial databases.')}*"
        )

        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=(
                [{"Metric": col, "Value": ""} for col in metrics_table["columns"]]
                + [{"Metric": row[0], "Value": row[1]} for row in metrics_table["rows"]]
                + [{"Metric": row[0], "Value": row[1]} for row in bullets_table["rows"]]
            ),
            table_columns=["Metric / Category", "Value / Detail"],
        )
