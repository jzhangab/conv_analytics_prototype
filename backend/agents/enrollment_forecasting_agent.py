"""
Enrollment & Site Activation Forecasting SubAgent.
Produces pessimistic / moderate / optimistic curves using a two-stage approach:
  1. LLM estimates domain parameters (enrollment rate, ramp period, dropout) per scenario.
  2. Deterministic Python math computes the curves (auditable, reproducible).
  3. LLM narrates the results.
"""
from __future__ import annotations

import logging
from datetime import datetime

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (ENROLLMENT_NARRATIVE_SYSTEM,
                                           ENROLLMENT_NARRATIVE_USER,
                                           ENROLLMENT_PARAMS_SYSTEM,
                                           ENROLLMENT_PARAMS_USER)
from backend.llm.response_parser import parse_enrollment_params
from backend.llm.web_search import WebSearchClient
from backend.state.conversation_state import ConversationState
from backend.utils.chart_builder import build_enrollment_figure, compute_scenario
from backend.utils.formatters import dict_list_to_table
from backend.utils.validators import normalize_date

logger = logging.getLogger(__name__)

SCENARIO_ORDER = ["pessimistic", "moderate", "optimistic"]


class EnrollmentForecastingAgent(BaseAgent):
    skill_id = "enrollment_forecasting"
    display_name = "Enrollment & Site Activation Forecasting"
    description = "Forecasts enrollment and site activation curves across three scenarios."

    def __init__(self, llm_client: LLMClient, web_search: WebSearchClient | None = None):
        self.llm = llm_client
        self.web_search = web_search

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group = params["age_group"]
        phase = params["phase"]
        num_sites = int(params["num_sites"])
        num_patients = int(params["num_patients"])

        # Parse start date, default to today
        raw_date = params.get("enrollment_start_date")
        if raw_date:
            normalized = normalize_date(str(raw_date))
            start_date = datetime.strptime(normalized, "%Y-%m-%d") if normalized else datetime.utcnow()
        else:
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Web search for supplementary enrollment benchmarks
        web_context = ""
        if self.web_search:
            raw = self.web_search.search_for_skill("enrollment_forecasting", params)
            if raw:
                web_context = f"\nSupplementary web search results:\n{raw}\n"

        # Stage 1: LLM estimates parameters for all three scenarios
        param_messages = [
            {"role": "system", "content": ENROLLMENT_PARAMS_SYSTEM},
            {"role": "user", "content": ENROLLMENT_PARAMS_USER.format(
                indication=indication,
                age_group=age_group,
                phase=phase,
                num_sites=num_sites,
                num_patients=num_patients,
                web_context=web_context,
            )},
        ]

        try:
            raw_params = self.llm.complete_json(param_messages, temperature=self.llm.temp_agents)
            scenario_params = parse_enrollment_params(raw_params)
        except Exception as e:
            logger.error("Enrollment param estimation failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Error estimating enrollment parameters: {e}"
            )

        # Stage 2: Compute curves deterministically
        scenario_results = {}
        for scenario in SCENARIO_ORDER:
            sp = scenario_params[scenario]
            scenario_results[scenario] = compute_scenario(
                num_sites=num_sites,
                num_patients=num_patients,
                enrollment_rate=sp["enrollment_rate_per_site_per_month"],
                ramp_period=sp["site_ramp_period_months"],
                dropout_rate_monthly_pct=sp["dropout_rate_monthly_percent"],
                start_date=start_date,
            )

        # Build Bokeh chart
        try:
            chart_json = build_enrollment_figure(
                scenarios=scenario_params,
                num_sites=num_sites,
                num_patients=num_patients,
                start_date=start_date,
                indication=indication,
                phase=phase,
            )
        except Exception as e:
            logger.error("Chart building failed: %s", e)
            chart_json = None

        # Stage 3: LLM narrative
        narrative_messages = [
            {"role": "system", "content": ENROLLMENT_NARRATIVE_SYSTEM},
            {"role": "user", "content": ENROLLMENT_NARRATIVE_USER.format(
                indication=indication,
                phase=phase,
                age_group=age_group,
                num_patients=num_patients,
                num_sites=num_sites,
                pessimistic_months=scenario_results["pessimistic"]["completion_month"],
                pessimistic_peak_sites=round(scenario_results["pessimistic"]["peak_active_sites"], 1),
                moderate_months=scenario_results["moderate"]["completion_month"],
                moderate_peak_sites=round(scenario_results["moderate"]["peak_active_sites"], 1),
                optimistic_months=scenario_results["optimistic"]["completion_month"],
                optimistic_peak_sites=round(scenario_results["optimistic"]["peak_active_sites"], 1),
            )},
        ]

        try:
            narrative = self.llm.complete(narrative_messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.warning("Narrative generation failed, using fallback: %s", e)
            narrative = self._fallback_narrative(scenario_results, indication, phase)

        response_text = (
            f"**Enrollment Forecast: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + narrative
        )

        # Summary table for the three scenarios
        table_rows = []
        for scenario in SCENARIO_ORDER:
            sr = scenario_results[scenario]
            sp = scenario_params[scenario]
            table_rows.append({
                "Scenario": scenario.capitalize(),
                "Enrollment Rate (pts/site/mo)": round(sp["enrollment_rate_per_site_per_month"], 2),
                "Site Ramp Period (months)": sp["site_ramp_period_months"],
                "Monthly Dropout (%)": sp["dropout_rate_monthly_percent"],
                "Enrollment Completion (months)": sr["completion_month"],
                "Peak Active Sites": round(sr["peak_active_sites"], 1),
                "Rationale": sp.get("rationale", ""),
            })

        columns = [
            "Scenario", "Enrollment Rate (pts/site/mo)", "Site Ramp Period (months)",
            "Monthly Dropout (%)", "Enrollment Completion (months)", "Peak Active Sites", "Rationale"
        ]

        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=table_rows,
            table_columns=columns,
            chart_json=chart_json,
        )

    def _fallback_narrative(self, results: dict, indication: str, phase: str) -> str:
        pess = results["pessimistic"]["completion_month"]
        mod = results["moderate"]["completion_month"]
        opt = results["optimistic"]["completion_month"]
        return (
            f"Based on the modeled parameters for {indication} {phase}, enrollment is projected "
            f"to complete in approximately {opt} months (optimistic), {mod} months (moderate), "
            f"or {pess} months (pessimistic) from the planned start date. "
            f"The chart above shows cumulative patient enrollment (solid lines) and site activation "
            f"(dashed lines) across all three scenarios. "
            f"Actual outcomes will depend on site activation pace, screen failure rates, and protocol complexity."
        )
