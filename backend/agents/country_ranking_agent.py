"""
Country Ranking SubAgent.
Ranks countries by their experience in executing clinical trials for a given indication.
Uses web search for supplementary data, then LLM synthesis into a ranked table.
"""
from __future__ import annotations

import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.web_search import WebSearchClient
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

COUNTRY_RANKING_SYSTEM = """\
You are a clinical trial strategy expert specializing in global site selection and country feasibility.

You will be given an indication (and optionally age group and phase). Your job is to rank countries \
globally by their experience and capability in executing clinical trials for that indication.

Consider the following factors for each country:
- **Number of trials**: Volume of registered/completed trials in this indication (ClinicalTrials.gov, EU CTR, etc.)
- **Investigator density**: Availability of experienced principal investigators and sites
- **Regulatory environment**: Speed and predictability of regulatory approval for trials
- **Patient access**: Size of the addressable patient population and recruitment feasibility
- **Infrastructure**: Quality of clinical research infrastructure (hospitals, CROs, labs)

If web search results are provided, use them to ground your ranking with real data. Cite specific \
numbers (e.g. trial counts, site counts) when available.

Return a JSON object:
{
  "summary": "<2-3 sentence overview of the global trial landscape for this indication>",
  "rankings": [
    {
      "rank": <int>,
      "country": "<country name>",
      "trial_count_estimate": "<approximate number of trials or 'N/A'>",
      "strengths": "<1-2 sentence summary of why this country ranks here>",
      "considerations": "<key risks or limitations>",
      "score": <float 1.0-10.0, overall suitability score>
    }
  ],
  "methodology_note": "<brief note on data sources and limitations>"
}

Rules:
- Rank at least 10 and up to 20 countries.
- Order by score descending (best first).
- Be specific — avoid generic statements. Reference trial counts and regulatory timelines where possible.
- If age group or phase narrows the field, reflect that in your ranking.
- Return ONLY the JSON object, no markdown fences, no other text."""

COUNTRY_RANKING_USER = """\
Indication: {indication}
{optional_context}
{web_context}
Rank countries globally by their experience and capability in executing clinical trials for this indication."""


class CountryRankingAgent(BaseAgent):
    skill_id = "country_ranking"
    display_name = "Country Ranking by Trial Experience"
    description = "Ranks countries by clinical trial experience for a given indication."

    def __init__(self, llm_client: LLMClient, web_search: WebSearchClient | None = None):
        self.llm = llm_client
        self.web_search = web_search

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group = params.get("age_group")
        phase = params.get("phase")

        # Build optional context line
        parts = []
        if age_group:
            parts.append(f"Age Group: {age_group}")
        if phase:
            parts.append(f"Phase: {phase}")
        optional_context = "\n".join(parts)

        # Web search for trial landscape data
        web_context = ""
        if self.web_search:
            raw = self.web_search.search_for_skill(
                "country_ranking", params,
                extra_terms="country trial count site selection",
            )
            if raw:
                web_context = f"\nWeb search results:\n{raw}\n"

        messages = [
            {"role": "system", "content": COUNTRY_RANKING_SYSTEM},
            {"role": "user", "content": COUNTRY_RANKING_USER.format(
                indication=indication,
                optional_context=optional_context,
                web_context=web_context,
            )},
        ]

        try:
            data = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Country ranking LLM call failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Error during country ranking: {e}",
            )

        rankings = data.get("rankings", [])
        summary = data.get("summary", "")
        methodology = data.get("methodology_note", "")

        # Build display
        title_parts = [indication]
        if phase:
            title_parts.append(phase)
        if age_group:
            title_parts.append(age_group.capitalize())
        title = " — ".join(title_parts)

        response_text = (
            f"**Country Ranking: {title}**\n\n"
            f"{summary}\n\n"
            f"*{methodology}*"
        )

        table_data = []
        for r in rankings:
            table_data.append({
                "Rank": r.get("rank", ""),
                "Country": r.get("country", ""),
                "Score": r.get("score", ""),
                "Est. Trial Count": r.get("trial_count_estimate", "N/A"),
                "Strengths": r.get("strengths", ""),
                "Considerations": r.get("considerations", ""),
            })

        table_columns = ["Rank", "Country", "Score", "Est. Trial Count",
                         "Strengths", "Considerations"]

        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=table_data,
            table_columns=table_columns,
        )
