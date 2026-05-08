"""
Competitive Intelligence SubAgent.
Like TrialBenchmarkingAgent but filters for trials that have NOT YET STARTED.

Strategy:
1. Infer a status column from the Citeline dataset (LLM / heuristic).
2. Use the LLM to identify which status values mean "not yet started"
   (e.g. "Not yet recruiting", "Planned"). Falls back to keyword matching.
3. Apply that filter on top of the standard indication/phase/age-group filters.
4. Fallback: if no status column exists, filter for year >= current calendar year.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from backend.agents.base_agent import AgentResult
from backend.agents.trial_benchmarking_agent import (
    DEFAULT_DATASET,
    _COL_GUESSES,
    TrialBenchmarkingAgent,
)
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (
    COMPETITIVE_INTELLIGENCE_SYSTEM,
    COMPETITIVE_INTELLIGENCE_USER,
)
from backend.llm.response_parser import parse_benchmarking_response
from backend.llm.web_search import WebSearchClient
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

_CI_COL_GUESSES = {
    **_COL_GUESSES,
    "status": [
        "status", "trial_status", "study_status", "recruitment_status",
        "trial_state", "state", "current_status",
    ],
}

_NOT_YET_STARTED_KEYWORDS = [
    "not yet recruiting", "not yet started", "planned", "pre-recruitment",
    "pre-study", "pending", "approved", "registered", "upcoming",
]


class CompetitiveIntelligenceAgent(TrialBenchmarkingAgent):
    skill_id     = "competitive_intelligence"
    display_name = "Competitive Intelligence"
    description  = (
        "Identifies upcoming competitor trials (not yet started) by indication, "
        "age group, and phase using Citeline data."
    )

    def __init__(
        self,
        llm_client: LLMClient,
        dataset_name: str = DEFAULT_DATASET,
        web_search: WebSearchClient | None = None,
    ):
        super().__init__(llm_client, dataset_name=dataset_name, web_search=web_search)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group  = params["age_group"]
        phase      = params["phase"]

        data_context, matched_rows, col_map = self._query_citeline_not_started(
            indication, age_group, phase
        )

        web_context = ""
        if self.web_search:
            raw = self.web_search.search_for_skill("competitive_intelligence", params)
            if raw:
                web_context = f"\nSupplementary web search results:\n{raw}\n"

        messages = [
            {"role": "system", "content": COMPETITIVE_INTELLIGENCE_SYSTEM},
            {"role": "user", "content": COMPETITIVE_INTELLIGENCE_USER.format(
                indication=indication,
                age_group=age_group,
                phase=phase,
                data_context=data_context,
                web_context=web_context,
            )},
        ]
        try:
            raw  = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
            data = parse_benchmarking_response(raw)
        except Exception as e:
            logger.error("Competitive intelligence LLM call failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Error during competitive intelligence analysis: {e}",
            )

        metrics    = data.get("key_metrics", {})
        patterns   = data.get("notable_patterns", [])
        challenges = data.get("key_challenges", [])

        metrics_md = (
            "| Metric | Value |\n|---|---|\n"
            f"| Upcoming trial count | {metrics.get('upcoming_trial_count', '—')} |\n"
            f"| Median planned sites | {metrics.get('median_planned_sites', '—')} |\n"
            f"| Median planned patients | {metrics.get('median_planned_patients', '—')} |\n"
            f"| Sponsors represented | {metrics.get('sponsors_represented', '—')} |\n"
        )
        response_text = (
            f"**Competitive Intelligence: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + data.get("benchmark_summary", "") + "\n\n"
            + "**Key Metrics**\n\n" + metrics_md + "\n"
            + ("**Notable Patterns**\n\n" + "\n".join(f"- {p}" for p in patterns) + "\n\n" if patterns else "")
            + ("**Competitive Risks**\n\n" + "\n".join(f"- {c}" for c in challenges) + "\n\n" if challenges else "")
            + f"*{data.get('data_source', '')}*\n\n"
            + f"*{data.get('caveats', '')}*"
        )

        table_data, table_cols = self._build_output_table(matched_rows, col_map)
        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=table_data or None,
            table_columns=table_cols or None,
        )

    # ------------------------------------------------------------------
    # Column inference — adds "status" to the LLM prompt
    # ------------------------------------------------------------------

    def _infer_columns(self, df) -> dict[str, str]:
        sample: dict[str, list] = {}
        for col in df.columns:
            raw_vals = df[col].dropna().head(3).tolist()
            expanded = []
            for v in raw_vals:
                items = self._parse_list_cell(v)
                expanded.append(items if len(items) > 1 else (items[0] if items else ""))
            sample[col] = expanded

        prompt = (
            "You are analysing a clinical trial dataset. The column names and sample "
            "values are shown below. Map each relevant column to one of the semantic "
            "roles listed. Some columns may contain lists of values.\n\n"
            f"Dataset columns and samples:\n{json.dumps(sample, default=str, indent=2)}\n\n"
            "Semantic roles to map:\n"
            "  indication          — therapeutic area / disease (often a list per trial)\n"
            "  age_group           — patient age range or group (may be a list)\n"
            "  phase               — clinical trial phase\n"
            "  trial_id            — unique trial identifier\n"
            "  year                — year the trial started or is planned to start\n"
            "  status              — trial status / recruitment status "
            "(e.g. 'Not yet recruiting', 'Recruiting', 'Planned')\n"
            "  num_sites           — number of investigator sites\n"
            "  num_patients        — number of patients enrolled or planned\n"
            "  enrollment_rate     — enrollment rate per site per month\n"
            "  dropout_rate        — dropout / discontinuation rate (%)\n"
            "  screen_failure_rate — screen failure rate (%)\n"
            "  total_duration      — total trial duration (months)\n"
            "  enrollment_duration — enrollment period duration (months)\n\n"
            "Return ONLY a JSON object mapping role → exact column name. "
            "Omit any role you cannot confidently map. Use only column names that "
            "appear verbatim in the dataset.\n"
            '{"indication": "Indications", "phase": "Phase", "status": "Trial Status", ...}'
        )
        try:
            result = self.llm.complete_json([{"role": "user", "content": prompt}])
            valid = {role: col for role, col in result.items() if col in df.columns}
            self._log_trace(
                "Competitive Intelligence Column Inference",
                f"Dataset columns: {list(df.columns)}\n\n"
                f"Inferred mapping:\n"
                + "\n".join(f"  {role:22s} → {col}" for role, col in valid.items()),
            )
            return valid
        except Exception as e:
            logger.warning("Column inference LLM call failed: %s — using heuristics", e)
            return self._guess_columns_ci(df)

    def _guess_columns_ci(self, df) -> dict[str, str]:
        cols_lower = {c.lower(): c for c in df.columns}
        result = {}
        for role, candidates in _CI_COL_GUESSES.items():
            for candidate in candidates:
                if candidate.lower() in cols_lower:
                    result[role] = cols_lower[candidate.lower()]
                    break
        self._log_trace(
            "Competitive Intelligence Column Inference (heuristic)",
            f"Dataset columns: {list(df.columns)}\n\n"
            f"Heuristic mapping:\n"
            + "\n".join(f"  {role:22s} → {col}" for role, col in result.items()),
        )
        return result

    # ------------------------------------------------------------------
    # Querying — adds "not yet started" filter on top of standard filters
    # ------------------------------------------------------------------

    def _query_citeline_not_started(
        self, indication: str, age_group: str, phase: str
    ) -> tuple[str, list[dict], dict[str, str]]:
        df, load_error = self._load_citeline_df()
        if load_error:
            return load_error, [], {}
        if df is None:
            return "Citeline data unavailable.", [], {}

        col_map    = self._get_col_map(df)
        ind_col    = col_map.get("indication")
        ag_col     = col_map.get("age_group")
        phase_col  = col_map.get("phase")
        status_col = col_map.get("status")
        year_col   = col_map.get("year")

        if not ind_col:
            return (
                "Could not identify an indication column in the Citeline dataset.",
                [], col_map,
            )

        unique_indications = self._extract_unique_values(df[ind_col])
        unique_age_groups  = self._extract_unique_values(df[ag_col]) if ag_col else []
        unique_phases      = self._extract_unique_values(df[phase_col]) if phase_col else []

        canonical = self._semantic_map(
            indication, age_group, phase,
            indications=unique_indications,
            phases=unique_phases,
            age_groups=unique_age_groups,
        )
        mapped_indications = canonical.get("indication_matches", [])
        mapped_phase       = canonical.get("phase_match")
        mapped_age_group   = canonical.get("age_group_match")

        self._log_trace(
            "Competitive Intelligence Semantic Mapping",
            f"User inputs:\n"
            f"  Indication: {indication!r}  Phase: {phase!r}  Age Group: {age_group!r}\n\n"
            f"Mapped columns:\n"
            f"  indication={ind_col!r}  age_group={ag_col!r}  phase={phase_col!r}  status={status_col!r}\n\n"
            f"LLM mapping result:\n"
            f"  indication_matches: {mapped_indications}\n"
            f"  phase_match: {mapped_phase!r}\n"
            f"  age_group_match: {mapped_age_group!r}",
        )

        def _ind_mask(d):
            if mapped_indications:
                return self._list_col_isin(d[ind_col], set(mapped_indications))
            return self._list_col_isin(d[ind_col], {indication.strip()})

        def _phase_mask(d):
            if not phase_col:
                return [True] * len(d)
            target = {mapped_phase} if mapped_phase else {phase.strip()}
            return self._list_col_isin(d[phase_col], target)

        def _ag_mask(d):
            if not ag_col:
                return [True] * len(d)
            target = {mapped_age_group} if mapped_age_group else {age_group.strip()}
            return self._list_col_isin(d[ag_col], target)

        not_started_mask, filter_note = self._not_yet_started_mask(df, status_col, year_col)

        fallback_note = ""

        matched = df[_ind_mask(df) & _phase_mask(df) & _ag_mask(df) & not_started_mask]

        if matched.empty:
            matched = df[_ind_mask(df) & _phase_mask(df) & not_started_mask]
            if not matched.empty:
                fallback_note = f" (age group '{age_group}' not matched — showing all age groups)"

        if matched.empty:
            matched = df[_ind_mask(df) & not_started_mask]
            if not matched.empty:
                fallback_note = f" (phase '{phase}' not matched — showing all phases)"

        self._log_trace(
            "Competitive Intelligence Filter Result",
            f"Rows matched: {len(matched)}"
            + (f"\nFallback: {fallback_note}" if fallback_note else ""),
        )

        if matched.empty:
            return (
                f"No upcoming (not-yet-started) trials found for indication '{indication}' "
                f"(mapped to: {mapped_indications or 'no match'}). "
                "Analysis will be based on general competitive landscape knowledge.",
                [], col_map,
            )

        s    = self._compute_stats(matched, col_map)
        rows = matched.to_dict(orient="records")
        n    = len(matched)

        lines = [
            f"Found {n} upcoming (not-yet-started) trial(s){fallback_note}.",
            f"Filter: {filter_note}",
            "\nAggregate Statistics:",
        ]
        if s.get("median_rate")     is not None: lines.append(f"  Median enrollment rate:     {s['median_rate']:.2f} pts/site/month")
        if s.get("median_dropout")  is not None: lines.append(f"  Median dropout rate:        {s['median_dropout']:.1f}%")
        if s.get("median_sf")       is not None: lines.append(f"  Median screen failure rate: {s['median_sf']:.1f}%")
        if s.get("median_duration") is not None: lines.append(f"  Median total duration:      {s['median_duration']:.0f} months")
        if s.get("median_sites")    is not None: lines.append(f"  Site count range:           {s['site_min']:.0f}–{s['site_max']:.0f} (median {s['median_sites']:.0f})")
        if s.get("indications"):                 lines.append(f"  Indications matched:        {', '.join(s['indications'])}")
        if s.get("phases"):                      lines.append(f"  Phases in match:            {', '.join(s['phases'])}")

        return "\n".join(lines), rows, col_map

    # ------------------------------------------------------------------
    # "Not yet started" mask
    # ------------------------------------------------------------------

    def _not_yet_started_mask(self, df, status_col: str | None, year_col: str | None):
        """
        Return (boolean Series, filter_note).
        Prefers status column; falls back to year >= current year.
        """
        import pandas as pd
        current_year = datetime.now().year

        if status_col and status_col in df.columns:
            unique_statuses = self._extract_unique_values(df[status_col])
            matched_statuses = self._identify_not_started_statuses(unique_statuses)
            if matched_statuses:
                mask = self._list_col_isin(df[status_col], set(matched_statuses))
                note = f"Status in {{{', '.join(matched_statuses)}}}"
                self._log_trace(
                    "Competitive Intelligence Status Filter",
                    f"Status column: {status_col!r}\n"
                    f"Unique statuses: {', '.join(unique_statuses)}\n"
                    f"Matched statuses: {matched_statuses}",
                )
                return mask, note

        if year_col and year_col in df.columns:
            try:
                year_series = pd.to_numeric(df[year_col], errors="coerce")
                mask = year_series.ge(current_year).fillna(False)
                note = f"Start year >= {current_year} (no status column found)"
                self._log_trace(
                    "Competitive Intelligence Year Filter",
                    f"Year column: {year_col!r}  Threshold: >= {current_year}",
                )
                return mask, note
            except Exception:
                pass

        self._log_trace(
            "Competitive Intelligence Filter",
            "No status or year column found — returning all rows without 'not yet started' filter.",
        )
        return (
            pd.Series([True] * len(df), index=df.index),
            "No status filter applied (column not found — showing all trials)",
        )

    def _identify_not_started_statuses(self, unique_statuses: list[str]) -> list[str]:
        """
        Ask the LLM which status values mean the trial has not yet started.
        Falls back to keyword matching.
        """
        if not unique_statuses:
            return []

        prompt = (
            "From the list of clinical trial status values below, identify which ones "
            "indicate the trial has NOT yet started (i.e. it is planned, approved, or "
            "in pre-recruitment but has not yet begun enrolling or running).\n\n"
            f"Available status values: {', '.join(repr(s) for s in unique_statuses)}\n\n"
            "Return ONLY a JSON array of the matching status strings.\n"
            'Example: ["Not yet recruiting", "Planned"]\n'
            "If none match, return an empty array []."
        )
        try:
            result = self.llm.complete_json([{"role": "user", "content": prompt}])
            if isinstance(result, list):
                valid = set(unique_statuses)
                return [s for s in result if s in valid]
        except Exception as e:
            logger.warning("Status classification LLM call failed: %s", e)

        # Keyword fallback
        lower_map = {s.lower(): s for s in unique_statuses}
        return [lower_map[kw] for kw in _NOT_YET_STARTED_KEYWORDS if kw in lower_map]
