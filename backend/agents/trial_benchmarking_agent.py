"""
Trial Benchmarking SubAgent.
Queries citeline_data.csv for matching historical trials, computes aggregate
statistics, then passes the real data to the LLM for narrative interpretation.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (TRIAL_BENCHMARKING_SYSTEM,
                                           TRIAL_BENCHMARKING_USER)
from backend.llm.response_parser import parse_benchmarking_response
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

# Path to the Citeline data file — relative to the project root
_DATA_FILE = Path(__file__).parent.parent.parent / "data" / "citeline_data.csv"


class TrialBenchmarkingAgent(BaseAgent):
    skill_id = "trial_benchmarking"
    display_name = "Clinical Trial Benchmarking"
    description = "Benchmarks clinical trials by indication, age group, and phase using Citeline data."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group = params["age_group"]
        phase = params["phase"]

        # Query CSV and build data context string for the LLM
        data_context, matched_rows = self._query_citeline(indication, age_group, phase)

        messages = [
            {"role": "system", "content": TRIAL_BENCHMARKING_SYSTEM},
            {"role": "user", "content": TRIAL_BENCHMARKING_USER.format(
                indication=indication,
                age_group=age_group,
                phase=phase,
                data_context=data_context,
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

        metrics = data.get("key_metrics", {})
        patterns = data.get("notable_patterns", [])
        challenges = data.get("key_challenges", [])

        # Build narrative with summary metrics embedded as markdown
        metrics_md = (
            f"| Metric | Value |\n|---|---|\n"
            f"| Median enrollment rate | {metrics.get('median_enrollment_rate_patients_per_site_per_month', '—')} pts/site/mo |\n"
            f"| Median dropout rate | {metrics.get('median_dropout_rate_percent', '—')}% |\n"
            f"| Typical total duration | {metrics.get('typical_duration_months', '—')} months |\n"
            f"| Typical site count | {metrics.get('typical_site_count_range', '—')} |\n"
            f"| Typical screen failure rate | {metrics.get('typical_screen_failure_rate_percent', '—')}% |\n"
        )
        patterns_md = "\n".join(f"- {p}" for p in patterns) if patterns else ""
        challenges_md = "\n".join(f"- {c}" for c in challenges) if challenges else ""

        response_text = (
            f"**Trial Benchmarking: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + data.get("benchmark_summary", "") + "\n\n"
            + "**Key Metrics**\n\n" + metrics_md + "\n"
            + (f"**Notable Patterns**\n\n{patterns_md}\n\n" if patterns_md else "")
            + (f"**Key Challenges**\n\n{challenges_md}\n\n" if challenges_md else "")
            + f"*{data.get('data_source', '')}*\n\n"
            + f"*{data.get('caveats', '')}*"
        )

        # Matched trial rows as a proper columnar table
        trial_table_columns = [
            "Trial ID", "Indication", "Phase", "Age Group", "Year",
            "Sites", "Patients", "Enroll Rate\n(pts/site/mo)",
            "Dropout %", "Screen Fail %", "Total Duration\n(mo)", "Enroll Duration\n(mo)",
        ]
        trial_table_data = []
        for row in matched_rows:
            trial_table_data.append({
                "Trial ID":                   row.get("trial_id", ""),
                "Indication":                 row.get("indication", ""),
                "Phase":                      row.get("phase", ""),
                "Age Group":                  row.get("age_group", ""),
                "Year":                       row.get("year_started", ""),
                "Sites":                      row.get("num_sites", ""),
                "Patients":                   row.get("num_patients_enrolled", ""),
                "Enroll Rate\n(pts/site/mo)": row.get("enrollment_rate_pts_per_site_per_month", ""),
                "Dropout %":                  row.get("dropout_rate_pct", ""),
                "Screen Fail %":              row.get("screen_failure_rate_pct", ""),
                "Total Duration\n(mo)":       row.get("total_duration_months", ""),
                "Enroll Duration\n(mo)":      row.get("enrollment_duration_months", ""),
            })

        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=trial_table_data if trial_table_data else None,
            table_columns=trial_table_columns if trial_table_data else None,
        )

    # ------------------------------------------------------------------
    # CSV querying
    # ------------------------------------------------------------------

    def _query_citeline(
        self, indication: str, age_group: str, phase: str
    ) -> tuple[str, list[dict]]:
        """
        Load citeline_data.csv, semantically map user inputs to canonical dataset
        values via LLM, then filter. Falls back progressively when no exact match.
        Returns (data_context_string, matched_rows_as_list_of_dicts).
        """
        try:
            import pandas as pd
        except ImportError:
            return "pandas not available; no database query performed.", []

        if not _DATA_FILE.exists():
            return (
                f"Citeline data file not found at {_DATA_FILE}. "
                "Metrics will be based on general industry knowledge.",
                [],
            )

        try:
            df = pd.read_csv(_DATA_FILE)
        except Exception as e:
            return f"Could not read Citeline data: {e}", []

        # Normalise column names and values
        df.columns = [c.strip().lower() for c in df.columns]
        df["indication"] = df["indication"].str.strip()
        df["age_group"]  = df["age_group"].str.strip().str.lower()
        df["phase"]      = df["phase"].str.strip()

        dataset_indications = sorted(df["indication"].unique().tolist())
        dataset_phases      = sorted(df["phase"].unique().tolist())
        dataset_age_groups  = sorted(df["age_group"].unique().tolist())

        # Semantically map user inputs → canonical dataset values via LLM
        canonical = self._semantic_map(
            indication, age_group, phase,
            indications=dataset_indications,
            phases=dataset_phases,
            age_groups=dataset_age_groups,
        )
        mapped_indications = canonical.get("indication_matches", [])
        mapped_phase       = canonical.get("phase_match")
        mapped_age_group   = canonical.get("age_group_match")

        # Log mapping result to call_log so the notebook trace pane shows it
        self._log_trace(
            label="Citeline Semantic Mapping",
            summary=(
                f"User inputs:\n"
                f"  Indication: {indication!r}\n"
                f"  Phase:      {phase!r}\n"
                f"  Age Group:  {age_group!r}\n\n"
                f"Dataset values available:\n"
                f"  Indications: {', '.join(dataset_indications)}\n"
                f"  Phases:      {', '.join(dataset_phases)}\n"
                f"  Age Groups:  {', '.join(dataset_age_groups)}\n\n"
                f"LLM mapping result:\n"
                f"  indication_matches: {mapped_indications}\n"
                f"  phase_match:        {mapped_phase!r}\n"
                f"  age_group_match:    {mapped_age_group!r}"
            ),
        )

        # Case-insensitive normalised lookup maps (handles apostrophe variants, spacing)
        def _norm(s):
            return s.lower().replace("\u2019", "'").replace("\u2018", "'").strip()

        ind_norm_map = {_norm(v): v for v in dataset_indications}
        phase_norm_map = {_norm(v): v for v in dataset_phases}
        ag_norm_map = {_norm(v): v for v in dataset_age_groups}

        # Re-validate mapped values using normalised comparison
        mapped_indications = [
            ind_norm_map[_norm(i)] for i in mapped_indications if _norm(i) in ind_norm_map
        ]
        if mapped_phase and _norm(mapped_phase) in phase_norm_map:
            mapped_phase = phase_norm_map[_norm(mapped_phase)]
        else:
            mapped_phase = None
        if mapped_age_group and _norm(mapped_age_group) in ag_norm_map:
            mapped_age_group = ag_norm_map[_norm(mapped_age_group)]
        else:
            mapped_age_group = None

        def _ind_mask(d):
            if mapped_indications:
                return d["indication"].isin(mapped_indications)
            return d["indication"].str.lower().str.contains(indication.strip().lower(), regex=False)

        def _phase_mask(d):
            if mapped_phase:
                return d["phase"] == mapped_phase
            return d["phase"].str.lower() == phase.strip().lower()

        def _ag_mask(d):
            if mapped_age_group:
                return d["age_group"] == mapped_age_group
            return d["age_group"] == age_group.strip().lower()

        fallback_note = ""

        # 1. Full match: indication + phase + age_group
        matched = df[_ind_mask(df) & _phase_mask(df) & _ag_mask(df)]

        # 2. Relax age group
        if matched.empty:
            matched = df[_ind_mask(df) & _phase_mask(df)]
            if not matched.empty:
                fallback_note = f" (age group '{age_group}' not found — showing all age groups for this phase)"

        # 3. Relax phase too
        if matched.empty:
            matched = df[_ind_mask(df)]
            if not matched.empty:
                fallback_note = f" (phase '{phase}' not found — showing all phases for this indication)"

        # Log match result
        self._log_trace(
            label="Citeline Filter Result",
            summary=(
                f"After normalised validation:\n"
                f"  indication_matches: {mapped_indications}\n"
                f"  phase_match:        {mapped_phase!r}\n"
                f"  age_group_match:    {mapped_age_group!r}\n\n"
                f"Rows matched: {len(matched)}"
                + (f"\nFallback applied: {fallback_note}" if fallback_note else "")
            ),
        )

        if matched.empty:
            return (
                f"No trials found in Citeline database for indication '{indication}' "
                f"(mapped to: {mapped_indications or 'no match'}). "
                "Metrics will be based on general industry knowledge.",
                [],
            )

        n = len(matched)
        s = self._compute_stats(matched)
        matched_rows = matched.to_dict(orient="records")

        context = (
            f"Found {n} matching trial(s) in the Citeline database{fallback_note}.\n\n"
            f"Aggregate Statistics (computed from matched trials):\n"
            f"  Median enrollment rate:      {s['median_rate']:.2f} pts/site/month\n"
            f"  Mean enrollment rate:        {s['mean_rate']:.2f} pts/site/month\n"
            f"  Median dropout rate:         {s['median_dropout']:.1f}%\n"
            f"  Median screen failure rate:  {s['median_sf']:.1f}%\n"
            f"  Median total duration:       {s['median_duration']:.0f} months\n"
            f"  Median enrollment duration:  {s['median_enroll_duration']:.0f} months\n"
            f"  Site count range:            {s['site_min']:.0f}–{s['site_max']:.0f} "
            f"(median {s['median_sites']:.0f})\n"
            f"  Indications matched:         {', '.join(s['indications'])}\n"
            f"  Phases in match:             {', '.join(s['phases'])}\n"
        )
        return context, matched_rows

    def _log_trace(self, label: str, summary: str) -> None:
        """Append a synthetic entry to the LLM client's call_log for the trace pane."""
        if not hasattr(self.llm, "call_log"):
            self.llm.call_log = []
        self.llm.call_log.append({
            "messages": [{"role": "system", "content": f"[{label}]"}, {"role": "user", "content": summary}],
            "response": summary,
            "synthetic": True,
        })

    def _semantic_map(
        self,
        indication: str,
        age_group: str,
        phase: str,
        indications: list[str],
        phases: list[str],
        age_groups: list[str],
    ) -> dict:
        """
        Call the LLM to map free-text user inputs to canonical values that exist
        in the Citeline dataset. Returns a dict with keys:
          indication_matches: list of matching indication strings (may be >1)
          phase_match:        single matching phase string or null
          age_group_match:    single matching age_group string or null
        """
        prompt = (
            "Map the user's clinical trial query parameters to the closest matching "
            "values from the provided database lists. Match semantically — e.g. "
            "'lung cancer' → ['NSCLC'], 'diabetes' → ['Type 2 Diabetes'], "
            "'Phase II' → 'Phase 2', 'adults' → 'adult', 'elderly patients' → 'elderly'.\n\n"
            f"User query:\n"
            f"  Indication: {indication}\n"
            f"  Phase: {phase}\n"
            f"  Age Group: {age_group}\n\n"
            f"Available indications: {', '.join(indications)}\n"
            f"Available phases: {', '.join(phases)}\n"
            f"Available age groups: {', '.join(age_groups)}\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "indication_matches": ["<exact string from available indications>", ...],\n'
            '  "phase_match": "<exact string from available phases or null>",\n'
            '  "age_group_match": "<exact string from available age groups or null>"\n'
            "}\n"
            "Use null if no reasonable match exists. Only return strings that appear "
            "verbatim in the available lists."
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self.llm.complete_json(messages)
            # Validate that returned values exist in the dataset
            valid_inds = set(indications)
            result["indication_matches"] = [
                i for i in result.get("indication_matches", []) if i in valid_inds
            ]
            if result.get("phase_match") not in phases:
                result["phase_match"] = None
            if result.get("age_group_match") not in age_groups:
                result["age_group_match"] = None
            return result
        except Exception as e:
            logger.warning("Semantic mapping LLM call failed: %s", e)
            return {"indication_matches": [], "phase_match": None, "age_group_match": None}

    def _compute_stats(self, df) -> dict:
        return {
            "median_rate":            float(df["enrollment_rate_pts_per_site_per_month"].median()),
            "mean_rate":              float(df["enrollment_rate_pts_per_site_per_month"].mean()),
            "median_dropout":         float(df["dropout_rate_pct"].median()),
            "median_sf":              float(df["screen_failure_rate_pct"].median()),
            "median_duration":        float(df["total_duration_months"].median()),
            "median_enroll_duration": float(df["enrollment_duration_months"].median()),
            "median_sites":           float(df["num_sites"].median()),
            "site_min":               float(df["num_sites"].min()),
            "site_max":               float(df["num_sites"].max()),
            "indications":            sorted(df["indication"].unique().tolist()),
            "phases":                 sorted(df["phase"].unique().tolist()),
        }
