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

        response_text = (
            f"**Trial Benchmarking: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + data.get("benchmark_summary", "")
            + "\n\n"
            + f"*{data.get('caveats', '')}*"
        )

        # Key metrics table
        metrics = data.get("key_metrics", {})
        metrics_rows = [
            {"Metric / Category": "Median Enrollment Rate (pts/site/mo)",
             "Value / Detail": str(metrics.get("median_enrollment_rate_patients_per_site_per_month", "—"))},
            {"Metric / Category": "Median Dropout Rate (%)",
             "Value / Detail": str(metrics.get("median_dropout_rate_percent", "—"))},
            {"Metric / Category": "Typical Total Duration (months)",
             "Value / Detail": str(metrics.get("typical_duration_months", "—"))},
            {"Metric / Category": "Typical Site Count Range",
             "Value / Detail": str(metrics.get("typical_site_count_range", "—"))},
            {"Metric / Category": "Typical Screen Failure Rate (%)",
             "Value / Detail": str(metrics.get("typical_screen_failure_rate_percent", "—"))},
            {"Metric / Category": "Data Source",
             "Value / Detail": data.get("data_source", "—")},
        ]

        patterns = data.get("notable_patterns", [])
        challenges = data.get("key_challenges", [])
        bullets_rows = (
            [{"Metric / Category": "Notable Pattern", "Value / Detail": p} for p in patterns]
            + [{"Metric / Category": "Key Challenge", "Value / Detail": c} for c in challenges]
        )

        # Matched trial detail rows (up to 15)
        detail_rows = []
        if matched_rows:
            detail_rows = [{"Metric / Category": "── Matched Citeline Trials ──", "Value / Detail": ""}]
            for row in matched_rows[:15]:
                detail_rows.append({
                    "Metric / Category": (
                        f"{row.get('indication', '')} | "
                        f"{row.get('phase', '')} | "
                        f"{row.get('age_group', '')} | "
                        f"{row.get('year_started', '')}"
                    ),
                    "Value / Detail": (
                        f"Sites: {row.get('num_sites', '')}, "
                        f"Pts: {row.get('num_patients_enrolled', '')}, "
                        f"Rate: {row.get('enrollment_rate_pts_per_site_per_month', '')} pts/site/mo, "
                        f"Dropout: {row.get('dropout_rate_pct', '')}%, "
                        f"Duration: {row.get('total_duration_months', '')} mo"
                    ),
                })

        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=metrics_rows + bullets_rows + detail_rows,
            table_columns=["Metric / Category", "Value / Detail"],
        )

    # ------------------------------------------------------------------
    # CSV querying
    # ------------------------------------------------------------------

    def _query_citeline(
        self, indication: str, age_group: str, phase: str
    ) -> tuple[str, list[dict]]:
        """
        Load citeline_data.csv, filter by indication (partial match), age_group,
        and phase. Falls back progressively when no exact match exists.
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

        # Normalise
        df.columns = [c.strip().lower() for c in df.columns]
        df["indication"] = df["indication"].str.strip()
        df["age_group"] = df["age_group"].str.strip().str.lower()
        df["phase"] = df["phase"].str.strip()

        ind_lower = indication.strip().lower()
        ag_lower = age_group.strip().lower()
        ph_lower = phase.strip().lower()

        def _ind_mask(d):
            return d["indication"].str.lower().str.contains(ind_lower, regex=False)

        # 1. Full match
        matched = df[_ind_mask(df) & (df["age_group"] == ag_lower) & (df["phase"].str.lower() == ph_lower)]
        fallback_note = ""

        # 2. Relax age group
        if matched.empty:
            matched = df[_ind_mask(df) & (df["phase"].str.lower() == ph_lower)]
            if not matched.empty:
                fallback_note = f" (age group '{age_group}' not in database — showing all age groups for this phase)"

        # 3. Relax phase too
        if matched.empty:
            matched = df[_ind_mask(df)]
            if not matched.empty:
                fallback_note = f" (phase '{phase}' not in database — showing all phases for this indication)"

        if matched.empty:
            return (
                f"No trials found in Citeline database matching indication '{indication}'. "
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
