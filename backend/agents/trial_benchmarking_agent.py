"""
Trial Benchmarking SubAgent.
Loads Citeline trial data from a configurable Dataiku dataset (default: CITELINE_DATA),
computes aggregate statistics, then passes the real data to the LLM for interpretation.

Column handling:
  - Column names are inferred via LLM on first load (cached per agent instance).
  - Indication and age_group columns may contain list values stored as strings
    (e.g. '["NSCLC", "Lung Cancer"]'). All filtering uses list-aware matching.
  - Falls back to a local citeline_data.csv when running outside Dataiku (dev/testing).
"""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (TRIAL_BENCHMARKING_SYSTEM,
                                           TRIAL_BENCHMARKING_USER)
from backend.llm.response_parser import parse_benchmarking_response
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

_LOCAL_CSV   = Path(__file__).parent.parent.parent / "data" / "citeline_data.csv"
DEFAULT_DATASET = "CITELINE_DATA"

# Heuristic candidates for each semantic role (checked case-insensitively)
_COL_GUESSES: dict[str, list[str]] = {
    "indication":          ["indication", "indications", "therapeutic_area", "disease",
                            "condition", "therapy_area"],
    "age_group":           ["age_group", "age_groups", "age_range", "patient_age",
                            "age_band", "age"],
    "phase":               ["phase", "trial_phase", "study_phase", "development_phase",
                            "clinical_phase"],
    "trial_id":            ["trial_id", "id", "nct_id", "study_id", "trial_number",
                            "protocol_id"],
    "year":                ["year", "year_started", "start_year", "year_initiated",
                            "start_date"],
    "num_sites":           ["num_sites", "sites", "site_count", "number_of_sites",
                            "investigator_sites"],
    "num_patients":        ["num_patients", "patients", "enrollment", "sample_size",
                            "enrolled_patients", "n"],
    "enrollment_rate":     ["enrollment_rate", "enroll_rate",
                            "enrollment_rate_pts_per_site_per_month",
                            "patients_per_site_per_month"],
    "dropout_rate":        ["dropout_rate", "dropout_rate_pct", "attrition_rate",
                            "discontinuation_rate"],
    "screen_failure_rate": ["screen_failure_rate", "screen_failure_rate_pct", "sfr",
                            "screening_failure_rate"],
    "total_duration":      ["total_duration", "total_duration_months", "duration",
                            "study_duration"],
    "enrollment_duration": ["enrollment_duration", "enrollment_duration_months",
                            "recruitment_duration"],
}


class TrialBenchmarkingAgent(BaseAgent):
    skill_id     = "trial_benchmarking"
    display_name = "Clinical Trial Benchmarking"
    description  = "Benchmarks clinical trials by indication, age group, and phase using Citeline data."

    def __init__(self, llm_client: LLMClient, dataset_name: str = DEFAULT_DATASET):
        self.llm          = llm_client
        self.dataset_name = dataset_name
        self._col_map: dict[str, str] | None = None   # cached after first load

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group  = params["age_group"]
        phase      = params["phase"]

        data_context, matched_rows, col_map = self._query_citeline(indication, age_group, phase)

        messages = [
            {"role": "system", "content": TRIAL_BENCHMARKING_SYSTEM},
            {"role": "user",   "content": TRIAL_BENCHMARKING_USER.format(
                indication=indication,
                age_group=age_group,
                phase=phase,
                data_context=data_context,
            )},
        ]
        try:
            raw  = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
            data = parse_benchmarking_response(raw)
        except Exception as e:
            logger.error("Trial benchmarking LLM call failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Error during trial benchmarking: {e}",
            )

        metrics    = data.get("key_metrics", {})
        patterns   = data.get("notable_patterns", [])
        challenges = data.get("key_challenges", [])

        metrics_md = (
            f"| Metric | Value |\n|---|---|\n"
            f"| Median enrollment rate | {metrics.get('median_enrollment_rate_patients_per_site_per_month', '—')} pts/site/mo |\n"
            f"| Median dropout rate | {metrics.get('median_dropout_rate_percent', '—')}% |\n"
            f"| Typical total duration | {metrics.get('typical_duration_months', '—')} months |\n"
            f"| Typical site count | {metrics.get('typical_site_count_range', '—')} |\n"
            f"| Typical screen failure rate | {metrics.get('typical_screen_failure_rate_percent', '—')}% |\n"
        )
        response_text = (
            f"**Trial Benchmarking: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + data.get("benchmark_summary", "") + "\n\n"
            + "**Key Metrics**\n\n" + metrics_md + "\n"
            + (f"**Notable Patterns**\n\n" + "\n".join(f"- {p}" for p in patterns) + "\n\n" if patterns else "")
            + (f"**Key Challenges**\n\n" + "\n".join(f"- {c}" for c in challenges) + "\n\n" if challenges else "")
            + f"*{data.get('data_source', '')}*\n\n"
            + f"*{data.get('caveats', '')}*"
        )

        # Build output table using inferred column names
        table_data, table_cols = self._build_output_table(matched_rows, col_map)
        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=table_data or None,
            table_columns=table_cols or None,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_citeline_df(self):
        """
        Returns (df, error_string). error_string is None on success.
        Priority: Dataiku dataset → local CSV fallback.
        """
        import pandas as pd

        # 1. Dataiku dataset
        try:
            import dataiku
            df = dataiku.Dataset(self.dataset_name).get_dataframe()
            self._log_trace(
                "Citeline Data Load",
                f"Loaded {len(df):,} rows from Dataiku dataset '{self.dataset_name}'.\n"
                f"Columns: {list(df.columns)}",
            )
            return df, None
        except ImportError:
            pass  # outside Dataiku
        except Exception as e:
            err = f"Could not read Dataiku dataset '{self.dataset_name}': {e}"
            logger.warning(err)
            return None, err

        # 2. Local CSV fallback
        if not _LOCAL_CSV.exists():
            return None, (
                f"Dataiku unavailable and local CSV not found at {_LOCAL_CSV}. "
                "Metrics will be based on general industry knowledge."
            )
        try:
            df = pd.read_csv(_LOCAL_CSV)
            self._log_trace(
                "Citeline Data Load",
                f"Loaded {len(df):,} rows from local CSV fallback.\n"
                f"Columns: {list(df.columns)}",
            )
            return df, None
        except Exception as e:
            return None, f"Could not read local Citeline CSV: {e}"

    # ------------------------------------------------------------------
    # Column inference (cached)
    # ------------------------------------------------------------------

    def _get_col_map(self, df) -> dict[str, str]:
        """Return cached column map, inferring it on first call."""
        if self._col_map is not None:
            return self._col_map
        self._col_map = self._infer_columns(df)
        return self._col_map

    def _infer_columns(self, df) -> dict[str, str]:
        """
        Ask the LLM to map dataset column names to semantic roles.
        Falls back to heuristic guessing if the LLM call fails.
        """
        # Build a compact sample: col → [first 3 non-null values]
        sample: dict[str, list] = {}
        for col in df.columns:
            raw_vals = df[col].dropna().head(3).tolist()
            # Expand list-type cells so the LLM can see what's inside
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
            "  year                — year the trial started\n"
            "  num_sites           — number of investigator sites\n"
            "  num_patients        — number of patients enrolled\n"
            "  enrollment_rate     — enrollment rate per site per month\n"
            "  dropout_rate        — dropout / discontinuation rate (%)\n"
            "  screen_failure_rate — screen failure rate (%)\n"
            "  total_duration      — total trial duration (months)\n"
            "  enrollment_duration — enrollment period duration (months)\n\n"
            "Return ONLY a JSON object mapping role → exact column name. "
            "Omit any role you cannot confidently map. Use only column names that "
            "appear verbatim in the dataset.\n"
            'Example: {"indication": "Indications", "phase": "Phase", ...}'
        )
        try:
            result = self.llm.complete_json([{"role": "user", "content": prompt}])
            # Keep only mappings that reference real columns
            valid = {role: col for role, col in result.items() if col in df.columns}
            self._log_trace(
                "Citeline Column Inference",
                f"Dataset columns: {list(df.columns)}\n\n"
                f"Inferred mapping:\n" +
                "\n".join(f"  {role:22s} → {col}" for role, col in valid.items()),
            )
            return valid
        except Exception as e:
            logger.warning("Column inference LLM call failed: %s — using heuristics", e)
            return self._guess_columns(df)

    def _guess_columns(self, df) -> dict[str, str]:
        """Heuristic fallback: match column names case-insensitively."""
        cols_lower = {c.lower(): c for c in df.columns}
        result = {}
        for role, candidates in _COL_GUESSES.items():
            for candidate in candidates:
                if candidate.lower() in cols_lower:
                    result[role] = cols_lower[candidate.lower()]
                    break
        self._log_trace(
            "Citeline Column Inference (heuristic)",
            f"Dataset columns: {list(df.columns)}\n\n"
            f"Heuristic mapping:\n" +
            "\n".join(f"  {role:22s} → {col}" for role, col in result.items()),
        )
        return result

    # ------------------------------------------------------------------
    # List-cell utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_list_cell(val) -> list[str]:
        """
        Parse a cell that may contain a list stored as a string.
        Handles: Python list, JSON array string, ast.literal_eval string,
        plain string (returned as single-element list).
        """
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
        if not isinstance(val, str):
            s = str(val).strip()
            return [s] if s and s.lower() not in ("nan", "none", "") else []
        val = val.strip()
        if not val or val.lower() in ("nan", "none"):
            return []
        if val.startswith("["):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(val)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
        return [val]

    def _extract_unique_values(self, series) -> list[str]:
        """
        Return sorted unique values from a Series, expanding list-type cells.
        """
        values: set[str] = set()
        for val in series.dropna():
            for item in self._parse_list_cell(val):
                if item:
                    values.add(item)
        return sorted(values)

    def _list_col_isin(self, series, targets: set[str]) -> "pd.Series":
        """
        Boolean mask: True when the cell's list contains any target value
        (case-insensitive).
        """
        targets_lower = {t.lower() for t in targets}

        def _check(val) -> bool:
            return any(v.lower() in targets_lower for v in self._parse_list_cell(val))

        return series.apply(_check)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _query_citeline(
        self, indication: str, age_group: str, phase: str
    ) -> tuple[str, list[dict], dict[str, str]]:
        """
        Returns (data_context_string, matched_rows, col_map).
        col_map maps semantic role → actual column name in df.
        """
        df, load_error = self._load_citeline_df()
        if load_error:
            return load_error, [], {}
        if df is None:
            return "Citeline data unavailable. Metrics will be based on general industry knowledge.", [], {}

        col_map = self._get_col_map(df)

        ind_col   = col_map.get("indication")
        ag_col    = col_map.get("age_group")
        phase_col = col_map.get("phase")

        if not ind_col:
            return (
                "Could not identify an indication column in the Citeline dataset. "
                "Metrics will be based on general industry knowledge.",
                [], col_map,
            )

        # ── Extract unique values (list-aware) ───────────────────────────────
        unique_indications = self._extract_unique_values(df[ind_col])
        unique_age_groups  = self._extract_unique_values(df[ag_col]) if ag_col else []
        unique_phases      = self._extract_unique_values(df[phase_col]) if phase_col else []

        # ── Semantic mapping ─────────────────────────────────────────────────
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
            "Citeline Semantic Mapping",
            f"User inputs:\n"
            f"  Indication: {indication!r}  Phase: {phase!r}  Age Group: {age_group!r}\n\n"
            f"Mapped columns:\n"
            f"  indication={ind_col!r}  age_group={ag_col!r}  phase={phase_col!r}\n\n"
            f"Unique values in dataset:\n"
            f"  Indications ({len(unique_indications)}): {', '.join(unique_indications[:20])}"
            + (" …" if len(unique_indications) > 20 else "") + "\n"
            f"  Phases: {', '.join(unique_phases)}\n"
            f"  Age Groups: {', '.join(unique_age_groups)}\n\n"
            f"LLM mapping result:\n"
            f"  indication_matches: {mapped_indications}\n"
            f"  phase_match: {mapped_phase!r}\n"
            f"  age_group_match: {mapped_age_group!r}",
        )

        # ── Build filter masks (list-aware) ──────────────────────────────────
        def _ind_mask(d):
            if mapped_indications:
                return self._list_col_isin(d[ind_col], set(mapped_indications))
            # Substring fallback
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

        import pandas as pd
        fallback_note = ""

        # 1. Full match
        matched = df[_ind_mask(df) & _phase_mask(df) & _ag_mask(df)]

        # 2. Relax age group
        if matched.empty:
            matched = df[_ind_mask(df) & _phase_mask(df)]
            if not matched.empty:
                fallback_note = f" (age group '{age_group}' not matched — showing all age groups)"

        # 3. Relax phase too
        if matched.empty:
            matched = df[_ind_mask(df)]
            if not matched.empty:
                fallback_note = f" (phase '{phase}' not matched — showing all phases)"

        self._log_trace(
            "Citeline Filter Result",
            f"Rows matched: {len(matched)}"
            + (f"\nFallback: {fallback_note}" if fallback_note else ""),
        )

        if matched.empty:
            return (
                f"No trials found for indication '{indication}' "
                f"(mapped to: {mapped_indications or 'no match'}). "
                "Metrics will be based on general industry knowledge.",
                [], col_map,
            )

        s       = self._compute_stats(matched, col_map)
        rows    = matched.to_dict(orient="records")
        n       = len(matched)

        lines = [f"Found {n} matching trial(s){fallback_note}."]
        lines.append("\nAggregate Statistics:")
        if s.get("median_rate")    is not None: lines.append(f"  Median enrollment rate:     {s['median_rate']:.2f} pts/site/month")
        if s.get("mean_rate")      is not None: lines.append(f"  Mean enrollment rate:       {s['mean_rate']:.2f} pts/site/month")
        if s.get("median_dropout") is not None: lines.append(f"  Median dropout rate:        {s['median_dropout']:.1f}%")
        if s.get("median_sf")      is not None: lines.append(f"  Median screen failure rate: {s['median_sf']:.1f}%")
        if s.get("median_duration")is not None: lines.append(f"  Median total duration:      {s['median_duration']:.0f} months")
        if s.get("median_sites")   is not None: lines.append(f"  Site count range:           {s['site_min']:.0f}–{s['site_max']:.0f} (median {s['median_sites']:.0f})")
        if s.get("indications"):                lines.append(f"  Indications matched:        {', '.join(s['indications'])}")
        if s.get("phases"):                     lines.append(f"  Phases in match:            {', '.join(s['phases'])}")

        return "\n".join(lines), rows, col_map

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats(self, df, col_map: dict[str, str]) -> dict:
        def _med(role):
            col = col_map.get(role)
            if not col or col not in df.columns:
                return None
            try:
                return float(df[col].median())
            except Exception:
                return None

        def _minmax(role):
            col = col_map.get(role)
            if not col or col not in df.columns:
                return None, None
            try:
                return float(df[col].min()), float(df[col].max())
            except Exception:
                return None, None

        ind_col   = col_map.get("indication")
        phase_col = col_map.get("phase")
        site_min, site_max = _minmax("num_sites")

        return {
            "median_rate":            _med("enrollment_rate"),
            "mean_rate":              float(df[col_map["enrollment_rate"]].mean()) if col_map.get("enrollment_rate") and col_map["enrollment_rate"] in df.columns else None,
            "median_dropout":         _med("dropout_rate"),
            "median_sf":              _med("screen_failure_rate"),
            "median_duration":        _med("total_duration"),
            "median_enroll_duration": _med("enrollment_duration"),
            "median_sites":           _med("num_sites"),
            "site_min":               site_min,
            "site_max":               site_max,
            "indications":            sorted(self._extract_unique_values(df[ind_col])) if ind_col and ind_col in df.columns else [],
            "phases":                 sorted(self._extract_unique_values(df[phase_col])) if phase_col and phase_col in df.columns else [],
        }

    # ------------------------------------------------------------------
    # Output table builder
    # ------------------------------------------------------------------

    def _build_output_table(
        self, rows: list[dict], col_map: dict[str, str]
    ) -> tuple[list[dict], list[str]]:
        """Build the matched-trial table using inferred column names."""
        if not rows:
            return [], []

        # Ordered display slots: (label, semantic_role)
        SLOTS = [
            ("Trial ID",               "trial_id"),
            ("Indication",             "indication"),
            ("Phase",                  "phase"),
            ("Age Group",              "age_group"),
            ("Year",                   "year"),
            ("Sites",                  "num_sites"),
            ("Patients",               "num_patients"),
            ("Enroll Rate\n(pts/site/mo)", "enrollment_rate"),
            ("Dropout %",              "dropout_rate"),
            ("Screen Fail %",          "screen_failure_rate"),
            ("Total Duration\n(mo)",   "total_duration"),
            ("Enroll Duration\n(mo)",  "enrollment_duration"),
        ]

        # Only include slots where the column was actually inferred
        active = [(lbl, col_map[role]) for lbl, role in SLOTS if role in col_map]
        if not active:
            return [], []

        labels, raw_cols = zip(*active)
        table_columns = list(labels)
        table_data = []
        for row in rows:
            rec = {}
            for lbl, col in active:
                val = row.get(col, "")
                # Flatten list values for display
                items = self._parse_list_cell(val)
                rec[lbl] = ", ".join(items) if len(items) > 1 else (items[0] if items else "")
            table_data.append(rec)

        return table_data, table_columns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_trace(self, label: str, summary: str) -> None:
        if not hasattr(self.llm, "call_log"):
            self.llm.call_log = []
        self.llm.call_log.append({
            "messages": [
                {"role": "system", "content": f"[{label}]"},
                {"role": "user",   "content": summary},
            ],
            "response": summary,
            "synthetic": True,
        })

    def _semantic_map(
        self,
        indication: str, age_group: str, phase: str,
        indications: list[str], phases: list[str], age_groups: list[str],
    ) -> dict:
        """
        LLM maps free-text user inputs → canonical values present in the dataset.
        Returns {indication_matches: [...], phase_match: str|None, age_group_match: str|None}.
        """
        prompt = (
            "Map the user's clinical trial query parameters to the closest matching "
            "values from the lists below. Match semantically — e.g. 'lung cancer' → "
            "'NSCLC', 'Phase II' → 'Phase 2', 'adults' → 'Adult'.\n\n"
            f"User query:  Indication={indication!r}  Phase={phase!r}  Age Group={age_group!r}\n\n"
            f"Available indications ({len(indications)}): {', '.join(indications)}\n"
            f"Available phases: {', '.join(phases)}\n"
            f"Available age groups: {', '.join(age_groups)}\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "indication_matches": ["<exact string from available indications>", ...],\n'
            '  "phase_match": "<exact string from available phases or null>",\n'
            '  "age_group_match": "<exact string from available age groups or null>"\n'
            "}\n"
            "Only use strings that appear verbatim in the available lists."
        )
        try:
            result = self.llm.complete_json([{"role": "user", "content": prompt}])
            valid_inds   = set(indications)
            valid_phases = set(phases)
            valid_ags    = set(age_groups)
            result["indication_matches"] = [
                i for i in result.get("indication_matches", []) if i in valid_inds
            ]
            if result.get("phase_match") not in valid_phases:
                result["phase_match"] = None
            if result.get("age_group_match") not in valid_ags:
                result["age_group_match"] = None
            return result
        except Exception as e:
            logger.warning("Semantic mapping failed: %s", e)
            return {"indication_matches": [], "phase_match": None, "age_group_match": None}
