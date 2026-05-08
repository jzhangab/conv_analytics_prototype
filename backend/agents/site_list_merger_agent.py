"""
CRO Site Profiling SubAgent.
Matches an uploaded CRO site list against the CTMS master site database
using a 2-step Jaro-Winkler algorithm:
  Step 1: site_name + city concatenation (JW > 0.9)
  Step 2: first 3 words of address + city concatenation (JW > 0.88)
Column mapping is inferred via LLM reasoning on column names.

For matched sites, calculates performance metrics from the CTMS dataset:
  - Avg Enrolled: average of the ENROLLED column across all rows for that site
  - Median Months Diff: median of the MONTHS_DIFF column across all rows for that site
  - % Non-Enrolling Trials: % of rows for that site where randomized patients == 0
  - Trial Experience: total number of rows (trials) for that site
  - Screen Failure Rate: median across all rows for that site of (SCREENFAILED/SCREENED*100);
    rows where SCREENED is 0 or NaN count as 1.0 (i.e. 100%)

Performance notes
-----------------
* Key building is fully vectorised via pandas string ops (no Python row loops).
* City-based blocking reduces the CTMS candidate set per uploaded site from O(M)
  to O(M/C) where C is the number of distinct cities — typically a 10-100x
  reduction in JW calls.
* Jaro-Winkler is delegated to rapidfuzz (C extension) when installed, giving
  another 50-100x speedup.  The pure-Python fallback is used when rapidfuzz is
  absent (see backend/utils/string_matching.py).
* An early-exit threshold of 0.98 skips the remainder of the candidate list once
  a near-perfect match is found.

Module-level caches (persist for the lifetime of the backend process):
  _ctms_df_cache       — raw CTMS DataFrame, keyed by dataset name
  _col_inference_cache — LLM column-mapping result, keyed by frozenset of column names
  _ctms_keys_cache     — pre-built JW matching keys + city index, keyed by
                         (dataset, name_col, city_col, addr_col)
  _ctms_metrics_cache  — pre-aggregated {site_id: {avg_enrolled, median_months_diff}},
                         keyed by (dataset, id_col, enrolled_col, months_diff_col)

Call CROSiteProfilingAgent.clear_caches() to invalidate all caches (e.g. after the
CTMS dataset is updated).
"""
from __future__ import annotations

import io
import logging

import pandas as pd

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (SITE_COLUMN_INFERENCE_SYSTEM,
                                           SITE_COLUMN_INFERENCE_USER)
from backend.state.conversation_state import ConversationState
from backend.utils.string_matching import jaro_winkler_similarity

logger = logging.getLogger(__name__)

DEFAULT_CTMS_DATASET = "CTMS_DATASET"

STEP1_THRESHOLD = 0.90
STEP2_THRESHOLD = 0.88
# Once a match this strong is found, skip remaining candidates — negligible accuracy loss.
EARLY_EXIT_SCORE = 0.98

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# dataset_name -> pd.DataFrame
_ctms_df_cache: dict[str, pd.DataFrame] = {}

# frozenset(uploaded_col_names | ctms_col_names) -> col_map dict
_col_inference_cache: dict[frozenset, dict] = {}

# (dataset_name, name_col, city_col, addr_col) ->
#   {"name_keys": list[str], "addr_keys": list[str], "city_index": dict[str, list[int]]}
_ctms_keys_cache: dict[tuple, dict] = {}

# (dataset_name, id_col, enrolled_col, months_diff_col)
#   -> {ctms_idx: {"avg_enrolled": float, "median_months_diff": float}}
_ctms_metrics_cache: dict[tuple, dict] = {}


class CROSiteProfilingAgent(BaseAgent):
    skill_id = "cro_site_profiling"
    display_name = "CRO Site Profiling"
    description = (
        "Matches an uploaded site list against the CTMS master database "
        "and calculates site performance metrics."
    )

    def __init__(self, llm_client: LLMClient, dataset_name: str = DEFAULT_CTMS_DATASET):
        self.llm = llm_client
        self.dataset_name = dataset_name

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @staticmethod
    def clear_caches() -> None:
        """Invalidate all module-level caches (call after the CTMS dataset is updated)."""
        _ctms_df_cache.clear()
        _col_inference_cache.clear()
        _ctms_keys_cache.clear()
        _ctms_metrics_cache.clear()
        logger.info("CROSiteProfilingAgent: all caches cleared.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_ctms_df(self):
        """Returns (df, error_string). error_string is None on success."""
        if self.dataset_name in _ctms_df_cache:
            logger.info("CTMS cache hit for dataset '%s'.", self.dataset_name)
            return _ctms_df_cache[self.dataset_name], None

        try:
            import dataiku
            df = dataiku.Dataset(self.dataset_name).get_dataframe()
            df.columns = [str(c).strip() for c in df.columns]
            df = df.reset_index(drop=True)
            logger.info("Loaded %d rows from Dataiku dataset '%s'; caching.", len(df), self.dataset_name)
            _ctms_df_cache[self.dataset_name] = df
            return df, None
        except ImportError:
            return None, "Dataiku SDK not available — cannot load CTMS dataset."
        except Exception as e:
            return None, f"Could not read Dataiku dataset '{self.dataset_name}': {e}"

    # ------------------------------------------------------------------
    # LLM column inference
    # ------------------------------------------------------------------

    def _infer_columns(self, uploaded_cols: list[str], ctms_cols: list[str]) -> dict:
        """Use LLM to map column names to semantic roles for both datasets.

        Cached by frozenset of all column names — called at most once per unique
        combination of uploaded + CTMS columns.
        """
        cache_key = frozenset(uploaded_cols) | frozenset(f"ctms::{c}" for c in ctms_cols)
        if cache_key in _col_inference_cache:
            logger.info("Column-inference cache hit.")
            return _col_inference_cache[cache_key]

        messages = [
            {"role": "system", "content": SITE_COLUMN_INFERENCE_SYSTEM},
            {"role": "user", "content": SITE_COLUMN_INFERENCE_USER.format(
                uploaded_columns=", ".join(uploaded_cols),
                ctms_columns=", ".join(ctms_cols),
            )},
        ]
        result = self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
        if self.llm.call_log:
            self.llm.call_log[-1]["label"] = "Column Inference"
        _col_inference_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Vectorised key builders
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_col(df: pd.DataFrame, col: str | None) -> pd.Series:
        """Return a Series of normalised strings (lowercase, collapsed whitespace)."""
        if col and col in df.columns:
            return df[col].fillna("").astype(str).str.lower().str.split().str.join(" ")
        return pd.Series("", index=df.index)

    @staticmethod
    def _build_keys(df: pd.DataFrame, name_col: str | None, city_col: str | None) -> list[str]:
        """Build site_name + city concatenation keys — fully vectorised."""
        name_s = CROSiteProfilingAgent._normalize_col(df, name_col)
        city_s = CROSiteProfilingAgent._normalize_col(df, city_col)
        return (name_s + " " + city_s).str.strip().tolist()

    @staticmethod
    def _build_address_keys(
        df: pd.DataFrame, addr_col: str | None, city_col: str | None,
    ) -> list[str]:
        """Build first-3-words-of-address + city concatenation keys — fully vectorised."""
        addr_s = CROSiteProfilingAgent._normalize_col(df, addr_col)
        city_s = CROSiteProfilingAgent._normalize_col(df, city_col)
        addr_short = addr_s.str.split().str[:3].str.join(" ")
        return (addr_short + " " + city_s).str.strip().tolist()

    @staticmethod
    def _build_city_index(df: pd.DataFrame, city_col: str | None) -> dict[str, list[int]]:
        """
        Build a mapping: normalised_city -> list of row positions in df.

        Used by _match_step to restrict JW comparisons to same-city CTMS rows
        instead of scanning the full CTMS table for every uploaded site.
        """
        city_index: dict[str, list[int]] = {}
        if not city_col or city_col not in df.columns:
            return city_index
        city_s = CROSiteProfilingAgent._normalize_col(df, city_col)
        for pos, city in enumerate(city_s):
            city_index.setdefault(city, []).append(pos)
        return city_index

    # ------------------------------------------------------------------
    # CTMS key + city-index cache
    # ------------------------------------------------------------------

    def _get_ctms_keys(
        self,
        ctms_df: pd.DataFrame,
        c_name_col: str | None,
        c_city_col: str | None,
        c_addr_col: str | None,
    ) -> tuple[list[str], list[str], dict[str, list[int]]]:
        """Return (name_keys, addr_keys, city_index), building and caching on first call."""
        cache_key = (self.dataset_name, c_name_col, c_city_col, c_addr_col)
        if cache_key in _ctms_keys_cache:
            logger.info("CTMS keys cache hit.")
            cached = _ctms_keys_cache[cache_key]
            return cached["name_keys"], cached["addr_keys"], cached["city_index"]

        name_keys = self._build_keys(ctms_df, c_name_col, c_city_col)
        addr_keys = (
            self._build_address_keys(ctms_df, c_addr_col, c_city_col)
            if c_addr_col else []
        )
        city_index = self._build_city_index(ctms_df, c_city_col)

        _ctms_keys_cache[cache_key] = {
            "name_keys": name_keys,
            "addr_keys": addr_keys,
            "city_index": city_index,
        }
        logger.info(
            "Built and cached CTMS keys for dataset '%s' (%d distinct cities).",
            self.dataset_name, len(city_index),
        )
        return name_keys, addr_keys, city_index

    # ------------------------------------------------------------------
    # Site metrics — pre-aggregated lookup table
    # ------------------------------------------------------------------

    def _get_site_metrics_lookup(
        self,
        ctms_df: pd.DataFrame,
        id_col: str | None,
    ) -> dict:
        """Return a pre-aggregated lookup: ctms_idx -> {avg_enrolled, median_months_diff, pct_non_enrolling, trial_experience, screen_failure_rate}."""
        col_lower = {c.lower(): c for c in ctms_df.columns}
        enrolled_col     = col_lower.get("enrolled")
        months_diff_col  = col_lower.get("months_diff")
        screened_col     = col_lower.get("screened")
        screenfailed_col = col_lower.get("screenfailed") or col_lower.get("screen_failed")
        # Accept any column whose name contains "random" (e.g. randomized, randomized_patients)
        randomized_col   = next(
            (c for lower, c in sorted(col_lower.items()) if "random" in lower), None
        )

        cache_key = (self.dataset_name, id_col, enrolled_col, months_diff_col,
                     randomized_col, screened_col, screenfailed_col)
        if cache_key in _ctms_metrics_cache:
            logger.info("Site metrics cache hit.")
            return _ctms_metrics_cache[cache_key]

        if enrolled_col is None and months_diff_col is None and randomized_col is None \
                and screened_col is None and screenfailed_col is None:
            logger.warning("CTMS dataset has no recognised metric columns; skipping metrics.")
            _ctms_metrics_cache[cache_key] = {}
            return {}

        if id_col and id_col in ctms_df.columns:
            # Standard aggregations (only when there are columns to aggregate)
            agg_cols: dict[str, str] = {}
            if enrolled_col:
                agg_cols[enrolled_col] = "mean"
            if months_diff_col:
                agg_cols[months_diff_col] = "median"

            grouped = (
                ctms_df.groupby(id_col, dropna=False).agg(agg_cols)
                if agg_cols else None
            )

            # % non-enrolling: rows where randomized == 0 / total rows per site
            # Computed independently so it works even when agg_cols is empty.
            # Trial experience = row count per site (computed once, reused below)
            trial_counts = ctms_df.groupby(id_col).size()

            non_enrolling_pct: pd.Series | None = None
            if randomized_col:
                is_zero    = ctms_df[randomized_col].eq(0)    # NaN → False, not counted
                zero_count = is_zero.groupby(ctms_df[id_col]).sum()
                non_enrolling_pct = (zero_count / trial_counts * 100).round(1)

            # Screen failure rate per row: SCREENFAILED / SCREENED * 100.
            # Rows where SCREENED is 0 or NaN default to 1.0 (100%).
            # Aggregated metric = median of per-row rates for each site.
            screen_failure_median: pd.Series | None = None
            if screened_col and screenfailed_col:
                screened_vals    = pd.to_numeric(ctms_df[screened_col],    errors="coerce")
                screenfailed_vals = pd.to_numeric(ctms_df[screenfailed_col], errors="coerce")
                screened_safe    = screened_vals.where(screened_vals > 0)  # 0 → NaN
                per_row_rate     = (screenfailed_vals / screened_safe).fillna(1.0) * 100
                screen_failure_median = per_row_rate.groupby(ctms_df[id_col]).median().round(1)

            lookup: dict[int, dict] = {}
            for c_idx in ctms_df.index:
                site_id_val = ctms_df.at[c_idx, id_col]
                if pd.isna(site_id_val):
                    lookup[c_idx] = {}
                    continue
                m: dict = {}
                if grouped is not None and site_id_val in grouped.index:
                    grp = grouped.loc[site_id_val]
                    if enrolled_col and pd.notna(grp[enrolled_col]):
                        m["avg_enrolled"] = round(float(grp[enrolled_col]), 2)
                    if months_diff_col and pd.notna(grp[months_diff_col]):
                        m["median_months_diff"] = round(float(grp[months_diff_col]), 2)
                if non_enrolling_pct is not None and site_id_val in non_enrolling_pct.index:
                    pct = non_enrolling_pct.loc[site_id_val]
                    if pd.notna(pct):
                        m["pct_non_enrolling"] = float(pct)
                if site_id_val in trial_counts.index:
                    m["trial_experience"] = int(trial_counts.loc[site_id_val])
                if screen_failure_median is not None and site_id_val in screen_failure_median.index:
                    sfr = screen_failure_median.loc[site_id_val]
                    if pd.notna(sfr):
                        m["screen_failure_rate"] = float(sfr)
                lookup[c_idx] = m
        else:
            # No site ID column — per-row values only; % non-enrolling requires grouping
            lookup = {}
            for c_idx in ctms_df.index:
                row = ctms_df.iloc[c_idx]
                m = {}
                if enrolled_col is not None:
                    val = row[enrolled_col]
                    m["avg_enrolled"] = round(float(val), 2) if pd.notna(val) else ""
                if months_diff_col is not None:
                    val = row[months_diff_col]
                    m["median_months_diff"] = round(float(val), 2) if pd.notna(val) else ""
                if screened_col is not None and screenfailed_col is not None:
                    screened_val    = pd.to_numeric(row.get(screened_col),    errors="coerce")
                    screenfailed_val = pd.to_numeric(row.get(screenfailed_col), errors="coerce")
                    if pd.isna(screened_val) or screened_val == 0:
                        m["screen_failure_rate"] = 100.0
                    else:
                        m["screen_failure_rate"] = round(float(screenfailed_val / screened_val * 100), 1)
                lookup[c_idx] = m

        logger.info("Pre-aggregated site metrics for dataset '%s'; caching.", self.dataset_name)
        _ctms_metrics_cache[cache_key] = lookup
        return lookup

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_step(
        uploaded_keys: list[str],
        ctms_keys: list[str],
        threshold: float,
        already_matched_uploaded: set[int],
        already_matched_ctms: set[int],
        uploaded_city_keys: list[str] | None = None,
        ctms_city_index: dict[str, list[int]] | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Return [(uploaded_idx, ctms_idx, score)] for matches above threshold.

        City blocking: when city keys are supplied, each uploaded site is only
        compared against CTMS rows in the same city bucket, reducing the inner
        loop from O(M) to O(M/C).  Falls back to the full CTMS table when the
        uploaded city is unknown or the city bucket is empty.

        Early exit: once a score > EARLY_EXIT_SCORE is found the inner loop
        stops immediately — effectively perfect matches don't need further search.
        """
        n_ctms = len(ctms_keys)
        candidates: list[tuple[int, int, float]] = []

        for u_idx, u_key in enumerate(uploaded_keys):
            if u_idx in already_matched_uploaded or not u_key:
                continue

            # City-based candidate pruning
            if uploaded_city_keys and ctms_city_index and u_idx < len(uploaded_city_keys):
                u_city = uploaded_city_keys[u_idx]
                ctms_candidates: list[int] | range = (
                    ctms_city_index.get(u_city) or range(n_ctms)
                )
            else:
                ctms_candidates = range(n_ctms)

            best_score = 0.0
            best_ctms = -1
            for c_idx in ctms_candidates:
                if c_idx in already_matched_ctms:
                    continue
                c_key = ctms_keys[c_idx]
                if not c_key:
                    continue
                score = jaro_winkler_similarity(u_key, c_key)
                if score > best_score:
                    best_score = score
                    best_ctms = c_idx
                if best_score >= EARLY_EXIT_SCORE:
                    break

            if best_score >= threshold and best_ctms >= 0:
                candidates.append((u_idx, best_ctms, best_score))

        # Conflict resolution: if multiple uploaded rows matched the same CTMS
        # row, keep only the highest-scoring pair.
        candidates.sort(key=lambda x: x[2], reverse=True)
        used_ctms: set[int] = set()
        used_uploaded: set[int] = set()
        final: list[tuple[int, int, float]] = []
        for u_idx, c_idx, score in candidates:
            if c_idx in used_ctms or u_idx in used_uploaded:
                continue
            used_ctms.add(c_idx)
            used_uploaded.add(u_idx)
            final.append((u_idx, c_idx, score))
        return final

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        file_info = state.uploaded_files.get("site_file")
        if not file_info:
            return AgentResult(
                success=False, text_response="",
                error_message="Missing uploaded file: Site List File.",
            )

        ctms_df, load_err = self._load_ctms_df()
        if load_err:
            return AgentResult(success=False, text_response="", error_message=load_err)

        uploaded_df = pd.DataFrame(file_info["data"]).reset_index(drop=True)
        n_uploaded = len(uploaded_df)
        n_ctms = len(ctms_df)

        # --- LLM column inference (cached) ---
        try:
            col_map = self._infer_columns(
                list(uploaded_df.columns), list(ctms_df.columns),
            )
        except Exception as e:
            logger.error("Column inference failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Column inference failed: {e}",
            )

        u_map = col_map.get("uploaded", {})
        c_map = col_map.get("ctms", {})

        u_name_col = u_map.get("site_name")
        u_city_col = u_map.get("city")
        u_addr_col = u_map.get("address")
        c_name_col = c_map.get("site_name")
        c_city_col = c_map.get("city")
        c_addr_col = c_map.get("address")
        c_id_col   = c_map.get("site_id")

        logger.info("Column mapping — uploaded: %s, ctms: %s", u_map, c_map)

        # --- Build uploaded keys (always fresh — varies per file) ---
        uploaded_name_keys = self._build_keys(uploaded_df, u_name_col, u_city_col)
        # Raw city strings used for blocking (separate from the combined key)
        uploaded_city_keys = self._normalize_col(uploaded_df, u_city_col).tolist()

        # --- Get CTMS keys + city index (cached) ---
        ctms_name_keys, ctms_addr_keys, ctms_city_index = self._get_ctms_keys(
            ctms_df, c_name_col, c_city_col, c_addr_col,
        )

        # --- Step 1: site_name + city (JW > 0.9) ---
        step1_matches = self._match_step(
            uploaded_name_keys, ctms_name_keys, STEP1_THRESHOLD,
            set(), set(),
            uploaded_city_keys=uploaded_city_keys,
            ctms_city_index=ctms_city_index,
        )
        matched_uploaded = {m[0] for m in step1_matches}
        matched_ctms     = {m[1] for m in step1_matches}

        # --- Step 2: first 3 words of address + city (JW > 0.88) ---
        step2_matches: list[tuple[int, int, float]] = []
        if u_addr_col and c_addr_col and ctms_addr_keys:
            uploaded_addr_keys = self._build_address_keys(uploaded_df, u_addr_col, u_city_col)
            step2_matches = self._match_step(
                uploaded_addr_keys, ctms_addr_keys, STEP2_THRESHOLD,
                matched_uploaded, matched_ctms,
                uploaded_city_keys=uploaded_city_keys,
                ctms_city_index=ctms_city_index,
            )

        # --- Combine results ---
        all_matches: dict[int, dict] = {}
        for u_idx, c_idx, score in step1_matches:
            all_matches[u_idx] = {"ctms_idx": c_idx, "score": round(score, 4), "step": "Step 1 (name+city)"}
        for u_idx, c_idx, score in step2_matches:
            all_matches[u_idx] = {"ctms_idx": c_idx, "score": round(score, 4), "step": "Step 2 (address+city)"}

        n_step1    = len(step1_matches)
        n_step2    = len(step2_matches)
        n_matched  = n_step1 + n_step2
        n_unmatched = n_uploaded - n_matched
        match_rate  = round(n_matched / max(n_uploaded, 1) * 100, 1)

        # --- Get pre-aggregated site metrics lookup (cached) ---
        site_metrics_lookup = self._get_site_metrics_lookup(ctms_df, c_id_col)

        summary_text = (
            f"**CRO Site Profiling Results**\n\n"
            f"Uploaded **{n_uploaded}** sites and compared against **{n_ctms}** CTMS sites.\n\n"
            f"- **Step 1** (site name + city, JW > {STEP1_THRESHOLD}): **{n_step1}** matches\n"
            f"- **Step 2** (address + city, JW > {STEP2_THRESHOLD}): **{n_step2}** additional matches\n"
            f"- **Total matched: {n_matched}** ({match_rate}%)\n"
            f"- **Unmatched: {n_unmatched}**\n\n"
            f"For matched sites, **Avg Enrolled**, **Median Months Diff**, "
            f"**% Non-Enrolling Trials**, **Trial Experience**, and **Screen Failure Rate** "
            f"are calculated from all rows of the matched site in the CTMS dataset."
        )

        # --- Build result table ---
        table_data = []
        for i in range(n_uploaded):
            if u_name_col and u_name_col in uploaded_df.columns:
                u_name = str(uploaded_df.at[i, u_name_col] or "").strip() or str(uploaded_df.iloc[i, 0])
            else:
                u_name = str(uploaded_df.iloc[i, 0])

            match = all_matches.get(i)
            if match:
                c_idx = match["ctms_idx"]
                c_name = str(ctms_df.at[c_idx, c_name_col]).strip() if c_name_col and c_name_col in ctms_df.columns else ""
                c_id   = str(ctms_df.at[c_idx, c_id_col]).strip()   if c_id_col   and c_id_col   in ctms_df.columns else ""
                metrics = site_metrics_lookup.get(c_idx, {})
                table_data.append({
                    "Row":                i + 1,
                    "Uploaded Site Name": u_name,
                    "CTMS Site Name":     c_name,
                    "CTMS Site ID":       c_id,
                    "Match Status":       "Matched",
                    "JW Score":           match["score"],
                    "Match Step":         match["step"],
                    "Avg Enrolled":           metrics.get("avg_enrolled", ""),
                    "Median Months Diff":     metrics.get("median_months_diff", ""),
                    "% Non-Enrolling Trials": metrics.get("pct_non_enrolling", ""),
                    "Trial Experience":       metrics.get("trial_experience", ""),
                    "Screen Failure Rate":    metrics.get("screen_failure_rate", ""),
                })
            else:
                table_data.append({
                    "Row":                    i + 1,
                    "Uploaded Site Name":     u_name,
                    "CTMS Site Name":         "",
                    "CTMS Site ID":           "",
                    "Match Status":           "Not matched",
                    "JW Score":               "",
                    "Match Step":             "",
                    "Avg Enrolled":           "",
                    "Median Months Diff":     "",
                    "% Non-Enrolling Trials": "",
                    "Trial Experience":       "",
                    "Screen Failure Rate":    "",
                })

        table_columns = [
            "Row", "Uploaded Site Name", "CTMS Site Name",
            "Match Status", "CTMS Site ID", "JW Score", "Match Step",
            "Avg Enrolled", "Median Months Diff", "% Non-Enrolling Trials", "Trial Experience", "Screen Failure Rate",
        ]

        return AgentResult(
            success=True,
            text_response=summary_text,
            table_data=table_data,
            table_columns=table_columns,
        )


def parse_uploaded_file(file_storage) -> dict:
    """
    Parse a file (CSV or Excel) into a dict with keys: filename, data, columns.
    Accepts Werkzeug FileStorage objects, UploadedFile, or any object with
    .filename and .read().
    Raises ValueError on unsupported format or parse error.
    """
    filename = file_storage.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    raw = file_storage.read()
    buf = io.BytesIO(raw)

    try:
        if ext == "csv":
            df = pd.read_csv(buf)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(buf)
        else:
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(raw))
                except Exception:
                    raise ValueError(
                        f"Could not parse file '{filename or 'upload'}' as CSV or Excel. "
                        "Please upload a .csv, .xlsx, or .xls file."
                    )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse file '{filename}': {e}")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)

    return {
        "filename": filename,
        "data": df.to_dict(orient="records"),
        "columns": list(df.columns),
    }
