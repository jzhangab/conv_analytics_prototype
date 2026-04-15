"""
CRO Site Profiling SubAgent.
Matches an uploaded CRO site list against the CTMS master site database
using a 2-step Jaro-Winkler algorithm:
  Step 1: site_name + city concatenation (JW > 0.9)
  Step 2: first 3 words of address + city concatenation (JW > 0.88)
Column mapping is inferred via LLM reasoning on column names.

For matched sites, calculates performance metrics from the CTMS dataset:
  - Avg Enrolled: average of the ENROLLED column across all rows for that site
  - Avg Months Diff: average of the MONTHS_DIFF column across all rows for that site

Performance caches (module-level, persist for the lifetime of the backend process):
  _ctms_df_cache       — raw CTMS DataFrame, keyed by dataset name
  _col_inference_cache — LLM column-mapping result, keyed by frozenset of column names
  _ctms_keys_cache     — pre-built JW matching keys, keyed by (dataset, name_col, city_col, addr_col)
  _ctms_metrics_cache  — pre-aggregated {site_id: {avg_enrolled, avg_months_diff}},
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
from backend.utils.string_matching import (first_n_words,
                                            jaro_winkler_similarity,
                                            normalize_for_matching)

logger = logging.getLogger(__name__)

DEFAULT_CTMS_DATASET = "CTMS_DATASET"

STEP1_THRESHOLD = 0.90
STEP2_THRESHOLD = 0.88

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# dataset_name -> pd.DataFrame
_ctms_df_cache: dict[str, pd.DataFrame] = {}

# frozenset(uploaded_col_names | ctms_col_names) -> col_map dict
_col_inference_cache: dict[frozenset, dict] = {}

# (dataset_name, name_col, city_col, addr_col) -> {"name_keys": list, "addr_keys": list}
_ctms_keys_cache: dict[tuple, dict] = {}

# (dataset_name, id_col, enrolled_col, months_diff_col)
#   -> {site_id_value: {"avg_enrolled": float, "avg_months_diff": float}}
_ctms_metrics_cache: dict[tuple, dict] = {}


class CROSiteProfilingAgent(BaseAgent):
    skill_id = "cro_site_profiling"
    display_name = "CRO Site Profiling"
    description = "Matches an uploaded site list against the CTMS master database and calculates site performance metrics."

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
        """Returns (df, error_string). error_string is None on success.

        The loaded DataFrame is cached for the lifetime of the backend process so
        subsequent calls skip the Dataiku network round-trip.
        """
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

        The result is cached by the frozenset of all column names so the LLM is
        only called once per unique combination of uploaded + CTMS columns.
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
    # Matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_str(df: pd.DataFrame, col: str | None, idx: int) -> str:
        """Get a cell value as a cleaned string, or empty if col is missing."""
        if col is None or col not in df.columns:
            return ""
        val = df.at[idx, col]
        if pd.isna(val):
            return ""
        return normalize_for_matching(str(val))

    @staticmethod
    def _build_keys(df: pd.DataFrame, name_col: str | None, city_col: str | None) -> list[str]:
        """Build site_name + city concatenation keys for each row."""
        keys = []
        for idx in df.index:
            name = CROSiteProfilingAgent._safe_str(df, name_col, idx)
            city = CROSiteProfilingAgent._safe_str(df, city_col, idx)
            keys.append(f"{name} {city}".strip())
        return keys

    @staticmethod
    def _build_address_keys(
        df: pd.DataFrame, addr_col: str | None, city_col: str | None,
    ) -> list[str]:
        """Build first-3-words-of-address + city concatenation keys."""
        keys = []
        for idx in df.index:
            addr = CROSiteProfilingAgent._safe_str(df, addr_col, idx)
            city = CROSiteProfilingAgent._safe_str(df, city_col, idx)
            addr_short = first_n_words(addr, 3)
            keys.append(f"{addr_short} {city}".strip())
        return keys

    def _match_step(
        self,
        uploaded_keys: list[str],
        ctms_keys: list[str],
        threshold: float,
        already_matched_uploaded: set[int],
        already_matched_ctms: set[int],
    ) -> list[tuple[int, int, float]]:
        """Return list of (uploaded_idx, ctms_idx, score) for matches above threshold.
        Each row matches at most once; best score wins."""
        candidates: list[tuple[int, int, float]] = []

        for u_idx, u_key in enumerate(uploaded_keys):
            if u_idx in already_matched_uploaded or not u_key:
                continue
            best_score = 0.0
            best_ctms = -1
            for c_idx, c_key in enumerate(ctms_keys):
                if c_idx in already_matched_ctms or not c_key:
                    continue
                score = jaro_winkler_similarity(u_key, c_key)
                if score > best_score:
                    best_score = score
                    best_ctms = c_idx
            if best_score >= threshold and best_ctms >= 0:
                candidates.append((u_idx, best_ctms, best_score))

        # Resolve conflicts: if multiple uploaded rows match the same CTMS row,
        # keep only the highest-scoring one.
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
    # CTMS key caching
    # ------------------------------------------------------------------

    def _get_ctms_keys(
        self,
        ctms_df: pd.DataFrame,
        c_name_col: str | None,
        c_city_col: str | None,
        c_addr_col: str | None,
    ) -> tuple[list[str], list[str]]:
        """Return (name_keys, addr_keys) for the CTMS df, building and caching on first call."""
        cache_key = (self.dataset_name, c_name_col, c_city_col, c_addr_col)
        if cache_key in _ctms_keys_cache:
            logger.info("CTMS keys cache hit.")
            cached = _ctms_keys_cache[cache_key]
            return cached["name_keys"], cached["addr_keys"]

        name_keys = self._build_keys(ctms_df, c_name_col, c_city_col)
        addr_keys = (
            self._build_address_keys(ctms_df, c_addr_col, c_city_col)
            if c_addr_col else []
        )
        _ctms_keys_cache[cache_key] = {"name_keys": name_keys, "addr_keys": addr_keys}
        logger.info("Built and cached CTMS keys for dataset '%s'.", self.dataset_name)
        return name_keys, addr_keys

    # ------------------------------------------------------------------
    # Site metrics — pre-aggregated lookup table
    # ------------------------------------------------------------------

    def _get_site_metrics_lookup(
        self,
        ctms_df: pd.DataFrame,
        id_col: str | None,
    ) -> dict:
        """Return a pre-aggregated lookup: ctms_idx -> {avg_enrolled, avg_months_diff}.

        On first call the full groupby aggregation is run and the result is stored in
        _ctms_metrics_cache.  Subsequent calls with the same dataset + column combination
        skip the computation entirely and return the cached dict in O(1).

        Without a site ID column the lookup is keyed by integer row index (no grouping).
        """
        # Resolve actual metric column names (case-insensitive)
        col_lower = {c.lower(): c for c in ctms_df.columns}
        enrolled_col = col_lower.get("enrolled")
        months_diff_col = col_lower.get("months_diff")

        cache_key = (self.dataset_name, id_col, enrolled_col, months_diff_col)
        if cache_key in _ctms_metrics_cache:
            logger.info("Site metrics cache hit.")
            return _ctms_metrics_cache[cache_key]

        if enrolled_col is None and months_diff_col is None:
            logger.warning("CTMS dataset has neither ENROLLED nor MONTHS_DIFF columns; skipping metrics.")
            result: dict = {}
            _ctms_metrics_cache[cache_key] = result
            return result

        if id_col and id_col in ctms_df.columns:
            # Aggregate all rows per site_id — these are the "stored rules"
            agg_cols: dict[str, str] = {}
            if enrolled_col:
                agg_cols[enrolled_col] = "mean"
            if months_diff_col:
                agg_cols[months_diff_col] = "mean"

            grouped = ctms_df.groupby(id_col, dropna=False).agg(agg_cols)

            # Build ctms_idx -> metrics dict using the site_id of each row as the key
            lookup: dict[int, dict] = {}
            for c_idx in ctms_df.index:
                site_id_val = ctms_df.at[c_idx, id_col]
                if pd.isna(site_id_val) or site_id_val not in grouped.index:
                    lookup[c_idx] = {}
                    continue
                grp = grouped.loc[site_id_val]
                m: dict = {}
                if enrolled_col and pd.notna(grp[enrolled_col]):
                    m["avg_enrolled"] = round(float(grp[enrolled_col]), 2)
                if months_diff_col and pd.notna(grp[months_diff_col]):
                    m["avg_months_diff"] = round(float(grp[months_diff_col]), 2)
                lookup[c_idx] = m
        else:
            # No ID column: per-row values (no aggregation possible)
            lookup = {}
            for c_idx in ctms_df.index:
                row = ctms_df.iloc[c_idx]
                m = {}
                if enrolled_col is not None:
                    val = row[enrolled_col]
                    m["avg_enrolled"] = round(float(val), 2) if pd.notna(val) else ""
                if months_diff_col is not None:
                    val = row[months_diff_col]
                    m["avg_months_diff"] = round(float(val), 2) if pd.notna(val) else ""
                lookup[c_idx] = m

        logger.info("Pre-aggregated site metrics for dataset '%s'; caching.", self.dataset_name)
        _ctms_metrics_cache[cache_key] = lookup
        return lookup

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

        uploaded_df = pd.DataFrame(file_info["data"])
        uploaded_df = uploaded_df.reset_index(drop=True)
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
        c_id_col = c_map.get("site_id")

        logger.info("Column mapping — uploaded: %s, ctms: %s", u_map, c_map)

        # --- Build uploaded keys (always fresh — varies by file) ---
        uploaded_name_keys = self._build_keys(uploaded_df, u_name_col, u_city_col)

        # --- Get CTMS keys (cached) ---
        ctms_name_keys, ctms_addr_keys = self._get_ctms_keys(
            ctms_df, c_name_col, c_city_col, c_addr_col,
        )

        # --- Step 1: site_name + city (JW > 0.9) ---
        step1_matches = self._match_step(
            uploaded_name_keys, ctms_name_keys, STEP1_THRESHOLD, set(), set(),
        )
        matched_uploaded = {m[0] for m in step1_matches}
        matched_ctms = {m[1] for m in step1_matches}

        # --- Step 2: first 3 words of address + city (JW > 0.88) ---
        step2_matches = []
        if u_addr_col and c_addr_col and ctms_addr_keys:
            uploaded_addr_keys = self._build_address_keys(uploaded_df, u_addr_col, u_city_col)
            step2_matches = self._match_step(
                uploaded_addr_keys, ctms_addr_keys, STEP2_THRESHOLD,
                matched_uploaded, matched_ctms,
            )

        # --- Combine results ---
        all_matches: dict[int, dict] = {}

        for u_idx, c_idx, score in step1_matches:
            all_matches[u_idx] = {
                "ctms_idx": c_idx,
                "score": round(score, 4),
                "step": "Step 1 (name+city)",
            }
        for u_idx, c_idx, score in step2_matches:
            all_matches[u_idx] = {
                "ctms_idx": c_idx,
                "score": round(score, 4),
                "step": "Step 2 (address+city)",
            }

        n_step1 = len(step1_matches)
        n_step2 = len(step2_matches)
        n_matched = n_step1 + n_step2
        n_unmatched = n_uploaded - n_matched
        match_rate = round(n_matched / max(n_uploaded, 1) * 100, 1)

        # --- Get pre-aggregated site metrics lookup (cached) ---
        site_metrics_lookup = self._get_site_metrics_lookup(ctms_df, c_id_col)

        summary_text = (
            f"**CRO Site Profiling Results**\n\n"
            f"Uploaded **{n_uploaded}** sites and compared against **{n_ctms}** CTMS sites.\n\n"
            f"- **Step 1** (site name + city, JW > {STEP1_THRESHOLD}): **{n_step1}** matches\n"
            f"- **Step 2** (address + city, JW > {STEP2_THRESHOLD}): **{n_step2}** additional matches\n"
            f"- **Total matched: {n_matched}** ({match_rate}%)\n"
            f"- **Unmatched: {n_unmatched}**\n\n"
            f"For matched sites, **Avg Enrolled** and **Avg Months Diff** are calculated "
            f"from all rows of the matched site in the CTMS dataset."
        )

        # --- Build result table ---
        table_data = []
        for i in range(n_uploaded):
            u_name = self._safe_str(uploaded_df, u_name_col, i) or str(uploaded_df.iloc[i, 0])
            match = all_matches.get(i)
            if match:
                c_idx = match["ctms_idx"]
                c_name = self._safe_str(ctms_df, c_name_col, c_idx)
                c_id = self._safe_str(ctms_df, c_id_col, c_idx) if c_id_col else ""
                metrics = site_metrics_lookup.get(c_idx, {})
                table_data.append({
                    "Row": i + 1,
                    "Uploaded Site Name": u_name,
                    "CTMS Site Name": c_name,
                    "CTMS Site ID": c_id,
                    "Match Status": "Matched",
                    "JW Score": match["score"],
                    "Match Step": match["step"],
                    "Avg Enrolled": metrics.get("avg_enrolled", ""),
                    "Avg Months Diff": metrics.get("avg_months_diff", ""),
                })
            else:
                table_data.append({
                    "Row": i + 1,
                    "Uploaded Site Name": u_name,
                    "CTMS Site Name": "",
                    "CTMS Site ID": "",
                    "Match Status": "Not matched",
                    "JW Score": "",
                    "Match Step": "",
                    "Avg Enrolled": "",
                    "Avg Months Diff": "",
                })

        table_columns = [
            "Row", "Uploaded Site Name", "CTMS Site Name",
            "Match Status", "CTMS Site ID", "JW Score", "Match Step",
            "Avg Enrolled", "Avg Months Diff",
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
    Accepts Werkzeug FileStorage objects or any object with .filename and .read().
    Falls back to content-based detection when the extension is missing or unknown.
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
