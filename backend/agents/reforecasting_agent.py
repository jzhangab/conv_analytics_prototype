"""
Reforecasting SubAgent.
Loads the REFORECAST Dataiku dataset, filters by protocol_number,
and plots lower_bound / mean_ / upper_bound enrollment curves with a
target_subjected horizontal reference line.
"""
from __future__ import annotations

import logging
from difflib import get_close_matches
from pathlib import Path

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "REFORECAST"
_LOCAL_CSV = Path(__file__).parent.parent.parent / "data" / "reforecast_data.csv"


class ReforecastingAgent(BaseAgent):
    skill_id = "reforecasting"
    display_name = "Enrollment Reforecasting"
    description = "Plots reforecast enrollment curves for a given protocol from the REFORECAST dataset."

    def __init__(self, dataset_name: str = DEFAULT_DATASET):
        self.dataset_name = dataset_name

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_reforecast_df(self):
        """Returns (df, error_string). error_string is None on success."""
        import pandas as pd

        # 1. Dataiku dataset
        try:
            import dataiku
            df = dataiku.Dataset(self.dataset_name).get_dataframe()
            df.columns = [str(c).strip().lower() for c in df.columns]
            logger.info("Loaded %d rows from Dataiku dataset '%s'.", len(df), self.dataset_name)
            return df, None
        except ImportError:
            pass  # outside Dataiku
        except Exception as e:
            err = f"Could not read Dataiku dataset '{self.dataset_name}': {e}"
            logger.warning(err)
            return None, err

        # 2. Local CSV fallback (dev / testing)
        if not _LOCAL_CSV.exists():
            return None, (
                f"Dataiku unavailable and local CSV not found at {_LOCAL_CSV}. "
                "Cannot load reforecast data."
            )
        try:
            df = pd.read_csv(_LOCAL_CSV)
            df.columns = [str(c).strip().lower() for c in df.columns]
            logger.info("Loaded %d rows from local CSV fallback.", len(df))
            return df, None
        except Exception as e:
            return None, f"Could not read local reforecast CSV: {e}"

    # ------------------------------------------------------------------
    # Fuzzy column resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_columns(df, canonical_names: list[str], cutoff: float = 0.6):
        """Match expected canonical column names to actual DataFrame columns.

        Returns (renamed_df, missing) where *missing* lists any canonical
        names that could not be matched.  Uses exact match first, then
        falls back to difflib closest match above *cutoff*.
        """
        actual = list(df.columns)
        rename_map: dict[str, str] = {}
        used: set[str] = set()

        for canon in canonical_names:
            if canon in actual and canon not in used:
                used.add(canon)
                continue  # already correct
            candidates = [c for c in actual if c not in used]
            matches = get_close_matches(canon, candidates, n=1, cutoff=cutoff)
            if matches:
                rename_map[matches[0]] = canon
                used.add(matches[0])

        missing = [c for c in canonical_names if c not in df.columns and c not in rename_map.values()]
        renamed_df = df.rename(columns=rename_map)
        if rename_map:
            logger.info("Fuzzy column renames: %s", rename_map)
        return renamed_df, missing

    # ------------------------------------------------------------------
    # Chart builder (Bokeh, same style as enrollment forecasting)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chart(filtered_df, protocol_id: str):
        """Build a Bokeh figure with lower/mean/upper curves + target line."""
        from bokeh.models import ColumnDataSource, HoverTool, Legend, Range1d, Span
        from bokeh.plotting import figure as bokeh_figure

        months = filtered_df["month"].tolist()
        lower = filtered_df["lower_bound"].tolist()
        mean = filtered_df["mean_"].tolist()
        upper = filtered_df["upper_bound"].tolist()
        target = filtered_df["target_subjected"].dropna().iloc[0] if filtered_df["target_subjected"].notna().any() else None

        source = ColumnDataSource({
            "month": months,
            "lower_bound": lower,
            "mean_": mean,
            "upper_bound": upper,
        })

        y_max = max(max(upper), target or 0) * 1.1

        p = bokeh_figure(
            title=f"Enrollment Reforecast \u2014 {protocol_id}",
            x_axis_label="Month",
            y_axis_label="Patients Enrolled",
            x_axis_type="datetime",
            height=420,
            width=780,
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.y_range = Range1d(0, y_max)

        # Upper bound (green, dashed)
        l_upper = p.line(
            "month", "upper_bound", source=source,
            line_color="#27ae60", line_width=2, line_dash="dashed",
        )
        # Mean (blue, solid)
        l_mean = p.line(
            "month", "mean_", source=source,
            line_color="#2980b9", line_width=2.5, line_dash="solid",
        )
        # Lower bound (red, dashed)
        l_lower = p.line(
            "month", "lower_bound", source=source,
            line_color="#e74c3c", line_width=2, line_dash="dashed",
        )

        legend_items = [
            ("Upper Bound", [l_upper]),
            ("Mean", [l_mean]),
            ("Lower Bound", [l_lower]),
        ]

        # Target horizontal line
        if target is not None:
            span = Span(
                location=target, dimension="width",
                line_color="gray", line_width=1.5, line_dash="dotted",
            )
            p.add_layout(span)
            # Invisible renderer just for the legend entry
            l_target = p.line(
                [months[0], months[-1]], [target, target],
                line_color="gray", line_width=1.5, line_dash="dotted",
            )
            legend_items.append((f"Target ({int(target)} patients)", [l_target]))

        legend = Legend(items=legend_items, location="top_left", label_text_font_size="10pt")
        p.add_layout(legend, "below")
        p.legend.click_policy = "hide"

        hover = HoverTool(tooltips=[
            ("Month", "@month{%Y-%m}"),
            ("Lower Bound", "@lower_bound{0.0}"),
            ("Mean", "@mean_{0.0}"),
            ("Upper Bound", "@upper_bound{0.0}"),
        ], formatters={"@month": "datetime"})
        p.add_tools(hover)

        p.xgrid.grid_line_alpha = 0.3
        p.ygrid.grid_line_alpha = 0.3

        return p

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        import pandas as pd

        protocol_id = str(params["protocol_id"]).strip()

        df, load_err = self._load_reforecast_df()
        if load_err:
            return AgentResult(success=False, text_response="", error_message=load_err)

        # Resolve columns via fuzzy matching
        canonical = ["protocol_number", "month", "lower_bound", "mean_", "upper_bound", "target_subjected"]
        df, missing = self._resolve_columns(df, canonical)

        if "protocol_number" in missing:
            return AgentResult(
                success=False, text_response="",
                error_message=f"Cannot identify a protocol column in dataset. Available columns: {list(df.columns)}",
            )

        required_value_cols = [c for c in ("month", "lower_bound", "mean_", "upper_bound") if c in missing]
        if required_value_cols:
            return AgentResult(
                success=False, text_response="",
                error_message=(
                    f"Could not match required columns {required_value_cols} "
                    f"to dataset columns: {list(df.columns)}"
                ),
            )

        # target_subjected is optional — fill with NaN if not matched
        if "target_subjected" in missing:
            import numpy as np
            df["target_subjected"] = np.nan

        # Parse month column to datetime so Bokeh renders a proper time axis.
        # Supports YYYY-MM strings as well as values already numeric or datetime.
        if not pd.api.types.is_datetime64_any_dtype(df["month"]):
            df["month"] = pd.to_datetime(df["month"], format="mixed", dayfirst=False)

        # Filter to the requested protocol
        mask = df["protocol_number"].astype(str).str.strip().str.upper() == protocol_id.upper()
        filtered = df[mask].copy()

        if filtered.empty:
            available = sorted(df["protocol_number"].astype(str).str.strip().unique().tolist())
            sample = ", ".join(available[:10])
            suffix = f" ... ({len(available)} total)" if len(available) > 10 else ""
            return AgentResult(
                success=False, text_response="",
                error_message=(
                    f"No data found for protocol '{protocol_id}'. "
                    f"Available protocols: {sample}{suffix}"
                ),
            )

        filtered = filtered.sort_values("month").reset_index(drop=True)

        # Build chart
        try:
            chart = self._build_chart(filtered, protocol_id)
        except Exception as e:
            logger.error("Reforecasting chart failed: %s", e)
            chart = None

        # Build summary
        target = filtered["target_subjected"].dropna().iloc[0] if filtered["target_subjected"].notna().any() else None
        last_month = filtered["month"].iloc[-1]
        last_month_label = last_month.strftime("%Y-%m")
        last_mean = filtered["mean_"].iloc[-1]

        summary = (
            f"**Enrollment Reforecast: {protocol_id}**\n\n"
            f"Showing **{len(filtered)}** monthly data points.\n\n"
            f"- **Mean enrollment at {last_month_label}:** {last_mean:,.0f} patients\n"
            f"- **Range at {last_month_label}:** {filtered['lower_bound'].iloc[-1]:,.0f} "
            f"(lower) \u2013 {filtered['upper_bound'].iloc[-1]:,.0f} (upper)\n"
        )
        if target is not None:
            summary += f"- **Target:** {int(target)} patients\n"

        # Build table data
        table_data = []
        for _, row in filtered.iterrows():
            table_data.append({
                "Month": row["month"].strftime("%Y-%m"),
                "Lower Bound": round(row["lower_bound"], 1),
                "Mean": round(row["mean_"], 1),
                "Upper Bound": round(row["upper_bound"], 1),
                "Target": int(target) if target is not None else "",
            })

        table_columns = ["Month", "Lower Bound", "Mean", "Upper Bound", "Target"]

        return AgentResult(
            success=True,
            text_response=summary,
            table_data=table_data,
            table_columns=table_columns,
            chart_json=chart,
        )
