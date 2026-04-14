"""
Reforecasting SubAgent.
Loads the REFORECAST Dataiku dataset, filters by protocol_number,
and plots lower_bound / mean_ / upper_bound enrollment curves with a
target_subjected horizontal reference line.
"""
from __future__ import annotations

import logging
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
            ("Month", "@month{0}"),
            ("Lower Bound", "@lower_bound{0.0}"),
            ("Mean", "@mean_{0.0}"),
            ("Upper Bound", "@upper_bound{0.0}"),
        ])
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

        if "protocol_number" not in df.columns:
            return AgentResult(
                success=False, text_response="",
                error_message=f"Column 'PROTOCOL_NUMBER' not found in dataset. Available columns: {list(df.columns)}",
            )

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

        # Resolve the month column
        month_col = None
        for candidate in ("month", "months", "time_months", "time"):
            if candidate in filtered.columns:
                month_col = candidate
                break
        if month_col is None:
            # Fallback: first numeric column that isn't one of the value columns
            value_cols = {"lower_bound", "mean_", "upper_bound", "target_subjected", "protocol_number"}
            for col in filtered.columns:
                if col not in value_cols and pd.api.types.is_numeric_dtype(filtered[col]):
                    month_col = col
                    break
        if month_col is None:
            return AgentResult(
                success=False, text_response="",
                error_message=f"Cannot identify a month/time column. Available columns: {list(filtered.columns)}",
            )

        # Rename to canonical 'month' for the chart builder
        if month_col != "month":
            filtered = filtered.rename(columns={month_col: "month"})

        # Validate required value columns exist
        for col in ("lower_bound", "mean_", "upper_bound", "target_subjected"):
            if col not in filtered.columns:
                return AgentResult(
                    success=False, text_response="",
                    error_message=f"Required column '{col}' not found in dataset. Available: {list(filtered.columns)}",
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
        last_mean = filtered["mean_"].iloc[-1]

        summary = (
            f"**Enrollment Reforecast: {protocol_id}**\n\n"
            f"Showing **{len(filtered)}** monthly data points.\n\n"
            f"- **Mean enrollment at month {int(last_month)}:** {last_mean:,.0f} patients\n"
            f"- **Range at month {int(last_month)}:** {filtered['lower_bound'].iloc[-1]:,.0f} "
            f"(lower) \u2013 {filtered['upper_bound'].iloc[-1]:,.0f} (upper)\n"
        )
        if target is not None:
            summary += f"- **Target:** {int(target)} patients\n"

        # Build table data
        table_data = []
        for _, row in filtered.iterrows():
            table_data.append({
                "Month": int(row["month"]),
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
