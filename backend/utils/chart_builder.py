"""
Bokeh chart construction for enrollment forecasting.
Generates a dual-axis line chart with three scenario curves (pessimistic,
moderate, optimistic) for both cumulative enrollment and active site count.
"""
import json
import math
from datetime import datetime, timedelta

import numpy as np
from bokeh.embed import json_item
from bokeh.models import (ColumnDataSource, DatetimeTickFormatter,
                           HoverTool, Legend, LinearAxis, Range1d)
from bokeh.plotting import figure


def _logistic_ramp(t_months: np.ndarray, n_total: int, ramp_period: float) -> np.ndarray:
    """
    Logistic site activation curve.
    ramp_period = months until ~90% of sites are activated (sets the slope).
    """
    # k chosen so that logistic(t=ramp_period) ≈ 0.90
    k = math.log(9) / (ramp_period / 2)
    t_mid = ramp_period / 2
    activated = n_total / (1 + np.exp(-k * (t_months - t_mid)))
    return np.clip(activated, 0, n_total)


def compute_scenario(
    num_sites: int,
    num_patients: int,
    enrollment_rate: float,        # patients / active site / month
    ramp_period: float,            # months
    dropout_rate_monthly_pct: float,
    start_date: datetime,
    max_months: int = 72,
) -> dict:
    """
    Compute monthly time-series for one scenario.
    Returns dict with keys: months, dates, active_sites, cumulative_patients
    """
    t = np.arange(0, max_months + 1, dtype=float)
    active_sites = _logistic_ramp(t, num_sites, ramp_period)

    monthly_dropout = dropout_rate_monthly_pct / 100.0
    cumulative = np.zeros(len(t))
    on_study = np.zeros(len(t))

    for i in range(1, len(t)):
        new_enrolled = active_sites[i] * enrollment_rate
        dropouts = on_study[i - 1] * monthly_dropout
        on_study[i] = max(0, on_study[i - 1] + new_enrolled - dropouts)
        cumulative[i] = cumulative[i - 1] + new_enrolled
        if cumulative[i] >= num_patients:
            # Enrollment target reached — cap and stop
            overshoot = cumulative[i] - num_patients
            cumulative[i] = num_patients
            cumulative[i + 1:] = num_patients
            active_sites[i + 1:] = active_sites[i]
            break

    dates = [start_date + timedelta(days=30 * int(ti)) for ti in t]

    # Trim trailing months beyond completion
    completion_idx = next(
        (i for i, v in enumerate(cumulative) if v >= num_patients),
        len(t) - 1
    )
    cutoff = min(completion_idx + 3, len(t) - 1)

    return {
        "months": t[:cutoff + 1].tolist(),
        "dates": dates[:cutoff + 1],
        "active_sites": active_sites[:cutoff + 1].tolist(),
        "cumulative_patients": cumulative[:cutoff + 1].tolist(),
        "completion_month": int(t[completion_idx]),
        "peak_active_sites": float(np.max(active_sites[:cutoff + 1])),
    }


def build_enrollment_figure(
    scenarios: dict,
    num_sites: int,
    num_patients: int,
    start_date: datetime,
    indication: str,
    phase: str,
):
    """
    Build a Bokeh Figure with three enrollment curves and three site activation curves.
    Returns the raw Bokeh Figure — use directly in pn.pane.Bokeh() for the notebook,
    or wrap with json_item() for the Flask webapp (see build_enrollment_chart).
    """
    colors = {
        "pessimistic": "#e74c3c",
        "moderate": "#2980b9",
        "optimistic": "#27ae60",
    }

    p = figure(
        title=f"Enrollment Forecast — {indication} {phase}",
        x_axis_type="datetime",
        height=420,
        width=780,
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Left y-axis: cumulative patients
    p.yaxis.axis_label = "Cumulative Patients Enrolled"
    p.y_range = Range1d(0, num_patients * 1.05)

    # Right y-axis: active sites
    p.extra_y_ranges = {"sites": Range1d(0, num_sites * 1.1)}
    p.add_layout(
        LinearAxis(y_range_name="sites", axis_label="Active Sites"),
        "right"
    )

    legend_items_patients = []
    legend_items_sites = []

    for scenario_name, params in scenarios.items():
        result = compute_scenario(
            num_sites=num_sites,
            num_patients=num_patients,
            enrollment_rate=params["enrollment_rate_per_site_per_month"],
            ramp_period=params["site_ramp_period_months"],
            dropout_rate_monthly_pct=params["dropout_rate_monthly_percent"],
            start_date=start_date,
        )

        source = ColumnDataSource({
            "date": result["dates"],
            "patients": result["cumulative_patients"],
            "sites": result["active_sites"],
            "month": result["months"],
            "scenario": [scenario_name.capitalize()] * len(result["dates"]),
        })

        color = colors[scenario_name]

        # Enrollment curve (solid)
        l_patients = p.line(
            "date", "patients", source=source,
            line_color=color, line_width=2.5,
            line_dash="solid",
        )
        legend_items_patients.append((f"{scenario_name.capitalize()} — Patients", [l_patients]))

        # Site activation curve (dashed), right axis
        l_sites = p.line(
            "date", "sites", source=source,
            line_color=color, line_width=1.5,
            line_dash="dashed", y_range_name="sites",
        )
        legend_items_sites.append((f"{scenario_name.capitalize()} — Sites (dashed)", [l_sites]))

    # Add target line
    all_sources_max_date = start_date + timedelta(days=30 * 72)
    p.line(
        [start_date, all_sources_max_date],
        [num_patients, num_patients],
        line_dash="dotted", line_color="gray", line_width=1.5,
        legend_label=f"Target ({num_patients} patients)"
    )

    # Hover tool
    hover = HoverTool(tooltips=[
        ("Scenario", "@scenario"),
        ("Month", "@month{0}"),
        ("Patients Enrolled", "@patients{0.0}"),
        ("Active Sites", "@sites{0.0}"),
    ])
    p.add_tools(hover)

    # Legend
    legend = Legend(
        items=legend_items_patients + legend_items_sites,
        location="top_left",
        label_text_font_size="10pt",
    )
    p.add_layout(legend, "below")
    p.legend.click_policy = "hide"

    p.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", years="%Y")
    p.xgrid.grid_line_alpha = 0.3
    p.ygrid.grid_line_alpha = 0.3

    return p


def build_enrollment_chart(
    scenarios: dict,
    num_sites: int,
    num_patients: int,
    start_date: datetime,
    indication: str,
    phase: str,
) -> dict:
    """
    Convenience wrapper for the Flask webapp: builds the Figure and converts
    it to a Bokeh json_item dict suitable for embedding via BokehJS.
    """
    p = build_enrollment_figure(scenarios, num_sites, num_patients, start_date, indication, phase)
    return json_item(p, "enrollment_chart")
