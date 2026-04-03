"""
Response formatting utilities — convert structured data into chat-ready output.
"""
from typing import Any


def dict_list_to_table(rows: list[dict], columns: list[str] = None) -> dict:
    """
    Convert a list of dicts to a table descriptor consumed by the frontend.
    Returns {"columns": [...], "rows": [...]} where rows are ordered lists.
    """
    if not rows:
        return {"columns": [], "rows": []}
    if columns is None:
        # Collect all unique keys preserving first-seen order
        seen = {}
        for row in rows:
            for k in row.keys():
                seen[k] = True
        columns = list(seen.keys())

    ordered_rows = [
        [row.get(col, "") for col in columns]
        for row in rows
    ]
    return {"columns": columns, "rows": ordered_rows}


def format_key_metrics_table(metrics: dict) -> dict:
    """
    Convert a flat metrics dict into a two-column name/value table.
    """
    label_map = {
        "median_enrollment_rate_patients_per_site_per_month": "Median Enrollment Rate (patients/site/month)",
        "median_dropout_rate_percent": "Median Dropout Rate (%)",
        "typical_duration_months": "Typical Trial Duration (months)",
        "typical_site_count_range": "Typical Site Count Range",
        "typical_screen_failure_rate_percent": "Typical Screen Failure Rate (%)",
    }
    rows = []
    for key, value in metrics.items():
        label = label_map.get(key, key.replace("_", " ").title())
        rows.append({"Metric": label, "Value": str(value)})
    return dict_list_to_table(rows, columns=["Metric", "Value"])


def format_reimbursement_table(assessments: list[dict]) -> dict:
    """Flatten reimbursement country assessments for table display."""
    rows = []
    for a in assessments:
        rows.append({
            "Country": a.get("country", ""),
            "Payer / HTA Body": a.get("payer_body", ""),
            "Reimbursement Likelihood": a.get("reimbursement_likelihood", ""),
            "Est. Timeline (months)": a.get("estimated_timeline_months", ""),
            "Key Requirements": "; ".join(a.get("key_requirements", [])),
            "Key Risks": "; ".join(a.get("key_risks", [])),
            "Notes": a.get("notes", ""),
        })
    return dict_list_to_table(rows, columns=[
        "Country", "Payer / HTA Body", "Reimbursement Likelihood",
        "Est. Timeline (months)", "Key Requirements", "Key Risks", "Notes"
    ])


def format_merger_summary(summary: dict) -> str:
    return (
        f"Merge complete. **{summary.get('total_sites', 0)} total sites** identified: "
        f"{summary.get('cro_only', 0)} from CRO only, "
        f"{summary.get('sponsor_only', 0)} from sponsor only, "
        f"{summary.get('in_both', 0)} present in both lists. "
        f"**{summary.get('conflicts_found', 0)} conflicts** flagged."
    )


def likelihood_badge(likelihood: str) -> str:
    """Return a text badge for reimbursement likelihood."""
    badges = {
        "favorable": "[FAVORABLE]",
        "uncertain": "[UNCERTAIN]",
        "challenging": "[CHALLENGING]",
    }
    return badges.get(likelihood.lower(), likelihood)
