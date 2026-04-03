"""
Builds and evaluates confirmation prompts.
Confirmation prompts are template-based (no LLM call) for speed and consistency.
"""
from backend.state.conversation_state import ConfirmationRequest
from backend.state.parameter_schema import SkillSchema


_SKILL_VERB = {
    "site_list_merger": "merge the uploaded site lists",
    "trial_benchmarking": "run Trial Benchmarking",
    "drug_reimbursement": "run Drug Reimbursement Assessment",
    "enrollment_forecasting": "run Enrollment Forecasting",
}

_YES_WORDS = {"yes", "y", "yep", "yeah", "sure", "proceed", "go", "go ahead",
              "confirm", "confirmed", "ok", "okay", "correct", "do it", "run it"}
_NO_WORDS = {"no", "n", "nope", "cancel", "stop", "abort", "don't", "do not",
             "negative", "not", "wrong", "incorrect"}
_EDIT_WORDS = {"edit", "change", "modify", "update", "adjust", "different",
               "instead", "actually"}


def build_confirmation_prompt(
    skill_schema: SkillSchema,
    params: dict,
    inherited_params: dict = None,
) -> ConfirmationRequest:
    """
    Build a human-readable confirmation request showing collected parameters.
    inherited_params: params suggested from a prior skill run (shown differently).
    """
    verb = _SKILL_VERB.get(skill_schema.skill_id, f"run {skill_schema.display_name}")
    lines = [f"I'm ready to **{verb}** with the following parameters:\n"]

    for spec in skill_schema.required_parameters:
        if spec.data_type == "file":
            continue
        value = params.get(spec.name)
        inherited = (inherited_params or {}).get(spec.name)
        if value is not None:
            source = " *(inherited from prior request)*" if (inherited and value == inherited) else ""
            if isinstance(value, list):
                formatted = ", ".join(str(v) for v in value)
            else:
                formatted = str(value)
            lines.append(f"  - **{spec.label}**: {formatted}{source}")

    for spec in skill_schema.optional_parameters:
        if spec.data_type == "file":
            continue
        value = params.get(spec.name)
        if value is not None:
            if isinstance(value, list):
                formatted = ", ".join(str(v) for v in value)
            else:
                formatted = str(value)
            lines.append(f"  - **{spec.label}** *(optional)*: {formatted}")

    # File uploads for site merger
    if skill_schema.skill_id == "site_list_merger":
        lines.append("  - CRO and sponsor site list files: uploaded")

    lines.append("\nShould I proceed? Reply **yes** to confirm, **no** to cancel, or **edit** to change a parameter.")

    summary_text = "\n".join(lines)

    return ConfirmationRequest(
        skill_id=skill_schema.skill_id,
        parameter_snapshot=dict(params),
        summary_text=summary_text,
    )


def parse_confirmation_reply(user_message: str) -> str:
    """
    Returns "yes", "no", or "edit" based on user reply.
    Defaults to "edit" if ambiguous.
    """
    lower = user_message.lower().strip().rstrip(".,!?")
    tokens = set(lower.split())

    if tokens & _YES_WORDS and not tokens & _NO_WORDS:
        return "yes"
    if tokens & _NO_WORDS and not tokens & _YES_WORDS:
        return "no"
    if tokens & _EDIT_WORDS:
        return "edit"

    # Single-word check
    if lower in _YES_WORDS:
        return "yes"
    if lower in _NO_WORDS:
        return "no"

    return "edit"
