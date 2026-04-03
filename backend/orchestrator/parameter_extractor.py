"""
LLM-based parameter extractor. Pulls required parameter values from the
user's message and conversation history for a specific skill.
"""
import logging

from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (PARAMETER_EXTRACTOR_SYSTEM,
                                           PARAMETER_EXTRACTOR_USER)
from backend.llm.response_parser import parse_parameter_extraction
from backend.state.parameter_schema import SkillSchema
from backend.utils.validators import parse_country_list, to_positive_integer

logger = logging.getLogger(__name__)


def extract_parameters(
    llm: LLMClient,
    skill_schema: SkillSchema,
    user_message: str,
    history: list[dict],
) -> dict:
    """
    Extract parameters relevant to the skill from the current message + history.
    Returns a dict {param_name: value | None}.
    Does not overwrite with None — caller merges into existing state.
    """
    param_names = [p.name for p in skill_schema.all_parameters()
                   if p.data_type != "file"]   # Files are handled via upload endpoint

    if not param_names:
        return {}

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history
    ) or "(no prior conversation)"

    messages = [
        {"role": "system", "content": PARAMETER_EXTRACTOR_SYSTEM},
        {"role": "user", "content": PARAMETER_EXTRACTOR_USER.format(
            skill_display_name=skill_schema.display_name,
            param_names=", ".join(param_names),
            history=history_text,
            user_message=user_message,
        )},
    ]

    try:
        raw = llm.complete_json(messages, temperature=llm.temp_extract)
        extracted = parse_parameter_extraction(raw)
    except Exception as e:
        logger.error("Parameter extraction failed: %s", e)
        return {}

    return _postprocess(extracted, skill_schema)


def _postprocess(extracted: dict, schema: SkillSchema) -> dict:
    """
    Normalize and validate extracted values.
    - Normalize choice values using schema aliases.
    - Coerce integers.
    - Parse country lists.
    - Drop None values.
    """
    result = {}
    param_map = {p.name: p for p in schema.all_parameters()}

    for name, value in extracted.items():
        if value is None:
            continue
        spec = param_map.get(name)
        if spec is None:
            continue

        if spec.data_type == "choice":
            normalized = schema.normalize_choice(name, str(value))
            if normalized:
                result[name] = normalized
            else:
                logger.debug("Unrecognized choice value '%s' for param '%s'", value, name)

        elif spec.data_type == "integer":
            coerced = to_positive_integer(value)
            if coerced is not None:
                result[name] = coerced

        elif spec.data_type == "list":
            # Expect either a list or a comma-separated string
            if isinstance(value, list):
                result[name] = [str(v).strip() for v in value if v]
            else:
                parsed = parse_country_list(str(value))
                if parsed:
                    result[name] = parsed

        else:
            # string, date — store as-is
            result[name] = str(value).strip()

    return result
